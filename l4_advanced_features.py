"""
Advanced NVIDIA L4 GPU Features for Phi-3.5 Vision
==================================================
This script demonstrates powerful capabilities of the NVIDIA L4 GPU with Phi-3.5 Vision model:
- Multi-query parallel batch processing
- Advanced image processing with higher resolution
- Region-of-interest feature extraction
- Cross-modal attention visualization
- Model throughput benchmarking
- Memory optimization techniques

For NVIDIA L4 with 24GB VRAM.
"""

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoProcessor
from torch.cuda.amp import autocast
import concurrent.futures
from io import BytesIO

# Configure for maximum L4 performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
device = torch.device("cuda")

# Optional: Enable DeepSpeed inference if available
USE_DEEPSPEED = False  # Set to True if DeepSpeed is installed


@dataclass
class L4ProcessingConfig:
    """Configuration for optimizing L4 GPU processing."""
    model_id: str = "microsoft/Phi-3.5-vision-instruct"
    batch_size: int = 4
    image_size: int = 1024  # Higher resolution for L4
    num_crops: int = 16  # Use more crops on L4
    max_new_tokens: int = 1500
    temperature: float = 0.2
    use_flash_attention: bool = True
    use_bettertransformer: bool = True
    enable_cuda_graph: bool = True  # Use CUDA graphs for repetitive operations
    dtype: torch.dtype = torch.float16
    inference_mode: str = "batch"  # Options: 'batch', 'streaming'
    extraction_layers: List[int] = None  # Layers to extract features from


class L4MultiBatchProcessor:
    """Advanced processor leveraging L4 GPU capabilities for multimodal processing."""
    
    def __init__(self, config: L4ProcessingConfig):
        self.config = config
        self.model = None
        self.processor = None
        self.hook_handles = []
        self.layer_features = {}
        self.current_batch_id = None
        
        # Initialize and optimize for L4
        self._initialize_model()
        
        # Performance metrics
        self.latency_history = []
        self.throughput_history = []
        self.memory_usage_history = []
    
    def _initialize_model(self):
        """Initialize model with L4-specific optimizations."""
        print(f"Initializing model {self.config.model_id} with L4 optimizations...")
        
        # Model loading with L4 optimizations
        model_args = {
            "device_map": "cuda",
            "torch_dtype": self.config.dtype,
            "trust_remote_code": True,
            "_attn_implementation": 'flash_attention_2' if self.config.use_flash_attention else 'eager',
            "use_cache": True
        }
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id, 
            **model_args
        )
        
        # Apply BetterTransformer if enabled
        if self.config.use_bettertransformer:
            try:
                from optimum.bettertransformer import BetterTransformer
                self.model = BetterTransformer.transform(self.model)
                print("Applied BetterTransformer optimizations")
            except ImportError:
                print("BetterTransformer not available. Install with: pip install optimum")
        
        # DeepSpeed Inference optimization if enabled and available
        if USE_DEEPSPEED:
            try:
                import deepspeed
                self.model = deepspeed.init_inference(
                    self.model,
                    mp_size=1,
                    dtype=self.config.dtype,
                    replace_with_kernel_inject=True,
                    enable_cuda_graph=self.config.enable_cuda_graph
                )
                print("Applied DeepSpeed optimizations")
            except ImportError:
                print("DeepSpeed not available. Install with: pip install deepspeed")
        
        # Initialize processor
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
            num_crops=self.config.num_crops
        )
        
        # Register hooks for feature extraction if layers specified
        if self.config.extraction_layers:
            self._register_feature_hooks()
        
        print("Model and processor initialized successfully")
        self._report_memory("After model initialization")
    
    def _register_feature_hooks(self):
        """Register hooks to extract features from specified layers."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        
        # Get layer modules
        layers = []
        for name, module in self.model.named_modules():
            if "layers" in name and "attention" in name:
                layer_num = int(name.split("layers.")[1].split(".")[0])
                if layer_num in self.config.extraction_layers or -1 in self.config.extraction_layers:
                    layers.append((name, module))
        
        # Register hooks
        for name, module in layers:
            handle = module.register_forward_hook(self._feature_hook(name))
            self.hook_handles.append(handle)
        
        print(f"Registered feature hooks on {len(self.hook_handles)} layers")
    
    def _feature_hook(self, name):
        """Create a hook function to capture features."""
        def hook(module, input, output):
            if self.current_batch_id is not None:
                if isinstance(output, tuple):
                    # Some models return tuples from attention
                    feature = output[0].detach()
                else:
                    feature = output.detach()
                
                # Store the feature
                if self.current_batch_id not in self.layer_features:
                    self.layer_features[self.current_batch_id] = {}
                self.layer_features[self.current_batch_id][name] = feature
        return hook
    
    def _report_memory(self, label):
        """Report current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"Memory {label}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
            self.memory_usage_history.append((label, allocated, reserved))
            return allocated
        return 0
    
    def _parallel_load_images(self, image_urls):
        """Load multiple images in parallel using thread pool."""
        all_images = []
        
        def load_image(url):
            try:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    img = img.convert('RGB')
                    
                    # Resize for optimal L4 processing if needed
                    if max(img.size) > self.config.image_size:
                        ratio = self.config.image_size / max(img.size)
                        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                        img = img.resize(new_size, Image.LANCZOS)
                    
                    return img
            except Exception as e:
                print(f"Error loading image {url}: {e}")
            return None
        
        # Process each batch of image URLs
        for urls_batch in image_urls:
            batch_images = []
            
            # Load images in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(load_image, url) for url in urls_batch]
                for future in concurrent.futures.as_completed(futures):
                    img = future.result()
                    if img is not None:
                        batch_images.append(img)
            
            all_images.append(batch_images)
        
        return all_images
    
    def process_batch(self, image_urls, queries, batch_id="batch_1"):
        """Process a batch of image-query pairs with L4 optimizations."""
        self.current_batch_id = batch_id
        print(f"Processing batch {batch_id} with {len(queries)} queries...")
        
        # Parallel image loading
        start_time = time.time()
        all_images = self._parallel_load_images(image_urls)
        image_load_time = time.time() - start_time
        print(f"Loaded {sum(len(imgs) for imgs in all_images)} images in {image_load_time:.2f}s")
        
        # Prepare inputs
        start_time = time.time()
        inputs_list = []
        
        for i, (images, query) in enumerate(zip(all_images, queries)):
            # Create placeholders
            placeholders = "".join([f"<|image_{j+1}|>\n" for j in range(len(images))])
            
            # Create messages
            messages = [{"role": "user", "content": placeholders + query}]
            
            # Create prompt
            prompt = self.processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(prompt, images, return_tensors="pt")
            inputs_list.append(inputs)
        
        # Combine inputs
        combined_inputs = {}
        for key in inputs_list[0].keys():
            combined_inputs[key] = torch.cat([inp[key] for inp in inputs_list], dim=0).to(device)
        
        prep_time = time.time() - start_time
        print(f"Prepared inputs in {prep_time:.2f}s")
        
        # Generate outputs
        start_time = time.time()
        
        # Use optimized generation
        with torch.inference_mode(), autocast(dtype=self.config.dtype):
            outputs = self.model.generate(
                **combined_inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
        
        generation_time = time.time() - start_time
        total_tokens = sum(output.shape[0] for output in outputs)
        throughput = total_tokens / generation_time
        
        self.latency_history.append(generation_time)
        self.throughput_history.append(throughput)
        
        print(f"Generated {total_tokens} tokens in {generation_time:.2f}s")
        print(f"Throughput: {throughput:.2f} tokens/sec")
        
        # Process outputs
        responses = []
        for i, query in enumerate(queries):
            # Get input length for this item
            input_length = combined_inputs['input_ids'][i].shape[0]
            
            # Extract generated IDs (skip input)
            generated_ids = outputs[i, input_length:]
            
            # Decode to text
            response = self.processor.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            responses.append(response)
        
        # Track memory after generation
        self._report_memory(f"After batch {batch_id} generation")
        
        return {
            "responses": responses,
            "metrics": {
                "image_load_time": image_load_time,
                "prep_time": prep_time,
                "generation_time": generation_time,
                "tokens_per_second": throughput,
                "total_time": image_load_time + prep_time + generation_time
            },
            "features": self.layer_features.get(batch_id, {})
        }
    
    def extract_region_features(self, image, regions=None):
        """Extract features from specific regions of an image."""
        if regions is None:
            # Default: split image into 4 quadrants
            w, h = image.size
            regions = [
                (0, 0, w//2, h//2),        # Top-left
                (w//2, 0, w, h//2),        # Top-right
                (0, h//2, w//2, h),        # Bottom-left
                (w//2, h//2, w, h)         # Bottom-right
            ]
        
        # Extract features from each region
        region_features = []
        
        for i, region in enumerate(regions):
            # Crop region
            region_img = image.crop(region)
            
            # Create a simple prompt for feature extraction
            messages = [{"role": "user", "content": f"<|image_1|>\nDescribe this region of the image."}]
            prompt = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process
            inputs = self.processor(prompt, [region_img], return_tensors="pt").to(device)
            
            # Extract features (no generation needed)
            self.current_batch_id = f"region_{i}"
            with torch.inference_mode(), autocast(dtype=self.config.dtype):
                # Just run forward pass up to last layer for feature extraction
                _ = self.model(**inputs, output_hidden_states=True)
            
            # Collect features
            region_features.append(self.layer_features.get(f"region_{i}", {}))
        
        return region_features
    
    def visualize_cross_attention(self, image, query, layer_idx=-1):
        """Visualize cross-attention between text and image regions."""
        # Process single image
        messages = [{"role": "user", "content": f"<|image_1|>\n{query}"}]
        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process
        inputs = self.processor(prompt, [image], return_tensors="pt").to(device)
        
        # Enable attention output
        self.model.config.output_attentions = True
        
        # Forward pass
        with torch.inference_mode(), autocast(dtype=self.config.dtype):
            outputs = self.model(**inputs, output_attentions=True)
        
        # Get attention weights
        attentions = outputs.attentions  # tuple of attention weights
        target_layer = attentions[layer_idx]  # get chosen layer
        
        # Average attention across heads
        avg_attention = target_layer.mean(dim=1).squeeze(0)  # shape: [seq_len, seq_len]
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        plt.imshow(avg_attention.cpu().numpy(), cmap='viridis')
        plt.colorbar(label='Attention Weight')
        plt.title(f'Cross-Attention Visualization for Query: "{query}"')
        plt.xlabel('Key (Image Tokens)')
        plt.ylabel('Query (Text Tokens)')
        
        # Save visualization
        os.makedirs('attention_maps', exist_ok=True)
        plt.savefig(f'attention_maps/cross_attention_{int(time.time())}.png')
        plt.close()
        
        # Reset attention output flag
        self.model.config.output_attentions = False
        
        return avg_attention
    
    def benchmark_throughput(self, image_urls, query, num_runs=5):
        """Benchmark throughput on L4 with different batch sizes."""
        results = []
        
        for batch_size in [1, 2, 4, 8]:
            print(f"\nBenchmarking with batch size {batch_size}")
            batch_times = []
            
            # Prepare data: repeat the same image/query for batch size
            batch_urls = [image_urls] * batch_size
            batch_queries = [query] * batch_size
            
            # Warmup run
            _ = self.process_batch(batch_urls, batch_queries, batch_id=f"warmup_bs{batch_size}")
            
            # Benchmark runs
            for run in range(num_runs):
                torch.cuda.empty_cache()
                batch_id = f"bench_bs{batch_size}_run{run}"
                
                result = self.process_batch(batch_urls, batch_queries, batch_id=batch_id)
                batch_times.append(result["metrics"]["generation_time"])
            
            # Calculate stats
            avg_time = np.mean(batch_times)
            std_time = np.std(batch_times)
            tokens_per_sec = self.config.max_new_tokens * batch_size / avg_time
            
            results.append({
                "batch_size": batch_size,
                "avg_time_sec": avg_time,
                "std_time_sec": std_time,
                "tokens_per_sec": tokens_per_sec
            })
            
            print(f"Batch size {batch_size}: {tokens_per_sec:.2f} tokens/sec "
                  f"(± {std_time/avg_time*100:.2f}%)")
        
        # Plot results
        plt.figure(figsize=(10, 6))
        batch_sizes = [r["batch_size"] for r in results]
        throughputs = [r["tokens_per_sec"] for r in results]
        
        plt.bar(batch_sizes, throughputs, color='royalblue')
        plt.xlabel('Batch Size')
        plt.ylabel('Throughput (tokens/sec)')
        plt.title('L4 GPU Throughput by Batch Size')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        for i, v in enumerate(throughputs):
            plt.text(batch_sizes[i], v + max(throughputs)*0.03, f"{v:.1f}", 
                    ha='center', va='bottom', fontweight='bold')
        
        os.makedirs('benchmarks', exist_ok=True)
        plt.savefig('benchmarks/l4_throughput_benchmark.png')
        plt.close()
        
        return results


# Example usage
if __name__ == "__main__":
    print("\n=== NVIDIA L4 Advanced Multimodal Processing Demo ===\n")
    
    # Configure for NVIDIA L4 GPU
    config = L4ProcessingConfig(
        model_id="microsoft/Phi-3.5-vision-instruct",
        batch_size=4,
        image_size=1024,
        num_crops=16,
        max_new_tokens=1500,
        extraction_layers=[-1, -2]  # Extract features from last 2 layers
    )
    
    # Initialize processor
    processor = L4MultiBatchProcessor(config)
    
    # Example image URLs - AI/ML conference presentations
    image_urls = [
        # Batch 1: AI conference presentation slides
        [
            "https://image.slidesharecdn.com/aiinmedicine-111118151146-phpapp02/95/ai-in-medicine-5-728.jpg",
            "https://image.slidesharecdn.com/aiinmedicine-111118151146-phpapp02/95/ai-in-medicine-6-728.jpg",
            "https://image.slidesharecdn.com/aiinmedicine-111118151146-phpapp02/95/ai-in-medicine-7-728.jpg",
        ],
        # Batch 2: ML architecture slides
        [
            "https://image.slidesharecdn.com/machinelearningfundamentals-190723183126/95/machine-learning-fundamentals-10-728.jpg",
            "https://image.slidesharecdn.com/machinelearningfundamentals-190723183126/95/machine-learning-fundamentals-11-728.jpg",
            "https://image.slidesharecdn.com/machinelearningfundamentals-190723183126/95/machine-learning-fundamentals-12-728.jpg",
        ]
    ]
    
    # Example queries
    queries = [
        "What medical applications of AI are shown in these slides?",
        "Explain the machine learning concepts shown in these slides."
    ]
    
    # 1. Process batch with full metrics
    print("\n=== 1. Batch Processing with Metrics ===\n")
    result = processor.process_batch(image_urls, queries, batch_id="demo_batch")
    
    # Display results
    for i, (query, response) in enumerate(zip(queries, result["responses"])):
        print(f"\nQuery {i+1}: {query}")
        print("-" * 80)
        print(response)
        print("-" * 80)
    
    # 2. Region feature extraction
    print("\n=== 2. Region Feature Extraction ===\n")
    
    # Load a single image for region analysis
    image_url = "https://image.slidesharecdn.com/machinelearningfundamentals-190723183126/95/machine-learning-fundamentals-10-728.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
    
    # Extract features from regions
    region_features = processor.extract_region_features(image)
    print(f"Extracted features from {len(region_features)} regions")
    
    # 3. Attention visualization
    print("\n=== 3. Cross-Attention Visualization ===\n")
    attention_map = processor.visualize_cross_attention(
        image,
        "What machine learning architecture is shown in this diagram?",
        layer_idx=-1
    )
    print("Attention visualization saved to 'attention_maps/' directory")
    
    # 4. Performance benchmarking
    print("\n=== 4. L4 Performance Benchmarking ===\n")
    benchmark_results = processor.benchmark_throughput(
        image_urls[0],
        "Describe these slides about AI in medicine.",
        num_runs=3
    )
    print("\nBenchmark results saved to 'benchmarks/' directory")
    
    print("\n=== Demo Complete ===\n") 