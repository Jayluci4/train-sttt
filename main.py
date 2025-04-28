import os
import math
import torch
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from functools import partial
import math
import tqdm
# Import necessary PyTorch and Transformers classes
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
    AutoProcessor
)
from torch.cuda.amp import GradScaler, autocast
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from PIL import Image

# Import all your custom MetaMorph components
from intrinsic_dimension_minimizer import IntrinsicDimensionMinimizer
from plasticity_weighted_reweighter import PlasticityWeightedReweighter
from dynamic_mutual_information import DynamicMutualInformationTracker
from bregman_dynamics import BregmanDynamicsController
from reward_weighted_plasticity import RewardWeightedPlasticityController
from adaptive_distillation import AdaptiveDistillationController
from convergent_neural_architecture_search import ConvergentNeuralArchitectureSearch
from chain_of_thought_injector import ChainOfThoughtInjector, PromptRouter
from enhanced_metaplasticity_optimizer import EnhancedMetaplasticityOptimizer
from enhanced_architecture_controller import EnhancedConvergentArchitectureController
from enhanced_activation_engineering import EnhancedActivationEngineering
# Import multimodal fusion components
from advanced_modal_fusion import AdvancedModalFusion, ModalFusionConfig
# Ensure EnhancedAdaFactorWithMetaplasticity is imported if used
# from enhanced_metaplasticity_optimizer import EnhancedAdaFactorWithMetaplasticity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("convergent_metamorph_qwen.log"), # Log to a new file
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("QwenConvergentMetaMorph")

# ------------------- CONFIGURATION -------------------
@dataclass
class ConvergentMetaMorphConfig:
    """Configuration for ConvergentMetaMorph with next-gen features."""
    # Model configuration
    model_name: str = "microsoft/Phi-3.5-vision-instruct" # Updated to Phi-3.5 Mini Vision
    output_dir: str = "./phi_vision_metamorph_output" # Updated output directory name

    # L4 GPU Advanced Features
    enable_tensor_parallel: bool = True # Enable tensor parallelism on multiple L4s if available
    enable_multimodal_fusion: bool = True # Enable advanced vision-text fusion capabilities
    enable_multi_gpu_checkpointing: bool = True # Checkpoint across multiple GPUs if available
    use_fused_adam: bool = True # Use fused Adam optimizer for L4
    enable_activation_checkpointing: bool = True # Save memory with activation checkpointing
    
    # L4 Memory Management
    dynamic_token_length: bool = True # Dynamically adjust sequence length during training
    auto_kv_cache_scaling: bool = True # Automatically scale KV cache based on batch size
    gradual_unfreezing: bool = True # Gradually unfreeze layers during training
    use_gradient_checkpointing: bool = True # Enable gradient checkpointing
    use_cpu_offloading: bool = False # Offload optimizer states to CPU when needed
    
    # L4 Inference Optimization
    use_bettertransformer: bool = True # Use BetterTransformer for inference
    enable_weight_streaming: bool = True # Stream weights for large models
    enable_continuous_batching: bool = True # Use continuous batching
    
    # Distributed Training (if multiple L4s)
    distributed_training: bool = False # Set to True if using multiple GPUs
    distributed_type: str = "deepspeed" # Options: "deepspeed", "accelerate"
    deepspeed_stage: int = 2 # ZeRO stage for DeepSpeed

    # Data configuration
    data_path: str = "path/to/your/full_dataset.csv" # <--- IMPORTANT: SET PATH TO YOUR FULL CSV
    max_seq_length: int = 768 # Increased for L4 GPU (24GB VRAM)
    max_image_size: int = 1024 # Increase image resolution for L4

    # Hardware optimization for NVIDIA L4
    device: str = "cuda" # Use CUDA for L4 GPU
    fp16_training: bool = True # Use FP16 for faster training on L4
    
    # Vision model specific settings
    use_flash_attention: bool = True # Keep flash attention for faster processing
    num_image_crops: int = 12 # Further increased for L4's higher capacity
    image_processing_threads: int = 8 # Parallel threads for image processing
    
    # Batch sizes optimized for L4 GPU
    micro_batch_size: int = 6 # Increased for maximum L4 utilization
    gradient_accumulation_steps: int = 8 # Increased for better optimization
    
    # Quantization settings - optimized for L4
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16" # Use float16 for L4 optimization
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True # Enable nested quantization for memory efficiency
    
    # Performance optimization
    use_scan_llm: bool = True # Enable ScanLLM optimization if available
    use_sdp_attention: bool = True # Enable SDP attention for faster processing
    use_scaled_dot_product: bool = True # Use scaled dot product attention
    
    # Advanced vision features
    enable_vision_feature_extraction: bool = True # Extract features from images
    vision_feature_pooling: str = "attention" # Use attention pooling for vision features
    vision_feature_layers: List[int] = field(default_factory=lambda: [-1, -2, -3, -4]) # Use multiple vision layers
    enable_cross_frame_attention: bool = True # Enable attention across multiple frames
    use_region_features: bool = True # Extract region-specific features
    
    # PEFT settings - increased for more capacity on L4
    use_peft: bool = True
    peft_method: str = "lora"
    lora_r: int = 16 # Increased from 8 to 16 for L4 GPU
    lora_alpha: int = 32 # Increased from 16 to 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ]) # Added more target modules for L4 GPU
    
    # Advanced Components Activation - enabled more for L4
    use_intrinsic_dimension: bool = True # Enabled for L4
    use_plasticity_reweighter: bool = True # Enabled for L4
    use_dynamic_mi: bool = True # Enabled for L4
    use_bregman_dynamics: bool = True # Enabled for L4 with more memory
    use_reward_plasticity: bool = True # Enabled for L4
    use_adaptive_distillation: bool = True # Enabled for L4 with teacher model
    use_convergent_nas: bool = True # Enabled for L4
    use_cot_injection: bool = True # Enabled for L4
    use_dynamic_architecture: bool = True # Enabled for L4
    use_metaplasticity: bool = True # Keep enabled
    use_activation_engineering: bool = True # Enabled for L4

    # Advanced data augmentation
    enable_data_augmentation: bool = True
    augmentation_methods: List[str] = field(default_factory=lambda: [
        "random_resize", "color_jitter", "random_crop", "random_flip"
    ])
    
    # Multi-task learning
    enable_multitask_heads: bool = True
    auxiliary_tasks: List[str] = field(default_factory=lambda: [
        "image_captioning", "text_classification", "object_detection"
    ])
    
    # Training settings - optimized for L4
    epochs: int = 8 # Increased for L4's higher efficiency
    learning_rate: float = 8e-5 # Increased for faster convergence on L4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    lr_scheduler: str = "cosine_with_restarts" # Use better scheduler for L4
    num_cycles: int = 3 # Number of LR cycles for cosine with restarts

    # Mixed precision (Recommended for L4)
    use_mixed_precision: bool = True
    amp_opt_level: str = "O2" # Advanced mixed precision option

    # Training logistics
    seed: int = 42
    logging_steps: int = 10 # Increased for longer runs
    eval_steps: int = 200 # Increased for longer runs
    save_steps: int = 500 # Increased for longer runs
    
    # Monitoring & visualization
    enable_wandb_logging: bool = True # Enable Weights & Biases monitoring
    enable_tensorboard: bool = True
    profiling_steps: int = 100 # Steps between profiling events
    
    # Performance profiling for L4
    enable_cuda_profiling: bool = False # Enable CUDA profiling (turn on temporarily)
    profile_memory_usage: bool = True # Profile memory usage
    
    # --- Component Specific Configs (Enhanced for L4) ---
    # Intrinsic Dimension settings
    dim_epsilon: float = 1e-6
    dim_window_size: int = 50
    dim_threshold: float = 0.1
    # Plasticity Reweighter settings
    reweighting_strength: float = 0.5
    max_weight_ratio: float = 5.0
    min_weight: float = 0.1
    # Dynamic MI settings
    mi_num_bins: int = 20
    mi_window_size: int = 50
    mi_update_freq: int = 10
    mi_threshold: float = 0.05
    # Bregman Dynamics settings
    divergence_type: str = "squared_euclidean"
    step_size_schedule: str = "inverse_sqrt_t"
    error_budget: float = 1.0 # Note: This gets scaled based on total steps later
    # Reward-weighted Plasticity settings
    rwp_update_rate: float = 0.01
    rwp_min_plasticity: float = 0.2
    rwp_max_plasticity: float = 5.0
    # Convergent NAS settings
    nas_error_budget: float = 1.0
    nas_exploration_factor: float = 0.1
    nas_warmup_steps: int = 100
    nas_morph_interval: int = 50
    # Convergent Architecture settings
    min_lora_rank: int = 4
    max_lora_rank: int = 64
    rank_step: int = 4
    architecture_scan_freq: int = 100
    morph_threshold: float = 0.15
    schedule_alpha: float = 0.7
    # Metaplasticity settings
    use_adafactor: bool = False
    plasticity_eta: float = 0.01
    plasticity_decay: float = 0.9999
    plasticity_growth: float = 1.001
    min_plasticity: float = 0.2
    max_plasticity: float = 5.0
    # Activation Engineering settings
    activation_update_freq: int = 100

    def __post_init__(self):
        """Ensure directories exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)


# ------------------- DATASET CLASS -------------------
import logging
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Any

# Configure logger (assuming setup elsewhere or add basicConfig here)
logger = logging.getLogger(__name__) # Use __name__ for module-level logger

class CompanyDataset(Dataset):
    """Dataset for company information including images."""
    def __init__(self, 
                 csv_path: str, 
                 target_column: str = 'Detailed Description', 
                 exclude_columns: Optional[List[str]] = None,
                 image_column: Optional[str] = None,
                 image_dir: Optional[str] = None,
                 processor=None):
        self.csv_path = csv_path
        self.target_column = target_column
        self.exclude_columns = exclude_columns or []
        self.image_column = image_column
        self.image_dir = image_dir
        self.processor = processor
        self.data = []
        self._load_and_prepare_data()
        
    def _load_and_prepare_data(self):
        """Load and prepare the dataset."""
        try:
            df = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(df)} records from {self.csv_path}")
            
            # Process each row
            for idx, row in df.iterrows():
                formatted_item = self._format_row(row)
                if formatted_item:
                    self.data.append(formatted_item)
            
            logger.info(f"Prepared {len(self.data)} valid items for training")
            
        except Exception as e:
            logger.error(f"Error loading or preparing data: {e}")
            raise
    
    def _format_row(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Format a row into a suitable training example."""
        # Check if target column exists and has content
        if self.target_column not in row or pd.isna(row[self.target_column]) or not row[self.target_column]:
                 return None

        # Create prompt from other columns
        prompt_parts = []
        images = []
        
        for col, value in row.items():
            # Skip excluded columns, empty values, and target column
            if (col in self.exclude_columns or 
                col == self.target_column or 
                pd.isna(value) or 
                not str(value).strip()):
                    continue
            
            # Handle image column if specified
            if self.image_column and col == self.image_column and self.image_dir:
                image_files = str(value).split(',')
                for img_file in image_files:
                    img_file = img_file.strip()
                    if img_file:
                        try:
                            img_path = os.path.join(self.image_dir, img_file)
                            if os.path.exists(img_path):
                                try:
                                    # Add image reference placeholder to the prompt
                                    img_id = len(images) + 1
                                    prompt_parts.append(f"<|image_{img_id}|>")
                                    # Load and store the image
                                    images.append(Image.open(img_path).convert('RGB'))
                                except Exception as img_err:
                                    logger.warning(f"Error processing image {img_path}: {img_err}")
                        except Exception as path_err:
                            logger.warning(f"Error with image path {img_file}: {path_err}")
            else:
                # Format text column with name and value
                prompt_parts.append(f"{col}: {value}")
        
        # Combine all parts into final prompt
        prompt = "\n".join(prompt_parts)
        
        # Create instruction with the prompt
        instruction = f"{prompt}\n\nDescribe the company in detail."
        
        # Return formatted item with target as completion
        return {
            "instruction": instruction,
            "target": str(row[self.target_column]),
            "images": images if images else None
        }
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item by index."""
        return self.data[idx]


# ------------------- DATA COLLATOR -------------------
def custom_collate_fn(batch, tokenizer, processor=None, max_length=512):
    """
    Custom collation function that handles batching with images.
    
    Args:
        batch: List of dictionaries from dataset
        tokenizer: The tokenizer to use
        processor: The vision processor to handle images
        max_length: Maximum sequence length
        
    Returns:
        Processed batch ready for model
    """
    instructions = []
    targets = []
    images_list = []
    
    for item in batch:
        instructions.append(item['instruction'])
        targets.append(item['target'])
        
        # Add images if present
        if 'images' in item and item['images'] is not None:
            images_list.append(item['images'])
        else:
            images_list.append([])
    
    # Prepare for vision model if processor is provided
    if processor is not None:
        # Create chat messages format
        messages = []
        for i, (instruction, images) in enumerate(zip(instructions, images_list)):
            # Create image placeholders
            placeholder = ""
            for j in range(len(images)):
                placeholder += f"<|image_{j+1}|>\n"
            
            # Create message with instruction including image placeholders
            messages.append({
                "role": "user", 
                "content": placeholder + instruction
            })
            
        # Process with vision processor
        inputs_list = []
        for i, msg in enumerate(messages):
            prompt = processor.tokenizer.apply_chat_template(
                [msg], 
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Process images if available
            if images_list[i]:
                inputs = processor(prompt, images_list[i], return_tensors="pt")
            else:
                # Text-only processing
                inputs = processor.tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=max_length, truncation=True)
                
            inputs_list.append(inputs)
        
        # Combine all inputs
        batch_inputs = {}
        for key in inputs_list[0].keys():
            batch_inputs[key] = torch.cat([inp[key] for inp in inputs_list], dim=0)
        
        # Add target labels for training
        if targets:
            output_tokens = processor.tokenizer(targets, return_tensors="pt", padding="max_length", max_length=max_length, truncation=True)
            batch_inputs["labels"] = output_tokens["input_ids"]
        
        return batch_inputs
    
    # Traditional text-only processing
    else:
        # Format as instruction-based format
        formatted_prompts = []
        for instruction in instructions:
            formatted_prompts.append(f"### Instruction:\n{instruction}\n\n### Response:\n")
        
        # Tokenize inputs
        tokenized_inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding="max_length",
        truncation=True,
            max_length=max_length
        )
        
        # Tokenize targets
        if targets:
            tokenized_targets = tokenizer(
                targets,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length
            )
            
            # Create labels
            tokenized_inputs["labels"] = _prepare_targets_for_training(
                tokenized_inputs["input_ids"], 
                tokenized_targets["input_ids"],
                tokenizer
            )
            
        return tokenized_inputs

def _prepare_targets_for_training(input_ids, target_ids, tokenizer):
    """
    Prepare target labels for training by masking the input tokens.
    
    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        target_ids: Target token IDs [batch_size, seq_len]
        tokenizer: The tokenizer
        
    Returns:
        Masked labels tensor with -100 for input tokens
    """
    # Clone target IDs as the starting point for labels
    labels = target_ids.clone()
    
    # Create mask tensor filled with -100
    batch_size, seq_len = input_ids.size()
    mask = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=input_ids.device)
    
    # Find the length of each input sequence (excluding padding)
    input_lens = torch.sum(input_ids != tokenizer.pad_token_id, dim=1)
    
    # For each example in the batch, mask the input tokens in the labels
    for i in range(batch_size):
        # Start with the original target tokens (default to -100 if past the end)
        labels[i] = mask[i]
    
    # Replace pad tokens with -100 in labels
    labels[target_ids == tokenizer.pad_token_id] = -100
    
    return labels


# ------------------- TRAINER CLASS (Minor changes needed)-------------------
class TrainingVisualizer:
    """Utility class for creating training visualizations and metrics reports."""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)
        
    def create_training_summary(self, training_stats, model_name, save_smoothed=True):
        """Create summary visualizations of training progress."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        if not training_stats or len(training_stats) == 0:
            logger.warning("No training statistics available for visualization")
            return {}
            
        # Extract metrics
        steps = [entry.get('step', i) for i, entry in enumerate(training_stats)]
        losses = [entry.get('loss', 0) for entry in training_stats]
        learning_rates = [entry.get('learning_rate', 0) for entry in training_stats]
        
        # Check if we have enough data to create meaningful visualizations
        if len(steps) < 2:
            logger.warning("Not enough training steps to create meaningful visualizations")
            return {}
        
        # Ensure we don't have invalid values
        losses = [float(loss) if not isinstance(loss, float) else loss for loss in losses]
        losses = [min(100, max(0, loss)) for loss in losses]  # Clip to reasonable range
        
        # 1. Loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, 'b-', linewidth=2, label='Loss')
        
        # Add smoothed curve if requested and we have sufficient data
        if save_smoothed and len(steps) > 10:
            try:
                window_size = max(3, min(len(steps) // 10, 20))  # Adaptive but reasonable window size
                smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
                smoothed_steps = steps[window_size-1:]
                
                if len(smoothed_losses) > 0 and len(smoothed_steps) > 0:
                    plt.plot(smoothed_steps, smoothed_losses, 'r-', linewidth=2, alpha=0.7, label='Smoothed')
                    plt.legend()
            except Exception as e:
                logger.warning(f"Error creating smoothed loss curve: {e}")
            
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title(f'Training Loss ({model_name})')
        plt.grid(True, alpha=0.3)
        
        # Handle extreme cases where losses are very similar
        loss_min, loss_max = min(losses), max(losses)
        if abs(loss_max - loss_min) < 1e-5:
            # Create reasonable margin around the values
            y_margin = max(0.1, loss_min * 0.1)
            plt.ylim(max(0, loss_min - y_margin), loss_max + y_margin)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'training_loss.png'), dpi=300)
        plt.close()
        
        # 2. Learning rate plot
        if any(lr != 0 for lr in learning_rates):
            plt.figure(figsize=(10, 4))
            plt.plot(steps, learning_rates, 'g-', linewidth=2)
            plt.xlabel('Training Steps')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'learning_rate.png'), dpi=300)
            plt.close()
        
        # 3. Training dashboard with epochs if available
        epochs = [entry.get('epoch', 0) for entry in training_stats]
        if max(epochs) > 0 and len(set(epochs)) > 1:
            try:
                self._create_epoch_dashboard(steps, losses, learning_rates, epochs, model_name)
            except Exception as e:
                logger.warning(f"Error creating epoch dashboard: {e}")
            
        return {"loss_plot": "training_loss.png", "lr_plot": "learning_rate.png"}
    
    def _create_epoch_dashboard(self, steps, losses, learning_rates, epochs, model_name):
        """Create a dashboard showing epochs and training metrics."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Find epoch boundaries
        boundaries = []
        epoch_nums = sorted(set(epochs))
        
        # Skip if we don't have multiple epochs
        if len(epoch_nums) <= 1:
            return
            
        for epoch_num in epoch_nums[1:]:  # Skip the first epoch boundary (start)
            try:
                # Find first occurrence of epoch
                idx = epochs.index(epoch_num)
                if 0 <= idx < len(steps):
                    boundaries.append((idx, f"Epoch {epoch_num}"))
            except (ValueError, IndexError):
                continue
                
        # Skip if we couldn't find any epoch boundaries
        if not boundaries:
            return
                
        # Create dashboard
        plt.figure(figsize=(12, 8))
        
        # Loss subplot
        plt.subplot(2, 1, 1)
        plt.plot(steps, losses, 'b-', linewidth=2)
        plt.ylabel('Loss')
        plt.title(f'Training Progress Dashboard ({model_name})')
        plt.grid(True, alpha=0.3)
        
        # Add epoch markers
        y_range = max(losses) - min(losses)
        if y_range <= 0:  # Handle flat loss curve
            y_range = 1.0
            
        for idx, label in boundaries:
            if idx < len(steps):  # Safety check
                plt.axvline(x=steps[idx], color='r', linestyle='--', alpha=0.5)
                plt.text(steps[idx], min(losses) + 0.1 * y_range, 
                       label, rotation=90, verticalalignment='bottom')
        
        # Learning rate subplot (only if we have valid learning rates)
        if any(lr > 0 for lr in learning_rates):
            plt.subplot(2, 1, 2)
            plt.plot(steps, learning_rates, 'g-', linewidth=2)
            plt.xlabel('Training Steps')
            plt.ylabel('Learning Rate')
            plt.grid(True, alpha=0.3)
            
            # Add epoch markers
            for idx, label in boundaries:
                if idx < len(steps):  # Safety check
                    plt.axvline(x=steps[idx], color='r', linestyle='--', alpha=0.5)
        else:
            # If no learning rates, adjust the loss plot to take full height
            plt.subplot(2, 1, 1)
            plt.xlabel('Training Steps')
            
        plt.tight_layout()
        try:
            plt.savefig(os.path.join(self.viz_dir, 'training_dashboard.png'), dpi=300)
        except Exception as e:
            logger.warning(f"Error saving training dashboard: {e}")
        finally:
            plt.close()
    
    def create_component_visualization(self, component_name, data, special_type=None):
        """Create component-specific visualization."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        if not data:
            return
            
        # Handle different component types
        if component_name == 'optimizer' and 'plasticity_stats' in data:
            return self.visualize_plasticity(data['plasticity_stats'])
            
        elif component_name == 'architecture':
            return self.visualize_architecture(data)
            
        elif component_name == 'intrinsic_dimension' and 'dimension_history' in data:
            return self.visualize_dimension_history(data)
            
        elif component_name == 'mutual_information' and 'mi_history' in data:
            return self.visualize_mutual_information(data)
            
        elif component_name == 'reward_plasticity':
            return self.visualize_reward_plasticity(data)
            
        elif component_name == 'neural_architecture_search' and 'action_stats' in data:
            return self.visualize_nas_exploration(data)
            
        elif component_name == 'adaptive_distillation':
            return self.visualize_distillation(data)
    
    def visualize_plasticity(self, stats):
        """Visualize plasticity statistics."""
        import matplotlib.pyplot as plt
        
        if not stats:
            return
            
        plt.figure(figsize=(8, 8))
        
        # Create pie chart if we have distribution data
        if 'high_plasticity_pct' in stats and 'low_plasticity_pct' in stats:
            plt.pie([
                stats['high_plasticity_pct'],
                100 - stats['high_plasticity_pct'] - stats['low_plasticity_pct'],
                stats['low_plasticity_pct']
            ], 
            labels=['High Plasticity', 'Medium Plasticity', 'Low Plasticity'],
            autopct='%1.1f%%',
            colors=['#ff9999', '#66b3ff', '#99ff99'],
            startangle=90)
            
            plt.title(f'Parameter Plasticity Distribution (Mean: {stats.get("mean", 0):.4f})')
        else:
            # Simple text display
            plt.text(0.5, 0.5, 
                   f"Mean Plasticity: {stats.get('mean', 0):.4f}\n"
                   f"Min: {stats.get('min', 0):.4f}\n"
                   f"Max: {stats.get('max', 0):.4f}",
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=14,
                   transform=plt.gca().transAxes)
            plt.axis('off')
            plt.title('Plasticity Statistics')
            
        plt.savefig(os.path.join(self.viz_dir, 'plasticity_distribution.png'), dpi=300)
        plt.close()
        
        return {"plasticity_dist": "plasticity_distribution.png"}
    
    def visualize_architecture(self, data):
        """Visualize architecture state and evolution."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        results = {}
        
        # Visualize module ranks
        if 'architecture_state' in data:
            arch_state = data['architecture_state']
            
            # Extract module info
            modules = []
            for name, info in arch_state.items():
                short_name = name[-30:] if len(name) > 30 else name
                modules.append({
                    'name': short_name,
                    'rank': info.get('rank', 0),
                    'params': info.get('params', 0)
                })
                
            # Sort by parameters for better visualization
            modules.sort(key=lambda x: x['params'])
            
            # Create plot
            plt.figure(figsize=(12, max(6, len(modules) * 0.3)))
            
            names = [m['name'] for m in modules]
            ranks = [m['rank'] for m in modules]
            params = [m['params'] for m in modules]
            
            y_pos = np.arange(len(names))
            
            # Plot ranks
            plt.barh(y_pos, ranks, color='skyblue')
            plt.yticks(y_pos, names, fontsize=8)
            plt.xlabel('Rank')
            plt.title('Module Ranks')
            plt.tight_layout()
            
            plt.savefig(os.path.join(self.viz_dir, 'module_ranks.png'), dpi=300)
            plt.close()
            
            results['module_ranks'] = 'module_ranks.png'
        
        # Visualize convergence status
        if 'convergence_status' in data:
            conv = data['convergence_status']
            
            plt.figure(figsize=(8, 5))
            
            # Create a gauge chart
            error = conv.get('cumulative_error', 0)
            budget = conv.get('error_budget', 1)
            pct_used = min(100, error/budget*100) if budget > 0 else 100
            
            # Background bar (full width)
            plt.barh(0, 100, color='lightgray', height=0.6)
            
            # Foreground bar (usage)
            bar_color = 'green' if pct_used < 75 else ('orange' if pct_used < 90 else 'red')
            plt.barh(0, pct_used, color=bar_color, height=0.6)
            
            # Add marker for current position
            plt.scatter(pct_used, 0, color='black', s=100, zorder=5)
            
            # Add labels
            plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            plt.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
            plt.axvline(x=100, color='gray', linestyle='--', alpha=0.5)
            
            plt.xlim(0, 100)
            plt.ylim(-1, 1)
            plt.yticks([])
            plt.xticks([0, 50, 100], ['0%', '50%', '100%'])
            
            plt.title('Convergence Budget Usage')
            
            # Add text details
            plt.figtext(0.5, 0.2, 
                      f"Error: {error:.4e}\nBudget: {budget:.4e}\n"
                      f"Guaranteed: {'Yes' if conv.get('convergence_guaranteed', False) else 'No'}",
                      ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
            
            plt.savefig(os.path.join(self.viz_dir, 'convergence_status.png'), dpi=300)
            plt.close()
            
            results['convergence'] = 'convergence_status.png'
            
        return results
        
    def visualize_dimension_history(self, data):
        """Visualize intrinsic dimension history."""
        import matplotlib.pyplot as plt
        
        dim_history = data['dimension_history']
        decision_history = data.get('decision_history', [0] * len(dim_history))
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(dim_history)), dim_history, 'b-', label='Intrinsic Dimension')
        
        # Add decision markers
        if len(decision_history) == len(dim_history):
            decisions = [d for d in decision_history if d != 0]
            if decisions:  # Only if we have non-zero decisions
                plt.scatter(range(len(decision_history)), 
                         [d * max(dim_history) * 0.1 for d in decision_history], 
                         c=['r' if d < 0 else 'g' if d > 0 else 'gray' for d in decision_history],
                         label='Capacity Decisions')
        
        plt.xlabel('Steps')
        plt.ylabel('Intrinsic Dimension')
        plt.title('Intrinsic Dimension Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.viz_dir, 'intrinsic_dimension.png'), dpi=300)
        plt.close()
        
        return {"dimension": "intrinsic_dimension.png"}
        
    def visualize_mutual_information(self, data):
        """Visualize mutual information history."""
        import matplotlib.pyplot as plt
        
        mi_history = data['mi_history']
        normalized_mi = data.get('normalized_mi', 0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(mi_history)), mi_history, 'b-', linewidth=2)
        
        plt.xlabel('Steps')
        plt.ylabel('Mutual Information')
        plt.title(f'Mutual Information Evolution (Normalized MI: {normalized_mi:.4f})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.viz_dir, 'mutual_information.png'), dpi=300)
        plt.close()
        
        return {"mi": "mutual_information.png"}
        
    def visualize_reward_plasticity(self, data):
        """Visualize reward plasticity data."""
        import matplotlib.pyplot as plt
        
        plasticity_history = data.get('plasticity_history', [])
        reward_history = data.get('reward_history', [])
        
        if not plasticity_history and not reward_history:
            return {}
            
        plt.figure(figsize=(10, 8))
        
        # Plot plasticity
        if plasticity_history:
            plt.subplot(2, 1, 1)
            plt.plot(range(len(plasticity_history)), plasticity_history, 'b-', linewidth=2)
            plt.ylabel('Mean Plasticity')
            plt.title('Reward-Weighted Plasticity Evolution')
            plt.grid(True, alpha=0.3)
            
        # Plot rewards
        if reward_history:
            plt.subplot(2, 1, 2)
            plt.plot(range(len(reward_history)), reward_history, 'g-', linewidth=2)
            plt.xlabel('Step')
            plt.ylabel('Reward')
            plt.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'reward_plasticity.png'), dpi=300)
        plt.close()
        
        return {"rwp": "reward_plasticity.png"}
        
    def visualize_nas_exploration(self, data):
        """Visualize NAS exploration data."""
        import matplotlib.pyplot as plt
        
        action_stats = data.get('action_stats', {})
        rewards = data.get('reward_history', [])
        
        results = {}
        
        # Plot rewards if available
        if rewards:
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(rewards)), rewards, 'b-', linewidth=2)
            plt.xlabel('Step')
            plt.ylabel('Reward')
            plt.title('NAS Exploration Rewards')
            plt.grid(True, alpha=0.3)
            
            plt.savefig(os.path.join(self.viz_dir, 'nas_rewards.png'), dpi=300)
            plt.close()
            
            results['rewards'] = 'nas_rewards.png'
            
        # Plot action exploration
        for action_type, stats in action_stats.items():
            if 'values' in stats and 'counts' in stats:
                values = stats['values']
                counts = stats['counts']
                rewards = stats.get('rewards', [0] * len(values))
                
                plt.figure(figsize=(10, 6))
                
                # Color bars by reward if available
                if rewards and max(rewards) != min(rewards):
                    norm_rewards = [(r - min(rewards)) / (max(rewards) - min(rewards)) for r in rewards]
                    colors = plt.cm.RdYlGn(norm_rewards)
                else:
                    colors = 'skyblue'
                
                plt.bar(range(len(values)), counts, color=colors)
                plt.xticks(range(len(values)), [str(v) for v in values], rotation=45)
                plt.xlabel('Value')
                plt.ylabel('Count')
                plt.title(f'NAS Exploration: {action_type}')
                plt.tight_layout()
                
                plt.savefig(os.path.join(self.viz_dir, f'nas_{action_type}.png'), dpi=300)
                plt.close()
                
                results[action_type] = f'nas_{action_type}.png'
                
        return results
        
    def visualize_distillation(self, data):
        """Visualize distillation data."""
        import matplotlib.pyplot as plt
        
        alpha_history = data.get('alpha_history', [])
        confidence_history = data.get('confidence_history', [])
        
        if not alpha_history and not confidence_history:
            return {}
            
        plt.figure(figsize=(10, 6))
        
        if alpha_history:
            plt.plot(range(len(alpha_history)), alpha_history, 'b-', 
                   label='Distillation Weight (α)')
                   
        if confidence_history:
            plt.plot(range(len(confidence_history)), confidence_history, 'r-', 
                   label='Teacher Confidence')
                   
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.title('Adaptive Distillation Parameters')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig(os.path.join(self.viz_dir, 'distillation.png'), dpi=300)
        plt.close()
        
        return {"distillation": "distillation.png"}
        
    def create_combined_report(self, viz_files, model_name, metadata=None):
        """Create an HTML report combining all visualizations."""
        import datetime
        
        report_path = os.path.join(self.output_dir, "training_report.html")
        
        # Organize visualizations by category
        sections = {
            "Training Progress": ["training_loss.png", "learning_rate.png", "training_dashboard.png"],
            "Architecture": ["module_ranks.png", "convergence_status.png"],
            "Optimization": ["plasticity_distribution.png", "reward_plasticity.png"],
            "Information Theory": ["intrinsic_dimension.png", "mutual_information.png"],
            "Architecture Search": [f for f in viz_files if f.startswith("nas_")],
            "Knowledge Distillation": ["distillation.png"]
        }
        
        # Start HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Training Report - {model_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .section {{ margin-bottom: 30px; }}
                .viz-container {{ display: flex; flex-wrap: wrap; }}
                .viz-item {{ margin: 10px; text-align: center; }}
                .viz-item img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Training Report: {model_name}</h1>
            <p>Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        """
        
        # Add metadata table if provided
        if metadata:
            html += """
            <div class="section">
                <h2>Training Metadata</h2>
                <table>
                    <tr><th>Property</th><th>Value</th></tr>
            """
            
            for key, value in metadata.items():
                html += f"<tr><td>{key}</td><td>{value}</td></tr>"
                
            html += """
                </table>
            </div>
            """
        
        # Add each section
        for section_name, files in sections.items():
            # Check if we have any files for this section
            section_files = [f for f in viz_files if f in files]
            if not section_files:
                continue
                
            html += f"""
            <div class="section">
                <h2>{section_name}</h2>
                <div class="viz-container">
            """
            
            for img_file in section_files:
                img_path = os.path.join("visualizations", img_file)
                name = img_file.replace(".png", "").replace("_", " ").title()
                html += f"""
                <div class="viz-item">
                    <img src="{img_path}" alt="{name}" />
                    <p>{name}</p>
                </div>
                """
                
            html += """
                </div>
            </div>
            """
            
        # Close HTML
        html += """
        </body>
        </html>
        """
        
        # Write HTML file
        with open(report_path, "w") as f:
            f.write(html)
            
        return report_path

class ConvergentMetaMorphTrainer:
    """
    Trainer with integrated next-gen MetaMorph features.
    (Largely reused from metamorph_integrated.py, ensure imports match above)
    """
    def __init__(self, model, tokenizer, config, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.components = {}
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self._initialize_components() # Call component initialization
        logger.info(f"Initialized ConvergentMetaMorphTrainer with device: {self.device}")
        # Add error budget tracking if needed for components
        self.cumulative_architecture_error = 0.0
        self.error_budget = config.error_budget

    def _initialize_components(self):
        """Initialize all the advanced MetaMorph components based on config."""
        logger.info("Initializing MetaMorph components...")
        # --- Metaplasticity Optimizer ---
        if self.config.use_metaplasticity:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            if not trainable_params:
                logger.warning("No trainable parameters found for optimizer!")
                return  # Cannot proceed without trainable params

            # Use standard optimizer with Enhanced Metaplasticity wrapper
            from torch.optim import AdamW
            # Check if AdaFactor is intended
            if self.config.use_adafactor:
                logger.warning("use_adafactor=True but only AdamW wrapper is shown here. Ensure EnhancedAdaFactorWithMetaplasticity is implemented and used if needed.")
            self.components['optimizer'] = EnhancedMetaplasticityOptimizer(
                trainable_params, AdamW,
                lr=self.config.learning_rate,
                plasticity_eta=self.config.plasticity_eta,
                plasticity_decay=self.config.plasticity_decay,
                plasticity_growth=self.config.plasticity_growth,
                min_plasticity=self.config.min_plasticity,
                max_plasticity=self.config.max_plasticity,
                betas=(0.9, 0.999), eps=1e-8,
                weight_decay=self.config.weight_decay
            )
            logger.info("Initialized Enhanced Metaplasticity Optimizer with AdamW base")
        else:
            logger.warning("Metaplasticity is disabled. Ensure a standard optimizer is used elsewhere if needed.")

        # --- Other Components (Initialize based on config flags) ---
        if self.config.use_intrinsic_dimension:
            self.components['dim_minimizer'] = IntrinsicDimensionMinimizer(
                model=self.model, epsilon=self.config.dim_epsilon, window_size=self.config.dim_window_size, dimension_threshold=self.config.dim_threshold)
            logger.info("Initialized Intrinsic Dimension Minimizer")
        if self.config.use_plasticity_reweighter and 'optimizer' in self.components:
            self.components['reweighter'] = PlasticityWeightedReweighter(
                model=self.model, optimizer=self.components['optimizer'], reweighting_strength=self.config.reweighting_strength, max_weight_ratio=self.config.max_weight_ratio, min_weight=self.config.min_weight)
            logger.info("Initialized Plasticity-Weighted Reweighter")
        if self.config.use_dynamic_mi:
            # Pass vocab size if possible
            vocab_size = getattr(self.model.config, 'vocab_size', 50272)  # Get vocab size
            self.components['mi_tracker'] = DynamicMutualInformationTracker(
                model=self.model, vocab_size=vocab_size, num_bins=self.config.mi_num_bins, window_size=self.config.mi_window_size, update_freq=self.config.mi_update_freq, mi_threshold=self.config.mi_threshold)
            logger.info(f"Initialized Dynamic Mutual Information Tracker (vocab size: {vocab_size})")
        if self.config.use_bregman_dynamics:
            self.components['bregman_controller'] = BregmanDynamicsController(
                model=self.model, divergence_type=self.config.divergence_type, step_size_schedule=self.config.step_size_schedule, error_budget=self.config.error_budget)
            logger.info("Initialized Bregman Dynamics Controller")
        if self.config.use_reward_plasticity and 'optimizer' in self.components:
            self.components['rwp_controller'] = RewardWeightedPlasticityController(
                model=self.model, optimizer=self.components['optimizer'], update_rate=self.config.rwp_update_rate, min_plasticity=self.config.rwp_min_plasticity, max_plasticity=self.config.rwp_max_plasticity)
            logger.info("Initialized Reward-Weighted Plasticity Controller")
        
        # Add initializations for NAS
        if self.config.use_convergent_nas:
            # Define proper action space for LoRA ranks and dropout rates
            action_space = {
                'lora_rank': [8, 16, 32],  # Use numerical rank values, not module names
                'hidden_dropout': [0.0, 0.1, 0.2]
            }
            
            self.components['nas'] = ConvergentNeuralArchitectureSearch(
                model=self.model, action_space=action_space, error_budget=self.config.nas_error_budget, 
                exploration_factor=self.config.nas_exploration_factor,
                warmup_steps=self.config.nas_warmup_steps, 
                morph_interval=self.config.nas_morph_interval, 
                optimizer=self.components.get('optimizer')
            )
            logger.info(f"Initialized Convergent Neural Architecture Search with action space: {action_space}")
        
        if self.config.use_activation_engineering:
            self.components['activation_engineer'] = EnhancedActivationEngineering(
                self.model, self.device, update_freq=self.config.activation_update_freq)
            logger.info("Initialized Enhanced Information-Theoretic Activation Engineering")

        # --- Initialize Multimodal Fusion if enabled ---
        if self.config.enable_multimodal_fusion:
            # Define modality dimensions
            # For vision models like Phi-3.5-vision-instruct, we need to map vision and text features
            modality_dims = {
                "text": self.model.config.hidden_size, 
                "image": self.model.config.vision_config.hidden_size if hasattr(self.model.config, 'vision_config') else self.model.config.hidden_size
            }
            
            # Create fusion config
            fusion_config = ModalFusionConfig(
                fusion_method="adaptive_attention",  # Default method
                hidden_dim=512,  # A reasonable size for fusion
                attention_heads=8,
                dropout_rate=0.1,
                dynamic_weighting=True,
                collect_stats=True,
                use_context_gating=True,
                mi_loss_weight=0.1  # Weight for mutual information loss
            )
            
            # Initialize fusion module
            self.components['modal_fusion'] = AdvancedModalFusion(
                config=fusion_config,
                model=self.model,
                modality_dims=modality_dims,
                output_dim=self.model.config.hidden_size,
                device=self.device
            )
            logger.info(f"Initialized Advanced Modal Fusion with method: {fusion_config.fusion_method}")

        # --- Architecture Controller ---
        # Initialize this *after* the optimizer
        if self.config.use_dynamic_architecture:
            self.components['architecture_controller'] = EnhancedConvergentArchitectureController(
                model=self.model, min_rank=self.config.min_lora_rank, max_rank=self.config.max_lora_rank,
                rank_step=self.config.rank_step, error_budget=self.config.error_budget,  # Use main budget
                scan_freq=self.config.architecture_scan_freq, morph_threshold=self.config.morph_threshold,
                optimizer=self.components.get('optimizer')  # Pass optimizer reference
            )
            self.components['architecture_controller'].schedule_alpha = self.config.schedule_alpha
            logger.info("Initialized Enhanced Convergent Architecture Controller")

        # --- Mixed Precision ---
        if self.config.use_mixed_precision and torch.cuda.is_available():
            # Use the updated torch.amp.GradScaler API
            self.components['scaler'] = torch.amp.GradScaler()
            logger.info("Initialized Mixed Precision Training with updated GradScaler")
        
        logger.info("Component initialization complete.")


    def setup_training(self, train_dataset_len, val_dataset_len=None):
        """Set up training components and calculate training steps."""
        config = self.config
        if train_dataset_len is None:
             logger.warning("Training dataset length not provided, using default total_steps=10000")
             total_steps = 10000 * config.epochs
        else:
             effective_batch_size = config.micro_batch_size * config.gradient_accumulation_steps
             steps_per_epoch = math.ceil(train_dataset_len / effective_batch_size)
             total_steps = steps_per_epoch * config.epochs

        self.total_steps = total_steps

        if 'optimizer' in self.components:
            warmup_steps = int(config.warmup_ratio * total_steps)
            self.components['scheduler'] = get_cosine_schedule_with_warmup(
                self.components['optimizer'],
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
            logger.info(f"Initialized scheduler with {warmup_steps} warmup steps and {total_steps} total steps")
        else:
             logger.warning("Optimizer not found, cannot initialize LR scheduler.")

        # Scale error budgets based on total steps if components exist
        if 'architecture_controller' in self.components:
            scaled_error_budget = min(config.error_budget, math.sqrt(total_steps) / 10) # Example scaling
            self.components['architecture_controller'].error_budget = scaled_error_budget
            logger.info(f"Set architecture error budget to {scaled_error_budget:.4f}")
        if 'nas' in self.components:
             scaled_nas_budget = min(config.nas_error_budget, math.sqrt(total_steps) / 10) # Example scaling
             self.components['nas'].error_budget = scaled_nas_budget
             logger.info(f"Set NAS error budget to {scaled_nas_budget:.4f}")
        if 'bregman_controller' in self.components:
             scaled_bregman_budget = min(config.error_budget, math.sqrt(total_steps) / 10) # Use main budget maybe?
             self.components['bregman_controller'].error_budget = scaled_bregman_budget
             logger.info(f"Set Bregman Dynamics error budget to {scaled_bregman_budget:.4f}")


        if 'adaptive_distillation' in self.components:
            self.components['adaptive_distillation'].setup(total_steps=total_steps)
            logger.info(f"Set adaptive distillation total steps to {total_steps}")

        return total_steps

    # --- _prepare_batch is NOT needed here, handled by collate_fn ---

    def _training_step(self, inputs):
        """Execute a single training step with all MetaMorph components."""
        config = self.config
        optimizer = self.components.get('optimizer')
        scaler = self.components.get('scaler')
        use_mixed_precision = scaler is not None and config.use_mixed_precision

        # --- Forward Pass ---
        outputs = None
        loss = None
        distillation_info = {}

        try:
            # Memory optimization: Clear CUDA cache before forward pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if use_mixed_precision:
                # Use updated torch.amp.autocast API
                with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    # Pass inputs directly, assuming collate_fn prepared them
                    outputs = self.model(**inputs)
                    loss = outputs.loss
            else:
                # Pass inputs directly
                outputs = self.model(**inputs)
                loss = outputs.loss

            if loss is None:
                logger.error("Loss is None after model forward pass!")
                return 0.0 # Or raise error

            # Scale loss for gradient accumulation
            accum_loss = loss / config.gradient_accumulation_steps

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"CUDA OOM during forward pass. Try reducing batch size or model size: {e}")
                # Try to free memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Return a valid but zero loss to continue training
                return 0.0
            else:
             logger.error(f"Error during forward pass: {e}", exc_info=True)
             return 0.0 # Return 0 loss on error to prevent crash, or raise

        # --- Backward Pass ---
        try:
            if use_mixed_precision:
                scaler.scale(accum_loss).backward()
            else:
                accum_loss.backward()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"CUDA OOM during backward pass. Try reducing batch size or model size: {e}")
                # Try to free memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return loss.item() if loss is not None else 0.0
            else:
                logger.error(f"Error during backward pass: {e}", exc_info=True)
                # Gradients might be None now, handle downstream
                return loss.item() if loss is not None else 0.0 # Return original unscaled loss

        # --- Optimizer Step (if not accumulating) ---
        should_step = (self.global_step + 1) % config.gradient_accumulation_steps == 0
        if should_step and optimizer:
            try:
                if use_mixed_precision:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], config.max_grad_norm)
                    optimizer.step()

                # Step the scheduler *after* the optimizer
                if self.components.get('scheduler'):
                    self.components['scheduler'].step()

                # Zero gradients *after* stepping
                optimizer.zero_grad(set_to_none=True) # More efficient

                # Memory optimization: Clear CUDA cache after step
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"CUDA OOM during optimizer step. Try reducing batch size or model size: {e}")
                    # Try to free memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                 logger.error(f"Error during optimizer step or gradient clipping: {e}", exc_info=True)

        return loss.item() if loss is not None else 0.0 # Return original unscaled loss


    # --- Keep train(), evaluate(), save/load_checkpoint(), _log_training_progress(), _check_architecture_morphing(), _generate_component_reports() ---
    # --- Ensure async/await is removed if not using CoT injector ---
    # --- Replace async def train/train_step/evaluate with regular def ---
    # --- Remove _maybe_apply_cot call if not using CoT ---

    def train(self, train_dataloader, val_dataloader=None, callbacks=None):
        """Train model on provided data and evaluate if validation data is provided."""
        config = self.config
        
        # Get dataset lengths if available
        train_len = len(train_dataloader.dataset) if hasattr(train_dataloader, 'dataset') else None
        val_len = len(val_dataloader.dataset) if val_dataloader and hasattr(val_dataloader, 'dataset') else None
        
        # Setup training components and get total steps
        total_steps = self.setup_training(train_len, val_len)

        # Calculate steps per epoch if total_steps is not provided
        if total_steps is None:
            steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
            total_steps = steps_per_epoch * config.epochs
        
        # Initialize tracking variables
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.training_stats = []  # Initialize training stats list

        logger.info(f"Starting training for {config.epochs} epochs, ~{total_steps} total steps")

        for epoch in range(config.epochs):
            self.epoch = epoch
            self.model.train() # Set model to train mode
            epoch_loss = 0.0
            processed_samples = 0

            # Setup progress bar (optional but helpful)
            from tqdm.auto import tqdm
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False)
            
            # Clear CUDA cache at start of epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for step, batch in enumerate(progress_bar):
                if not batch: # Skip empty batches from collate_fn errors
                     logger.warning(f"Skipping empty batch at step {step}")
                     continue

                # Prepare batch (move to device)
                try:
                    # Memory optimization: Move tensors to device one by one and clear them from CPU
                    inputs = {}
                    for k, v in batch.items():
                        if hasattr(v, 'to'):
                            # Move to device and ensure CPU data is freed
                            inputs[k] = v.to(self.device, non_blocking=True)
                            # Free CPU memory
                            del v
                    del batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.error(f"CUDA OOM while moving batch to device. Skipping batch: {e}")
                        # Clear any partial data
                        inputs = {}
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        logger.error(f"Error preparing batch: {e}", exc_info=True)
                        continue
                    
                if not inputs:
                    logger.warning(f"Skipping batch with no tensors after moving to device at step {step}")
                    continue

                # Perform training step
                loss = self._training_step(inputs) # Call the refactored step function
                
                # Free up memory from inputs after training step
                del inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # Update epoch stats
                batch_size = config.micro_batch_size  # Use configured batch size for stats
                epoch_loss += loss * batch_size
                processed_samples += batch_size

                # Check if an optimizer step happened
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    self.global_step += 1

                    # Optional: Update progress bar description
                    lr = self.components['optimizer'].param_groups[0]['lr'] if 'optimizer' in self.components else 0
                    progress_bar.set_postfix({'loss': f"{loss:.3f}", 'lr': f"{lr:.2e}", 'step': self.global_step})

                    # Check for architecture morphing
                    if config.use_dynamic_architecture:
                        self._check_architecture_morphing()

                    # Logging
                    if self.global_step % config.logging_steps == 0:
                        self._log_training_progress(loss)
                        # Save stats for visualization
                        self.training_stats.append({
                            'step': self.global_step, 
                            'epoch': epoch + 1, 
                            'loss': loss,
                            'learning_rate': lr
                        })

                    # Validation
                    if val_dataloader is not None and self.global_step > 0 and self.global_step % config.eval_steps == 0:
                        val_loss = self.evaluate(val_dataloader) # Use non-async evaluate
                        logger.info(f"Step {self.global_step} Validation loss: {val_loss:.4f}")
                        self.model.train() # Ensure model is back in training mode
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint(
                                os.path.join(config.output_dir, "checkpoints", "best_model"),
                                {'val_loss': val_loss, 'step': self.global_step}
                            )

                    # Save periodic checkpoint
                    if self.global_step > 0 and self.global_step % config.save_steps == 0:
                        self.save_checkpoint(
                            os.path.join(config.output_dir, "checkpoints", f"step_{self.global_step}"),
                            {'step': self.global_step}
                        )

            # End of epoch
            avg_epoch_loss = epoch_loss / processed_samples if processed_samples > 0 else 0
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")

            if val_dataloader is not None:
                val_loss = self.evaluate(val_dataloader)
                logger.info(f"End of Epoch {epoch+1} Validation loss: {val_loss:.4f}")
                self.model.train()
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(
                        os.path.join(config.output_dir, "checkpoints", "best_model"),
                        {'val_loss': val_loss, 'step': self.global_step, 'epoch': epoch+1}
                    )

            self.save_checkpoint(
                os.path.join(config.output_dir, "checkpoints", f"epoch_{epoch+1}"),
                {'step': self.global_step, 'epoch': epoch+1}
            )
            progress_bar.close()


        logger.info(f"Training complete - {self.global_step} steps across {config.epochs} epochs")
        self._generate_component_reports()
        self.save_checkpoint(
            os.path.join(config.output_dir, "final_model"),
            {'step': self.global_step, 'epochs': config.epochs}
        )
        logger.info(f"Final model saved to {os.path.join(config.output_dir, 'final_model')}")

        return {'training_stats': self.training_stats, 'best_val_loss': self.best_val_loss, 'final_step': self.global_step}

    def evaluate(self, eval_dataloader):
        """Evaluate model (Non-async version)."""
        self.model.eval() # Set model to evaluation mode
        total_loss = 0
        total_samples = 0

        # Setup progress bar (optional)
        from tqdm.auto import tqdm
        progress_bar = tqdm(eval_dataloader, desc="Evaluating", leave=False)

        with torch.no_grad():
            for batch in progress_bar:
                if not batch: continue # Skip empty batches
                inputs = {k: v.to(self.device) for k, v in batch.items() if hasattr(v, 'to')}
                if not inputs: continue

                try:
                    outputs = self.model(**inputs)
                    loss = outputs.loss
                    if loss is not None:
                        batch_size = inputs['input_ids'].shape[0]
                        total_loss += loss.item() * batch_size
                        total_samples += batch_size
                except Exception as e:
                     logger.error(f"Error during evaluation step: {e}")

        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        progress_bar.close()
        self.model.train() # Set model back to training mode
        return avg_loss

    # --- Need _log_training_progress, _check_architecture_morphing ---
    # --- save/load_checkpoint, _generate_component_reports methods ---
    # --- (Copy these from the original metamorph_integrated.py or adapt) ---
    # --- Ensure all methods called by train() exist and are non-async ---

    # Placeholder implementations for methods assumed from original script:
    def _log_training_progress(self, loss):
        # Simplified logging (expand as in original)
        lr = self.components['optimizer'].param_groups[0]['lr'] if 'optimizer' in self.components else 0
        logger.info(f"Step {self.global_step}: Loss={loss:.4f}, LR={lr:.2e}")

    def _check_architecture_morphing(self):
         # Simplified check (expand as in original)
         if 'architecture_controller' in self.components:
             arch_ctrl = self.components['architecture_controller']
             scan_results = arch_ctrl.scan_architecture(self.global_step)
             if scan_results and scan_results.get('morph_needed', False):
                 morph_plan = arch_ctrl.morph_architecture(scan_results)
                 if morph_plan:
                     logger.info(f"Architecture controller morphed architecture at step {self.global_step}")
         if 'nas' in self.components:
             nas = self.components['nas']
             if nas.should_morph_architecture():
                 modifications, error = nas.morph_architecture()
                 if modifications:
                      logger.info(f"NAS morphed architecture: {modifications}")


    def save_checkpoint(self, save_path, info=None):
        """Save model checkpoint with components state."""
        try:
            os.makedirs(save_path, exist_ok=True)
            logger.info(f"Saving checkpoint to {save_path}...")

            # Save model using save_pretrained
            # If using PEFT, save adapter; otherwise, save full model
            if hasattr(self.model, "save_pretrained"):
                 self.model.save_pretrained(save_path)
            else:
                 torch.save(self.model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))

            # Save tokenizer
            if self.tokenizer:
                self.tokenizer.save_pretrained(save_path)

            # Save component states (optimizer, scheduler, etc.)
            component_states = {}
            if 'optimizer' in self.components:
                component_states['optimizer_state_dict'] = self.components['optimizer'].state_dict()
            if 'scheduler' in self.components:
                component_states['scheduler_state_dict'] = self.components['scheduler'].state_dict()
            # Add other component states if they have saveable state_dict() methods

            # Save training info and component states
            if info is None: info = {}
            checkpoint_data = {
                'epoch': self.epoch, 'global_step': self.global_step,
                'best_val_loss': self.best_val_loss, 'config': self.config, # Save config
                'component_states': component_states, **info
            }
            torch.save(checkpoint_data, os.path.join(save_path, "trainer_state.pt")) # Use .pt extension

            logger.info(f"Checkpoint saved successfully to {save_path}")
        except Exception as e:
            logger.error(f"Error saving checkpoint to {save_path}: {e}", exc_info=True)


    def load_checkpoint(self, load_path):
        """Load checkpoint with component states."""
        logger.info(f"Attempting to load checkpoint from {load_path}...")
        # Note: Model loading should happen *before* trainer initialization typically.
        # This method assumes model structure matches checkpoint.
        state_path = os.path.join(load_path, "trainer_state.pt")
        if not os.path.exists(state_path):
             logger.error(f"Trainer state file not found: {state_path}")
             return False

        try:
            checkpoint_data = torch.load(state_path, map_location=self.device)

            # Load trainer state
            self.epoch = checkpoint_data.get('epoch', 0)
            self.global_step = checkpoint_data.get('global_step', 0)
            self.best_val_loss = checkpoint_data.get('best_val_loss', float('inf'))
            # Could potentially load and verify config here if needed:
            # self.config = checkpoint_data.get('config', self.config)

            # Load component states
            component_states = checkpoint_data.get('component_states', {})
            if 'optimizer' in self.components and 'optimizer_state_dict' in component_states:
                self.components['optimizer'].load_state_dict(component_states['optimizer_state_dict'])
                logger.info("Optimizer state loaded.")
            if 'scheduler' in self.components and 'scheduler_state_dict' in component_states:
                self.components['scheduler'].load_state_dict(component_states['scheduler_state_dict'])
                logger.info("Scheduler state loaded.")
            # Load other component states if implemented

            logger.info(f"Checkpoint loaded successfully from {load_path} (Step: {self.global_step})")
            return True
        except Exception as e:
             logger.error(f"Error loading checkpoint from {load_path}: {e}", exc_info=True)
             return False

    def _generate_component_reports(self):
         # Simplified report generation (expand as in original)
         logger.info("Generating final component reports...")
         
         try:
             # Import visualization libraries
             import matplotlib.pyplot as plt
             import json
             import numpy as np
             from datetime import datetime
             
             # Initialize reports dictionary
             reports = {}
             
             # Create report directory
             report_dir = os.path.join(self.config.output_dir, "component_reports")
             os.makedirs(report_dir, exist_ok=True)
             
             # Create visualizer
             visualizer = TrainingVisualizer(report_dir)
             generated_visualizations = []
             
             # 1. Optimizer and Metaplasticity report
             if 'optimizer' in self.components:
                 try:
                     if hasattr(self.components['optimizer'], 'get_stability_report'):
                         reports['optimizer'] = self.components['optimizer'].get_stability_report()
                     elif hasattr(self.components['optimizer'], 'get_plasticity_stats'):
                         reports['optimizer'] = {
                             'plasticity_stats': self.components['optimizer'].get_plasticity_stats()
                         }
                 except Exception as e:
                     logger.warning(f"Error getting optimizer stats: {e}")
             
             # 2. Architecture Controller report
             if 'architecture_controller' in self.components:
                 try:
                     if hasattr(self.components['architecture_controller'], 'get_architecture_report'):
                         reports['architecture'] = self.components['architecture_controller'].get_architecture_report()
                     
                     # Verify convergence guarantee
                     arch_ctrl = self.components['architecture_controller']
                     if hasattr(arch_ctrl, 'cumulative_error') and hasattr(arch_ctrl, 'error_budget'):
                         if arch_ctrl.cumulative_error <= arch_ctrl.error_budget:
                             logger.info(f"✓ Convergence guarantee satisfied: "
                                      f"Total error {arch_ctrl.cumulative_error:.4e} <= "
                                      f"Budget {arch_ctrl.error_budget:.4e}")
                         else:
                             logger.warning(f"⚠ Convergence guarantee not satisfied: "
                                         f"Total error {arch_ctrl.cumulative_error:.4e} > "
                                         f"Budget {arch_ctrl.error_budget:.4e}")
                 except Exception as e:
                     logger.warning(f"Error getting architecture stats: {e}")
             
             # 3. Intrinsic Dimension report
             if 'dim_minimizer' in self.components:
                 try:
                     if hasattr(self.components['dim_minimizer'], 'get_stats'):
                         reports['intrinsic_dimension'] = self.components['dim_minimizer'].get_stats()
                 except Exception as e:
                     logger.warning(f"Error getting dimension stats: {e}")
             
             # 4. Mutual Information report
             if 'mi_tracker' in self.components:
                 try:
                     if hasattr(self.components['mi_tracker'], 'get_stats'):
                         reports['mutual_information'] = self.components['mi_tracker'].get_stats()
                 except Exception as e:
                     logger.warning(f"Error getting MI stats: {e}")
             
             # 5. Bregman Dynamics report
             if 'bregman_controller' in self.components:
                 try:
                     if hasattr(self.components['bregman_controller'], 'get_stats'):
                         reports['bregman_dynamics'] = self.components['bregman_controller'].get_stats()
                 except Exception as e:
                     logger.warning(f"Error getting Bregman dynamics stats: {e}")
             
             # 6. Reward-Weighted Plasticity report
             if 'rwp_controller' in self.components:
                 try:
                     if hasattr(self.components['rwp_controller'], 'get_stats'):
                         reports['reward_plasticity'] = self.components['rwp_controller'].get_stats()
                 except Exception as e:
                     logger.warning(f"Error getting reward plasticity stats: {e}")
             
             # 7. Adaptive Distillation report
             if 'adaptive_distillation' in self.components:
                 try:
                     if hasattr(self.components['adaptive_distillation'], 'get_stats'):
                         reports['adaptive_distillation'] = self.components['adaptive_distillation'].get_stats()
                 except Exception as e:
                     logger.warning(f"Error getting distillation stats: {e}")
             
             # 8. Neural Architecture Search report
             if 'nas' in self.components:
                 try:
                     if hasattr(self.components['nas'], 'get_architecture_exploration_report'):
                         reports['neural_architecture_search'] = self.components['nas'].get_architecture_exploration_report()
                 except Exception as e:
                     logger.warning(f"Error getting NAS stats: {e}")
             
             # 9. Chain-of-Thought Injection report
             if 'cot_injector' in self.components:
                 try:
                     if hasattr(self.components['cot_injector'], 'get_stats'):
                         reports['cot_injection'] = self.components['cot_injector'].get_stats()
                     
                     if 'prompt_router' in self.components and hasattr(self.components['prompt_router'], 'get_stats'):
                         reports['prompt_router'] = self.components['prompt_router'].get_stats()
                 except Exception as e:
                     logger.warning(f"Error getting CoT stats: {e}")
             
             # 10. Activation Engineering report
             if 'activation_engineer' in self.components:
                 try:
                     if hasattr(self.components['activation_engineer'], 'get_activation_selection_report'):
                         reports['activation_engineering'] = self.components['activation_engineer'].get_activation_selection_report()
                 except Exception as e:
                     logger.warning(f"Error getting activation engineering stats: {e}")
             
             # Get training stats if available from the train() return value
             training_stats = getattr(self, 'training_stats', [])
             
             # Visualize training metrics if available
             if training_stats:
                 try:
                     training_viz = visualizer.create_training_summary(
                         training_stats, 
                         self.config.model_name
                     )
                     if training_viz:
                         generated_visualizations.extend(training_viz.values())
                 except Exception as e:
                     logger.warning(f"Error visualizing training stats: {e}")
             
             # Save reports to output directory and create visualizations
             for name, report in reports.items():
                 try:
                     # Convert any non-serializable values to strings
                     serializable_report = {}
                     for k, v in report.items():
                         if isinstance(v, (dict, list, str, int, float, bool)) or v is None:
                             serializable_report[k] = v
                         else:
                             serializable_report[k] = str(v)
                     
                     # Save report as JSON
                     with open(os.path.join(report_dir, f"{name}_report.json"), 'w') as f:
                         json.dump(serializable_report, f, indent=2)
                     
                     # Create visualizations for this component
                     try:
                         component_viz = visualizer.create_component_visualization(name, report)
                         if component_viz:
                             generated_visualizations.extend(component_viz.values())
                     except Exception as e:
                         logger.warning(f"Error creating visualizations for {name}: {e}")
                 except Exception as e:
                     logger.warning(f"Error saving report for {name}: {e}")
             
             # Create metadata for the HTML report
             lr_value = 'N/A'
             if training_stats and len(training_stats) > 0:
                 lr_value = training_stats[-1].get('learning_rate', 'N/A')
             
             metadata = {
                 "Model": self.config.model_name,
                 "Training Steps": self.global_step,
                 "Epochs": self.epoch + 1 if hasattr(self, 'epoch') else 1,
                 "Best Validation Loss": f"{getattr(self, 'best_val_loss', float('inf')):.4f}",
                 "Final Learning Rate": lr_value,
                 "Batch Size": self.config.micro_batch_size,
                 "Gradient Accumulation": self.config.gradient_accumulation_steps,
                 "Effective Batch Size": self.config.micro_batch_size * self.config.gradient_accumulation_steps,
                 "Active Components": ", ".join(reports.keys())
             }
             
             # Generate HTML report
             try:
                 report_path = visualizer.create_combined_report(
                     generated_visualizations,
                     self.config.model_name,
                     metadata
                 )
                 logger.info(f"Generated training report at {report_path}")
             except Exception as e:
                 logger.warning(f"Error creating combined visualization report: {e}")
         
         except Exception as e:
             logger.error(f"Error generating component reports: {e}", exc_info=True)
         
         return reports


# ------------------- MAIN EXECUTION -------------------
def main():
    # --- Configuration ---
    config = ConvergentMetaMorphConfig()
    
    # --- Setup logging ---
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.output_dir, "training.log")),
            logging.StreamHandler()
        ]
    )
    logger.info(f"Starting with config: {config}")
    
    # --- NVIDIA L4 optimizations ---
    logger.info("Setting up optimizations for NVIDIA L4 GPU")
    
    # Enable TF32 for faster matrix multiplications on L4
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # --- Seed for reproducibility ---
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        # Report GPU info
        device_props = torch.cuda.get_device_properties(0)
        logger.info(f"Using GPU: {device_props.name} with {device_props.total_memory / 1024**3:.2f} GB memory")

    # --- Model Initialization ---
    logger.info(f"Loading model: {config.model_name}")

    # Set computation type based on config
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
    
    # Additional model args for L4 optimizations
    model_args = {
        "device_map": config.device,
        "trust_remote_code": True,
        "_attn_implementation": 'flash_attention_2' if config.use_flash_attention else 'eager',
        "use_cache": True  # Enable KV caching for inference
    }
    
    # Load model with quantization if enabled
    if config.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.use_nested_quant,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=quantization_config,
            **model_args
        )
    else:
            model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.fp16_training else "auto",
            **model_args
        )
    
    # Get tokenizer and processor
    logger.info("Loading tokenizer and processor")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    
    # Initialize processor for vision model with optimized settings
    processor = AutoProcessor.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        num_crops=config.num_image_crops,
    )
    
    # Set padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # Prepare model for k-bit training if using PEFT
    if config.use_peft:
        logger.info("Preparing model for PEFT (Parameter-Efficient Fine-Tuning)")
        model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA with L4-optimized settings
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply PEFT
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # --- Data preparation ---
    logger.info(f"Loading dataset from {config.data_path}")
    
    # Initialize dataset with image support
    train_dataset = CompanyDataset(
            csv_path=config.data_path,
        target_column='Detailed Description',
        exclude_columns=['Contact Detail', 'Email ID'],
        image_column='Images',  # Specify the column that contains image filenames
        image_dir='./company_images',  # Directory containing images
        processor=processor  # Pass the processor for image handling
    )
    
    # Create DataLoader with custom collation and L4-optimized batch size
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.micro_batch_size,
        shuffle=True,
        collate_fn=lambda batch: custom_collate_fn(batch, tokenizer, processor, config.max_seq_length),
        num_workers=4,  # L4 can handle more parallel workers
        pin_memory=True  # Enable pinned memory for faster data transfer to GPU
    )
    
    # Optionally create validation dataloader
    val_dataloader = None
    
    # --- Initialize Trainer ---
    trainer = ConvergentMetaMorphTrainer(model, tokenizer, config)

    # If dynamic MI is enabled, provide vocab size
    if config.use_dynamic_mi:
        trainer.vocab_size = len(tokenizer)
    
    # Make processor available to trainer
    trainer.processor = processor
    
    # Setup training with L4 optimizations
    trainer.setup_training(len(train_dataset))
    
    # --- Clear CUDA cache before training ---
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # --- Start Training ---
    logger.info("Starting training with L4-optimized settings")
    try:
        trainer.train(train_dataloader, val_dataloader)
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Error during training: {e}")
    finally:
        # Clean up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()