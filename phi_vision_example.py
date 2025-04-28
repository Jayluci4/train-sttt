from PIL import Image 
import requests 
import torch
import time
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor

# Optimize for NVIDIA L4 GPU with 24GB VRAM
print("Setting up optimizations for NVIDIA L4 GPU...")

# L4-optimized model configuration
model_id = "microsoft/Phi-3.5-vision-instruct" 
device = "cuda"

# Performance configuration for L4
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster matrix multiplications
torch.backends.cudnn.benchmark = True  # Use cuDNN benchmarking for faster convolutions

# Note: L4 supports flash attention 2, which is faster
model = AutoModelForCausalLM.from_pretrained(
  model_id, 
  device_map=device,
  trust_remote_code=True, 
  torch_dtype=torch.float16,  # Use float16 for faster performance on L4
  _attn_implementation='flash_attention_2',
  use_cache=True  # Enable KV caching for faster generation
)

# L4 can handle more crops (8-16) for better quality
processor = AutoProcessor.from_pretrained(
  model_id, 
  trust_remote_code=True, 
  num_crops=8  # Increased for better quality on L4
) 

# Batch processing example - process multiple queries in parallel for L4
def process_batch(image_urls, queries, batch_size=2):
    """Process multiple image-query pairs in parallel batches"""
    results = []
    total_time = 0
    
    # Process in batches
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i+batch_size]
        batch_images = []
        
        # Load images for this batch
        for urls in image_urls[i:i+batch_size]:
            images = []
            for url in urls:
                try:
                    img = Image.open(requests.get(url, stream=True).raw).convert('RGB')
                    images.append(img)
                except Exception as e:
                    print(f"Error loading image {url}: {e}")
            batch_images.append(images)
        
        # Create message format
        batch_messages = []
        for j, (images, query) in enumerate(zip(batch_images, batch_queries)):
            # Create image placeholders
            placeholder = ""
            for k in range(len(images)):
                placeholder += f"<|image_{k+1}|>\n"
            
            # Add query
            batch_messages.append({"role": "user", "content": placeholder + query})
        
        # Process each message with its images
        batch_inputs_list = []
        for msg_idx, message in enumerate(batch_messages):
            prompt = processor.tokenizer.apply_chat_template(
                [message], 
                tokenize=False, 
                add_generation_prompt=True
            )
            inputs = processor(prompt, batch_images[msg_idx], return_tensors="pt")
            batch_inputs_list.append(inputs)
        
        # Consolidate inputs for batched processing
        consolidated_inputs = {}
        for key in batch_inputs_list[0].keys():
            consolidated_inputs[key] = torch.cat([inp[key] for inp in batch_inputs_list], dim=0).to(device)
        
        # Optimized generation for L4
        start_time = time.time()
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model.generate(
                **consolidated_inputs,
                max_new_tokens=1000,
                temperature=0.2,  # Slight randomness for better results
                do_sample=True,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
        
        batch_time = time.time() - start_time
        total_time += batch_time
        
        # Process outputs
        for j in range(len(batch_queries)):
            # Extract the generated text for this item
            input_length = consolidated_inputs['input_ids'][j].shape[0]
            generated_ids = outputs[j, input_length:]
            
            response = processor.tokenizer.decode(
                generated_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            results.append(response)
            
        print(f"Processed batch of {len(batch_queries)} in {batch_time:.2f} seconds")
    
    print(f"Total processing time: {total_time:.2f} seconds for {len(queries)} queries")
    return results

# Example usage with multiple slides and queries
image_urls = [
    # First query images
    [f"https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-{i}-2048.jpg" 
     for i in range(1, 10)],
    
    # Second query images
    [f"https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-{i}-2048.jpg" 
     for i in range(10, 20)]
]

queries = [
    "Summarize these Azure introduction slides.",
    "What are the key benefits of Azure mentioned in these slides?"
]

# Execute batch processing - L4 GPU can handle this efficiently
results = process_batch(image_urls, queries, batch_size=2)

# Print results
for i, result in enumerate(results):
    print(f"\nQuery {i+1}: {queries[i]}")
    print("-" * 40)
    print(result)
    print("-" * 40)

# Single query example (original implementation)
print("\nRunning single query example...")

images = []
placeholder = ""

# Limit to fewer slides for single example
for i in range(1, 10):
    url = f"https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-{i}-2048.jpg" 
    images.append(Image.open(requests.get(url, stream=True).raw))
    placeholder += f"<|image_{i}|>\n"

messages = [
    {"role": "user", "content": placeholder+"What are the main cloud computing models shown in these slides?"},
]

prompt = processor.tokenizer.apply_chat_template(
  messages, 
  tokenize=False, 
  add_generation_prompt=True
)

inputs = processor(prompt, images, return_tensors="pt").to(device)

# Optimized generation settings for L4
with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
    start_time = time.time()
    generate_ids = model.generate(
        **inputs, 
        max_new_tokens=1000,
        temperature=0.2,
        do_sample=True,
        top_p=0.95,
        repetition_penalty=1.1,
        eos_token_id=processor.tokenizer.eos_token_id
    )
    gen_time = time.time() - start_time

# Remove input tokens 
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(
    generate_ids, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False
)[0]

print(f"Generation completed in {gen_time:.2f} seconds")
print("\nResponse:")
print("-" * 40)
print(response)
print("-" * 40) 