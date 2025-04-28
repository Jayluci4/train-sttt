"""
Model Factory for ConvergentMetaMorph

This module provides functions to load and initialize models, tokenizers,
and processors for different model architectures.
"""

import torch
import logging
from typing import Dict, Any, Union, Optional, List

# Import necessary PyTorch and Transformers classes
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoProcessor
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelFactory")

def load_model(config: Any) -> torch.nn.Module:
    """
    Load a model based on configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Initialized model
    """
    logger.info(f"Loading model: {config.model_name}")
    
    # Check if the model is a vision model
    is_vision_model = any(name in config.model_name.lower() for name in ["vision", "clip", "blip", "phi-3.5-vision"])
    
    # Prepare model arguments
    model_args = {
        "device_map": config.device,
        "trust_remote_code": True,
    }
    
    # Add flash attention if supported
    if hasattr(config, "use_flash_attention") and config.use_flash_attention:
        model_args["_attn_implementation"] = 'flash_attention_2'
    
    # Add cache settings if available
    if hasattr(config, "use_cache"):
        model_args["use_cache"] = config.use_cache
    
    # Prepare quantization config if enabled
    if hasattr(config, "use_4bit") and config.use_4bit:
        logger.info("Using 4-bit quantization")
        compute_dtype = torch.float16
        if hasattr(config, "bnb_4bit_compute_dtype"):
            if config.bnb_4bit_compute_dtype == "float16":
                compute_dtype = torch.float16
            elif config.bnb_4bit_compute_dtype == "bfloat16":
                compute_dtype = torch.bfloat16
                
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type if hasattr(config, "bnb_4bit_quant_type") else "nf4",
            bnb_4bit_use_double_quant=config.use_nested_quant if hasattr(config, "use_nested_quant") else False
        )
        model_args["quantization_config"] = quantization_config
    else:
        # Handle FP16 training
        if hasattr(config, "fp16_training") and config.fp16_training:
            model_args["torch_dtype"] = torch.float16
        else:
            model_args["torch_dtype"] = "auto"
    
    # Load the model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **model_args
        )
        logger.info(f"Model loaded successfully: {type(model).__name__}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    # Apply PEFT if enabled
    if hasattr(config, "use_peft") and config.use_peft:
        logger.info("Applying PEFT (Parameter-Efficient Fine-Tuning)")
        
        # Prepare model for k-bit training if using quantization
        if hasattr(config, "use_4bit") and config.use_4bit:
            model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        target_modules = config.lora_target_modules if hasattr(config, "lora_target_modules") else None
        
        peft_config = LoraConfig(
            r=config.lora_r if hasattr(config, "lora_r") else 8,
            lora_alpha=config.lora_alpha if hasattr(config, "lora_alpha") else 16,
            lora_dropout=config.lora_dropout if hasattr(config, "lora_dropout") else 0.05,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply PEFT
        model = get_peft_model(model, peft_config)
        logger.info("PEFT applied successfully")
        
        # Print trainable parameters
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
    
    return model

def initialize_tokenizer(config: Any) -> Optional[Any]:
    """
    Initialize a tokenizer based on configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Initialized tokenizer
    """
    logger.info(f"Initializing tokenizer for {config.model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )
        
        # Set padding token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        logger.info("Tokenizer initialized successfully")
        return tokenizer
    except Exception as e:
        logger.error(f"Error initializing tokenizer: {e}")
        raise

def initialize_processor(config: Any) -> Optional[Any]:
    """
    Initialize a processor for vision models.
    
    Args:
        config: Configuration object
        
    Returns:
        Initialized processor or None if not applicable
    """
    # Check if the model is a vision model
    is_vision_model = any(name in config.model_name.lower() for name in ["vision", "clip", "blip", "phi-3.5-vision"])
    
    if not is_vision_model:
        logger.info("Model is not a vision model, processor not initialized")
        return None
        
    logger.info(f"Initializing processor for {config.model_name}")
    
    try:
        processor_args = {
            "trust_remote_code": True
        }
        
        # Add additional processor args if available
        if hasattr(config, "num_image_crops"):
            processor_args["num_crops"] = config.num_image_crops
            
        processor = AutoProcessor.from_pretrained(
            config.model_name,
            **processor_args
        )
        
        logger.info("Processor initialized successfully")
        return processor
    except Exception as e:
        logger.error(f"Error initializing processor: {e}")
        if is_vision_model:
            raise
        return None

def create_optimizer(model: torch.nn.Module, config: Any) -> torch.optim.Optimizer:
    """
    Create an optimizer for the model.
    
    Args:
        model: The model to optimize
        config: Configuration object
        
    Returns:
        Optimizer
    """
    # Get trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if not trainable_params:
        logger.warning("No trainable parameters found!")
        return None
    
    logger.info(f"Creating optimizer with learning rate {config.learning_rate}")
    
    # Check if we should use special optimizer
    if hasattr(config, "use_fused_adam") and config.use_fused_adam and torch.cuda.is_available():
        try:
            from apex.optimizers import FusedAdam
            optimizer = FusedAdam(
                trainable_params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay if hasattr(config, "weight_decay") else 0.01
            )
            logger.info("Using FusedAdam optimizer")
        except ImportError:
            logger.warning("apex not installed, falling back to AdamW")
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay if hasattr(config, "weight_decay") else 0.01
            )
    else:
        # Default to AdamW
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay if hasattr(config, "weight_decay") else 0.01
        )
        logger.info("Using AdamW optimizer")
    
    return optimizer 