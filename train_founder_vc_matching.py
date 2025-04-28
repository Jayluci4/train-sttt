#!/usr/bin/env python3
"""
MetaMorph Phase 2 Training Script for Founder-VC Matching

This script demonstrates how to use the MetaMorph Phase 2 training pipeline
to train a model for founder-VC matching. It includes:

1. Data loading and preprocessing
2. Model initialization with PEFT (LoRA)
3. Configuration of all Phase 2 components:
   - STTT Cycle
   - Dynamic Curriculum
   - Hard Example Amplification
   - Latent Space Regularization
4. Training with the integrated pipeline
5. Evaluation and reporting
"""

import os
import torch
import argparse
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import transformers and PEFT
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

# Import bitsandbytes for quantization (optional)
try:
    import bitsandbytes as bnb
    has_bnb = True
except ImportError:
    has_bnb = False

# Import dataset utilities
from torch.utils.data import Dataset, DataLoader

# Import MetaMorph Phase 2 components
from metamorph_phase2_trainer import (
    MetaMorphPhase2Trainer, 
    MetaMorphPhase2Config,
    create_default_metamorph_phase2_config
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"founder_vc_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FounderVC")


class FounderVCDataset(Dataset):
    """
    Dataset for founder-VC matching task.
    
    This dataset loads and preprocesses founder profiles and matching VCs.
    """
    
    def __init__(
        self, 
        data_path: str, 
        tokenizer, 
        max_length: int = 1024, 
        is_train: bool = True,
        add_eos_token: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the dataset file/directory
            tokenizer: Tokenizer for encoding texts
            max_length: Maximum sequence length
            is_train: Whether this is a training dataset
            add_eos_token: Whether to add EOS token to each example
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.add_eos_token = add_eos_token
        
        # Load and preprocess data
        self.examples = self._load_data()
        
        logger.info(f"Loaded {len(self.examples)} examples from {data_path}")
        logger.info(f"Sample example: {self.examples[0] if self.examples else 'No examples'}")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """
        Load founder-VC matching data from file.
        
        This is a placeholder - implement based on your actual data format.
        
        Returns:
            List of examples, each containing founder info and matching VCs
        """
        examples = []
        
        try:
            # Check if path is a file or directory
            data_path = Path(self.data_path)
            
            if data_path.is_file():
                # Single file - load based on extension
                if data_path.suffix == '.json':
                    import json
                    with open(data_path, 'r') as f:
                        data = json.load(f)
                        
                        # Process based on expected format
                        # This assumes a list of dictionaries with founder and VC info
                        for item in data:
                            examples.append(self._process_example(item))
                
                elif data_path.is_dir():
                # Directory - process all JSON and CSV files
                    for file_path in data_path.glob("*.json"):
                        import json
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            for item in data:
                                examples.append(self._process_example(item))
                
                    for file_path in data_path.glob("*.csv"):
                        import pandas as pd
                        df = pd.read_csv(file_path)
                        for _, row in df.iterrows():
                            examples.append(self._process_example(row.to_dict()))
            
            else:
                logger.warning(f"Data path does not exist: {data_path}")
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
        
        return examples
    
    def _process_example(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single example from the raw data.
        
        Args:
            data: Raw data for a single example
            
        Returns:
            Processed example
        """
        # Extract founder information
        founder_info = {
            'name': data.get('founder_name', ''),
            'company': data.get('company_name', ''),
            'sector': data.get('sector', ''),
            'stage': data.get('funding_stage', ''),
            'location': data.get('location', ''),
            'description': data.get('description', '')
        }
        
        # Extract VC information
        vc_info = {
            'name': data.get('vc_name', ''),
            'focus': data.get('vc_focus', ''),
            'stage_preference': data.get('vc_stage_preference', ''),
            'geo_preference': data.get('vc_geo_preference', '')
        }
        
        # Create formatted input and output texts
        input_text = self._format_input(founder_info)
        output_text = self._format_output(vc_info)
        
        # Tokenize
        tokenized_input = self.tokenize(input_text)
        tokenized_output = self.tokenize(output_text)
        
        # Combine for training (input + output) or keep separate for inference
        if self.is_train:
            # For training, we want input_ids to contain both input and output
            # with labels being -100 (ignored) for input tokens and actual output token ids for output tokens
            combined = self._combine_input_output(tokenized_input, tokenized_output)
            return combined
        else:
            # For inference, keep input and expected output separate
            return {
                'input_ids': tokenized_input['input_ids'],
                'attention_mask': tokenized_input['attention_mask'],
                'labels': tokenized_output['input_ids'],
                'founder_info': founder_info,
                'vc_info': vc_info
            }
    
    def _format_input(self, founder_info: Dict[str, Any]) -> str:
        """
        Format founder information into input text.
        
        Args:
            founder_info: Founder information
            
        Returns:
            Formatted input text
        """
        return f"""Find investors for this founder:
Name: {founder_info['name']}
Company: {founder_info['company']}
Sector: {founder_info['sector']}
Stage: {founder_info['stage']}
Location: {founder_info['location']}
Description: {founder_info['description']}

Matching investors:"""
    
    def _format_output(self, vc_info: Dict[str, Any]) -> str:
        """
        Format VC information into output text.
        
        Args:
            vc_info: VC information
            
        Returns:
            Formatted output text
        """
        return f"""{vc_info['name']}

Reasoning: This investor is a good match because they focus on {vc_info['focus']}, typically invest in {vc_info['stage_preference']} stage companies, and prefer investments in {vc_info['geo_preference']}."""
    
    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize text.
        
        Args:
            text: Text to tokenize
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Add EOS token if specified
        if self.add_eos_token and not text.endswith(self.tokenizer.eos_token):
            text = text + self.tokenizer.eos_token
        
        # Tokenize
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Remove batch dimension (added by return_tensors='pt')
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0)
        }
    
    def _combine_input_output(
        self, 
        tokenized_input: Dict[str, torch.Tensor], 
        tokenized_output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Combine tokenized input and output for training.
        
        For training, we need:
        - input_ids: concatenation of input and output token ids
        - attention_mask: 1s for all tokens
        - labels: -100 for input tokens (so loss is not computed) and output token ids for output tokens
        
        Args:
            tokenized_input: Tokenized input
            tokenized_output: Tokenized output
            
        Returns:
            Combined example for training
        """
        # Get input length (excluding padding)
        input_length = tokenized_input['attention_mask'].sum().item()
        
        # Get output length (excluding padding)
        output_length = tokenized_output['attention_mask'].sum().item()
        
        # Calculate total length (ensure it doesn't exceed max_length)
        total_length = min(input_length + output_length, self.max_length)
        
        # Initialize tensors
        combined_input_ids = torch.zeros(self.max_length, dtype=torch.long)
        combined_attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        labels = torch.full((self.max_length,), -100, dtype=torch.long)  # Initialize all as -100 (ignored in loss)
        
        # Copy input tokens
        combined_input_ids[:input_length] = tokenized_input['input_ids'][:input_length]
        combined_attention_mask[:input_length] = 1
        
        # Copy output tokens
        remaining_length = total_length - input_length
        if remaining_length > 0:
            combined_input_ids[input_length:total_length] = tokenized_output['input_ids'][:remaining_length]
            combined_attention_mask[input_length:total_length] = 1
            labels[input_length:total_length] = tokenized_output['input_ids'][:remaining_length]
        
        return {
            'input_ids': combined_input_ids,
            'attention_mask': combined_attention_mask,
            'labels': labels
        }
    
    def __len__(self):
        """Return the number of examples."""
        return len(self.examples)
    
    def __getitem__(self, idx):
        """Get an example by index."""
        return self.examples[idx]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a founder-VC matching model with MetaMorph Phase 2")
    
    # Data arguments
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="gemma-3b-it", help="Base model name or path")
    parser.add_argument("--quantize", action="store_true", help="Use 4-bit quantization (requires bitsandbytes)")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./founder_vc_model", help="Output directory")
    parser.add_argument("--training_steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=500, help="Checkpoint saving frequency")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # MetaMorph Phase 2 arguments
    parser.add_argument("--disable_sttt", action="store_true", help="Disable STTT Cycle")
    parser.add_argument("--disable_curriculum", action="store_true", help="Disable Dynamic Curriculum")
    parser.add_argument("--disable_hard_examples", action="store_true", help="Disable Hard Example Amplification")
    parser.add_argument("--enable_latent_reg", action="store_true", help="Enable Latent Space Regularization")
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Initialize tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.eos_token = "</s>"
    
    # Initialize model
    logger.info(f"Loading model: {args.model_name}")
    
    if args.quantize and has_bnb:
        logger.info("Using 4-bit quantization")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            load_in_4bit=True,
            quantization_config=bnb.nn.modules.LinearFP4_Dynamic,
            device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="auto"
        )
    
    # Apply LoRA
    logger.info(f"Applying LoRA with rank={args.lora_r}, alpha={args.lora_alpha}")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)
    
    # Load datasets
    logger.info(f"Loading training data from {args.train_data}")
    train_dataset = FounderVCDataset(
        data_path=args.train_data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        is_train=True
    )
    
    logger.info(f"Loading validation data from {args.val_data}")
    val_dataset = FounderVCDataset(
        data_path=args.val_data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        is_train=True  # Use same format as training for validation
    )
    
    # Create MetaMorph Phase 2 config
    config = create_default_metamorph_phase2_config()
    
    # Update config with CLI arguments
    config.model_name = args.model_name
    config.peft_method = "lora"
    config.output_dir = args.output_dir
    config.training_steps = args.training_steps
    config.eval_steps = args.eval_steps
    config.save_steps = args.save_steps
    config.learning_rate = args.learning_rate
    config.seed = args.seed
    
    # Enable/disable components based on arguments
    config.enable_sttt = not args.disable_sttt
    config.enable_curriculum = not args.disable_curriculum
    config.enable_hard_examples = not args.disable_hard_examples
    config.enable_latent_reg = args.enable_latent_reg
    
    # Initialize trainer
    logger.info("Initializing MetaMorph Phase 2 Trainer")
    trainer = MetaMorphPhase2Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config
    )
    
    # Train model
    logger.info(f"Starting training for {args.training_steps} steps")
    results = trainer.train()
    
    # Log results
    logger.info(f"Training completed in {results.get('training_time', 0) / 3600:.2f} hours")
    logger.info(f"Best validation loss: {results.get('best_eval_loss', float('inf')):.4f}")
    logger.info(f"Final step: {results.get('global_step', 0)}")
    logger.info(f"Model saved to {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main()) 
    