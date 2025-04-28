#!/usr/bin/env python3
"""
Unified Training System for ConvergentMetaMorph

This script provides a single entry point for the entire training pipeline,
integrating both Phase 1 (vision-language pre-training) and Phase 2 
(specialized founder-VC matching) components.

Usage:
  python train.py --config configs/default.json --mode phase1
  python train.py --config configs/founder_vc.json --mode phase2
  python train.py --config configs/combined.json --mode full
"""

import os
import torch
import argparse
import logging
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

# Import core configuration systems
from config import (
    BaseConfig, 
    ConvergentMetaMorphConfig, 
    MetaMorphPhase2Config,
    STTTConfig,
    DynamicCurriculumConfig,
    LatentSpaceRegConfig,
    load_config_from_file,
    create_default_config
)

# Import Phase 1 components
from main import ConvergentMetaMorphTrainer

# Import Phase 2 components
from metamorph_phase2_trainer import MetaMorphPhase2Trainer
from train_founder_vc_matching import FounderVCDataset
from sttt_cycle import STTTCycle
from dynamic_curriculum import DynamicCurriculumConstructor
from hard_example_amplification import HardExampleAmplifier
from latent_space_regularization import LatentSpaceRegularizer

# Import model loaders
from model_factory import load_model, initialize_tokenizer, initialize_processor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"unified_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("UnifiedTrainer")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Unified ConvergentMetaMorph Training System")
    
    # Core arguments
    parser.add_argument("--config", type=str, default=None, help="Path to configuration file")
    parser.add_argument("--mode", type=str, default="full", choices=["phase1", "phase2", "full"],
                       help="Training mode: phase1 (vision-language), phase2 (founder-VC), or full (both)")
    
    # Output directory
    parser.add_argument("--output_dir", type=str, default=None, 
                       help="Output directory for models and logs")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to training data (CSV for phase1, JSON for phase2)")
    parser.add_argument("--image_dir", type=str, default=None,
                       help="Path to images directory (for phase1)")
    parser.add_argument("--founder_profiles", type=str, default=None,
                       help="Path to founder profiles JSON (for phase2)")
    parser.add_argument("--founder_vc_matches", type=str, default=None,
                       help="Path to founder-VC matches (for phase2)")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default=None,
                       help="HuggingFace model name/path")
    parser.add_argument("--load_checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Training batch size")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Learning rate")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda, cpu)")
    
    # Component-specific flags
    parser.add_argument("--enable_sttt", action="store_true",
                       help="Enable STTT cycle (for phase2)")
    parser.add_argument("--enable_curriculum", action="store_true",
                       help="Enable dynamic curriculum (for phase2)")
    parser.add_argument("--enable_hard_examples", action="store_true",
                       help="Enable hard example amplification (for phase2)")
    parser.add_argument("--enable_latent_reg", action="store_true",
                       help="Enable latent space regularization")
    
    # Debug flags
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    return parser.parse_args()

def create_config(args) -> Union[BaseConfig, Dict[str, BaseConfig]]:
    """
    Create configuration based on command-line arguments and config file.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Configuration object or dictionary of configurations
    """
    # Start with default config based on mode
    if args.mode == "phase1":
        config = create_default_config("phase1")
    elif args.mode == "phase2":
        config = create_default_config("phase2")
    else:  # full
        config = {
            "phase1": create_default_config("phase1"),
            "phase2": create_default_config("phase2")
        }
    
    # Load from file if provided
    if args.config:
        try:
            file_config = load_config_from_file(args.config)
            
            # Merge file config with default
            if args.mode == "full":
                if isinstance(file_config, dict) and "phase1" in file_config and "phase2" in file_config:
                    config["phase1"].update(file_config["phase1"])
                    config["phase2"].update(file_config["phase2"])
                else:
                    logger.warning("Config file doesn't have phase1/phase2 sections. Using same config for both phases.")
                    config["phase1"].update(file_config)
                    config["phase2"].update(file_config)
            else:
                if isinstance(config, dict):
                    config.update(file_config)
                else:
                    for k, v in file_config.items():
                        setattr(config, k, v)
                        
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            if args.debug:
                raise
    
    # Override with command-line arguments
    override_config_from_args(config, args)
    
    return config

def override_config_from_args(config, args):
    """Override configuration with command-line arguments."""
    # Helper function to set attribute in nested config
    def set_attr(cfg, name, value):
        if value is None:
            return
            
        if isinstance(cfg, dict):
            for k in cfg.keys():
                if hasattr(cfg[k], name):
                    setattr(cfg[k], name, value)
        else:
            if hasattr(cfg, name):
                setattr(cfg, name, value)
    
    # Core settings
    if args.output_dir:
        set_attr(config, "output_dir", args.output_dir)
    
    # Data settings
    if args.data_path:
        set_attr(config, "data_path", args.data_path)
    if args.image_dir:
        set_attr(config, "image_dir", args.image_dir)
    if args.founder_profiles:
        set_attr(config, "founder_profiles", args.founder_profiles)
    if args.founder_vc_matches:
        set_attr(config, "founder_vc_matches", args.founder_vc_matches)
    
    # Model settings
    if args.model_name:
        set_attr(config, "model_name", args.model_name)
    
    # Training settings
    if args.batch_size:
        set_attr(config, "micro_batch_size", args.batch_size)
        set_attr(config, "batch_size", args.batch_size)
    if args.epochs:
        set_attr(config, "epochs", args.epochs)
    if args.learning_rate:
        set_attr(config, "learning_rate", args.learning_rate)
    if args.seed:
        set_attr(config, "seed", args.seed)
    if args.device:
        set_attr(config, "device", args.device)
    
    # Component flags
    if args.enable_sttt:
        if isinstance(config, dict) and "phase2" in config:
            config["phase2"].enable_sttt = True
        elif hasattr(config, "enable_sttt"):
            config.enable_sttt = True
    
    if args.enable_curriculum:
        if isinstance(config, dict) and "phase2" in config:
            config["phase2"].enable_curriculum = True
        elif hasattr(config, "enable_curriculum"):
            config.enable_curriculum = True
            
    if args.enable_hard_examples:
        if isinstance(config, dict) and "phase2" in config:
            config["phase2"].enable_hard_examples = True
        elif hasattr(config, "enable_hard_examples"):
            config.enable_hard_examples = True
            
    if args.enable_latent_reg:
        if isinstance(config, dict):
            if "phase1" in config:
                if hasattr(config["phase1"], "use_latent_regularization"):
                    config["phase1"].use_latent_regularization = True
            if "phase2" in config:
                if hasattr(config["phase2"], "enable_latent_reg"):
                    config["phase2"].enable_latent_reg = True
        else:
            if hasattr(config, "use_latent_regularization"):
                config.use_latent_regularization = True
            if hasattr(config, "enable_latent_reg"):
                config.enable_latent_reg = True

def prepare_phase1_training(config):
    """
    Prepare Phase 1 training (vision-language pre-training).
    
    Args:
        config: Phase 1 configuration
        
    Returns:
        Tuple of (model, tokenizer, processor, train_dataloader, val_dataloader)
    """
    logger.info("Preparing Phase 1 training (vision-language pre-training)")
    
    # Initialize model, tokenizer, and processor
    model = load_model(config)
    tokenizer = initialize_tokenizer(config)
    processor = initialize_processor(config)
    
    # Initialize dataset
    from main import CompanyDataset, custom_collate_fn
    
    train_dataset = CompanyDataset(
        csv_path=config.data_path,
        target_column='Detailed Description',
        exclude_columns=['Contact Detail', 'Email ID'],
        image_column='Images',
        image_dir=config.image_dir,
        processor=processor
    )
    
    # Create DataLoader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.micro_batch_size,
        shuffle=True,
        collate_fn=lambda batch: custom_collate_fn(batch, tokenizer, processor, config.max_seq_length),
        num_workers=4,
        pin_memory=True
    )
    
    # Create a small validation set
    val_size = min(int(len(train_dataset) * 0.1), 100)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=config.micro_batch_size,
        shuffle=False,
        collate_fn=lambda batch: custom_collate_fn(batch, tokenizer, processor, config.max_seq_length),
        num_workers=2,
        pin_memory=True
    )
    
    return model, tokenizer, processor, train_dataloader, val_dataloader

def prepare_phase2_training(config):
    """
    Prepare Phase 2 training (founder-VC matching).
    
    Args:
        config: Phase 2 configuration
        
    Returns:
        Tuple of (model, tokenizer, train_dataset, val_dataset)
    """
    logger.info("Preparing Phase 2 training (founder-VC matching)")
    
    # Initialize model and tokenizer
    model = load_model(config)
    tokenizer = initialize_tokenizer(config)
    
    # Load founder and VC data
    train_dataset = FounderVCDataset(
        data_path=config.founder_profiles,
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        is_train=True
    )
    
    # Create validation set
    val_size = min(int(len(train_dataset) * 0.1), 100)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    return model, tokenizer, train_subset, val_subset

def run_phase1_training(config, model, tokenizer, processor, train_dataloader, val_dataloader=None):
    """
    Run Phase 1 training (vision-language pre-training).
    
    Args:
        config: Phase 1 configuration
        model: The model to train
        tokenizer: Tokenizer for the model
        processor: Processor for vision inputs
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        
    Returns:
        Trained model and training results
    """
    logger.info("Starting Phase 1 training")
    
    # Initialize trainer
    trainer = ConvergentMetaMorphTrainer(model, tokenizer, config)
    
    # Setup processor
    trainer.processor = processor
    
    # Train the model
    training_results = trainer.train(train_dataloader, val_dataloader)
    
    # Save final model
    final_output_dir = os.path.join(config.output_dir, "phase1_final")
    trainer.save_checkpoint(final_output_dir)
    
    return model, training_results

def run_phase2_training(config, model, tokenizer, train_dataset, val_dataset, phase1_results=None):
    """
    Run Phase 2 training (founder-VC matching).
    
    Args:
        config: Phase 2 configuration
        model: The model to train
        tokenizer: Tokenizer for the model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        phase1_results: Optional results from Phase 1 training
        
    Returns:
        Trained model and training results
    """
    logger.info("Starting Phase 2 training")
    
    # Initialize trainer
    trainer = MetaMorphPhase2Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config
    )
    
    # Initialize training
    training_results = trainer.train()
    
    # Save final model
    final_output_dir = os.path.join(config.output_dir, "phase2_final")
    trainer.save_checkpoint(final_output_dir)
    
    return model, training_results

def main():
    """Main entry point."""
    # Parse command-line arguments
    args = parse_args()
    
    # Set debug logging level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create configuration
    config = create_config(args)
    
    # Run in selected mode
    if args.mode == "phase1":
        # Prepare and run Phase 1 training
        model, tokenizer, processor, train_dataloader, val_dataloader = prepare_phase1_training(config)
        model, results = run_phase1_training(config, model, tokenizer, processor, train_dataloader, val_dataloader)
        
    elif args.mode == "phase2":
        # Prepare and run Phase 2 training
        model, tokenizer, train_dataset, val_dataset = prepare_phase2_training(config)
        model, results = run_phase2_training(config, model, tokenizer, train_dataset, val_dataset)
        
    else:  # full mode - run both phases
        # Phase 1
        model, tokenizer, processor, train_dataloader, val_dataloader = prepare_phase1_training(config["phase1"])
        model, phase1_results = run_phase1_training(config["phase1"], model, tokenizer, processor, train_dataloader, val_dataloader)
        
        # Phase 2
        if config["phase2"].model_name is None:
            # Use Phase 1 output as Phase 2 input
            config["phase2"].model_name = os.path.join(config["phase1"].output_dir, "phase1_final")
            
        model, tokenizer, train_dataset, val_dataset = prepare_phase2_training(config["phase2"])
        model, phase2_results = run_phase2_training(config["phase2"], model, tokenizer, train_dataset, val_dataset, phase1_results)
        
        # Combine results
        results = {
            "phase1": phase1_results,
            "phase2": phase2_results
        }
    
    # Save final results
    with open(os.path.join(config.output_dir if not isinstance(config, dict) else config["phase2"].output_dir, 
                           "training_results.json"), "w") as f:
        # Convert any non-serializable objects to strings
        serializable_results = {}
        for k, v in (results.items() if isinstance(results, dict) else {"results": results}.items()):
            if isinstance(v, dict):
                serializable_results[k] = {sk: str(sv) if not isinstance(sv, (int, float, str, bool, list, dict)) else sv
                                         for sk, sv in v.items()}
            else:
                serializable_results[k] = str(v)
        
        json.dump(serializable_results, f, indent=2)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main() 