import os
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup
from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Optional, Union, Any

# Import our enhanced components
# You would need to replace these imports with the actual module paths
from enhanced_metaplasticity_optimizer import EnhancedMetaplasticityOptimizer, EnhancedAdaFactorWithMetaplasticity
from enhanced_architecture_controller import EnhancedConvergentArchitectureController
from enhanced_activation_engineering import EnhancedActivationEngineering

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("convergent_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ConvergentMetaMorph")

@dataclass
class ConvergentMetaMorphConfig:
    """Configuration for ConvergentMetaMorph with convergence guarantees."""
    # Model configuration
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir: str = "./qwen_finetune_output_memory_optimized"
    
    # Quantization settings
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True  # Double quantization for 4-bit
    
    # PEFT settings
    use_peft: bool = True
    peft_method: str = "lora"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Convergent Architecture settings
    use_dynamic_architecture: bool = True
    min_lora_rank: int = 4
    max_lora_rank: int = 64
    rank_step: int = 4
    architecture_scan_freq: int = 100
    morph_threshold: float = 0.15
    error_budget: float = 1.0  # Total error budget for convergence guarantee
    schedule_alpha: float = 0.7  # Sublinear schedule factor for architecture morphing
    
    # Metaplasticity settings
    use_metaplasticity: bool = True
    use_adafactor: bool = False  # Use AdaFactor with metaplasticity
    plasticity_eta: float = 0.01  # Learning rate for plasticity
    plasticity_decay: float = 0.9999  # Decay factor for plasticity
    plasticity_growth: float = 1.001  # Growth factor for plasticity
    min_plasticity: float = 0.2  # Minimum plasticity value
    max_plasticity: float = 5.0  # Maximum plasticity value
    
    # Activation Engineering settings
    use_activation_engineering: bool = True
    activation_update_freq: int = 100  # Steps between MI calculations
    
    # Training settings
    epochs: int = 3
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    
    # Mixed precision
    use_mixed_precision: bool = True
    
    # Training logistics
    seed: int = 42
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200
    
    def __post_init__(self):
        """Ensure directories exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)


class EnhancedConvergentMetaMorph:
    """
    Enhanced MetaMorph framework with stronger convergence guarantees 
    based on mathematical analysis.
    
    This implementation integrates:
    1. Enhanced Convergent Architecture Controller (non-expansive operators with bounded error)
    2. Enhanced Two-Timescale Metaplasticity Optimizer (following the convergence proofs)
    3. Information-Theoretic Activation Engineering (with mutual information maximization)
    4. Rigorous error tracking and bounds to guarantee overall convergence
    """
    def __init__(self, model, tokenizer, config, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize component registry
        self.components = {}
        
        # Initialize convergence tracking
        self.cumulative_architecture_error = 0.0
        self.error_budget = config.error_budget
        
        # Initialize training tracking
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Set up components
        self._initialize_components()
        
        logger.info(f"Initialized EnhancedConvergentMetaMorph with device: {self.device}")
    
    def _initialize_components(self):
        """Initialize the convergent MetaMorph components."""
        # 1. Initialize Enhanced Metaplasticity Optimizer
        if self.config.use_metaplasticity:
            self._initialize_optimizer()
        
        # 2. Initialize Enhanced Convergent Architecture Controller (after optimizer)
        if self.config.use_dynamic_architecture:
            self._initialize_architecture_controller()
        
        # 3. Initialize Information-Theoretic Activation Engineering
        if self.config.use_activation_engineering:
            self.components['activation_engineer'] = EnhancedActivationEngineering(
                self.model, 
                self.device,
                update_freq=self.config.activation_update_freq
            )
            logger.info("Initialized Enhanced Information-Theoretic Activation Engineering")
        
        # 4. Initialize mixed precision training
        if self.config.use_mixed_precision and torch.cuda.is_available():
            self.components['scaler'] = torch.amp.GradScaler()
            logger.info("Initialized Mixed Precision Training")
    
    def _initialize_optimizer(self):
        """Initialize enhanced metaplasticity optimizer with convergence guarantees."""
        # Filter trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Choose base optimizer type
        if self.config.use_adafactor:
            # Use Enhanced AdaFactor with Metaplasticity
            self.components['optimizer'] = EnhancedAdaFactorWithMetaplasticity(
                trainable_params,
                lr=self.config.learning_rate,
                plasticity_eta=self.config.plasticity_eta,
                min_plasticity=self.config.min_plasticity,
                max_plasticity=self.config.max_plasticity
            )
            logger.info("Initialized Enhanced AdaFactor with Metaplasticity")
        else:
            # Use standard optimizer with Enhanced Metaplasticity wrapper
            from torch.optim import AdamW
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
    
    def _initialize_architecture_controller(self):
        """Initialize enhanced convergent architecture controller."""
        self.components['architecture_controller'] = EnhancedConvergentArchitectureController(
            model=self.model,
            min_rank=self.config.min_lora_rank,
            max_rank=self.config.max_lora_rank,
            rank_step=self.config.rank_step,
            error_budget=self.config.error_budget,
            scan_freq=self.config.architecture_scan_freq,
            morph_threshold=self.config.morph_threshold
        )
        self.components['architecture_controller'].schedule_alpha = self.config.schedule_alpha
        logger.info("Initialized Enhanced Convergent Architecture Controller")
    
    def setup_training(self, train_dataset, val_dataset=None):
        """Set up training components and calculate training steps."""
        config = self.config
        
        # Calculate total steps
        if train_dataset is not None:
            num_examples = len(train_dataset)
            effective_batch_size = config.micro_batch_size * config.gradient_accumulation_steps
            total_steps = (num_examples // effective_batch_size) * config.epochs
        else:
            # Fallback for custom datasets without __len__
            total_steps = 10000 * config.epochs
            
        self.total_steps = total_steps
        
        # Initialize learning rate scheduler
        if 'optimizer' in self.components:
            warmup_steps = int(config.warmup_ratio * total_steps)
            
            scheduler = get_cosine_schedule_with_warmup(
                self.components['optimizer'],
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
            self.components['scheduler'] = scheduler
            
            logger.info(f"Initialized scheduler with {warmup_steps} warmup steps and {total_steps} total steps")
        
        # Scale error budget based on total steps to ensure convergence
        if 'architecture_controller' in self.components:
            # Configure error budget based on total steps
            scaled_error_budget = min(config.error_budget, math.sqrt(total_steps) / 10)
            self.components['architecture_controller'].error_budget = scaled_error_budget
            logger.info(f"Set architecture error budget to {scaled_error_budget:.4f}")
        
        return total_steps
    
    def train(self, train_dataloader, val_dataloader=None, callbacks=None):
        """
        Main training loop with convergence guarantees.
        
        Args:
            train_dataloader: DataLoader for training
            val_dataloader: Optional DataLoader for validation
            callbacks: Optional dict of callback functions
            
        Returns:
            Dict of training stats
        """
        config = self.config
        
        # Training tracking
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        training_stats = []
        
        # Calculate total steps
        if hasattr(train_dataloader, 'dataset') and hasattr(train_dataloader.dataset, '__len__'):
            num_examples = len(train_dataloader.dataset)
            steps_per_epoch = len(train_dataloader)
            total_steps = steps_per_epoch * config.epochs
        else:
            # Fallback for unknown length
            steps_per_epoch = 1000
            total_steps = steps_per_epoch * config.epochs
            num_examples = steps_per_epoch * train_dataloader.batch_size
            
        self.total_steps = total_steps
        
        # Setup scheduler if not already initialized
        if 'optimizer' in self.components and 'scheduler' not in self.components:
            warmup_steps = int(config.warmup_ratio * total_steps)
            
            scheduler = get_cosine_schedule_with_warmup(
                self.components['optimizer'],
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
            self.components['scheduler'] = scheduler
        
        logger.info(f"Starting training for {config.epochs} epochs, {total_steps} total steps")
        logger.info(f"Training on {num_examples} examples with batch size {train_dataloader.batch_size}")
        
        # Training loop
        for epoch in range(config.epochs):
            self.epoch = epoch
            self.model.train()
            epoch_loss = 0
            epoch_start_time = torch.cuda.Event(enable_timing=True)
            epoch_end_time = torch.cuda.Event(enable_timing=True)
            
            epoch_start_time.record()
            logger.info(f"Starting epoch {epoch+1}/{config.epochs}")
            
            # Batch loop
            for step, batch in enumerate(train_dataloader):
                # Process batch
                batch_inputs = self._prepare_batch(batch)
                
                # Forward and backward passes
                loss = self._training_step(batch_inputs)
                epoch_loss += loss
                
                # Update global step if accumulation is complete
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % config.logging_steps == 0:
                        # Get current LR
                        if 'optimizer' in self.components:
                            current_lr = self.components['optimizer'].param_groups[0]['lr']
                            
                            # Get plasticity stats if available
                            if hasattr(self.components['optimizer'], 'get_plasticity_stats'):
                                plasticity_stats = self.components['optimizer'].get_plasticity_stats()
                                plasticity_info = f", Mean plasticity: {plasticity_stats['mean']:.4f}"
                            else:
                                plasticity_info = ""
                        else:
                            current_lr = 0
                            plasticity_info = ""
                        
                        # Get architecture stats if available
                        if 'architecture_controller' in self.components:
                            arch_ctrl = self.components['architecture_controller']
                            cumulative_error = arch_ctrl.cumulative_error
                            error_budget = arch_ctrl.error_budget
                            arch_info = f", Arch error: {cumulative_error:.4e}/{error_budget:.4e}"
                        else:
                            arch_info = ""
                        
                        logger.info(f"Step {self.global_step}/{total_steps} - "
                                   f"Loss: {loss:.4f}, LR: {current_lr:.8f}"
                                   f"{plasticity_info}{arch_info}")
                        
                        # Record stats
                        training_stats.append({
                            'step': self.global_step,
                            'epoch': epoch + 1,
                            'loss': loss,
                            'learning_rate': current_lr,
                            'timestamp': torch.cuda.Event.elapsed_time(epoch_start_time, epoch_end_time) if torch.cuda.is_available() else 0
                        })
                    
                    # Architecture scanning
                    if 'architecture_controller' in self.components:
                        arch_ctrl = self.components['architecture_controller']
                        scan_results = arch_ctrl.scan_architecture(self.global_step)
                        
                        if scan_results and scan_results.get('morph_needed', False):
                            # Apply architecture changes with error tracking
                            morph_plan = arch_ctrl.morph_architecture(scan_results)
                            
                            if morph_plan:
                                logger.info(f"Architecture morphing applied at step {self.global_step}")
                                
                                # Check convergence guarantee
                                if arch_ctrl.cumulative_error > arch_ctrl.error_budget:
                                    logger.warning(f"Warning: Architecture error budget exceeded: "
                                                 f"{arch_ctrl.cumulative_error:.4e} > {arch_ctrl.error_budget:.4e}")
                    
                    # Validation
                    if val_dataloader is not None and self.global_step % config.eval_steps == 0:
                        val_loss = self.evaluate(val_dataloader)
                        
                        logger.info(f"Validation loss: {val_loss:.4f}")
                        
                        # Save best model
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            
                            # Save checkpoint using more informative name
                            checkpoint_path = os.path.join(config.output_dir, "checkpoints", "best_model")
                            os.makedirs(checkpoint_path, exist_ok=True)
                            self.save_checkpoint(
                                checkpoint_path,
                                {'val_loss': val_loss, 'step': self.global_step}
                            )
                    
                    # Save periodic checkpoint
                    if self.global_step % config.save_steps == 0:
                        checkpoint_path = os.path.join(config.output_dir, "checkpoints", f"step_{self.global_step}")
                        os.makedirs(checkpoint_path, exist_ok=True)
                        self.save_checkpoint(
                            checkpoint_path,
                            {'step': self.global_step}
                        )
                        
                    # Callbacks
                    if callbacks and 'on_step_end' in callbacks:
                        callbacks['on_step_end'](self, step=self.global_step, loss=loss)
            
            # End of epoch
            epoch_end_time.record()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                epoch_time_ms = torch.cuda.Event.elapsed_time(epoch_start_time, epoch_end_time)
                epoch_time = epoch_time_ms / 1000.0  # Convert to seconds
            else:
                epoch_time = 0
                
            avg_loss = epoch_loss / len(train_dataloader)
            
            # Log epoch stats
            logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s - "
                       f"Average loss: {avg_loss:.4f}")
            
            # Run validation at the end of each epoch
            if val_dataloader is not None:
                val_loss = self.evaluate(val_dataloader)
                logger.info(f"End of epoch validation - Loss: {val_loss:.4f}")
                
                # Save if best
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    
                    # Save checkpoint
                    checkpoint_path = os.path.join(config.output_dir, "checkpoints", "best_model")
                    os.makedirs(checkpoint_path, exist_ok=True)
                    self.save_checkpoint(
                        checkpoint_path,
                        {'val_loss': val_loss, 'step': self.global_step, 'epoch': epoch+1}
                    )
            
            # Save epoch checkpoint
            checkpoint_path = os.path.join(config.output_dir, "checkpoints", f"epoch_{epoch+1}")
            os.makedirs(checkpoint_path, exist_ok=True)
            self.save_checkpoint(
                checkpoint_path,
                {'step': self.global_step, 'epoch': epoch+1}
            )
            
            # Callbacks
            if callbacks and 'on_epoch_end' in callbacks:
                callbacks['on_epoch_end'](self, epoch=epoch, loss=avg_loss)
        
        # Training complete
        logger.info(f"Training complete - {self.global_step} steps across {config.epochs} epochs")
        
        # Save final model
        final_path = os.path.join(config.output_dir, "final_model")
        os.makedirs(final_path, exist_ok=True)
        self.save_checkpoint(
            final_path,
            {'step': self.global_step, 'epochs': config.epochs}
        )
        
        # Generate final reports
        if 'architecture_controller' in self.components:
            arch_report = self.components['architecture_controller'].get_architecture_report()
            logger.info(f"Architecture report: {arch_report}")
            
            # Verify convergence guarantee
            arch_ctrl = self.components['architecture_controller']
            if arch_ctrl.cumulative_error <= arch_ctrl.error_budget:
                logger.info(f"✓ Convergence guarantee satisfied: "
                          f"Total error {arch_ctrl.cumulative_error:.4e} <= "
                          f"Budget {arch_ctrl.error_budget:.4e}")
            else:
                logger.warning(f"⚠ Convergence guarantee not satisfied: "
                             f"Total error {arch_ctrl.cumulative_error:.4e} > "
                             f"Budget {arch_ctrl.error_budget:.4e}")
        
        if 'optimizer' in self.components and hasattr(self.components['optimizer'], 'get_stability_report'):
            stability_report = self.components['optimizer'].get_stability_report()
            logger.info(f"Optimizer stability report: {stability_report['convergence_status']}")
        
        if 'activation_engineer' in self.components:
            activation_report = self.components['activation_engineer'].get_activation_selection_report()
            logger.info(f"Activation engineering report: {len(activation_report)} layers optimized")
        
        # Log training summary
        avg_loss = sum(stat['loss'] for stat in training_stats) / len(training_stats) if training_stats else 0
        logger.info(f"Training summary - Avg loss: {avg_loss:.4f}, Best val loss: {self.best_val_loss:.4f}")
        
        return {
            'training_stats': training_stats,
            'best_val_loss': self.best_val_loss,
            'final_step': self.global_step
        }
    
    def _prepare_batch(self, batch):
        """Prepare batch inputs for model."""
        # Format input dict with appropriate tensors
        if isinstance(batch, dict):
            inputs = batch
        else:
            # Convert batch to model inputs
            if hasattr(batch, 'keys'):
                # Dataset returns dict-like object
                text_key = 'text' if 'text' in batch else 'formatted_text'
                texts = batch[text_key] if text_key in batch else batch
            else:
                # Direct list/tensor
                texts = batch
            
            # Tokenize
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            # Add labels for LM loss (copy input_ids)
            inputs['labels'] = inputs['input_ids'].clone()
        
        # Move to device
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        return inputs
        
    def _training_step(self, inputs):
        """Execute a single training step with all convergent components."""
        config = self.config
        
        # Check if we're accumulating gradients
        is_accumulating = (self.global_step + 1) % config.gradient_accumulation_steps != 0
        should_update = not is_accumulating
        
        # Zero gradients at the beginning of accumulation
        if not is_accumulating or self.global_step == 0:
            if 'optimizer' in self.components:
                self.components['optimizer'].zero_grad()
        
        # Mixed precision training
        use_mixed_precision = 'scaler' in self.components
        
        # Forward and backward pass with mixed precision
        if use_mixed_precision:
            with autocast():
                outputs = self.model(**inputs)
                loss = outputs.loss / config.gradient_accumulation_steps
            
            # Scale loss and backward
            self.components['scaler'].scale(loss).backward()
            
            # Update weights if needed
            if should_update:
                # Clip gradients
                if 'optimizer' in self.components:
                    self.components['scaler'].unscale_(self.components['optimizer'])
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        config.max_grad_norm
                    )
                
                    # Update parameters
                    self.components['scaler'].step(self.components['optimizer'])
                    self.components['scaler'].update()
                    
                    # Update learning rate
                    if 'scheduler' in self.components:
                        self.components['scheduler'].step()
        else:
            # Standard training without mixed precision
            outputs = self.model(**inputs)
            loss = outputs.loss / config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights if needed
            if should_update and 'optimizer' in self.components:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    config.max_grad_norm
                )
                
                # Update parameters
                self.components['optimizer'].step()
                