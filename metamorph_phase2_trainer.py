import os
import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
import json
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("metamorph_phase2.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MetaMorphPhase2")

@dataclass
class MetaMorphPhase2Config:
    """Master configuration for MetaMorph Phase 2."""
    # Model configuration
    model_name: str = "gemma-3b-it"  # Base model
    peft_method: str = "lora"       # Parameter-efficient fine-tuning method
    
    # STTT Cycle configuration
    enable_sttt: bool = True          # Enable STTT cycle
    sttt_config: Dict[str, Any] = field(default_factory=dict)
    
    # Dynamic Curriculum configuration
    enable_curriculum: bool = True    # Enable dynamic curriculum
    curriculum_config: Dict[str, Any] = field(default_factory=dict)
    
    # Hard Example Amplification configuration
    enable_hard_examples: bool = True  # Enable hard example amplification
    hard_example_config: Dict[str, Any] = field(default_factory=dict)
    
    # Latent Space Regularization configuration
    enable_latent_reg: bool = True    # Enable latent space regularization
    latent_reg_config: Dict[str, Any] = field(default_factory=dict)
    
    # Internal Consistency Checking configuration
    enable_consistency: bool = False  # Enable internal consistency checking
    consistency_config: Dict[str, Any] = field(default_factory=dict)
    
    # Synthetic Data configuration
    enable_synthetic: bool = False    # Enable synthetic data generation
    synthetic_config: Dict[str, Any] = field(default_factory=dict)
    
    # Training settings
    output_dir: str = "./metamorph_phase2_output"
    training_steps: int = 10000
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 10
    
    # Optimizer settings
    optimizer_type: str = "adamw"     # Optimizer type
    learning_rate: float = 2e-5       # Learning rate
    weight_decay: float = 0.01        # Weight decay
    max_grad_norm: float = 1.0        # Gradient clipping
    
    # Hardware settings
    device: str = "cuda"              # Device to use
    mixed_precision: bool = True      # Use mixed precision training
    
    # Reproducibility
    seed: int = 42                    # Random seed
    
    def __post_init__(self):
        """Ensure output directory exists."""
        os.makedirs(self.output_dir, exist_ok=True)


class MetaMorphPhase2Trainer:
    """
    MetaMorph Phase 2 Trainer - Integrates all Phase 2 components.
    
    This trainer implements:
    1. STTT Cycle (Study-Test-Test-Test Loop)
    2. Dynamic Curriculum Construction
    3. Hard Example Amplification
    4. Latent Space Regularization
    5. Internal Consistency Checking (optional)
    6. Synthetic Founder Creation (optional)
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        val_dataset,
        config: MetaMorphPhase2Config = None,
        custom_modules: Dict[str, Any] = None
    ):
        """
        Initialize the MetaMorph Phase 2 Trainer.
        
        Args:
            model: The model to train
            tokenizer: Tokenizer for the model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Training configuration
            custom_modules: Optional custom module implementations
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or MetaMorphPhase2Config()
        self.custom_modules = custom_modules or {}
        
        # Set device
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Set random seed for reproducibility
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        
        # Initialize optimizer
        self.optimizer = self._initialize_optimizer()
        
        # Initialize components
        logger.info("Initializing MetaMorph Phase 2 components...")
        self.components = {}
        self._initialize_components()
        
        # Training state
        self.global_step = 0
        self.best_eval_loss = float('inf')
        self.training_start_time = None
        
        # Metrics tracking
        self.metrics_history = []
        
        logger.info(f"MetaMorph Phase 2 Trainer initialized on device: {self.device}")
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Components enabled: {self._get_enabled_components()}")
    
    def _initialize_optimizer(self):
        """Initialize the optimizer."""
        # Get trainable parameters
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Create optimizer based on config
        if self.config.optimizer_type.lower() == "adamw":
            return torch.optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "adam":
            return torch.optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            logger.warning(f"Unknown optimizer type: {self.config.optimizer_type}, using AdamW")
            return torch.optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
    
    def _initialize_components(self):
        """Initialize all enabled components."""
        # Import component modules
        from sttt_cycle import STTTConfig, STTTCycle
        from dynamic_curriculum import DynamicCurriculumConfig, DynamicCurriculumConstructor
        from hard_example_amplification import HardExampleAmplificationConfig, HardExampleAmplifier
        from latent_space_regularization import LatentSpaceRegConfig, LatentSpaceRegularizer
        
        # 1. Initialize STTT Cycle
        if self.config.enable_sttt:
            logger.info("Initializing STTT Cycle...")
            
            # Create STTT config
            sttt_config = STTTConfig(**self.config.sttt_config)
            
            # Basic train/val dataloaders for initial setup
            # (these will be replaced once other components are initialized)
            train_dataloader = torch.utils.data.DataLoader(
                dataset=self.train_dataset,
                batch_size=8,
                shuffle=True
            )
            
            val_dataloader = torch.utils.data.DataLoader(
                dataset=self.val_dataset,
                batch_size=8,
                shuffle=False
            )
            
            # Initialize STTT component
            self.components['sttt'] = STTTCycle(
                model=self.model,
                optimizer=self.optimizer,
                study_dataloader=train_dataloader,
                t1_dataloader=val_dataloader,
                config=sttt_config,
                device=self.device
            )
        
        # 2. Initialize Dynamic Curriculum
        if self.config.enable_curriculum:
            logger.info("Initializing Dynamic Curriculum...")
            
            # Create curriculum config
            curriculum_config = DynamicCurriculumConfig(**self.config.curriculum_config)
            
            # Initialize curriculum component
            self.components['curriculum'] = DynamicCurriculumConstructor(
                model=self.model,
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset,
                config=curriculum_config,
                device=self.device
            )
        
        # 3. Initialize Hard Example Amplification
        if self.config.enable_hard_examples:
            logger.info("Initializing Hard Example Amplification...")
            
            # Create hard example config
            hard_example_config = HardExampleAmplificationConfig(**self.config.hard_example_config)
            
            # Initialize hard example component
            self.components['hard_examples'] = HardExampleAmplifier(
                model=self.model,
                train_dataset=self.train_dataset,
                config=hard_example_config,
                device=self.device
            )
        
        # 4. Initialize Latent Space Regularization
        if self.config.enable_latent_reg:
            logger.info("Initializing Latent Space Regularization...")
            
            # Create latent reg config
            latent_reg_config = LatentSpaceRegConfig(**self.config.latent_reg_config)
            
            # Initialize latent space regularization component
            self.components['latent_reg'] = LatentSpaceRegularizer(
                model=self.model,
                config=latent_reg_config,
                device=self.device
            )
        
        # 5. Initialize custom components from custom_modules
        for name, module_class in self.custom_modules.items():
            if name in self.config.__dict__ and self.config.__dict__[f"enable_{name}"]:
                logger.info(f"Initializing custom component: {name}...")
                
                # Get config for this component
                component_config = self.config.__dict__[f"{name}_config"]
                
                # Initialize component
                self.components[name] = module_class(
                    model=self.model,
                    train_dataset=self.train_dataset,
                    val_dataset=self.val_dataset,
                    config=component_config,
                    device=self.device
                )
        
        # 6. Connect components by updating dataloaders
        self._connect_components()
    
    def _connect_components(self):
        """Connect components by updating dataloaders and references."""
        logger.info("Connecting components...")
        
        # Create integrated dataloader
        integrated_dataloader = self._create_integrated_dataloader()
        
        # Update STTT dataloader
        if 'sttt' in self.components:
            self.components['sttt'].study_dataloader = integrated_dataloader
            self.components['sttt'].study_iter = iter(integrated_dataloader)
            
            # Update T2/T3 generators
            self.components['sttt'].t2_generator = self._t2_generator
            self.components['sttt'].t3_generator = self._t3_generator
    
    def _create_integrated_dataloader(self, batch_size: int = 8) -> torch.utils.data.DataLoader:
        """
        Create a dataloader that integrates curriculum and hard example components.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Integrated DataLoader
        """
        # Start with uniform sampling weights
        all_indices = list(range(len(self.train_dataset)))
        weights = np.ones(len(all_indices))
        
        # Apply curriculum weights if enabled
        if 'curriculum' in self.components:
            # Get available indices from curriculum
            available_indices = self.components['curriculum'].curriculum_dataset._get_available_examples()
            
            if available_indices:
                # Update indices list to only include curriculum-available examples
                all_indices = available_indices
                
                # Get curriculum weights
                curriculum_weights = self.components['curriculum'].curriculum_dataset.sampling_weights[available_indices]
                weights = curriculum_weights
        
        # Apply hard example weights if enabled
        if 'hard_examples' in self.components:
            # Get hard example weights
            hard_example_weights = self.components['hard_examples'].tracker.get_sampling_weights(all_indices)
            
            # Multiply weights (AND effect)
            weights = weights * hard_example_weights
        
        # Ensure valid weights
        if np.sum(weights) <= 0:
            weights = np.ones_like(weights)
        
        # Create sampler
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(all_indices),
            replacement=True
        )
        
        # Create dataloader
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
    
    def _t2_generator(self, batch):
        """
        Generator for T2 phase (counterfactual examples).
        
        Args:
            batch: Input batch
            
        Returns:
            Transformed batch
        """
        # Use curriculum's higher difficulty examples if available
        if 'curriculum' in self.components:
            curriculum = self.components['curriculum']
            phase = min(curriculum.curriculum_dataset.current_phase + 1, 
                       len(curriculum.curriculum_dataset.phase_boundaries) - 2)
            
            harder_indices = curriculum.curriculum_dataset.phase_examples.get(phase, [])
            
            if harder_indices:
                # Sample from harder examples
                sampled_indices = np.random.choice(harder_indices, size=len(batch['input_ids']))
                harder_batch = {
                    k: torch.stack([
                        torch.tensor(self.train_dataset[idx][k]) for idx in sampled_indices
                    ]).to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                return harder_batch
        
        # Fallback to simple transformation
        return self._simple_counterfactual_transform(batch)
    
    def _t3_generator(self, batch):
        """
        Generator for T3 phase (adversarial examples).
        
        Args:
            batch: Input batch
            
        Returns:
            Transformed batch
        """
        # Use hard example augmentations if available
        if 'hard_examples' in self.components:
            augmented_batch = self.components['hard_examples'].get_augmented_hard_example_batch(
                batch_size=len(batch['input_ids'])
            )
            
            if augmented_batch:
                return augmented_batch
        
        # Try to use curriculum's hardest examples
        if 'curriculum' in self.components:
            curriculum = self.components['curriculum']
            hardest_phase = len(curriculum.curriculum_dataset.phase_boundaries) - 2
            hardest_indices = curriculum.curriculum_dataset.phase_examples.get(hardest_phase, [])
            
            if hardest_indices:
                # Sample from hardest examples
                sampled_indices = np.random.choice(hardest_indices, size=len(batch['input_ids']))
                hardest_batch = {
                    k: torch.stack([
                        torch.tensor(self.train_dataset[idx][k]) for idx in sampled_indices
                    ]).to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                return hardest_batch
        
        # Fallback to simple transformation
        return self._simple_adversarial_transform(batch)
    
    def _simple_counterfactual_transform(self, batch):
        """
        Simple counterfactual transformation for T2 phase.
        
        Args:
            batch: Input batch
            
        Returns:
            Transformed batch
        """
        # Clone the batch to avoid modifying the original
        counterfactual_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # For text inputs, apply simple transformations
        if 'input_ids' in counterfactual_batch:
            input_ids = counterfactual_batch['input_ids']
            batch_size, seq_len = input_ids.shape
            
            # Simple transformation: swap tokens in 10-20% of positions
            swap_prob = 0.15
            
            for i in range(batch_size):
                # Create mask for tokens to swap
                swap_mask = torch.rand(seq_len, device=input_ids.device) < swap_prob
                swap_mask = swap_mask & (input_ids[i] != 0)  # Don't swap padding
                
                # Get indices to swap
                swap_indices = torch.nonzero(swap_mask).squeeze(-1)
                
                # Shuffle these indices
                if len(swap_indices) > 1:
                    perm = torch.randperm(len(swap_indices))
                    shuffled_indices = swap_indices[perm]
                    
                    # Swap tokens
                    input_ids[i, swap_indices] = input_ids[i, shuffled_indices]
        
        return counterfactual_batch
    
    def _simple_adversarial_transform(self, batch):
        """
        Simple adversarial transformation for T3 phase.
        
        Args:
            batch: Input batch
            
        Returns:
            Transformed batch
        """
        # Clone the batch to avoid modifying the original
        adversarial_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # For text inputs, apply more aggressive transformations
        if 'input_ids' in adversarial_batch:
            input_ids = adversarial_batch['input_ids']
            attention_mask = adversarial_batch.get('attention_mask', None)
            batch_size, seq_len = input_ids.shape
            
            # Adversarial transformation: mask 15-30% of tokens and replace 10-20% with random tokens
            mask_prob = 0.2
            replace_prob = 0.15
            
            for i in range(batch_size):
                # Create masks for tokens to modify
                mask_mask = torch.rand(seq_len, device=input_ids.device) < mask_prob
                replace_mask = torch.rand(seq_len, device=input_ids.device) < replace_prob
                
                # Don't modify padding or special tokens
                mask_mask = mask_mask & (input_ids[i] > 100)  # Assume special tokens are < 100
                replace_mask = replace_mask & (input_ids[i] > 100) & (~mask_mask)  # Don't overlap with mask
                
                # Apply masking (replace with mask token, typically 1 or similar)
                mask_token_id = 1  # Adjust based on tokenizer
                input_ids[i, mask_mask] = mask_token_id
                
                # Apply random replacements
                if replace_mask.sum() > 0:
                    # Generate random token IDs between 1000 and 10000 (common vocab range)
                    random_ids = torch.randint(1000, 10000, (replace_mask.sum(),), device=input_ids.device)
                    input_ids[i, replace_mask] = random_ids
                
                # Optionally modify attention mask to create missing information
                if attention_mask is not None:
                    # Zero out 5-10% of attention positions
                    attn_mask = torch.rand(seq_len, device=attention_mask.device) < 0.075
                    attn_mask = attn_mask & (attention_mask[i] > 0)  # Only modify non-padding
                    attention_mask[i, attn_mask] = 0
        
        return adversarial_batch
    
    def train_step(self) -> Dict[str, Any]:
        """
        Execute a single training step with all components.
        
        Returns:
            Dictionary of metrics for this step
        """
        # Step 1: STTT Cycle step
        sttt_metrics = {}
        if 'sttt' in self.components:
            sttt_metrics = self.components['sttt'].step()
            current_phase = sttt_metrics.get('phase', 'S')
        else:
            # If STTT not enabled, do a basic training step
            batch = next(iter(self._create_integrated_dataloader()))
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            self.model.train()
            self.optimizer.zero_grad()
            
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config.max_grad_norm
            )
            
            self.optimizer.step()
            
            sttt_metrics = {
                'phase': 'S',
                's_loss': loss.item()
            }
            current_phase = 'S'
        
        # Step 2: Apply latent space regularization if enabled and in study phase
        latent_reg_components = {}
        if 'latent_reg' in self.components and current_phase == 'S':
            # Get validation loss delta if available
            val_loss_delta = None
            if 't1_loss' in sttt_metrics and hasattr(self, 'last_val_loss') and self.last_val_loss is not None:
                val_loss_delta = sttt_metrics['t1_loss'] - self.last_val_loss
            
            # Apply latent regularization
            latent_reg_components = self.components['latent_reg'].step(val_loss_delta)
            
            # Store current validation loss
            if 't1_loss' in sttt_metrics:
                self.last_val_loss = sttt_metrics['t1_loss']
        
        # Step 3: Update curriculum if enabled
        if 'curriculum' in self.components:
            self.components['curriculum'].update_curriculum(sttt_metrics)
        
        # Step 4: Update hard example tracking if enabled and in study phase
        if 'hard_examples' in self.components and current_phase == 'S':
            # Try to extract batch info from STTT component
            indices = getattr(self.components['sttt'], 'current_batch_indices', None)
            batch = getattr(self.components['sttt'], 'current_batch', None)
            outputs = getattr(self.components['sttt'], 'current_batch_outputs', None)
            losses = getattr(self.components['sttt'], 'current_batch_losses', None)
            
            # Update tracking if we have all needed info
            if indices is not None and batch is not None and outputs is not None and losses is not None:
                self.components['hard_examples'].update_with_batch_results(indices, batch, outputs, losses)
        
        # Step 5: Update custom components if any
        for name, component in self.components.items():
            if name not in ['sttt', 'curriculum', 'hard_examples', 'latent_reg']:
                # Call update method if it exists
                if hasattr(component, 'update') and callable(getattr(component, 'update')):
                    component.update(self.global_step, sttt_metrics)
        
        # Periodically update the integrated dataloader (every 100 steps)
        if self.global_step % 100 == 0 and 'sttt' in self.components:
            updated_dataloader = self._create_integrated_dataloader()
            self.components['sttt'].study_dataloader = updated_dataloader
            self.components['sttt'].study_iter = iter(updated_dataloader)
        
        # Combine metrics from all components
        combined_metrics = {
            'global_step': self.global_step,
            **sttt_metrics,
            **latent_reg_components
        }
        
        # Add metrics from other components
        if 'curriculum' in self.components:
            combined_metrics['curriculum_phase'] = self.components['curriculum'].curriculum_dataset.current_phase
        
        if 'hard_examples' in self.components:
            combined_metrics['hard_examples'] = len(self.components['hard_examples'].tracker.hard_examples)
        
        # Record metrics
        self.metrics_history.append(combined_metrics)
        
        # Log metrics periodically
        if self.global_step % self.config.logging_steps == 0:
            self._log_metrics(combined_metrics)
        
        # Increment global step
        self.global_step += 1
        
        return combined_metrics
    
    def _log_metrics(self, metrics: Dict[str, Any]):
        """
        Log metrics from the current step.
        
        Args:
            metrics: Dictionary of metrics
        """
        # Basic metrics
        log_msg = f"Step {metrics['global_step']}: Phase={metrics.get('phase', 'S')}"
        
        # Add loss based on phase
        phase = metrics.get('phase', 'S').lower()
        if f'{phase}_loss' in metrics:
            log_msg += f", Loss={metrics[f'{phase}_loss']:.4f}"
        
        # Add curriculum phase if available
        if 'curriculum_phase' in metrics:
            log_msg += f", Curriculum={metrics['curriculum_phase']}"
        
        # Add hard examples if available
        if 'hard_examples' in metrics:
            log_msg += f", Hard Examples={metrics['hard_examples']}"
        
        # Add latent regularization loss if available
        if 'combined_loss' in metrics:
            log_msg += f", Reg Loss={metrics['combined_loss']:.4e}"
        
        # Log accumulated message
        logger.info(log_msg)
    
    def train(self) -> Dict[str, Any]:
        """
        Train the model for the specified number of steps.
        
        Returns:
            Dictionary of training results
        """
        logger.info(f"Starting training for {self.config.training_steps} steps")
        self.training_start_time = time.time()
        
        try:
            # Training loop
            for _ in range(self.config.training_steps):
                # Execute training step
                step_metrics = self.train_step()
                
                # Evaluation
                if self.global_step % self.config.eval_steps == 0:
                    eval_results = self.evaluate()
                    
                    # Save best model if applicable
                    if eval_results['eval_loss'] < self.best_eval_loss:
                        self.best_eval_loss = eval_results['eval_loss']
                        
                        # Save best model
                        self.save_checkpoint(
                            os.path.join(self.config.output_dir, "best_model"),
                            info={"eval_loss": self.best_eval_loss}
                        )
                        
                        logger.info(f"New best model saved with eval loss: {self.best_eval_loss:.4f}")
                
                # Save checkpoint periodically
                if self.global_step % self.config.save_steps == 0:
                    # Save checkpoint
                    self.save_checkpoint(
                        os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
                    )
                    
                    logger.info(f"Checkpoint saved at step {self.global_step}")
                
                # Check for early stopping or other conditions
                if self._should_stop_training():
                    logger.info("Early stopping triggered")
                    break
            
            # Save final model
            self.save_checkpoint(os.path.join(self.config.output_dir, "final_model"))
            logger.info("Final model saved")
            
            # Generate final reports
            final_reports = self._generate_final_reports()
            
            # Save final reports
            self._save_reports(final_reports)
            
            # Return training results
            training_time = time.time() - self.training_start_time
            
            return {
                'global_step': self.global_step,
                'best_eval_loss': self.best_eval_loss,
                'training_time': training_time,
                'final_reports': final_reports
            }
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            
            # Save interrupt checkpoint
            self.save_checkpoint(os.path.join(self.config.output_dir, "interrupt_checkpoint"))
            logger.info("Interrupt checkpoint saved")
            
            return {
                'global_step': self.global_step,
                'best_eval_loss': self.best_eval_loss,
                'interrupted': True
            }
        
        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            
            # Try to save emergency checkpoint
            try:
                self.save_checkpoint(os.path.join(self.config.output_dir, "error_checkpoint"))
                logger.info("Emergency checkpoint saved")
            except:
                logger.error("Failed to save emergency checkpoint")
            
            raise
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the validation dataset.
        
        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Evaluating at step {self.global_step}")
        
        # Create evaluation dataloader
        eval_dataloader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=16,  # Larger batch size for evaluation
            shuffle=False
        )
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Track metrics
        total_loss = 0.0
        num_eval_steps = 0
        
        # Evaluation loop
        with torch.no_grad():
            for batch in eval_dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Accumulate loss
                total_loss += loss.item()
                num_eval_steps += 1
        
        # Calculate average loss
        avg_loss = total_loss / max(1, num_eval_steps)
        
        # Log evaluation results
        logger.info(f"Evaluation results at step {self.global_step}: Loss={avg_loss:.4f}")
        
        # Return evaluation results
        return {
            'eval_loss': avg_loss,
            'global_step': self.global_step
        }
    
    def save_checkpoint(self, output_dir: str, info: Dict[str, Any] = None):
        """
        Save a model checkpoint.
        
        Args:
            output_dir: Directory to save checkpoint
            info: Optional additional information to save
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save optimizer state
        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        
        # Save component states
        component_states = {}
        for name, component in self.components.items():
            if hasattr(component, 'get_state_dict') and callable(getattr(component, 'get_state_dict')):
                component_states[name] = component.get_state_dict()
            elif hasattr(component, 'get_statistics') and callable(getattr(component, 'get_statistics')):
                component_states[name] = component.get_statistics()
            elif hasattr(component, 'get_metrics') and callable(getattr(component, 'get_metrics')):
                component_states[name] = component.get_metrics()
        
        # Save training state
        training_state = {
            'global_step': self.global_step,
            'best_eval_loss': self.best_eval_loss,
            'component_states': component_states,
            'metrics_history': self.metrics_history[-1000:],  # Save last 1000 steps of metrics
            'config': self.config.__dict__,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add additional info if provided
        if info:
            training_state.update(info)
        
        # Save training state
        torch.save(training_state, os.path.join(output_dir, "training_state.pt"))
        
        # Save config as JSON for easy inspection
        with open(os.path.join(output_dir, "config.json"), 'w') as f:
            # Convert config to dict with basic types
            config_dict = {k: v if isinstance(v, (int, float, str, bool, list, dict)) else str(v) 
                          for k, v in self.config.__dict__.items()}
            json.dump(config_dict, f, indent=2)
    
    def _should_stop_training(self) -> bool:
        """
        Check if training should be stopped early.
        
        Returns:
            True if training should stop, False otherwise
        """
        # Simple implementation - override with more sophisticated logic
        return False
    
    def _generate_final_reports(self) -> Dict[str, Any]:
        """
        Generate final reports for all components.
        
        Returns:
            Dictionary of component reports
        """
        reports = {}
        
        # Get reports from each component
        for name, component in self.components.items():
            if hasattr(component, 'get_statistics') and callable(getattr(component, 'get_statistics')):
                reports[name] = component.get_statistics()
            elif hasattr(component, 'get_metrics') and callable(getattr(component, 'get_metrics')):
                reports[name] = component.get_metrics()
        
        # Add global training statistics
        reports['training'] = {
            'global_step': self.global_step,
            'best_eval_loss': self.best_eval_loss,
            'training_time': time.time() - self.training_start_time if self.training_start_time else 0,
            'enabled_components': self._get_enabled_components()
        }
        
        return reports
    
    def _save_reports(self, reports: Dict[str, Any]):
        """
        Save component reports to disk.
        
        Args:
            reports: Dictionary of component reports
        """
        report_dir = os.path.join(self.config.output_dir, "reports")
        os.makedirs(report_dir, exist_ok=True)
        
        # Save each component report
        for name, report in reports.items():
            # Convert report to JSON-serializable format
            serializable_report = self._make_serializable(report)
            
            # Save as JSON
            with open(os.path.join(report_dir, f"{name}_report.json"), 'w') as f:
                json.dump(serializable_report, f, indent=2)
        
        # Save combined report
        with open(os.path.join(report_dir, "combined_report.json"), 'w') as f:
            json.dump(self._make_serializable(reports), f, indent=2)
        
        logger.info(f"Saved component reports to {report_dir}")
    
    def _make_serializable(self, obj):
        """
        Convert an object to a JSON-serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif hasattr(obj, 'item'):  # torch tensors
            return obj.item()
        else:
            return str(obj)
    
    def _get_enabled_components(self) -> List[str]:
        """
        Get the list of enabled components.
        
        Returns:
            List of enabled component names
        """
        enabled_components = []
        
        if self.config.enable_sttt:
            enabled_components.append("STTT Cycle")
            
        if self.config.enable_curriculum:
            enabled_components.append("Dynamic Curriculum")
            
        if self.config.enable_hard_examples:
            enabled_components.append("Hard Example Amplification")
            
        if self.config.enable_latent_reg:
            enabled_components.append("Latent Space Regularization")
            
        if self.config.enable_consistency:
            enabled_components.append("Internal Consistency")
            
        if self.config.enable_synthetic:
            enabled_components.append("Synthetic Data")
        
        return enabled_components
    
    def load_checkpoint(self, checkpoint_dir: str) -> bool:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_dir: Directory containing checkpoint
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading checkpoint from {checkpoint_dir}")
            
            # Load model
            self.model = type(self.model).from_pretrained(checkpoint_dir)
            self.model.to(self.device)
            
            # Load training state
            training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
            if os.path.exists(training_state_path):
                training_state = torch.load(training_state_path, map_location=self.device)
                
                # Restore training state
                self.global_step = training_state.get('global_step', 0)
                self.best_eval_loss = training_state.get('best_eval_loss', float('inf'))
                
                # Restore metrics history
                if 'metrics_history' in training_state:
                    self.metrics_history = training_state['metrics_history']
                
                logger.info(f"Restored training state: step={self.global_step}, best_loss={self.best_eval_loss}")
                
                # Try to restore component states
                component_states = training_state.get('component_states', {})
                for name, state in component_states.items():
                    if name in self.components:
                        if hasattr(self.components[name], 'load_state_dict') and callable(getattr(self.components[name], 'load_state_dict')):
                            self.components[name].load_state_dict(state)
                            logger.info(f"Restored state for component: {name}")
            
            # Load optimizer state
            optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
            if os.path.exists(optimizer_path):
                optimizer_state = torch.load(optimizer_path, map_location=self.device)
                self.optimizer.load_state_dict(optimizer_state)
                logger.info("Restored optimizer state")
            
            logger.info(f"Successfully loaded checkpoint from {checkpoint_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}", exc_info=True)
            return False


def create_default_metamorph_phase2_config() -> MetaMorphPhase2Config:
    """
    Create a default configuration for MetaMorph Phase 2.
    
    Returns:
        Default configuration object
    """
    # Import config classes
    from sttt_cycle import STTTConfig
    from dynamic_curriculum import DynamicCurriculumConfig
    from hard_example_amplification import HardExampleAmplificationConfig
    from latent_space_regularization import LatentSpaceRegConfig
    
    # Create default STTT config
    sttt_config = STTTConfig()
    
    # Create default curriculum config
    curriculum_config = DynamicCurriculumConfig()
    
    # Create default hard example config
    hard_example_config = HardExampleAmplificationConfig()
    
    # Create default latent space reg config
    latent_reg_config = LatentSpaceRegConfig()
    
    # Create MetaMorph Phase 2 config
    return MetaMorphPhase2Config(
        # Enable core components based on the progression order from the document
        enable_sttt=True,               # Priority 10/10
        enable_curriculum=True,         # Priority 8.5/10
        enable_hard_examples=True,      # Priority 8/10
        enable_latent_reg=False,        # Priority 6/10 - disabled initially
        enable_consistency=False,       # Not implemented in initial phase
        enable_synthetic=False,         # Not implemented in initial phase
        
        # Component configs
        sttt_config=sttt_config.__dict__,
        curriculum_config=curriculum_config.__dict__,
        hard_example_config=hard_example_config.__dict__,
        latent_reg_config=latent_reg_config.__dict__,
        
        # Training settings
        output_dir="./metamorph_phase2_output",
        training_steps=10000,
        eval_steps=100,
        save_steps=500,
        logging_steps=10,
        
        # Model settings
        model_name="gemma-3b-it",
        peft_method="lora",
        
        # Optimizer settings
        optimizer_type="adamw",
        learning_rate=2e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # Hardware settings
        device="cuda",
        mixed_precision=True,
        
        # Reproducibility
        seed=42
    )


# Example usage
"""
# Import required libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

# Create configuration
config = create_default_metamorph_phase2_config()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    device_map="auto"
)

# Apply LoRA if specified
if config.peft_method == "lora":
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)

# Load datasets
train_dataset = ...  # Your training dataset
val_dataset = ...    # Your validation dataset

# Initialize trainer
trainer = MetaMorphPhase2Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config=config
)

# Train the model
results = trainer.train()

# Print results
print(f"Training completed in {results['training_time'] / 3600:.2f} hours")
print(f"Best validation loss: {results['best_eval_loss']:.4f}")
print(f"Final step: {results['global_step']}")
"""
