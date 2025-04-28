import os
import math
import torch
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from transformers import get_cosine_schedule_with_warmup
from torch.cuda.amp import GradScaler, autocast

# Import all our components
from intrinsic_dimension_minimizer import IntrinsicDimensionMinimizer
from plasticity_weighted_reweighter import PlasticityWeightedReweighter
from dynamic_mutual_information import DynamicMutualInformationTracker
from bregman_dynamics import BregmanDynamicsController
from reward_weighted_plasticity import RewardWeightedPlasticityController
from adaptive_distillation import AdaptiveDistillationController
from convergent_neural_architecture_search import ConvergentNeuralArchitectureSearch
from chain_of_thought_injector import ChainOfThoughtInjector, PromptRouter

# Import original components for integration
from enhanced_metaplasticity_optimizer import EnhancedMetaplasticityOptimizer
from enhanced_architecture_controller import EnhancedConvergentArchitectureController
from enhanced_activation_engineering import EnhancedActivationEngineering

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("convergent_metamorph.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ConvergentMetaMorph")

@dataclass
class ConvergentMetaMorphConfig:
    """Configuration for ConvergentMetaMorph with next-gen features."""
    # Model configuration
    model_name: str = "meta-llama/Llama-3-8b-instruct"
    output_dir: str = "./convergent_metamorph_model"
    
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
    
    # Advanced Components Activation Flags
    use_intrinsic_dimension: bool = True
    use_plasticity_reweighter: bool = True
    use_dynamic_mi: bool = True
    use_bregman_dynamics: bool = True
    use_reward_plasticity: bool = True
    use_adaptive_distillation: bool = False  # Requires teacher model
    use_convergent_nas: bool = True
    use_cot_injection: bool = False  # Requires specific settings
    
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
    
    # Reward-weighted Plasticity settings
    rwp_update_rate: float = 0.01
    rwp_min_plasticity: float = 0.2
    rwp_max_plasticity: float = 5.0
    
    # Adaptive Distillation settings
    distill_alpha: float = 0.5
    distill_temperature: float = 2.0
    distill_min_confidence: float = 0.2
    
    # Convergent NAS settings
    nas_error_budget: float = 1.0
    nas_exploration_factor: float = 0.1
    nas_warmup_steps: int = 100
    nas_morph_interval: int = 50
    
    # CoT Injection settings
    cot_adaptive_mode: bool = True
    cot_max_length: int = 512
    cot_lambda: float = 0.5
    
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


class ConvergentMetaMorphTrainer:
    """
    Trainer with integrated next-gen MetaMorph features.
    
    This implementation combines all advanced components into a unified training framework:
    1. Intrinsic Dimension Minimization (entropy-constrained morphing)
    2. Plasticity-Weighted Data Reweighting
    3. Dynamic Mutual Information Objectives
    4. Bregman Dynamics for Formal Convergence
    5. Reward-Weighted Plasticity Alignment
    6. Adaptive Distillation with Confidence Reweighting
    7. Convergent Neural Architecture Search (C-NAS)
    8. Chain-of-Thought Injection
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
        
        logger.info(f"Initialized ConvergentMetaMorphTrainer with device: {self.device}")
    
    def _initialize_components(self):
        """Initialize all the advanced MetaMorph components."""
        # 1. Initialize Enhanced Metaplasticity Optimizer
        if self.config.use_metaplasticity:
            self._initialize_optimizer()
        
        # 2. Initialize Intrinsic Dimension Minimizer
        if self.config.use_intrinsic_dimension:
            self.components['dim_minimizer'] = IntrinsicDimensionMinimizer(
                model=self.model,
                epsilon=self.config.dim_epsilon,
                window_size=self.config.dim_window_size,
                dimension_threshold=self.config.dim_threshold
            )
            logger.info("Initialized Intrinsic Dimension Minimizer")
        
        # 3. Initialize Plasticity-Weighted Reweighter
        if self.config.use_plasticity_reweighter and 'optimizer' in self.components:
            self.components['reweighter'] = PlasticityWeightedReweighter(
                model=self.model,
                optimizer=self.components['optimizer'],
                reweighting_strength=self.config.reweighting_strength,
                max_weight_ratio=self.config.max_weight_ratio,
                min_weight=self.config.min_weight
            )
            logger.info("Initialized Plasticity-Weighted Reweighter")
        
        # 4. Initialize Dynamic Mutual Information Tracker
        if self.config.use_dynamic_mi:
            self.components['mi_tracker'] = DynamicMutualInformationTracker(
                model=self.model,
                num_bins=self.config.mi_num_bins,
                window_size=self.config.mi_window_size,
                update_freq=self.config.mi_update_freq,
                mi_threshold=self.config.mi_threshold
            )
            logger.info("Initialized Dynamic Mutual Information Tracker")
        
        # 5. Initialize Bregman Dynamics Controller
        if self.config.use_bregman_dynamics:
            self.components['bregman_controller'] = BregmanDynamicsController(
                model=self.model,
                divergence_type=self.config.divergence_type,
                step_size_schedule=self.config.step_size_schedule,
                initial_step_size=1.0,
                error_budget=self.config.error_budget
            )
            logger.info("Initialized Bregman Dynamics Controller")
        
        # 6. Initialize Reward-Weighted Plasticity Controller
        if self.config.use_reward_plasticity and 'optimizer' in self.components:
            self.components['rwp_controller'] = RewardWeightedPlasticityController(
                model=self.model,
                optimizer=self.components['optimizer'],
                update_rate=self.config.rwp_update_rate,
                min_plasticity=self.config.rwp_min_plasticity,
                max_plasticity=self.config.rwp_max_plasticity
            )
            logger.info("Initialized Reward-Weighted Plasticity Controller")
        
        # 7. Initialize Adaptive Distillation Controller
        if self.config.use_adaptive_distillation:
            # Note: For real use, you'd need a separate teacher model
            # For this integration example, we're using the same model
            self.components['adaptive_distillation'] = AdaptiveDistillationController(
                student_model=self.model,
                teacher_model=self.model,  # In practice, would be different
                tokenizer=self.tokenizer,
                alpha=self.config.distill_alpha,
                temperature=self.config.distill_temperature,
                min_confidence=self.config.distill_min_confidence,
                adaptive_alpha=True
            )
            logger.info("Initialized Adaptive Distillation Controller")
        
        # 8. Initialize Convergent Neural Architecture Search
        if self.config.use_convergent_nas:
            # Define NAS action space
            action_space = {
                'lora_rank': [4, 8, 16, 32, 64],
                'hidden_dropout': [0.0, 0.1, 0.2, 0.3]
            }
            
            self.components['nas'] = ConvergentNeuralArchitectureSearch(
                model=self.model,
                action_space=action_space,
                error_budget=self.config.nas_error_budget,
                exploration_factor=self.config.nas_exploration_factor,
                warmup_steps=self.config.nas_warmup_steps,
                morph_interval=self.config.nas_morph_interval,
                optimizer=self.components.get('optimizer')
            )
            logger.info("Initialized Convergent Neural Architecture Search")
        
        # 9. Initialize Chain-of-Thought Injector
        if self.config.use_cot_injection:
            # Note: In practice, you'd want a separate teacher model
            self.components['cot_injector'] = ChainOfThoughtInjector(
                teacher_model=None,  # We'll use rule-based generation for integration
                tokenizer=self.tokenizer,
                adaptive_mode=self.config.cot_adaptive_mode,
                max_cot_length=self.config.cot_max_length,
                cot_lambda=self.config.cot_lambda
            )
            
            # Initialize prompt router
            self.components['prompt_router'] = PromptRouter(
                reward_fn=None,  # Will be set during training
                learning_rate=0.01,
                exploration_rate=0.1
            )
            
            logger.info("Initialized Chain-of-Thought Injector and Prompt Router")
        
        # 10. Initialize Enhanced Convergent Architecture Controller
        if self.config.use_dynamic_architecture:
            self.components['architecture_controller'] = EnhancedConvergentArchitectureController(
                model=self.model,
                min_rank=self.config.min_lora_rank,
                max_rank=self.config.max_lora_rank,
                rank_step=self.config.rank_step,
                error_budget=self.config.error_budget,
                scan_freq=self.config.architecture_scan_freq,
                morph_threshold=self.config.morph_threshold,
                optimizer=self.components.get('optimizer')
            )
            self.components['architecture_controller'].schedule_alpha = self.config.schedule_alpha
            logger.info("Initialized Enhanced Convergent Architecture Controller")
        
        # 11. Initialize Information-Theoretic Activation Engineering
        if self.config.use_activation_engineering:
            self.components['activation_engineer'] = EnhancedActivationEngineering(
                self.model, 
                self.device,
                update_freq=self.config.activation_update_freq
            )
            logger.info("Initialized Enhanced Information-Theoretic Activation Engineering")
        
        # 12. Initialize mixed precision training
        if self.config.use_mixed_precision and torch.cuda.is_available():
            self.components['scaler'] = GradScaler()
            logger.info("Initialized Mixed Precision Training")
    
    def _initialize_optimizer(self):
        """Initialize enhanced metaplasticity optimizer with convergence guarantees."""
        # Filter trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Choose base optimizer type
        if self.config.use_adafactor:
            # Use Enhanced AdaFactor with Metaplasticity
            from enhanced_metaplasticity_optimizer import EnhancedAdaFactorWithMetaplasticity
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
        
        # Set total steps for adaptive distillation
        if 'adaptive_distillation' in self.components:
            self.components['adaptive_distillation'].setup(total_steps=total_steps)
            logger.info(f"Set adaptive distillation total steps to {total_steps}")
        
        return total_steps
    
    def _prepare_batch(self, batch):
        """Prepare batch inputs for model."""
        # Format input dict with appropriate tensors
        if isinstance(batch, dict):
            inputs = batch
        else:
            # Convert batch to model inputs
            if hasattr(batch, 'keys'):
                # Dataset returns dict-like object
                text_key = 'text' if 'text' in batch else 'input_text'
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
    
    async def _maybe_apply_cot(self, batch):
        """Optionally apply Chain-of-Thought transformation."""
        if 'cot_injector' not in self.components:
            return batch, None
        
        cot_injector = self.components['cot_injector']
        prompt_router = self.components.get('prompt_router')
        
        # Extract text from batch
        if 'input_ids' in batch and self.tokenizer is not None:
            texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['input_ids']]
        elif 'text' in batch:
            texts = batch['text']
        elif 'input_text' in batch:
            texts = batch['input_text']
        else:
            return batch, None
        
        # Process each text
        transformed_texts = []
        transform_info = []
        
        for text in texts:
            # Decide whether to apply CoT based on router or adaptive mode
            apply_cot = False
            if prompt_router:
                apply_cot = prompt_router.decide(text)
            
            if apply_cot:
                # Apply CoT transformation
                transformed_text, info = await cot_injector.transform_input(text)
                transformed_texts.append(transformed_text)
                transform_info.append(info)
            else:
                transformed_texts.append(text)
                transform_info.append({'applied': False})
        
        # Create new batch with transformed texts
        if 'input_ids' in batch and self.tokenizer is not None:
            # Re-tokenize
            transformed_inputs = self.tokenizer(
                transformed_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=batch['input_ids'].shape[1]
            )
            
            # Copy all other keys
            transformed_batch = {k: v for k, v in batch.items() if k != 'input_ids' and k != 'attention_mask'}
            transformed_batch['input_ids'] = transformed_inputs['input_ids'].to(self.device)
            transformed_batch['attention_mask'] = transformed_inputs['attention_mask'].to(self.device)
            
            if 'labels' in batch:
                transformed_batch['labels'] = batch['labels']  # Keep original labels
            
            return transformed_batch, transform_info
        else:
            # Just replace text
            transformed_batch = {k: v for k, v in batch.items()}
            
            if 'text' in batch:
                transformed_batch['text'] = transformed_texts
            elif 'input_text' in batch:
                transformed_batch['input_text'] = transformed_texts
            
            return transformed_batch, transform_info
    
    def _training_step(self, inputs):
        """Execute a single training step with all MetaMorph components."""
        config = self.config
        
        # Check if we're accumulating gradients
        is_accumulating = (self.global_step + 1) % config.gradient_accumulation_steps != 0
        should_update = not is_accumulating
        
        # Zero gradients at the beginning of accumulation
        if not is_accumulating or self.global_step == 0:
            if 'optimizer' in self.components:
                self.components['optimizer'].zero_grad()
        
        # Apply plasticity-weighted reweighting if available
        sample_weights = None
        if 'reweighter' in self.components:
            sample_weights = self.components['reweighter'].compute_sample_weights(inputs)
        
        # Mixed precision training
        use_mixed_precision = 'scaler' in self.components
        
        # Forward and backward pass with mixed precision
        if use_mixed_precision:
            with autocast():
                # Standard forward pass or distillation
                if 'adaptive_distillation' in self.components:
                    # Use adaptive distillation
                    loss, distillation_info = self.components['adaptive_distillation'].distill_step(inputs)
                else:
                    # Standard forward pass
                    outputs = self.model(**inputs)
                    loss = outputs.loss
                    
                    # Apply sample weights if available
                    if sample_weights is not None:
                        loss = self.components['reweighter'].apply_weight_to_loss(loss)
                
                # Scale loss for gradient accumulation
                loss = loss / config.gradient_accumulation_steps
            
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
            if 'adaptive_distillation' in self.components:
                # Use adaptive distillation
                loss, distillation_info = self.components['adaptive_distillation'].distill_step(inputs)
            else:
                # Standard forward pass
                outputs = self.model(**inputs)
                loss = outputs.loss
                
                # Apply sample weights if available
                if sample_weights is not None:
                    loss = self.components['reweighter'].apply_weight_to_loss(loss)
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / config.gradient_accumulation_steps
            
            # Backward pass
            scaled_loss.backward()
            
            # Update weights if needed
            if should_update and 'optimizer' in self.components:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    config.max_grad_norm
                )
                
                # Update parameters
                self.components['optimizer'].step()
                
                # Update learning rate
                if 'scheduler' in self.components:
                    self.components['scheduler'].step()
        
        # Update trackers after backward pass
        if 'dim_minimizer' in self.components:
            self.components['dim_minimizer'].update(self.global_step)
        
        if 'mi_tracker' in self.components:
            self.components['mi_tracker'].update(self.global_step, inputs)
        
        if 'bregman_controller' in self.components:
            self.components['bregman_controller'].step()
        
        # Get rewards for reward-weighted plasticity
        if 'rwp_controller' in self.components:
            if 'adaptive_distillation' in self.components:
                reward = -distillation_info.get('hybrid_loss', loss.item())
            else:
                reward = -loss.item()
            
            self.components['rwp_controller'].update_with_reward(
                loss=loss.item(),
                outputs=outputs if 'outputs' in locals() else None,
                targets=inputs.get('labels')
            )
        
        # Update NAS reward
        if 'nas' in self.components:
            if 'adaptive_distillation' in self.components:
                reward = -distillation_info.get('hybrid_loss', loss.item())
            else:
                reward = -loss.item()
                
            self.components['nas'].update(self.global_step, reward)
        
        # Return full loss (not scaled by accumulation steps)
        return loss.item(), outputs if 'outputs' in locals() else None
    
    async def train_step(self, batch):
        """Process a single training step with all MetaMorph components."""
        # 1. Prepare batch
        inputs = self._prepare_batch(batch)
        
        # 2. Apply Chain-of-Thought transformation if enabled
        cot_inputs, cot_info = await self._maybe_apply_cot(inputs)
        
        # 3. Perform training step
        if cot_info and any(info['applied'] for info in cot_info):
            # If CoT was applied, we train on both original and CoT versions
            # First train on original
            orig_loss, orig_outputs = self._training_step(inputs)
            
            # Then train on CoT version
            cot_loss, cot_outputs = self._training_step(cot_inputs)
            
            # Get mixed loss
            if 'cot_injector' in self.components:
                loss = self.components['cot_injector'].get_mixed_loss(
                    torch.tensor(orig_loss), torch.tensor(cot_loss)
                ).item()
            else:
                loss = (orig_loss + cot_loss) / 2
            
            # Update prompt router if available
            if 'prompt_router' in self.components:
                # Compute reward as improvement from CoT
                reward = orig_loss - cot_loss
                
                for text, info in zip(batch.get('text', []), cot_info):
                    if isinstance(text, str):
                        self.components['prompt_router'].update(
                            text, info['applied'], reward
                        )
            
            outputs = cot_outputs  # Return the CoT outputs
        else:
            # Standard training step
            loss, outputs = self._training_step(inputs)
        
        return loss, outputs
    
    async def train(self, train_dataloader, val_dataloader=None, callbacks=None):
        """Main training loop with all MetaMorph components integrated."""
        config = self.config
        
        # Calculate total steps
        total_steps = self.setup_training(
            train_dataloader.dataset if hasattr(train_dataloader, 'dataset') else None
        )
        
        # Training tracking
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        training_stats = []
        
        logger.info(f"Starting training for {config.epochs} epochs, {total_steps} total steps")
        
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
                # Process batch with all components
                loss, outputs = await self.train_step(batch)
                epoch_loss += loss
                
                # Update global step if accumulation is complete
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    self.global_step += 1
                    
                    # Check for architecture morphing
                    self._check_architecture_morphing()
                    
                    # Logging
                    if self.global_step % config.logging_steps == 0:
                        self._log_training_progress(loss)
                        
                        # Record stats
                        training_stats.append({
                            'step': self.global_step,
                            'epoch': epoch + 1,
                            'loss': loss,
                            'learning_rate': self.components['optimizer'].param_groups[0]['lr'] 
                                           if 'optimizer' in self.components else 0
                        })
                    
                    # Validation
                    if val_dataloader is not None and self.global_step % config.eval_steps == 0:
                        val_loss = await self.evaluate(val_dataloader)
                        logger.info(f"Validation loss: {val_loss:.4f}")
                        
                        # Save best model
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint(
                                os.path.join(config.output_dir, "checkpoints", "best_model"),
                                {'val_loss': val_loss, 'step': self.global_step}
                            )
                    
                    # Save periodic checkpoint
                    if self.global_step % config.save_steps == 0:
                        self.save_checkpoint(
                            os.path.join(config.output_dir, "checkpoints", f"step_{self.global_step}"),
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
                val_loss = await self.evaluate(val_dataloader)
                logger.info(f"End of epoch validation - Loss: {val_loss:.4f}")
                
                # Save if best
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(
                        os.path.join(config.output_dir, "checkpoints", "best_model"),
                        {'val_loss': val_loss, 'step': self.global_step, 'epoch': epoch+1}
                    )
            
            # Save epoch checkpoint
            self.save_checkpoint(
                os.path.join(config.output_dir, "checkpoints", f"epoch_{epoch+1}"),
                {'step': self.global_step, 'epoch': epoch+1}
            )
            
            # Callbacks
            if callbacks and 'on_epoch_end' in callbacks:
                callbacks['on_epoch_end'](self, epoch=epoch, loss=avg_loss)
        
        # Training complete
        logger.info(f"Training complete - {self.global_step} steps across {config.epochs} epochs")
        
        # Generate final reports for all components
        self._generate_component_reports()
        
        # Save final model
        self.save_checkpoint(
            os.path.join(config.output_dir, "final_model"),
            {'step': self.global_step, 'epochs': config.epochs}
        )
        
        return {
            'training_stats': training_stats,
            'best_val_loss': self.best_val_loss,
            'final_step': self.global_step
        }
    
    def _check_architecture_morphing(self):
        """Check if architecture should be morphed based on different components."""
        # 1. Check based on Intrinsic Dimension
        dim_decision = False
        if 'dim_minimizer' in self.components:
            dim_minimizer = self.components['dim_minimizer']
            increase_rank = dim_minimizer.should_increase_rank()
            decrease_rank = dim_minimizer.should_decrease_rank()
            
            if increase_rank or decrease_rank:
                logger.info(f"Intrinsic dimension suggests {'increasing' if increase_rank else 'decreasing'} rank")
                dim_decision = True
        
        # 2. Check based on Dynamic Mutual Information
        mi_decision = False
        if 'mi_tracker' in self.components:
            mi_tracker = self.components['mi_tracker']
            increase_capacity = mi_tracker.should_increase_capacity()
            decrease_capacity = mi_tracker.should_decrease_capacity()
            
            if increase_capacity or decrease_capacity:
                logger.info(f"Mutual information suggests {'increasing' if increase_capacity else 'decreasing'} capacity")
                mi_decision = True
        
        # 3. Convergent Neural Architecture Search
        nas_decision = False
        if 'nas' in self.components:
            nas = self.components['nas']
            if nas.should_morph_architecture():
                logger.info("NAS suggests architecture morphing")
                modifications, error = nas.morph_architecture()
                if modifications:
                    logger.info(f"NAS morphed architecture: {modifications}")
                    nas_decision = True
        
        # 4. EnhancedConvergentArchitectureController
        if 'architecture_controller' in self.components:
            arch_ctrl = self.components['architecture_controller']
            scan_results = arch_ctrl.scan_architecture(self.global_step)
            
            if scan_results and scan_results.get('morph_needed', False):
                # Apply architecture changes with error tracking
                morph_plan = arch_ctrl.morph_architecture(scan_results)
                
                if morph_plan:
                    logger.info(f"Architecture controller morphed architecture at step {self.global_step}")
                    
                    # Check convergence guarantee
                    if arch_ctrl.cumulative_error > arch_ctrl.error_budget:
                        logger.warning(f"Warning: Architecture error budget exceeded: "
                                     f"{arch_ctrl.cumulative_error:.4e} > {arch_ctrl.error_budget:.4e}")
    
    def _log_training_progress(self, loss):
        """Log training progress with component stats."""
        config = self.config
        
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
        
        # Get intrinsic dimension if available
        if 'dim_minimizer' in self.components:
            dim = self.components['dim_minimizer'].current_intrinsic_dim
            dim_info = f", Intrinsic dim: {dim:.2f}" if dim is not None else ""
        else:
            dim_info = ""
        
        # Get mutual information if available
        if 'mi_tracker' in self.components:
            mi = self.components['mi_tracker'].get_normalized_mi()
            mi_info = f", Normalized MI: {mi:.4f}"
        else:
            mi_info = ""
        
        # Get CoT stats if available
        if 'cot_injector' in self.components:
            cot_stats = self.components['cot_injector'].get_stats()
            cot_rate = cot_stats.get('cot_application_rate', 0)
            cot_info = f", CoT rate: {cot_rate:.2f}"
        else:
            cot_info = ""
        
        logger.info(f"Step {self.global_step}: "
                   f"Loss: {loss:.4f}, LR: {current_lr:.8f}"
                   f"{plasticity_info}{arch_info}{dim_info}{mi_info}{cot_info}")
    
    async def evaluate(self, eval_dataloader):
        """Evaluate model on validation data."""
        self.model.eval()
        total_loss = 0
        eval_steps = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                inputs = self._prepare_batch(batch)
                
                # Forward pass
                outputs = self.model(**inputs)
                loss = outputs.loss
                
                total_loss += loss.item()
                eval_steps += 1
        
        avg_loss = total_loss / max(1, eval_steps)
        return avg_loss
    
    def save_checkpoint(self, save_path, info=None):
        """Save model checkpoint with components state."""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save component states
        component_states = {}
        
        # Save optimizer and scheduler state
        if 'optimizer' in self.components:
            component_states['optimizer'] = {
                'state_dict': self.components['optimizer'].state_dict()
            }
            
            if 'scheduler' in self.components:
                component_states['scheduler'] = {
                    'state_dict': self.components['scheduler'].state_dict()
                }
        
        # Save component metrics
        for name, component in self.components.items():
            if hasattr(component, 'get_stats'):
                component_states[f"{name}_stats"] = component.get_stats()
        
        # Save training info
        if info is None:
            info = {}
        
        checkpoint_data = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'component_states': component_states,
            **info
        }
        
        torch.save(checkpoint_data, os.path.join(save_path, "training_state.bin"))
        logger.info(f"Saved checkpoint to {save_path}")
    
    def load_checkpoint(self, load_path):
        """Load checkpoint with component states."""
        # Load model and tokenizer
        self.model.from_pretrained(load_path)
        self.tokenizer.from_pretrained(load_path)
        
        # Load training state
        try:
            checkpoint_data = torch.load(os.path.join(load_path, "training_state.bin"))
            
            # Restore training state
            self.epoch = checkpoint_data.get('epoch', 0)
            self.global_step = checkpoint_data.get('global_step', 0)
            self.best_val_loss = checkpoint_data.get('best_val_loss', float('inf'))
            
            # Restore component states
            component_states = checkpoint_data.get('component_states', {})
            
            # Restore optimizer and scheduler
            if 'optimizer' in component_states and 'optimizer' in self.components:
                state_dict = component_states['optimizer'].get('state_dict')
                if state_dict:
                    self.components['optimizer'].load_state_dict(state_dict)
                    
                if 'scheduler' in component_states and 'scheduler' in self.components:
                    state_dict = component_states['scheduler'].get('state_dict')
                    if state_dict:
                        self.components['scheduler'].load_state_dict(state_dict)
            
            logger.info(f"Loaded checkpoint from {load_path} at epoch {self.epoch}, step {self.global_step}")
            return True
        except (FileNotFoundError, KeyError) as e:
            logger.error(f"Error loading checkpoint: {e}")
            return False
    
    def _generate_component_reports(self):
        """Generate detailed reports for all components at the end of training."""
        reports = {}
        
        # 1. Optimizer and Metaplasticity report
        if 'optimizer' in self.components:
            if hasattr(self.components['optimizer'], 'get_stability_report'):
                reports['optimizer'] = self.components['optimizer'].get_stability_report()
            elif hasattr(self.components['optimizer'], 'get_plasticity_stats'):
                reports['optimizer'] = {
                    'plasticity_stats': self.components['optimizer'].get_plasticity_stats()
                }
        
        # 2. Architecture Controller report
        if 'architecture_controller' in self.components:
            reports['architecture'] = self.components['architecture_controller'].get_architecture_report()
            
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
        
        # 3. Intrinsic Dimension report
        if 'dim_minimizer' in self.components:
            reports['intrinsic_dimension'] = self.components['dim_minimizer'].get_stats()
        
        # 4. Mutual Information report
        if 'mi_tracker' in self.components:
            reports['mutual_information'] = self.components['mi_tracker'].get_stats()
        
        # 5. Bregman Dynamics report
        if 'bregman_controller' in self.components:
            reports['bregman_dynamics'] = self.components['bregman_controller'].get_stats()
        
        # 6. Reward-Weighted Plasticity report
        if 'rwp_controller' in self.components:
            reports['reward_plasticity'] = self.components['rwp_controller'].get_stats()
        
        # 7. Adaptive Distillation report
        if 'adaptive_distillation' in self.components:
            reports['adaptive_distillation'] = self.components['adaptive_distillation'].get_stats()
        
        # 8. Neural Architecture Search report
        if 'nas' in self.components:
            reports['neural_architecture_search'] = self.components['nas'].get_architecture_exploration_report()
        
        # 9. Chain-of-Thought Injection report
        if 'cot_injector' in self.components:
            reports['cot_injection'] = self.components['cot_injector'].get_stats()
            
            if 'prompt_router' in self.components:
                reports['prompt_router'] = self.components['prompt_router'].get_stats()
        
        # 10. Activation Engineering report
        if 'activation_engineer' in self.components:
            reports['activation_engineering'] = self.components['activation_engineer'].get_activation_selection_report()
        
        # Save reports to output directory
        import json
        report_dir = os.path.join(self.config.output_dir, "component_reports")
        os.makedirs(report_dir, exist_ok=True)
        
        for name, report in reports.items():
            # Convert any non-serializable values to strings
            serializable_report = {}
            for k, v in report.items():
                if isinstance(v, (dict, list, str, int, float, bool)) or v is None:
                    serializable_report[k] = v
                else:
                    serializable_report[k] = str(v)
            
            with open(os.path.join(report_dir, f"{name}_report.json"), 'w') as f:
                json.dump(serializable_report, f, indent=2)
        
        logger.info(f"Saved component reports to {report_dir}")
        return reports


# Example usage
def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
    
    # Set up config
    config = ConvergentMetaMorphConfig(
        model_name="facebook/opt-125m",  # Use a small model for demo
        output_dir="./metamorph_demo",
        
        # Disable components requiring teacher model for this demo
        use_adaptive_distillation=False,
        use_cot_injection=False,
        
        # Training settings for demo
        epochs=1,
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        
        # Enable core features
        use_intrinsic_dimension=True, 
        use_plasticity_reweighter=True,
        use_dynamic_mi=True,
        use_bregman_dynamics=True,
        use_reward_plasticity=True,
        use_convergent_nas=True,
        use_dynamic_architecture=True,
        use_activation_engineering=True
    )
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Apply LoRA for parameter-efficient fine-tuning
    if config.use_peft:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, peft_config)
        
    # Create dummy dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, n_samples=100):
            self.data = [f"This is a test sample {i} with some text." for i in range(n_samples)]
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            return {"text": self.data[idx]}
    
    train_dataset = DummyDataset(100)
    val_dataset = DummyDataset(20)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.micro_batch_size, 
        shuffle=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config.micro_batch_size, 
        shuffle=False
    )
    
    # Create trainer
    trainer = ConvergentMetaMorphTrainer(model, tokenizer, config)
    
    # Run training
    import asyncio
    asyncio.run(trainer.train(train_loader, val_loader))
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
