import torch
import logging
import math
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
import time
import random
import copy
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.distributions as dist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import pickle
import os

# Add import for the latent space regularization
from latent_space_regularization import (
    LatentSpaceRegularizer, 
    LatentSpaceRegConfig,
    add_latent_regularization_to_loss,
    create_latent_space_regularizer
)
from hard_example_amplification import HardExampleAmplifier
from dynamic_curriculum import DynamicCurriculumConstructor
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("STTTCycle")

@dataclass
class STTTConfig:
    """Configuration for STTT Cycle (Study-Test-Test-Test Loop)."""
    # Study phase settings
    study_batch_size: int = 4
    study_grad_accum_steps: int = 8
    study_learning_rate: float = 2e-5
    
    # Test phases settings
    t1_batch_size: int = 8  # Held-out validation
    t2_batch_size: int = 4  # Counterfactual validation
    t3_batch_size: int = 4  # Adversarial validation
    
    # STTT Cycle control
    cycle_length: int = 100  # Steps per complete STTT cycle
    study_steps: int = 80    # Number of study steps per cycle
    t1_steps: int = 10       # Number of T1 validation steps per cycle
    t2_steps: int = 5        # Number of T2 counterfactual validation steps per cycle
    t3_steps: int = 5        # Number of T3 adversarial validation steps per cycle
    
    # Intervention thresholds
    t1_loss_threshold: float = 0.2  # Maximum allowed T1 loss increase from baseline
    t2_loss_threshold: float = 0.3  # Maximum allowed T2 loss increase from baseline
    t3_loss_threshold: float = 0.4  # Maximum allowed T3 loss increase from baseline
    
    # Dynamic adjustment
    dynamic_cycle_adjustment: bool = True  # Adjust cycle parameters based on performance
    
    # Counterfactual generation
    counterfactual_types: List[str] = field(default_factory=lambda: ["sector_change", "stage_change", "geography_change"])
    counterfactual_probability: float = 0.8  # Probability of applying counterfactual transformation
    
    # Adversarial generation
    adversarial_types: List[str] = field(default_factory=lambda: ["missing_fields", "noisy_descriptions", "conflicting_signals"])
    adversarial_severity: float = 0.5  # Severity of adversarial perturbations (0-1)
    
    # Test phase weights for interventions
    t1_weight: float = 0.5  # Weight of T1 loss for intervention decisions
    t2_weight: float = 0.3  # Weight of T2 loss for intervention decisions
    t3_weight: float = 0.2  # Weight of T3 loss for intervention decisions
    
    # Logging
    log_frequency: int = 10  # Log stats every N steps
    verbose: bool = True     # Detailed logging
    
    # Enhancement 1: Test Loss Derivative Monitoring
    enable_loss_derivative_monitoring: bool = True
    derivative_warning_threshold: float = 0.05  # Threshold for derivative-based warnings
    
    # Enhancement 2: Meta-Baseline Volatility
    track_baseline_volatility: bool = True
    volatility_threshold: float = 0.1  # Threshold for baseline volatility alerts
    
    # Enhancement 3: Dynamic Grad Accum Tuning
    dynamic_grad_accum: bool = True
    min_grad_accum_steps: int = 4
    max_grad_accum_steps: int = 16
    intervention_lookback_window: int = 100  # Window for calculating intervention rate
    
    # Enhancement 4: Dynamic Adversarial Boosting
    dynamic_adversarial_severity: bool = True
    min_adversarial_severity: float = 0.3
    max_adversarial_severity: float = 0.8
    
    # Enhancement 5: Intervention-Based Curriculum Shift
    enable_curriculum_shift: bool = True
    curriculum_difficulty_boost: float = 0.2  # Increase in difficulty after intervention
    difficulty_decay_rate: float = 0.95  # Rate at which difficulty returns to normal
    
    # New Enhancement: Feature-Specific Curriculum Difficulty
    feature_correlation_window: int = 50  # Window size for feature-loss correlation
    feature_difficulty_step: float = 0.1  # Step size for feature difficulty updates
    feature_noise_std: float = 0.05  # Standard deviation for feature noise
    
    # New Enhancement: Adversarial Loss Regularization
    adversarial_reg_weight: float = 0.05  # Weight for adversarial loss in study phase
    adversarial_reg_samples: int = 1  # Number of adversarial samples per study step
    
    # New Enhancement: Cross-Cycle Knowledge Distillation
    enable_knowledge_distillation: bool = True
    distillation_alpha: float = 0.7  # Weight for KL divergence in distillation loss
    distillation_beta: float = 0.3  # Weight for original loss in distillation loss
    teacher_model_cycles: int = 3  # Number of cycles to keep teacher models
    
    # New Enhancement: Enhanced Counterfactual Generation
    use_enhanced_counterfactuals: bool = True
    counterfactual_vae_weight: float = 0.5  # Weight for VAE-generated counterfactuals
    counterfactual_loss_constraint: float = 0.2  # Max allowed loss change in counterfactuals
    
    # New Enhancement: Gradient-Based Adversarial Attacks
    use_gradient_adversarial: bool = True
    fgsm_epsilon_base: float = 0.01  # Base perturbation size for FGSM
    fgsm_epsilon_scale: float = 0.1  # Scales epsilon based on severity
    
    # New Enhancement: Concrete Intervention Strategies
    intervention_lr_alpha: float = 0.1  # Learning rate adjustment factor
    intervention_example_reweight_lambda: float = 1.0  # Failed example reweighting
    intervention_reg_boost_beta: float = 0.01  # Regularization boost factor
    intervention_reg_reset_steps: int = 50  # Steps to maintain boosted regularization
    
    # New Enhancement: Smoothed Loss Derivatives
    use_ema_derivatives: bool = True
    ema_alpha: float = 0.3  # EMA smoothing factor
    
    # New Enhancement: Curriculum Learning Optimization
    curriculum_learning_step_size: float = 0.05  # Step size for curriculum updates
    
    # New Enhancement: Soft Phase Transitions
    use_soft_phase_transitions: bool = True
    transition_window_size: int = 5  # Steps for soft transition between phases
    
    # New Enhancement: Memory-Efficient History
    use_reservoir_sampling: bool = True
    reservoir_size: int = 100  # Size of reservoir for sampling history
    history_downsample_factor: int = 5  # Only store every n-th observation
    
    # New Enhancement: Multi-Objective Optimization
    use_multi_objective_opt: bool = True
    pareto_weight_update_rate: float = 0.2  # Rate to update Pareto weights
    
    # New Enhancement: Generalization Gap Metrics
    track_generalization_gaps: bool = True
    
    # Enhancement: Variational Counterfactual Generation
    use_vae_counterfactuals: bool = True
    vae_latent_dim: int = 32  # Dimensionality of VAE latent space
    vae_kl_weight: float = 0.5  # KL divergence weight in VAE loss
    vae_training_epochs: int = 50  # Epochs to train the VAE
    vae_batch_size: int = 64  # Batch size for VAE training
    vae_cache_latents: bool = True  # Whether to cache latent codes for efficiency
    
    # Enhancement: PGD for Adversarial Attacks
    use_pgd_adversarial: bool = True
    pgd_steps: int = 3  # Number of PGD iterations
    pgd_alpha: float = 0.005  # Step size for PGD updates
    pgd_constraint_types: List[str] = field(default_factory=lambda: ["non_negative", "norm_preserving"])
    
    # Enhancement: Adversarial Loss Regularization
    use_adversarial_regularization: bool = True
    adversarial_reg_weight: float = 0.05  # Weight of adversarial loss in study phase
    adversarial_reg_warmup_steps: int = 500  # Warmup steps before applying regularization
    adversarial_reg_sample_freq: int = 5  # How often to sample adversarial examples for regularization
    
    # Enhancement: Uncertainty-Aware Interventions
    use_uncertainty_interventions: bool = True
    uncertainty_samples: int = 5  # Number of samples for GP
    gp_kernel_scale: float = 1.0  # Scale parameter for RBF kernel
    gp_noise: float = 0.1  # Noise parameter for GP
    intervention_params: List[str] = field(default_factory=lambda: ["learning_rate", "dropout", "weight_decay", "reg_weight", "curriculum_difficulty"])
    
    # Latent Space Regularization integration
    use_latent_regularization: bool = False  # Whether to use latent space regularization
    latent_reg_weight: float = 0.1  # Weight for latent space regularization
    phase_specific_reg: bool = True  # Whether to use phase-specific regularization
    study_reg_weight: float = 0.1  # Weight for study phase
    t1_reg_weight: float = 0.05  # Weight for T1 phase
    t2_reg_weight: float = 0.15  # Weight for T2 phase
    t3_reg_weight: float = 0.2  # Weight for T3 phase
    adapt_reg_on_intervention: bool = True  # Adapt regularization on intervention
    intervention_reg_boost: float = 0.05  # Boost to regularization on intervention
    
    # NEW: Bayesian regularization tuning
    bayesian_weight: float = 0.05  # Weight for Bayesian regularization
    t2_bayesian_boost: float = 2.0  # Boost factor for Bayesian weight in T2 phase
    
    # NEW: PGD parameters for adversarial training
    pgd_steps: int = 3  # Steps for PGD adversarial generation
    pgd_alpha: float = 0.005  # Step size for PGD
    pgd_adaptive: bool = True  # Adaptively adjust PGD parameters based on T3 performance
    
    # NEW: Adversarial regularization warmup
    adversarial_reg_warmup_steps: int = 500  # Steps before applying adversarial regularization
    
    # NEW: Enhanced adversarial types
    adversarial_types: List[str] = field(default_factory=lambda: [
        "missing_fields", "noisy_descriptions", "conflicting_signals", 
        "outlier_metrics", "extreme_claims", "inconsistent_data"
    ])
    
    # NEW: VAE pre-training
    vae_pretrain_steps: int = 1000  # Steps for VAE pre-training
    use_pretrained_embeddings: bool = False  # Use pre-trained embeddings (e.g., BERT)
    pretrained_embedding_model: str = "bert-base-uncased"  # Model for pre-trained embeddings
    
    # NEW: Visualization
    enable_latent_visualization: bool = True  # Enable latent space visualization
    visualization_frequency: int = 1000  # Frequency of visualization generation
    visualization_path: str = "./visualizations"  # Path to save visualizations
    
    # NEW: Pruning
    enable_pruning: bool = False  # Enable model pruning after training
    pruning_threshold: float = 0.05  # Threshold for pruning
    pruning_importance_threshold: float = 0.1  # Importance threshold for pruning
    gradual_pruning_steps: int = 100  # Steps for gradual pruning
    
    # ... other parameters ...


class STTTCycle:
    """
    Implements the Study-Test-Test-Test (STTT) Cycle for robust model training.
    
    The STTT cycle consists of:
    - Study Phase (S): Gradient descent on labeled founder-VC matching examples
    - Test Phase 1 (T1): Immediate validation on held-out founders
    - Test Phase 2 (T2): Validation against counterfactual founders (changed sectors/stages)
    - Test Phase 3 (T3): Adversarial validation (partial information, noisy founders)
    
    This cycle detects shallow fitting, inability to reason over data perturbations,
    and dynamically intervenes when test loss surges.
    """
    
    def __init__(
        self, 
        model,
        optimizer,
        study_dataloader,
        t1_dataloader,
        t2_generator: Optional[Callable] = None,
        t3_generator: Optional[Callable] = None,
        config: STTTConfig = None,
        device = None,
        val_dataset = None,
        hard_example_config=None,
        curriculum_config=None, 
        sttt_config=None,
      
        latent_reg_config: LatentSpaceRegConfig = None,
        external_data_for_vae: Optional[torch.utils.data.DataLoader] = None  # NEW: External data for VAE
    ):
        """
        Initialize the STTT Cycle.
        
        Args:
            model: The model to train
            optimizer: The optimizer to use for training
            study_dataloader: DataLoader for study phase (training data)
            t1_dataloader: DataLoader for T1 phase (held-out validation data)
            t2_generator: Function to generate counterfactual examples for T2 phase
            t3_generator: Function to generate adversarial examples for T3 phase
            config: Configuration for the STTT cycle
            device: Device to run on ('cuda' or 'cpu')
            latent_reg_config: Configuration for latent space regularization
            external_data_for_vae: Optional DataLoader for external data for VAE pre-training
        """
        self.model = model
        self.optimizer = optimizer
        self.study_dataloader = study_dataloader
        self.t1_dataloader = t1_dataloader
        self.t2_generator = t2_generator
        self.t3_generator = t3_generator
        self.config = config or STTTConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize phase iterators
        self.study_iter = iter(self.study_dataloader)
        self.t1_iter = iter(self.t1_dataloader)
        
        # Phase tracking
        self.current_phase = "S"  # One of {"S", "T1", "T2", "T3"}
        self.phase_step = 0
        self.global_step = 0
        
        # Initialize buffers for tracking loss and metrics
        # Replace unbounded deques with ReservoirBuffer for memory efficiency
        self.t1_loss_history = ReservoirBuffer(max_size=100)
        self.t2_loss_history = ReservoirBuffer(max_size=100)
        self.t3_loss_history = ReservoirBuffer(max_size=100)
        self.study_loss_history = ReservoirBuffer(max_size=100)
        
        # Track baseline losses for comparison
        self.t1_baseline_loss = None
        self.t2_baseline_loss = None
        self.t3_baseline_loss = None
        
        # Track metrics for all phases
        self.phase_metrics = {
            "S": ReservoirBuffer(max_size=100),
            "T1": ReservoirBuffer(max_size=100),
            "T2": ReservoirBuffer(max_size=100),
            "T3": ReservoirBuffer(max_size=100)
        }
        
        # Track accuracy for all phases
        self.phase_accuracy = {
            "S": ReservoirBuffer(max_size=100),
            "T1": ReservoirBuffer(max_size=100),
            "T2": ReservoirBuffer(max_size=100),
            "T3": ReservoirBuffer(max_size=100)
        }
        
        # Enhancement 1: Test Loss Derivative Monitoring
        self.prev_t1_loss = None
        self.prev_t2_loss = None
        self.prev_t3_loss = None
        self.t1_derivatives = ReservoirBuffer(max_size=100)
        self.t2_derivatives = ReservoirBuffer(max_size=100)
        self.t3_derivatives = ReservoirBuffer(max_size=100)
        
        # Enhancement 2: Meta-Baseline Volatility
        self.t1_baseline_history = ReservoirBuffer(max_size=20)
        self.t2_baseline_history = ReservoirBuffer(max_size=20)
        self.t3_baseline_history = ReservoirBuffer(max_size=20)
        
        # Enhancement 3: Dynamic Grad Accum Tuning
        self.current_grad_accum_steps = self.config.study_grad_accum_steps
        
        # Enhancement 4: Dynamic Adversarial Boosting
        self.current_adversarial_severity = self.config.adversarial_severity
        
        # Enhancement 5: Intervention-Based Curriculum Shift
        self.current_curriculum_difficulty = 0.0  # Starts at normal difficulty
        
        # Enhancement: Smoothed Loss Derivatives with EMA
        if self.config.use_ema_derivatives:
            self.ema_t1_loss = None
            self.ema_t2_loss = None
            self.ema_t3_loss = None
            self.ema_t1_derivatives = ReservoirBuffer(max_size=100)
            self.ema_t2_derivatives = ReservoirBuffer(max_size=100)
            self.ema_t3_derivatives = ReservoirBuffer(max_size=100)
        
        # Enhancement: Concrete Intervention Strategies
        self.original_learning_rate = self.config.study_learning_rate
        self.intervention_reg_boost_steps_left = 0
        
        # Enhancement: Failed examples replay buffer
        self.failed_examples_buffer = ReservoirBuffer(max_size=100)
        
        # Store feature difficulties for curriculum learning
        self.feature_difficulties = defaultdict(float)
        
        # Feature history for correlation with loss
        self.feature_history = defaultdict(lambda: ReservoirBuffer(max_size=self.config.feature_correlation_window))
        self.loss_history = ReservoirBuffer(max_size=self.config.feature_correlation_window)
        
        # Initialize teacher models for knowledge distillation
        self.teacher_models = []
        if self.config.enable_knowledge_distillation:
            self._initialize_teacher_models()
            
        # Track intervention history
        self.intervention_history = ReservoirBuffer(max_size=100)
        self.intervention_steps = ReservoirBuffer(max_size=100)
        
        # Track VAE constraint violations
        self.vae_constraint_violations = 0
        self.vae_total_counterfactuals = 0
        self.vae_constraint_logs = ReservoirBuffer(max_size=100)
        
        # Track generalization gaps
        self.generalization_gaps = {
            't1_gap': ReservoirBuffer(max_size=20),
            't2_gap': ReservoirBuffer(max_size=20),
            't3_gap': ReservoirBuffer(max_size=20)
        }
        
        # Initialize Pareto weights for multi-objective optimization
        self.pareto_weights = {'t1': 0.5, 't2': 0.3, 't3': 0.2}
        
        # Store examples from each phase for analysis and VAE training
        self.s_examples_buffer = ReservoirBuffer(max_size=50)
        self.t1_examples_buffer = ReservoirBuffer(max_size=50)
        self.t2_examples_buffer = ReservoirBuffer(max_size=50)
        self.t3_examples_buffer = ReservoirBuffer(max_size=50)
        
        # Create visualization directory if needed
        if self.config.enable_latent_visualization:
            os.makedirs(self.config.visualization_path, exist_ok=True)
        
        self.hard_amplifier = HardExampleAmplifier(model, train_dataset, hard_example_config, device)
        self.curriculum = DynamicCurriculumConstructor(model, train_dataset, val_dataset, curriculum_config, device=device)
        self.sttt_cycle = STTTCycle(
            model=model,
            optimizer=optimizer,
            study_dataloader=self.curriculum.get_dataloader(batch_size=32),
            t1_dataloader=DataLoader(val_dataset, batch_size=32),
            t2_generator=self.curriculum.t2_generator,
            t3_generator=lambda batch: self.hard_amplifier.get_augmented_hard_example_batch(len(batch['input_ids'])),
            config=sttt_config,
            device=device
        )
        
    def _reset_iterators(self):
        """Reset data iterators for each phase."""
        self.study_iter = iter(self.study_dataloader)
        self.t1_iter = iter(self.t1_dataloader)
    
    def _get_next_batch(self, phase: str) -> Dict[str, torch.Tensor]:
        """
        Get the next batch for the given phase.
        
        Args:
            phase: One of {"S", "T1", "T2", "T3"}
            
        Returns:
            Batch of data for the given phase
        """
        # Check if dataloaders are valid and not empty
        if phase == "S" and (not hasattr(self, 'study_dataloader') or len(self.study_dataloader) == 0):
            raise ValueError("Study dataloader is empty or not initialized")
        if phase in ["T1", "T2", "T3"] and (not hasattr(self, 't1_dataloader') or len(self.t1_dataloader) == 0):
            raise ValueError(f"T1 dataloader needed for phase {phase} is empty or not initialized")
        
        try:
            if phase == "S":
                # Study phase - get next training batch
                batch = next(self.study_iter)
            elif phase == "T1":
                # T1 phase - get next validation batch
                batch = next(self.t1_iter)
            elif phase == "T2":
                # T2 phase - generate counterfactual examples
                # First get a regular batch, then transform it
                base_batch = next(self.t1_iter)
                if self.t2_generator is not None:
                    batch = self.t2_generator(base_batch)
                else:
                    # Fallback to simple counterfactual generation
                    batch = self._generate_counterfactual(base_batch)
            elif phase == "T3":
                # T3 phase - generate adversarial examples
                # First get a regular batch, then transform it
                base_batch = next(self.t1_iter)
                if self.t3_generator is not None:
                    batch = self.t3_generator(base_batch)
                else:
                    # Fallback to simple adversarial generation
                    batch = self._generate_adversarial(base_batch)
            else:
                raise ValueError(f"Unknown phase: {phase}")
        except StopIteration:
            # Reset the iterator if we've gone through the dataset
            self._reset_iterators()
            return self._get_next_batch(phase)
        
        # Validate the batch is not None and has data
        if batch is None:
            raise ValueError(f"Received None batch from {phase} dataloader")
            
        if len(batch) == 0:
            raise ValueError(f"Received empty batch from {phase} dataloader")
            
        # Verify at least one tensor exists in the batch
        tensor_found = False
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                tensor_found = True
                break
                
        if not tensor_found:
            raise ValueError(f"No tensor found in batch from {phase} dataloader")
            
        # Move batch to device
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    def _generate_counterfactual(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Generate counterfactual examples (T2 phase) using an improved approach that 
        ensures more realistic transformations.
        
        Args:
            batch: Input batch
            
        Returns:
            Batch with counterfactual examples
        """
        # Clone the batch to avoid modifying the original
        counterfactual_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        if not self.config.use_enhanced_counterfactuals:
            # Apply simple counterfactual transformations (original approach)
            if 'sector' in counterfactual_batch and 'sector_change' in self.config.counterfactual_types:
                # Change the sector for a subset of examples
                batch_size = counterfactual_batch['sector'].size(0)
                mask = torch.rand(batch_size) < self.config.counterfactual_probability
                
                if mask.sum() > 0:
                    sector_ids = counterfactual_batch['sector'][mask]
                    # Shift sectors by 1 (wrapping around)
                    num_sectors = sector_ids.max() + 1
                    counterfactual_batch['sector'][mask] = (sector_ids + 1) % num_sectors
            
            if 'stage' in counterfactual_batch and 'stage_change' in self.config.counterfactual_types:
                # Change the funding stage for a subset of examples
                batch_size = counterfactual_batch['stage'].size(0)
                mask = torch.rand(batch_size) < self.config.counterfactual_probability
                
                if mask.sum() > 0:
                    stage_ids = counterfactual_batch['stage'][mask]
                    # Shift stages by 1 (wrapping around)
                    num_stages = stage_ids.max() + 1
                    counterfactual_batch['stage'][mask] = (stage_ids + 1) % num_stages
            
            # Return simple counterfactuals
            return counterfactual_batch
        
        # Enhanced counterfactual generation
        # Step 1: Define a probabilistic model p(x'|x) for each feature
        
        # Step 2: Apply feature-specific transformations that maintain realism
        batch_size = next(iter([v for v in counterfactual_batch.values() if isinstance(v, torch.Tensor)])).size(0)
        
        # For sector changes, use conditional probabilities instead of simple shifts
        if 'sector' in counterfactual_batch and 'sector_change' in self.config.counterfactual_types:
            # In a real implementation, we would have a learned transition matrix
            # Here we simulate one with some domain knowledge
            
            # Example sector transition matrix (probability of transitioning from sector i to j)
            # This would ideally be learned from data or defined with domain expertise
            if hasattr(self, 'sector_transition_matrix'):
                transition_matrix = self.sector_transition_matrix
            else:
                # Create a mock transition matrix
                num_sectors = int(counterfactual_batch['sector'].max().item()) + 1
                # Initialize with small random values
                transition_matrix = torch.rand(num_sectors, num_sectors) * 0.1
                # Increase probability of realistic transitions
                for i in range(num_sectors):
                    # Higher probability of staying in the same sector
                    transition_matrix[i, i] = 0.0  # We want to change sectors
                    # Higher probability for related sectors (assuming sectors near each other are related)
                    for j in range(max(0, i-2), min(num_sectors, i+3)):
                        if i != j:
                            transition_matrix[i, j] += 0.2
                # Normalize rows to sum to 1
                transition_matrix = transition_matrix / transition_matrix.sum(dim=1, keepdim=True)
                # Cache for future use
                self.sector_transition_matrix = transition_matrix.to(self.device)
            
            # Apply sector transitions
            mask = torch.rand(batch_size) < self.config.counterfactual_probability
            if mask.sum() > 0:
                current_sectors = counterfactual_batch['sector'][mask].to(torch.long)
                
                # For each example, sample new sector from transition distribution
                new_sectors = current_sectors.clone()
                for i in range(current_sectors.size(0)):
                    sector_probs = self.sector_transition_matrix[current_sectors[i]]
                    new_sectors[i] = torch.multinomial(sector_probs, 1).item()
                
                counterfactual_batch['sector'][mask] = new_sectors
        
        # For stage changes, maintain realistic progression
        if 'stage' in counterfactual_batch and 'stage_change' in self.config.counterfactual_types:
            mask = torch.rand(batch_size) < self.config.counterfactual_probability
            if mask.sum() > 0:
                current_stages = counterfactual_batch['stage'][mask]
                
                # Define more realistic stage transitions
                # For simplicity: 40% chance to go one stage up, 40% one stage down, 20% two stages difference
                stage_change = torch.zeros_like(current_stages)
                
                # Generate random change direction with appropriate probabilities
                rand_val = torch.rand(current_stages.size())
                stage_change[rand_val < 0.4] = 1    # 40% chance to increase stage
                stage_change[rand_val >= 0.4] = -1  # 40% chance to decrease stage
                stage_change[rand_val >= 0.8] = 2 * torch.sign(torch.randn(torch.sum(rand_val >= 0.8).item()))  # 20% chance +/-2
                
                # Apply changes while ensuring stages remain valid
                num_stages = int(counterfactual_batch['stage'].max().item()) + 1
                new_stages = (current_stages + stage_change).clamp(0, num_stages - 1)
                
                # Ensure stage actually changed
                unchanged = (new_stages == current_stages)
                if unchanged.sum() > 0:
                    # Force a stage change by +1 or -1
                    direction = torch.sign(torch.randn(unchanged.sum().item())).to(current_stages.device)
                    forced_change = torch.ones_like(new_stages[unchanged]) * direction
                    # Ensure valid stages
                    new_stages_tmp = (current_stages[unchanged] + forced_change).clamp(0, num_stages - 1)
                    # If still unchanged (can happen at boundaries), go the other direction
                    still_unchanged = (new_stages_tmp == current_stages[unchanged])
                    if still_unchanged.sum() > 0:
                        new_stages_tmp[still_unchanged] = (current_stages[unchanged][still_unchanged] - forced_change[still_unchanged]).clamp(0, num_stages - 1)
                    new_stages[unchanged] = new_stages_tmp
                
                counterfactual_batch['stage'][mask] = new_stages
        
        # For numerical features (like traction metrics), apply realistic transformations
        for field in [f for f in counterfactual_batch.keys() if f.startswith('traction_') or f.startswith('metric_')]:
            if isinstance(counterfactual_batch[field], torch.Tensor) and counterfactual_batch[field].dtype.is_floating_point:
                mask = torch.rand(batch_size) < self.config.counterfactual_probability
                if mask.sum() > 0:
                    # Get current values
                    current_values = counterfactual_batch[field][mask]
                    
                    # Apply multiplicative noise based on the field
                    # For revenue/traction metrics that follow power laws, use log-normal noise
                    log_factor = torch.randn(current_values.size()) * 0.5  # σ = 0.5 for moderate changes
                    multiplier = torch.exp(log_factor).to(current_values.device)
                    
                    # Apply transformation
                    new_values = current_values * multiplier
                    
                    # Ensure values stay realistic (non-negative for most metrics)
                    new_values = torch.max(new_values, torch.zeros_like(new_values))
                    
                    counterfactual_batch[field][mask] = new_values
        
        # For text descriptions (if represented as token IDs), we might make semantically related changes
        # This would typically require more sophisticated NLP approaches
        if 'input_ids' in counterfactual_batch and 'description_mask' in counterfactual_batch:
            # In a real implementation, this would involve using language models or embeddings
            # For simulation, we can swap some tokens or phrases
            pass  # Requires more sophisticated NLP functions
        
        # Step 3: Validate counterfactuals with loss constraint if possible
        # In a real implementation, we would compute the loss on these counterfactuals
        # and ensure they don't deviate too much from original examples
        if 'labels' in batch and hasattr(self.config, 'counterfactual_constraint'):
            # Track counterfactual constraints (from EnhancedSTTTCycle)
            if hasattr(self, 'vae_total_counterfactuals'):
                self.vae_total_counterfactuals += 1
                
                # Check if the counterfactual violates constraints
                try:
                    with torch.no_grad():
                        # Forward pass for original batch
                        orig_outputs = self.model(**{k: v for k, v in batch.items() if k != 'labels'})
                        orig_logits = orig_outputs.logits
                        orig_loss = torch.nn.functional.cross_entropy(orig_logits, batch['labels'])
                        
                        # Forward pass for counterfactual batch
                        cf_outputs = self.model(**{k: v for k, v in counterfactual_batch.items() if k != 'labels'})
                        cf_logits = cf_outputs.logits
                        cf_loss = torch.nn.functional.cross_entropy(cf_logits, batch['labels'])
                        
                        # Calculate relative change
                        rel_change = abs(cf_loss.item() - orig_loss.item()) / max(0.001, orig_loss.item())
                        
                        # Get constraint threshold
                        constraint = getattr(self.config, 'counterfactual_constraint', 0.2)
                        
                        # Check if constraint is violated
                        if rel_change > constraint:
                            self.vae_constraint_violations += 1
                            self.vae_constraint_logs.append(1)
                            
                            # Partially revert the counterfactual to reduce the loss difference
                            if rel_change > constraint * 2:  # If severely violated
                                # Interpolate between original and counterfactual
                                alpha = constraint / rel_change
                                for k, v in counterfactual_batch.items():
                                    if isinstance(v, torch.Tensor) and k in batch:
                                        counterfactual_batch[k] = alpha * v + (1 - alpha) * batch[k]
                        else:
                            self.vae_constraint_logs.append(0)
                except Exception as e:
                    logger.warning(f"Error checking counterfactual constraint: {e}")
        
        return counterfactual_batch
    
    def _generate_adversarial(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Generate adversarial examples (T3 phase) using gradient-based methods.
        
        Args:
            batch: Input batch
            
        Returns:
            Batch with adversarial examples
        """
        # Clone the batch to avoid modifying the original
        adversarial_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Use the dynamically adjusted adversarial severity
        severity = self.current_adversarial_severity
        
        # Use PGD adversarial attacks if enabled
        if self.config.use_pgd_adversarial:
            try:
                # Apply PGD attack
                adversarial_batch = self._pgd_adversarial(batch)
                
                if self.config.verbose and hasattr(self, 'global_step') and self.global_step % 50 == 0:
                    logger.info(f"Generated PGD adversarial examples with severity {severity:.2f}")
                
                return adversarial_batch
                
            except Exception as e:
                logger.warning(f"PGD adversarial generation failed: {str(e)}. Falling back to FGSM or heuristic method.")
                # Continue to FGSM or heuristic methods
        
        # If not using PGD or PGD failed, fall back to previous methods
        if not self.config.use_gradient_adversarial:
            # Use the original heuristic approach
            # Apply adversarial transformations based on config
            if 'missing_fields' in self.config.adversarial_types:
                # Randomly mask out fields
                for field in adversarial_batch:
                    if isinstance(adversarial_batch[field], torch.Tensor) and field != 'labels':
                        batch_size = adversarial_batch[field].size(0)
                        mask = torch.rand(batch_size) < severity
                        
                        if mask.sum() > 0 and field.endswith('_mask'):  # Assuming attention masks
                            # For attention masks, we can zero out random positions
                            seq_len = adversarial_batch[field].size(1)
                            for i in range(batch_size):
                                if mask[i]:
                                    # Zero out 10-30% of the positions, scaled by severity
                                    zero_positions = torch.rand(seq_len) < (0.1 + 0.2 * severity)
                                    adversarial_batch[field][i, zero_positions] = 0
                                    
            if 'noisy_descriptions' in self.config.adversarial_types and 'input_ids' in adversarial_batch:
                # Add noise to input tokens
                batch_size = adversarial_batch['input_ids'].size(0)
                seq_len = adversarial_batch['input_ids'].size(1)
                
                # Randomly swap tokens for 5-15% of positions, scaled by severity
                noise_prob = 0.05 + 0.1 * severity
                
                for i in range(batch_size):
                    noise_mask = torch.rand(seq_len) < noise_prob
                    if noise_mask.sum() > 0:
                        # Shuffle the masked positions
                        noisy_positions = torch.nonzero(noise_mask).squeeze()
                        shuffled_indices = torch.randperm(noisy_positions.size(0))
                        shuffled_positions = noisy_positions[shuffled_indices]
                        
                        # Swap tokens
                        temp = adversarial_batch['input_ids'][i, noisy_positions].clone()
                        adversarial_batch['input_ids'][i, noisy_positions] = adversarial_batch['input_ids'][i, shuffled_positions]
                        adversarial_batch['input_ids'][i, shuffled_positions] = temp
            
            if 'conflicting_signals' in self.config.adversarial_types:
                # Add conflicting signals
                if 'sector' in adversarial_batch and 'sector_description' in adversarial_batch:
                    batch_size = adversarial_batch['sector'].size(0)
                    conflict_mask = torch.rand(batch_size) < (severity * 0.7)
                    
                    if conflict_mask.sum() > 0:
                        # Mismatch sectors and descriptions
                        sector_ids = adversarial_batch['sector'][conflict_mask]
                        num_sectors = sector_ids.max() + 1
                        
                        # Assign a different sector (ensuring it's different)
                        new_sectors = (sector_ids + 1 + torch.randint(0, num_sectors-1, size=sector_ids.size())) % num_sectors
                        adversarial_batch['sector'][conflict_mask] = new_sectors
            
            return adversarial_batch
        
        # Gradient-based adversarial example generation (FGSM)
        # First, we need to identify which inputs we want to perturb
        # We'll focus on continuous features since directly perturbing discrete 
        # features like token IDs can be problematic
        
        # Get epsilon based on severity
        epsilon = self.config.fgsm_epsilon_base + severity * self.config.fgsm_epsilon_scale
        
        # Save original requires_grad state
        grad_state = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                grad_state[k] = v.requires_grad
                # For features we want to perturb, enable gradient tracking
                if v.dtype.is_floating_point and k not in ['labels', 'position_ids', 'attention_mask']:
                    batch[k].requires_grad_(True)
        
        # Forward pass with gradient tracking
        try:
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Backward pass to get gradients
            loss.backward()
            
            # Apply FGSM perturbation: x_adv = x + ε * sign(∇_x L(x, y; θ))
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and v.requires_grad and v.grad is not None:
                    # Get sign of gradient
                    grad_sign = v.grad.sign()
                    
                    # Apply perturbation
                    perturbation = epsilon * grad_sign
                    adversarial_batch[k] = (v + perturbation).detach()
                    
                    # Apply constraints to keep inputs in valid ranges
                    if k.startswith('traction_') or k.startswith('metric_'):
                        # Ensure metrics stay non-negative
                        adversarial_batch[k] = torch.max(adversarial_batch[k], torch.zeros_like(adversarial_batch[k]))
                    
                    # For embeddings or other normalized features, we might want to renormalize
                    if k.endswith('_embeddings') and v.norm(dim=-1).mean() > 0:
                        # Renormalize to preserve embedding norms
                        orig_norms = v.norm(dim=-1, keepdim=True)
                        adv_norms = adversarial_batch[k].norm(dim=-1, keepdim=True)
                        adversarial_batch[k] = adversarial_batch[k] * (orig_norms / adv_norms.clamp(min=1e-8))
            
            # Clean up by zeroing gradients
            self.model.zero_grad()
            
        except Exception as e:
            # If gradient-based method fails, fall back to heuristic method
            if self.config.verbose:
                logger.warning(f"Gradient-based adversarial generation failed: {str(e)}. Falling back to heuristic method.")
            
            # Fall back to the original heuristic approach
            # We won't repeat the code here since it would be wasteful
            # Instead, we'll recursively call this method with use_gradient_adversarial=False
            old_setting = self.config.use_gradient_adversarial
            self.config.use_gradient_adversarial = False
            adversarial_batch = self._generate_adversarial(batch)
            self.config.use_gradient_adversarial = old_setting
        
        finally:
            # Restore original requires_grad state
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and k in grad_state:
                    v.requires_grad_(grad_state[k])
        
        # Additional adversarial transformations that are not gradient-based
        # For discrete features like token IDs or categorical features,
        # we still need to use heuristic approaches
        
        batch_size = next(iter([v for v in adversarial_batch.values() if isinstance(v, torch.Tensor)])).size(0)
        
        # Token masking for text inputs (this can't easily be done with gradients)
        if 'input_ids' in adversarial_batch and 'attention_mask' in adversarial_batch:
            # Randomly mask tokens
            mask_prob = 0.05 * severity
            seq_len = adversarial_batch['input_ids'].size(1)
            
            for i in range(batch_size):
                mask = torch.rand(seq_len) < mask_prob
                if mask.sum() > 0:
                    # Set attention to 0 for these positions
                    adversarial_batch['attention_mask'][i, mask] = 0
        
        # For categorical features, apply targeted adversarial changes
        if 'sector' in adversarial_batch and severity > 0.5:
            # For high severity, introduce categorical errors in a small percentage of examples
            error_prob = (severity - 0.5) * 0.4  # Max 20% chance at max severity
            error_mask = torch.rand(batch_size) < error_prob
            
            if error_mask.sum() > 0:
                # Change to the least likely sector (assuming we have sector_probs)
                if hasattr(self, 'sector_probs'):
                    sector_probs = self.sector_probs
                    min_prob_sectors = sector_probs.argmin(dim=1)
                    adversarial_batch['sector'][error_mask] = min_prob_sectors[error_mask]
                else:
                    # Fallback: just pick a random sector
                    num_sectors = int(adversarial_batch['sector'].max().item()) + 1
                    random_sectors = torch.randint(0, num_sectors, (error_mask.sum(),)).to(adversarial_batch['sector'].device)
                    adversarial_batch['sector'][error_mask] = random_sectors
        
        return adversarial_batch
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor], phase: str) -> torch.Tensor:
        """
        Compute loss for the given batch.
        
        Args:
            batch: Input batch
            phase: One of {"S", "T1", "T2", "T3"}
            
        Returns:
            Loss tensor
        """
        # Validate batch contents and tensor dimensions
        required_keys = set()
        if hasattr(self.model, 'expected_inputs'):
            required_keys = set(self.model.expected_inputs)
        elif hasattr(self.model, 'forward'):
            # Try to infer required keys from forward method signature
            import inspect
            sig = inspect.signature(self.model.forward)
            required_keys = {p.name for p in sig.parameters.values() 
                            if p.default is inspect.Parameter.empty and p.name != 'self'}
        
        # Validate required keys are present
        if required_keys:
            missing_keys = required_keys - set(batch.keys())
            if missing_keys:
                raise ValueError(f"Missing required keys in batch for phase {phase}: {missing_keys}")
        
        # Validate tensor dimensions
        batch_size = None
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0:
                if batch_size is None:
                    batch_size = v.size(0)
                elif v.size(0) != batch_size:
                    raise ValueError(f"Inconsistent batch sizes in {phase} batch: {k} has size {v.size(0)}, expected {batch_size}")
        
        # Forward pass with model
        outputs = self.model(**{k: v for k, v in batch.items() if k != 'loss_weight'})
        
        # Get loss
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            loss = outputs.loss
        elif 'labels' in batch and hasattr(outputs, 'logits'):
            # Use cross entropy if we have labels and logits
            loss = torch.nn.functional.cross_entropy(outputs.logits, batch['labels'])
        else:
            # Fallback - assume the model returns the loss directly
            loss = outputs if isinstance(outputs, torch.Tensor) else torch.tensor(0.0).to(self.device)
            
        # Check if loss is a valid tensor with gradient
        if not isinstance(loss, torch.Tensor):
            raise TypeError(f"Loss from phase {phase} is not a tensor: {type(loss)}")
            
        if not loss.requires_grad:
            raise ValueError(f"Loss from phase {phase} doesn't require gradient")
            
        # Apply any phase-specific loss scaling/weighting
        if 'loss_weight' in batch:
            loss = loss * batch['loss_weight']
            
        # Apply curriculum difficulty as loss scaling
        if phase == "S" and self.current_curriculum_difficulty > 0:
            # Scale loss based on difficulty (higher difficulty = higher gradient scaling)
            difficulty_scale = 1.0 + self.current_curriculum_difficulty
            loss = loss * difficulty_scale
        
        # Apply T2-specific Bayesian boosting (from enhancement)
        if phase == "T2" and hasattr(self.config, 'bayesian_boost') and self.latent_regularizer is not None:
            if hasattr(self.latent_regularizer, 'config'):
                orig_weight = self.latent_regularizer.config.bayesian_weight
                # Boost bayesian weight for T2 phase
                bayesian_boost = getattr(self.config, 'bayesian_boost', 2.0)
                self.latent_regularizer.config.bayesian_weight *= bayesian_boost
                logger.info(f"Boosting Bayesian weight for T2: {orig_weight} → {self.latent_regularizer.config.bayesian_weight}")
            
        # Apply regularization if enabled
        if self.config.use_latent_regularization and hasattr(self, 'latent_regularizer'):
            reg_weight = 0.0
            
            if phase == "S":
                reg_weight = self.reg_weights.get('study', 0.0)
            elif phase == "T1":
                reg_weight = self.reg_weights.get('t1', 0.0)
            elif phase == "T2":
                reg_weight = self.reg_weights.get('t2', 0.0)
            elif phase == "T3":
                reg_weight = self.reg_weights.get('t3', 0.0)
                
            if reg_weight > 0:
                # Apply regularization if weight is positive
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    # Use hidden states from model output if available
                    reg_loss = self.latent_regularizer(outputs.hidden_states)
                    loss = loss + reg_weight * reg_loss
                else:
                    # Otherwise extract representations from model manually if possible
                    try:
                        reg_loss = self.latent_regularizer.get_model_regularization(
                            self.model, batch, embedding_key='last_hidden_state'
                        )
                        loss = loss + reg_weight * reg_loss
                    except Exception as e:
                        if self.config.verbose:
                            logging.warning(f"Failed to apply regularization: {str(e)}")
        
        # Log VAE counterfactual constraint violations (from enhancement)
        if phase == "T2" and hasattr(self, 'vae_constraint_violations') and self.vae_total_counterfactuals > 0:
            violation_rate = (self.vae_constraint_violations / max(1, self.vae_total_counterfactuals)) * 100
            logger.info(f"VAE constraint violations: {violation_rate:.2f}% ({self.vae_constraint_violations}/{self.vae_total_counterfactuals})")
                            
        # Verify loss is finite
        if not torch.isfinite(loss).all():
            logging.warning(f"Non-finite loss detected in phase {phase}. Using substitute loss.")
            # Replace with a small positive value to avoid breaking the training
            loss = torch.tensor(1.0, device=self.device, requires_grad=True)
                            
        return loss
    
    def _update_baselines(self):
        """Update baseline losses for detecting surges."""
        # Use mean of recent losses as baseline
        if len(self.t1_loss_history) > 0:
            self.t1_baseline_loss = sum(self.t1_loss_history) / len(self.t1_loss_history)
        
        if len(self.t2_loss_history) > 0:
            self.t2_baseline_loss = sum(self.t2_loss_history) / len(self.t2_loss_history)
        
        if len(self.t3_loss_history) > 0:
            self.t3_baseline_loss = sum(self.t3_loss_history) / len(self.t3_loss_history)
    
    def _check_for_intervention(self, current_losses: Dict[str, float]) -> Tuple[bool, str]:
        """
        Check if we need to intervene based on test losses with adaptive thresholds.
        """
        if self.t1_baseline_loss is None or self.t2_baseline_loss is None or self.t3_baseline_loss is None:
            return False, ""
        
        # Calculate volatilities
        volatility = self._calculate_baseline_volatility()
        mean_volatility = sum(volatility.values()) / len(volatility) if volatility else 0.1
        
        # Adjust thresholds based on volatility
        adaptive_t1_threshold = self.config.t1_loss_threshold * (1 + 0.1 * (volatility.get("T1", 0) / mean_volatility))
        adaptive_t2_threshold = self.config.t2_loss_threshold * (1 + 0.1 * (volatility.get("T2", 0) / mean_volatility))
        adaptive_t3_threshold = self.config.t3_loss_threshold * (1 + 0.1 * (volatility.get("T3", 0) / mean_volatility))
        
        # Check for surges with adaptive thresholds
        t1_surge = current_losses.get("T1", 0) > self.t1_baseline_loss * (1 + adaptive_t1_threshold)
        t2_surge = current_losses.get("T2", 0) > self.t2_baseline_loss * (1 + adaptive_t2_threshold)
        t3_surge = current_losses.get("T3", 0) > self.t3_baseline_loss * (1 + adaptive_t3_threshold)
        
        # Weight the surges according to config
        weighted_surge = (
            t1_surge * self.config.t1_weight +
            t2_surge * self.config.t2_weight +
            t3_surge * self.config.t3_weight
        )
        
        # Determine if intervention is needed (more than 50% of weighted factors)
        need_intervention = weighted_surge > 0.5
        
        # Determine the reason
        reason = ""
        if need_intervention:
            reasons = []
            if t1_surge:
                reasons.append(f"T1 loss surge: {current_losses.get('T1', 0):.4f} > {self.t1_baseline_loss:.4f} * {1 + adaptive_t1_threshold}")
            if t2_surge:
                reasons.append(f"T2 loss surge: {current_losses.get('T2', 0):.4f} > {self.t2_baseline_loss:.4f} * {1 + adaptive_t2_threshold}")
            if t3_surge:
                reasons.append(f"T3 loss surge: {current_losses.get('T3', 0):.4f} > {self.t3_baseline_loss:.4f} * {1 + adaptive_t3_threshold}")
            
            reason = ", ".join(reasons)
        
        return need_intervention, reason
    
    def _apply_intervention(self, reason: str) -> None:
        """
        Apply intervention when test losses surge, using Bayesian optimization
        for uncertainty-aware parameter selection.
        
        Args:
            reason: Reason for intervention
        """
        # Record the intervention
        intervention = {
            "step": self.global_step,
            "reason": reason,
            "study_loss": np.mean(list(self.study_loss_history)[-10:]) if self.study_loss_history else None,
            "t1_loss": np.mean(list(self.t1_loss_history)[-10:]) if self.t1_loss_history else None,
            "t2_loss": np.mean(list(self.t2_loss_history)[-10:]) if self.t2_loss_history else None,
            "t3_loss": np.mean(list(self.t3_loss_history)[-10:]) if self.t3_loss_history else None
        }
        self.interventions.append(intervention)
        if hasattr(self, 'metrics') and "intervention_steps" in self.metrics:
            self.metrics["intervention_steps"].append(self.global_step)
        
        logger.info(f"Intervention at step {self.global_step}: {reason}")
        
        # Calculate surge score to determine intervention strength
        # Higher surge = stronger intervention
        surge_score = 0.0
        surge_count = 0
        
        if self.t1_baseline_loss is not None and "T1" in reason:
            t1_loss = np.mean(list(self.t1_loss_history)[-5:])
            t1_surge = (t1_loss / self.t1_baseline_loss) - 1.0
            surge_score += t1_surge * self.config.t1_weight
            surge_count += 1
        
        if self.t2_baseline_loss is not None and "T2" in reason:
            t2_loss = np.mean(list(self.t2_loss_history)[-5:])
            t2_surge = (t2_loss / self.t2_baseline_loss) - 1.0
            surge_score += t2_surge * self.config.t2_weight
            surge_count += 1
        
        if self.t3_baseline_loss is not None and "T3" in reason:
            t3_loss = np.mean(list(self.t3_loss_history)[-5:])
            t3_surge = (t3_loss / self.t3_baseline_loss) - 1.0
            surge_score += t3_surge * self.config.t3_weight
            surge_count += 1
        
        if surge_count > 0:
            surge_score /= surge_count
        
        # Use Uncertainty-Aware Bayesian Optimization if enabled
        if self.config.use_uncertainty_interventions:
            try:
                # Apply Bayesian optimization for intervention parameters
                self._apply_bayesian_intervention(surge_score, reason)
                
                # After Bayesian intervention, adjust latent space regularization if enabled
                if self.config.use_latent_regularization and self.config.adapt_reg_on_intervention:
                    self._adjust_latent_regularization(surge_score, reason)
                    
                return
            except Exception as e:
                logger.warning(f"Bayesian intervention failed: {str(e)}. Falling back to standard intervention.")
                # Continue with standard intervention below
        
        # Apply concrete intervention strategies (standard approach)
        
        # 1. Learning Rate Adjustment: Reduce learning rate when test losses spike
        current_lr = self.optimizer.param_groups[0]['lr']
        alpha = self.config.intervention_lr_alpha
        
        # Decrease learning rate based on surge score: η' = η * exp(-α * S)
        new_lr = current_lr * math.exp(-alpha * surge_score)
        
        # Apply new learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        logger.info(f"Adjusted learning rate: {current_lr:.6f} -> {new_lr:.6f}")
        
        # 2. Example Reweighting: Identify and collect failed examples
        # In a real implementation, we would identify specific examples that caused the surge
        # and increase their sampling probability in future training
        
        # For this simulation, we'll store the current batch if we're in a test phase
        # and use it to augment future study phases
        if hasattr(self, 'current_phase') and self.current_phase in ["T1", "T2", "T3"]:
            try:
                # Get the current batch (assuming it contributed to the surge)
                # In practice, we'd be more selective about which examples to store
                batch = self._get_next_batch(self.current_phase)
                
                # Add to failed examples buffer
                if hasattr(self, 'failed_examples_buffer'):
                    self.failed_examples_buffer.append({
                        'batch': batch,
                        'phase': self.current_phase,
                        'weight': math.exp(self.config.intervention_example_reweight_lambda)
                    })
                
                logger.info(f"Added examples from {self.current_phase} to replay buffer")
            except Exception as e:
                logger.warning(f"Failed to store examples for reweighting: {str(e)}")
        
        # 3. Regularization Boost: Temporarily increase regularization
        # For simplicity, we'll assume the model supports dropout_probability and weight_decay
        # In a real implementation, this would depend on the model architecture
        
        # Set regularization boost countdown
        self.intervention_reg_boost_steps_left = self.config.intervention_reg_reset_steps
        
        # Try to increase dropout if the model supports it
        try:
            # This is a proxy - in reality, you'd need to know the model's structure 
            # to modify dropout appropriately
            if hasattr(self.model, 'dropout'):
                current_dropout = self.model.dropout.p
                new_dropout = min(0.9, current_dropout + self.config.intervention_reg_boost_beta * surge_score)
                self.model.dropout.p = new_dropout
                logger.info(f"Increased dropout probability: {current_dropout:.2f} -> {new_dropout:.2f}")
            
            # If model has multiple dropouts
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Dropout):
                    current_p = module.p
                    new_p = min(0.9, current_p + self.config.intervention_reg_boost_beta * surge_score)
                    module.p = new_p
        except Exception as e:
            logger.warning(f"Failed to adjust dropout: {str(e)}")
        
        # Try to increase weight decay
        try:
            for param_group in self.optimizer.param_groups:
                if 'weight_decay' in param_group:
                    current_wd = param_group['weight_decay']
                    new_wd = current_wd + self.config.intervention_reg_boost_beta * surge_score
                    param_group['weight_decay'] = new_wd
                    logger.info(f"Increased weight decay: {current_wd:.5f} -> {new_wd:.5f}")
        except Exception as e:
            logger.warning(f"Failed to adjust weight decay: {str(e)}")
        
        # Additional intervention: data augmentation/diversity boost
        # In the next few study steps, increase the likelihood of using diverse examples
        # This is implemented through the failed examples buffer being used in _get_next_batch
        
        # Adjust latent space regularization if enabled
        if self.config.use_latent_regularization and self.config.adapt_reg_on_intervention:
            self._adjust_latent_regularization(surge_score, reason)
    
    def _adjust_latent_regularization(self, surge_score: float, reason: str) -> None:
        """
        Adjust latent space regularization weights based on surge score and reason.
        
        Args:
            surge_score: Score indicating how severe the performance loss was
            reason: Reason for the intervention
        """
        if not self.config.use_latent_regularization or self.latent_regularizer is None:
            return
            
        # Determine which phases to boost based on reason
        phases_to_boost = []
        if "T1" in reason:
            phases_to_boost.append("t1")
        if "T2" in reason:
            phases_to_boost.append("t2")
        if "T3" in reason:
            phases_to_boost.append("t3")
        
        # Always include study phase
        phases_to_boost.append("study")
        
        # Apply boost scaled by surge score
        boost_amount = self.config.intervention_reg_boost * (1.0 + surge_score)
        
        for phase in phases_to_boost:
            self.reg_weights[phase] += boost_amount
            
        # Log the adjustment
        if self.verbose:
            logging.info(f"Adjusted latent regularization weights due to intervention: {reason}")
            logging.info(f"New weights: {self.reg_weights}")
            
        # Optionally adjust the regularizer's internal parameters
        try:
            # Increase L1 penalty for more sparsity
            if hasattr(self.latent_regularizer, 'config'):
                # Make a temporary reference for shorter code
                reg_config = self.latent_regularizer.config
                
                # Boost L1 penalty
                if hasattr(reg_config, 'l1_penalty_weight'):
                    reg_config.l1_penalty_weight *= (1.0 + 0.5 * surge_score)
                    
                # Boost orthogonal penalty
                if hasattr(reg_config, 'orthogonal_penalty_weight'):
                    reg_config.orthogonal_penalty_weight *= (1.0 + 0.5 * surge_score)
                    
                logging.info("Adjusted internal regularization parameters")
        except Exception as e:
            logging.warning(f"Failed to adjust regularizer's internal parameters: {str(e)}")
    
    def _apply_bayesian_intervention(self, surge_score: float, reason: str) -> None:
        """
        Apply uncertainty-aware intervention using Bayesian optimization.
        
        This uses a Gaussian Process (GP) to model the expected loss as a function 
        of intervention parameters, and selects optimal parameters under uncertainty:
        
        η' = argmax_η E_p(L|η)[-L]
        
        Args:
            surge_score: Severity of the loss surge
            reason: Reason for intervention
        """
        # Initialize the Bayesian optimization system if not already done
        if not hasattr(self, 'intervention_gp'):
            self._initialize_bayesian_intervention_system()
        
        # Get current parameter values to use as baseline
        current_params = self._get_current_intervention_params()
        
        # Add noise to parameters based on config to explore parameter space
        param_noise = np.random.normal(0, 0.1, len(current_params))
        
        # Create parameter samples
        # We generate multiple candidates, evaluate them with the GP, and select the best
        n_samples = self.config.uncertainty_samples
        param_samples = []
        
        # Generate candidate parameter sets by exploring around current values
        # with some random perturbations for exploration
        for i in range(n_samples):
            # Different noise level for each sample to explore parameter space
            noise_scale = 0.05 + (i / n_samples) * 0.2  # Gradually increase exploration
            sample_noise = np.random.normal(0, noise_scale, len(current_params))
            
            # Create sample with noise
            param_sample = current_params.copy()
            
            # Apply noise to each parameter, respecting their bounds
            for j, param_name in enumerate(self.config.intervention_params):
                if param_name == 'learning_rate':
                    # Log scale noise for learning rate (multiplicative)
                    factor = math.exp(sample_noise[j])
                    param_sample[j] *= factor
                    # Bound learning rate
                    param_sample[j] = max(1e-6, min(1e-2, param_sample[j]))
                elif param_name == 'dropout':
                    # Additive noise for dropout, bounded between 0.1 and 0.9
                    param_sample[j] += sample_noise[j] * 0.2  # Scale noise
                    param_sample[j] = max(0.1, min(0.9, param_sample[j]))
                elif param_name == 'weight_decay':
                    # Log scale for weight decay
                    factor = math.exp(sample_noise[j])
                    param_sample[j] *= factor
                    # Bound weight decay
                    param_sample[j] = max(1e-6, min(0.1, param_sample[j]))
                elif param_name == 'reg_weight':
                    # Additive noise for regularization weight
                    param_sample[j] += sample_noise[j] * 0.05
                    param_sample[j] = max(0.01, min(0.2, param_sample[j]))
                elif param_name == 'curriculum_difficulty':
                    # Additive noise for curriculum difficulty
                    param_sample[j] += sample_noise[j] * 0.1
                    param_sample[j] = max(0.1, min(1.0, param_sample[j]))
            
            param_samples.append(param_sample)
        
        # If we have sufficient history, use GP to predict expected loss for each sample
        if len(self.intervention_history) >= 3:
            # Convert history to training data for GP
            X_train = np.array([entry['params'] for entry in self.intervention_history])
            y_train = np.array([entry['post_loss'] for entry in self.intervention_history])
            
            # Normalize y values for numerical stability
            y_mean = y_train.mean()
            y_std = y_train.std() if y_train.std() > 0 else 1.0
            y_train_norm = (y_train - y_mean) / y_std
            
            # Update GP with observed data
            self.intervention_gp.fit(X_train, y_train_norm)
            
            # Evaluate each sample with GP
            X_test = np.array(param_samples)
            y_pred, y_std = self.intervention_gp.predict(X_test, return_std=True)
            
            # Denormalize predictions
            y_pred = y_pred * y_std + y_mean
            y_std = y_std * y_std  # Scale std dev back
            
            # Calculate acquisition function (Expected Improvement)
            # For minimum loss: EI = -μ + β*σ (exploration-exploitation tradeoff)
            beta = 1.0  # Exploration parameter
            acquisition_values = -y_pred + beta * y_std
            
            # Select parameter set with highest acquisition value
            best_idx = np.argmax(acquisition_values)
            best_params = param_samples[best_idx]
            
            if self.config.verbose:
                logger.info(f"Selected intervention params using Bayesian optimization, "
                           f"pred_loss={y_pred[best_idx]:.4f}, uncertainty={y_std[best_idx]:.4f}")
        else:
            # Not enough history for GP - use heuristic approach
            # Select based on surge score - larger perturbation for worse surges
            # Apply largest perturbation for the most severe surges
            sorted_by_perturbation = sorted(param_samples, key=lambda x: np.linalg.norm(x - current_params))
            perturbation_idx = min(int(surge_score * len(sorted_by_perturbation)), len(sorted_by_perturbation) - 1)
            best_params = sorted_by_perturbation[perturbation_idx]
            
            if self.config.verbose:
                logger.info(f"Selected intervention params using heuristic (limited history), "
                           f"perturbation_magnitude={np.linalg.norm(best_params - current_params):.4f}")
        
        # Apply the selected parameters
        self._apply_intervention_params(best_params)
        
        # Record this intervention for future GP training
        pre_intervention_loss = np.mean([
            self.study_loss_history[-1] if self.study_loss_history else 0,
            self.t1_loss_history[-1] if self.t1_loss_history else 0,
            self.t2_loss_history[-1] if self.t2_loss_history else 0,
            self.t3_loss_history[-1] if self.t3_loss_history else 0
        ])
        
        self.intervention_history.append({
            'step': self.global_step,
            'params': best_params,
            'surge_score': surge_score,
            'pre_loss': pre_intervention_loss,
            'post_loss': None  # Will be updated after observing next losses
        })
        
        # Update the post_loss of the previous intervention if available
        if len(self.intervention_history) > 1:
            prev_intervention = self.intervention_history[-2]
            if prev_intervention['post_loss'] is None:
                prev_intervention['post_loss'] = pre_intervention_loss
    
    def _initialize_bayesian_intervention_system(self):
        """Initialize the Gaussian Process for Bayesian intervention optimization."""
        # Initialize history of interventions for GP training
        self.intervention_history = []
        
        # Define the kernel for the GP
        # RBF kernel with some noise for stability
        kernel = (
            self.config.gp_kernel_scale * 
            (1.0 * np.ones(len(self.config.intervention_params)) ** 2)
        )
        
        # Create GP regressor
        # Simple numpy implementation - in practice you'd use GPyTorch or similar
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        
        # Create a combination of RBF kernel and white noise
        kernel = RBF(length_scale=kernel) + WhiteKernel(noise_level=self.config.gp_noise)
        
        # Create the GP regressor
        self.intervention_gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,  # Small regularization
            normalize_y=False,  # We'll normalize manually
            n_restarts_optimizer=2  # For kernel hyperparameter optimization
        )
        
        logger.info("Initialized Gaussian Process for uncertainty-aware interventions")
    
    def _get_current_intervention_params(self) -> np.ndarray:
        """Get current values of intervention parameters as array."""
        params = []
        
        for param_name in self.config.intervention_params:
            if param_name == 'learning_rate':
                params.append(self.optimizer.param_groups[0]['lr'])
            elif param_name == 'dropout':
                # Find average dropout in model
                dropout_vals = []
                for name, module in self.model.named_modules():
                    if isinstance(module, torch.nn.Dropout):
                        dropout_vals.append(module.p)
                # Use mean or default
                params.append(np.mean(dropout_vals) if dropout_vals else 0.1)
            elif param_name == 'weight_decay':
                wd = 0.0
                if 'weight_decay' in self.optimizer.param_groups[0]:
                    wd = self.optimizer.param_groups[0]['weight_decay']
                params.append(wd)
            elif param_name == 'reg_weight':
                # Regularization weight (e.g., adversarial reg)
                params.append(self.config.adversarial_reg_weight)
            elif param_name == 'curriculum_difficulty':
                # Current curriculum difficulty
                if hasattr(self, 'current_curriculum_difficulty'):
                    params.append(self.current_curriculum_difficulty)
                else:
                    params.append(0.5)  # Default mid-level difficulty
        
        return np.array(params)
    
    def _apply_intervention_params(self, params: np.ndarray) -> None:
        """Apply the selected intervention parameters."""
        for i, param_name in enumerate(self.config.intervention_params):
            if param_name == 'learning_rate':
                # Update learning rate in optimizer
                old_lr = self.optimizer.param_groups[0]['lr']
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = params[i]
                logger.info(f"Adjusted learning rate: {old_lr:.6f} -> {params[i]:.6f}")
            
            elif param_name == 'dropout':
                # Update dropout rates in model
                for name, module in self.model.named_modules():
                    if isinstance(module, torch.nn.Dropout):
                        old_p = module.p
                        module.p = params[i]
                        if old_p != params[i]:
                            logger.info(f"Adjusted dropout in {name}: {old_p:.2f} -> {params[i]:.2f}")
            
            elif param_name == 'weight_decay':
                # Update weight decay in optimizer
                for param_group in self.optimizer.param_groups:
                    if 'weight_decay' in param_group:
                        old_wd = param_group['weight_decay']
                        param_group['weight_decay'] = params[i]
                        logger.info(f"Adjusted weight decay: {old_wd:.5f} -> {params[i]:.5f}")
            
            elif param_name == 'reg_weight':
                # Update regularization weight in config
                old_reg = self.config.adversarial_reg_weight
                self.config.adversarial_reg_weight = params[i]
                logger.info(f"Adjusted adversarial regularization weight: {old_reg:.4f} -> {params[i]:.4f}")
            
            elif param_name == 'curriculum_difficulty':
                # Update curriculum difficulty
                if hasattr(self, 'current_curriculum_difficulty'):
                    old_diff = self.current_curriculum_difficulty
                    self.current_curriculum_difficulty = params[i]
                    logger.info(f"Adjusted curriculum difficulty: {old_diff:.2f} -> {params[i]:.2f}")
                else:
                    self.current_curriculum_difficulty = params[i]
                    logger.info(f"Set curriculum difficulty: {params[i]:.2f}")
        
        # Set regularization boost countdown
        self.intervention_reg_boost_steps_left = self.config.intervention_reg_reset_steps
    
    def _update_cycle_parameters(self):
        """Update STTT cycle parameters based on performance if dynamic adjustment is enabled."""
        if not self.config.dynamic_cycle_adjustment:
            return
        
        # Simple heuristic: If we've had interventions recently, increase test phases
        recent_steps = 100
        recent_interventions = [i for i in self.interventions if self.global_step - i["step"] < recent_steps]
        
        if len(recent_interventions) > 0:
            # Increase test phases, decrease study phase
            self.config.study_steps = max(60, self.config.study_steps - 5)
            self.config.t1_steps = min(20, self.config.t1_steps + 2)
            self.config.t2_steps = min(10, self.config.t2_steps + 2)
            self.config.t3_steps = min(10, self.config.t3_steps + 1)
            
            logger.info(f"Adjusted cycle parameters due to recent interventions: "
                        f"Study={self.config.study_steps}, "
                        f"T1={self.config.t1_steps}, "
                        f"T2={self.config.t2_steps}, "
                        f"T3={self.config.t3_steps}")
        else:
            # No recent interventions, gradually return to normal
            self.config.study_steps = min(80, self.config.study_steps + 2)
            self.config.t1_steps = max(10, self.config.t1_steps - 1)
            self.config.t2_steps = max(5, self.config.t2_steps - 1)
            self.config.t3_steps = max(5, self.config.t3_steps - 1)
    
    def _calculate_loss_derivatives(self) -> Dict[str, float]:
        """
        Calculate the rate of change (derivative) of test losses.
        Uses exponential moving averages for smoother estimates.
        
        Returns:
            Dictionary of loss derivatives for each test phase
        """
        derivatives = {}
        
        if not self.config.use_ema_derivatives:
            # Original approach - direct step-by-step derivatives
            # Calculate T1 derivative
            if self.prev_t1_loss is not None and len(self.t1_loss_history) > 0:
                current_t1 = self.t1_loss_history[-1]
                t1_derivative = current_t1 - self.prev_t1_loss
                self.t1_derivatives.append(t1_derivative)
                derivatives["T1"] = t1_derivative
            
            # Calculate T2 derivative
            if self.prev_t2_loss is not None and len(self.t2_loss_history) > 0:
                current_t2 = self.t2_loss_history[-1]
                t2_derivative = current_t2 - self.prev_t2_loss
                self.t2_derivatives.append(t2_derivative)
                derivatives["T2"] = t2_derivative
            
            # Calculate T3 derivative
            if self.prev_t3_loss is not None and len(self.t3_loss_history) > 0:
                current_t3 = self.t3_loss_history[-1]
                t3_derivative = current_t3 - self.prev_t3_loss
                self.t3_derivatives.append(t3_derivative)
                derivatives["T3"] = t3_derivative
            
            # Update previous losses for next calculation
            if len(self.t1_loss_history) > 0:
                self.prev_t1_loss = self.t1_loss_history[-1]
            if len(self.t2_loss_history) > 0:
                self.prev_t2_loss = self.t2_loss_history[-1]
            if len(self.t3_loss_history) > 0:
                self.prev_t3_loss = self.t3_loss_history[-1]
            
        else:
            # Enhanced approach - EMA for smoother derivative estimates
            alpha = self.config.ema_alpha  # Smoothing factor
            
            # Calculate T1 EMA and derivative
            if len(self.t1_loss_history) > 0:
                current_t1 = self.t1_loss_history[-1]
                
                # Update EMA
                if self.ema_t1_loss is None:
                    self.ema_t1_loss = current_t1
                else:
                    prev_ema_t1 = self.ema_t1_loss
                    self.ema_t1_loss = alpha * current_t1 + (1 - alpha) * self.ema_t1_loss
                    
                    # Calculate smoothed derivative
                    t1_derivative = self.ema_t1_loss - prev_ema_t1
                    self.ema_t1_derivatives.append(t1_derivative)
                    derivatives["T1"] = t1_derivative
            
            # Calculate T2 EMA and derivative
            if len(self.t2_loss_history) > 0:
                current_t2 = self.t2_loss_history[-1]
                
                # Update EMA
                if self.ema_t2_loss is None:
                    self.ema_t2_loss = current_t2
                else:
                    prev_ema_t2 = self.ema_t2_loss
                    self.ema_t2_loss = alpha * current_t2 + (1 - alpha) * self.ema_t2_loss
                    
                    # Calculate smoothed derivative
                    t2_derivative = self.ema_t2_loss - prev_ema_t2
                    self.ema_t2_derivatives.append(t2_derivative)
                    derivatives["T2"] = t2_derivative
            
            # Calculate T3 EMA and derivative
            if len(self.t3_loss_history) > 0:
                current_t3 = self.t3_loss_history[-1]
                
                # Update EMA
                if self.ema_t3_loss is None:
                    self.ema_t3_loss = current_t3
                else:
                    prev_ema_t3 = self.ema_t3_loss
                    self.ema_t3_loss = alpha * current_t3 + (1 - alpha) * self.ema_t3_loss
                    
                    # Calculate smoothed derivative
                    t3_derivative = self.ema_t3_loss - prev_ema_t3
                    self.ema_t3_derivatives.append(t3_derivative)
                    derivatives["T3"] = t3_derivative
            
            # Also update regular derivatives for comparison/backup
            if self.prev_t1_loss is not None and len(self.t1_loss_history) > 0:
                t1_derivative = self.t1_loss_history[-1] - self.prev_t1_loss
                self.t1_derivatives.append(t1_derivative)
            
            if self.prev_t2_loss is not None and len(self.t2_loss_history) > 0:
                t2_derivative = self.t2_loss_history[-1] - self.prev_t2_loss
                self.t2_derivatives.append(t2_derivative)
            
            if self.prev_t3_loss is not None and len(self.t3_loss_history) > 0:
                t3_derivative = self.t3_loss_history[-1] - self.prev_t3_loss
                self.t3_derivatives.append(t3_derivative)
            
            # Update previous raw losses for next calculation
            if len(self.t1_loss_history) > 0:
                self.prev_t1_loss = self.t1_loss_history[-1]
            if len(self.t2_loss_history) > 0:
                self.prev_t2_loss = self.t2_loss_history[-1]
            if len(self.t3_loss_history) > 0:
                self.prev_t3_loss = self.t3_loss_history[-1]
        
        return derivatives
    
    def _check_derivative_warnings(self, derivatives: Dict[str, float]) -> Tuple[bool, str]:
        """
        Check if loss derivatives exceed warning thresholds.
        
        Args:
            derivatives: Dictionary of loss derivatives
            
        Returns:
            Tuple of (has_warning, reason)
        """
        if not self.config.enable_loss_derivative_monitoring:
            return False, ""
        
        warnings = []
        for phase, derivative in derivatives.items():
            if derivative > self.config.derivative_warning_threshold:
                warnings.append(f"{phase} loss increasing rapidly (dL/dt={derivative:.4f})")
        
        has_warning = len(warnings) > 0
        reason = ", ".join(warnings)
        
        return has_warning, reason
    
    def _update_baseline_history(self):
        """Update history of baseline losses for volatility tracking."""
        if not self.config.track_baseline_volatility:
            return
        
        if self.t1_baseline_loss is not None:
            self.t1_baseline_history.append(self.t1_baseline_loss)
        
        if self.t2_baseline_loss is not None:
            self.t2_baseline_history.append(self.t2_baseline_loss)
        
        if self.t3_baseline_loss is not None:
            self.t3_baseline_history.append(self.t3_baseline_loss)
    
    def _calculate_baseline_volatility(self) -> Dict[str, float]:
        """
        Calculate volatility (standard deviation) of baseline losses.
        
        Returns:
            Dictionary of baseline volatilities for each test phase
        """
        volatility = {}
        
        # Calculate T1 baseline volatility
        if len(self.t1_baseline_history) > 3:  # Need enough samples for meaningful std
            t1_volatility = np.std(list(self.t1_baseline_history))
            volatility["T1"] = t1_volatility
        
        # Calculate T2 baseline volatility
        if len(self.t2_baseline_history) > 3:
            t2_volatility = np.std(list(self.t2_baseline_history))
            volatility["T2"] = t2_volatility
        
        # Calculate T3 baseline volatility
        if len(self.t3_baseline_history) > 3:
            t3_volatility = np.std(list(self.t3_baseline_history))
            volatility["T3"] = t3_volatility
        
        return volatility
    
    def _check_volatility_warnings(self, volatility: Dict[str, float]) -> Tuple[bool, str]:
        """
        Check if baseline volatility exceeds warning thresholds.
        
        Args:
            volatility: Dictionary of baseline volatilities
            
        Returns:
            Tuple of (has_warning, reason)
        """
        if not self.config.track_baseline_volatility:
            return False, ""
        
        warnings = []
        for phase, vol in volatility.items():
            if vol > self.config.volatility_threshold:
                warnings.append(f"{phase} baseline highly volatile (σ={vol:.4f})")
        
        has_warning = len(warnings) > 0
        reason = ", ".join(warnings)
        
        return has_warning, reason
    
    def _calculate_intervention_rate(self) -> float:
        """
        Calculate the recent intervention rate.
        
        Returns:
            Intervention rate (interventions per step)
        """
        lookback = self.config.intervention_lookback_window
        recent_steps = min(lookback, self.global_step)
        
        if recent_steps == 0:
            return 0.0
        
        # Count recent interventions
        recent_interventions = sum(
            1 for i in self.interventions 
            if self.global_step - i["step"] < recent_steps
        )
        
        return recent_interventions / recent_steps
    
    def _update_grad_accum_steps(self):
        if not self.config.dynamic_grad_accum:
            return
        
        # Original intervention-based adjustment
        intervention_rate = self._calculate_intervention_rate()
        
        # Add memory-based adjustment
        if torch.cuda.is_available():
            try:
                # Get memory stats
                mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                mem_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                mem_free = torch.cuda.get_device_properties(0).total_memory / 1024**3 - mem_reserved  # GB
                
                # Calculate optimal steps based on memory
                batch_mem_estimate = mem_allocated / self.current_grad_accum_steps  # Rough estimate
                memory_optimal_steps = max(1, int(mem_free / batch_mem_estimate))
                
                # Combine with intervention-based recommendation
                if intervention_rate > 0:
                    rate_factor = 0.05 / max(0.05, intervention_rate)
                    intervention_steps = int(rate_factor * self.config.min_grad_accum_steps)
                    # Take the minimum (both memory and intervention constraints should be satisfied)
                    new_steps = min(memory_optimal_steps, intervention_steps)
                else:
                    # No interventions, use memory-based steps
                    new_steps = memory_optimal_steps
                
                # Clamp to config limits
                new_steps = max(self.config.min_grad_accum_steps, 
                              min(self.config.max_grad_accum_steps, new_steps))
                
                if new_steps != self.current_grad_accum_steps:
                    self.current_grad_accum_steps = new_steps
                    if self.config.verbose:
                        logger.info(f"Adjusted gradient accumulation steps to {new_steps} "
                                   f"(memory: {mem_free:.2f}GB free, intervention rate: {intervention_rate:.4f})")
            except:
                # Fall back to intervention-based adjustment if memory query fails
                pass
    
    def _update_adversarial_severity(self):
        """Update adversarial severity based on intervention density."""
        if not self.config.dynamic_adversarial_severity:
            return
        
        intervention_rate = self._calculate_intervention_rate()
        
        # Direct relationship: higher intervention rate -> higher severity
        # Scale between min and max based on intervention rate
        severity_range = self.config.max_adversarial_severity - self.config.min_adversarial_severity
        
        # Scale intervention rate (typically small) to full severity range
        # Max rate of 0.2 (intervention every 5 steps) would give max severity
        new_severity = self.config.min_adversarial_severity + (
            min(0.2, intervention_rate) / 0.2
        ) * severity_range
        
        # Update current severity
        if abs(new_severity - self.current_adversarial_severity) > 0.05:  # Only update on significant changes
            self.current_adversarial_severity = new_severity
            if self.config.verbose:
                logger.info(f"Adjusted adversarial severity to {new_severity:.4f} "
                           f"(intervention rate: {intervention_rate:.4f})")
    
    def _update_curriculum_difficulty(self, intervention_applied: bool):
        """
        Update curriculum difficulty based on interventions and test performance.
        
        Args:
            intervention_applied: Whether an intervention was applied this step
        """
        if not self.config.enable_curriculum_shift:
            return
        
        if intervention_applied:
            # Original approach: simple boost after intervention
            old_difficulty = self.current_curriculum_difficulty
            self.current_curriculum_difficulty += self.config.curriculum_difficulty_boost
            self.current_curriculum_difficulty = min(1.0, self.current_curriculum_difficulty)
            
            if self.config.verbose and abs(old_difficulty - self.current_curriculum_difficulty) > 0.01:
                logger.info(f"Boosted curriculum difficulty to {self.current_curriculum_difficulty:.4f} "
                           f"after intervention")
            return
        
        # Enhanced curriculum optimization approach
        # Dynamically adjust difficulty based on test phase performance
        if (self.t1_baseline_loss is not None and 
            self.t2_baseline_loss is not None and 
            self.t3_baseline_loss is not None):
            
            # Get recent losses for each test phase
            if len(self.t1_loss_history) > 0 and len(self.t2_loss_history) > 0 and len(self.t3_loss_history) > 0:
                t1_loss = self.t1_loss_history[-1]
                t2_loss = self.t2_loss_history[-1]
                t3_loss = self.t3_loss_history[-1]
                
                # Calculate relative increase over baseline for each test phase
                t1_gap = max(0, t1_loss - self.t1_baseline_loss)
                t2_gap = max(0, t2_loss - self.t2_baseline_loss)
                t3_gap = max(0, t3_loss - self.t3_baseline_loss)
                
                # Weight the gaps according to configuration
                weighted_gap = (
                    t1_gap * self.config.t1_weight +
                    t2_gap * self.config.t2_weight +
                    t3_gap * self.config.t3_weight
                )
                
                # Calculate step: D(t+1) = D(t) + η_D * (weighted_gap)
                difficulty_step = self.config.curriculum_learning_step_size * weighted_gap
                
                # Apply step (can be positive or negative)
                old_difficulty = self.current_curriculum_difficulty
                self.current_curriculum_difficulty += difficulty_step
                
                # Ensure difficulty stays in valid range [0, 1]
                self.current_curriculum_difficulty = max(0.0, min(1.0, self.current_curriculum_difficulty))
                
                # Log significant changes
                if self.config.verbose and abs(old_difficulty - self.current_curriculum_difficulty) > 0.05:
                    direction = "increased" if difficulty_step > 0 else "decreased"
                    logger.info(f"Curriculum difficulty {direction} to {self.current_curriculum_difficulty:.4f} "
                               f"based on test performance (gap: {weighted_gap:.4f})")
            
        # Gradually decay difficulty when no specific signal
        # Only apply decay if we didn't make a targeted adjustment
        else:
            # Gradually decay difficulty back to normal
            old_difficulty = self.current_curriculum_difficulty
            self.current_curriculum_difficulty *= self.config.difficulty_decay_rate
            
            # Reset to 0 if it gets very small
            if self.current_curriculum_difficulty < 0.01:
                self.current_curriculum_difficulty = 0.0
    
    def _apply_curriculum_difficulty(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply curriculum difficulty to a batch, making it harder if needed.
        
        Args:
            batch: Input batch
            
        Returns:
            Modified batch with adjusted difficulty
        """
        if not self.config.enable_curriculum_shift or self.current_curriculum_difficulty <= 0:
            return batch
        
        # Clone the batch
        modified_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Apply difficulty modifications based on the current curriculum difficulty
        # This is a simplified example - actual implementation would depend on your data structure
        
        # 1. For sequence inputs like 'input_ids', we could truncate some tokens
        if 'input_ids' in modified_batch and 'attention_mask' in modified_batch:
            batch_size = modified_batch['input_ids'].size(0)
            seq_len = modified_batch['input_ids'].size(1)
            
            # Only apply to a subset of examples based on difficulty
            mask = torch.rand(batch_size) < self.current_curriculum_difficulty
            
            if mask.sum() > 0:
                # Truncate some portion of the sequence
                truncate_percent = 0.1 * self.current_curriculum_difficulty
                truncate_pos = int((1.0 - truncate_percent) * seq_len)
                
                # Apply truncation to selected examples
                for i in range(batch_size):
                    if mask[i]:
                        modified_batch['attention_mask'][i, truncate_pos:] = 0
        
        # 2. For numerical features, we could add noise
        for key in modified_batch:
            if isinstance(modified_batch[key], torch.Tensor) and modified_batch[key].dtype.is_floating_point:
                # Add scaled noise based on difficulty
                noise_scale = 0.05 * self.current_curriculum_difficulty
                noise = torch.randn_like(modified_batch[key]) * noise_scale
                modified_batch[key] = modified_batch[key] + noise
        
        return modified_batch
    
    def step(self) -> Dict[str, Any]:
        """
        Execute a single step in the STTT cycle.
        
        Returns:
            Dictionary of metrics for this step
        """
        next_phase = None
        
        # Determine if we need to transition to a new phase
        if not self.in_transition:
            if self.current_phase == "S" and self.phase_step >= self.config.study_steps:
                # Transition from Study to T1
                if self.config.use_soft_phase_transitions:
                    self.in_transition = True
                    self.transition_phase_from = "S"
                    self.transition_phase_to = "T1"
                    self.transition_step = 0
                    if self.config.verbose:
                        logger.info(f"Starting soft transition S → T1 at step {self.global_step}")
                else:
                    # Hard transition
                    self.current_phase = "T1"
                    self.phase_step = 0
                    if self.config.verbose:
                        logger.info(f"Transitioning to T1 phase at step {self.global_step}")
            elif self.current_phase == "T1" and self.phase_step >= self.config.t1_steps:
                # Transition from T1 to T2
                if self.config.use_soft_phase_transitions:
                    self.in_transition = True
                    self.transition_phase_from = "T1"
                    self.transition_phase_to = "T2"
                    self.transition_step = 0
                    if self.config.verbose:
                        logger.info(f"Starting soft transition T1 → T2 at step {self.global_step}")
                else:
                    # Hard transition
                    self.current_phase = "T2"
                    self.phase_step = 0
                    if self.config.verbose:
                        logger.info(f"Transitioning to T2 phase at step {self.global_step}")
            elif self.current_phase == "T2" and self.phase_step >= self.config.t2_steps:
                # Transition from T2 to T3
                if self.config.use_soft_phase_transitions:
                    self.in_transition = True
                    self.transition_phase_from = "T2"
                    self.transition_phase_to = "T3"
                    self.transition_step = 0
                    if self.config.verbose:
                        logger.info(f"Starting soft transition T2 → T3 at step {self.global_step}")
                else:
                    # Hard transition
                    self.current_phase = "T3"
                    self.phase_step = 0
                    if self.config.verbose:
                        logger.info(f"Transitioning to T3 phase at step {self.global_step}")
            elif self.current_phase == "T3" and self.phase_step >= self.config.t3_steps:
                # Transition from T3 back to Study
                if self.config.use_soft_phase_transitions:
                    self.in_transition = True
                    self.transition_phase_from = "T3"
                    self.transition_phase_to = "S"
                    self.transition_step = 0
                    if self.config.verbose:
                        logger.info(f"Starting soft transition T3 → S at step {self.global_step}")
                else:
                    # Hard transition
                    self.current_phase = "S"
                    self.phase_step = 0
                    
                    # Update baselines after a complete cycle
                    self._update_baselines()
                    
                    # Enhancement 2: Meta-Baseline Volatility
                    self._update_baseline_history()
                    volatility = self._calculate_baseline_volatility()
                    has_volatility_warning, volatility_reason = self._check_volatility_warnings(volatility)
                    
                    if has_volatility_warning and self.config.verbose:
                        logger.warning(f"Baseline volatility warning: {volatility_reason}")
                    
                    # Enhancement 3: Dynamic Grad Accum Tuning
                    self._update_grad_accum_steps()
                    
                    # Enhancement 4: Dynamic Adversarial Boosting
                    self._update_adversarial_severity()
                    
                    # Update cycle parameters
                    self._update_cycle_parameters()
                    
                    if self.config.verbose:
                        logger.info(f"Completed STTT cycle at step {self.global_step}")
        
        # Handle transition updates if we're in a soft transition
        if self.in_transition:
            self.transition_step += 1
            
            # Check if transition is complete
            if self.transition_step >= self.config.transition_window_size:
                # Transition complete, move to the next phase
                self.current_phase = self.transition_phase_to
                self.phase_step = 0
                self.in_transition = False
                
                # If transitioning back to Study, update baselines and other cycle parameters
                if self.transition_phase_to == "S":
                    # Update baselines after a complete cycle
                    self._update_baselines()
                    
                    # Enhancement 2: Meta-Baseline Volatility
                    self._update_baseline_history()
                    volatility = self._calculate_baseline_volatility()
                    has_volatility_warning, volatility_reason = self._check_volatility_warnings(volatility)
                    
                    if has_volatility_warning and self.config.verbose:
                        logger.warning(f"Baseline volatility warning: {volatility_reason}")
                    
                    # Enhancement 3: Dynamic Grad Accum Tuning
                    self._update_grad_accum_steps()
                    
                    # Enhancement 4: Dynamic Adversarial Boosting
                    self._update_adversarial_severity()
                    
                    # Update cycle parameters
                    self._update_cycle_parameters()
                
                if self.config.verbose:
                    logger.info(f"Completed transition to {self.current_phase} phase at step {self.global_step}")
        
        # Track current losses for all phases
        current_losses = {}
        
        # Execute the step based on the current phase
        if not self.in_transition:
            # Standard step within a single phase
            # Get the next batch for the current phase
            batch = self._get_next_batch(self.current_phase)
            
            # Enhancement 5: Apply curriculum difficulty for study phase
            if self.current_phase == "S" and self.config.enable_curriculum_shift:
                batch = self._apply_curriculum_difficulty(batch)
            
            if self.current_phase == "S":
                # Study phase - gradient update
                self.model.train()
                loss = self._compute_loss(batch, "S")
                
                # Backward pass and optimizer step
                loss.backward()
                
                # Enhancement 3: Dynamic grad accumulation steps
                if (self.phase_step + 1) % self.current_grad_accum_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                current_losses["S"] = loss.item()
            else:
                # Test phases - no gradient update
                self.model.eval()
                with torch.no_grad():
                    loss = self._compute_loss(batch, self.current_phase)
                
                current_losses[self.current_phase] = loss.item()
        else:
            # Soft transition between phases
            # Get one batch from each phase in the transition
            from_batch = self._get_next_batch(self.transition_phase_from)
            to_batch = self._get_next_batch(self.transition_phase_to)
            
            # Calculate transition weight (linear interpolation)
            # λ(t) = 1 - (t - t_switch) / m
            # where m is transition_window_size
            lambda_weight = 1.0 - (self.transition_step / self.config.transition_window_size)
            
            # Apply appropriate model mode based on phases involved
            if self.transition_phase_from == "S" or self.transition_phase_to == "S":
                # Any transition involving the study phase requires train mode
                self.model.train()
                
                # Process the 'from' phase batch
                from_loss = self._compute_loss(from_batch, self.transition_phase_from)
                if self.transition_phase_from == "S":
                    # Only compute gradients for study phase
                    from_loss.backward()
                    # Don't step optimizer yet - will blend with 'to' phase
                
                # Process the 'to' phase batch
                to_loss = self._compute_loss(to_batch, self.transition_phase_to)
                if self.transition_phase_to == "S":
                    # Only compute gradients for study phase
                    to_loss.backward()
                
                # Weighted combination of losses
                blended_loss = lambda_weight * from_loss.item() + (1 - lambda_weight) * to_loss.item()
                
                # Update optimizer if either phase is a study phase
                if self.transition_phase_from == "S" or self.transition_phase_to == "S":
                    # Step the optimizer with the blended gradient
                    if (self.transition_step % self.current_grad_accum_steps == 0):
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                
                # Record losses
                current_losses[self.transition_phase_from] = from_loss.item()
                current_losses[self.transition_phase_to] = to_loss.item()
                current_losses["blended"] = blended_loss
            else:
                # Transition between two test phases - both are eval mode
                self.model.eval()
                with torch.no_grad():
                    from_loss = self._compute_loss(from_batch, self.transition_phase_from)
                    to_loss = self._compute_loss(to_batch, self.transition_phase_to)
                    
                    # Weighted combination
                    blended_loss = lambda_weight * from_loss.item() + (1 - lambda_weight) * to_loss.item()
                    
                    # Record losses
                    current_losses[self.transition_phase_from] = from_loss.item()
                    current_losses[self.transition_phase_to] = to_loss.item()
                    current_losses["blended"] = blended_loss
        
        # Enhancement 1: Calculate loss derivatives
        if self.config.enable_loss_derivative_monitoring and not self.current_phase == "S":
            derivatives = self._calculate_loss_derivatives()
            has_derivative_warning, derivative_reason = self._check_derivative_warnings(derivatives)
            
            # Early warning based on derivatives
            if has_derivative_warning and self.config.verbose:
                logger.warning(f"Loss derivative warning: {derivative_reason}")
        
        # Update generalization gap metrics if we have losses from multiple phases
        if self.config.track_generalization_gaps and len(current_losses) > 1:
            self._update_generalization_gaps(current_losses)
        
        # Check for interventions (only during test phases or transitions involving test phases)
        intervention_applied = False
        if not self.current_phase == "S" or (self.in_transition and self.transition_phase_to != "S"):
            need_intervention, reason = self._check_for_intervention(current_losses)
            if need_intervention:
                self._apply_intervention(reason)
                intervention_applied = True
        
        # Enhancement 5: Update curriculum difficulty based on intervention
        self._update_curriculum_difficulty(intervention_applied)
        
        # Update metrics
        for phase in ["S", "T1", "T2", "T3", "blended"]:
            if phase in current_losses:
                self.metrics[f"{phase.lower()}_loss"].append(current_losses[phase])
        
        # Logging
        if self.global_step % self.config.log_frequency == 0:
            if not self.in_transition:
                log_msg = f"Step {self.global_step} ({self.current_phase} phase, {self.phase_step}/{getattr(self.config, f'{self.current_phase.lower()}_steps')}) - "
            else:
                progress = int(100 * self.transition_step / self.config.transition_window_size)
                log_msg = f"Step {self.global_step} (Transition {self.transition_phase_from} → {self.transition_phase_to}, {progress}%) - "
            
            # Add losses to log message
            for phase, loss_value in current_losses.items():
                log_msg += f"{phase} Loss: {loss_value:.4f} "
            
            # Add baseline losses
            if self.t1_baseline_loss is not None:
                log_msg += f"Baselines: T1={self.t1_baseline_loss:.4f} "
            if self.t2_baseline_loss is not None:
                log_msg += f"T2={self.t2_baseline_loss:.4f} "
            if self.t3_baseline_loss is not None:
                log_msg += f"T3={self.t3_baseline_loss:.4f} "
            
            # Add enhanced metrics to log
            if self.config.enable_loss_derivative_monitoring and 'derivatives' in locals():
                log_msg += f"dL/dt: "
                for phase, deriv in derivatives.items():
                    log_msg += f"{phase}={deriv:.4f} "
            
            if self.config.track_baseline_volatility and 'volatility' in locals():
                log_msg += f"σ(L): "
                for phase, vol in volatility.items():
                    log_msg += f"{phase}={vol:.4f} "
            
            if self.config.dynamic_grad_accum:
                log_msg += f"GradAccum: {self.current_grad_accum_steps} "
            
            if self.config.dynamic_adversarial_severity:
                log_msg += f"AdvSeverity: {self.current_adversarial_severity:.2f} "
            
            if self.config.enable_curriculum_shift and self.current_curriculum_difficulty > 0:
                log_msg += f"Difficulty: {self.current_curriculum_difficulty:.2f}"
            
            logger.info(log_msg)
        
        # Update step counters
        if not self.in_transition:
            self.phase_step += 1
        self.global_step += 1
        
        # Return metrics for this step
        step_metrics = {
            "global_step": self.global_step,
            "phase": self.current_phase if not self.in_transition else f"{self.transition_phase_from}→{self.transition_phase_to}",
            "phase_step": self.phase_step if not self.in_transition else self.transition_step,
            **{f"{phase.lower()}_loss": value for phase, value in current_losses.items()},
            "intervention": intervention_applied,
            # Add enhanced metrics
            "grad_accum_steps": self.current_grad_accum_steps,
            "adversarial_severity": self.current_adversarial_severity,
            "curriculum_difficulty": self.current_curriculum_difficulty,
            "in_transition": self.in_transition
        }
        
        # Add derivatives if available
        if self.config.enable_loss_derivative_monitoring and 'derivatives' in locals():
            step_metrics["loss_derivatives"] = derivatives
        
        # Add volatility if available
        if self.config.track_baseline_volatility and 'volatility' in locals():
            step_metrics["baseline_volatility"] = volatility
        
        # Add generalization gap metrics if tracking
        if self.config.track_generalization_gaps:
            gap_metrics = self._get_generalization_gap_metrics()
            if gap_metrics:
                step_metrics["generalization_gaps"] = gap_metrics
        
        return step_metrics
    
    def train(self, num_steps: int) -> Dict[str, Any]:
        """Train the model for the specified number of steps.
        
        Args:
            num_steps: Number of steps to train
            
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Starting training for {num_steps} steps")
        metrics = {}
        
        for _ in range(num_steps):
            step_results = self.step()
            
            # Track metrics
            for k, v in step_results.items():
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(v)
            
            # Update teacher models at the end of each cycle
            if self.step_count % self.config.cycle_length == 0 and self.config.enable_knowledge_distillation:
                self._update_teacher_models()
                
            # Optionally visualize latent space periodically
            if hasattr(self.config, 'enable_visualization') and self.config.enable_visualization:
                if hasattr(self.config, 'visualization_frequency'):
                    if self.step_count % self.config.visualization_frequency == 0:
                        vis_path = self.visualize_latent_space()
                        if vis_path:
                            metrics['visualization_path'] = vis_path
        
        # Final metrics
        final_metrics = {k: np.mean(v[-10:]) if len(v) >= 10 else np.mean(v) for k, v in metrics.items() 
                         if isinstance(v[0], (int, float))}
                         
        # Generate visualizations if enabled (final)
        if hasattr(self.config, 'enable_visualization') and self.config.enable_visualization:
            vis_path = self.visualize_latent_space()
            if vis_path:
                final_metrics['visualization_path'] = vis_path
                
        # Prune model if enabled
        if hasattr(self.config, 'enable_pruning') and self.config.enable_pruning:
            pruned_model = self.prune_model()
            if pruned_model is not self.model:
                self.model = pruned_model
                final_metrics['pruned'] = True
        
        return final_metrics
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current training metrics."""
        metrics = {
            "global_step": self.global_step,
            "current_phase": self.current_phase,
            "study_loss": np.mean(list(self.study_loss_history)[-10:]) if self.study_loss_history else None,
            "t1_loss": np.mean(list(self.t1_loss_history)[-10:]) if self.t1_loss_history else None,
            "t2_loss": np.mean(list(self.t2_loss_history)[-10:]) if self.t2_loss_history else None,
            "t3_loss": np.mean(list(self.t3_loss_history)[-10:]) if self.t3_loss_history else None,
            "total_interventions": len(self.interventions),
            "recent_interventions": sum(1 for i in self.interventions if self.global_step - i["step"] < 100),
            "cycle_config": {
                "study_steps": self.config.study_steps,
                "t1_steps": self.config.t1_steps,
                "t2_steps": self.config.t2_steps,
                "t3_steps": self.config.t3_steps
            }
        }
        
        # Enhancement 1: Loss derivatives
        if self.config.enable_loss_derivative_monitoring:
            derivatives = {}
            if self.t1_derivatives:
                derivatives["t1_derivative"] = np.mean(list(self.t1_derivatives)[-5:])
            if self.t2_derivatives:
                derivatives["t2_derivative"] = np.mean(list(self.t2_derivatives)[-5:])
            if self.t3_derivatives:
                derivatives["t3_derivative"] = np.mean(list(self.t3_derivatives)[-5:])
            
            metrics["loss_derivatives"] = derivatives
        
        # Enhancement 2: Baseline volatility
        if self.config.track_baseline_volatility:
            volatility = {}
            if len(self.t1_baseline_history) > 3:
                volatility["t1_volatility"] = np.std(list(self.t1_baseline_history))
            if len(self.t2_baseline_history) > 3:
                volatility["t2_volatility"] = np.std(list(self.t2_baseline_history))
            if len(self.t3_baseline_history) > 3:
                volatility["t3_volatility"] = np.std(list(self.t3_baseline_history))
            
            metrics["baseline_volatility"] = volatility
        
        # Enhancement 3: Dynamic grad accum
        if self.config.dynamic_grad_accum:
            metrics["current_grad_accum_steps"] = self.current_grad_accum_steps
            metrics["intervention_rate"] = self._calculate_intervention_rate()
        
        # Enhancement 4: Dynamic adversarial severity
        if self.config.dynamic_adversarial_severity:
            metrics["current_adversarial_severity"] = self.current_adversarial_severity
        
        # Enhancement 5: Curriculum difficulty
        if self.config.enable_curriculum_shift:
            metrics["current_curriculum_difficulty"] = self.current_curriculum_difficulty
        
        # Soft phase transitions
        if self.config.use_soft_phase_transitions:
            metrics["in_transition"] = self.in_transition
            if self.in_transition:
                metrics["transition_from"] = self.transition_phase_from
                metrics["transition_to"] = self.transition_phase_to
                metrics["transition_step"] = self.transition_step
                metrics["transition_progress"] = self.transition_step / self.config.transition_window_size
        
        # Generalization gap metrics
        if self.config.track_generalization_gaps:
            gap_metrics = self._get_generalization_gap_metrics()
            if gap_metrics:
                metrics["generalization_gaps"] = gap_metrics
        
        # Multi-objective optimization
        if self.config.use_multi_objective_opt:
            metrics["pareto_weights"] = self.pareto_weights
        
        # Failed examples information
        metrics["failed_examples_count"] = len(self.failed_examples_buffer)
        
        # Enhanced counterfactuals & adversarials
        if self.config.use_enhanced_counterfactuals:
            metrics["counterfactual_settings"] = {
                "loss_constraint": self.config.counterfactual_loss_constraint,
                "vae_weight": self.config.counterfactual_vae_weight
            }
        
        if self.config.use_gradient_adversarial:
            metrics["adversarial_settings"] = {
                "epsilon_base": self.config.fgsm_epsilon_base,
                "epsilon_effective": self.config.fgsm_epsilon_base + 
                                     self.current_adversarial_severity * self.config.fgsm_epsilon_scale
            }
        
        # Regularization status
        metrics["regularization_boost_steps_left"] = self.intervention_reg_boost_steps_left
        
        return metrics
    
    def _update_generalization_gaps(self, current_losses: Dict[str, float]):
        """
        Update generalization gap metrics between different phases.
        
        Args:
            current_losses: Dictionary of current losses for each phase
        """
        if not self.config.track_generalization_gaps:
            return
        
        # We need losses from all phases to calculate gaps
        needed_phases = ["S", "T1", "T2", "T3"]
        if not all(phase in current_losses for phase in needed_phases):
            return
        
        # Calculate generalization gaps between study and test phases
        # G_i = L_{T_i} - L_S
        self.generalization_gaps["G1"].append(current_losses["T1"] - current_losses["S"])
        self.generalization_gaps["G2"].append(current_losses["T2"] - current_losses["S"])
        self.generalization_gaps["G3"].append(current_losses["T3"] - current_losses["S"])
        
        # Calculate cross-test gaps
        # G_{ij} = L_{T_i} - L_{T_j}
        self.generalization_gaps["G12"].append(current_losses["T1"] - current_losses["T2"])
        self.generalization_gaps["G13"].append(current_losses["T1"] - current_losses["T3"])
        self.generalization_gaps["G23"].append(current_losses["T2"] - current_losses["T3"])
    
    def _get_generalization_gap_metrics(self) -> Dict[str, float]:
        """
        Get summary statistics for generalization gaps.
        
        Returns:
            Dictionary of gap metrics
        """
        if not self.config.track_generalization_gaps:
            return {}
        
        metrics = {}
        
        # Calculate moving averages of gaps
        for gap_name, gap_values in self.generalization_gaps.items():
            if len(gap_values) > 0:
                # Calculate moving average
                window_size = min(len(gap_values), 20)  # Use last 20 values at most
                recent_values = list(gap_values)[-window_size:]
                metrics[f"{gap_name}_avg"] = sum(recent_values) / len(recent_values)
                
                # Calculate trend (average of derivatives)
                if len(recent_values) > 1:
                    derivatives = [recent_values[i] - recent_values[i-1] for i in range(1, len(recent_values))]
                    metrics[f"{gap_name}_trend"] = sum(derivatives) / len(derivatives)
        
        return metrics
    
    def _update_curriculum_difficulty_from_gaps(self):
        if not self.config.track_generalization_gaps:
            return
        
        # Get recent gap values (T1-S is G1)
        if len(self.generalization_gaps["G1"]) > 0:
            g1_recent = list(self.generalization_gaps["G1"])[-5:]
            g1_avg = sum(g1_recent) / len(g1_recent)
            g1_trend = g1_recent[-1] - g1_recent[0] if len(g1_recent) > 1 else 0
            
            # If G1 is increasing rapidly, target sector and stage difficulty
            if g1_trend > 0.05:  # Gap is growing
                # Increase difficulty specifically for problematic features
                self.feature_difficulty = {
                    "sector": min(1.0, self.feature_difficulty.get("sector", 0.2) + 0.1),
                    "stage": min(1.0, self.feature_difficulty.get("stage", 0.2) + 0.1)
                }
                
                if self.config.verbose:
                    logger.info(f"Increased difficulty for sector ({self.feature_difficulty['sector']:.2f}) and "
                               f"stage ({self.feature_difficulty['stage']:.2f}) due to G1 gap trend: {g1_trend:.4f}")
    
    def _update_pareto_weights(self):
        if not self.config.use_multi_objective_opt:
            return
        
        # Check if T3 is consistently lagging (in the "red zone")
        t3_lagging = False
        if self.config.use_ema_derivatives and len(self.ema_t3_derivatives) > 5:
            recent_t3_derivatives = list(self.ema_t3_derivatives)[-5:]
            t3_lagging = all(d > 0 for d in recent_t3_derivatives)  # All derivatives positive
        
        # Use faster update rate if T3 is lagging
        update_rate = self.config.pareto_weight_update_rate * (3.0 if t3_lagging else 1.0)
        
        # Update weights based on loss derivatives
        if hasattr(self, 'ema_t1_loss') and hasattr(self, 'ema_t2_loss') and hasattr(self, 'ema_t3_loss'):
            # Use losses or derivatives to adjust weights
            total_loss = (self.ema_t1_loss + self.ema_t2_loss + self.ema_t3_loss)
            
            # Higher weight for phases with higher loss
            new_weights = {
                "T1": self.ema_t1_loss / total_loss,
                "T2": self.ema_t2_loss / total_loss,
                "T3": self.ema_t3_loss / total_loss
            }
            
            # Blend with old weights
            for phase in ["T1", "T2", "T3"]:
                self.pareto_weights[phase] = (1 - update_rate) * self.pareto_weights[phase] + update_rate * new_weights[phase]
            
            # Keep S weight fixed at 1.0
            self.pareto_weights["S"] = 1.0
    
    def _generate_vae_counterfactual(self, batch):
        """
        Generate counterfactuals using a Variational Autoencoder (VAE) model.
        
        This approach creates more realistic counterfactuals by learning a continuous 
        latent manifold that preserves the underlying data distribution while allowing
        controlled modifications in the latent space.
        
        The VAE follows the loss function:
        L_VAE = E_q(z|x)[log p(x'|z)] + 0.5 * KL(q(z|x) || p(z))
        
        Args:
            batch: Input batch to transform
            
        Returns:
            Counterfactual batch
        """
        # Initialize VAE if not already done
        if not hasattr(self, 'counterfactual_vae'):
            self._initialize_counterfactual_vae()
        
        # Extract numerical features for VAE processing
        features, feature_masks, categorical_features = self._extract_features_for_vae(batch)
        
        # Encode features to get latent representation (μ, log σ²)
        z_mean, z_log_var = self.counterfactual_vae.encode(features)
        
        # Sample from latent distribution: z ~ N(μ, σ²)
        z = self._sample_latent(z_mean, z_log_var)
        
        # Store original encoding for comparison
        original_z = z.clone()
        
        # Perturb latent space in a targeted way
        # Different perturbation strategies based on what we want to change
        
        # 1. Sector change: Perturb sector-related dimensions
        if hasattr(self, 'vae_feature_dims') and "sector" in self.vae_feature_dims and "sector_change" in self.config.counterfactual_types:
            sector_dims = self.vae_feature_dims["sector"]  # This could be a list of influential dimensions
            
            # Calculate perturbation magnitude based on config
            perturbation_scale = 0.5 + self.current_curriculum_difficulty * 0.5  # Scale with difficulty
            
            if isinstance(sector_dims, list):
                for dim in sector_dims:
                    # Apply targeted perturbation
                    z[:, dim] = z[:, dim] + perturbation_scale * torch.randn_like(z[:, dim])
            else:
                # Single dimension case
                z[:, sector_dims] = z[:, sector_dims] + perturbation_scale * torch.randn_like(z[:, sector_dims])
        
        # 2. Stage change: Perturb stage-related dimensions
        if hasattr(self, 'vae_feature_dims') and "stage" in self.vae_feature_dims and "stage_change" in self.config.counterfactual_types:
            stage_dims = self.vae_feature_dims["stage"]
            
            # For stage, we can use a more structured perturbation (e.g., moving forward in time)
            perturbation_scale = 0.4 + self.current_curriculum_difficulty * 0.6
            
            if isinstance(stage_dims, list):
                for dim in stage_dims:
                    # More structured perturbation (e.g., positive direction = later stage)
                    z[:, dim] = z[:, dim] + perturbation_scale * torch.abs(torch.randn_like(z[:, dim]))
            else:
                # Single dimension case with directed perturbation
                z[:, stage_dims] = z[:, stage_dims] + perturbation_scale * torch.abs(torch.randn_like(z[:, stage_dims]))
        
        # 3. Geography/market change
        if hasattr(self, 'vae_feature_dims') and "geography" in self.vae_feature_dims and "geography_change" in self.config.counterfactual_types:
            geo_dims = self.vae_feature_dims["geography"]
            perturbation_scale = 0.4 + self.current_curriculum_difficulty * 0.6
            
            if isinstance(geo_dims, list):
                for dim in geo_dims:
                    z[:, dim] = z[:, dim] + perturbation_scale * torch.randn_like(z[:, dim])
            else:
                z[:, geo_dims] = z[:, geo_dims] + perturbation_scale * torch.randn_like(z[:, geo_dims])
        
        # Decode to get counterfactual features
        counterfactual_features = self.counterfactual_vae.decode(z)
        
        # Verify semantic fidelity by checking loss constraint
        if self.config.counterfactual_loss_constraint > 0:
            # Calculate reconstruction loss between original and counterfactual
            # This ensures the counterfactual isn't too far from original distribution
            try:
                with torch.no_grad():
                    # Map to full batch format
                    temp_cf_batch = self._map_features_to_batch(counterfactual_features, batch, 
                                                               feature_masks, categorical_features)
                    
                    # Compute original loss
                    original_outputs = self.model(**batch)
                    original_loss = original_outputs.loss.item()
                    
                    # Compute counterfactual loss
                    cf_outputs = self.model(**temp_cf_batch)
                    cf_loss = cf_outputs.loss.item()
                    
                    # Check if loss change exceeds constraint
                    relative_change = abs(cf_loss - original_loss) / original_loss
                    if relative_change > self.config.counterfactual_loss_constraint:
                        if self.config.verbose:
                            logger.info(f"VAE counterfactual exceeds loss constraint: {relative_change:.4f} > {self.config.counterfactual_loss_constraint}")
                        
                        # Interpolate between original and perturbed latent to satisfy constraint
                        # α * original_z + (1-α) * perturbed_z where α is adjusted to satisfy constraint
                        alpha = 0.5  # Start with middle interpolation
                        for _ in range(5):  # Binary search for acceptable α
                            interp_z = alpha * original_z + (1 - alpha) * z
                            interp_features = self.counterfactual_vae.decode(interp_z)
                            
                            # Map to batch and test
                            interp_batch = self._map_features_to_batch(interp_features, batch, 
                                                                      feature_masks, categorical_features)
                            interp_outputs = self.model(**interp_batch)
                            interp_loss = interp_outputs.loss.item()
                            
                            relative_change = abs(interp_loss - original_loss) / original_loss
                            if relative_change <= self.config.counterfactual_loss_constraint:
                                # This interpolation satisfies constraint
                                counterfactual_features = interp_features
                                break
                            
                            # Adjust alpha based on constraint
                            if relative_change > self.config.counterfactual_loss_constraint:
                                # Too far, move closer to original
                                alpha = alpha + (1 - alpha) * 0.5
                            else:
                                # Could be more aggressive
                                alpha = alpha * 0.5
            except Exception as e:
                logger.warning(f"Error validating VAE counterfactual: {str(e)}")
        
        # Map back to original batch format
        counterfactual_batch = self._map_features_to_batch(counterfactual_features, batch, 
                                                          feature_masks, categorical_features)
        
        return counterfactual_batch
    
    def _initialize_counterfactual_vae(self):
        """Initialize the VAE model for counterfactual generation."""
        if hasattr(self, 'counterfactual_vae'):
            return  # Already initialized
        
        # Here we implement a simple VAE architecture
        # In a real implementation, this would be more sophisticated
        class VAE(torch.nn.Module):
            def __init__(self, input_dim, latent_dim, hidden_dim=64):
                super().__init__()
                
                # Encoder
                self.encoder = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU()
                )
                
                # Mean and variance for latent space
                self.fc_mu = torch.nn.Linear(hidden_dim, latent_dim)
                self.fc_var = torch.nn.Linear(hidden_dim, latent_dim)
                
                # Decoder
                self.decoder = torch.nn.Sequential(
                    torch.nn.Linear(latent_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, input_dim)
                )
            
            def encode(self, x):
                h = self.encoder(x)
                return self.fc_mu(h), self.fc_var(h)
            
            def decode(self, z):
                return self.decoder(z)
            
            def forward(self, x):
                mu, log_var = self.encode(x)
                
                # Reparameterization trick
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                z = mu + eps * std
                
                # Decode
                return self.decode(z), mu, log_var
        
        # Create a simple dataset from collected examples to train the VAE
        vae_dataset = []
        
        # First try to use external data if provided
        if hasattr(self, 'external_data_for_vae') and self.external_data_for_vae is not None:
            logger.info("Using external data for VAE initialization")
            dataloader_iter = iter(self.external_data_for_vae)
            for _ in range(min(100, len(self.external_data_for_vae))):
                try:
                    batch = next(dataloader_iter)
                    # Move batch to device
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    features, _, _ = self._extract_features_for_vae(batch)
                    vae_dataset.append(features)
                except StopIteration:
                    break
        
        # Use collected examples from all phases if needed
        if len(vae_dataset) < 50:
            for phase in ["S", "T1", "T2", "T3"]:
                if hasattr(self, f'{phase.lower()}_examples_buffer'):
                    buffer = getattr(self, f'{phase.lower()}_examples_buffer')
                    for example in buffer:
                        if isinstance(example, dict) and 'batch' in example:
                            features, _, _ = self._extract_features_for_vae(example['batch'])
                            vae_dataset.append(features)
        
        # If we don't have enough examples, collect more from study dataloader
        if len(vae_dataset) < 50 and hasattr(self, 'study_dataloader'):
            # Sample batches from study dataloader
            dataloader_iter = iter(self.study_dataloader)
            for _ in range(min(20, len(self.study_dataloader))):
                try:
                    batch = next(dataloader_iter)
                    # Move batch to device
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    features, _, _ = self._extract_features_for_vae(batch)
                    vae_dataset.append(features)
                except StopIteration:
                    break
        
        # Create synthetic data as a last resort if we still don't have enough examples
        if len(vae_dataset) < 10:
            logger.warning("Insufficient data for VAE training. Generating synthetic training data.")
            # If we have at least one example, use its shape to generate more
            if vae_dataset:
                example_shape = vae_dataset[0].shape
                # Generate synthetic data with similar statistics
                mean_feature = torch.mean(vae_dataset[0], dim=0)
                std_feature = torch.std(vae_dataset[0], dim=0) + 1e-6  # Avoid zero std
                # Generate 50 synthetic examples
                for _ in range(50):
                    synthetic_features = mean_feature + torch.randn(example_shape) * std_feature
                    vae_dataset.append(synthetic_features)
            else:
                # If no examples at all, create some default synthetic data
                # Assume a reasonable feature dimension (update based on your use case)
                input_dim = 32
                batch_size = 8
                synthetic_features = torch.randn(batch_size, input_dim).to(self.device)
                for _ in range(50):
                    vae_dataset.append(synthetic_features + torch.randn_like(synthetic_features) * 0.1)
        
        # Determine input dimension for VAE
        input_dim = vae_dataset[0].shape[1] if vae_dataset else 32
        
        # Create and train the VAE
        self.counterfactual_vae = VAE(
            input_dim=input_dim,
            latent_dim=self.config.vae_latent_dim,
            hidden_dim=input_dim * 2
        ).to(self.device)
        
        # If we have enough data, train the VAE
        if len(vae_dataset) >= 10:
            self._train_vae(vae_dataset)
        else:
            logger.warning("Not enough data to properly train the VAE. Using randomly initialized VAE.")
            
        # Try to load pretrained weights if available
        pretrained_path = getattr(self.config, 'vae_pretrained_weights_path', None)
        if pretrained_path and os.path.exists(pretrained_path):
            try:
                logger.info(f"Loading pretrained VAE weights from {pretrained_path}")
                state_dict = torch.load(pretrained_path, map_location=self.device)
                self.counterfactual_vae.load_state_dict(state_dict)
            except Exception as e:
                logger.warning(f"Failed to load pretrained VAE weights: {str(e)}")
        
        # Create a map of feature dimensions to help with targeted perturbations
        self._identify_feature_dimensions()
        
        logger.info(f"Initialized counterfactual VAE with latent dim {self.config.vae_latent_dim}")
        # Track VAE data metrics
        self.vae_data_size = len(vae_dataset)
        self.vae_synthetic_ratio = 0.0 if len(vae_dataset) < 10 else max(0, 50 - len(vae_dataset)) / len(vae_dataset)
    
    def _train_vae(self, dataset):
        """Train the VAE on collected examples."""
        if not dataset:
            return
        
        # Stack features into a tensor
        features = torch.cat(dataset, dim=0)
        
        # Create dataloader
        dataset = torch.utils.data.TensorDataset(features)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=min(self.config.vae_batch_size, len(dataset)),
            shuffle=True
        )
        
        # Setup optimizer
        optimizer = torch.optim.Adam(self.counterfactual_vae.parameters(), lr=1e-3)
        
        # Train loop
        self.counterfactual_vae.train()
        for epoch in range(self.config.vae_training_epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0]
                
                # Forward pass
                optimizer.zero_grad()
                x_recon, mu, log_var = self.counterfactual_vae(x)
                
                # Compute loss
                recon_loss = torch.nn.functional.mse_loss(x_recon, x)
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                
                # Total loss with KL weight
                loss = recon_loss + self.config.vae_kl_weight * kl_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0 and self.config.verbose:
                logger.info(f"VAE training epoch {epoch}, loss: {total_loss/len(dataloader):.6f}")
        
        # Switch to eval mode
        self.counterfactual_vae.eval()
        
        # Cache latent representations if desired
        if self.config.vae_cache_latents:
            self.cached_latents = {}
            with torch.no_grad():
                for i, feat in enumerate(dataset):
                    mu, log_var = self.counterfactual_vae.encode(feat[0].unsqueeze(0))
                    self.cached_latents[i] = (mu, log_var)
    
    def _identify_feature_dimensions(self):
        """
        Identify which dimensions in the latent space correspond to different features.
        
        This is a simplified approach to identify important latent dimensions.
        In a real implementation, this would use a more sophisticated method like
        disentanglement metrics or controlled perturbation analysis.
        """
        if not hasattr(self, 'counterfactual_vae'):
            return
        
        # For simplicity, we'll use a heuristic approach to assign dimensions
        # In a real implementation, this would involve training with labeled data
        # or using disentanglement techniques
        
        latent_dim = self.config.vae_latent_dim
        self.vae_feature_dims = {}
        
        # Simple heuristic: Assign specific ranges to features
        # In reality, these would be learned through feature interaction analysis
        sector_dims = list(range(0, latent_dim // 4))
        stage_dims = list(range(latent_dim // 4, latent_dim // 2))
        geography_dims = list(range(latent_dim // 2, 3 * latent_dim // 4))
        traction_dims = list(range(3 * latent_dim // 4, latent_dim))
        
        self.vae_feature_dims = {
            "sector": sector_dims,
            "stage": stage_dims,
            "geography": geography_dims,
            "traction": traction_dims
        }
    
    def _extract_features_for_vae(self, batch):
        """
        Extract numerical features from batch for VAE processing.
        
        Returns:
            features: Tensor of numerical features
            feature_masks: Dict mapping feature name to mask for reconstruction
            categorical_features: Dict of categorical features to preserve
        """
        # Identify numerical features to include in VAE
        feature_tensors = []
        feature_masks = {}
        categorical_features = {}
        
        # Extract all tensor fields that aren't labels or masks
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if key != 'labels' and not key.endswith('_mask') and not key.endswith('_ids'):
                    if value.dtype.is_floating_point:
                        # Store mask for reconstruction
                        feature_masks[key] = (len(feature_tensors), value.shape)
                        
                        # Flatten and add to list
                        feature_tensors.append(value.view(value.size(0), -1))
                    elif key in ['sector', 'stage', 'geography']:
                        # Save categorical features that we want to modify separately
                        categorical_features[key] = value.clone()
        
        # Concatenate all features
        if feature_tensors:
            features = torch.cat(feature_tensors, dim=1)
        else:
            # If no suitable features, create a dummy tensor
            # This would be rare but handles edge cases
            features = torch.randn(batch['labels'].shape[0], 32).to(self.device)
            logger.warning("No suitable features found for VAE, using random features")
        
        return features, feature_masks, categorical_features
    
    def _map_features_to_batch(self, counterfactual_features, original_batch, feature_masks, categorical_features):
        """
        Map the VAE-generated features back to batch format.
        
        Args:
            counterfactual_features: Generated features from VAE
            original_batch: Original batch with full structure
            feature_masks: Dict mapping feature name to position/shape in VAE input
            categorical_features: Dict of categorical features to incorporate
            
        Returns:
            Complete counterfactual batch matching original structure
        """
        # Clone the original batch
        cf_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in original_batch.items()}
        
        # Reconstruct numerical features
        for key, (idx, shape) in feature_masks.items():
            feature_len = shape[1] if len(shape) > 1 else 1
            if len(shape) > 1:
                # Multi-dimensional feature (reshape needed)
                start_idx = 0
                for i in range(idx):
                    prev_shape = feature_masks[list(feature_masks.keys())[i]][1]
                    prev_len = prev_shape[1] if len(prev_shape) > 1 else 1
                    start_idx += prev_len
                
                # Extract and reshape
                end_idx = start_idx + feature_len
                cf_batch[key] = counterfactual_features[:, start_idx:end_idx].view(shape)
            else:
                # 1D feature
                cf_batch[key] = counterfactual_features[:, idx]
        
        # Handle categorical features (special case)
        for key, value in categorical_features.items():
            # If VAE has instructions for this feature, apply them
            if key in self.vae_feature_dims:
                # For now, keep original categorical values
                # In a real implementation, we would decode them from the latent space
                cf_batch[key] = value
        
        return cf_batch
    
    def _sample_latent(self, mu, log_var):
        """
        Sample from the latent distribution using the reparameterization trick.
        
        Args:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
        
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def _pgd_adversarial(self, batch, steps=None, alpha=None):
        """Generate adversarial examples using PGD with adaptive parameters."""
        # Use config parameters if not provided
        steps = steps or self.config.pgd_steps
        alpha = alpha or self.config.pgd_alpha
        
        # NEW: Adaptively adjust PGD parameters based on T3 performance
        if self.config.pgd_adaptive and hasattr(self, 't3_loss_history') and len(self.t3_loss_history) > 10:
            # If T3 performance is poor (high loss), strengthen attack
            recent_t3_loss = np.mean(list(self.t3_loss_history)[-10:])
            if hasattr(self, 't3_baseline_loss') and self.t3_baseline_loss is not None:
                t3_ratio = recent_t3_loss / self.t3_baseline_loss
                
                # Adjust PGD steps and alpha based on T3 performance
                if t3_ratio < 1.1:  # T3 performance is good, strengthen attack
                    steps = min(10, steps + 1)
                    alpha = min(0.02, alpha * 1.2)
                elif t3_ratio > 1.5:  # T3 performance is poor, weaken attack slightly
                    steps = max(2, steps - 1)
                    alpha = max(0.001, alpha * 0.9)
                    
                logging.info(f"Adaptive PGD: steps={steps}, alpha={alpha:.6f} (T3 ratio: {t3_ratio:.2f})")
        
        # Create a copy of the batch for adversarial perturbation
        adversarial_batch = {k: v.clone().detach() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Identify which tensors to perturb (numerical features)
        perturb_keys = []
        for k, v in adversarial_batch.items():
            if isinstance(v, torch.Tensor) and v.dtype.is_floating_point and v.requires_grad == False and len(v.shape) > 0:
                v.requires_grad = True
                perturb_keys.append(k)
        
        # If no tensors to perturb, return the original batch
        if not perturb_keys:
            return adversarial_batch
        
        # Get original labels if they exist
        labels = batch.get('labels')
        
        # Iterative PGD loop
        for i in range(steps):
            # Forward pass to compute loss
            self.model.zero_grad()
            outputs = self.model(**{k: v for k, v in adversarial_batch.items() if k != 'labels'})
            
            # Compute loss
            if labels is not None and hasattr(outputs, 'logits'):
                loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
            elif hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                # If no clear loss to maximize, use negative confidence as proxy
                if hasattr(outputs, 'logits'):
                    probs = torch.softmax(outputs.logits, dim=-1)
                    confidence = probs.max(dim=-1)[0].mean()
                    loss = -confidence
                else:
                    # Fallback - the forward method should return a loss
                    loss = outputs if isinstance(outputs, torch.Tensor) else torch.tensor(0.0).to(self.device)
            
            # Backward pass
            loss.backward()
            
            # Update adversarial batch
            with torch.no_grad():
                for k in perturb_keys:
                    if adversarial_batch[k].grad is not None:
                        # Normalized gradient (sign method similar to FGSM)
                        grad_sign = adversarial_batch[k].grad.sign()
                        
                        # Apply perturbation
                        perturbation = alpha * grad_sign
                        adversarial_batch[k] = adversarial_batch[k] + perturbation
                        
                        # Apply constraints based on config
                        if 'non_negative' in self.config.pgd_constraint_types:
                            adversarial_batch[k] = torch.maximum(adversarial_batch[k], torch.zeros_like(adversarial_batch[k]))
                        
                        if 'norm_preserving' in self.config.pgd_constraint_types:
                            # Calculate original norm
                            orig_norm = batch[k].norm(dim=-1, keepdim=True)
                            # Normalize and rescale to preserve the norm
                            current_norm = adversarial_batch[k].norm(dim=-1, keepdim=True)
                            scale_factor = orig_norm / (current_norm + 1e-8)
                            adversarial_batch[k] = adversarial_batch[k] * scale_factor
                        
                        # Re-attach gradient for next iteration
                        adversarial_batch[k].requires_grad = True
            
            # Clear gradients for next iteration
            self.model.zero_grad()
            for k in perturb_keys:
                if adversarial_batch[k].grad is not None:
                    adversarial_batch[k].grad.zero_()
        
        # Detach all tensors for final output
        for k in perturb_keys:
            adversarial_batch[k] = adversarial_batch[k].detach()
            adversarial_batch[k].requires_grad = False
        
        return adversarial_batch
    
    def _generate_adversarial(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate adversarial examples with enhanced diversity."""
        # ... existing adversarial generation code ...
        
        # NEW: Add support for additional adversarial types
        adversarial_type = random.choice(self.config.adversarial_types)
        
        if adversarial_type == "outlier_metrics":
            # Create batch with extreme metric values (e.g., 10x normal)
            if "metrics" in batch:
                batch["metrics"] = batch["metrics"] * 10.0
            
        elif adversarial_type == "extreme_claims":
            # Simulate extreme startup claims
            if "text_description" in batch and isinstance(batch["text_description"], str):
                extreme_claims = [
                    "revolutionary", "disruptive", "groundbreaking", "10x better",
                    "unprecedented", "game-changing", "unicorn potential"
                ]
                claim = random.choice(extreme_claims)
                batch["text_description"] = f"{claim} {batch['text_description']}"
                
        elif adversarial_type == "inconsistent_data":
            # Create inconsistency between fields
            if "sector" in batch and "stage" in batch:
                # Mismatch sector and stage (early stage with mature sector)
                if batch["stage"] == "seed" and batch["sector"] == "fintech":
                    batch["metrics"] = batch["metrics"] * 5  # Unrealistic metrics for seed stage
        
        # Use PGD for gradient-based attacks
        if self.config.use_pgd_adversarial:
            return self._pgd_adversarial(batch)
        
        return batch
    
    def _update_feature_difficulties(self, batch: Dict[str, torch.Tensor], current_loss: float):
        """Update feature difficulties based on correlation with loss."""
        # Store current loss
        self.loss_history.append(current_loss)
        
        # Update feature histories and compute correlations
        for feature_name, feature_values in batch.items():
            if isinstance(feature_values, torch.Tensor) and feature_values.dim() == 1:
                self.feature_history[feature_name].append(feature_values.cpu().numpy())
                
                if len(self.feature_history[feature_name]) == self.config.feature_correlation_window:
                    # Compute correlation
                    feature_array = np.array(self.feature_history[feature_name])
                    loss_array = np.array(self.loss_history)
                    
                    # Compute correlation coefficient
                    correlation = np.corrcoef(feature_array, loss_array)[0, 1]
                    if not np.isnan(correlation):
                        # Update difficulty based on correlation and generalization gap
                        generalization_gap = self._get_generalization_gap_metrics()['t1_gap']
                        difficulty_update = self.config.feature_difficulty_step * np.sign(generalization_gap * correlation)
                        self.feature_difficulties[feature_name] = np.clip(
                            self.feature_difficulties[feature_name] + difficulty_update,
                            0.0, 1.0
                        )
    
    def _apply_feature_noise(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply feature-specific noise based on difficulty."""
        noisy_batch = {}
        for feature_name, feature_values in batch.items():
            if isinstance(feature_values, torch.Tensor) and feature_values.dim() == 1:
                difficulty = self.feature_difficulties.get(feature_name, 0.0)
                if difficulty > 0:
                    noise = torch.randn_like(feature_values) * self.config.feature_noise_std * difficulty
                    noisy_batch[feature_name] = feature_values + noise
                else:
                    noisy_batch[feature_name] = feature_values
            else:
                noisy_batch[feature_name] = feature_values
        return noisy_batch
    
    def _apply_curriculum_difficulty(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply curriculum difficulty including feature-specific noise."""
        # Apply base curriculum difficulty
        modified_batch = super()._apply_curriculum_difficulty(batch)
        
        # Apply feature-specific noise
        return self._apply_feature_noise(modified_batch)
    
    def _initialize_teacher_models(self):
        """Initialize teacher models for knowledge distillation."""
        if self.config.enable_knowledge_distillation:
            # Create a copy of the current model as the first teacher
            teacher_model = type(self.model)(**self.model.config)
            teacher_model.load_state_dict(self.model.state_dict())
            self.teacher_models.append(teacher_model)
    
    def _update_teacher_models(self):
        """Update teacher models at the end of each cycle."""
        if self.config.enable_knowledge_distillation:
            # Create a copy of the current model
            new_teacher = type(self.model)(**self.model.config)
            new_teacher.load_state_dict(self.model.state_dict())
            
            # Add new teacher and maintain only the last N cycles
            self.teacher_models.append(new_teacher)
            if len(self.teacher_models) > self.config.teacher_model_cycles:
                self.teacher_models.pop(0)
    
    def _compute_distillation_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute knowledge distillation loss using teacher models."""
        if not self.config.enable_knowledge_distillation or not self.teacher_models:
            return torch.tensor(0.0, device=self.device)
        
        # Get student model predictions
        student_outputs = self.model(**batch)
        student_logits = student_outputs.logits
        
        # Get average teacher predictions
        teacher_logits = []
        for teacher in self.teacher_models:
            with torch.no_grad():
                teacher_outputs = teacher(**batch)
                teacher_logits.append(teacher_outputs.logits)
        
        avg_teacher_logits = torch.mean(torch.stack(teacher_logits), dim=0)
        
        # Compute KL divergence
        kl_div = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student_logits, dim=-1),
            torch.nn.functional.softmax(avg_teacher_logits, dim=-1),
            reduction='batchmean'
        )
        
        return kl_div

    def auto_tune_regularization(
        self,
        num_tune_steps: int = 100,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None
    ) -> Dict[str, float]:
        """
        Auto-tune regularization weights using latent_space_regularization's auto-tuning.
        
        Args:
            num_tune_steps: Number of steps for tuning
            val_dataloader: Optional separate validation dataloader for tuning
            
        Returns:
            Dictionary of optimized weights for each phase
        """
        if not self.config.use_latent_regularization or self.latent_regularizer is None:
            logging.warning("Latent space regularization not enabled. Cannot auto-tune.")
            return {}
        
        logging.info("Starting auto-tuning of regularization weights...")
        
        # Use provided validation dataloader or T1 dataloader
        val_dataloader = val_dataloader or self.t1_dataloader
        
        from latent_space_regularization import auto_tune_regularization
        
        # Auto-tune for each phase
        phase_weights = {}
        for phase in ['study', 't1', 't2', 't3']:
            logging.info(f"Tuning {phase} phase regularization...")
            
            # Create temporary dataloader for T2/T3 phases
            if phase == 't2' and self.t2_generator:
                temp_loader = self._create_phase_dataloader(val_dataloader, self.t2_generator)
            elif phase == 't3' and self.t3_generator:
                temp_loader = self._create_phase_dataloader(val_dataloader, self.t3_generator)
            else:
                temp_loader = self.study_dataloader if phase == 'study' else val_dataloader
            
            # Run auto-tuning
            optimized_config = auto_tune_regularization(
                self.model,
                temp_loader,
                val_dataloader,
                self.original_loss_fn,
                n_trials=5,  # Reduced trials for efficiency
                trial_steps=num_tune_steps,
                device=self.device
            )
            
            # Extract optimized weight
            phase_weights[phase] = optimized_config.latent_reg_weight
        
        # Update weights
        self.reg_weights = phase_weights
        logging.info(f"Auto-tuning complete. Optimized weights: {phase_weights}")
        
        return phase_weights

    def _create_phase_dataloader(
        self,
        base_dataloader: torch.utils.data.DataLoader,
        generator_fn: Callable
    ) -> torch.utils.data.DataLoader:
        """Helper to create temporary dataloader for T2/T3 phases during auto-tuning."""
        class GeneratorDataset(torch.utils.data.Dataset):
            def __init__(self, base_loader, transform_fn):
                self.base_loader = base_loader
                self.transform_fn = transform_fn
                self.data = []
                
                # Generate transformed examples
                for batch in base_loader:
                    self.data.append(transform_fn(batch))
        
            def __len__(self):
                return len(self.data)
        
            def __getitem__(self, idx):
                return self.data[idx]
        
        # Create dataset with transformed examples
        dataset = GeneratorDataset(base_dataloader, generator_fn)
        
        # Create new dataloader
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=base_dataloader.batch_size,
            shuffle=True
        )

    def visualize_latent_space(self):
        """Visualize latent space using tools from latent_space_regularization."""
        if not self.config.enable_latent_visualization or self.latent_regularizer is None:
            return
        
        try:
            from latent_space_regularization import visualize_regularization_effects, latent_space_analysis_report
            
            # Generate timestamp for filenames
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            
            # Create visualization path with proper error handling
            vis_path = os.path.join(self.config.visualization_path, f"latent_vis_{timestamp}")
            try:
                os.makedirs(vis_path, exist_ok=True)
            except (PermissionError, OSError) as e:
                # Handle permission issues or other OS errors
                alt_path = os.path.join(".", f"latent_vis_{timestamp}")
                logging.warning(f"Failed to create directory at {vis_path}: {str(e)}. Using {alt_path} instead.")
                try:
                    os.makedirs(alt_path, exist_ok=True)
                    vis_path = alt_path
                except Exception as e2:
                    logging.error(f"Failed to create alternative directory {alt_path}: {str(e2)}. Visualization will be skipped.")
                    return None
            
            # Generate visualizations
            logging.info(f"Generating latent space visualizations at {vis_path}")
            
            # Visualize regularization effects with error handling
            vis_file = os.path.join(vis_path, "regularization_effects.png")
            try:
                visualize_regularization_effects(self.latent_regularizer, save_path=vis_file)
            except Exception as e:
                logging.error(f"Failed to generate regularization effects visualization: {str(e)}")
            
            # Generate analysis report with error handling
            report_file = os.path.join(vis_path, "latent_analysis.html")
            try:
                latent_space_analysis_report(self.latent_regularizer, filepath=report_file)
            except Exception as e:
                logging.error(f"Failed to generate latent analysis report: {str(e)}")
            
            # Verify files were actually created
            vis_success = os.path.exists(vis_file) and os.path.getsize(vis_file) > 0
            report_success = os.path.exists(report_file) and os.path.getsize(report_file) > 0
            
            if vis_success or report_success:
                logging.info(f"Latent space visualizations generated at {vis_path}")
                return vis_path
            else:
                logging.warning("Visualization files appear to be empty or were not created properly.")
                return None
                
        except ImportError as e:
            logging.error(f"Failed to import required visualization modules: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Error generating latent space visualizations: {str(e)}")
            return None

    def prune_model(self) -> torch.nn.Module:
        """Prune model based on regularization importance scores."""
        if not self.config.enable_pruning or self.latent_regularizer is None:
            return self.model
        
        try:
            from latent_space_regularization import prune_model_based_on_regularization
            
            logging.info("Pruning model based on regularization importance scores...")
            
            # Run pruning
            pruned_model, pruning_stats = prune_model_based_on_regularization(
                self.model,
                self.latent_regularizer,
                pruning_threshold=self.config.pruning_threshold,
                importance_threshold=self.config.pruning_importance_threshold,
                gradual_steps=self.config.gradual_pruning_steps
            )
            
            # Log pruning results
            total_params_before = sum(p.numel() for p in self.model.parameters())
            total_params_after = sum(p.numel() for p in pruned_model.parameters())
            pruning_ratio = 1.0 - (total_params_after / total_params_before)
            
            logging.info(f"Model pruned: {total_params_before:,} → {total_params_after:,} parameters ({pruning_ratio:.2%} reduction)")
            
            # More detailed stats
            for layer_name, stats in pruning_stats.items():
                if 'pruned_percentage' in stats:
                    logging.info(f"Layer {layer_name}: {stats['pruned_percentage']:.2%} pruned")
            
            return pruned_model
        except Exception as e:
            logging.error(f"Error pruning model: {str(e)}")
            return self.model

    def get_metrics(self) -> Dict[str, Any]:
        """Get enhanced metrics including STTT-specific and regularization metrics."""
        # Get base metrics
        metrics = super().get_metrics() if hasattr(super(), 'get_metrics') else {}
        
        # Add STTT-specific metrics
        if hasattr(self, 't1_loss_history') and self.t1_loss_history:
            metrics['t1_loss'] = np.mean(list(self.t1_loss_history)[-10:])
            
        if hasattr(self, 't2_loss_history') and self.t2_loss_history:
            metrics['t2_loss'] = np.mean(list(self.t2_loss_history)[-10:])
            
        if hasattr(self, 't3_loss_history') and self.t3_loss_history:
            metrics['t3_loss'] = np.mean(list(self.t3_loss_history)[-10:])
        
        # Add phase-specific accuracy metrics
        if hasattr(self, 'phase_accuracy'):
            for phase, acc_history in self.phase_accuracy.items():
                if acc_history:
                    metrics[f'{phase}_accuracy'] = np.mean(list(acc_history)[-10:])
        
        # Add regularization metrics if available
        if self.config.use_latent_regularization and self.latent_regularizer is not None:
            # Add regularization weights
            metrics['reg_weights'] = self.reg_weights
            
            # Add component-specific metrics
            if hasattr(self.latent_regularizer, 'layer_stats'):
                for layer_name, stats in self.latent_regularizer.layer_stats.items():
                    # Add sparsity metrics
                    if 'sparsity' in stats:
                        metrics[f'{layer_name}_sparsity'] = stats['sparsity']
                        
                    # Add spectral metrics
                    if 'spectral_norm' in stats:
                        metrics[f'{layer_name}_spectral_norm'] = stats['spectral_norm']
            
            # Add VAE counterfactual metrics
            if hasattr(self, 'vae_constraint_violations'):
                violation_rate = (self.vae_constraint_violations / max(1, self.vae_total_counterfactuals)) * 100
                metrics['vae_constraint_violation_rate'] = violation_rate
                
                if self.vae_constraint_logs:
                    recent_violations = sum(self.vae_constraint_logs[-10:]) / min(10, len(self.vae_constraint_logs))
                    metrics['vae_recent_violation_rate'] = recent_violations * 100
            
            # Add T3 robustness metrics
            if hasattr(self, 'adversarial_types_performance'):
                for adv_type, perf in self.adversarial_types_performance.items():
                    metrics[f't3_robustness_{adv_type}'] = perf
        
        return metrics

    def train(self, num_steps: int) -> Dict[str, Any]:
        """Enhanced training loop with regularization visualization and pruning."""
        # ... existing training logic ...
        
        # Run base training
        training_results = super().train(num_steps)
        
        # Generate latent space visualizations periodically during training
        if self.config.enable_latent_visualization and self.global_step % self.config.visualization_frequency == 0:
            self.visualize_latent_space()
        
        # Generate final visualizations after training
        if self.config.enable_latent_visualization:
            vis_path = self.visualize_latent_space()
            if vis_path:
                training_results['visualization_path'] = vis_path
        
        # Prune model if enabled
        if self.config.enable_pruning:
            pruned_model = self.prune_model()
            if pruned_model is not self.model:
                self.model = pruned_model
                training_results['pruned_model'] = True
        
        return training_results


# Example usage:
"""
# Initialize model and optimizer
model = ...
optimizer = ...

# Initialize data loaders
study_dataloader = ...
t1_dataloader = ...

# Define counterfactual and adversarial generators
def t2_generator(batch):
    # Transform batch into counterfactual examples
    ...
    return counterfactual_batch

def t3_generator(batch):
    # Transform batch into adversarial examples
    ...
    return adversarial_batch

# Initialize STTT Cycle
sttt_config = STTTConfig(
    study_batch_size=4,
    study_grad_accum_steps=8,
    study_learning_rate=2e-5,
    ...
)

sttt_cycle = STTTCycle(
    model=model,
    optimizer=optimizer,
    study_dataloader=study_dataloader,
    t1_dataloader=t1_dataloader,
    t2_generator=t2_generator,
    t3_generator=t3_generator,
    config=sttt_config
)

# Train for 1000 steps
metrics = sttt_cycle.train(1000)

# Get current metrics
current_metrics = sttt_cycle.get_metrics()
"""

# Add ReservoirBuffer class for memory-efficient history tracking
class ReservoirBuffer:
    """Implements reservoir sampling for memory-efficient history tracking."""
    
    def __init__(self, max_size=100):
        self.buffer = []
        self.max_size = max_size
        self.count = 0
    
    def append(self, item):
        """Add an item using reservoir sampling algorithm."""
        self.count += 1
        
        # If buffer not full, just append
        if len(self.buffer) < self.max_size:
            self.buffer.append(item)
        else:
            # Reservoir sampling: keep with probability max_size/count
            idx = np.random.randint(0, self.count)
            if idx < self.max_size:
                self.buffer[idx] = item
    
    def __len__(self):
        return len(self.buffer)
    
    def __iter__(self):
        return iter(self.buffer)
    
    def __getitem__(self, idx):
        return self.buffer[idx]

# Function to set up STTT cycle with latent space regularization
def setup_sttt_with_regularization(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    study_dataloader: torch.utils.data.DataLoader,
    t1_dataloader: torch.utils.data.DataLoader,
    t2_generator: Optional[Callable] = None,
    t3_generator: Optional[Callable] = None,
    sttt_config: STTTConfig = None,
    latent_reg_config: LatentSpaceRegConfig = None,
    device = None
) -> STTTCycle:
    """
    Set up an STTT cycle with latent space regularization.
    
    Args:
        model: The model to train
        optimizer: Optimizer for training
        study_dataloader: DataLoader for study phase
        t1_dataloader: DataLoader for T1 phase
        t2_generator: Optional generator for T2 counterfactual examples
        t3_generator: Optional generator for T3 adversarial examples
        sttt_config: Configuration for STTT cycle
        latent_reg_config: Configuration for latent space regularization
        device: Device to use for computation
        
    Returns:
        Configured STTTCycle instance
    """
    # Create default configs if not provided
    if sttt_config is None:
        sttt_config = STTTConfig(use_latent_regularization=True)
    else:
        # Enable latent regularization if not specified
        sttt_config.use_latent_regularization = True
    
    if latent_reg_config is None and sttt_config.use_latent_regularization:
        _, latent_reg_config = create_latent_space_regularizer(model)
    
    # Create and return the STTT cycle
    return STTTCycle(
        model=model,
        optimizer=optimizer,
        study_dataloader=study_dataloader,
        t1_dataloader=t1_dataloader,
        t2_generator=t2_generator,
        t3_generator=t3_generator,
        config=sttt_config,
        device=device,
        latent_reg_config=latent_reg_config
    )

def enable_latent_regularization_for_sttt(
    sttt_cycle: STTTCycle,
    latent_reg_config: LatentSpaceRegConfig = None,
    reg_weight: float = 0.1,
    phase_specific_weights: Dict[str, float] = None
) -> STTTCycle:
    """
    Enable latent space regularization for an existing STTT cycle.
    
    Args:
        sttt_cycle: Existing STTT cycle instance
        latent_reg_config: Configuration for latent space regularization (optional)
        reg_weight: General weight for latent space regularization
        phase_specific_weights: Optional dictionary with phase-specific weights
        
    Returns:
        STTT cycle with regularization enabled
    """
    # Skip if already enabled
    if sttt_cycle.config.use_latent_regularization and sttt_cycle.latent_regularizer is not None:
        logging.info("Latent space regularization already enabled for this STTT cycle")
        return sttt_cycle
    
    # Update configuration
    sttt_cycle.config.use_latent_regularization = True
    sttt_cycle.config.latent_reg_weight = reg_weight
    
    # Create regularizer if needed
    if sttt_cycle.latent_regularizer is None:
        if latent_reg_config is None:
            # Create default regularizer and config
            sttt_cycle.latent_regularizer, latent_reg_config = create_latent_space_regularizer(sttt_cycle.model)
        else:
            # Use provided config
            sttt_cycle.latent_regularizer = LatentSpaceRegularizer(
                sttt_cycle.model, 
                latent_reg_config, 
                device=sttt_cycle.device
            )
    
    # Set up phase-specific weights
    if phase_specific_weights is None:
        sttt_cycle.reg_weights = {
            'study': sttt_cycle.config.study_reg_weight,
            't1': sttt_cycle.config.t1_reg_weight,
            't2': sttt_cycle.config.t2_reg_weight,
            't3': sttt_cycle.config.t3_reg_weight
        }
    else:
        sttt_cycle.reg_weights = phase_specific_weights
        # Update config to match
        if 'study' in phase_specific_weights:
            sttt_cycle.config.study_reg_weight = phase_specific_weights['study']
        if 't1' in phase_specific_weights:
            sttt_cycle.config.t1_reg_weight = phase_specific_weights['t1']
        if 't2' in phase_specific_weights:
            sttt_cycle.config.t2_reg_weight = phase_specific_weights['t2']
        if 't3' in phase_specific_weights:
            sttt_cycle.config.t3_reg_weight = phase_specific_weights['t3']
    
    if sttt_cycle.verbose:
        logging.info(f"Latent space regularization enabled with weights: {sttt_cycle.reg_weights}")
    
    return sttt_cycle

def create_enhanced_sttt(
    model,
    optimizer,
    study_dataloader,
    t1_dataloader,
    t2_generator=None,
    t3_generator=None,
    sttt_config=None,
    latent_reg_config=None,
    device=None,
    external_vae_data=None,
    enable_visualization=True,
    enable_pruning=False,
    bayesian_boost=2.0,
    pgd_steps=5,
    pgd_alpha=0.01
):
    """Create an enhanced STTT cycle with all improvements.
    
    Args:
        model: The model to train
        optimizer: The optimizer to use
        study_dataloader: Dataloader for study phase
        t1_dataloader: Dataloader for T1 phase
        t2_generator: Optional callable for generating counterfactual examples
        t3_generator: Optional callable for generating adversarial examples
        sttt_config: Optional configuration
        latent_reg_config: Optional latent regularization config
        device: Device to use (cpu/cuda)
        external_vae_data: Optional external data for VAE training
        enable_visualization: Whether to enable latent space visualization
        enable_pruning: Whether to enable model pruning
        bayesian_boost: Boost factor for Bayesian regularization in T2 phase
        pgd_steps: Number of steps for PGD adversarial generation
        pgd_alpha: Step size for PGD updates
        
    Returns:
        Configured STTTCycle instance
    """
    # Create default config if needed
    if sttt_config is None:
        sttt_config = STTTConfig(use_latent_regularization=True)
    
    # Add enhanced parameters
    sttt_config.bayesian_boost = bayesian_boost
    sttt_config.pgd_steps = pgd_steps
    sttt_config.pgd_alpha = pgd_alpha
    sttt_config.adaptive_pgd = True
    sttt_config.enable_visualization = enable_visualization
    sttt_config.visualization_path = "./visualizations"
    sttt_config.enable_pruning = enable_pruning
    sttt_config.counterfactual_constraint = 0.2
    
    # Create cycle
    return STTTCycle(
        model=model,
        optimizer=optimizer,
        study_dataloader=study_dataloader,
        t1_dataloader=t1_dataloader,
        t2_generator=t2_generator,
        t3_generator=t3_generator,
        config=sttt_config,
        device=device,
        latent_reg_config=latent_reg_config,
        external_data_for_vae=external_vae_data
    )

# Example usage
if __name__ == "__main__":
    print("Import this module to use STTT cycle with latent space regularization")
    print("Example: from sttt_cycle import create_enhanced_sttt, STTTCycle")
