"""
Unified Configuration System for ConvergentMetaMorph

This module provides a hierarchical configuration system for the entire
training pipeline, with a base configuration class and specialized 
configurations for each component.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Config")

@dataclass
class BaseConfig:
    """Base configuration class with common parameters."""
    # Basic info
    model_name: str = "microsoft/Phi-3.5-vision-instruct"
    output_dir: str = "./output"
    
    # Training parameters
    batch_size: int = 8
    max_seq_length: int = 512
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    epochs: int = 3
    
    # Hardware settings
    device: str = "cuda"
    mixed_precision: bool = True
    seed: int = 42
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 50
    
    def __post_init__(self):
        """Ensure output directory exists."""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def update(self, config_dict: Dict[str, Any]):
        """Update config from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown config parameter: {key}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


@dataclass
class ConvergentMetaMorphConfig(BaseConfig):
    """Configuration for ConvergentMetaMorph with vision support (Phase 1)."""
    # Model configuration
    model_name: str = "microsoft/Phi-3.5-vision-instruct"
    output_dir: str = "./phi_vision_metamorph_output"

    # Data configuration
    data_path: str = "path/to/your/full_dataset.csv"
    image_dir: str = "./company_images"
    
    # Vision model specific settings
    max_image_size: int = 1024
    use_flash_attention: bool = True
    num_image_crops: int = 12
    image_processing_threads: int = 8
    
    # Batch sizes
    micro_batch_size: int = 6
    gradient_accumulation_steps: int = 8
    
    # Quantization settings
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True
    
    # PEFT settings
    use_peft: bool = True
    peft_method: str = "lora"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    
    # Advanced Component Activation
    use_intrinsic_dimension: bool = True
    use_plasticity_reweighter: bool = True
    use_dynamic_mi: bool = True
    use_bregman_dynamics: bool = True
    use_reward_plasticity: bool = True
    use_adaptive_distillation: bool = True
    use_convergent_nas: bool = True
    use_cot_injection: bool = True
    use_dynamic_architecture: bool = True
    use_metaplasticity: bool = True
    use_activation_engineering: bool = True
    
    # L4 GPU Advanced Features
    enable_tensor_parallel: bool = True
    enable_multimodal_fusion: bool = True
    enable_multi_gpu_checkpointing: bool = True
    use_fused_adam: bool = True
    enable_activation_checkpointing: bool = True
    
    # L4 Memory Management
    dynamic_token_length: bool = True
    auto_kv_cache_scaling: bool = True
    gradual_unfreezing: bool = True
    use_gradient_checkpointing: bool = True
    use_cpu_offloading: bool = False
    
    # Latent Space Regularization Integration
    use_latent_regularization: bool = False
    latent_reg_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetaMorphPhase2Config(BaseConfig):
    """Master configuration for MetaMorph Phase 2."""
    # Model configuration
    model_name: str = "microsoft/Phi-3.5-mini"
    peft_method: str = "lora"
    output_dir: str = "./metamorph_phase2_output"
    
    # Data configuration
    founder_profiles: str = "1000 Founder Profiles.json"
    founder_vc_matches: str = "Founder-VC Match Pairs 101-200.markdown"
    
    # STTT Cycle configuration
    enable_sttt: bool = True
    sttt_config: Dict[str, Any] = field(default_factory=dict)
    
    # Dynamic Curriculum configuration
    enable_curriculum: bool = True
    curriculum_config: Dict[str, Any] = field(default_factory=dict)
    
    # Hard Example Amplification configuration
    enable_hard_examples: bool = True
    hard_example_config: Dict[str, Any] = field(default_factory=dict)
    
    # Latent Space Regularization configuration
    enable_latent_reg: bool = False
    latent_reg_config: Dict[str, Any] = field(default_factory=dict)
    
    # Internal Consistency Checking configuration
    enable_consistency: bool = False
    consistency_config: Dict[str, Any] = field(default_factory=dict)
    
    # Synthetic Data configuration
    enable_synthetic: bool = False
    synthetic_config: Dict[str, Any] = field(default_factory=dict)
    
    # Training settings
    training_steps: int = 10000


@dataclass
class STTTConfig:
    """Configuration for STTT Cycle (Study-Test-Test-Test Loop)."""
    # Study phase settings
    study_batch_size: int = 4
    study_grad_accum_steps: int = 8
    study_learning_rate: float = 2e-5
    
    # Test phases settings
    t1_batch_size: int = 8
    t2_batch_size: int = 4
    t3_batch_size: int = 4
    
    # STTT Cycle control
    cycle_length: int = 100
    study_steps: int = 80
    t1_steps: int = 10
    t2_steps: int = 5
    t3_steps: int = 5
    
    # Intervention thresholds
    t1_loss_threshold: float = 0.2
    t2_loss_threshold: float = 0.3
    t3_loss_threshold: float = 0.4
    
    # Logging
    log_frequency: int = 10
    verbose: bool = True


@dataclass
class DynamicCurriculumConfig:
    """Configuration for Dynamic Curriculum Construction."""
    # Curriculum phases
    num_curriculum_phases: int = 5
    phase_duration: int = 1000
    
    # Data difficulty scoring
    difficulty_metric: str = "loss"
    min_difficulty: float = 0.0
    max_difficulty: float = 1.0
    
    # Phase progression
    auto_progress: bool = True
    progress_threshold: float = 0.1
    plateau_patience: int = 5
    min_steps_per_phase: int = 500
    
    # Logging
    log_frequency: int = 100
    verbose: bool = True


@dataclass
class LatentSpaceRegConfig:
    """Configuration for Latent Space Regularization."""
    # Core regularization parameters
    l1_penalty_weight: float = 1e-5
    kl_penalty_weight: float = 1e-5
    orthogonal_penalty_weight: float = 1e-5
    group_sparsity_weight: float = 1e-5
    hessian_penalty_weight: float = 1e-6
    spectral_penalty_weight: float = 1e-6
    
    # Dynamic adjustment
    dynamic_penalty_adjustment: bool = True
    min_penalty_weight: float = 1e-8
    max_penalty_weight: float = 1e-2
    
    # Scheduling
    warmup_steps: int = 1000
    cooldown_steps: int = 5000
    schedule_type: str = "cosine"
    
    # Logging
    log_frequency: int = 100
    verbose: bool = True


def load_config_from_file(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        filepath: Path to the JSON configuration file
        
    Returns:
        Dictionary with configuration parameters
    """
    try:
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return config_dict
    except Exception as e:
        logger.error(f"Error loading config from {filepath}: {e}")
        raise


def create_default_config(config_type: str) -> BaseConfig:
    """
    Create a default configuration object based on type.
    
    Args:
        config_type: Type of configuration to create ("phase1", "phase2", etc.)
        
    Returns:
        Configuration object
    """
    if config_type == "phase1":
        return ConvergentMetaMorphConfig()
    elif config_type == "phase2":
        return MetaMorphPhase2Config()
    elif config_type == "sttt":
        return STTTConfig()
    elif config_type == "curriculum":
        return DynamicCurriculumConfig()
    elif config_type == "latent_reg":
        return LatentSpaceRegConfig()
    else:
        logger.warning(f"Unknown config type: {config_type}, using base config")
        return BaseConfig()


def save_config_to_file(config: BaseConfig, filepath: str):
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration object
        filepath: Path to save the configuration
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving config to {filepath}: {e}")
        raise 