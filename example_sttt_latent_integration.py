# Example: Using Latent Space Regularization with STTT Cycle
import torch
from sttt_cycle import STTTConfig, setup_sttt_with_regularization, STTTCycle, enable_latent_regularization_for_sttt
from latent_space_regularization import LatentSpaceRegConfig
# Import the enhanced STTT cycle
from sttt_enhancements import EnhancedSTTTCycle, create_enhanced_sttt

def example_integration():
    """
    This function demonstrates how to integrate latent space regularization
    with the STTT cycle. It's for illustrative purposes only.
    """
    # 1. Define model and dataloaders (placeholder)
    class YourModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(100, 50),
                torch.nn.ReLU(),
                torch.nn.Linear(50, 10)
            )
            
        def forward(self, x):
            return self.layers(x)
            
        def compute_loss(self, batch):
            # Placeholder loss function
            return torch.tensor(0.5, requires_grad=True)
    
    # Create model and optimizer
    model = YourModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Placeholder dataloaders - replace with actual dataloaders in real use
    your_study_dataloader = torch.utils.data.DataLoader([])
    your_validation_dataloader = torch.utils.data.DataLoader([])
    
    # Optional external data for VAE pre-training (improves counterfactual quality)
    external_data = torch.utils.data.DataLoader([])  # Replace with actual external data
    
    #-----------------------------------------------------------------
    # Method 1: Create a basic STTT cycle with latent regularization
    #-----------------------------------------------------------------
    
    # 2. Configure STTT cycle with enhanced settings
    sttt_config = STTTConfig(
        # Basic parameters
        study_batch_size=4,
        t1_batch_size=8,
        study_steps=80,
        t1_steps=10,
        
        # Enable latent space regularization with scheduling
        use_latent_regularization=True,
        latent_reg_weight=0.1,
        phase_specific_reg=True,
        study_reg_weight=0.1,  # Study phase gets more regularization 
        t1_reg_weight=0.05,    # Validation phase gets less regularization
        t2_reg_weight=0.15,    # Counterfactual phase gets more regularization
        t3_reg_weight=0.2,     # Adversarial phase gets even more regularization
        adapt_reg_on_intervention=True,  # Increase regularization during interventions
        
        # NEW: Enhanced logging settings
        verbose=True,
        log_frequency=10  # Log more frequently for debugging
    )
    
    # 3. Configure latent space regularization with scheduling
    latent_reg_config = LatentSpaceRegConfig(
        # Sparsity parameters
        l1_penalty_weight=1e-5,
        group_sparsity_weight=1e-5,
        sparsity_target=0.2,
        
        # Orthogonality
        orthogonal_penalty_weight=1e-5,
        
        # Advanced parameters
        hessian_penalty_weight=1e-6,
        spectral_penalty_weight=1e-6,
        
        # NEW: Enhanced scheduling settings
        dynamic_penalty_adjustment=True,
        penalty_schedule="adaptive",  # Use adaptive scheduling
        cycle_length=1000,
        warmup_steps=100,
        cooldown_steps=1000,
        
        # NEW: Track more metrics
        track_spectral_norm=True,
        track_generalization_gaps=True,
        use_multi_objective_opt=True
    )
    
    # 4. Create STTT cycle with regularization
    sttt_cycle = setup_sttt_with_regularization(
        model=model,
        optimizer=optimizer,
        study_dataloader=your_study_dataloader,
        t1_dataloader=your_validation_dataloader,
        sttt_config=sttt_config,
        latent_reg_config=latent_reg_config,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    # NEW: Auto-tune regularization weights before training
    print("\nAuto-tuning regularization weights...")
    optimized_weights = sttt_cycle.auto_tune_regularization(
        num_tune_steps=50,  # Quick tuning for example
        val_dataloader=your_validation_dataloader
    )
    print(f"Optimized weights: {optimized_weights}")
    
    # 5. Run the cycle (in real use)
    # results = sttt_cycle.train(num_steps=10000)
    print("\nCreated STTT cycle with latent space regularization")
    
    #-----------------------------------------------------------------
    # NEW METHOD: Create an ENHANCED STTT cycle with advanced features
    #-----------------------------------------------------------------
    enhanced_cycle = create_enhanced_sttt(
        model=model,
        optimizer=optimizer,
        study_dataloader=your_study_dataloader,
        t1_dataloader=your_validation_dataloader,
        # Enhanced counterfactual generator
        t2_generator=lambda batch: {k: v * 1.2 if isinstance(v, torch.Tensor) else v for k, v in batch.items()},
        # Enhanced adversarial generator
        t3_generator=lambda batch: {k: v + 0.1 if isinstance(v, torch.Tensor) else v for k, v in batch.items()},
        # Use the same configs as before
        sttt_config=sttt_config,
        latent_reg_config=latent_reg_config,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        # NEW: External data for VAE pre-training
        external_vae_data=external_data,
        # NEW: Enable visualization and model pruning
        enable_visualization=True,
        enable_pruning=True,
        # NEW: Enhanced parameters
        bayesian_boost=2.0,  # Boost Bayesian regularization for T2 phase
        pgd_steps=5,         # Increase PGD steps for stronger adversarial examples
        pgd_alpha=0.01       # Increase PGD step size
    )
    
    print("\nCreated Enhanced STTT cycle with advanced features")
    
    # Example: run training and visualize latent space
    # results = enhanced_cycle.train(num_steps=1000)
    # vis_path = enhanced_cycle.visualize_latent_space()
    # pruned_model = enhanced_cycle.prune_model()
    
    #-----------------------------------------------------------------
    # Method 3: Enable regularization for an existing STTT cycle
    #-----------------------------------------------------------------
    
    # Create normal STTT cycle without regularization
    normal_cycle = STTTCycle(
        model=model,
        optimizer=optimizer,
        study_dataloader=your_study_dataloader,
        t1_dataloader=your_validation_dataloader
    )
    
    # Enable regularization later with custom phase weights
    phase_weights = {
        'study': 0.1,
        't1': 0.05,
        't2': 0.15,
        't3': 0.2
    }
    
    regularized_cycle = enable_latent_regularization_for_sttt(
        sttt_cycle=normal_cycle,
        reg_weight=0.1,
        phase_specific_weights=phase_weights
    )
    
    print("\nAdded latent space regularization to existing STTT cycle")
    return {
        "with_reg": sttt_cycle, 
        "enhanced": enhanced_cycle,
        "added_later": regularized_cycle
    }

# Example of benefits and usage
"""
Benefits of this integration:

1. Phase-Specific Regularization:
   - Study phase: Moderate regularization to prevent overfitting
   - T1 (validation): Light regularization to assess generalization
   - T2 (counterfactual): Stronger regularization to enforce invariance
   - T3 (adversarial): Strongest regularization to enhance robustness

2. Adaptive Intervention:
   - When the STTT cycle detects performance issues, latent regularization
     is automatically strengthened to improve robustness
   - Different regularization types can be increased based on the specific
     issues detected (e.g., more sparsity for overfitting)

3. NEW: Dynamic Scheduling with RegularizationScheduler:
   - Automatically adjusts regularization strength based on performance
   - Supports curriculum learning with gradual increase in difficulty
   - Adapts to different phases of training for optimal regularization

4. NEW: Enhanced Monitoring and Debugging:
   - Detailed logging of all regularization components
   - Track spectral norms and generalization gaps
   - Multi-objective optimization for balanced regularization

5. NEW: Auto-Tuning Capabilities:
   - Automatically find optimal regularization weights for each phase
   - Separate tuning for different types of regularization
   - Phase-specific optimization for better performance

6. NEW: Advanced VAE Counterfactual Generation:
   - Pre-train VAE with external data for better distribution coverage
   - Track counterfactual constraint violations to monitor VAE quality
   - Constraint enforcement for more realistic counterfactuals

7. NEW: Adaptive PGD for Adversarial Training:
   - Automatically adjust PGD parameters based on T3 performance
   - Stronger attacks when T3 is doing well, weaker when struggling
   - Better robustness against adversarial examples

8. NEW: Visualization and Model Pruning:
   - Visualize latent space regularization effects
   - Generate analysis reports for debugging
   - Prune model after training for faster inference

9. Regularization Types Available:
   - Sparsity (L1): Encourages sparse latent representations
   - Group sparsity: Encourages entire feature groups to be sparse
   - Orthogonality: Encourages feature disentanglement
   - Spectral: Controls the spectral properties of representations
   - Hessian: Encourages smoother manifolds in latent space
   - Bayesian: Improves uncertainty quantification for T2 robustness

10. Benefits for different STTT phases:
   - Study: More efficient learning of meaningful features
   - T1: Better generalization to held-out data
   - T2: More robust to counterfactual scenarios
   - T3: Enhanced resistance to adversarial examples
"""

if __name__ == "__main__":
    print("This is an example module and not meant to be run directly.")
    print("Import and use the integration functions in your actual code.")
