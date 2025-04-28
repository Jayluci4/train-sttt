import os
import sys
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our components
from intrinsic_dimension_minimizer import IntrinsicDimensionMinimizer
from plasticity_weighted_reweighter import PlasticityWeightedReweighter
from dynamic_mutual_information import DynamicMutualInformationTracker
from bregman_dynamics import BregmanDynamicsController
from reward_weighted_plasticity import RewardWeightedPlasticityController
from adaptive_distillation import AdaptiveDistillationController
from convergent_neural_architecture_search import ConvergentNeuralArchitectureSearch

# Import original components for integration testing
from enhanced_metaplasticity_optimizer import EnhancedMetaplasticityOptimizer
from enhanced_architecture_controller import EnhancedConvergentArchitectureController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AdvancedFeaturesTest")

class TestHarness:
    """Test harness for advanced MetaMorph features."""
    
    def __init__(self, output_dir="./test_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Track test results
        self.test_results = {}
        
        # Track initialized components
        self.components = {}
        
        # Initialize model
        self.model = None
        self.tokenizer = None
        self._initialize_model()
    
    def _initialize_model(self, model_name="facebook/opt-125m"):
        """Initialize a small model for testing."""
        logger.info(f"Initializing model: {model_name}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Apply LoRA for testing architecture components
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj"]
        )
        
        self.model = get_peft_model(self.model, lora_config)
        logger.info(f"Applied LoRA to model with rank={lora_config.r}")
        
        # Move to device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        from torch.optim import AdamW
        self.optimizer = EnhancedMetaplasticityOptimizer(
            self.model.parameters(), AdamW,
            lr=2e-5,
            plasticity_eta=0.01,
            plasticity_decay=0.9999,
            plasticity_growth=1.001,
            min_plasticity=0.2,
            max_plasticity=5.0,
            betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0.01
        )
        
        logger.info(f"Model initialized on device: {self.device}")
    
    def generate_dummy_batch(self, batch_size=4, seq_length=16):
        """Generate a dummy batch for testing."""
        input_ids = torch.randint(
            0, self.tokenizer.vocab_size, 
            (batch_size, seq_length), 
            device=self.device
        )
        attention_mask = torch.ones_like(input_ids)
        labels = torch.randint(
            0, self.tokenizer.vocab_size, 
            (batch_size, seq_length), 
            device=self.device
        )
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def forward_backward_step(self, inputs):
        """Perform a forward and backward step."""
        self.model.train()
        
        # Forward pass
        outputs = self.model(**inputs)
        loss = outputs.loss
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        return loss.item(), outputs
    
    def test_intrinsic_dimension_minimizer(self):
        """Test IntrinsicDimensionMinimizer component."""
        logger.info("Testing IntrinsicDimensionMinimizer...")
        
        try:
            # Initialize component
            dim_minimizer = IntrinsicDimensionMinimizer(
                model=self.model,
                epsilon=1e-6,
                window_size=10,
                dimension_threshold=0.1
            )
            self.components['dim_minimizer'] = dim_minimizer
            
            # Run simulation
            dimension_history = []
            decision_history = []
            
            for step in range(20):
                # Generate batch and do forward-backward
                inputs = self.generate_dummy_batch()
                loss, _ = self.forward_backward_step(inputs)
                
                # Update intrinsic dimension tracking
                dim_minimizer.update(step)
                
                # Check decisions
                increase_rank = dim_minimizer.should_increase_rank()
                decrease_rank = dim_minimizer.should_decrease_rank()
                
                # Get current intrinsic dimension
                current_dim = dim_minimizer.current_intrinsic_dim
                if current_dim is not None:
                    dimension_history.append(current_dim)
                
                decision = 1 if increase_rank else (-1 if decrease_rank else 0)
                decision_history.append(decision)
                
                logger.info(f"Step {step}: Loss={loss:.4f}, Dim={current_dim}, Decision={decision}")
                
                # Optimizer step
                self.optimizer.step()
            
            # Plot intrinsic dimension
            if dimension_history:
                plt.figure(figsize=(10, 6))
                plt.plot(range(len(dimension_history)), dimension_history, 'b-', label='Intrinsic Dimension')
                plt.scatter(range(len(decision_history)), 
                         [d * max(dimension_history) * 0.1 for d in decision_history], 
                         c=['r' if d < 0 else 'g' if d > 0 else 'gray' for d in decision_history],
                         label='Morph Decision')
                plt.xlabel('Step')
                plt.ylabel('Intrinsic Dimension')
                plt.title('Intrinsic Dimension Evolution')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(self.output_dir, 'intrinsic_dimension.png'))
                
                # Save results
                self.test_results['intrinsic_dimension'] = {
                    'passed': True,
                    'dimension_history': dimension_history,
                    'decision_history': decision_history,
                    'final_dimension': dimension_history[-1] if dimension_history else None
                }
                
                logger.info("IntrinsicDimensionMinimizer test completed successfully")
                return True
            else:
                logger.error("No dimension history generated")
                self.test_results['intrinsic_dimension'] = {'passed': False}
                return False
                
        except Exception as e:
            logger.error(f"Error testing IntrinsicDimensionMinimizer: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['intrinsic_dimension'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_plasticity_weighted_reweighter(self):
        """Test PlasticityWeightedReweighter component."""
        logger.info("Testing PlasticityWeightedReweighter...")
        
        try:
            # Initialize component
            reweighter = PlasticityWeightedReweighter(
                model=self.model,
                optimizer=self.optimizer,
                reweighting_strength=0.5,
                max_weight_ratio=5.0,
                min_weight=0.1
            )
            self.components['reweighter'] = reweighter
            
            # Run simulation
            weight_history = []
            loss_history = []
            
            for step in range(15):
                # Generate batch
                inputs = self.generate_dummy_batch(batch_size=8)
                
                # Compute sample weights
                sample_weights = reweighter.compute_sample_weights(inputs)
                
                # Forward-backward pass
                self.model.train()
                outputs = self.model(**inputs)
                raw_loss = outputs.loss
                
                # Apply sample weights to loss
                weighted_loss = reweighter.apply_weight_to_loss(raw_loss)
                
                # Backward pass
                self.optimizer.zero_grad()
                weighted_loss.backward()
                self.optimizer.step()
                
                # Track history
                weight_stats = {
                    'mean': sample_weights.mean().item(),
                    'min': sample_weights.min().item(),
                    'max': sample_weights.max().item(),
                    'std': sample_weights.std().item()
                }
                weight_history.append(weight_stats)
                loss_history.append({
                    'raw': raw_loss.item(),
                    'weighted': weighted_loss.item()
                })
                
                logger.info(
                    f"Step {step}: Raw Loss={raw_loss.item():.4f}, "
                    f"Weighted Loss={weighted_loss.item():.4f}, "
                    f"Mean Weight={weight_stats['mean']:.4f}"
                )
            
            # Plot weight distribution
            if weight_history:
                plt.figure(figsize=(10, 6))
                steps = range(len(weight_history))
                plt.plot(steps, [w['mean'] for w in weight_history], 'b-', label='Mean Weight')
                plt.fill_between(
                    steps,
                    [w['min'] for w in weight_history],
                    [w['max'] for w in weight_history],
                    alpha=0.3, color='blue'
                )
                plt.xlabel('Step')
                plt.ylabel('Sample Weight')
                plt.title('Plasticity-Weighted Sample Weights')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(self.output_dir, 'sample_weights.png'))
                
                # Plot loss comparison
                plt.figure(figsize=(10, 6))
                plt.plot(steps, [l['raw'] for l in loss_history], 'r-', label='Raw Loss')
                plt.plot(steps, [l['weighted'] for l in loss_history], 'g-', label='Weighted Loss')
                plt.xlabel('Step')
                plt.ylabel('Loss')
                plt.title('Effect of Plasticity-Weighted Reweighting on Loss')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(self.output_dir, 'reweighted_loss.png'))
                
                # Save results
                self.test_results['plasticity_reweighter'] = {
                    'passed': True,
                    'weight_history': weight_history,
                    'loss_history': loss_history
                }
                
                logger.info("PlasticityWeightedReweighter test completed successfully")
                return True
            else:
                logger.error("No weight history generated")
                self.test_results['plasticity_reweighter'] = {'passed': False}
                return False
                
        except Exception as e:
            logger.error(f"Error testing PlasticityWeightedReweighter: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['plasticity_reweighter'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_dynamic_mutual_information(self):
        """Test DynamicMutualInformationTracker component."""
        logger.info("Testing DynamicMutualInformationTracker...")
        
        try:
            # Initialize component
            mi_tracker = DynamicMutualInformationTracker(
                model=self.model,
                num_bins=20,
                window_size=50,
                update_freq=5,
                mi_threshold=0.05
            )
            self.components['mi_tracker'] = mi_tracker
            
            # Run simulation
            mi_history = []
            decision_history = []
            
            for step in range(20):
                # Generate batch and do forward-backward
                inputs = self.generate_dummy_batch()
                loss, _ = self.forward_backward_step(inputs)
                
                # Update MI tracking
                mi_tracker.update(step, inputs)
                
                # Check decisions
                increase_capacity = mi_tracker.should_increase_capacity()
                decrease_capacity = mi_tracker.should_decrease_capacity()
                
                # Get current MI
                normalized_mi = mi_tracker.get_normalized_mi()
                mi_history.append(normalized_mi)
                
                decision = 1 if increase_capacity else (-1 if decrease_capacity else 0)
                decision_history.append(decision)
                
                logger.info(f"Step {step}: Loss={loss:.4f}, MI={normalized_mi:.4f}, Decision={decision}")
                
                # Optimizer step
                self.optimizer.step()
            
            # Plot MI
            if mi_history:
                plt.figure(figsize=(10, 6))
                plt.plot(range(len(mi_history)), mi_history, 'b-', label='Normalized MI')
                plt.scatter(range(len(decision_history)), 
                         [d * max(mi_history) * 0.1 for d in decision_history], 
                         c=['r' if d < 0 else 'g' if d > 0 else 'gray' for d in decision_history],
                         label='Capacity Decision')
                plt.xlabel('Step')
                plt.ylabel('Normalized Mutual Information')
                plt.title('Mutual Information Evolution')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(self.output_dir, 'mutual_information.png'))
                
                # Save results
                self.test_results['mutual_information'] = {
                    'passed': True,
                    'mi_history': mi_history,
                    'decision_history': decision_history
                }
                
                # Get MI stats
                mi_stats = mi_tracker.get_stats()
                logger.info(f"MI stats: {mi_stats}")
                
                logger.info("DynamicMutualInformationTracker test completed successfully")
                return True
            else:
                logger.error("No MI history generated")
                self.test_results['mutual_information'] = {'passed': False}
                return False
                
        except Exception as e:
            logger.error(f"Error testing DynamicMutualInformationTracker: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['mutual_information'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_bregman_dynamics(self):
        """Test BregmanDynamicsController component."""
        logger.info("Testing BregmanDynamicsController...")
        
        try:
            # Initialize component
            bregman_controller = BregmanDynamicsController(
                model=self.model,
                divergence_type='squared_euclidean',
                step_size_schedule='inverse_sqrt_t',
                initial_step_size=1.0,
                error_budget=1.0
            )
            self.components['bregman_controller'] = bregman_controller
            
            # Run simulation
            error_history = []
            validation_history = []
            
            for step in range(20):
                # Generate batch and do forward-backward
                inputs = self.generate_dummy_batch()
                loss, _ = self.forward_backward_step(inputs)
                
                # Record a step in Bregman dynamics
                bregman_controller.step()
                
                # Get current parameter vector
                current_params = []
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        current_params.append(param.data.detach().flatten())
                
                if current_params:
                    current_vector = torch.cat(current_params)
                    
                    # Create a slightly perturbed vector for testing
                    perturbed_vector = current_vector + torch.randn_like(current_vector) * 0.01
                    
                    # Check if morph is valid
                    is_valid, error = bregman_controller.check_morph_valid(perturbed_vector)
                    
                    error_history.append(error)
                    validation_history.append(is_valid)
                    
                    logger.info(f"Step {step}: Loss={loss:.4f}, Error={error:.6f}, Valid={is_valid}")
                
                # Optimizer step
                self.optimizer.step()
            
            # Plot error
            if error_history:
                plt.figure(figsize=(10, 6))
                plt.plot(range(len(error_history)), error_history, 'b-', label='Error Norm')
                plt.scatter(range(len(validation_history)), 
                         [v * max(error_history) * 0.1 for v in validation_history], 
                         c=['g' if v else 'r' for v in validation_history],
                         label='Validation Result')
                plt.xlabel('Step')
                plt.ylabel('Error Norm')
                plt.title('Bregman Dynamics Error Evolution')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(self.output_dir, 'bregman_dynamics.png'))
                
                # Save results
                self.test_results['bregman_dynamics'] = {
                    'passed': True,
                    'error_history': error_history,
                    'validation_history': validation_history
                }
                
                # Get Bregman dynamics stats
                stats = bregman_controller.get_stats()
                logger.info(f"Bregman dynamics stats: {stats}")
                
                logger.info("BregmanDynamicsController test completed successfully")
                return True
            else:
                logger.error("No error history generated")
                self.test_results['bregman_dynamics'] = {'passed': False}
                return False
                
        except Exception as e:
            logger.error(f"Error testing BregmanDynamicsController: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['bregman_dynamics'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_reward_weighted_plasticity(self):
        """Test RewardWeightedPlasticityController component."""
        logger.info("Testing RewardWeightedPlasticityController...")
        
        try:
            # Initialize component with a reward function
            def dummy_reward_fn(outputs, targets):
                # Simple reward: negative loss
                return -outputs.loss.item()
            
            rwp_controller = RewardWeightedPlasticityController(
                model=self.model,
                optimizer=self.optimizer,
                reward_fn=dummy_reward_fn,
                update_rate=0.01,
                min_plasticity=0.2,
                max_plasticity=5.0
            )
            self.components['rwp_controller'] = rwp_controller
            
            # Run simulation
            plasticity_history = []
            reward_history = []
            
            for step in range(20):
                # Generate batch and do forward-backward
                inputs = self.generate_dummy_batch()
                loss, outputs = self.forward_backward_step(inputs)
                
                # Update plasticity with reward
                rwp_controller.update_with_reward(
                    loss=loss, 
                    outputs=outputs, 
                    targets=inputs['labels']
                )
                
                # Get group plasticities
                plasticities = rwp_controller.get_group_plasticities()
                mean_plasticity = np.mean(list(plasticities.values())) if plasticities else 0
                plasticity_history.append(mean_plasticity)
                
                # Get reward
                reward = dummy_reward_fn(outputs, inputs['labels'])
                reward_history.append(reward)
                
                logger.info(f"Step {step}: Loss={loss:.4f}, Reward={reward:.4f}, Mean Plasticity={mean_plasticity:.4f}")
                
                # Optimizer step
                self.optimizer.step()
            
            # Plot plasticity and reward
            if plasticity_history and reward_history:
                plt.figure(figsize=(10, 6))
                plt.subplot(2, 1, 1)
                plt.plot(range(len(plasticity_history)), plasticity_history, 'b-', label='Mean Plasticity')
                plt.xlabel('Step')
                plt.ylabel('Plasticity')
                plt.title('Reward-Weighted Plasticity Evolution')
                plt.legend()
                plt.grid(True)
                
                plt.subplot(2, 1, 2)
                plt.plot(range(len(reward_history)), reward_history, 'g-', label='Reward')
                plt.xlabel('Step')
                plt.ylabel('Reward')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'reward_weighted_plasticity.png'))
                
                # Save results
                self.test_results['reward_weighted_plasticity'] = {
                    'passed': True,
                    'plasticity_history': plasticity_history,
                    'reward_history': reward_history
                }
                
                # Get controller stats
                stats = rwp_controller.get_stats()
                logger.info(f"RWP Controller stats: {stats}")
                
                logger.info("RewardWeightedPlasticityController test completed successfully")
                return True
            else:
                logger.error("No plasticity or reward history generated")
                self.test_results['reward_weighted_plasticity'] = {'passed': False}
                return False
                
        except Exception as e:
            logger.error(f"Error testing RewardWeightedPlasticityController: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['reward_weighted_plasticity'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_adaptive_distillation(self):
        """Test AdaptiveDistillationController component."""
        logger.info("Testing AdaptiveDistillationController...")
        
        try:
            # Use the same model as both student and teacher for testing
            # In practice, teacher would be a different (larger) model
            adaptive_distillation = AdaptiveDistillationController(
                student_model=self.model,
                teacher_model=self.model,  # Same model for testing
                tokenizer=self.tokenizer,
                alpha=0.5,
                temperature=2.0,
                min_confidence=0.2,
                adaptive_alpha=True
            )
            adaptive_distillation.setup(total_steps=100)  # Set total training steps
            self.components['adaptive_distillation'] = adaptive_distillation
            
            # Run simulation
            alpha_history = []
            confidence_history = []
            loss_history = []
            
            for step in range(20):
                # Generate batch
                inputs = self.generate_dummy_batch()
                
                # Perform distillation step
                hybrid_loss, distillation_info = adaptive_distillation.distill_step(inputs)
                
                # Track history
                alpha_history.append(distillation_info['alpha'])
                confidence_history.append(distillation_info.get('confidence', 0))
                loss_history.append(distillation_info.get('hybrid_loss', 0))
                
                logger.info(
                    f"Step {step}: "
                    f"Alpha={distillation_info['alpha']:.4f}, "
                    f"Confidence={distillation_info.get('confidence', 0):.4f}, "
                    f"Loss={distillation_info.get('hybrid_loss', 0):.4f}"
                )
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                hybrid_loss.backward()
                self.optimizer.step()
            
            # Plot distillation metrics
            if alpha_history and confidence_history and loss_history:
                plt.figure(figsize=(10, 8))
                
                plt.subplot(3, 1, 1)
                plt.plot(range(len(alpha_history)), alpha_history, 'b-', label='Alpha')
                plt.xlabel('Step')
                plt.ylabel('Alpha')
                plt.title('Distillation Alpha')
                plt.legend()
                plt.grid(True)
                
                plt.subplot(3, 1, 2)
                plt.plot(range(len(confidence_history)), confidence_history, 'g-', label='Confidence')
                plt.xlabel('Step')
                plt.ylabel('Confidence')
                plt.title('Teacher Confidence')
                plt.legend()
                plt.grid(True)
                
                plt.subplot(3, 1, 3)
                plt.plot(range(len(loss_history)), loss_history, 'r-', label='Hybrid Loss')
                plt.xlabel('Step')
                plt.ylabel('Loss')
                plt.title('Hybrid Loss')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'adaptive_distillation.png'))
                
                # Save results
                self.test_results['adaptive_distillation'] = {
                    'passed': True,
                    'alpha_history': alpha_history,
                    'confidence_history': confidence_history,
                    'loss_history': loss_history
                }
                
                # Get distillation stats
                stats = adaptive_distillation.get_stats()
                logger.info(f"Adaptive Distillation stats: {stats}")
                
                logger.info("AdaptiveDistillationController test completed successfully")
                return True
            else:
                logger.error("No distillation metrics generated")
                self.test_results['adaptive_distillation'] = {'passed': False}
                return False
                
        except Exception as e:
            logger.error(f"Error testing AdaptiveDistillationController: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['adaptive_distillation'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_convergent_nas(self):
        """Test ConvergentNeuralArchitectureSearch component."""
        logger.info("Testing ConvergentNeuralArchitectureSearch...")
        
        try:
            # Define action space for architecture search
            action_space = {
                'lora_rank': [4, 8, 16, 32],
                'hidden_dropout': [0.0, 0.1, 0.2, 0.3]
            }
            
            # Define simple reward function based on loss
            def reward_fn(loss_val):
                return -loss_val  # Negative loss as reward
            
            # Initialize component
            nas = ConvergentNeuralArchitectureSearch(
                model=self.model,
                action_space=action_space,
                error_budget=1.0,
                exploration_factor=0.2,
                warmup_steps=5,
                morph_interval=5,
                optimizer=self.optimizer
            )
            self.components['nas'] = nas
            
            # Run simulation
            rewards = []
            architectures = []
            errors = []
            
            for step in range(30):
                # Generate batch and do forward-backward
                inputs = self.generate_dummy_batch()
                loss, _ = self.forward_backward_step(inputs)
                
                # Compute reward
                reward = reward_fn(loss)
                rewards.append(reward)
                
                # Update NAS statistics
                nas.update(step, reward)
                
                # Check if we should morph
                if nas.should_morph_architecture():
                    # Morph architecture
                    modifications, error = nas.morph_architecture()
                    
                    if modifications:
                        architectures.append({
                            'step': step,
                            'modifications': modifications
                        })
                        errors.append(error)
                        
                        logger.info(f"Step {step}: Morphed architecture: {modifications}, Error: {error:.4f}")
                    else:
                        logger.info(f"Step {step}: No modifications applied")
                
                # Optimizer step
                self.optimizer.step()
                
                # Log progress
                logger.info(f"Step {step}: Loss={loss:.4f}, Reward={reward:.4f}")
            
            # Plot rewards and architecture changes
            if rewards:
                plt.figure(figsize=(10, 8))
                
                plt.subplot(2, 1, 1)
                plt.plot(range(len(rewards)), rewards, 'b-', label='Reward')
                
                # Mark architecture changes
                if architectures:
                    arch_steps = [a['step'] for a in architectures]
                    arch_rewards = [rewards[step] for step in arch_steps]
                    plt.scatter(arch_steps, arch_rewards, c='r', marker='o', s=50, label='Architecture Change')
                
                plt.xlabel('Step')
                plt.ylabel('Reward')
                plt.title('Reward and Architecture Changes')
                plt.legend()
                plt.grid(True)
                
                if errors:
                    plt.subplot(2, 1, 2)
                    plt.bar(range(len(errors)), errors, label='Transformation Error')
                    plt.xlabel('Morph Index')
                    plt.ylabel('Error')
                    plt.title('Architecture Transformation Error')
                    plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'convergent_nas.png'))
                
                # Save results
                self.test_results['convergent_nas'] = {
                    'passed': True,
                    'rewards': rewards,
                    'architectures': architectures,
                    'errors': errors
                }
                
                logger.info("ConvergentNeuralArchitectureSearch test completed successfully")
                return True
            else:
                logger.error("No rewards generated")
                self.test_results['convergent_nas'] = {'passed': False}
                return False
                
        except Exception as e:
            logger.error(f"Error testing ConvergentNeuralArchitectureSearch: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['convergent_nas'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_combined_features(self):
        """Test all components working together for integration testing."""
        logger.info("Testing combined features integration...")
        
        try:
            # Initialize all components
            # 1. Intrinsic Dimension Minimizer
            dim_minimizer = IntrinsicDimensionMinimizer(
                model=self.model,
                epsilon=1e-6,
                window_size=10,
                dimension_threshold=0.1
            )
            self.components['dim_minimizer'] = dim_minimizer
            
            # 2. Plasticity Weighted Reweighter
            reweighter = PlasticityWeightedReweighter(
                model=self.model,
                optimizer=self.optimizer,
                reweighting_strength=0.3,
                max_weight_ratio=3.0,
                min_weight=0.2
            )
            self.components['reweighter'] = reweighter
            
            # 3. Dynamic Mutual Information Tracker
            mi_tracker = DynamicMutualInformationTracker(
                model=self.model,
                num_bins=16,
                window_size=30,
                update_freq=5
            )
            self.components['mi_tracker'] = mi_tracker
            
            # 4. Bregman Dynamics Controller
            bregman_controller = BregmanDynamicsController(
                model=self.model,
                divergence_type='squared_euclidean',
                step_size_schedule='inverse_sqrt_t',
                error_budget=0.8
            )
            self.components['bregman_controller'] = bregman_controller
            
            # 5. Reward Weighted Plasticity Controller
            def dummy_reward_fn(outputs, targets):
                # Simple reward: negative loss
                return -outputs.loss.item()
                
            rwp_controller = RewardWeightedPlasticityController(
                model=self.model,
                optimizer=self.optimizer,
                reward_fn=dummy_reward_fn,
                update_rate=0.01
            )
            self.components['rwp_controller'] = rwp_controller
            
            # 6. Convergent Neural Architecture Search
            action_space = {
                'lora_rank': [4, 8, 16, 32],
                'hidden_dropout': [0.0, 0.1, 0.2]
            }
            
            nas = ConvergentNeuralArchitectureSearch(
                model=self.model,
                action_space=action_space,
                exploration_factor=0.2,
                warmup_steps=10,
                morph_interval=10,
                optimizer=self.optimizer
            )
            self.components['nas'] = nas
            
            # Run integration simulation
            loss_history = []
            decisions = []
            
            for step in range(40):
                # Generate batch
                inputs = self.generate_dummy_batch()
                
                # Apply plasticity-weighted reweighting
                sample_weights = reweighter.compute_sample_weights(inputs)
                
                # Forward pass
                self.model.train()
                outputs = self.model(**inputs)
                raw_loss = outputs.loss
                
                # Apply sample weights
                weighted_loss = reweighter.apply_weight_to_loss(raw_loss)
                
                # Backward pass
                self.optimizer.zero_grad()
                weighted_loss.backward()
                
                # Update trackers and controllers
                dim_minimizer.update(step)
                mi_tracker.update(step, inputs)
                bregman_controller.step()
                
                # Record loss
                loss_val = weighted_loss.item()
                loss_history.append(loss_val)
                
                # Make architecture decisions
                decision = {}
                
                # Intrinsic dimension decision
                increase_rank = dim_minimizer.should_increase_rank()
                decrease_rank = dim_minimizer.should_decrease_rank()
                if increase_rank or decrease_rank:
                    direction = "increase" if increase_rank else "decrease"
                    decision['dim_rank'] = direction
                
                # Mutual information decision
                increase_capacity = mi_tracker.should_increase_capacity()
                decrease_capacity = mi_tracker.should_decrease_capacity()
                if increase_capacity or decrease_capacity:
                    direction = "increase" if increase_capacity else "decrease"
                    decision['mi_capacity'] = direction
                
                # NAS decision
                if nas.should_morph_architecture():
                    modifications, error = nas.morph_architecture()
                    if modifications:
                        decision['nas'] = modifications
                
                if decision:
                    decisions.append({
                        'step': step,
                        'decisions': decision
                    })
                    logger.info(f"Step {step} decisions: {decision}")
                
                # Update reward-weighted plasticity
                rwp_controller.update_with_reward(loss_val, outputs, inputs['labels'])
                
                # Update NAS reward
                reward = dummy_reward_fn(outputs, inputs['labels'])
                nas.update(step, reward)
                
                # Log progress
                logger.info(f"Step {step}: Loss={loss_val:.4f}")
                
                # Optimizer step
                self.optimizer.step()
            
            # Plot combined results
            plt.figure(figsize=(10, 8))
            
            # Loss plot
            plt.subplot(2, 1, 1)
            plt.plot(range(len(loss_history)), loss_history, 'b-', label='Loss')
            
            # Mark decision points
            for d in decisions:
                plt.axvline(x=d['step'], color='r', linestyle='--', alpha=0.3)
            
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.title('Training Loss with Architecture Decisions')
            plt.legend()
            plt.grid(True)
            
            # Decision types plot
            if decisions:
                decision_types = set()
                for d in decisions:
                    for k in d['decisions'].keys():
                        decision_types.add(k)
                
                decision_types = list(decision_types)
                decision_counts = {t: [0] * len(loss_history) for t in decision_types}
                
                for d in decisions:
                    for k in d['decisions'].keys():
                        decision_counts[k][d['step']] = 1
                
                plt.subplot(2, 1, 2)
                for i, t in enumerate(decision_types):
                    counts = decision_counts[t]
                    plt.scatter(
                        [i for i, x in enumerate(counts) if x > 0],
                        [i + 1 for x in counts if x > 0],
                        label=t, alpha=0.7
                    )
                    
                plt.xlabel('Step')
                plt.yticks(range(1, len(decision_types) + 1), decision_types)
                plt.title('Architecture Decision Types')
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'combined_features.png'))
            
            # Save results
            self.test_results['combined_features'] = {
                'passed': True,
                'loss_history': loss_history,
                'decisions': decisions
            }
            
            logger.info("Combined features integration test completed successfully")
            return True
                
        except Exception as e:
            logger.error(f"Error testing combined features: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['combined_features'] = {'passed': False, 'error': str(e)}
            return False
    
    def run_all_tests(self):
        """Run all tests and generate a summary report."""
        test_methods = [
            self.test_intrinsic_dimension_minimizer,
            self.test_plasticity_weighted_reweighter,
            self.test_dynamic_mutual_information,
            self.test_bregman_dynamics,
            self.test_reward_weighted_plasticity,
            self.test_adaptive_distillation,
            self.test_convergent_nas,
            self.test_combined_features
        ]
        
        results = {}
        
        for test_method in test_methods:
            test_name = test_method.__name__
            logger.info(f"\n=== Running {test_name} ===\n")
            
            success = test_method()
            results[test_name] = success
            
            logger.info(f"\n=== {test_name} completed with result: {success} ===\n")
        
        # Generate summary report
        logger.info("\n=== Test Summary ===")
        for test_name, success in results.items():
            logger.info(f"{test_name}: {'PASSED' if success else 'FAILED'}")
        
        # Save report to file
        with open(os.path.join(self.output_dir, 'test_summary.txt'), 'w') as f:
            f.write("=== MetaMorph Advanced Features Test Summary ===\n\n")
            
            for test_name, success in results.items():
                f.write(f"{test_name}: {'PASSED' if success else 'FAILED'}\n")
            
            f.write("\n=== Detailed Results ===\n\n")
            for test_name, test_result in self.test_results.items():
                f.write(f"{test_name}:\n")
                f.write(f"  Passed: {test_result.get('passed', False)}\n")
                if 'error' in test_result:
                    f.write(f"  Error: {test_result['error']}\n")
                f.write("\n")
        
        logger.info(f"Test summary saved to {os.path.join(self.output_dir, 'test_summary.txt')}")
        
        return results


if __name__ == "__main__":
    # Create output directory with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"./test_results_{timestamp}"
    
    # Initialize and run tests
    test_harness = TestHarness(output_dir=output_dir)
    results = test_harness.run_all_tests()
    
    # Print overall result
    all_passed = all(results.values())
    print(f"\nOverall result: {'SUCCESS' if all_passed else 'FAILURE'}")
