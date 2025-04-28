#!/usr/bin/env python3
"""
Comprehensive Test Suite for Unified ConvergentMetaMorph System

This script tests all components of the unified system working together:
1. Phase 1 (Vision-Language) components
2. Phase 2 (Founder-VC) components
3. Integration between phases
4. Advanced features and optimizations
"""

import os
import torch
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

# Import configuration
from config import (
    BaseConfig,
    ConvergentMetaMorphConfig,
    MetaMorphPhase2Config,
    STTTConfig,
    DynamicCurriculumConfig,
    LatentSpaceRegConfig
)

# Import model factory
from model_factory import load_model, initialize_tokenizer, initialize_processor

# Import Phase 1 components
from main import ConvergentMetaMorphTrainer
from intrinsic_dimension_minimizer import IntrinsicDimensionMinimizer
from plasticity_weighted_reweighter import PlasticityWeightedReweighter
from dynamic_mutual_information import DynamicMutualInformationTracker
from bregman_dynamics import BregmanDynamicsController
from reward_weighted_plasticity import RewardWeightedPlasticityController
from adaptive_distillation import AdaptiveDistillationController
from convergent_neural_architecture_search import ConvergentNeuralArchitectureSearch
from enhanced_metaplasticity_optimizer import EnhancedMetaplasticityOptimizer
from enhanced_architecture_controller import EnhancedConvergentArchitectureController

# Import Phase 2 components
from metamorph_phase2_trainer import MetaMorphPhase2Trainer
from train_founder_vc_matching import FounderVCDataset
from sttt_cycle import STTTCycle
from dynamic_curriculum import DynamicCurriculumConstructor
from hard_example_amplification import HardExampleAmplifier
from latent_space_regularization import LatentSpaceRegularizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"unified_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("UnifiedTest")

class FounderVCDataLoader:
    """Data loader for founder-VC matching data."""
    
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
    def load_prompt_response_pairs(self, file_path):
        """Load prompt-response pairs from markdown file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            pairs = content.split('\n\n')
            for pair in pairs:
                if '**Prompt**:' in pair and '**Response**:' in pair:
                    prompt = pair.split('**Response**:')[0].split('**Prompt**:')[1].strip()
                    response = pair.split('**Response**:')[1].strip()
                    self.data.append({
                        'prompt': prompt,
                        'response': response,
                        'type': 'prompt_response'
                    })
    
    def load_founder_vc_pairs(self, file_path):
        """Load founder-VC match pairs from markdown file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            pairs = content.split('\n\n')
            for pair in pairs:
                if '**Founder**:' in pair and '**Matched VC**:' in pair:
                    founder_info = {}
                    for line in pair.split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            key = key.strip('* ')
                            founder_info[key] = value.strip()
                    self.data.append({
                        'founder_info': founder_info,
                        'type': 'founder_vc_match'
                    })
    
    def load_founder_profiles(self, file_path):
        """Load founder profiles from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            profiles = json.load(f)
            for profile in profiles:
                self.data.append({
                    'profile': profile,
                    'type': 'founder_profile'
                })
    
    def prepare_input(self, item):
        """Prepare input for the model."""
        if item['type'] == 'prompt_response':
            text = f"Prompt: {item['prompt']}\nResponse: {item['response']}"
        elif item['type'] == 'founder_vc_match':
            info = item['founder_info']
            text = f"Founder: {info.get('Founder', '')}\n"
            text += f"Sector: {info.get('Sector', '')}\n"
            text += f"Stage: {info.get('Stage', '')}\n"
            text += f"Pitch: {info.get('Pitch', '')}\n"
            text += f"Matched VC: {info.get('Matched VC', '')}\n"
            text += f"Reason: {info.get('Reason', '')}"
        else:  # founder_profile
            profile = item['profile']
            text = f"Name: {profile.get('name', '')}\n"
            text += f"Sector: {profile.get('sector', '')}\n"
            text += f"Location: {profile.get('location', '')}\n"
            text += f"Stage: {profile.get('funding_stage', '')}\n"
            text += f"Experience: {profile.get('past_experience', '')}\n"
            text += f"Goal: {profile.get('goal', '')}"
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'labels': encoded['input_ids'].clone()  # For autoregressive training
        }
    
    def get_batch(self, batch_size=8):
        """Get a batch of data."""
        indices = np.random.choice(len(self.data), batch_size)
        batch_list = [self.prepare_input(self.data[i]) for i in indices]
        
        # Collate
        return {
            'input_ids': torch.cat([b['input_ids'] for b in batch_list]),
            'attention_mask': torch.cat([b['attention_mask'] for b in batch_list]),
            'labels': torch.cat([b['labels'] for b in batch_list])
        }

class UnifiedSystemTester:
    """Test harness for the unified ConvergentMetaMorph system."""
    
    def __init__(self, output_dir: str = "./test_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Track test results
        self.test_results = {}
        
        # Track components
        self.components = {}
        
        # Initialize configs
        self.phase1_config = ConvergentMetaMorphConfig()
        self.phase2_config = MetaMorphPhase2Config()
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.processor = None
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self._initialize_model()
        
        # Initialize data loader
        self.data_loader = FounderVCDataLoader(self.tokenizer)
        
        # Load all data sources
        self.data_loader.load_prompt_response_pairs("data/Prompt-Response Pairs 1-25.markdown")
        self.data_loader.load_founder_vc_pairs("data/Founder-VC Match Pairs 101-200.markdown")
        self.data_loader.load_founder_profiles("data/1000 Founder Profiles.json")
        
        logger.info(f"Loaded {len(self.data_loader.data)} total training examples")
    
    def _initialize_model(self):
        """Initialize a small model for testing."""
        logger.info("Initializing test model...")
        
        # Update configs for testing
        self.phase1_config.model_name = "microsoft/phi-2"  # Smaller model for testing
        self.phase1_config.micro_batch_size = 2
        self.phase1_config.max_seq_length = 128
        
        self.phase2_config.model_name = "microsoft/phi-2"
        self.phase2_config.batch_size = 2
        self.phase2_config.max_seq_length = 128
        
        # Load model, tokenizer, and processor
        self.model = load_model(self.phase1_config)
        self.tokenizer = initialize_tokenizer(self.phase1_config)
        self.processor = initialize_processor(self.phase1_config)
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer with metaplasticity
        self.optimizer = EnhancedMetaplasticityOptimizer(
            self.model.parameters(),
            torch.optim.AdamW,
            lr=2e-5,
            plasticity_eta=0.01,
            plasticity_decay=0.9999,
            plasticity_growth=1.001,
            min_plasticity=0.2,
            max_plasticity=5.0
        )
        
        logger.info("Model initialization complete")
    
    def generate_dummy_batch(self, batch_size=2, seq_length=16, include_images=False):
        """Generate dummy batch for testing."""
        batch = {
            'input_ids': torch.randint(
                0, self.tokenizer.vocab_size,
                (batch_size, seq_length),
                device=self.device
            ),
            'attention_mask': torch.ones(
                (batch_size, seq_length),
                device=self.device
            ),
            'labels': torch.randint(
                0, self.tokenizer.vocab_size,
                (batch_size, seq_length),
                device=self.device
            )
        }
        
        if include_images:
            # Generate dummy images (3 channels, 224x224)
            batch['pixel_values'] = torch.randn(
                (batch_size, 3, 224, 224),
                device=self.device
            )
        
        return batch
    
    def test_phase1_components(self):
        """Test Phase 1 (vision-language) components."""
        logger.info("Testing Phase 1 components...")
        
        try:
            # Initialize components
            components = {
                'dim_minimizer': IntrinsicDimensionMinimizer(
                    model=self.model,
                    epsilon=1e-6,
                    window_size=10
                ),
                'reweighter': PlasticityWeightedReweighter(
                    model=self.model,
                    optimizer=self.optimizer,
                    reweighting_strength=0.5
                ),
                'mi_tracker': DynamicMutualInformationTracker(
                    model=self.model,
                    num_bins=16,
                    window_size=20
                ),
                'bregman': BregmanDynamicsController(
                    model=self.model,
                    error_budget=1.0
                ),
                'rwp': RewardWeightedPlasticityController(
                    model=self.model,
                    optimizer=self.optimizer,
                    update_rate=0.01
                ),
                'nas': ConvergentNeuralArchitectureSearch(
                    model=self.model,
                    action_space={'lora_rank': [4, 8, 16]},
                    error_budget=1.0,
                    optimizer=self.optimizer
                )
            }
            
            # Run test iterations
            metrics_history = []
            
            for step in range(10):
                # Generate batch with images
                batch = self.generate_dummy_batch(include_images=True)
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Update components
                metrics = {'loss': loss.item(), 'step': step}
                
                # Dimension minimization
                components['dim_minimizer'].update(step)
                metrics['intrinsic_dim'] = components['dim_minimizer'].current_intrinsic_dim
                
                # Plasticity reweighting
                sample_weights = components['reweighter'].compute_sample_weights(batch)
                metrics['mean_weight'] = sample_weights.mean().item()
                
                # Mutual information
                components['mi_tracker'].update(step, batch)
                metrics['normalized_mi'] = components['mi_tracker'].get_normalized_mi()
                
                # Bregman dynamics
                components['bregman'].step()
                
                # Reward plasticity
                components['rwp'].update_with_reward(loss.item(), outputs, batch['labels'])
                
                # Neural architecture search
                if step % 5 == 0:  # Check every 5 steps
                    if components['nas'].should_morph_architecture():
                        modifications, error = components['nas'].morph_architecture()
                        metrics['nas_modifications'] = modifications
                        metrics['nas_error'] = error
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                metrics_history.append(metrics)
                logger.info(f"Step {step}: {metrics}")
            
            # Plot results
            self._plot_phase1_results(metrics_history)
            
            self.test_results['phase1'] = {
                'passed': True,
                'metrics_history': metrics_history
            }
            
            logger.info("Phase 1 component test completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error testing Phase 1 components: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['phase1'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_phase2_components(self):
        """Test Phase 2 (founder-VC matching) components."""
        logger.info("Testing Phase 2 components...")
        
        try:
            # Initialize STTT cycle
            sttt_config = STTTConfig()
            sttt = STTTCycle(
                model=self.model,
                study_dataloader=None,  # Will use dummy data
                t1_dataloader=None,
                config=sttt_config,
                device=self.device
            )
            
            # Initialize curriculum
            curriculum_config = DynamicCurriculumConfig()
            curriculum = DynamicCurriculumConstructor(
                model=self.model,
                train_dataset=None,  # Will use dummy data
                config=curriculum_config
            )
            
            # Initialize hard example amplification
            hard_examples = HardExampleAmplifier(
                model=self.model,
                amplification_factor=2.0,
                difficulty_threshold=0.7
            )
            
            # Initialize latent regularization
            latent_reg_config = LatentSpaceRegConfig()
            latent_reg = LatentSpaceRegularizer(
                model=self.model,
                config=latent_reg_config
            )
            
            # Run test iterations
            metrics_history = []
            
            for step in range(10):
                # Generate batch
                batch = self.generate_dummy_batch()
                
                # STTT cycle step
                sttt_metrics = sttt.step()
                
                # Curriculum step
                curriculum_batch = curriculum._apply_curriculum_difficulty(batch)
                
                # Hard example step
                hard_examples.update(batch, loss=1.0)  # Dummy loss
                
                # Latent regularization
                reg_loss = latent_reg.compute_regularization_loss()[0]
                
                # Forward pass with regularization
                outputs = self.model(**batch)
                loss = outputs.loss + reg_loss
                
                # Collect metrics
                metrics = {
                    'step': step,
                    'loss': loss.item(),
                    'reg_loss': reg_loss.item(),
                    'sttt_metrics': sttt_metrics
                }
                metrics_history.append(metrics)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                logger.info(f"Step {step}: {metrics}")
            
            # Plot results
            self._plot_phase2_results(metrics_history)
            
            self.test_results['phase2'] = {
                'passed': True,
                'metrics_history': metrics_history
            }
            
            logger.info("Phase 2 component test completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error testing Phase 2 components: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['phase2'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_full_pipeline(self):
        """Test the full training pipeline with real data."""
        logger.info("Testing full training pipeline...")
        
        try:
            # Phase 1: Vision-Language Pre-training
            trainer_p1 = ConvergentMetaMorphTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                config=self.phase1_config
            )
            trainer_p1.processor = self.processor
            
            # Run Phase 1 training with real data
            phase1_metrics = []
            for step in range(10):
                batch = self.data_loader.get_batch(batch_size=self.phase1_config.micro_batch_size)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss = trainer_p1._training_step(batch)
                phase1_metrics.append({
                    'step': step,
                    'loss': loss
                })
            
            # Phase 2: Founder-VC Matching
            trainer_p2 = MetaMorphPhase2Trainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=self.data_loader,  # Use our data loader
                val_dataset=self.data_loader,    # Use same for validation
                config=self.phase2_config
            )
            
            # Run Phase 2 training
            phase2_metrics = []
            for step in range(10):
                batch = self.data_loader.get_batch(batch_size=self.phase2_config.batch_size)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                metrics = trainer_p2.train_step(batch)
                metrics['step'] = step
                phase2_metrics.append(metrics)
            
            # Plot combined results
            self._plot_pipeline_results(phase1_metrics, phase2_metrics)
            
            self.test_results['pipeline'] = {
                'passed': True,
                'phase1_metrics': phase1_metrics,
                'phase2_metrics': phase2_metrics
            }
            
            logger.info("Full pipeline test completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error testing full pipeline: {e}")
            import traceback
            traceback.print_exc()
            self.test_results['pipeline'] = {'passed': False, 'error': str(e)}
            return False
    
    def _plot_phase1_results(self, metrics_history):
        """Plot Phase 1 test results."""
        plt.figure(figsize=(15, 10))
        
        # Plot loss
        plt.subplot(2, 2, 1)
        steps = [m['step'] for m in metrics_history]
        losses = [m['loss'] for m in metrics_history]
        plt.plot(steps, losses, 'b-', label='Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
        
        # Plot intrinsic dimension
        plt.subplot(2, 2, 2)
        dims = [m.get('intrinsic_dim', 0) for m in metrics_history]
        plt.plot(steps, dims, 'g-', label='Intrinsic Dimension')
        plt.xlabel('Step')
        plt.ylabel('Dimension')
        plt.title('Intrinsic Dimension')
        plt.grid(True)
        
        # Plot mutual information
        plt.subplot(2, 2, 3)
        mis = [m.get('normalized_mi', 0) for m in metrics_history]
        plt.plot(steps, mis, 'r-', label='Normalized MI')
        plt.xlabel('Step')
        plt.ylabel('MI')
        plt.title('Mutual Information')
        plt.grid(True)
        
        # Plot sample weights
        plt.subplot(2, 2, 4)
        weights = [m.get('mean_weight', 0) for m in metrics_history]
        plt.plot(steps, weights, 'm-', label='Mean Sample Weight')
        plt.xlabel('Step')
        plt.ylabel('Weight')
        plt.title('Sample Weights')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'phase1_results.png'))
        plt.close()
    
    def _plot_phase2_results(self, metrics_history):
        """Plot Phase 2 test results."""
        plt.figure(figsize=(15, 10))
        
        # Plot combined loss
        plt.subplot(2, 2, 1)
        steps = [m['step'] for m in metrics_history]
        losses = [m['loss'] for m in metrics_history]
        reg_losses = [m['reg_loss'] for m in metrics_history]
        
        plt.plot(steps, losses, 'b-', label='Total Loss')
        plt.plot(steps, reg_losses, 'r--', label='Reg Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True)
        
        # Plot STTT metrics if available
        plt.subplot(2, 2, 2)
        if 'sttt_metrics' in metrics_history[0]:
            sttt_losses = [m['sttt_metrics'].get('loss', 0) for m in metrics_history]
            plt.plot(steps, sttt_losses, 'g-', label='STTT Loss')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.title('STTT Cycle')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'phase2_results.png'))
        plt.close()
    
    def _plot_pipeline_results(self, phase1_metrics, phase2_metrics):
        """Plot full pipeline results."""
        plt.figure(figsize=(15, 5))
        
        # Plot Phase 1 loss
        plt.subplot(1, 2, 1)
        steps = [m['step'] for m in phase1_metrics]
        losses = [m['loss'] for m in phase1_metrics]
        plt.plot(steps, losses, 'b-', label='Phase 1 Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Phase 1: Vision-Language Pre-training')
        plt.grid(True)
        
        # Plot Phase 2 loss
        plt.subplot(1, 2, 2)
        steps = [m['step'] for m in phase2_metrics]
        losses = [m.get('loss', 0) for m in phase2_metrics]
        plt.plot(steps, losses, 'r-', label='Phase 2 Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Phase 2: Founder-VC Matching')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pipeline_results.png'))
        plt.close()
    
    def run_all_tests(self):
        """Run all tests and generate a summary report."""
        test_methods = [
            self.test_phase1_components,
            self.test_phase2_components,
            self.test_full_pipeline
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
        
        # Save report
        with open(os.path.join(self.output_dir, 'test_summary.txt'), 'w') as f:
            f.write("=== Unified System Test Summary ===\n\n")
            
            for test_name, success in results.items():
                f.write(f"{test_name}: {'PASSED' if success else 'FAILED'}\n")
            
            f.write("\n=== Detailed Results ===\n\n")
            for test_name, test_result in self.test_results.items():
                f.write(f"{test_name}:\n")
                f.write(f"  Passed: {test_result.get('passed', False)}\n")
                if 'error' in test_result:
                    f.write(f"  Error: {test_result['error']}\n")
                f.write("\n")
        
        return results


if __name__ == "__main__":
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"./unified_test_results_{timestamp}"
    
    # Initialize and run tests
    tester = UnifiedSystemTester(output_dir=output_dir)
    results = tester.run_all_tests()
    
    # Print overall result
    all_passed = all(results.values())
    print(f"\nOverall result: {'SUCCESS' if all_passed else 'FAILURE'}") 