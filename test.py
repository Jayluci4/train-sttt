import os
import sys
import torch
import logging
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FeatureImportTester")

def check_import(module_name, alternative_import=None):
    """Try to import a module and return success status."""
    try:
        if alternative_import:
            exec(f"from {module_name} import {alternative_import}")
            logger.info(f"✓ Successfully imported {alternative_import} from {module_name}")
        else:
            exec(f"import {module_name}")
            logger.info(f"✓ Successfully imported {module_name}")
        return True
    except ImportError as e:
        logger.error(f"✗ Failed to import {module_name}: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Error when importing {module_name}: {e}")
        return False

def test_torch_functionality():
    """Test basic PyTorch functionality."""
    try:
        # Check if CUDA is available
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA available: {cuda_available}")
        
        # Create a simple tensor
        x = torch.randn(3, 3)
        logger.info(f"Created tensor with shape {x.shape}")
        
        # Test basic operations
        y = torch.nn.functional.relu(x)
        logger.info(f"Applied ReLU with result shape {y.shape}")
        
        return True
    except Exception as e:
        logger.error(f"PyTorch functionality test failed: {e}")
        return False

def test_transformers_functionality():
    """Test basic transformers functionality."""
    try:
        # Load a tiny model just to test
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        logger.info(f"Loaded tokenizer with vocab size: {len(tokenizer)}")
        
        # Tokenize a sample text
        text = "Hello, this is a test."
        tokens = tokenizer(text, return_tensors="pt")
        logger.info(f"Tokenized text with shape: {tokens.input_ids.shape}")
        
        return True
    except Exception as e:
        logger.error(f"Transformers functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting feature import tests...")
    
    # Create results directory
    output_dir = "./test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Track test results
    results = {}
    
    # 1. Test core libraries
    core_libs = [
        "torch", "transformers", "peft", "numpy", "logging", 
        "matplotlib.pyplot", "os", "sys", "pathlib"
    ]
    
    for lib in core_libs:
        results[f"import_{lib}"] = check_import(lib)
    
    # 2. Test custom component imports
    custom_components = [
        "intrinsic_dimension_minimizer.IntrinsicDimensionMinimizer",
        "plasticity_weighted_reweighter.PlasticityWeightedReweighter",
        "dynamic_mutual_information.DynamicMutualInformationTracker",
        "bregman_dynamics.BregmanDynamicsController",
        "reward_weighted_plasticity.RewardWeightedPlasticityController",
        "adaptive_distillation.AdaptiveDistillationController",
        "convergent_neural_architecture_search.ConvergentNeuralArchitectureSearch",
        "enhanced_metaplasticity_optimizer.EnhancedMetaplasticityOptimizer",
        "enhanced_architecture_controller.EnhancedConvergentArchitectureController",
        "advanced_modal_fusion.AdvancedModalFusion"
    ]
    
    for component in custom_components:
        module, cls = component.split(".")
        results[f"import_{cls}"] = check_import(module, cls)
    
    # 3. Test PyTorch functionality
    results["torch_functionality"] = test_torch_functionality()
    
    # 4. Test transformers functionality
    results["transformers_functionality"] = test_transformers_functionality()
    
    # Print summary
    logger.info("\n=== Test Summary ===")
    passed = 0
    failed = 0
    
    for test_name, success in results.items():
        status = "PASSED" if success else "FAILED"
        logger.info(f"{test_name}: {status}")
        if success:
            passed += 1
        else:
            failed += 1
    
    # Save report to file
    with open(os.path.join(output_dir, 'import_test_summary.txt'), 'w') as f:
        f.write("=== MetaMorph Features Import Test Summary ===\n\n")
        f.write(f"Tests Passed: {passed}\n")
        f.write(f"Tests Failed: {failed}\n\n")
        
        for test_name, success in results.items():
            f.write(f"{test_name}: {'PASSED' if success else 'FAILED'}\n")
    
    logger.info(f"\nTests completed: {passed} passed, {failed} failed")
    logger.info(f"Test summary saved to {os.path.join(output_dir, 'import_test_summary.txt')}")
    
    return passed, failed

if __name__ == "__main__":
    passed, failed = main()
    print(f"\nImport tests completed with {passed} passed and {failed} failed tests")
    
    if failed == 0:
        print("All tests PASSED! System is ready for advanced features.")
    else:
        print(f"Some tests FAILED. Please check the logs and fix {failed} failed dependencies.") 