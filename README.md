# Unified ConvergentMetaMorph Training System

This repository contains a unified training system for advanced language model training, specifically designed for vision-language models and specialized founder-VC matching tasks.

## System Architecture

The training system is divided into two main phases:

### Phase 1: Vision-Language Pre-training

The first phase uses the ConvergentMetaMorph framework to train vision-language models with advanced optimization techniques:

- **Intrinsic Dimension Minimization**: Measures and reduces the effective dimensionality of the parameter space
- **Plasticity-Weighted Reweighting**: Adapts parameter updates based on their plasticity
- **Dynamic Mutual Information**: Tracks information flow between layers 
- **Bregman Dynamics**: Provides convergence guarantees
- **Neural Architecture Search**: Automatically optimizes model architecture
- **Metaplasticity Optimization**: Dynamically adjusts parameter-specific learning rates

### Phase 2: Founder-VC Matching

The second phase specializes a pre-trained model for founder-VC matching with:

- **STTT Cycle**: Study-Test-Test-Test cycle with robust validation on held-out, counterfactual, and adversarial examples
- **Dynamic Curriculum**: Adaptive example difficulty progression
- **Hard Example Amplification**: Focuses on challenging examples
- **Latent Space Regularization**: Sophisticated regularization techniques

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/unified-convergent-metamorph.git
cd unified-convergent-metamorph

# Install dependencies
pip install -r requirements.txt
```

## Configuration

The system uses a hierarchical configuration system with:

- Base configuration parameters
- Phase-specific settings
- Component-specific parameters

Configuration can be defined through JSON files or command-line arguments.

Example configuration files are provided in the `configs/` directory:
- `configs/phase1.json` - Configuration for Phase 1 (vision-language pre-training)
- `configs/phase2.json` - Configuration for Phase 2 (founder-VC matching)
- `configs/combined.json` - Combined configuration for both phases

## Usage

The system provides a single entry point through `train.py`:

### Phase 1 Training (Vision-Language Pre-training)

```bash
python train.py --mode phase1 --config configs/phase1.json \
    --data_path data/company_dataset.csv \
    --image_dir data/company_images \
    --output_dir output/vision_phase1
```

### Phase 2 Training (Founder-VC Matching)

```bash
python train.py --mode phase2 --config configs/phase2.json \
    --model_name output/vision_phase1/phase1_final \
    --founder_profiles data/1000\ Founder\ Profiles.json \
    --founder_vc_matches data/Founder-VC\ Match\ Pairs\ 101-200.markdown \
    --output_dir output/founder_vc_phase2
```

### Full Pipeline (Both Phases)

```bash
python train.py --mode full --config configs/combined.json \
    --output_dir output/complete_pipeline
```

## Command-Line Arguments

- `--mode`: Training mode (`phase1`, `phase2`, or `full`)
- `--config`: Path to configuration file
- `--output_dir`: Output directory for models and logs
- `--data_path`: Path to training data (CSV for phase1)
- `--image_dir`: Path to images directory (for phase1)
- `--founder_profiles`: Path to founder profiles JSON (for phase2)
- `--founder_vc_matches`: Path to founder-VC matches (for phase2)
- `--model_name`: HuggingFace model name or path to checkpoint
- `--batch_size`: Training batch size
- `--epochs`/`--training_steps`: Number of training epochs/steps
- `--learning_rate`: Learning rate
- `--seed`: Random seed for reproducibility
- `--device`: Device to use (`cuda`, `cpu`)

Component-specific flags:
- `--enable_sttt`: Enable STTT cycle (for phase2)
- `--enable_curriculum`: Enable dynamic curriculum (for phase2)
- `--enable_hard_examples`: Enable hard example amplification (for phase2)
- `--enable_latent_reg`: Enable latent space regularization

## Data Preparation

### Phase 1 (Vision-Language)

For Phase 1, prepare a CSV file with company information and images:
- CSV should include columns for company descriptions, image references, etc.
- Images should be placed in the specified image directory

### Phase 2 (Founder-VC Matching)

For Phase 2, prepare:
- JSON file with founder profiles (see `1000 Founder Profiles.json`)
- Markdown/JSON file with founder-VC match pairs (see `Founder-VC Match Pairs 101-200.markdown`)

## Advanced Components

### STTT Cycle

The STTT cycle consists of four phases:
- **Study Phase (S)**: Gradient descent on labeled founder-VC matching examples
- **Test Phase 1 (T1)**: Validation on held-out founders
- **Test Phase 2 (T2)**: Validation against counterfactual founders (changed sectors/stages)
- **Test Phase 3 (T3)**: Adversarial validation (partial information, noisy founders)

This cycle detects shallow fitting and dynamically intervenes when performance degrades.

### Dynamic Curriculum

The dynamic curriculum:
- Assesses example difficulty
- Organizes training into progressive phases
- Adapts to model performance
- Gradually introduces harder examples

### Latent Space Regularization

Advanced regularization techniques including:
- L1/Group sparsity penalties
- Orthogonal regularization
- Topological regularization
- Spectral norm constraints
- Bayesian regularization

## Credits

This system integrates multiple advanced training techniques for robust language model training.

## License

[MIT License](LICENSE) 