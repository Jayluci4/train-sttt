# Convergent MetaMorph Framework

A mathematically rigorous framework for adaptive architecture learning with formal convergence guarantees and advanced features that enhance neural network training.

## Overview

Convergent MetaMorph is a next-generation training framework that applies several mathematically-grounded strategies to dynamically adapt neural network architectures during training while maintaining theoretical guarantees of convergence. By integrating principles from information theory, Bregman dynamics, and intrinsic dimensionality analysis, this framework enables models to adapt their capacity to the demands of the task while ensuring stable optimization.

## Core Components

### 1. Intrinsic Dimension Minimization

Tracks the empirical Fisher Information Matrix to determine the intrinsic dimensionality of the parameter space. This guides architecture morphing to ensure model complexity aligns with task requirements.

```python
# Mathematical definition of intrinsic dimension:
# dim_ε(F) = (tr F)² / (tr F² + ε)
```

### 2. Plasticity-Weighted Data Reweighting

Adaptively upweights training samples that impact highly plastic parameters, accelerating learning on dimensions that still "move" during training.

```python
# Let ψ_i be the plasticity of parameter group i
# ω(x) ∝ ∑_i ψ_i · ‖∇_{θ_i} ℓ(f_θ(x), y)‖²
```

### 3. Dynamic Mutual Information Objectives

Uses mutual information between inputs and latent representations to guide architectural updates, ensuring updates increase information flow.

```python
# Every morphing step should increase:
# I_{t+1}(x; z_{t+1}) - I_{t}(x; z_{t}) > 0
```

### 4. Bregman Dynamics for Formal Convergence

Models architecture morphing as mirror descent steps with Bregman divergence, providing formal convergence guarantees.

```python
# θ_{t+1} = argmin_θ ⟨∇L_t, θ⟩ + (1/η_t) D_ϕ(θ || θ_t)
```

### 5. Reward-Weighted Plasticity Alignment

Integrates task performance into parameter plasticity, making weights sensitive to rewards.

```python
# ψ_{i,t+1} = ψ_{i,t} + β · r_t(x,y) · ‖∇_{θ_i} ℓ(f_θ(x),y)‖²
```

### 6. Adaptive Distillation with Confidence Reweighting

Uses teacher confidence to adjust distillation pressure, backing off when teacher is uncertain.

```python
# L_hybrid = (1 - α_t) · ℓ(f_θ(x), y) + α_t · c(x) · KL(f_θ(x) || p_T(y|x))
```

### 7. Convergent Neural Architecture Search (C-NAS)

Auto-morphs architecture using Thompson sampling to maximize expected reward while respecting convergence constraints.

```python
# max_{a_t ∈ A} E[r_t(a_t)] subject to ∑_{t} ‖T_{a_t}(θ_t) - θ_t‖ < B
```

### 8. Chain-of-Thought (CoT) Injection

Mathematically grounded prompt transformation that reduces sample complexity by making reasoning traces explicit.

```python
# x ↦ x̃ = T_CoT(x)
# L_total = λ · L(f_θ(x), y) + (1-λ) · L(f_θ(T_CoT(x)), y)
```

## Implementation Structure

```
convergent-metamorph/
├── intrinsic_dimension_minimizer.py
├── plasticity_weighted_reweighter.py
├── dynamic_mutual_information.py
├── bregman_dynamics.py
├── reward_weighted_plasticity.py
├── adaptive_distillation.py
├── convergent_neural_architecture_search.py
├── chain_of_thought_injector.py
├── enhanced_metaplasticity_optimizer.py
├── enhanced_architecture_controller.py
├── enhanced_activation_engineering.py
├── convergent_metamorph_trainer.py
└── enhanced_features_test.py
```

## Mathematical Foundations

Our framework is built on rigorous mathematical foundations from:

1. **Information Theory**: Measuring and maximizing mutual information flows
2. **Non-expansive Operators**: Ensuring bounded transformations with guaranteed convergence
3. **Bregman Divergences**: Non-Euclidean geometry for optimal weight-space navigation
4. **Two-Timescale Stochastic Approximation**: Separating architecture and weight optimization processes
5. **Thompson Sampling**: Principled exploration-exploitation trade-off for architecture search
6. **Metaplasticity**: Biologically-inspired adaptive learning rate modulation

## Usage Example

```python
from convergent_metamorph_trainer import ConvergentMetaMorphTrainer, ConvergentMetaMorphConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Configure the framework
config = ConvergentMetaMorphConfig(
    model_name="facebook/opt-125m",
    output_dir="./metamorph_output",
    use_intrinsic_dimension=True,
    use_plasticity_reweighter=True,
    use_dynamic_mi=True,
    use_bregman_dynamics=True,
    use_reward_plasticity=True,
    use_convergent_nas=True,
    epochs=3
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.micro_batch_size)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.micro_batch_size)

# Create trainer and train
trainer = ConvergentMetaMorphTrainer(model, tokenizer, config)
import asyncio
asyncio.run(trainer.train(train_loader, val_loader))
```

## Convergence Guarantees

This framework provides theoretical convergence guarantees under the following conditions:

1. The error introduced by architecture morphing is summable: ∑_t ‖T_{a_t}(θ_t) - θ_t‖ < B
2. The architecture morphing operator is non-expansive
3. Plasticity updates operate on a slower timescale than weight updates
4. The learning rate schedule follows standard assumptions for stochastic optimization

## Citation

If you use this framework in your research, please cite:

```
@software{ConvergentMetaMorph2025,
  author = {Convergent MetaMorph Team},
  title = {Convergent MetaMorph: A Mathematically Rigorous Framework for Adaptive Architecture Learning},
  year = {2025},
  month = {4},
  url = {https://github.com/convergent-metamorph/framework}
}
```

## License

MIT
