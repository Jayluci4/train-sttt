import torch
import logging
import math
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class RewardWeightedPlasticityController:
    """
    Implements reward-weighted plasticity alignment to make plasticity sensitive to performance.
    
    This component adjusts parameter plasticity based on task performance:
        - Highly rewarded samples strengthen plasticity in their contributing weights
        - Poorly scored samples suppress plasticity
    
    Mathematical update rule:
        ψ_{i,t+1} = ψ_{i,t} + β · r_t(x,y) · ‖∇_{θ_i} ℓ(f_θ(x),y)‖²
    
    Where:
        - ψ_i is the plasticity of parameter group i
        - r_t(x,y) is the reward for the current sample
        - β is the update rate
        - ∇_{θ_i} ℓ(f_θ(x),y) is the gradient for parameter group i
    """
    def __init__(
        self, 
        model,
        optimizer,
        reward_fn: Optional[Callable] = None,
        update_rate: float = 0.01,
        smoothing_factor: float = 0.9,
        min_plasticity: float = 0.1,
        max_plasticity: float = 10.0,
        reward_scaling: str = 'centered'  # 'raw', 'centered', or 'normalized'
    ):
        self.model = model
        self.optimizer = optimizer
        self.reward_fn = reward_fn
        self.update_rate = update_rate
        self.smoothing_factor = smoothing_factor
        self.min_plasticity = min_plasticity
        self.max_plasticity = max_plasticity
        self.reward_scaling = reward_scaling
        
        # Parameter group tracking
        self.param_groups = self._initialize_param_groups()
        
        # Reward history for scaling
        self.reward_history = deque(maxlen=100)
        
        # Plasticity tracking
        self.baseline_plasticities = {}
        self.plasticity_history = defaultdict(list)
        
        # Initialize baseline plasticities
        self._initialize_baseline_plasticities()
        
        logger.info(f"Initialized RewardWeightedPlasticityController with {len(self.param_groups)} groups")
    
    def _initialize_param_groups(self) -> List[Tuple[str, List[torch.nn.Parameter]]]:
        """Initialize parameter groups for tracking."""
        # Group parameters by module
        param_groups = defaultdict(list)
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Group by top-level module
                top_level = name.split('.')[0]
                param_groups[top_level].append((name, param))
        
        # Convert to list of tuples
        return [(group_name, [p for _, p in params]) 
                for group_name, params in param_groups.items()]
    
    def _initialize_baseline_plasticities(self) -> None:
        """Initialize baseline plasticity values from optimizer state if available."""
        # Try to extract current plasticity values from optimizer
        for group_name, params in self.param_groups:
            group_plasticity = 1.0  # Default
            
            # Try to extract from neural plasticity optimizer
            if hasattr(self.optimizer, 'state'):
                plasticity_sum = 0.0
                param_count = 0
                
                for param in params:
                    if param in self.optimizer.state:
                        state = self.optimizer.state[param]
                        if 'plasticity_factor' in state:
                            plasticity_sum += state['plasticity_factor']
                            param_count += 1
                
                if param_count > 0:
                    group_plasticity = plasticity_sum / param_count
            
            self.baseline_plasticities[group_name] = group_plasticity
            logger.debug(f"Initialized baseline plasticity for {group_name}: {group_plasticity:.4f}")
    
    def update_with_reward(
        self, 
        loss: torch.Tensor, 
        outputs: Any, 
        targets: Any, 
        reward: Optional[float] = None
    ) -> None:
        """
        Update plasticity values based on reward signal.
        
        Args:
            loss: Loss tensor
            outputs: Model outputs
            targets: Target values
            reward: Optional explicit reward value (if None, computed from reward_fn)
        """
        # Compute reward if not provided
        if reward is None:
            if self.reward_fn is not None:
                reward = self.reward_fn(outputs, targets)
            else:
                # Default: negative loss as reward
                reward = -loss.item()
        
        # Record reward for scaling
        self.reward_history.append(reward)
        
        # Scale reward
        scaled_reward = self._scale_reward(reward)
        
        # Compute gradient contributions for each parameter group
        group_contributions = self._compute_gradient_contributions()
        
        # Update plasticity values based on reward and gradient contributions
        self._update_plasticities(scaled_reward, group_contributions)
    
    def _scale_reward(self, reward: float) -> float:
        """Scale reward based on history."""
        if not self.reward_history or len(self.reward_history) < 2:
            return np.clip(reward, -1.0, 1.0)
        
        if self.reward_scaling == 'raw':
            # Use raw reward value
            return reward
        elif self.reward_scaling == 'centered':
            # Center around mean
            mean_reward = np.mean(self.reward_history)
            return reward - mean_reward
        elif self.reward_scaling == 'normalized':
            # Z-score normalization
            mean_reward = np.mean(self.reward_history)
            std_reward = np.std(self.reward_history) + 1e-8  # Avoid division by zero
            return (reward - mean_reward) / std_reward
        else:
            # Default: clip to [-1, 1]
            return np.clip(reward, -1.0, 1.0)
    
    def _compute_gradient_contributions(self) -> Dict[str, float]:
        """Compute gradient contributions for each parameter group."""
        group_contributions = {}
        
        for group_name, params in self.param_groups:
            # Compute squared gradient norm for this group
            group_grad_norm_sq = 0.0
            param_count = 0
            
            for param in params:
                if param.grad is not None:
                    group_grad_norm_sq += torch.sum(param.grad**2).item()
                    param_count += 1
            
            # Normalize by parameter count
            if param_count > 0:
                group_contributions[group_name] = group_grad_norm_sq / param_count
            else:
                group_contributions[group_name] = 0.0
        
        return group_contributions
    
    def _update_plasticities(self, reward: float, group_contributions: Dict[str, float]) -> None:
        """
        Update plasticity values based on reward and gradient contributions.
        
        Args:
            reward: Scaled reward value
            group_contributions: Gradient contributions per group
        """
        # Skip if optimizer doesn't have a state dict (e.g., not a neural plasticity optimizer)
        if not hasattr(self.optimizer, 'state'):
            return
        
        # Normalize contributions if any are non-zero
        total_contribution = sum(group_contributions.values())
        if total_contribution > 0:
            normalized_contributions = {
                group: contribution / total_contribution
                for group, contribution in group_contributions.items()
            }
        else:
            normalized_contributions = group_contributions
        
        # Update plasticity for each parameter
        for group_name, params in self.param_groups:
            if group_name not in normalized_contributions:
                continue
                
            group_contribution = normalized_contributions[group_name]
            
            # Compute plasticity update for this group
            # ψ_{i,t+1} = ψ_{i,t} + β · r_t(x,y) · ‖∇_{θ_i} ℓ(f_θ(x),y)‖²
            plasticity_update = self.update_rate * reward * group_contribution
            
            # Apply update to each parameter in the group
            for param in params:
                if param in self.optimizer.state:
                    state = self.optimizer.state[param]
                    
                    if 'plasticity_factor' in state:
                        # Get current plasticity
                        current_plasticity = state['plasticity_factor']
                        
                        # Apply update with smoothing
                        new_plasticity = (
                            self.smoothing_factor * current_plasticity + 
                            (1 - self.smoothing_factor) * (current_plasticity + plasticity_update)
                        )
                        
                        # Ensure bounds
                        new_plasticity = max(self.min_plasticity, 
                                          min(self.max_plasticity, new_plasticity))
                        
                        # Update plasticity
                        state['plasticity_factor'] = new_plasticity
            
            # Record updated plasticity
            self._record_group_plasticity(group_name)
    
    def _record_group_plasticity(self, group_name: str) -> None:
        """Record current plasticity for a parameter group."""
        total_plasticity = 0.0
        param_count = 0
        
        for param in self.param_groups[list(map(lambda x: x[0], self.param_groups)).index(group_name)][1]:
            if param in self.optimizer.state:
                state = self.optimizer.state[param]
                if 'plasticity_factor' in state:
                    total_plasticity += state['plasticity_factor']
                    param_count += 1
        
        if param_count > 0:
            avg_plasticity = total_plasticity / param_count
            self.plasticity_history[group_name].append(avg_plasticity)
    
    def get_group_plasticities(self) -> Dict[str, float]:
        """Get current average plasticity for each parameter group."""
        group_plasticities = {}
        
        for group_name, params in self.param_groups:
            total_plasticity = 0.0
            param_count = 0
            
            for param in params:
                if param in self.optimizer.state:
                    state = self.optimizer.state[param]
                    if 'plasticity_factor' in state:
                        total_plasticity += state['plasticity_factor']
                        param_count += 1
            
            if param_count > 0:
                group_plasticities[group_name] = total_plasticity / param_count
            else:
                group_plasticities[group_name] = 1.0  # Default
        
        return group_plasticities
    
    def get_reward_model(self, loss_fn=None) -> Callable:
        """
        Get a reward function that computes reward based on predictive performance.
        
        Args:
            loss_fn: Optional loss function (defaults to cross-entropy)
            
        Returns:
            Reward function that takes (outputs, targets) and returns a scalar reward
        """
        # Define a default reward function
        def default_reward_fn(outputs, targets):
            # For classification tasks
            if hasattr(outputs, 'logits') and hasattr(targets, 'dim') and targets.dim() == 1:
                # Use accuracy as reward
                predictions = outputs.logits.argmax(dim=-1)
                correct = (predictions == targets).float()
                return correct.mean().item() * 2 - 1  # Scale to [-1, 1]
            
            # For regression tasks or if loss is available
            if hasattr(outputs, 'loss'):
                # Negative loss as reward
                loss_val = outputs.loss.item()
                # Scale based on history
                if self.reward_history:
                    mean_loss = np.mean([r for r in self.reward_history if r < 0])
                    std_loss = np.std([r for r in self.reward_history if r < 0]) + 1e-8
                    scaled_loss = (loss_val - mean_loss) / std_loss
                    return -np.clip(scaled_loss, -1.0, 1.0)
                return -np.clip(loss_val, 0, 1)
            
            # Fallback
            return 0.0
        
        return default_reward_fn if self.reward_fn is None else self.reward_fn
    
    def reset_plasticity(self) -> None:
        """Reset plasticity values to baseline."""
        for group_name, params in self.param_groups:
            baseline = self.baseline_plasticities.get(group_name, 1.0)
            
            for param in params:
                if param in self.optimizer.state:
                    state = self.optimizer.state[param]
                    if 'plasticity_factor' in state:
                        state['plasticity_factor'] = baseline
    
    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics."""
        group_plasticities = self.get_group_plasticities()
        
        # Compute overall plasticity metrics
        all_plasticities = list(group_plasticities.values())
        
        # Compute reward statistics
        reward_stats = {}
        if self.reward_history:
            reward_stats = {
                'mean_reward': np.mean(self.reward_history),
                'min_reward': np.min(self.reward_history),
                'max_reward': np.max(self.reward_history),
                'std_reward': np.std(self.reward_history)
            }
        
        return {
            'group_plasticities': group_plasticities,
            'mean_plasticity': np.mean(all_plasticities) if all_plasticities else 0.0,
            'min_plasticity': np.min(all_plasticities) if all_plasticities else 0.0,
            'max_plasticity': np.max(all_plasticities) if all_plasticities else 0.0,
            'reward_stats': reward_stats,
            'update_rate': self.update_rate,
            'reward_scaling': self.reward_scaling
        }
            