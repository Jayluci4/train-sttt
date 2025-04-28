import torch
import torch.nn as nn
import logging
import math
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import numpy as np
from collections import defaultdict, deque
import scipy.stats as stats

logger = logging.getLogger(__name__)

class ConvergentNeuralArchitectureSearch:
    """
    Implements Convergent Neural Architecture Search (C-NAS) with Thompson sampling
    and bounded error guarantees for architecture morphing.
    
    This component formulates architecture search as a structured bandit problem:
        max_a E[r_t(a)]  subject to  sum_t ||T_a(θ_t) - θ_t|| < B
    
    Where:
        - r_t(a) is the expected reward for architecture action a
        - T_a(θ_t) is the transformation operator applied to parameters θ
        - B is the total error budget for convergence guarantees
    """
    def __init__(
        self, 
        model,
        action_space: Dict[str, List[Any]],
        reward_fn: Optional[Callable] = None,
        error_budget: float = 1.0,
        exploration_factor: float = 0.1,
        prior_strength: float = 2.0,
        min_samples: int = 5,
        warmup_steps: int = 100,
        morph_interval: int = 50,
        horizon: int = 3,
        convergence_factor: float = 1.1,  # For error decay: 1/t^convergence_factor
        optimizer: Optional[Any] = None
    ):
        """
        Initialize the Convergent Neural Architecture Search component.
        
        Args:
            model: The neural network model
            action_space: Dictionary of architecture action types and their possible values
            reward_fn: Function to compute reward for an architecture (default: loss reduction)
            error_budget: Total error budget for morphing
            exploration_factor: Controls exploration vs exploitation (higher = more exploration)
            prior_strength: Strength of prior beliefs (higher = stronger priors)
            min_samples: Minimum samples before using Thompson sampling
            warmup_steps: Number of steps before starting architecture search
            morph_interval: Minimum interval between architecture morphs
            horizon: Planning horizon for reward estimation
            convergence_factor: Factor for error decay schedule (> 1 for convergence)
            optimizer: Optional optimizer for parameter group repair
        """
        self.model = model
        self.action_space = action_space
        self.reward_fn = reward_fn
        self.error_budget = error_budget
        self.exploration_factor = exploration_factor
        self.prior_strength = prior_strength
        self.min_samples = min_samples
        self.warmup_steps = warmup_steps
        self.morph_interval = morph_interval
        self.horizon = horizon
        self.convergence_factor = convergence_factor
        self.optimizer = optimizer
        
        # Initialize action statistics
        self.action_stats = {}
        self._initialize_action_stats()
        
        # History tracking
        self.reward_history = []
        self.action_history = []
        self.error_history = []
        self.cumulative_error = 0.0
        
        # State tracking
        self.step_count = 0
        self.last_morph_step = 0
        self.current_architecture = self._get_current_architecture()
        
        # Estimated reward model
        self.reward_model = {}
        
        logger.info(f"Initialized Convergent Neural Architecture Search with {len(self.action_space)} action dimensions")
        logger.info(f"Action space: {self.action_space}")
    
    def _initialize_action_stats(self):
        """Initialize statistics for each action in the action space."""
        for action_type, values in self.action_space.items():
            self.action_stats[action_type] = {}
            
            for value in values:
                # Initialize with Beta prior for Thompson sampling
                # (alpha, beta) parameters for Beta distribution
                # Higher alpha/beta = stronger prior
                self.action_stats[action_type][value] = {
                    'alpha': self.prior_strength,  # Prior successes
                    'beta': self.prior_strength,   # Prior failures
                    'count': 0,                    # Number of times tried
                    'rewards': [],                 # History of rewards
                    'errors': []                   # History of errors
                }
    
    def _get_current_architecture(self) -> Dict[str, Any]:
        """Get the current architecture configuration."""
        current_arch = {}
        
        # Extract current architecture for each action type
        for action_type in self.action_space.keys():
            # This needs to be customized based on the architecture aspects being searched
            if action_type == 'lora_rank':
                # Find LoRA modules and get their ranks
                lora_ranks = {}
                for name, module in self.model.named_modules():
                    # Check if lora_A exists and is a ModuleDict with a 'default' key
                    if hasattr(module, 'lora_A') and isinstance(module.lora_A, nn.ModuleDict) and 'default' in module.lora_A:
                        # Access the 'default' adapter's weight
                        lora_ranks[name] = module.lora_A['default'].weight.shape[0]
                    elif hasattr(module, 'lora_A') and isinstance(module.lora_A, nn.Linear):
                         # Handle cases where lora_A might be a direct Linear layer (less common with PEFT)
                         lora_ranks[name] = module.lora_A.weight.shape[0]
                    # --- MODIFICATION END ---
                    # if hasattr(module, 'lora_A'):
                    #     lora_ranks[name] = module.lora_A.weight.shape[0]
                
                if lora_ranks:
                    # Use average rank as representative value
                    current_arch[action_type] = sum(lora_ranks.values()) / len(lora_ranks)
                else:
                    # Default value if no LoRA modules found
                    current_arch[action_type] = self.action_space[action_type][0]
            
            elif action_type == 'hidden_dropout':
                # Find dropout layers and get their rates
                dropout_rates = []
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Dropout):
                        dropout_rates.append(module.p)
                
                if dropout_rates:
                    current_arch[action_type] = sum(dropout_rates) / len(dropout_rates)
                else:
                    current_arch[action_type] = self.action_space[action_type][0]
            
            # Add more action types as needed for your architecture search
            else:
                # Default: use first value as placeholder
                current_arch[action_type] = self.action_space[action_type][0]
        
        return current_arch
    
    def update(self, step: int, reward: float) -> None:
        """
        Update action statistics based on reward.
        
        Args:
            step: Current training step
            reward: Reward value (higher is better)
        """
        self.step_count = step
        
        # Skip during warmup
        if step < self.warmup_steps:
            return
        
        # Record reward for current architecture
        self.reward_history.append((step, reward))
        
        # Update action statistics for current architecture
        for action_type, value in self.current_architecture.items():
            # Find closest value in action space
            closest_value = self._find_closest_value(action_type, value)
            
            if closest_value is not None:
                stats = self.action_stats[action_type][closest_value]
                
                # Update counts
                stats['count'] += 1
                stats['rewards'].append(reward)
                
                # Update Beta distribution parameters based on reward
                # Normalize reward to [0, 1] for Beta distribution
                normalized_reward = self._normalize_reward(reward)
                
                # Update success (alpha) and failure (beta) parameters
                # Higher reward = more success
                stats['alpha'] += normalized_reward
                stats['beta'] += (1.0 - normalized_reward)
    
    def _normalize_reward(self, reward: float) -> float:
        """
        Normalize reward to [0, 1] range for Beta distribution.
        
        Args:
            reward: Raw reward value
            
        Returns:
            Normalized reward in [0, 1]
        """
        # Use reward history to normalize
        if len(self.reward_history) >= 10:
            recent_rewards = [r for _, r in self.reward_history[-10:]]
            min_reward = min(recent_rewards)
            max_reward = max(recent_rewards)
            
            # Avoid division by zero
            if max_reward > min_reward:
                normalized = (reward - min_reward) / (max_reward - min_reward)
                return max(0.0, min(1.0, normalized))
        
        # Default normalization
        # Assuming reward is negative loss or accuracy-like metric
        return max(0.0, min(1.0, (reward + 1.0) / 2.0))
    
    def _find_closest_value(self, action_type: str, value: Any) -> Optional[Any]:
        """Find closest value in action space for continuous action types."""
        if action_type not in self.action_space:
            return None
            
        values = self.action_space[action_type]
        
        if isinstance(value, (int, float)) and all(isinstance(v, (int, float)) for v in values):
            # For numerical values, find closest
            return min(values, key=lambda v: abs(v - value))
        else:
            # For categorical values, exact match
            return value if value in values else values[0]
    
    def should_morph_architecture(self) -> bool:
        """
        Determine if architecture should be morphed at this step.
        
        Returns:
            True if architecture should be morphed
        """
        # Skip during warmup
        if self.step_count < self.warmup_steps:
            return False
            
        # Check interval between morphs
        steps_since_last_morph = self.step_count - self.last_morph_step
        if steps_since_last_morph < self.morph_interval:
            return False
        
        # Check if we have enough samples for meaningful Thompson sampling
        for action_type, stats_dict in self.action_stats.items():
            has_sufficient_samples = any(
                stats['count'] >= self.min_samples 
                for stats in stats_dict.values()
            )
            
            if has_sufficient_samples:
                return True
        
        # Default: morph at regular intervals after warmup
        return steps_since_last_morph >= self.morph_interval * 2
    
    def sample_next_architecture(self) -> Dict[str, Any]:
        """
        Sample next architecture using Thompson sampling with warm-start.
        
        Returns:
            Dictionary of architecture action values
        """
        next_arch = {}
        
        # Sample from posterior for each action type
        for action_type, values_dict in self.action_stats.items():
            # Extract values and compute their sampling weights
            values = list(values_dict.keys())
            
            # Use Thompson sampling (sample from Beta distribution)
            samples = []
            for value in values:
                stats = values_dict[value]
                alpha = stats['alpha']
                beta = stats['beta']
                
                # Sample from Beta
                sample = np.random.beta(alpha, beta)
                
                # Apply warm-start strategy for less-explored actions
                # This balances exploration and exploitation more effectively
                if stats['count'] < self.min_samples:
                    # Progressive exploration bonus that decreases with sample count
                    # but increases with the difference between count and min_samples
                    exploration_bonus = self.exploration_factor * (
                        (self.min_samples - stats['count']) / self.min_samples
                    )
                    
                    # Apply UCB-like bonus based on uncertainty
                    uncertainty = alpha / (alpha + beta) * (1 - alpha / (alpha + beta))
                    uncertainty_bonus = np.sqrt(uncertainty * 2 * np.log(sum(
                        [s['count'] for s in values_dict.values()]) + 1))
                    
                    # Combined bonus with progressive reduction
                    sample += exploration_bonus * uncertainty_bonus
                
                samples.append(sample)
            
            # Choose action with highest sampled value
            best_idx = np.argmax(samples)
            next_arch[action_type] = values[best_idx]
        
        logger.info(f"Thompson sampling selected architecture: {next_arch}")
        return next_arch
    
    def morph_architecture(self) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Apply architecture morphing using Thompson sampling while respecting error budget.
        
        Returns:
            Tuple of (modifications, error) where modifications is a dictionary of
            architecture changes and error is the transformation error
        """
        if not self.should_morph_architecture():
            return None, 0.0
            
        # Sample next architecture
        next_arch = self.sample_next_architecture()
        
        # Calculate expected error for this transformation
        expected_error = self._estimate_transformation_error(next_arch)
        
        # Calculate step-specific error budget based on convergence schedule
        # Using harmonic series for bounded total error
        step_error_budget = self.error_budget / ((1 + self.step_count) ** self.convergence_factor)
        
        # Check against budget
        if expected_error > step_error_budget:
            logger.warning(f"Expected error {expected_error:.4e} exceeds budget {step_error_budget:.4e}, aborting morph")
            return None, 0.0
        
        # Apply the architecture changes
        actual_modifications, actual_error = self._apply_architecture_changes(next_arch)
        
        if actual_modifications:
            # Update state tracking
            self.last_morph_step = self.step_count
            self.current_architecture = {**self.current_architecture, **actual_modifications}
            
            # Record error
            self.cumulative_error += actual_error
            self.error_history.append((self.step_count, actual_error, self.cumulative_error))
            
            # Record action taken
            self.action_history.append({
                'step': self.step_count,
                'architecture': self.current_architecture,
                'error': actual_error
            })
            
            logger.info(f"Architecture morphed at step {self.step_count}. "
                       f"Changes: {actual_modifications}, "
                       f"Error: {actual_error:.4e}, "
                       f"Cumulative: {self.cumulative_error:.4e}")
            
            return actual_modifications, actual_error
        
        return None, 0.0
    
    def _estimate_transformation_error(self, new_arch: Dict[str, Any]) -> float:
        """
        Estimate transformation error for proposed architecture change.
        
        Args:
            new_arch: Proposed new architecture
            
        Returns:
            Estimated error norm
        """
        # Start with base error estimation
        base_error = 0.1  # Default base error
        
        # Add error terms for each changed dimension
        for action_type, new_value in new_arch.items():
            current_value = self.current_architecture.get(action_type, new_value)
            
            # Skip if no change
            if new_value == current_value:
                continue
                
            if action_type == 'lora_rank':
                # Error scales with rank change magnitude and current step
                rank_diff = abs(new_value - current_value)
                # Error decreases with step count (model is more stable later in training)
                step_factor = 1.0 / math.sqrt(1 + self.step_count / 1000.0)
                rank_error = 0.1 * rank_diff * step_factor
                base_error += rank_error
            
            elif action_type == 'hidden_dropout':
                # Error scales with dropout rate change
                dropout_diff = abs(new_value - current_value)
                base_error += 0.05 * dropout_diff
            
            # Add more action types as needed
        
        return base_error
    
    def _apply_architecture_changes(self, new_arch: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """
        Apply architecture changes to the model.
        
        Args:
            new_arch: New architecture configuration
            
        Returns:
            Tuple of (actual modifications, transformation error)
        """
        # Track actual modifications and error
        actual_modifications = {}
        total_error = 0.0
        
        # Apply changes for each action type
        for action_type, new_value in new_arch.items():
            current_value = self.current_architecture.get(action_type, new_value)
            
            # Skip if no change
            if new_value == current_value:
                continue
            
            # Different handling based on action type
            if action_type == 'lora_rank':
                # Change LoRA ranks
                error = self._change_lora_ranks(new_value)
                if error > 0:
                    actual_modifications[action_type] = new_value
                    total_error += error
            
            elif action_type == 'hidden_dropout':
                # Change dropout rates
                error = self._change_dropout_rates(new_value)
                if error > 0:
                    actual_modifications[action_type] = new_value
                    total_error += error
            
            # Add more action types as needed
        
        return actual_modifications, total_error
    
    def _change_lora_ranks(self, new_rank: int) -> float:
        """
        Change LoRA ranks throughout model.
        
        Args:
            new_rank: New LoRA rank
            
        Returns:
            Error norm
        """
        total_error = 0.0
        modules_changed = 0
        
        # Find all LoRA modules
        lora_modules = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                lora_modules.append((name, module))
        
        if not lora_modules:
            logger.warning("No LoRA modules found to change ranks")
            return 0.0
        
        # Change rank for each module
        for name, module in lora_modules:
            try:
                # Get current rank
                current_rank = module.lora_A.weight.shape[0]
                
                # Skip if already at target
                if current_rank == new_rank:
                    continue
                
                # Get current weights
                current_A = module.lora_A.weight.detach().clone()  # [r, in_features]
                current_B = module.lora_B.weight.detach().clone()  # [out_features, r]
                
                # Save references to old parameters for optimizer state transfer
                old_params = [module.lora_A.weight, module.lora_B.weight]
                
                # Get device and dtype
                device = current_A.device
                dtype = current_A.dtype
                
                # Create new Linear modules with appropriate device & dtype
                in_features = module.lora_A.in_features
                out_features = module.lora_B.out_features
                
                new_lora_A = nn.Linear(in_features, new_rank, bias=False, device=device, dtype=dtype)
                new_lora_B = nn.Linear(new_rank, out_features, bias=False, device=device, dtype=dtype)
                
                # Handle rank increase
                if new_rank > current_rank:
                    # Use SVD-based expansion for optimal subspace preservation
                    try:
                        # Compute AB decomposition
                        AB = current_B @ current_A  # [out_features, in_features]
                        
                        # Compute SVD
                        U, S, Vh = torch.linalg.svd(AB, full_matrices=False)
                        
                        # Calculate new weights with preserved subspace
                        S_sqrt = torch.sqrt(S)
                        
                        # Initialize new weights, preserving existing subspace
                        new_lora_A.weight[:current_rank, :] = (
                            torch.diag(S_sqrt[:current_rank]) @ Vh[:current_rank, :]
                        ).to(device=device, dtype=dtype)
                        new_lora_B.weight[:, :current_rank] = (
                            U[:, :current_rank] @ torch.diag(S_sqrt[:current_rank])
                        ).to(device=device, dtype=dtype)
                        
                        # Initialize new dimensions with orthogonal initialization
                        if new_rank > current_rank:
                            # Calculate scaling factor for orthogonal initialization
                            fan_in = in_features
                            gain = 1.0 / math.sqrt(fan_in)
                            
                            # Initialize new rows of A with orthogonal values
                            nn.init.orthogonal_(new_lora_A.weight[current_rank:, :])
                            new_lora_A.weight[current_rank:, :].mul_(gain)
                            
                            # Initialize new columns of B with zeros (non-expansive)
                            new_lora_B.weight[:, current_rank:].zero_()
                    except Exception as e:
                        # If SVD fails, fall back to simpler initialization
                        logger.warning(f"SVD expansion failed for {name}, falling back to zero-init: {e}")
                        
                        # Copy existing weights
                        new_lora_A.weight[:current_rank, :] = current_A
                        new_lora_B.weight[:, :current_rank] = current_B
                        
                        # Zero-initialize new dimensions (strictly non-expansive)
                        new_lora_A.weight[current_rank:, :].zero_()
                        new_lora_B.weight[:, current_rank:].zero_()
                
                # Handle rank decrease
                else:
                    # Using SVD for optimal compression
                    try:
                        # Compute AB decomposition
                        AB = current_B @ current_A  # [out_features, in_features]
                        
                        # Compute SVD
                        U, S, Vh = torch.linalg.svd(AB, full_matrices=False)
                        
                        # Use top singular values/vectors for reduction
                        S_sqrt = torch.sqrt(S[:new_rank])
                        
                        # Set the new weights based on truncated SVD
                        new_lora_A.weight = nn.Parameter((torch.diag(S_sqrt) @ Vh[:new_rank, :]).to(device=device, dtype=dtype))
                        new_lora_B.weight = nn.Parameter((U[:, :new_rank] @ torch.diag(S_sqrt)).to(device=device, dtype=dtype))
                    
                    except Exception as e:
                        # If SVD fails, use simple slicing
                        logger.warning(f"SVD reduction failed for {name}, falling back to slicing: {e}")
                        new_lora_A.weight = nn.Parameter(current_A[:new_rank, :].clone())
                        new_lora_B.weight = nn.Parameter(current_B[:, :new_rank].clone())
                
                # Record requires_grad state
                requires_grad_A = old_params[0].requires_grad
                requires_grad_B = old_params[1].requires_grad
                
                # Set requires_grad to match original
                new_lora_A.weight.requires_grad = requires_grad_A
                new_lora_B.weight.requires_grad = requires_grad_B
                
                # New parameters for parameter group repair
                new_params = [new_lora_A.weight, new_lora_B.weight]
                
                # Transfer optimizer state if optimizer is provided (CRITICAL)
                if self.optimizer is not None:
                    try:
                        self._repair_optimizer_state(old_params, new_params, self.optimizer)
                    except Exception as e:
                        logger.warning(f"Error repairing optimizer state: {e}")
                
                # Update the module
                module.lora_A = new_lora_A
                module.lora_B = new_lora_B
                
                # Update scaling factor if needed
                if hasattr(module, 'scaling'):
                    # Get LoRA alpha if available
                    lora_alpha = getattr(module, 'lora_alpha', current_rank)
                    module.scaling = lora_alpha / new_rank
                
                # Calculate error norm for this module
                old_params_vector = torch.cat([p.data.flatten() for p in old_params])
                new_params_vector = torch.cat([p.data.flatten() for p in new_params])
                
                # Handle dimension mismatch
                if old_params_vector.shape != new_params_vector.shape:
                    min_length = min(old_params_vector.shape[0], new_params_vector.shape[0])
                    error = torch.norm(old_params_vector[:min_length] - new_params_vector[:min_length]).item()
                    # Add error from the dimension change itself
                    if old_params_vector.shape[0] < new_params_vector.shape[0]:
                        error += torch.norm(new_params_vector[min_length:]).item()
                    else:
                        error += torch.norm(old_params_vector[min_length:]).item()
                else:
                    error = torch.norm(old_params_vector - new_params_vector).item()
                
                total_error += error
                modules_changed += 1
                logger.info(f"Changed LoRA rank for {name} from {current_rank} to {new_rank}, error: {error:.4e}")
                
            except Exception as e:
                logger.error(f"Error changing LoRA rank for {name}: {e}")
        
        if modules_changed > 0:
            logger.info(f"Changed rank for {modules_changed} LoRA modules to {new_rank}, total error: {total_error:.4e}")
        
        return total_error
    
    def _change_dropout_rates(self, new_rate: float) -> float:
        """
        Change dropout rates throughout model.
        
        Args:
            new_rate: New dropout rate
            
        Returns:
            Error norm (always 0 for dropout as it doesn't affect parameters)
        """
        modules_changed = 0
        
        # Find all dropout modules
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Dropout):
                old_rate = module.p
                module.p = new_rate
                modules_changed += 1
                logger.info(f"Changed dropout rate for {name} from {old_rate} to {new_rate}")
        
        if modules_changed > 0:
            logger.info(f"Changed dropout rate for {modules_changed} modules to {new_rate}")
        
        # Changing dropout doesn't affect parameters, so error is 0
        return 0.0
    
    def _repair_optimizer_state(self, old_params, new_params, optimizer):
        """
        Repair optimizer state after parameter replacement.
        
        Args:
            old_params: List of old parameters
            new_params: List of new parameters
            optimizer: The optimizer
        """
        # Repair parameter groups
        for group in optimizer.param_groups:
            # Find and remove old parameters
            old_param_indices = []
            for i, p in enumerate(group['params']):
                if any(p is old_param for old_param in old_params):
                    old_param_indices.append(i)
            
            # Remove old parameters (in reverse order to avoid index issues)
            for i in sorted(old_param_indices, reverse=True):
                del group['params'][i]
            
            # Add new parameters
            group['params'].extend(new_params)
        
        # Transfer optimizer state
        if hasattr(optimizer, 'state'):
            opt_state = optimizer.state
            
            for old_p, new_p in zip(old_params, new_params):
                if old_p in opt_state:
                    # Create a new state dictionary for the new parameter
                    new_state = {}
                    
                    # Clone transferable state values
                    for k, v in opt_state[old_p].items():
                        if isinstance(v, torch.Tensor):
                            # For tensor states (like momentum buffers)
                            if v.dim() > 0 and v.size(0) == old_p.size(0):
                                # Resize if dimensions changed (e.g., for exp_avg in Adam)
                                if old_p.size(0) != new_p.size(0):
                                    if old_p.size(0) < new_p.size(0):
                                        # Expanding - create new tensor with appropriate shape
                                        resized_v = torch.zeros_like(new_p)
                                        # Copy values from the smaller tensor
                                        if v.dim() == 1:
                                            resized_v[:old_p.size(0)] = v
                                        elif v.dim() == 2:
                                            resized_v[:old_p.size(0), :] = v
                                        new_state[k] = resized_v
                                    else:
                                        # Shrinking - slice the tensor
                                        if v.dim() == 1:
                                            new_state[k] = v[:new_p.size(0)].clone()
                                        elif v.dim() == 2:
                                            new_state[k] = v[:new_p.size(0), :].clone()
                                else:
                                    # Same size, just clone
                                    new_state[k] = v.clone()
                            else:
                                # For other tensor states (like step counters)
                                new_state[k] = v.clone()
                        else:
                            # For non-tensor states (like step counters)
                            new_state[k] = v
                    
                    # Apply the state to the new parameter
                    opt_state[new_p] = new_state
                    
                    # Optional: Scale momentum if rank changed
                    if 'exp_avg' in new_state and old_p.size(0) != new_p.size(0):
                        scale_factor = math.sqrt(old_p.size(0) / new_p.size(0))
                        new_state['exp_avg'].mul_(scale_factor)
    
    def estimate_reward(self, architecture: Dict[str, Any]) -> Tuple[float, float]:
        """
        Estimate expected reward for a given architecture.
        
        Args:
            architecture: Architecture configuration
            
        Returns:
            Tuple of (estimated reward, standard deviation)
        """
        rewards = []
        weights = []
        
        # For each action type and value, gather rewards
        for action_type, value in architecture.items():
            closest_value = self._find_closest_value(action_type, value)
            
            if closest_value is not None:
                stats = self.action_stats[action_type][closest_value]
                
                if stats['rewards']:
                    # Compute weighted average reward
                    # More recent rewards get higher weight
                    recent_rewards = stats['rewards'][-self.horizon:]
                    if recent_rewards:
                        # Exponential weights: more recent = higher weight
                        exp_weights = [math.exp(i/self.horizon) for i in range(len(recent_rewards))]
                        
                        # Normalize weights
                        sum_weights = sum(exp_weights)
                        norm_weights = [w / sum_weights for w in exp_weights]
                        
                        # Weighted average
                        weighted_reward = sum(r * w for r, w in zip(recent_rewards, norm_weights))
                        rewards.append(weighted_reward)
                        
                        # Weight by sample count (more samples = more confidence)
                        sample_weight = min(1.0, stats['count'] / self.min_samples)
                        weights.append(sample_weight)
        
        if not rewards:
            # No reward data available, return default
            return 0.0, 1.0
        
        # Calculate weighted average across all action dimensions
        if not weights:
            weights = [1.0] * len(rewards)
        
        # Normalize weights
        sum_weights = sum(weights)
        if sum_weights > 0:
            norm_weights = [w / sum_weights for w in weights]
        else:
            norm_weights = [1.0 / len(weights)] * len(weights)
        
        # Weighted average reward
        avg_reward = sum(r * w for r, w in zip(rewards, norm_weights))
        
        # Calculate weighted standard deviation
        if len(rewards) > 1:
            variance = sum(w * ((r - avg_reward) ** 2) for r, w in zip(rewards, norm_weights))
            std_dev = math.sqrt(variance)
        else:
            std_dev = 0.5  # Default uncertainty
        
        return avg_reward, std_dev
    
    def get_action_stats(self) -> Dict[str, Any]:
        """Get statistics about action sampling."""
        stats = {}
        
        for action_type, values_dict in self.action_stats.items():
            action_stats = {}
            
            for value, value_stats in values_dict.items():
                # Calculate MAP estimate from Beta distribution
                alpha = value_stats['alpha']
                beta = value_stats['beta']
                
                # Mode of Beta distribution: (alpha - 1) / (alpha + beta - 2) for alpha, beta > 1
                if alpha > 1 and beta > 1:
                    map_estimate = (alpha - 1) / (alpha + beta - 2)
                else:
                    map_estimate = alpha / (alpha + beta)
                
                # Calculate 95% credible interval
                if alpha > 0 and beta > 0:
                    lower = stats.beta.ppf(0.025, alpha, beta)
                    upper = stats.beta.ppf(0.975, alpha, beta)
                else:
                    lower, upper = 0.0, 1.0
                
                action_stats[value] = {
                    'count': value_stats['count'],
                    'map_estimate': map_estimate,
                    'uncertainty': upper - lower,
                    'credible_interval': (lower, upper)
                }
            
            stats[action_type] = action_stats
        
        return stats
    
    def get_best_architecture(self) -> Dict[str, Any]:
        """
        Get the best architecture based on current knowledge.
        
        Returns:
            Dictionary of best architecture configuration
        """
        best_arch = {}
        
        # For each action type, select the value with highest MAP estimate
        for action_type, values_dict in self.action_stats.items():
            best_value = None
            best_estimate = -float('inf')
            
            for value, value_stats in values_dict.items():
                # Calculate MAP estimate from Beta distribution
                alpha = value_stats['alpha']
                beta = value_stats['beta']
                
                # Mode of Beta distribution
                if alpha > 1 and beta > 1:
                    map_estimate = (alpha - 1) / (alpha + beta - 2)
                else:
                    map_estimate = alpha / (alpha + beta)
                
                # Keep track of best value
                if map_estimate > best_estimate:
                    best_estimate = map_estimate
                    best_value = value
            
            if best_value is not None:
                best_arch[action_type] = best_value
        
        return best_arch
    
    def get_architecture_exploration_report(self) -> Dict[str, Any]:
        """
        Generate a report on architecture exploration and search progress.
        
        Returns:
            Dictionary containing exploration metrics and best architecture
        """
        # Get action statistics
        action_stats = self.get_action_stats()
        
        # Get best architecture
        best_arch = self.get_best_architecture()
        
        # Calculate exploration progress for each action dimension
        exploration_progress = {}
        for action_type, values_dict in self.action_stats.items():
            # Count explored values
            explored = sum(1 for stats in values_dict.values() if stats['count'] >= self.min_samples)
            total = len(values_dict)
            
            exploration_progress[action_type] = {
                'explored': explored,
                'total': total,
                'progress': explored / total if total > 0 else 0.0
            }
        
        # Calculate overall exploration progress
        total_explored = sum(prog['explored'] for prog in exploration_progress.values())
        total_actions = sum(prog['total'] for prog in exploration_progress.values())
        overall_progress = total_explored / total_actions if total_actions > 0 else 0.0
        
        # Get exploitation metrics
        if len(self.reward_history) >= 10:
            recent_rewards = [r for _, r in self.reward_history[-10:]]
            reward_trend = np.mean(recent_rewards[-5:]) - np.mean(recent_rewards[:5])
        else:
            reward_trend = 0.0
        
        # Compute confidence in current best architecture
        confidence = {}
        for action_type, value in best_arch.items():
            if action_type in action_stats and value in action_stats[action_type]:
                confidence[action_type] = 1.0 - action_stats[action_type][value].get('uncertainty', 1.0)
        
        avg_confidence = np.mean(list(confidence.values())) if confidence else 0.0
        
        return {
            'best_architecture': best_arch,
            'exploration_progress': exploration_progress,
            'overall_exploration': overall_progress,
            'reward_trend': reward_trend,
            'confidence': confidence,
            'average_confidence': avg_confidence,
            'error_budget_remaining': self.error_budget - self.cumulative_error,
            'morph_count': len(self.action_history),
            'architecture_modifications': self.action_history[-5:] if self.action_history else []
        }
    
    def perform_targeted_exploration(self, action_type: str, values: Optional[List[Any]] = None) -> None:
        """
        Actively explore specific action dimensions to gather more information.
        
        Args:
            action_type: Action dimension to explore
            values: Specific values to explore (if None, selects least explored values)
        """
        if action_type not in self.action_space:
            logger.warning(f"Action type '{action_type}' not in action space")
            return
            
        # If no specific values provided, select least explored values
        if values is None:
            values_dict = self.action_stats[action_type]
            values = sorted(
                values_dict.keys(), 
                key=lambda v: values_dict[v]['count']
            )[:2]  # Select 2 least explored values
        
        # For each value, set up targeted exploration
        for value in values:
            if value not in self.action_space[action_type]:
                logger.warning(f"Value {value} not in action space for {action_type}")
                continue
                
            # Modify exploration bonus for this value
            stats = self.action_stats[action_type][value]
            
            # Add exploration bonus by resetting count and adding uncertainty
            # This makes the value more likely to be selected in Thompson sampling
            bonus_factor = 2.0
            stats['alpha'] = stats['alpha'] / bonus_factor
            stats['beta'] = stats['beta'] / bonus_factor
            
            logger.info(f"Added exploration bonus for {action_type}={value}")
        
        # Force a morph on next check
        self.last_morph_step = self.step_count - self.morph_interval
    
    def reset_error_budget(self, new_budget: Optional[float] = None) -> None:
        """
        Reset the error budget, optionally to a new value.
        
        Args:
            new_budget: New error budget (if None, resets to original value)
        """
        if new_budget is not None:
            self.error_budget = new_budget
        
        # Reset cumulative error
        self.cumulative_error = 0.0
        
        # Record reset in history
        self.error_history.append((self.step_count, 0.0, 0.0))
        
        logger.info(f"Reset error budget to {self.error_budget}")
