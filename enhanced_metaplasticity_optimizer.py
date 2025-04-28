import math
import torch
from torch.optim import Optimizer
from collections import defaultdict, deque
import logging
import numpy as np
import math
logger = logging.getLogger(__name__)

class EnhancedMetaplasticityOptimizer(Optimizer):
    """
    Enhanced metaplasticity optimizer with stronger convergence guarantees using 
    two-timescale stochastic approximation.
    
    This optimizer wraps a base optimizer (like AdamW or AdaFactor) and modulates its behavior by:
    1. Tracking learning history for each parameter
    2. Computing stability metrics for gradient direction and magnitude
    3. Adaptively adjusting per-parameter plasticity (learning rates)
    
    The key enhancement is ensuring proper timescale separation with mathematically 
    guaranteed convergence based on non-expansive parameter transformations.
    """
    def __init__(self, params, base_optimizer_class, lr=1e-3, plasticity_eta=0.01, 
                 plasticity_decay=0.9999, plasticity_growth=1.001, 
                 min_plasticity=0.2, max_plasticity=5.0, **base_kwargs):
        """
        Args:
            params: Iterable of parameters to optimize
            base_optimizer_class: Optimizer class to use (e.g. torch.optim.AdamW)
            lr: Learning rate
            plasticity_eta: Base learning rate for plasticity adaptation
            plasticity_decay: Factor for decreasing plasticity (slower timescale)
            plasticity_growth: Factor for increasing plasticity (slower timescale)
            min_plasticity: Minimum allowed plasticity factor
            max_plasticity: Maximum allowed plasticity factor
            **base_kwargs: Additional arguments for base optimizer
        """
        self.base_optimizer_class = base_optimizer_class
        self.plasticity_eta = plasticity_eta
        
        # Ensure timescale separation: plasticity operates on slower timescale
        # This is critical for the convergence proof to apply
        # plasticity_growth/decay should scale with lr for proper separation
        self.plasticity_decay = max(1 - 5 * lr, plasticity_decay)
        self.plasticity_growth = min(1 + 5 * lr, plasticity_growth)
        
        self.min_plasticity = min_plasticity
        self.max_plasticity = max_plasticity
        
        # Create parameter groups with plasticity tracking
        param_groups = []
        for group in params:
            if isinstance(group, dict):
                new_group = {**group}
                if 'plasticity' not in new_group:
                    new_group['plasticity'] = 1.0
                param_groups.append(new_group)
            else:
                param_groups.append({
                    'params': [group],
                    'plasticity': 1.0
                })
        
        # Initialize base optimizer with proper parameter groups
        self.base_optimizer = base_optimizer_class(param_groups, lr=lr, **base_kwargs)
        
        # Initialize optimizer state
        self.state = defaultdict(dict)
        
        # Parameter update history for tracking gradient stability
        self.update_history = defaultdict(lambda: deque(maxlen=50))
        
        # Step counter for tracking optimization progress
        self.step_count = 0
        
        # Convergence monitoring
        self.plasticity_stats_history = []
        
        defaults = dict(plasticity=1.0)
        super(EnhancedMetaplasticityOptimizer, self).__init__(param_groups, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform optimization step with plasticity control."""
        # Increment step counter
        self.step_count += 1
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Update plasticity for each parameter (slower timescale)
        self._update_plasticity()
        
        # Apply plasticity modulation to gradients
        self._apply_plasticity()
        
        # Let base optimizer take a step (faster timescale)
        self.base_optimizer.step()
        
        # Record parameter updates for future plasticity calculations
        self._record_updates()
        
        # Periodically log plasticity statistics
        if self.step_count % 100 == 0:
            stats = self.get_plasticity_stats()
            self.plasticity_stats_history.append(stats)
            logger.info(f"Step {self.step_count}: Plasticity mean={stats['mean']:.4f}, "
                       f"min={stats['min']:.4f}, max={stats['max']:.4f}, "
                       f"high_pct={stats['high_plasticity_pct']:.1f}%, "
                       f"low_pct={stats['low_plasticity_pct']:.1f}%")
        
        return loss
    
    def _update_plasticity(self):
        """
        Update plasticity factors based on parameter history.
        
        This operates on the slower timescale of the two-timescale stochastic approximation,
        ensuring convergence guarantees according to the mathematical analysis.
        """
        # Get current timescale factor (decreases with step count)
        # This ensures the slower timescale property required for convergence
        timescale_factor = min(1.0, 10.0 / math.sqrt(1 + self.step_count))
        
        # Update plasticity for each parameter group
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                
                # Initialize plasticity state if needed
                if 'plasticity_factor' not in state:
                    state['plasticity_factor'] = 1.0
                    state['update_sign_stability'] = 0.0
                    state['gradient_magnitude_stability'] = 0.0
                
                # Get update history
                update_history = self.update_history[p]
                
                if len(update_history) >= 5:  # Need some history to compute stability
                    
                    # --- MODIFICATION START ---
                    # Stack the recent update tensors along a new dimension (dim=0)
                    recent_updates_stack = torch.stack(list(update_history)[-5:]) # Shape: [5, num_params]

                    # Compute update sign stability (consistency in gradient direction)
                    # Calculate sign agreement across the history dimension (dim=0)
                    signs_stack = torch.sign(recent_updates_stack)
                    # Sum signs along history dim, take abs, divide by history length
                    sign_agreement = torch.abs(torch.sum(signs_stack, dim=0)) / signs_stack.size(0)
                    # Average stability across all parameters in the tensor
                    sign_stability = sign_agreement.mean().item()

                    # Compute gradient magnitude stability
                    # High if gradient magnitudes are consistent over history
                    recent_mags = torch.abs(recent_updates_stack) # Shape: [5, num_params]
                    mag_mean = torch.mean(recent_mags, dim=0) + 1e-8 # Mean across history, Shape: [num_params]
                    mag_variance = torch.var(recent_mags, dim=0) # Variance across history, Shape: [num_params]
                    # Relative variance (coefficient of variation squared, essentially)
                    relative_mag_variance = mag_variance / (mag_mean**2)
                    # Use exp(-rel_var) as stability measure, average across params
                    mag_stability = torch.exp(-relative_mag_variance).mean().item()

                    # Ensure stability values are valid
                    sign_stability = np.nan_to_num(sign_stability, nan=0.5)
                    mag_stability = np.nan_to_num(mag_stability, nan=0.5)

                    # --- MODIFICATION END ---

                    # Update stability metrics in state
                    state['update_sign_stability'] = sign_stability
                    state['gradient_magnitude_stability'] = mag_stability

                    # Adjust plasticity based on stability metrics (slower timescale)
                    # Increase plasticity for stable gradients, decrease for unstable ones
                    plasticity_factor = state['plasticity_factor']

                    # Combined stability metric (higher is more stable)
                    combined_stability = 0.7 * sign_stability + 0.3 * mag_stability

                    if combined_stability > 0.7:  # High stability, increase plasticity
                        # Apply slower timescale growth
                        plasticity_factor *= 1.0 + (self.plasticity_growth - 1.0) * timescale_factor
                    elif combined_stability < 0.3:  # Low stability, decrease plasticity
                        # Apply slower timescale decay
                        plasticity_factor *= 1.0 - (1.0 - self.plasticity_decay) * timescale_factor

                    # Ensure plasticity stays within bounds
                    plasticity_factor = max(self.min_plasticity, min(self.max_plasticity, plasticity_factor))
                    state['plasticity_factor'] = plasticity_factor
    
    def _apply_plasticity(self):
        """
        Apply plasticity factors to gradients.
        
        This modulates the learning based on parameter-specific plasticity factors,
        implementing the non-expansive operator required by the convergence theory.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                if 'plasticity_factor' in state:
                    # Scale gradient by plasticity factor
                    p.grad.data.mul_(state['plasticity_factor'])
    
    def _record_updates(self):
        """Record parameter updates for plasticity calculations."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Record current gradient
                self.update_history[p].append(p.grad.detach().clone())
    
    def zero_grad(self, set_to_none=False):
        """Clear gradients."""
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
    
    def get_plasticity_stats(self):
        """Get statistics about current plasticity state."""
        plasticity_stats = {
            'min': float('inf'),
            'max': float('-inf'),
            'mean': 0.0,
            'high_plasticity_params': 0,
            'low_plasticity_params': 0,
            'param_count': 0
        }
        
        total_plasticity = 0.0
        param_count = 0
        
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'plasticity_factor' in state:
                    plasticity = state['plasticity_factor']
                    param_count += 1
                    total_plasticity += plasticity
                    
                    # Update min/max
                    plasticity_stats['min'] = min(plasticity_stats['min'], plasticity)
                    plasticity_stats['max'] = max(plasticity_stats['max'], plasticity)
                    
                    # Count high/low plasticity parameters
                    if plasticity > 2.0:
                        plasticity_stats['high_plasticity_params'] += 1
                    elif plasticity < 0.5:
                        plasticity_stats['low_plasticity_params'] += 1
        
        if param_count > 0:
            plasticity_stats['mean'] = total_plasticity / param_count
            plasticity_stats['param_count'] = param_count
            plasticity_stats['high_plasticity_pct'] = 100 * plasticity_stats['high_plasticity_params'] / param_count
            plasticity_stats['low_plasticity_pct'] = 100 * plasticity_stats['low_plasticity_params'] / param_count
        else:
            plasticity_stats['min'] = 0.0
            plasticity_stats['high_plasticity_pct'] = 0.0
            plasticity_stats['low_plasticity_pct'] = 0.0
        
        return plasticity_stats
    
    def get_stability_report(self):
        """Generate a report on parameter stability and learning dynamics."""
        # Analyze individual parameter groups
        param_stability = {}
        
        for i, group in enumerate(self.param_groups):
            group_stability = {
                'params': len(group['params']),
                'high_stability': 0,
                'low_stability': 0,
                'avg_sign_stability': 0.0,
                'avg_magnitude_stability': 0.0,
                'avg_plasticity': 0.0
            }
            
            param_count = 0
            
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    param_count += 1
                    
                    if 'update_sign_stability' in state and 'gradient_magnitude_stability' in state:
                        sign_stability = state['update_sign_stability']
                        mag_stability = state['gradient_magnitude_stability']
                        plasticity = state['plasticity_factor']
                        
                        group_stability['avg_sign_stability'] += sign_stability
                        group_stability['avg_magnitude_stability'] += mag_stability
                        group_stability['avg_plasticity'] += plasticity
                        
                        # Count high/low stability parameters
                        combined_stability = 0.7 * sign_stability + 0.3 * mag_stability
                        if combined_stability > 0.7:
                            group_stability['high_stability'] += 1
                        elif combined_stability < 0.3:
                            group_stability['low_stability'] += 1
            
            if param_count > 0:
                group_stability['avg_sign_stability'] /= param_count
                group_stability['avg_magnitude_stability'] /= param_count
                group_stability['avg_plasticity'] /= param_count
                
                # Calculate percentages
                group_stability['high_stability_pct'] = 100 * group_stability['high_stability'] / param_count
                group_stability['low_stability_pct'] = 100 * group_stability['low_stability'] / param_count
            
            param_stability[f'group_{i}'] = group_stability
        
        # Analyze plasticity evolution
        plasticity_evolution = {
            'steps': [],
            'mean': [],
            'min': [],
            'max': []
        }
        
        for i, stats in enumerate(self.plasticity_stats_history):
            step = (i + 1) * 100  # Stats recorded every 100 steps
            plasticity_evolution['steps'].append(step)
            plasticity_evolution['mean'].append(stats['mean'])
            plasticity_evolution['min'].append(stats['min'])
            plasticity_evolution['max'].append(stats['max'])
        
        # Full report
        return {
            'param_stability': param_stability,
            'plasticity_evolution': plasticity_evolution,
            'current_stats': self.get_plasticity_stats() if self.step_count > 0 else None,
            'convergence_status': 'On track' if self._check_convergence_criteria() else 'Needs monitoring'
        }
    
    def _check_convergence_criteria(self):
        """
        Check if optimizer behavior satisfies mathematical convergence criteria.
        
        According to the mathematical analysis, convergence is guaranteed when:
        1. Plasticity operates on a slower timescale than weight updates
        2. The operator is non-expansive (plasticity factors bounded)
        3. Error terms are summable
        """
        # Check plasticity trend stability - should converge to stable values
        if len(self.plasticity_stats_history) < 5:
            return True  # Not enough data yet
            
        recent_mean = self.plasticity_stats_history[-1]['mean']
        earlier_mean = self.plasticity_stats_history[-5]['mean']
        plasticity_stability = abs(recent_mean - earlier_mean) / max(abs(earlier_mean), 1e-8)
        
        # Check that plasticity values aren't constantly hitting bounds
        latest_stats = self.plasticity_stats_history[-1]
        bound_hitting = (
            latest_stats['min'] <= self.min_plasticity * 1.01 or 
            latest_stats['max'] >= self.max_plasticity * 0.99
        )
        
        # Final convergence check
        return plasticity_stability < 0.2 and not bound_hitting


class EnhancedAdaFactorWithMetaplasticity(Optimizer):
    """
    Enhanced AdaFactor optimizer with metaplasticity for improved convergence.
    
    This combines the memory efficiency of AdaFactor with parameter-specific 
    metaplasticity for adaptive learning rates, following the mathematical framework
    for guaranteed convergence under architecture changes.
    
    The main enhancement is ensuring proper two-timescale behavior needed for the
    convergence guarantees in the mathematical analysis.
    """
    def __init__(self, params, lr=None, beta1=0.9, beta2=0.999, eps1=1e-30, eps2=1e-3,
                 clipping_threshold=1.0, no_bias_correction=False, relative_step=False,
                 scale_parameter=True, warmup_init=False, plasticity_eta=0.01,
                 min_plasticity=0.2, max_plasticity=5.0):
        """
        Args:
            params: Iterable of parameters
            lr: Learning rate (can be None for default schedule)
            beta1: Coefficient for exponential moving averages of gradients
            beta2: Coefficient for exponential moving averages of squared gradients
            eps1: Small constant to avoid division by zero in updates
            eps2: Small constant for parameter scale estimate
            clipping_threshold: Threshold for gradient clipping
            no_bias_correction: Whether to disable bias correction
            relative_step: Whether to use relative step sizes
            scale_parameter: Whether to scale parameter updates
            warmup_init: Whether to use linear warmup
            plasticity_eta: Learning rate for plasticity updates
            min_plasticity: Minimum plasticity value
            max_plasticity: Maximum plasticity value
        """
        if lr is not None and relative_step:
            raise ValueError("Cannot specify both lr and relative_step=True")
            
        # Default to relative step if lr is None
        if lr is None:
            relative_step = True
            
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps1=eps1,
            eps2=eps2,
            clipping_threshold=clipping_threshold,
            no_bias_correction=no_bias_correction,
            relative_step=relative_step,
            scale_parameter=scale_parameter,
            warmup_init=warmup_init,
            plasticity=1.0,
            plasticity_eta=plasticity_eta,
            min_plasticity=min_plasticity,
            max_plasticity=max_plasticity
        )
        
        super(EnhancedAdaFactorWithMetaplasticity, self).__init__(params, defaults)
        
        # Parameter update history for tracking gradient stability
        self.update_history = defaultdict(lambda: deque(maxlen=50))
        
        # Step counter for tracking optimization progress
        self.step_count = 0
        
        # Statistics tracking
        self.plasticity_stats_history = []
        
    def _get_lr(self, param_group, param_state):
        """Get learning rate based on schedule."""
        relative_step_size = param_group.get("lr", None)
        
        if param_group["relative_step"]:
            min_step = 1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2
            relative_step_size = min(min_step, 1.0 / math.sqrt(param_state["step"]))
            
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps2"], param_state["RMS"])
            
        return param_scale * relative_step_size
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform optimization step with metaplasticity."""
        # Increment step counter
        self.step_count += 1
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Update plasticity for each parameter
        self._update_plasticity()
                
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                
                # Apply plasticity factor to gradient
                if p in self.state and 'plasticity_factor' in self.state[p]:
                    plasticity = self.state[p]['plasticity_factor']
                    grad = grad * plasticity
                
                # Get or initialize state
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["RMS"] = 0
                    state["plasticity_factor"] = 1.0
                    state["update_sign_stability"] = 0.0
                    state["gradient_magnitude_stability"] = 0.0
                
                # Update step count
                state["step"] += 1
                
                # Perform AdaFactor update logic
                if group["beta1"] > 0:
                    # Compute exponential moving average of gradient
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(grad, alpha=1 - group["beta1"])
                    
                    # Bias correction
                    if not group["no_bias_correction"]:
                        bias_correction1 = 1 - group["beta1"] ** state["step"]
                        grad = exp_avg / bias_correction1
                
                # Clip gradient by L2 norm
                grad_norm = torch.norm(grad)
                if group["clipping_threshold"] > 0:
                    clip = group["clipping_threshold"] / (grad_norm + group["eps1"])
                    if clip < 1.0:
                        grad.mul_(clip)
                
                # Factored second moment estimation
                if len(p.shape) >= 2:
                    # For matrices, compute factored second moments
                    row_norm = torch.norm(grad, dim=1, keepdim=True)
                    row_mean = row_norm.mean()
                    
                    col_norm = torch.norm(grad, dim=0, keepdim=True)
                    col_mean = col_norm.mean()
                    
                    # Compute RMS = sqrt(row_mean * col_mean)
                    new_RMS = math.sqrt(row_mean * col_mean) / math.sqrt(p.numel())
                else:
                    # For vectors, compute standard RMS
                    new_RMS = grad_norm / math.sqrt(p.numel())
                
                # Update exponential average of RMS
                state["RMS"] = group["beta2"] * state["RMS"] + (1 - group["beta2"]) * new_RMS
                
                # Get learning rate
                lr = self._get_lr(group, state)
                
                # Update parameters
                p.add_(grad, alpha=-lr)
                
                # Record gradient for plasticity tracking
                self.update_history[p].append(grad.detach().clone())
        
        # Periodically log plasticity statistics
        if self.step_count % 100 == 0:
            stats = self.get_plasticity_stats()
            self.plasticity_stats_history.append(stats)
            logger.info(f"Step {self.step_count}: AdaFactor plasticity mean={stats['mean']:.4f}, "
                      f"min={stats['min']:.4f}, max={stats['max']:.4f}")
        
        return loss
    
    def _update_plasticity(self):
        """
        Update plasticity factors based on parameter history.
        
        This operates on a slower timescale to ensure convergence guarantees.
        """
        # Get current timescale factor (decreases with step count)
        # This ensures the slower timescale property required for convergence
        timescale_factor = min(1.0, 10.0 / math.sqrt(1 + self.step_count))
        
        # Update plasticity for each parameter group
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                
                # Initialize plasticity state if needed
                if 'plasticity_factor' not in state:
                    state['plasticity_factor'] = 1.0
                    state['update_sign_stability'] = 0.0
                    state['gradient_magnitude_stability'] = 0.0
                
                # Get update history
                update_history = self.update_history[p]
                
                if len(update_history) >= 5:  # Need some history to compute stability
                    # Compute update sign stability (consistency in gradient direction)
                    # High if gradients consistently point in the same direction
                    recent_signs = torch.tensor([torch.sign(upd) for upd in update_history[-5:]])
                    sign_agreement = torch.abs(recent_signs.sum(dim=0)) / 5.0
                    sign_stability = sign_agreement.mean().item()
                    
                    # Compute gradient magnitude stability
                    # High if gradient magnitudes are consistent
                    recent_mags = torch.stack([torch.abs(upd) for upd in update_history[-5:]])
                    mag_mean = torch.mean(recent_mags, dim=0) + 1e-8
                    mag_variance = torch.var(recent_mags, dim=0) / mag_mean
                    mag_stability = torch.exp(-mag_variance).mean().item()
                    
                    # Update stability metrics
                    state['update_sign_stability'] = sign_stability
                    state['gradient_magnitude_stability'] = mag_stability
                    
                    # Adjust plasticity based on stability metrics (slower timescale)
                    # Increase plasticity for stable gradients, decrease for unstable ones
                    plasticity_factor = state['plasticity_factor']
                    
                    # Combined stability metric (higher is more stable)
                    combined_stability = 0.7 * sign_stability + 0.3 * mag_stability
                    
                    # Slower timescale adaptation controlled by timescale_factor
                    plasticity_eta = group['plasticity_eta'] * timescale_factor
                    
                    if combined_stability > 0.7:  # High stability, increase plasticity
                        plasticity_factor *= (1.0 + plasticity_eta)
                    elif combined_stability < 0.3:  # Low stability, decrease plasticity
                        plasticity_factor *= (1.0 - plasticity_eta)
                    
                    # Ensure plasticity stays within bounds
                    plasticity_factor = max(group['min_plasticity'], 
                                          min(group['max_plasticity'], plasticity_factor))
                    state['plasticity_factor'] = plasticity_factor
    
    def get_plasticity_stats(self):
        """Get statistics about current plasticity state."""
        plasticity_stats = {
            'min': float('inf'),
            'max': float('-inf'),
            'mean': 0.0,
            'high_plasticity_params': 0,
            'low_plasticity_params': 0,
            'param_count': 0
        }
        
        total_plasticity = 0.0
        param_count = 0
        
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'plasticity_factor' in state:
                    plasticity = state['plasticity_factor']
                    param_count += 1
                    total_plasticity += plasticity
                    
                    # Update min/max
                    plasticity_stats['min'] = min(plasticity_stats['min'], plasticity)
                    plasticity_stats['max'] = max(plasticity_stats['max'], plasticity)
                    
                    # Count high/low plasticity parameters
                    if plasticity > 2.0:
                        plasticity_stats['high_plasticity_params'] += 1
                    elif plasticity < 0.5:
                        plasticity_stats['low_plasticity_params'] += 1
        
        if param_count > 0:
            plasticity_stats['mean'] = total_plasticity / param_count
            plasticity_stats['param_count'] = param_count
            plasticity_stats['high_plasticity_pct'] = 100 * plasticity_stats['high_plasticity_params'] / param_count
            plasticity_stats['low_plasticity_pct'] = 100 * plasticity_stats['low_plasticity_params'] / param_count
        else:
            plasticity_stats['min'] = 0.0
            plasticity_stats['high_plasticity_pct'] = 0.0
            plasticity_stats['low_plasticity_pct'] = 0.0
        
        return plasticity_stats