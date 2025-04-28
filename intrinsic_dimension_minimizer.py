import torch
import logging
import math
from typing import Dict, List, Optional, Union, Any
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)

class IntrinsicDimensionMinimizer:
    """
    Implements Intrinsic Dimension Minimization (Entropy-Constrained Morphing).
    
    This component tracks the empirical Fisher Information or covariance of gradients
    to estimate the intrinsic dimension of the parameter space, and uses this to make
    decisions about architecture morphing.
    
    Mathematical definition of intrinsic dimension:
        dim_ε(F) = (tr F)² / (tr F² + ε)
    
    Where F is the Fisher Information Matrix.
    """
    def __init__(
        self, 
        model, 
        epsilon: float = 1e-6, 
        window_size: int = 50,
        dimension_threshold: float = 0.1  # Relative increase threshold
    ):
        self.model = model
        self.epsilon = epsilon
        self.window_size = window_size
        self.dimension_threshold = dimension_threshold
        
        # Gradient statistics tracking
        self.gradient_history = defaultdict(lambda: deque(maxlen=window_size))
        self.fisher_stats = {}
        self.intrinsic_dim_history = []
        self.current_intrinsic_dim = None
        
        # Layer grouping for efficiency
        self.parameter_groups = self._initialize_parameter_groups()
    
    def _initialize_parameter_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        """Group parameters by module for efficient tracking."""
        param_groups = {}
        
        # Group by top-level module
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            # Extract top-level module name
            top_level = name.split('.')[0]
            
            if top_level not in param_groups:
                param_groups[top_level] = []
                
            param_groups[top_level].append(param)
        
        return param_groups
    
    def update(self, step: int) -> None:
        """
        Update gradient statistics and intrinsic dimension estimate.
        Should be called after backward pass but before optimizer step.
        """
        # Record current gradients for each parameter group
        for group_name, params in self.parameter_groups.items():
            # Collect gradients for this group
            grad_vectors = []
            for param in params:
                if param.grad is not None:
                    grad_vectors.append(param.grad.detach().flatten())
            
            if not grad_vectors:
                continue
                
            # Concatenate gradients into a single vector for this group
            grads = torch.cat(grad_vectors)
            
            # Store gradient for Fisher estimation
            self.gradient_history[group_name].append(grads)
            
            # Update Fisher statistics if we have enough samples
            if len(self.gradient_history[group_name]) >= min(10, self.window_size):
                self._update_fisher_stats(group_name)
        
        # Update global intrinsic dimension
        if step % 10 == 0:  # Update periodically to save computation
            self._update_intrinsic_dimension()
    
    def _update_fisher_stats(self, group_name: str) -> None:
        """Update Fisher Information Matrix statistics for a parameter group."""
        recent_grads = list(self.gradient_history[group_name])
        
        # We don't compute the full FIM as that would be too expensive
        # Instead, we compute summary statistics: tr(F) and tr(F²)
        
        # Compute squared gradient norms and their sum
        grad_norms_squared = []
        for grad in recent_grads:
            grad_norm_sq = torch.sum(grad * grad).item()
            grad_norms_squared.append(grad_norm_sq)
        
        # Compute trace of F (sum of squared gradient norms)
        tr_F = sum(grad_norms_squared) / len(grad_norms_squared)
        
        # Compute trace of F² (needs the outer products)
        # This is approximated via the squared Frobenius norm of the empirical FIM
        tr_F_squared = 0.0
        
        if len(recent_grads) >= 2:
            # Compute empirical covariance using sample covariance estimator
            # This is a fast approximation - in a full implementation we would
            # compute the pairwise gradient products more carefully
            
            # Concatenate gradients into a matrix
            grad_matrix = torch.stack(recent_grads[-min(10, len(recent_grads)):])
            
            # Center gradients (subtract mean)
            grad_mean = torch.mean(grad_matrix, dim=0, keepdim=True)
            centered_grads = grad_matrix - grad_mean
            
            # Compute empirical covariance
            n_samples = centered_grads.size(0)
            cov_factor = 1.0 / max(1, n_samples - 1)
            
            # This gives us the Frobenius norm of the covariance matrix
            # which approximates the trace of F²
            tr_F_squared = torch.sum(torch.matmul(centered_grads, centered_grads.t())**2).item() * cov_factor**2
        else:
            # Not enough samples, use a simple approximation
            tr_F_squared = tr_F**2
        
        # Store statistics
        self.fisher_stats[group_name] = {
            'tr_F': tr_F,
            'tr_F_squared': tr_F_squared,
            'group_size': sum(p.numel() for p in self.parameter_groups[group_name])
        }
    
    def _update_intrinsic_dimension(self) -> None:
        """Update global intrinsic dimension estimate across all parameter groups."""
        if not self.fisher_stats:
            return
        
        # Compute weighted sum of traces across all groups
        total_tr_F = 0.0
        total_tr_F_squared = 0.0
        total_size = 0
        
        for stats in self.fisher_stats.values():
            group_size = stats['group_size']
            weight = group_size / (1 + total_size)  # Weighted by parameter count
            
            total_tr_F += stats['tr_F'] * weight
            total_tr_F_squared += stats['tr_F_squared'] * weight
            total_size += group_size
        
        # Compute intrinsic dimension
        # dim_ε(F) = (tr F)² / (tr F² + ε)
        intrinsic_dim = (total_tr_F**2) / (total_tr_F_squared + self.epsilon)
        
        # Record history
        self.intrinsic_dim_history.append(intrinsic_dim)
        self.current_intrinsic_dim = intrinsic_dim
        
        logger.debug(f"Updated intrinsic dimension: {intrinsic_dim:.4f}")
    
    def should_increase_rank(self) -> bool:
        """
        Determine if rank should be increased based on intrinsic dimension.
        Returns True if the intrinsic dimension is increasing, indicating
        the model is exploring a higher dimensional space.
        """
        if len(self.intrinsic_dim_history) < 3:
            # Not enough history to make a decision
            return False
        
        # Look at recent trend
        recent_dims = self.intrinsic_dim_history[-3:]
        if recent_dims[-1] > recent_dims[0] * (1.0 + self.dimension_threshold):
            # Intrinsic dimension is increasing significantly
            # This suggests the model is exploring a higher dimensional space
            # and could benefit from increased capacity
            logger.info(f"Intrinsic dimension increasing: {recent_dims[0]:.2f} -> {recent_dims[-1]:.2f}")
            return True
        
        return False
    
    def should_decrease_rank(self) -> bool:
        """
        Determine if rank should be decreased based on intrinsic dimension.
        Returns True if the intrinsic dimension is decreasing, indicating
        the model is converging to a lower dimensional manifold.
        """
        if len(self.intrinsic_dim_history) < 5:
            # Need more history to be confident in decreasing rank
            return False
        
        # Look at longer trend for decreasing (more conservative)
        recent_dims = self.intrinsic_dim_history[-5:]
        if recent_dims[-1] < recent_dims[0] * (1.0 - self.dimension_threshold):
            # Intrinsic dimension is decreasing significantly
            # This suggests the model has found a lower dimensional manifold
            # and we can reduce capacity while maintaining performance
            logger.info(f"Intrinsic dimension decreasing: {recent_dims[0]:.2f} -> {recent_dims[-1]:.2f}")
            return True
        
        return False
    
    def get_group_intrinsic_dimensions(self) -> Dict[str, float]:
        """Get intrinsic dimension estimates for each parameter group."""
        group_dims = {}
        
        for group_name, stats in self.fisher_stats.items():
            tr_F = stats['tr_F']
            tr_F_squared = stats['tr_F_squared']
            
            # dim_ε(F) = (tr F)² / (tr F² + ε)
            dim = (tr_F**2) / (tr_F_squared + self.epsilon)
            group_dims[group_name] = dim
        
        return group_dims
    
    def get_stats(self) -> Dict[str, Any]:
        """Get all intrinsic dimension statistics."""
        group_dims = self.get_group_intrinsic_dimensions()
        
        return {
            'current_intrinsic_dim': self.current_intrinsic_dim,
            'dimension_history': self.intrinsic_dim_history[-10:] if self.intrinsic_dim_history else [],
            'group_dimensions': group_dims
        }
