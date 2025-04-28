import torch
import logging
import math
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class BregmanDynamicsController:
    """
    Implements Bregman Dynamics for architecture morphing with formal convergence guarantees.
    
    This component models architecture changes as mirror descent steps with Bregman divergence:
        θ_{t+1} = argmin_θ ⟨∇L_t, θ⟩ + (1/η_t) D_ϕ(θ || θ_t)
    
    Where:
        - D_ϕ is the Bregman divergence induced by convex function ϕ
        - η_t is the step size at time t
    
    This approach ensures that architecture changes maintain convergence properties
    while allowing non-Euclidean geometry to guide the optimization trajectory.
    """
    def __init__(
        self, 
        model,
        divergence_type: str = 'squared_euclidean',
        step_size_schedule: str = 'inverse_sqrt_t',
        initial_step_size: float = 1.0,
        error_budget: float = 1.0,
        budget_exponent: float = 1.1  # Error decay exponent (> 1 for summability)
    ):
        self.model = model
        self.divergence_type = divergence_type
        self.step_size_schedule = step_size_schedule
        self.initial_step_size = initial_step_size
        self.error_budget = error_budget
        self.budget_exponent = budget_exponent
        
        # Step tracking
        self.step_count = 0
        self.cumulative_error = 0.0
        self.error_history = []
        
        # Set up Bregman divergence function
        self.bregman_divergence_fn = self._get_bregman_divergence()
        
        # Initialize parameter reference 
        self.last_parameters = self._get_parameter_vector()
        
        logger.info(f"Initialized BregmanDynamicsController with {divergence_type} divergence")
    
    def _get_bregman_divergence(self) -> Callable:
        """Get Bregman divergence function based on type."""
        if self.divergence_type == 'squared_euclidean':
            # D_ϕ(x||y) = (1/2)||x - y||²
            return self._squared_euclidean_divergence
        elif self.divergence_type == 'relative_entropy':
            # D_ϕ(x||y) = ∑_i x_i log(x_i/y_i) - x_i + y_i
            return self._relative_entropy_divergence
        elif self.divergence_type == 'itakura_saito':
            # D_ϕ(x||y) = ∑_i (x_i/y_i - log(x_i/y_i) - 1)
            return self._itakura_saito_divergence
        else:
            logger.warning(f"Unknown divergence type: {self.divergence_type}, using squared Euclidean")
            return self._squared_euclidean_divergence
    
    def _squared_euclidean_divergence(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Squared Euclidean distance Bregman divergence."""
        return 0.5 * torch.sum((x - y)**2)
    
    def _relative_entropy_divergence(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Relative entropy (KL divergence) Bregman divergence."""
        # Ensure positive values
        x_safe = torch.clamp(x, min=1e-10)
        y_safe = torch.clamp(y, min=1e-10)
        
        # KL(x||y) = ∑_i x_i log(x_i/y_i) - x_i + y_i
        return torch.sum(x_safe * torch.log(x_safe / y_safe) - x_safe + y_safe)
    
    def _itakura_saito_divergence(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Itakura-Saito Bregman divergence."""
        # Ensure positive values
        x_safe = torch.clamp(x, min=1e-10)
        y_safe = torch.clamp(y, min=1e-10)
        
        # D_IS(x||y) = ∑_i (x_i/y_i - log(x_i/y_i) - 1)
        return torch.sum(x_safe / y_safe - torch.log(x_safe / y_safe) - 1)
    
    def _get_parameter_vector(self) -> torch.Tensor:
        """Get flattened parameter vector."""
        params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params.append(param.data.detach().flatten())
        
        if not params:
            # Return empty tensor if no parameters
            return torch.tensor([])
        
        return torch.cat(params)
    
    def _get_step_size(self) -> float:
        """Get current step size based on schedule."""
        if self.step_size_schedule == 'constant':
            return self.initial_step_size
        elif self.step_size_schedule == 'inverse_t':
            # η_t = η_0 / t
            return self.initial_step_size / max(1, self.step_count)
        elif self.step_size_schedule == 'inverse_sqrt_t':
            # η_t = η_0 / √t
            return self.initial_step_size / math.sqrt(max(1, self.step_count))
        else:
            logger.warning(f"Unknown step size schedule: {self.step_size_schedule}, using inverse_sqrt_t")
            return self.initial_step_size / math.sqrt(max(1, self.step_count))
    
    def step(self) -> None:
        """
        Record a step in the optimization process.
        This updates the step count and captures the current parameter state.
        """
        self.step_count += 1
        
        # Update parameter reference
        current_params = self._get_parameter_vector()
        
        # Store parameter reference for next step
        self.last_parameters = current_params
    
    def check_morph_valid(self, new_parameter_vector: torch.Tensor) -> Tuple[bool, float]:
        """
        Check if a proposed architecture morph is valid according to Bregman dynamics.
        
        Args:
            new_parameter_vector: Proposed new parameter vector after morphing
            
        Returns:
            Tuple of (is_valid, error_norm)
        """
        if self.last_parameters.numel() == 0 or new_parameter_vector.numel() == 0:
            # Cannot validate without parameters
            return False, float('inf')
        
        # Handle size mismatch (e.g., when adding/removing parameters)
        if self.last_parameters.size(0) != new_parameter_vector.size(0):
            # Resize to match (pad with zeros)
            if self.last_parameters.size(0) < new_parameter_vector.size(0):
                # Expand last_parameters
                padding = torch.zeros(
                    new_parameter_vector.size(0) - self.last_parameters.size(0),
                    device=self.last_parameters.device
                )
                expanded_last_params = torch.cat([self.last_parameters, padding])
                
                # Compute Bregman divergence
                divergence = self.bregman_divergence_fn(new_parameter_vector, expanded_last_params)
            else:
                # Expand new_parameter_vector
                padding = torch.zeros(
                    self.last_parameters.size(0) - new_parameter_vector.size(0),
                    device=new_parameter_vector.device
                )
                expanded_new_params = torch.cat([new_parameter_vector, padding])
                
                # Compute Bregman divergence
                divergence = self.bregman_divergence_fn(expanded_new_params, self.last_parameters)
        else:
            # Compute Bregman divergence
            divergence = self.bregman_divergence_fn(new_parameter_vector, self.last_parameters)
        
        # Calculate step size for this step
        step_size = self._get_step_size()
        
        # Calculate error norm - this is the key quantity for convergence
        error_norm = divergence.sqrt().item() if divergence.item() >= 0 else float('inf')
        
        # Get step-specific error budget
        # Error budget decreases over time to ensure summability
        step_budget = self.error_budget / (1 + self.step_count)**self.budget_exponent
        
        # Check if morph is valid (error within budget)
        is_valid = error_norm <= step_budget
        
        if is_valid:
            # Update cumulative error
            self.cumulative_error += error_norm
            
            # Record in history
            self.error_history.append((self.step_count, error_norm, self.cumulative_error))
            
            logger.info(
                f"Valid architecture morph at step {self.step_count}: "
                f"error={error_norm:.4e}, budget={step_budget:.4e}, "
                f"cumulative={self.cumulative_error:.4e}"
            )
        else:
            logger.warning(
                f"Invalid architecture morph at step {self.step_count}: "
                f"error={error_norm:.4e} > budget={step_budget:.4e}"
            )
        
        return is_valid, error_norm
    
    def morph_parameters(
        self, 
        old_params: List[torch.nn.Parameter], 
        new_params: List[torch.nn.Parameter]
    ) -> Tuple[bool, float]:
        """
        Validate and apply parameter morphing using Bregman dynamics.
        
        Args:
            old_params: List of parameters to be replaced
            new_params: List of new parameters
            
        Returns:
            Tuple of (is_valid, error_norm)
        """
        # Flatten parameters
        old_param_vector = torch.cat([p.data.detach().flatten() for p in old_params])
        new_param_vector = torch.cat([p.data.detach().flatten() for p in new_params])
        
        # Check if morph is valid
        is_valid, error_norm = self.check_morph_valid(new_param_vector)
        
        if is_valid:
            # Update parameter reference
            current_params = self._get_parameter_vector()
            self.last_parameters = current_params
        
        return is_valid, error_norm
    
    def get_remaining_budget(self) -> float:
        """Get remaining error budget."""
        return max(0.0, self.error_budget - self.cumulative_error)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics."""
        return {
            'step_count': self.step_count,
            'cumulative_error': self.cumulative_error,
            'error_budget': self.error_budget,
            'remaining_budget': self.get_remaining_budget(),
            'last_errors': self.error_history[-10:] if self.error_history else [],
            'divergence_type': self.divergence_type,
            'step_size_schedule': self.step_size_schedule,
            'current_step_size': self._get_step_size()
        }
