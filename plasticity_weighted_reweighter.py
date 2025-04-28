import torch
import logging
import math
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class PlasticityWeightedReweighter:
    """
    Implements plasticity-weighted data reweighting to focus training on
    samples that impact highly plastic parameters.
    
    Mathematical formulation:
        ω(x) ∝ ∑_i ψ_i · ‖∇_θ_i ℓ(f_θ(x), y)‖²
    
    Where:
        - ψ_i is the plasticity of parameter group i
        - ∇_θ_i ℓ(f_θ(x), y) is the gradient for parameter group i
        - ω(x) is the importance weight for sample x
    """
    def __init__(
        self, 
        model, 
        optimizer,
        reweighting_strength: float = 0.5,
        max_weight_ratio: float = 5.0,
        min_weight: float = 0.1,
        smoothing_factor: float = 0.9
    ):
        self.model = model
        self.optimizer = optimizer
        self.reweighting_strength = reweighting_strength
        self.max_weight_ratio = max_weight_ratio
        self.min_weight = min_weight
        self.smoothing_factor = smoothing_factor
        
        # Sample weight tracking
        self.sample_weights = {}
        self.current_batch_weights = None
        self.weight_history = []
        
        # Parameter group tracking
        self.param_groups = list(self._get_parameter_groups())
        
        logger.info(f"Initialized PlasticityWeightedReweighter with {len(self.param_groups)} parameter groups")
    
    def _get_parameter_groups(self) -> List[Tuple[str, List[torch.nn.Parameter]]]:
        """Get parameter groups with their names."""
        # First try to extract parameter groups from a neural plasticity optimizer
        if hasattr(self.optimizer, 'state') and hasattr(self.optimizer, 'param_groups'):
            # Assume this is a plasticity optimizer with parameter-specific plasticity factors
            param_to_idx = {}
            for i, group in enumerate(self.optimizer.param_groups):
                for param in group['params']:
                    param_to_idx[param] = i
            
            # Group by module name
            param_groups = defaultdict(list)
            for name, param in self.model.named_parameters():
                if param in param_to_idx and param.requires_grad:
                    # Extract module name (first component)
                    module_name = name.split('.')[0]
                    param_groups[module_name].append(param)
            
            return [(name, params) for name, params in param_groups.items()]
        
        # Fallback: group by top-level module name
        param_groups = defaultdict(list)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                module_name = name.split('.')[0]
                param_groups[module_name].append(param)
        
        return [(name, params) for name, params in param_groups.items()]
    
    def compute_sample_weights(self, batch_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute importance weights for each sample in the batch based on
        their interaction with plastic parameters.
        
        Args:
            batch_inputs: Dictionary of input tensors
            
        Returns:
            Tensor of sample weights with shape [batch_size]
        """
        # Forward pass to get gradients
        self.model.train()
        outputs = self.model(**batch_inputs)
        loss = outputs.loss
        
        # Get batch size
        batch_size = self._get_batch_size(batch_inputs)
        if batch_size <= 1:
            # No need to reweight for a single sample
            return torch.ones(1, device=loss.device)
        
        # Create sample weights tensor
        sample_weights = torch.ones(batch_size, device=loss.device)
        
        # Compute per-sample gradients if possible
        if hasattr(outputs, 'logits') and 'labels' in batch_inputs:
            # Compute per-sample weights
            weights = self._compute_per_sample_weights(
                outputs.logits, batch_inputs['labels'], outputs.loss
            )
            
            if weights is not None:
                sample_weights = weights
        
        # Normalize weights
        if sample_weights.numel() > 1:
            mean_weight = sample_weights.mean()
            if mean_weight > 0:
                sample_weights = sample_weights / mean_weight
            
            # Apply maximum weight constraint
            sample_weights = torch.clamp(
                sample_weights, 
                min=self.min_weight, 
                max=self.max_weight_ratio
            )
        
        # Store current weights
        self.current_batch_weights = sample_weights.detach()
        
        # Record statistics
        self.weight_history.append({
            'mean': sample_weights.mean().item(),
            'min': sample_weights.min().item(),
            'max': sample_weights.max().item(),
            'std': sample_weights.std().item() if sample_weights.numel() > 1 else 0.0
        })
        
        return sample_weights
    
    def _compute_per_sample_weights(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss: torch.Tensor # Keep the original loss calculation from the model forward pass
    ) -> Optional[torch.Tensor]:
        """
        Compute weights for each sample based on plasticity and gradients.
        """
        batch_size = logits.size(0)
        vocab_size = logits.size(-1) # Get vocab size from logits

        # Get plasticity factors for each parameter group
        param_plasticities = self._get_parameter_plasticities()
        if not param_plasticities:
            logger.warning("Could not get parameter plasticities.")
            return None

        sample_weights = torch.zeros(batch_size, device=logits.device)

        # Compute per-sample gradient norms weighted by plasticity
        for i in range(batch_size):
            # Zero gradients before computing for this sample
            self.model.zero_grad()

            # --- MODIFICATION START ---
            # Recompute loss for this specific sample
            # Ensure labels are not shifted/ignored (-100) for loss calculation
            sample_logits_flat = logits[i].view(-1, vocab_size) # Shape: [seq_len, vocab_size]
            sample_labels_flat = labels[i].view(-1)             # Shape: [seq_len]

            # Filter out ignored indices (e.g., padding)
            active_loss = sample_labels_flat != -100
            active_logits = sample_logits_flat[active_loss]
            active_labels = sample_labels_flat[active_loss]

            if active_logits.shape[0] == 0:
                 # Skip if no active labels in this sample
                 sample_weights[i] = 1.0 # Assign default weight
                 continue

            # Calculate loss for active tokens only
            sample_loss = torch.nn.functional.cross_entropy(
                active_logits, active_labels, reduction='sum'
            )

            # Backward pass for this sample to get gradients
            # Check if graph needs to be retained (depends on outer loop)
            # We assume retain_graph=True might be needed if this function is called
            # multiple times within a single outer backward pass context,
            # but for typical usage, it should be safe here if called once per batch.
            # Let's assume the outer loop structure handles retain_graph if needed.
            # If this is called outside the main backward(), retain_graph=False is safer.
            try:
                sample_loss.backward(retain_graph=True) # Retain graph might be needed if used inside another autograd context
            except RuntimeError as e:
                 if "specify retain_graph=True" in str(e):
                     logger.error("RuntimeError during per-sample backward. Ensure retain_graph=True if needed.")
                     sample_loss.backward(retain_graph=True)
                 else:
                     raise e

            # --- MODIFICATION END ---

            # Compute weighted gradient norm for this sample
            weighted_norm = 0.0
            for group_name, group_params in self.param_groups:
                if group_name in param_plasticities:
                    plasticity = param_plasticities[group_name]

                    # Sum gradient norms for this group
                    group_grad_norm_sq = 0.0
                    for param in group_params:
                        if param.grad is not None:
                            group_grad_norm_sq += torch.sum(param.grad ** 2).item()

                    # Add weighted contribution
                    weighted_norm += plasticity * math.sqrt(max(0, group_grad_norm_sq))

            # Store weight for this sample (ensure non-zero)
            sample_weights[i] = max(1e-4, weighted_norm) # Use a small epsilon

            # Crucially, zero gradients again after processing the sample
            # to avoid interference with the next sample's gradient calculation
            self.model.zero_grad(set_to_none=True) # Use set_to_none=True for efficiency

        # Apply reweighting strength
        if self.reweighting_strength < 1.0:
            # Blend with uniform weights
            uniform_weights = torch.ones_like(sample_weights)
            sample_weights = (
                self.reweighting_strength * sample_weights +
                (1 - self.reweighting_strength) * uniform_weights
            )

        return sample_weights
            
    
    def _get_parameter_plasticities(self) -> Dict[str, float]:
        """Get plasticity factors for parameter groups from optimizer."""
        plasticities = {}
        
        # Try to extract from NeuralPlasticityOptimizer or similar
        if hasattr(self.optimizer, 'state'):
            # Map parameters to their groups
            param_to_group = {}
            for i, group in enumerate(self.param_groups):
                group_name, params = group
                for param in params:
                    param_to_group[param] = group_name
            
            # Extract plasticity factors from optimizer state
            for param, state in self.optimizer.state.items():
                if param in param_to_group and 'plasticity_factor' in state:
                    group_name = param_to_group[param]
                    
                    # Average plasticity for parameters in the same group
                    if group_name not in plasticities:
                        plasticities[group_name] = state['plasticity_factor']
                    else:
                        # Exponential moving average
                        plasticities[group_name] = (
                            self.smoothing_factor * plasticities[group_name] + 
                            (1 - self.smoothing_factor) * state['plasticity_factor']
                        )
        
        return plasticities
    
    def _get_batch_size(self, batch_inputs: Dict[str, torch.Tensor]) -> int:
        """Extract batch size from inputs dictionary."""
        for tensor in batch_inputs.values():
            if isinstance(tensor, torch.Tensor) and tensor.dim() > 0:
                return tensor.size(0)
        return 1
    
    def apply_weight_to_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Apply the computed sample weights to the loss.
        This should be called after computing sample weights.
        
        Args:
            loss: The loss tensor from the model
            
        Returns:
            Weighted loss tensor
        """
        if self.current_batch_weights is None or self.current_batch_weights.numel() == 1:
            return loss
        
        # Ensure loss has appropriate shape for weighting
        # This assumes the loss is already per-sample
        if loss.dim() == 0:
            # Scalar loss, can't apply per-sample weights
            return loss
        
        batch_size = loss.size(0)
        if batch_size != self.current_batch_weights.size(0):
            logger.warning(
                f"Batch size mismatch: loss has size {batch_size}, "
                f"weights has size {self.current_batch_weights.size(0)}"
            )
            return loss
        
        # Apply weights to loss
        weighted_loss = loss * self.current_batch_weights
        
        # Return mean loss
        return weighted_loss.mean()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reweighting statistics."""
        if not self.weight_history:
            return {}
            
        # Compute statistics from history
        recent_history = self.weight_history[-min(10, len(self.weight_history)):]
        
        return {
            'weight_mean': sum(h['mean'] for h in recent_history) / len(recent_history),
            'weight_min': min(h['min'] for h in recent_history),
            'weight_max': max(h['max'] for h in recent_history),
            'weight_std': sum(h['std'] for h in recent_history) / len(recent_history),
            'reweighting_strength': self.reweighting_strength
        }
