import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class EnhancedActivationEngineering:
    """
    Dynamically modifies model activation patterns based on information-theoretic principles.
    
    This implementation uses mutual information maximization to select optimal activation 
    functions for each layer, following the mathematical convergence theory from the analysis.
    """
    def __init__(self, model, device='cuda', activation_types=None, update_freq=100):
        self.model = model
        self.device = device
        self.hooks = []
        self.activation_stats = {}
        
        # Define available activation types
        self.activation_types = activation_types or {
            'standard': F.gelu,
            'focused': lambda x: F.gelu(x) * torch.sigmoid(x),
            'sparse': lambda x: F.gelu(x) * (torch.abs(x) > 0.5).float(),
            'dynamic': self._dynamic_activation,
            'shifted': lambda x: F.gelu(x + 0.1) # Small shift helps break symmetry
        }
        
        # Track activation selection statistics
        self.activation_selection = {}
        self.mi_scores_history = defaultdict(lambda: defaultdict(list))
        
        # Update frequency (compute MI occasionally to save compute)
        self.update_frequency = update_freq
        self.step_counter = 0
        
        # Initialize activation tracking hooks
        self._initialize_hooks()
    
    def _initialize_hooks(self):
        """Initialize hooks for activation tracking and engineering."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and any(x in name.lower() for x in ['mlp', 'ffn', 'feed']):
                # Found an MLP/feedforward layer in transformer
                # Add pre-activation hook (before non-linearity)
                hook = module.register_forward_hook(
                    lambda mod, inp, out, layer_name=name: self._activation_hook(mod, inp, out, layer_name)
                )
                self.hooks.append(hook)
                
                # Initialize activation stats for this layer
                self.activation_stats[name] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'sparsity': 0.0,
                    'updates': 0,
                    'best_activation': 'standard',
                    'mi_scores': {}
                }
                
                # Initialize activation selection
                self.activation_selection[name] = 'standard'
                
                logger.info(f"Registered activation hook for {name}")
    
    def _activation_hook(self, module, inputs, output, layer_name):
        """
        Hook for tracking and modifying activations using information theory.
        Uses mutual information to select optimal activation function.
        """
        # Only perform MI computation occasionally (expensive)
        should_update_mi = (self.step_counter % self.update_frequency == 0)
        self.step_counter += 1
        
        # Update activation statistics
        with torch.no_grad():
            # Compute basic statistics
            curr_mean = output.mean().item()
            curr_std = output.std().item()
            curr_sparsity = (output.abs() < 0.1).float().mean().item()
            
            # Update statistics with exponential moving average
            if self.activation_stats[layer_name]['updates'] == 0:
                self.activation_stats[layer_name]['mean'] = curr_mean
                self.activation_stats[layer_name]['std'] = curr_std
                self.activation_stats[layer_name]['sparsity'] = curr_sparsity
            else:
                alpha = 0.1  # Update rate
                self.activation_stats[layer_name]['mean'] = (
                    (1 - alpha) * self.activation_stats[layer_name]['mean'] + 
                    alpha * curr_mean
                )
                self.activation_stats[layer_name]['std'] = (
                    (1 - alpha) * self.activation_stats[layer_name]['std'] + 
                    alpha * curr_std
                )
                self.activation_stats[layer_name]['sparsity'] = (
                    (1 - alpha) * self.activation_stats[layer_name]['sparsity'] + 
                    alpha * curr_sparsity
                )
            
            self.activation_stats[layer_name]['updates'] += 1
            
            # Compute mutual information occasionally
            if should_update_mi and self.model.training:
                # Calculate MI for each activation function
                mi_scores = {}
                
                # Get a sample of the outputs for efficiency
                sample_size = min(1000, output.shape[0] * output.shape[1])
                if sample_size > 0:
                    # Flatten and sample
                    x_flat = output.reshape(-1)
                    if x_flat.shape[0] > sample_size:
                        indices = torch.randperm(x_flat.shape[0], device=x_flat.device)[:sample_size]
                        x_sample = x_flat[indices]
                    else:
                        x_sample = x_flat
                    
                    # Compute MI for each activation function
                    for name, activation_fn in self.activation_types.items():
                        y_sample = activation_fn(x_sample)
                        mi_score = self.approx_mutual_information(x_sample, y_sample)
                        mi_scores[name] = mi_score
                        
                        # Update history
                        self.mi_scores_history[layer_name][name].append(mi_score)
                    
                    # Store scores
                    self.activation_stats[layer_name]['mi_scores'] = mi_scores
                    
                    # Select best activation based on MI
                    if mi_scores:
                        best_activation = max(mi_scores.items(), key=lambda x: x[1])[0]
                        self.activation_selection[layer_name] = best_activation
                        self.activation_stats[layer_name]['best_activation'] = best_activation
                        
                        logger.info(f"Layer {layer_name}: Selected activation '{best_activation}' with MI={mi_scores[best_activation]:.4f}")
        
        # Apply engineered activation if in training mode
        if self.model.training:
            # Apply the selected activation function
            activation_fn = self.activation_types[self.activation_selection[layer_name]]
            return activation_fn(output)
        
        return output
    
    def approx_mutual_information(self, x, y, bins=16):
        """
        Approximate mutual information between two continuous variables using histogram estimator.
        
        MI(X;Y) = H(X) + H(Y) - H(X,Y)
        """
        # Make sure we have tensors on the same device
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, device=self.device)
            
        # Move to CPU for histogram computation if needed
        x_cpu = x.detach().float().cpu()
        y_cpu = y.detach().float().cpu()
        
        # Normalize to [0, 1] for binning
        x_min, x_max = x_cpu.min(), x_cpu.max()
        y_min, y_max = y_cpu.min(), y_cpu.max()
        
        # Avoid division by zero
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        if x_range.item() == 0 or y_range.item() == 0:
            return 0.0
            
        x_norm = (x_cpu - x_min) / x_range
        y_norm = (y_cpu - y_min) / y_range
        
        # Compute histograms (marginal distributions)
        x_hist = torch.histc(x_norm, bins=bins, min=0, max=1)
        y_hist = torch.histc(y_norm, bins=bins, min=0, max=1)
        
        # Normalize histograms to get probability distributions
        x_hist = x_hist / x_hist.sum()
        y_hist = y_hist / y_hist.sum()
        
        # Compute joint histogram
        joint_hist = torch.zeros((bins, bins))
        
        # Bin indices for each sample
        x_indices = torch.clamp((x_norm * bins).long(), 0, bins-1)
        y_indices = torch.clamp((y_norm * bins).long(), 0, bins-1)
        
        # Count joint occurrences
        for i in range(len(x_indices)):
            joint_hist[x_indices[i], y_indices[i]] += 1
            
        # Normalize joint histogram
        joint_hist = joint_hist / joint_hist.sum()
        
        # Compute entropy terms
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        
        # H(X)
        h_x = -torch.sum(x_hist * torch.log2(x_hist + epsilon))
        
        # H(Y)
        h_y = -torch.sum(y_hist * torch.log2(y_hist + epsilon))
        
        # H(X,Y)
        h_xy = -torch.sum(joint_hist * torch.log2(joint_hist + epsilon))
        
        # I(X;Y) = H(X) + H(Y) - H(X,Y)
        mi = h_x + h_y - h_xy
        
        return mi.item()
    
    def _dynamic_activation(self, x):
        """Dynamic activation function that adapts to input distribution."""
        # Compute statistics of current input
        mean = x.mean()
        std = x.std()
        
        # Normalize for stability
        x_norm = (x - mean) / (std + 1e-5)
        
        # Dynamic blend of different activation components
        gelu_comp = F.gelu(x_norm)
        tanh_comp = torch.tanh(x_norm)
        
        # Data-dependent mixing weights
        pos_ratio = (x > 0).float().mean()
        
        # Blend activation components
        result = pos_ratio * gelu_comp + (1 - pos_ratio) * tanh_comp
        
        # Re-scale and re-center to original distribution
        return result * std + mean
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        logger.info(f"Removed {len(self.hooks)} activation hooks")
    
    def get_activation_stats(self):
        """Get activation statistics for all tracked layers."""
        return {name: {k: v for k, v in stats.items() if k != 'updates'} 
                for name, stats in self.activation_stats.items()}
    
    def get_activation_selection_report(self):
        """Get report on activation selections and mutual information scores."""
        report = {}
        
        for layer_name, selection in self.activation_selection.items():
            # Get MI scores if available
            mi_scores = self.activation_stats[layer_name].get('mi_scores', {})
            
            # Get recent history
            history = {act_name: scores[-10:] if scores else [] 
                      for act_name, scores in self.mi_scores_history[layer_name].items()}
            
            report[layer_name] = {
                'selected_activation': selection,
                'mi_scores': mi_scores,
                'recent_history': history
            }
        
        return report