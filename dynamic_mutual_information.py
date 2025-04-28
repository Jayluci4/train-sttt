import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class DynamicMutualInformationTracker:
    """
    Implements dynamic mutual information objectives to guide architecture updates.
    
    This component monitors the mutual information between inputs and latent
    representations to ensure that architecture changes increase information flow.
    
    Mathematical goal:
        I_{t+1}(x; z_{t+1}) - I_{t}(x; z_{t}) > 0
    
    Where:
        - I(x; z) is the mutual information between input x and latent z
        - t indicates the time step or architecture version
    """
    def __init__(
        self,
        model,
        # --- ADD vocab_size argument ---
        vocab_size: Optional[int] = None,
        # --- END ADD ---
        latent_extractor=None,
        num_bins: int = 20,
        window_size: int = 50,
        update_freq: int = 10,
        mi_threshold: float = 0.05
    ):
        self.model = model
        self.num_bins = num_bins
        self.window_size = window_size
        self.update_freq = update_freq
        self.mi_threshold = mi_threshold
        self.latent_extractor = latent_extractor or self._default_latent_extractor

        # --- Store vocab_size or try to get it ---
        if vocab_size is not None:
            self.vocab_size = vocab_size
        else:
            self.vocab_size = self._get_vocab_size() # Try to get it if not provided
            if self.vocab_size is None or self.vocab_size <= 0:
                 logger.error("Could not determine vocabulary size for MI tracker! Ensure model config is accessible or pass vocab_size explicitly.")
                 # Handle error: raise exception or use a default? Raising is safer.
                 raise ValueError("Vocabulary size is required and could not be determined for DynamicMutualInformationTracker")
        # --- END Store ---

        # Initialize other attributes
        self.mi_history = []
        self.input_entropy_history = []
        self.representation_entropy_history = []
        self.current_mi = None
        self.last_update_step = 0
        self.input_buffer = deque(maxlen=window_size)
        self.representation_buffer = deque(maxlen=window_size)
        self.hooks = []
        self.current_representation = None # Add this attribute
        self.collecting_representations = False # Add this attribute

        self._register_hooks()
        logger.info(f"Initialized DynamicMutualInformationTracker (vocab size: {self.vocab_size})")

    def _get_vocab_size(self) -> Optional[int]:
        """Try to get vocabulary size from model config."""
        try:
            # Try accessing directly
            return self.model.config.vocab_size
        except AttributeError:
            # Try base_model if PEFT adapter is used
            try:
                 return self.model.base_model.config.vocab_size
            except AttributeError:
                 logger.warning("Could not automatically determine vocab_size from model config.")
                 return None # Return None if not found

    def _register_hooks(self):
        """Register hooks to capture intermediate representations."""
        # Clear any existing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        registered_count = 0

        try:
            for name, module in self.model.named_modules():
                # Target LayerNorms within the transformer blocks (e.g., after MLP or Attention)
                # Qwen1.5 layers might use 'ln_1', 'ln_2' or similar inside 'layers.N'
                if isinstance(module, nn.LayerNorm) and '.layers.' in name and ('ln_1' in name or 'ln_2' in name or 'input_layernorm' in name or 'post_attention_layernorm' in name):
                    hook = module.register_forward_hook(self._representation_hook)
                    self.hooks.append(hook)
                    logger.info(f"Registered MI tracker hook on {name}")
                    registered_count += 1

                # Target the final LayerNorm (often named 'norm' at the top level of the model)
                elif isinstance(module, nn.LayerNorm) and name.endswith('.norm') and '.layers.' not in name:
                     hook = module.register_forward_hook(self._representation_hook)
                     self.hooks.append(hook)
                     logger.info(f"Registered MI tracker hook on FINAL {name}")
                     registered_count += 1

        except Exception as e:
             logger.error(f"Error during MI hook registration: {e}", exc_info=True)

        if registered_count == 0:
            logger.warning("Failed to register any hooks for representation extraction using standard LayerNorm patterns.")
            # Add alternative hook strategy if needed, e.g., hook specific MLP layers


    # --- Make sure methods below use self.vocab_size ---

    def _extract_input_features(self, inputs: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Extract features from input tensors."""
        if 'input_ids' in inputs:
            input_ids = inputs['input_ids']
            batch_size = input_ids.size(0)
            vocab_size = self.vocab_size # <-- USE STORED vocab_size

            try:
                # Ensure vocab_size is valid
                if vocab_size is None or vocab_size <= 0:
                     logger.error("Invalid vocab_size stored in MI tracker for feature extraction.")
                     return None

                features = torch.zeros(batch_size, vocab_size, device=input_ids.device, dtype=torch.float)
                for i in range(batch_size):
                    valid_indices = input_ids[i][input_ids[i] < vocab_size]
                    if valid_indices.numel() > 0:
                        token_counts = torch.bincount(valid_indices, minlength=vocab_size)
                        length_to_copy = min(features.shape[1], token_counts.shape[0])
                        features[i, :length_to_copy] = token_counts[:length_to_copy].float()

                features = features / (features.sum(dim=1, keepdim=True) + 1e-8)
                return features
            except Exception as e:
                logger.warning(f"Error extracting input features: {e}")
                return None
        # ... (rest of the method remains same) ...
        logger.warning("Could not extract suitable input features for MI calculation.")
        return None

    def _estimate_entropy(self, data: torch.Tensor) -> float:
        """Estimate entropy of a distribution using histogram binning."""
        # (Ensure this method doesn't rely on an uninitialized vocab_size)
        # This method primarily works on feature distributions, not directly vocab size
        # ... (existing implementation should be okay) ...
        if data is None or data.numel() == 0:
            return 0.0
        # Apply dimensionality reduction if data is high-dimensional
        if data.size(1) > 50:
             try:
                  # Use PCA to reduce dimensions
                  data = data - data.mean(dim=0, keepdim=True)
                  # Ensure matrix is on CPU for SVD if needed, or handle potential CUDA SVD issues
                  U, S, V = torch.linalg.svd(data.cpu() if not data.is_cuda else data, full_matrices=False)
                  data = torch.mm(data, V[:, :min(50, data.size(1))])
             except Exception as e:
                  logger.warning(f"SVD failed during entropy estimation: {e}. Using original data.")


        # Check again after potential SVD failure or if originally small
        if data is None or data.numel() == 0:
            return 0.0

        # Normalize data to [0, 1] for binning
        # Add small epsilon to range to prevent division by zero
        epsilon = 1e-8
        data_min = data.min(dim=0, keepdim=True)[0]
        data_max = data.max(dim=0, keepdim=True)[0]
        data_range = data_max - data_min + epsilon
        # data_range[data_range < epsilon] = epsilon # Ensure range is not too small

        normalized_data = (data - data_min) / data_range

        # Compute histogram entropy (average across features)
        entropy_sum = 0.0
        valid_features = 0

        # Limit feature processing for efficiency, ensure at least 1 feature processed if available
        num_features_to_process = min(10, normalized_data.size(1)) if normalized_data.size(1) > 0 else 0

        for i in range(num_features_to_process):
            # Extract feature column
            feature = normalized_data[:, i]

            # Create histogram
            try:
                 hist = torch.histc(feature.cpu(), bins=self.num_bins, min=0, max=1) # histc might need CPU tensor
                 hist = hist / hist.sum() # Normalize histogram

                 # Compute entropy: -sum(p * log2(p))
                 entropy = -torch.sum(hist * torch.log2(hist + epsilon))

                 # Only include in average if valid
                 if not torch.isnan(entropy) and not torch.isinf(entropy):
                     entropy_sum += entropy.item()
                     valid_features += 1
            except Exception as e:
                 logger.warning(f"Error calculating histogram/entropy for feature {i}: {e}")


        # Return average entropy
        if valid_features > 0:
            return entropy_sum / valid_features
        else:
            return 0.0


    def _estimate_mutual_information(self, X: torch.Tensor, Z: torch.Tensor) -> float:
        """Estimate mutual information between X and Z using histogram method."""
        # (Ensure this method doesn't rely on an uninitialized vocab_size)
        # This method works on input/representation features
        # ... (existing implementation) ...
        if X is None or Z is None or X.numel() == 0 or Z.numel() == 0:
             return 0.0

        # Reduce dimensions for computational efficiency if needed
        if X.size(1) > 10:
             try:
                  X = X - X.mean(dim=0, keepdim=True)
                  U, S, V = torch.linalg.svd(X.cpu() if not X.is_cuda else X, full_matrices=False)
                  X = torch.mm(X, V[:, :min(10, X.size(1))])
             except Exception as e:
                  logger.warning(f"SVD failed for X during MI estimation: {e}. Using original data.")

        if Z.size(1) > 10:
             try:
                  Z = Z - Z.mean(dim=0, keepdim=True)
                  U, S, V = torch.linalg.svd(Z.cpu() if not Z.is_cuda else Z, full_matrices=False)
                  Z = torch.mm(Z, V[:, :min(10, Z.size(1))])
             except Exception as e:
                  logger.warning(f"SVD failed for Z during MI estimation: {e}. Using original data.")


        # Check again after potential SVD failure
        if X is None or Z is None or X.numel() == 0 or Z.numel() == 0:
             return 0.0

        # Use only the first dimension for joint entropy calculation (simplification)
        X1 = X[:, 0]
        Z1 = Z[:, 0]

        # Compute marginal entropies using the _estimate_entropy helper
        # Pass the full (potentially reduced-dim) tensors
        H_X = self._estimate_entropy(X)
        H_Z = self._estimate_entropy(Z)

        # Compute joint entropy using only the first dimension (as per original logic)
        joint_entropy = 0.0
        try:
             # Normalize first dimensions to [0, 1]
             epsilon = 1e-8
             X1_min, X1_max = X1.min(), X1.max()
             Z1_min, Z1_max = Z1.min(), Z1.max()
             X1_range = X1_max - X1_min + epsilon
             Z1_range = Z1_max - Z1_min + epsilon

             X1_normalized = (X1 - X1_min) / X1_range
             Z1_normalized = (Z1 - Z1_min) / Z1_range

             # Create 2D histogram
             joint_hist = torch.zeros(self.num_bins, self.num_bins, device='cpu') # Ensure hist is on CPU

             # Bin indices
             X_bins = torch.clamp((X1_normalized * self.num_bins).long(), 0, self.num_bins - 1)
             Z_bins = torch.clamp((Z1_normalized * self.num_bins).long(), 0, self.num_bins - 1)

             # Fill joint histogram (move indices to CPU if needed)
             X_bins_cpu = X_bins.cpu()
             Z_bins_cpu = Z_bins.cpu()
             for i in range(X.size(0)):
                 joint_hist[X_bins_cpu[i], Z_bins_cpu[i]] += 1

             # Normalize to get joint probability
             joint_hist_sum = joint_hist.sum()
             if joint_hist_sum > 0:
                  joint_hist = joint_hist / joint_hist_sum
             else:
                  logger.warning("Joint histogram sum is zero in MI calculation.")
                  return 0.0 # Avoid division by zero


             # Compute joint entropy
             joint_entropy = -torch.sum(joint_hist * torch.log2(joint_hist + epsilon)).item()

             if np.isnan(joint_entropy) or np.isinf(joint_entropy):
                 logger.warning(f"Joint entropy calculation resulted in invalid value ({joint_entropy}), using fallback.")
                 joint_entropy = H_X + H_Z # Fallback: assume independence

        except Exception as e:
             logger.error(f"Error calculating joint entropy in MI estimation: {e}", exc_info=True)
             joint_entropy = H_X + H_Z # Fallback on error

        # Compute mutual information: MI(X;Z) = H(X) + H(Z) - H(X,Z)
        mi = H_X + H_Z - joint_entropy

        # Ensure non-negative
        return max(0.0, mi)

    
    def should_increase_capacity(self) -> bool:
        """
        Determine if model capacity should be increased based on mutual information.
        
        Returns:
            True if capacity should be increased (MI is growing)
        """
        if len(self.mi_history) < 3:
            # Not enough history
            return False
        
        # Check if MI is consistently increasing
        recent_mi = self.mi_history[-3:]
        
        # MI should be increasing over time
        if recent_mi[-1] > recent_mi[0] * (1.0 + self.mi_threshold):
            logger.info(f"MI increasing: {recent_mi[0]:.4f} -> {recent_mi[-1]:.4f}")
            return True
        
        return False
    
    def should_decrease_capacity(self) -> bool:
        """
        Determine if model capacity should be decreased based on mutual information.
        
        Returns:
            True if capacity should be decreased (MI has plateaued or decreased)
        """
        if len(self.mi_history) < 5:
            # Need more history for decreasing capacity
            return False
        
        # Check if MI has plateaued or decreased
        recent_mi = self.mi_history[-5:]
        
        # MI should be decreasing or flat
        if recent_mi[-1] < recent_mi[0] * (1.0 - self.mi_threshold):
            logger.info(f"MI decreasing: {recent_mi[0]:.4f} -> {recent_mi[-1]:.4f}")
            return True
        
        # Check for plateau
        mi_std = np.std(recent_mi)
        mi_mean = np.mean(recent_mi)
        
        # If variation is small relative to mean, consider it a plateau
        if mi_mean > 0 and mi_std / mi_mean < 0.05:
            logger.info(f"MI plateau detected: {mi_mean:.4f} ± {mi_std:.4f}")
            return True
        
        return False
    
    def get_normalized_mi(self) -> float:
        """
        Get current mutual information normalized by input entropy.
        This indicates how much of the input information is captured in the representation.
        
        Returns:
            Normalized mutual information value in [0, 1]
        """
        if not self.mi_history or not self.input_entropy_history:
            return 0.0
        
        mi = self.mi_history[-1]
        input_entropy = self.input_entropy_history[-1]
        
        if input_entropy > 0:
            # Normalize by input entropy: MI(X;Z) / H(X)
            return min(1.0, mi / input_entropy)
        
        return 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get mutual information statistics."""
        if not self.mi_history:
            return {}
        
        return {
            'current_mi': self.mi_history[-1] if self.mi_history else 0.0,
            'mi_history': self.mi_history[-10:],
            'input_entropy': self.input_entropy_history[-1] if self.input_entropy_history else 0.0,
            'representation_entropy': self.representation_entropy_history[-1] if self.representation_entropy_history else 0.0,
            'normalized_mi': self.get_normalized_mi()
        }

    def __del__(self):
        """Clean up hooks when object is deleted."""
        try:
            self._remove_hooks()
        except:
            # Ignore errors during cleanup
            pass
            
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []