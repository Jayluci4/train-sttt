import math
import torch
import torch.nn as nn
import logging
from typing import Dict, Optional, List, Tuple, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

class EnhancedConvergentArchitectureController:
    """
    Enhanced architecture controller with stronger convergence guarantees based on 
    non-expansive operators with bounded cumulative error.
    
    This implementation follows the mathematical convergence theory from the analysis document,
    ensuring that architecture changes preserve optimization trajectories with provable guarantees.
    """
    def __init__(
        self, 
        model, 
        min_rank=4, 
        max_rank=64, 
        rank_step=4, 
        error_budget=1.0, 
        scan_freq=100,
        morph_threshold=0.15,
        optimizer=None
    ):
        self.model = model
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.rank_step = rank_step
        self.error_budget = error_budget
        self.morph_threshold = morph_threshold
        self.optimizer = optimizer  # Store reference to optimizer for parameter group repair
        
        # Track cumulative error for convergence guarantee
        self.cumulative_error = 0.0
        self.error_history = []
        
        # Module tracking
        self.module_map = {}  # Maps module names to convergent adapters
        self.init_adapters()
        
        # Deterministic schedule for architecture morphing
        self.next_morph_step = scan_freq
        self.base_scan_freq = scan_freq
        self.schedule_alpha = 0.7  # Between 0.5 and 1.0, per math analysis
        
        # Metrics
        self.morph_history = []
        self.step_count = 0
        
        # Keep track of total module count for budget redistribution
        self.total_module_count = len(self.module_map) or 1
    
    def init_adapters(self):
        """Initialize convergent adapters for all LoRA modules."""
        lora_modules = {}
        
        # Find all LoRA modules in the model
        for name, module in self.model.named_modules():
            # Check for LoRA modules (this may need adjustment based on your PEFT implementation)
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                lora_modules[name] = module
                logger.info(f"Found LoRA module: {name}")
        
        # Register convergent adapters
        for name, module in lora_modules.items():
            try:
                individual_error_budget = self.error_budget / max(1, len(lora_modules))
                self.module_map[name] = ConvergentLoraAdapter(
                    module, 
                    name=name,
                    error_budget=individual_error_budget
                )
                logger.info(f"Registered convergent adapter for {name}")
            except Exception as e:
                logger.warning(f"Could not register module {name}: {e}")
                
        # Keep track of total module count for budget redistribution
        self.total_module_count = len(self.module_map) or 1
    
    def register_optimizer(self, optimizer):
        """Register the optimizer for parameter group repair during architecture changes."""
        self.optimizer = optimizer
        logger.info(f"Registered optimizer for architecture controller: {type(optimizer).__name__}")
        
    def scan_for_new_modules(self):
        """
        Scan for new LoRA modules that might have been lazily initialized.
        This is important for models that create modules on demand.
        """
        current_modules = set(self.module_map.keys())
        new_modules = {}
        
        # Find all LoRA modules in the model
        for name, module in self.model.named_modules():
            # Check for LoRA modules that are not already registered
            if name not in current_modules and hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                new_modules[name] = module
                logger.info(f"Found new LoRA module: {name}")
        
        # Register new modules if any
        if new_modules:
            # Recalculate error budget for all modules
            new_total_count = self.total_module_count + len(new_modules)
            individual_error_budget = self.error_budget / new_total_count
            
            # Adjust existing modules' budgets
            for adapter in self.module_map.values():
                adapter.error_budget = individual_error_budget
            
            # Register new modules
            for name, module in new_modules.items():
                try:
                    self.module_map[name] = ConvergentLoraAdapter(
                        module, 
                        name=name,
                        error_budget=individual_error_budget
                    )
                    logger.info(f"Registered new convergent adapter for {name}")
                except Exception as e:
                    logger.warning(f"Could not register new module {name}: {e}")
            
            # Update total count
            self.total_module_count = new_total_count
            logger.info(f"Redistributed error budget among {self.total_module_count} modules")
            
        return len(new_modules) > 0    
    def should_morph_architecture(self, step):
        """
        Determine if architecture should be morphed at this step using deterministic schedule.
        This replaces the original probabilistic approach for better reproducibility.
        """
        if step >= self.next_morph_step:
            # Update next morph step using sublinear schedule with self.schedule_alpha parameter
            self.next_morph_step += int(self.base_scan_freq * (1 + step)**self.schedule_alpha)
            return True
        return False
    
    def scan_architecture(self, step):
        """Scan model architecture for bottlenecks and opportunities."""
        self.step_count = step
        
        # Check if it's time to morph using deterministic schedule
        if not self.should_morph_architecture(step):
            return None
        
        logger.info(f"Scanning architecture at step {step}")
        
        # Collect gradient flow statistics for each module
        grad_flow = {}
        for name, adapter in self.module_map.items():
            grad_flow[name] = adapter.compute_grad_flow()
        
        # Calculate importance scores
        importance_scores = self._compute_importance_scores(grad_flow)
        
        # Identify bottlenecks (high importance, low capacity)
        bottlenecks = self._detect_bottlenecks(importance_scores)
        
        # Identify underutilized modules (low importance, high capacity)
        underutilized = self._detect_underutilized(importance_scores)
        
        # Determine if morphing is needed based on bottleneck scores
        morph_needed = any(score > self.morph_threshold for _, score in bottlenecks[:3])
        
        # Log findings
        if bottlenecks:
            top_bottlenecks = bottlenecks[:3]
            logger.info(f"Top bottlenecks: {top_bottlenecks}")
        
        if underutilized:
            top_underutilized = underutilized[:3] 
            logger.info(f"Top underutilized modules: {top_underutilized}")
        
        return {
            'bottlenecks': bottlenecks,
            'underutilized': underutilized,
            'importance_scores': importance_scores,
            'morph_needed': morph_needed
        }
    
    def _compute_importance_scores(self, grad_flow):
        """Compute importance scores for each adapter based on gradient flow."""
        importance_scores = {}
        
        for name, adapter in self.module_map.items():
            # Base importance on gradient flow
            base_importance = grad_flow[name]
            
            # Get dimensional importance scores
            dim_importance_scores = adapter.compute_importance_scores()
            
            # Calculate dimensional importance metrics
            if dim_importance_scores is not None and dim_importance_scores.numel() > 0:
                variance = torch.var(dim_importance_scores).item() if dim_importance_scores.numel() > 1 else 0.0
                mean = torch.mean(dim_importance_scores).item()
                dim_importance = mean * (1.0 + variance)
                
                # Compute rank utilization (ratio of important dimensions to total)
                rank_utilization = min(1.0, dim_importance / max(1e-10, base_importance))
            else:
                dim_importance = 0.0
                rank_utilization = 0.5  # Default value
            
            # Final importance score
            importance_scores[name] = {
                'score': base_importance * (1.0 + rank_utilization),
                'gradient_flow': base_importance,
                'dimensional_importance': dim_importance,
                'rank_utilization': rank_utilization,
                'current_rank': adapter.current_rank
            }
        
        return importance_scores
    
    def _detect_bottlenecks(self, importance_scores):
        """Detect bottlenecks in the architecture."""
        bottlenecks = []
        
        for name, metrics in importance_scores.items():
            # Calculate bottleneck score
            score = metrics['score']
            rank = metrics['current_rank']
            
            # Bottleneck score increases when:
            # - Importance is high
            # - Current rank is small relative to max_rank
            # - Rank utilization is high (efficient use of current dimensions)
            bottleneck_score = score * metrics['rank_utilization'] * (1.0 - (rank / self.max_rank)**0.5)
            
            bottlenecks.append((name, bottleneck_score))
        
        # Sort by bottleneck score (descending)
        bottlenecks.sort(key=lambda x: x[1], reverse=True)
        
        return bottlenecks
    
    def _detect_underutilized(self, importance_scores):
        """Detect underutilized modules in the architecture."""
        underutilized = []
        
        for name, metrics in importance_scores.items():
            # Calculate underutilization score
            score = metrics['score']
            rank = metrics['current_rank']
            
            # Only consider modules with rank above minimum
            if rank <= self.min_rank:
                continue
            
            # Underutilization score increases when:
            # - Importance is low
            # - Current rank is large relative to min_rank
            # - Rank utilization is low (inefficient use of dimensions)
            underutilization_score = (1.0 - min(1.0, score)) * (rank / self.min_rank) * (1.0 - metrics['rank_utilization'])
            
            underutilized.append((name, underutilization_score))
        
        # Sort by underutilization score (descending)
        underutilized.sort(key=lambda x: x[1], reverse=True)
        
        return underutilized
    
    def morph_architecture(self, scan_results):
        """
        Apply architecture modifications based on scan results.
        Uses non-expansive transformations with bounded error for guaranteed convergence.
        """
        if scan_results is None or not scan_results.get('morph_needed', False):
            return None
        
        global_step = self.step_count
        modifications = {}
        total_error = 0.0
        
        # Scale error budget by current step to ensure summability
        # We divide by (1 + step)^1.1 to ensure the sum converges
        step_error_budget = self.error_budget / ((1 + global_step)**1.1)
        
        # Process bottlenecks (expand capacity)
        bottlenecks = scan_results.get('bottlenecks', [])
        for name, score in bottlenecks[:3]:  # Consider top 3 bottlenecks
            if score < self.morph_threshold:
                continue
                
            adapter = self.module_map.get(name)
            if adapter is None:
                continue
                
            # Calculate new rank (increase by rank_step but respect max_rank)
            current_rank = adapter.current_rank
            new_rank = min(self.max_rank, current_rank + self.rank_step)
            
            # Skip if no change
            if new_rank == current_rank:
                continue
            
            # Apply the change with convergence guarantees
            old_params = self._get_flat_params(adapter.module)
            
            # Get importance scores to guide expansion
            dim_scores = adapter.compute_importance_scores()
            
            # Pass the optimizer to the adapter for parameter group repair
            success = adapter.change_rank(new_rank, dim_scores, optimizer=self.optimizer)
            
            if success:
                # Calculate error norm for convergence tracking
                new_params = self._get_flat_params(adapter.module)
                error_norm = torch.norm(new_params - old_params).item()
                
                # Scale error by step for summability
                scaled_error = error_norm / ((1 + global_step)**1.1)
                total_error += scaled_error
                
                # Check against budget
                if total_error > step_error_budget:
                    logger.warning(f"Error budget exceeded: {total_error:.4e} > {step_error_budget:.4e}")
                    # Revert the change if it would exceed error budget
                    adapter.change_rank(current_rank, dim_scores, optimizer=self.optimizer)
                    continue
                
                # Record the change
                modifications[name] = {
                    'action': 'expand',
                    'old_rank': current_rank,
                    'new_rank': new_rank,
                    'error': error_norm,
                    'scaled_error': scaled_error
                }
                
                logger.info(f"Expanded {name} from rank {current_rank} to {new_rank}, error: {error_norm:.4e}")
        
        # Process underutilized modules (reduce capacity)
        underutilized = scan_results.get('underutilized', [])
        for name, score in underutilized[:3]:  # Consider top 3 underutilized
            if score < self.morph_threshold:
                continue
                
            adapter = self.module_map.get(name)
            if adapter is None:
                continue
                
            # Calculate new rank
            current_rank = adapter.current_rank
            
            # Calculate new rank (decrease by rank_step but respect min_rank)
            new_rank = max(self.min_rank, current_rank - self.rank_step)
            
            # Skip if no change
            if new_rank == current_rank:
                continue
            
            # Apply the change with convergence guarantees
            old_params = self._get_flat_params(adapter.module)
            
            # Get importance scores for optimal reduction
            dim_scores = adapter.compute_importance_scores()
            
            # Pass the optimizer to the adapter for parameter group repair
            success = adapter.change_rank(new_rank, dim_scores, optimizer=self.optimizer)
            
            if success:
                # Calculate error norm for convergence tracking
                new_params = self._get_flat_params(adapter.module)
                error_norm = torch.norm(new_params - old_params).item()
                
                # Scale error by step for summability
                scaled_error = error_norm / ((1 + global_step)**1.1)
                total_error += scaled_error
                
                # Check against budget
                if total_error > step_error_budget:
                    logger.warning(f"Error budget exceeded: {total_error:.4e} > {step_error_budget:.4e}")
                    # Revert the change if it would exceed error budget
                    adapter.change_rank(current_rank, dim_scores, optimizer=self.optimizer)
                    continue
                
                # Record the change
                modifications[name] = {
                    'action': 'reduce',
                    'old_rank': current_rank,
                    'new_rank': new_rank,
                    'error': error_norm,
                    'scaled_error': scaled_error
                }
                
                logger.info(f"Reduced {name} from rank {current_rank} to {new_rank}, error: {error_norm:.4e}")
        
        # Update cumulative error
        self.cumulative_error += total_error
        self.error_history.append((global_step, total_error, self.cumulative_error))
        
        # Record this morphing event
        if modifications:
            self.morph_history.append({
                'step': global_step,
                'modifications': modifications,
                'total_error': total_error,
                'cumulative_error': self.cumulative_error
            })
            
            logger.info(f"Architecture morphing complete at step {global_step}. " 
                       f"Changes: {len(modifications)}, "
                       f"Error: {total_error:.4e}, "
                       f"Cumulative: {self.cumulative_error:.4e}")
        
        return modifications if modifications else None
    
    def _get_flat_params(self, module):
        """Get flattened parameters from a module for error calculation."""
        params_list = []
        for name, param in module.named_parameters():
            if param.requires_grad:
                params_list.append(param.data.view(-1))
        return torch.cat(params_list) if params_list else torch.tensor([])
    
    def get_architecture_report(self):
        """Generate a report on the current architecture state and convergence metrics."""
        # Count total parameters in LoRA adapters
        total_params = 0
        architecture_state = {}
        
        for name, adapter in self.module_map.items():
            rank = adapter.current_rank
            in_features = adapter.module.lora_A.in_features if hasattr(adapter.module, 'lora_A') else 0
            out_features = adapter.module.lora_B.out_features if hasattr(adapter.module, 'lora_B') else 0
            params = rank * (in_features + out_features)
            total_params += params
            
            architecture_state[name] = {
                'rank': rank,
                'params': params
            }
        
        # Report on convergence guarantee
        convergence_status = {
            'cumulative_error': self.cumulative_error,
            'error_budget': self.error_budget,
            'budget_remaining': self.error_budget - self.cumulative_error,
            'convergence_guaranteed': self.cumulative_error <= self.error_budget
        }
        
        # Report on morphing activity
        morphing_summary = {
            'total_morphs': len(self.morph_history),
            'param_count': total_params
        }
        
        return {
            'architecture_state': architecture_state,
            'convergence_status': convergence_status,
            'morphing_summary': morphing_summary
        }


class ConvergentLoraAdapter:
    """
    Wrapper for LoRA modules that enables convergent rank changes.
    Implements non-expansive transformations with bounded error guarantees.
    """
    def __init__(self, module, name="", error_budget=0.1):
        self.module = module
        self.name = name
        self.error_budget = error_budget
        self.cumulative_error = 0.0
        
        # Get current rank and dimensions - handle PEFT structure variants
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            # Check if lora_A is a ModuleDict (PEFT structure for newer versions)
            if isinstance(module.lora_A, nn.ModuleDict) and 'default' in module.lora_A:
                # Use the 'default' adapter
                lora_A = module.lora_A['default']
                lora_B = module.lora_B['default']
                self.current_rank = lora_A.weight.shape[0]
                self.in_features = lora_A.in_features
                self.out_features = lora_B.out_features
            # Check if lora_A is a direct Linear layer (older PEFT structure)
            elif isinstance(module.lora_A, nn.Linear):
                self.current_rank = module.lora_A.weight.shape[0]
                self.in_features = module.lora_A.in_features
                self.out_features = module.lora_B.out_features
            else:
                raise ValueError(f"Module {name} has lora_A/B attributes but in an unrecognized format")
        else:
            raise ValueError(f"Module {name} does not have LoRA layers")
        
        # History tracking
        self.change_history = []
        
    def change_rank(self, new_rank, importance_scores=None, optimizer=None):
        """
        Change the rank of the LoRA adapter with convergence guarantees.
        Uses SVD for optimal compression during reduction and orthogonal initialization for expansion.
        
        Args:
            new_rank: New rank value
            importance_scores: Optional importance scores for dimensions
            optimizer: The optimizer to update parameter groups (critical for proper convergence)
            
        Returns:
            bool: Whether the rank change was successful
        """
        # Validate parameters
        if new_rank <= 0:
            logger.warning(f"Invalid new rank {new_rank} for {self.name}")
            return False
        
        if new_rank == self.current_rank:
            return True
        
        logger.info(f"Changing rank for {self.name} from {self.current_rank} to {new_rank}")
        
        try:
            with torch.no_grad():
                # Get current module
                lora_A = self.module.lora_A
                lora_B = self.module.lora_B
                
                # Get current weights
                current_A = lora_A.weight.detach().clone()  # [r, in_features]
                current_B = lora_B.weight.detach().clone()  # [out_features, r]
                
                # Save references to old parameters for optimizer state transfer
                old_params = [lora_A.weight, lora_B.weight]
                
                # Get device and dtype
                device = current_A.device
                dtype = current_A.dtype
                
                # Create new Linear modules with appropriate device & dtype
                new_lora_A = nn.Linear(self.in_features, new_rank, bias=False, device=device, dtype=dtype)
                new_lora_B = nn.Linear(new_rank, self.out_features, bias=False, device=device, dtype=dtype)
                
                # Handle rank increase
                if new_rank > self.current_rank:
                    # Approach 1: SVD-based expansion (better preserves information)
                    if self.current_rank > 0:
                        try:
                            # Try SVD-based expansion for optimal subspace preservation
                            # Compute AB decomposition
                            AB = current_B @ current_A  # [out_features, in_features]
                            
                            # Compute SVD
                            U, S, Vh = torch.linalg.svd(AB, full_matrices=False)
                            
                            # Calculate new weights with preserved subspace
                            S_sqrt = torch.sqrt(S)
                            
                            # Initialize new weights, preserving existing subspace
                            new_lora_A.weight[:self.current_rank, :] = (
                                torch.diag(S_sqrt[:self.current_rank]) @ Vh[:self.current_rank, :]
                            ).to(device=device, dtype=dtype)
                            new_lora_B.weight[:, :self.current_rank] = (
                                U[:, :self.current_rank] @ torch.diag(S_sqrt[:self.current_rank])
                            ).to(device=device, dtype=dtype)
                            
                            # Initialize new dimensions with orthogonal initialization
                            if new_rank > self.current_rank:
                                # Calculate scaling factor for orthogonal initialization
                                fan_in = self.in_features
                                gain = 1.0 / math.sqrt(fan_in)
                                
                                # Initialize new rows of A with orthogonal values
                                nn.init.orthogonal_(new_lora_A.weight[self.current_rank:, :])
                                new_lora_A.weight[self.current_rank:, :].mul_(gain)
                                
                                # Initialize new columns of B with zeros (non-expansive)
                                new_lora_B.weight[:, self.current_rank:].zero_()
                        except Exception as e:
                            # If SVD fails, fall back to simpler initialization
                            logger.warning(f"SVD expansion failed for {self.name}, falling back to zero-init: {e}")
                            
                            # Copy existing weights
                            new_lora_A.weight[:self.current_rank, :] = current_A
                            new_lora_B.weight[:, :self.current_rank] = current_B
                            
                            # Zero-initialize new dimensions (strictly non-expansive)
                            new_lora_A.weight[self.current_rank:, :].zero_()
                            new_lora_B.weight[:, self.current_rank:].zero_()
                    else:
                        # For first initialization, just use zeros
                        new_lora_A.weight.zero_()
                        new_lora_B.weight.zero_()
                
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
                        # If SVD fails, use importance scores or simple slicing
                        logger.warning(f"SVD reduction failed for {self.name}, falling back to alternative method: {e}")
                        
                        if importance_scores is not None and importance_scores.numel() >= new_rank:
                            # Select dimensions with highest importance scores
                            _, indices = torch.topk(importance_scores, new_rank)
                            indices, _ = torch.sort(indices)  # Sort indices for cleaner computation
                            
                            # Initialize new weights by keeping only important dimensions
                            for i, idx in enumerate(indices):
                                new_lora_A.weight[i, :] = current_A[idx, :]
                                new_lora_B.weight[:, i] = current_B[:, idx]
                        else:
                            # Simple slicing (if no better method works)
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
                if optimizer is not None:
                    # Repair parameter groups in optimizer
                    self._repair_param_groups(old_params, new_params, optimizer)
                    
                    # Transfer optimizer state for smooth learning curve
                    self._transfer_optimizer_state(old_params, new_params, optimizer)
                
                # Update the module
                self.module.lora_A = new_lora_A
                self.module.lora_B = new_lora_B
                
                # Update scaling factor if needed
                if hasattr(self.module, 'scaling'):
                    # Get LoRA alpha if available
                    lora_alpha = getattr(self.module, 'lora_alpha', self.current_rank)
                    self.module.scaling = lora_alpha / new_rank
                
                # Update current rank
                self.current_rank = new_rank
                
                # Record change in history
                self.change_history.append({
                    'old_rank': self.current_rank, 
                    'new_rank': new_rank
                })
                
                return True
                
        except Exception as e:
            logger.error(f"Error changing rank for {self.name}: {e}")
            return False
    
    def _repair_param_groups(self, old_params, new_params, optimizer):
        """
        Repair optimizer parameter groups when parameters are replaced.
        This is CRITICAL for proper convergence.
        
        Args:
            old_params: List of old parameters to be removed
            new_params: List of new parameters to be added
            optimizer: The optimizer to update
        """
        if optimizer is None:
            return
            
        logger.info(f"Repairing parameter groups for {self.name}")
        
        try:
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
                
            logger.info(f"Parameter groups repaired successfully for {self.name}")
        except Exception as e:
            logger.error(f"Error repairing parameter groups: {e}")
    
    def _transfer_optimizer_state(self, old_params, new_params, optimizer):
        """
        Transfer optimizer state from old parameters to new parameters.
        This maintains momentum and other state for smooth training.
        
        Args:
            old_params: List of old parameters
            new_params: List of new parameters
            optimizer: The optimizer containing states
        """
        if optimizer is None or not hasattr(optimizer, 'state'):
            return
            
        logger.info(f"Transferring optimizer state for {self.name}")
        
        try:
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
            
            logger.info(f"Optimizer state transferred successfully for {self.name}")
        except Exception as e:
            logger.error(f"Error transferring optimizer state: {e}")

    
    def compute_importance_scores(self):
        """
        Compute importance scores for each LoRA dimension.
        Higher score means the dimension contributes more to the output.
        """
        try:
            # Get the current weights
            lora_A = self.module.lora_A.weight.detach()  # [r, in_features]
            lora_B = self.module.lora_B.weight.detach()  # [out_features, r]
            
            # Compute importance of each dimension
            # Method 1: Norm-based importance
            A_importance = torch.norm(lora_A, dim=1)  # [r]
            B_importance = torch.norm(lora_B, dim=0)  # [r]
            
            # Combine importances (elementwise product captures dimensions important in both)
            combined_importance = A_importance * B_importance  # [r]
            
            return combined_importance
        except Exception as e:
            logger.warning(f"Could not compute importance scores for {self.name}: {e}")
            return None
    
    def compute_grad_flow(self):
        """
        Compute gradient flow through this adapter.
        Returns a score of how much this adapter is learning.
        """
        try:
            # Check if gradients exist
            if (not hasattr(self.module.lora_A, 'weight') or
                not hasattr(self.module.lora_B, 'weight') or
                self.module.lora_A.weight.grad is None or
                self.module.lora_B.weight.grad is None):
                return 0.0
            
            # Get gradient magnitudes
            grad_A = self.module.lora_A.weight.grad.abs().mean().item()
            grad_B = self.module.lora_B.weight.grad.abs().mean().item()
            
            # Combined gradient flow (geometric mean)
            return (grad_A * grad_B) ** 0.5
        except Exception as e:
            logger.warning(f"Could not compute grad flow for {self.name}: {e}")
            return 0.0
