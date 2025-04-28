import torch
import logging
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HardExampleAmp")

@dataclass
class HardExampleAmplificationConfig:
    """Configuration for Hard Example Amplification."""
    # Hard example criteria
    loss_threshold_factor: float = 1.5  # Multiplier on mean loss to identify hard examples
    uncertainty_threshold: float = 0.3  # Threshold for model uncertainty to identify hard examples
    min_loss_percentile: float = 75.0   # Minimum loss percentile to consider an example "hard"
    
    # Sampling adjustments
    reweighting_factor: float = 3.0     # How much to increase probability of hard examples
    max_weight_ratio: float = 5.0       # Maximum weight ratio for hard to easy examples
    beta_factor: float = 0.5            # β factor in p(x) ∝ 1 + β(L(x) - μL) formula
    
    # Dynamic adjustment
    adaptive_thresholds: bool = True    # Adjust thresholds based on model progress
    adaptive_decay: float = 0.99        # Decay rate for adaptive thresholds
    
    # Hard example generation
    augment_hard_examples: bool = True  # Create augmented versions of hard examples
    max_augmentations: int = 3          # Maximum number of augmentations per hard example
    
    # History tracking
    history_window: int = 100           # Number of recent examples to track for statistics
    min_examples_seen: int = 50         # Minimum examples seen before starting amplification
    
    # Integration control
    integration_start_step: int = 200   # Step to start applying hard example amplification
    integration_warmup_steps: int = 100 # Gradual ramp-up of amplification effect
    
    # Logging
    log_frequency: int = 100            # Log stats every N steps
    verbose: bool = True                # Detailed logging


class HardExampleTracker:
    """
    Tracks example difficulty based on loss values, uncertainty, and other metrics.
    Identifies hard examples that should be amplified during training.
    """
    
    def __init__(
        self, 
        config: HardExampleAmplificationConfig = None,
        dataset_size: int = 10000
    ):
        """
        Initialize the hard example tracker.
        
        Args:
            config: Configuration for hard example amplification
            dataset_size: Size of the dataset being tracked
        """
        self.config = config or HardExampleAmplificationConfig()
        self.dataset_size = dataset_size
        self.example_features = {}  # idx -> feature vector
        self.example_losses = {}
        self.hard_examples = set()
        self.global_step = 0
        
        # Example tracking
        self.example_losses = {}  # idx -> list of losses
        self.example_uncertainties = {}  # idx -> list of uncertainties
        self.example_last_seen = {}  # idx -> step last seen
        
        # Recent losses for statistics
        self.recent_losses = deque(maxlen=self.config.history_window)
        self.recent_uncertainties = deque(maxlen=self.config.history_window)
        
        # Hard example tracking
        self.hard_examples = set()  # Set of hard example indices
        self.hard_example_history = []  # History of hard example sets
        
        # Loss statistics
        self.loss_mean = 0.0
        self.loss_std = 0.0
        self.uncertainty_mean = 0.0
        self.uncertainty_std = 0.0
        
        # Current thresholds
        self.current_loss_threshold = float('inf')  # Will be updated
        self.current_uncertainty_threshold = self.config.uncertainty_threshold
        
        # Step counting
        self.global_step = 0
        self.examples_seen = 0
        
        logger.info(f"Initialized HardExampleTracker with config: {self.config}")
    
    def update_example(
        self, 
        idx: int, 
        loss: float, 
        uncertainty: Optional[float] = None, 
        metrics: Optional[Dict[str, float]] = None,
        outputs: Optional[Any] = None,
        # uncertainty=None
    ):
        """
        Update tracking information for an example.
        
        Args:
            idx: Example index
            loss: Loss value for this example
            uncertainty: Optional uncertainty measure (e.g., entropy of predictions)
            metrics: Optional additional metrics for this example
        """
        # Initialize tracking for this example if needed
        if idx not in self.example_losses:
            self.example_losses[idx] = []
            self.example_uncertainties[idx] = []
        self.example_losses[idx].append(loss)
        if outputs is not None and hasattr(outputs, 'logits'):
            self.example_features[idx] = outputs.logits.detach().cpu().numpy()
            # self.examples_seen += 1
        if self.global_step % 100 == 0 and len(self.example_features) > 50:
            self._update_hard_examples()
        # Update loss history
        self.example_losses[idx].append(loss)
        self.recent_losses.append(loss)
        
        # Update uncertainty if provided
        if uncertainty is not None:
            self.example_uncertainties[idx].append(uncertainty)
            self.recent_uncertainties.append(uncertainty)
        
        # Record last seen step
        self.example_last_seen[idx] = self.global_step
        
        # Update statistics
        self._update_statistics()
        
        # Update hard example set if we have enough data
        if self.examples_seen >= self.config.min_examples_seen:
            self._update_hard_examples()
        
        if outputs is not None and hasattr(outputs, 'logits'):
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            epsilon = 1e-8
            example_entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=-1).mean().item()
            mi_score = self.predictive_entropy - example_entropy
            self.example_mi_scores[idx] = mi_score
    
    def _update_hard_examples(self):
        indices = list(self.example_features.keys())
        if len(indices) < 50:
            return

        # Compute feature matrix
        features = np.stack([self.example_features[idx] for idx in indices])
        losses = np.array([np.mean(self.example_losses[idx]) for idx in indices])

        # Target distribution: high-loss examples
        target_weights = np.exp(losses / np.std(losses))
        target_weights /= target_weights.sum()
        source_weights = np.ones(len(indices)) / len(indices)

        # Compute Wasserstein distance via Sinkhorn
        cost_matrix = ot.dist(features, features, metric='sqeuclidean')
        transport_plan = ot.sinkhorn(source_weights, target_weights, cost_matrix, reg=0.1)
        scores = transport_plan.sum(axis=1)
        threshold = np.percentile(scores, 75)
        self.hard_examples = set(indices[i] for i in range(len(indices)) if scores[i] > threshold)

        if self.config.verbose:
            logger.info(f"Updated hard examples: {len(self.hard_examples)} selected via OT")
    
    def update_batch(
        self, 
        indices: List[int], 
        losses: List[float],
        uncertainties: Optional[List[float]] = None
    ):
        """
        Update tracking information for a batch of examples.
        
        Args:
            indices: List of example indices
            losses: List of loss values
            uncertainties: Optional list of uncertainty values
        """
        if uncertainties is None:
            uncertainties = [None] * len(indices)
        
        for idx, loss, uncertainty in zip(indices, losses, uncertainties):
            self.update_example(idx, loss, uncertainty)
        
        self.global_step += 1
    
    def _update_statistics(self):
        """Update loss and uncertainty statistics."""
        if len(self.recent_losses) > 0:
            self.loss_mean = np.mean(self.recent_losses)
            self.loss_std = np.std(self.recent_losses) if len(self.recent_losses) > 1 else 0.0
            
            # Update loss threshold
            self.current_loss_threshold = self.loss_mean + self.config.loss_threshold_factor * self.loss_std
        
        if len(self.recent_uncertainties) > 0:
            self.uncertainty_mean = np.mean(self.recent_uncertainties)
            self.uncertainty_std = np.std(self.recent_uncertainties) if len(self.recent_uncertainties) > 1 else 0.0
            
            # Update uncertainty threshold if adaptive
            if self.config.adaptive_thresholds:
                target_threshold = self.uncertainty_mean + self.uncertainty_std
                self.current_uncertainty_threshold = (
                    self.config.adaptive_decay * self.current_uncertainty_threshold + 
                    (1 - self.config.adaptive_decay) * target_threshold
                )
    
    def _update_hard_examples(self):
        """Update the set of hard examples based on current statistics."""
        new_hard_examples = set()
        # new_hard_examples = set()
        mi_threshold = np.percentile(list(self.example_mi_scores.values()), 75) if self.example_mi_scores else 0.0
        for idx, mi_score in self.example_mi_scores.items():
            if mi_score > mi_threshold or idx in self.hard_examples:
                new_hard_examples.add(idx)
        self.hard_examples = new_hard_examples
        
        # Examine all tracked examples
        for idx, losses in self.example_losses.items():
            if not losses:
                continue
            
            # Use most recent loss
            latest_loss = losses[-1]
            
            # Check if this example exceeds the loss threshold
            is_hard_by_loss = latest_loss > self.current_loss_threshold
            
            # Check uncertainty if available
            is_hard_by_uncertainty = False
            if idx in self.example_uncertainties and self.example_uncertainties[idx]:
                latest_uncertainty = self.example_uncertainties[idx][-1]
                is_hard_by_uncertainty = latest_uncertainty > self.current_uncertainty_threshold
            
            # Combine criteria (loss is primary, uncertainty is secondary)
            if is_hard_by_loss or is_hard_by_uncertainty:
                new_hard_examples.add(idx)
        
        # Check if hard example set has changed significantly
        if len(new_hard_examples) > 0 and (
            len(self.hard_examples) == 0 or 
            len(new_hard_examples.symmetric_difference(self.hard_examples)) > 0.2 * len(self.hard_examples)
        ):
            # Record history
            self.hard_example_history.append({
                'step': self.global_step,
                'hard_examples': new_hard_examples.copy(),
                'loss_threshold': self.current_loss_threshold,
                'uncertainty_threshold': self.current_uncertainty_threshold,
                'num_hard': len(new_hard_examples)
            })
            
            # Update current set
            self.hard_examples = new_hard_examples
            
            if self.config.verbose:
                logger.info(f"Updated hard example set at step {self.global_step}: {len(self.hard_examples)} examples")
                logger.info(f"Loss threshold: {self.current_loss_threshold:.4f}, "
                           f"Uncertainty threshold: {self.current_uncertainty_threshold:.4f}")
    
    def get_sampling_weights(self, indices: List[int]) -> np.ndarray:
        """
        Get sampling weights for a set of examples based on their difficulty.
        
        Args:
            indices: List of example indices
            
        Returns:
            Array of sampling weights
        """
        weights = np.ones(len(indices))
        
        # Check if we have enough data and are past the integration start step
        if (self.examples_seen < self.config.min_examples_seen or 
            self.global_step < self.config.integration_start_step):
            return weights
        
        # Compute amplification factor based on warmup
        if self.global_step < self.config.integration_start_step + self.config.integration_warmup_steps:
            # Gradually ramp up amplification
            warmup_progress = (self.global_step - self.config.integration_start_step) / self.config.integration_warmup_steps
            amplification_factor = self.config.reweighting_factor * warmup_progress
        else:
            amplification_factor = self.config.reweighting_factor
        
        # Adjust weights based on whether examples are hard
        for i, idx in enumerate(indices):
            if idx in self.hard_examples:
                weights[i] *= amplification_factor
            
            # Additional adjustment based on exact loss difference from mean
            if idx in self.example_losses and self.example_losses[idx]:
                latest_loss = self.example_losses[idx][-1]
                # Apply formula: p(x) ∝ 1 + β(L(x) - μL)
                relative_loss = latest_loss - self.loss_mean
                if relative_loss > 0:  # Only boost examples with above-average loss
                    added_weight = 1.0 + self.config.beta_factor * relative_loss
                    weights[i] *= added_weight
        
        # Ensure weights are positive
        weights = np.maximum(weights, 0.1)
        
        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum() * len(weights)
        
        # Apply max weight ratio constraint
        if weights.max() > self.config.max_weight_ratio:
            weights = np.minimum(weights, self.config.max_weight_ratio)
            # Re-normalize
            if weights.sum() > 0:
                weights = weights / weights.sum() * len(weights)
        
        return weights
    
    def is_hard_example(self, idx: int) -> bool:
        """
        Check if an example is currently considered hard.
        
        Args:
            idx: Example index
            
        Returns:
            True if the example is hard, False otherwise
        """
        return idx in self.hard_examples
    
    def get_hard_example_indices(self) -> List[int]:
        """
        Get the indices of all current hard examples.
        
        Returns:
            List of hard example indices
        """
        return list(self.hard_examples)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current statistics about hard examples.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'global_step': self.global_step,
            'examples_seen': self.examples_seen,
            'num_hard_examples': len(self.hard_examples),
            'hard_example_percentage': len(self.hard_examples) / max(1, self.examples_seen) * 100,
            'loss_mean': self.loss_mean,
            'loss_std': self.loss_std,
            'loss_threshold': self.current_loss_threshold,
            'uncertainty_mean': self.uncertainty_mean,
            'uncertainty_std': self.uncertainty_std,
            'uncertainty_threshold': self.current_uncertainty_threshold
        }
import torch_geometric as pyg

class HGNN_Augmentor:
    def __init__(self, dataset, device):
        self.dataset = dataset
        self.device = device
        self.hgnn = pyg.nn.HyperGraphConv(in_channels=512, out_channels=512)

    def build_hypergraph(self, batch):
        node_features = []
        hyperedges = []
        for i, example in enumerate(batch['input_ids']):
            node_features.append(example)
            # Create hyperedge for each example (e.g., founder + 2 VCs)
            if torch.rand(1) < 0.1:
                hyperedges.append([i, (i + 1) % len(batch['input_ids']), (i + 2) % len(batch['input_ids'])])
        return pyg.data.HyperGraphData(
            x=torch.stack(node_features),
            hyperedge_index=torch.tensor(hyperedges, dtype=torch.long).t().to(self.device)
        )

    def augment(self, hypergraph, beta=0.1, noise_std=0.05):
        hyperedge_index = hypergraph.hyperedge_index
        mask = torch.rand(hyperedge_index.size(1)) > beta
        hyperedge_index = hyperedge_index[:, mask]
        noisy_features = hypergraph.x + torch.randn_like(hypergraph.x) * noise_std
        new_nodes = torch.cat([hypergraph.x, noisy_features], dim=0)
        new_edges = torch.cat([hyperedge_index, hyperedge_index + hypergraph.x.size(0)], dim=1)
        return pyg.data.HyperGraphData(x=new_nodes, hyperedge_index=new_edges)


class HardExampleAmplifier:
    """
    Implements hard example amplification during training.
    
    This component identifies training examples with high residual loss after epochs,
    increases their sampling probability, and aggressively resamples them to force
    model error correction.
    """
    
    def __init__(
        self,
        model,
        train_dataset: Dataset,
        config: HardExampleAmplificationConfig = None,
        device = None,
        dataset_size=10000
    ):
        """
        Initialize the hard example amplifier.
        
        Args:
            model: The model being trained
            train_dataset: Training dataset
            config: Configuration for hard example amplification
            device: Device to use ('cuda' or 'cpu')
        """
        self.model = model
        self.train_dataset = train_dataset
        self.config = config or HardExampleAmplificationConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_size = dataset_size
        self.example_mi_scores = {}  # idx -> mutual information scores
        self.predictive_entropy = 0.0  # H(Y | θ)
        # Initialize hard example tracker
        self.tracker = HardExampleTracker(
            config=self.config,
            dataset_size=len(train_dataset)
        )
        
        # Augmented examples storage
        self.augmented_examples = {}  # idx -> list of augmented versions
        
        # Metrics tracking
        self.metrics_history = []
        self.global_step = 0
        # self.gnn_augmentor = GNNAugmentor(train_dataset, self.device)
        self.hgnn_augmentor = HGNN_Augmentor(train_dataset, self.device)
    def _create_integrated_dataloader(self, batch_size: int = 8) -> DataLoader:
        """
        Create a dataloader that integrates curriculum learning and hard example amplification.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Integrated DataLoader
        """
        # For true integration, we need to combine the samplers/weights from both components
        
        # Get curriculum phase examples
        available_indices = self.curriculum.curriculum_dataset._get_available_examples()
        
        if not available_indices:
            # Fallback to all examples
            available_indices = list(range(len(self.train_dataset)))
        
        # Get curriculum weights
        curriculum_weights = self.curriculum.curriculum_dataset.sampling_weights[available_indices]
        
        # Get hard example weights for these indices
        hard_example_weights = self.hard_amplifier.tracker.get_sampling_weights(available_indices)
        
        # Combine weights (multiply for AND effect, emphasizing examples that are both
        # in the right curriculum phase AND identified as hard)
        combined_weights = curriculum_weights * hard_example_weights
        
        # Ensure valid weights
        if np.sum(combined_weights) <= 0:
            combined_weights = np.ones_like(combined_weights)
        
        # Create sampler
        sampler = WeightedRandomSampler(
            weights=combined_weights,
            num_samples=len(available_indices),
            replacement=True
        )
        
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
    
    # def train_step(self) -> Dict[str, Any]:
    #     """
    #     Execute a single training step with the integrated components.
        
    #     Returns:
    #         Dictionary of metrics for this step
    #     """
    #     # Execute STTT step
    #     sttt_metrics = self.sttt_cycle.step()
        
    #     # Extract per-example metrics for hard example tracking
    #     if self.sttt_cycle.current_phase == "S" and hasattr(self.sttt_cycle, 'current_batch_indices'):
    #         # Get indices and losses
    #         indices = self.sttt_cycle.current_batch_indices
    #         losses = self.sttt_cycle.current_batch_losses
    #         outputs = self.sttt_cycle.current_batch_outputs
            
    #         # Update hard example tracker
    #         if indices is not None and losses is not None and outputs is not None:
    #             self.hard_amplifier.update_with_batch_results(indices, self.sttt_cycle.current_batch, outputs, losses)
        
    #     # Update curriculum based on STTT metrics
    #     self.curriculum.update_curriculum(sttt_metrics)
        
    #     # Periodically update the integrated dataloader
    #     if self.global_step % 100 == 0:
    #         self.train_dataloader = self._create_integrated_dataloader()
    #         self.sttt_cycle.study_dataloader = self.train_dataloader
    #         self.sttt_cycle.study_iter = iter(self.train_dataloader)
        
    #     # Combine metrics
    #     combined_metrics = {
    #         'global_step': self.global_step,
    #         **sttt_metrics,
    #         'curriculum_phase': self.curriculum.curriculum_dataset.current_phase,
    #         'hard_examples': len(self.hard_amplifier.tracker.hard_examples)
    #     }
        
    #     # Record metrics
    #     self.metrics_history.append(combined_metrics)
        
    #     # Update step counter
    #     self.global_step += 1
        
    #     return combined_metrics
    
    def train_step(self):
        metrics = self.sttt_cycle.step()
        self.hard_amplifier.update_with_batch_results(
            indices=self.sttt_cycle.current_batch_indices,
            batch=self.sttt_cycle.current_batch,
            outputs=self.sttt_cycle.current_batch_outputs,
            losses=self.sttt_cycle.current_batch_losses
        )
        self.curriculum.update_curriculum(metrics)
        if self.sttt_cycle.global_step % 100 == 0:
            self.sttt_cycle.study_dataloader = self.curriculum.get_dataloader(batch_size=32)
            self.sttt_cycle.study_iter = iter(self.sttt_cycle.study_dataloader)
        return metrics
    
    def train(self, num_steps: int) -> Dict[str, Any]:
        """
        Train the model for a specified number of steps.
        
        Args:
            num_steps: Number of steps to train for
            
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Starting integrated STTT-Curriculum-HardExample training for {num_steps} steps")
        
        all_metrics = []
        for _ in range(num_steps):
            step_metrics = self.train_step()
            all_metrics.append(step_metrics)
            
            # Logging
            if self.global_step % 10 == 0:
                logger.info(f"Step {self.global_step}: "
                           f"STTT Phase={step_metrics['phase']}, "
                           f"Curriculum Phase={step_metrics['curriculum_phase']}, "
                           f"Hard Examples={step_metrics['hard_examples']}, "
                           f"Loss={step_metrics.get(f'{step_metrics['phase'].lower()}_loss', 'N/A')}")
        
        # Compile final metrics
        final_metrics = {
            'total_steps': num_steps,
            'sttt_metrics': self.sttt_cycle.get_metrics(),
            'curriculum_state': self.curriculum.get_curriculum_state(),
            'hard_example_stats': self.hard_amplifier.get_statistics(),
            'metrics_history': self.metrics_history
        }
        
        logger.info(f"Integrated training completed")
        return final_metrics
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current training metrics."""
        return {
            'global_step': self.global_step,
            'sttt_metrics': self.sttt_cycle.get_metrics(),
            'curriculum_state': self.curriculum.get_curriculum_state(),
            'hard_example_stats': self.hard_amplifier.get_statistics()
        } 
        history = []
        self.global_step = 0
        
        logger.info(f"Initialized HardExampleAmplifier for {len(train_dataset)} training examples")
    
    def compute_example_uncertainty(self, outputs: Any, is_classification: bool = True, num_samples: int = 10) -> float:
        """
        Compute uncertainty measure for model outputs.
        
        Args:
            outputs: Model outputs
            is_classification: Whether the task is classification (vs regression)
            
        Returns:
            Uncertainty score (higher means more uncertain)
        """
        # if is_classification and hasattr(outputs, 'logits'):
        #     # Use entropy of the predicted distribution as uncertainty measure
        #     logits = outputs.logits
        #     probs = torch.nn.functional.softmax(logits, dim=-1)
            
        #     # Add small epsilon to avoid log(0)
        #     epsilon = 1e-8
        #     entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=-1)
            
        #     # Normalize by maximum possible entropy (log of num_classes)
        #     num_classes = probs.size(-1)
        #     max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float))
        #     normalized_entropy = entropy / max_entropy
            
        #     return normalized_entropy.mean().item()
        if is_classification and hasattr(outputs, 'logits'):
            self.model.eval()
            with torch.no_grad():
                probs = []
                for _ in range(num_samples):
                    logits = self.model(batch['input_ids'], dropout=True)  # Assume dropout-enabled model
                    probs.append(torch.nn.functional.softmax(logits, dim=-1))
                probs = torch.stack(probs)
                mean_probs = probs.mean(dim=0)
                epistemic_uncertainty = ((probs - mean_probs) ** 2).mean(dim=0).mean().item()
            self.model.train()
            return epistemic_uncertainty
        return 0.5
        # Fallback: use inverse confidence/probability as uncertainty
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            confidence = probs.max(dim=-1)[0]
            return (1.0 - confidence.mean()).item()
        
        # If no appropriate outputs, return default uncertainty
        return 0.5
    
    def augment_hard_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an augmented version of a hard example.
        This is a simple implementation - in practice, use more sophisticated augmentations.
        
        Args:
            example: Original example
            
        Returns:
            Augmented example
        """
        # Clone the example to avoid modifying the original
        augmented = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in example.items()}
        
        # For text inputs, we could:
        # 1. Swap/replace some tokens
        # 2. Add noise to embeddings
        # 3. Apply dropout to input tokens
        
        if 'input_ids' in augmented and isinstance(augmented['input_ids'], torch.Tensor):
            # Simple augmentation: randomly mask 5-10% of tokens
            mask_prob = np.random.uniform(0.05, 0.1)
            batch_size = augmented['input_ids'].size(0) if augmented['input_ids'].dim() > 1 else 1
            seq_len = augmented['input_ids'].size(-1)
            
            # Create mask
            mask = torch.rand(batch_size, seq_len) < mask_prob
            
            # Find a suitable mask token (assuming 0 is padding, 1 might be mask token for some models)
            mask_token_id = 1  # Default
            
            # Apply masking
            for i in range(batch_size):
                for j in range(seq_len):
                    if mask[i, j]:
                        augmented['input_ids'][i, j] = mask_token_id
        
        # batch = {k: v.unsqueeze(0) for k, v in example.items()}
        batch = {k: v.unsqueeze(0) for k, v in example.items()}
        hypergraph = self.hgnn_augmentor.build_hypergraph(batch)
        augmented_hypergraph = self.hgnn_augmentor.augment(hypergraph)
        return {'input_ids': augmented_hypergraph.x}
        # return augmented
    
    def update_predictive_entropy(self, batch_outputs):
        probs = torch.nn.functional.softmax(batch_outputs.logits, dim=-1)
        epsilon = 1e-8
        self.predictive_entropy = -torch.sum(probs.mean(dim=0) * torch.log(probs.mean(dim=0) + epsilon)).item()
    
    def update_with_batch_results(
        self, 
        indices: List[int],
        batch: Dict[str, torch.Tensor],
        outputs: Any,
        losses: torch.Tensor
    ):
        """
        Update hard example tracking with batch results.
        
        Args:
            indices: Indices of examples in the batch
            batch: Input batch
            outputs: Model outputs
            losses: Loss values for each example
        """
        # Compute uncertainty
        uncertainty = self.compute_example_uncertainty(outputs)
        
        # Convert losses to numpy array
        losses_np = losses.detach().cpu().numpy()
        
        # For simplicity, use the same uncertainty for all examples in batch
        uncertainties = [uncertainty] * len(indices)
        
        # Update tracker with batch results
        self.tracker.update_batch(indices, losses_np, uncertainties)
        
        # Generate augmentations for newly identified hard examples
        if self.config.augment_hard_examples and self.global_step >= self.config.integration_start_step:
            for i, idx in enumerate(indices):
                if self.tracker.is_hard_example(idx) and idx not in self.augmented_examples:
                    # Get the example
                    example = {k: v[i:i+1] for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    
                    # Create augmentations
                    self.augmented_examples[idx] = []
                    for _ in range(self.config.max_augmentations):
                        augmented = self.augment_hard_example(example)
                        self.augmented_examples[idx].append(augmented)
                    
                    if self.config.verbose:
                        logger.info(f"Created {len(self.augmented_examples[idx])} augmentations for hard example {idx}")
        
        # Update step counter
        self.global_step += 1
        
        # Log statistics periodically
        if self.global_step % self.config.log_frequency == 0:
            stats = self.tracker.get_statistics()
            self.metrics_history.append(stats)
            
            if self.config.verbose:
                logger.info(f"Hard example stats at step {self.global_step}: "
                           f"{stats['num_hard_examples']} hard examples "
                           f"({stats['hard_example_percentage']:.1f}% of seen examples)")
                logger.info(f"Loss threshold: {stats['loss_threshold']:.4f}, "
                           f"Mean loss: {stats['loss_mean']:.4f}, "
                           f"StdDev: {stats['loss_std']:.4f}")
    
    def get_weighted_sampler(self) -> Sampler:
        """
        Get a sampler that emphasizes hard examples.
        
        Returns:
            PyTorch sampler
        """
        # Get indices of all examples
        indices = list(range(len(self.train_dataset)))
        
        # Get weights based on difficulty
        weights = self.tracker.get_sampling_weights(indices)
        
        # Create weighted sampler
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
    
    def get_hard_example_batch(self, batch_size: int) -> Tuple[List[int], Dict[str, torch.Tensor]]:
        """
        Create a batch focused on hard examples.
        
        Args:
            batch_size: Desired batch size
            
        Returns:
            Tuple of (indices, batch)
        """
        # Get hard example indices
        hard_indices = self.tracker.get_hard_example_indices()
        
        if not hard_indices:
            # Fallback to random indices if no hard examples
            indices = np.random.choice(len(self.train_dataset), size=batch_size)
        else:
            # Sample from hard examples, with replacement if needed
            if len(hard_indices) < batch_size:
                indices = np.random.choice(hard_indices, size=batch_size, replace=True)
            else:
                indices = np.random.choice(hard_indices, size=batch_size, replace=False)
        
        # Create batch
        batch = {}
        for i, idx in enumerate(indices):
            example = self.train_dataset[idx]
            
            # First example - initialize batch
            if i == 0:
                batch = {k: [v] for k, v in example.items()}
            else:
                # Add to batch
                for k, v in example.items():
                    batch[k].append(v)
        
        # Convert to tensors
        batch = {k: torch.stack(v) if isinstance(v[0], torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Move to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        return indices.tolist(), batch
    
    def get_augmented_hard_example_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Create a batch from augmented hard examples.
        
        Args:
            batch_size: Desired batch size
            
        Returns:
            Batch of augmented examples
        """
        if not self.augmented_examples:
            # Fallback to regular hard example batch if no augmentations
            _, batch = self.get_hard_example_batch(batch_size)
            return batch
        
        # Get all hard examples with augmentations
        hard_indices = [idx for idx in self.tracker.get_hard_example_indices() 
                       if idx in self.augmented_examples]
        
        if not hard_indices:
            # Fallback to regular hard example batch
            _, batch = self.get_hard_example_batch(batch_size)
            return batch
        
        # Sample hard examples
        sampled_indices = np.random.choice(hard_indices, size=min(batch_size, len(hard_indices)), replace=False)
        
        # Create batch from augmentations
        augmented_batch = {}
        example_count = 0
        
        for idx in sampled_indices:
            # Get augmentations for this example
            augmentations = self.augmented_examples[idx]
            
            if not augmentations:
                continue
                
            # Sample an augmentation
            augmentation = np.random.choice(augmentations)
            
            # Add to batch
            if not augmented_batch:
                # First example - initialize batch
                augmented_batch = {k: v for k, v in augmentation.items()}
                example_count = 1
            else:
                # Concat to batch
                for k, v in augmentation.items():
                    if isinstance(v, torch.Tensor):
                        augmented_batch[k] = torch.cat([augmented_batch[k], v], dim=0)
                
                example_count += 1
            
            # Break if we have enough examples
            if example_count >= batch_size:
                break
        
        # If we don't have enough examples, pad with regular hard examples
        if example_count < batch_size:
            remaining = batch_size - example_count
            _, remaining_batch = self.get_hard_example_batch(remaining)
            
            # Combine batches
            for k, v in remaining_batch.items():
                if k in augmented_batch and isinstance(v, torch.Tensor):
                    augmented_batch[k] = torch.cat([augmented_batch[k], v], dim=0)
                else:
                    augmented_batch[k] = v
        
        return augmented_batch
    
    def get_dataloader(self, batch_size: int, num_workers: int = 4) -> DataLoader:
        """
        Get a dataloader that emphasizes hard examples.
        
        Args:
            batch_size: Batch size
            num_workers: Number of worker processes
            
        Returns:
            PyTorch DataLoader
        """
        sampler = self.get_weighted_sampler()
        
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current statistics about hard examples.
        
        Returns:
            Dictionary of statistics
        """
        stats = self.tracker.get_statistics()
        stats.update({
            'global_step': self.global_step,
            'num_augmented_examples': len(self.augmented_examples),
            'total_augmentations': sum(len(augs) for augs in self.augmented_examples.values())
        })
        return stats


class IntegratedHardExampleTrainer:
    """
    Integrates Hard Example Amplification with STTT Cycle and Dynamic Curriculum.
    """
    
    def __init__(
        self,
        model,
        optimizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
        hard_example_config=None,
        curriculum_config=None,
        sttt_config=None,
        device=None
    ):
        """
        Initialize the integrated trainer.
        
        Args:
            model: The model to train
            optimizer: The optimizer to use for training
            train_dataset: Training dataset
            val_dataset: Validation dataset
            hard_example_config: Configuration for hard example amplification
            curriculum_config: Configuration for dynamic curriculum
            sttt_config: Configuration for STTT cycle
            device: Device to use ('cuda' or 'cpu')
        """
        from sttt_cycle import STTTConfig, STTTCycle
        from dynamic_curriculum import DynamicCurriculumConfig, DynamicCurriculumConstructor
        
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize hard example amplifier
        self.hard_amplifier = HardExampleAmplifier(model, train_dataset, hard_example_config, device)
        self.curriculum = DynamicCurriculumConstructor(model, train_dataset, val_dataset, curriculum_config, device=device)
        self.sttt_cycle = STTTCycle(
            model=model,
            optimizer=optimizer,
            study_dataloader=self.curriculum.get_dataloader(batch_size=32),
            t1_dataloader=DataLoader(val_dataset, batch_size=32),
            t2_generator=self.curriculum.t2_generator,
            t3_generator=lambda batch: self.hard_amplifier.get_augmented_hard_example_batch(len(batch['input_ids'])),
            config=sttt_config,
            device=device
        )
        
        # Initialize curriculum
        self.curriculum_config = curriculum_config or DynamicCurriculumConfig()
        self.curriculum = DynamicCurriculumConstructor(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=self.curriculum_config,
            device=self.device
        )
        
        # Get weighted dataloader (combining curriculum and hard example emphasis)
        self.train_dataloader = self._create_integrated_dataloader()
        
        # Initialize validation dataloader
        self.val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=self.curriculum_config.phase_duration // 10,
            shuffle=False
        )
        
        # Initialize STTT cycle
        self.sttt_config = sttt_config or STTTConfig()
        
        # Custom generators for T2 and T3 phases that incorporate hard examples
        def t2_generator(batch):
            # Use curriculum's higher difficulty examples
            phase = min(self.curriculum.curriculum_dataset.current_phase + 1, 
                       self.curriculum_config.num_curriculum_phases - 1)
            
            harder_indices = self.curriculum.curriculum_dataset.phase_examples[phase]
            if not harder_indices:
                return batch  # Fallback
            
            # Sample from harder examples
            sampled_indices = np.random.choice(harder_indices, size=len(batch['input_ids']))
            harder_batch = {
                k: torch.stack([
                    torch.tensor(self.train_dataset[idx][k]) for idx in sampled_indices
                ]).to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            return harder_batch
        
        def t3_generator(batch):
            # For T3 (adversarial), use hard examples with augmentation
            # This creates a true adversarial test
            augmented_batch = self.hard_amplifier.get_augmented_hard_example_batch(
                batch_size=len(batch['input_ids'])
            )
            
            # If no augmented batch, fallback to hardest curriculum examples
            if not augmented_batch:
                hardest_phase = self.curriculum_config.num_curriculum_phases - 1
                hardest_indices = self.curriculum.curriculum_dataset.phase_examples[hardest_phase]
                
                if not hardest_indices:
                    return batch  # Fallback
                
                # Sample from hardest examples
                sampled_indices = np.random.choice(hardest_indices, size=len(batch['input_ids']))
                adversarial_batch = {
                    k: torch.stack([
                        torch.tensor(self.train_dataset[idx][k]) for idx in sampled_indices
                    ]).to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                return adversarial_batch
            
            return augmented_batch
        
        self.sttt_cycle = STTTCycle(
            model=model,
            optimizer=optimizer,
            study_dataloader=self.train_dataloader,
            t1_dataloader=self.val_dataloader,
            t2_generator=t2_generator,
            t3_generator=t3_generator,
            config=self.sttt_config,
            device=self.device
        )
        
        # Metrics tracking
        self.metrics