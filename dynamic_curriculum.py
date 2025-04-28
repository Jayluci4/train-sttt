import torch
import torch.nn as nn
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
logger = logging.getLogger("DynamicCurriculum")

@dataclass
class DynamicCurriculumConfig:
    """Configuration for Dynamic Curriculum Construction."""
    # Curriculum phases
    num_curriculum_phases: int = 5  # Number of difficulty levels
    phase_duration: int = 1000      # Base number of steps per phase
    
    # Data difficulty scoring
    difficulty_metric: str = "loss"   # One of {"loss", "confidence", "entropy", "manual"}
    min_difficulty: float = 0.0       # Minimum difficulty score
    max_difficulty: float = 1.0       # Maximum difficulty score
    
    # Phase progression
    auto_progress: bool = True              # Automatically progress through phases
    progress_threshold: float = 0.1         # Loss improvement threshold for progression
    plateau_patience: int = 5               # Number of evaluations with minimal improvement before progression
    min_steps_per_phase: int = 500          # Minimum steps to stay in a phase
    
    # Data sampling
    adaptive_sampling: bool = True          # Use weighted sampling based on loss/difficulty
    sampling_strategy: str = "difficulty"   # One of {"difficulty", "loss", "combined"}
    sampling_temperature: float = 1.0       # Temperature for softmax sampling weights
    sampling_update_freq: int = 100         # Frequency to update sampling weights
    
    # Example mixing
    mix_ratio: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0, 0.0])  # Initial mix ratio of phases
    target_mix_ratio: List[float] = field(default_factory=lambda: [0.2, 0.2, 0.2, 0.2, 0.2])  # Target mix ratio
    mix_ratio_decay: float = 0.995          # Rate at which mix ratio approaches target
    
    # Metrics tracking
    track_per_example_metrics: bool = True  # Track metrics for each example
    metrics_memory_size: int = 10000        # Maximum number of examples to track
    
    # Integration with STTT
    sttt_integration: bool = True           # Integrate with STTT cycle
    sttt_weight: float = 0.5                # Weight of STTT metrics in difficulty scoring
    
    # Logging
    log_frequency: int = 100                # Log stats every N steps
    verbose: bool = True                    # Detailed logging


class CurriculumDataset(Dataset):
    """
    Wrapper dataset that implements curriculum learning by managing access to examples
    based on their difficulty scores and the current curriculum phase.
    """
    
    def __init__(
        self, 
        base_dataset: Dataset,
        config: DynamicCurriculumConfig = None,
        example_difficulties: Optional[List[float]] = None,
        device = None
    ):
        """
        Initialize the curriculum dataset.
        
        Args:
            base_dataset: The underlying dataset to wrap
            config: Configuration for curriculum learning
            example_difficulties: Optional pre-computed difficulty scores for each example
            device: Device to use ('cuda' or 'cpu')
        """
        self.base_dataset = base_dataset
        self.config = config or DynamicCurriculumConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Example difficulty scores (0.0 = easiest, 1.0 = hardest)
        if example_difficulties is not None:
            self.example_difficulties = example_difficulties
        else:
            # Initialize with default uniform scores if not provided
            self.example_difficulties = np.ones(len(base_dataset)) * 0.5
        
        # Per-example metrics tracking
        self.example_metrics = defaultdict(list)
        self.example_loss_history = defaultdict(lambda: deque(maxlen=10))
        
        # Curriculum state
        self.current_phase = 0  # Start with easiest phase
        self.phase_step = 0
        self.global_step = 0
        
        # Phase progression tracking
        self.phase_losses = {i: [] for i in range(self.config.num_curriculum_phases)}
        self.plateau_counter = 0
        self.best_loss = float('inf')
        
        # Sample weights for adaptive sampling
        self.sampling_weights = np.ones(len(base_dataset))
        
        # Phase boundaries - divide difficulty range into phases
        self.phase_boundaries = np.linspace(
            self.config.min_difficulty, 
            self.config.max_difficulty, 
            self.config.num_curriculum_phases + 1
        )
        
        # Identify examples for each phase
        self.phase_examples = self._assign_examples_to_phases()
        
        # Current mix ratio
        self.current_mix_ratio = self.config.mix_ratio.copy()
        
        logger.info(f"Initialized CurriculumDataset with {len(base_dataset)} examples and {self.config.num_curriculum_phases} phases")
        logger.info(f"Phase boundaries: {self.phase_boundaries}")
        logger.info(f"Examples per phase: {[len(indices) for _, indices in self.phase_examples.items()]}")
    
    def _assign_examples_to_phases(self) -> Dict[int, List[int]]:
        """
        Assign examples to curriculum phases based on difficulty scores.
        
        Returns:
            Dictionary mapping phase index to list of example indices
        """
        phase_examples = defaultdict(list)
        
        for idx, difficulty in enumerate(self.example_difficulties):
            # Find the appropriate phase for this example
            for phase in range(self.config.num_curriculum_phases):
                lower_bound = self.phase_boundaries[phase]
                upper_bound = self.phase_boundaries[phase + 1]
                
                if lower_bound <= difficulty < upper_bound or (
                    phase == self.config.num_curriculum_phases - 1 and difficulty == upper_bound
                ):
                    phase_examples[phase].append(idx)
                    break
        
        return phase_examples
    
    def _get_available_examples(self) -> List[int]:
        """
        Get the indices of examples available in the current curriculum phase.
        
        Returns:
            List of example indices
        """
        available_indices = []
        
        # Include examples from current and previous phases based on mix ratio
        for phase in range(self.config.num_curriculum_phases):
            if self.current_mix_ratio[phase] > 0 and phase <= self.current_phase:
                available_indices.extend(self.phase_examples[phase])
        
        return available_indices
    
    def _update_sampling_weights(self):
        """Update sampling weights based on difficulty and loss history."""
        if not self.config.adaptive_sampling or self.global_step % self.config.sampling_update_freq != 0:
            return
        
        # Reset weights
        self.sampling_weights = np.ones(len(self.base_dataset))
        
        if self.config.sampling_strategy == "difficulty":
            # Weight directly by difficulty (harder examples get higher weights)
            self.sampling_weights = self.example_difficulties
            
        elif self.config.sampling_strategy == "loss":
            # Weight by loss (higher loss examples get higher weights)
            for idx, losses in self.example_loss_history.items():
                if losses:  # Only update if we have loss history
                    avg_loss = sum(losses) / len(losses)
                    self.sampling_weights[idx] = avg_loss
                    
        elif self.config.sampling_strategy == "combined":
            # Combine difficulty and loss
            for idx, losses in self.example_loss_history.items():
                if losses:  # Only update if we have loss history
                    avg_loss = sum(losses) / len(losses)
                    self.sampling_weights[idx] = (self.example_difficulties[idx] + avg_loss) / 2
        
        # Apply temperature scaling
        if self.config.sampling_temperature != 1.0:
            self.sampling_weights = np.power(self.sampling_weights, 1.0 / self.config.sampling_temperature)
        
        # Normalize weights
        if np.sum(self.sampling_weights) > 0:
            self.sampling_weights = self.sampling_weights / np.sum(self.sampling_weights)
    
    def _update_mix_ratio(self):
        """Gradually update mix ratio towards target."""
        for i in range(len(self.current_mix_ratio)):
            self.current_mix_ratio[i] = (
                self.current_mix_ratio[i] * self.config.mix_ratio_decay + 
                self.config.target_mix_ratio[i] * (1 - self.config.mix_ratio_decay)
            )
        
        # Normalize to ensure sum is 1
        self.current_mix_ratio = [r / sum(self.current_mix_ratio) for r in self.current_mix_ratio]
    
    def update_curriculum_state(self, metrics: Dict[str, Any] = None):
        """
        Update curriculum state based on training metrics.
        
        Args:
            metrics: Dictionary of training metrics
        """
        self.global_step += 1
        self.phase_step += 1
        
        # Update example-specific metrics if provided
        if metrics and 'example_metrics' in metrics:
            example_metrics = metrics['example_metrics']
            for idx, loss in example_metrics.items():
                if isinstance(idx, int) and 0 <= idx < len(self.base_dataset):
                    self.example_loss_history[idx].append(loss)
                    
                    if self.config.track_per_example_metrics:
                        self.example_metrics[idx].append(loss)
        
        # Update sampling weights
        self._update_sampling_weights()
        
        # Update mix ratio
        self._update_mix_ratio()
        
        # Check for phase progression
        if (metrics and self.config.auto_progress and 
            self.phase_step >= self.config.min_steps_per_phase):
            
            # Get current validation loss
            current_loss = None
            if 't1_loss' in metrics:  # STTT integration
                current_loss = metrics['t1_loss']
            elif 'val_loss' in metrics:
                current_loss = metrics['val_loss']
            
            if current_loss is not None:
                # Record loss for current phase
                self.phase_losses[self.current_phase].append(current_loss)
                
                # Check for improvement
                improvement = self.best_loss - current_loss
                relative_improvement = improvement / self.best_loss if self.best_loss > 0 else 0
                
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                
                # Check for plateau
                if relative_improvement < self.config.progress_threshold:
                    self.plateau_counter += 1
                else:
                    self.plateau_counter = 0
                
                # Progress to next phase if plateau reached
                if (self.plateau_counter >= self.config.plateau_patience and 
                    self.current_phase < self.config.num_curriculum_phases - 1):
                    
                    self.current_phase += 1
                    self.phase_step = 0
                    self.plateau_counter = 0
                    
                    logger.info(f"Progressing to curriculum phase {self.current_phase} at step {self.global_step}")
                    logger.info(f"New phase includes {len(self.phase_examples[self.current_phase])} additional examples")
                    logger.info(f"Current mix ratio: {[round(r, 2) for r in self.current_mix_ratio]}")
        
        # Logging
        if self.global_step % self.config.log_frequency == 0 and self.config.verbose:
            logger.info(f"Curriculum step {self.global_step} (Phase {self.current_phase}, step {self.phase_step})")
            logger.info(f"Mix ratio: {[round(r, 2) for r in self.current_mix_ratio]}")
            logger.info(f"Available examples: {sum(len(self.phase_examples[p]) for p in range(self.current_phase + 1))}")
    
    def update_example_difficulty(self, idx: int, difficulty: float):
        """
        Update difficulty score for a specific example.
        
        Args:
            idx: Example index
            difficulty: New difficulty score (0.0 to 1.0)
        """
        if 0 <= idx < len(self.base_dataset):
            # Update difficulty score
            self.example_difficulties[idx] = difficulty
            
            # Re-assign example to appropriate phase
            for phase in range(self.config.num_curriculum_phases):
                # Remove from current phase assignment
                for p, indices in self.phase_examples.items():
                    if idx in indices:
                        self.phase_examples[p].remove(idx)
                
                # Add to appropriate phase
                lower_bound = self.phase_boundaries[phase]
                upper_bound = self.phase_boundaries[phase + 1]
                
                if lower_bound <= difficulty < upper_bound or (
                    phase == self.config.num_curriculum_phases - 1 and difficulty == upper_bound
                ):
                    self.phase_examples[phase].append(idx)
                    break
    
    def get_sampler(self) -> Sampler:
        """
        Get a sampler for the curriculum dataset based on the current phase.
        
        Returns:
            PyTorch sampler
        """
        # Get indices of available examples
        available_indices = self._get_available_examples()
        
        if not available_indices:
            # Fallback to all examples
            available_indices = list(range(len(self.base_dataset)))
        
        # Get weights for available examples
        weights = self.sampling_weights[available_indices]
        
        # Create weighted sampler
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(available_indices),
            replacement=True
        )
    
    def __len__(self):
        """Return the length of the dataset."""
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        """Get an item from the base dataset."""
        return self.base_dataset[idx]


import torch
import torch.nn as nn
import torchdiffeq
from torch.distributions import Normal
import numpy as np

class CurriculumCNF(nn.Module):
    def __init__(self, num_phases, dataset_size, hidden_dim=64):
        super().__init__()
        self.num_phases = num_phases
        self.dataset_size = dataset_size
        state_dim = num_phases + dataset_size
        self.net = nn.Sequential(
            nn.Linear(state_dim + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        self.trace_estimator = lambda x: torch.autograd.grad(
            self.net(x).sum(), x, create_graph=True
        )[0].sum(dim=-1)

    def forward(self, t, state, metrics):
        with torch.enable_grad():
            inputs = torch.cat([state, metrics], dim=-1)
            velocity = self.net(inputs)
            divergence = self.trace_estimator(inputs)
        return velocity, divergence

    
class WGANGenerator(nn.Module):
    def __init__(self, input_dim, difficulty_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + difficulty_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z, difficulty):
        inputs = torch.cat([z, difficulty], dim=-1)
        return self.net(inputs)

class WGANDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical

class CausalVAE(nn.Module):
    def __init__(self, sector_dim=10, traction_dim=1, metadata_dim=1, match_dim=1, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(sector_dim + traction_dim + metadata_dim + match_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * latent_dim)  # Mean and log-variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + sector_dim, 128),
            nn.ReLU(),
            nn.Linear(128, traction_dim + metadata_dim + match_dim)
        )
        self.sector_decoder = nn.Linear(latent_dim, sector_dim)

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    def decode(self, z, sector):
        inputs = torch.cat([z, sector], dim=-1)
        outputs = self.decoder(inputs)
        traction, metadata, match = outputs.split([1, 1, 1], dim=-1)
        sector_logits = self.sector_decoder(z)
        return sector_logits, traction, metadata, match

    def forward(self, x):
        sector, traction, metadata, match = x.split([10, 1, 1, 1], dim=-1)
        mu, logvar = self.encode(x)
        z = Normal(mu, logvar.exp().sqrt()).rsample()
        sector_logits, traction, metadata, match = self.decode(z, sector)
        return sector_logits, traction, metadata, match, mu, logvar

class CausalCounterfactualGenerator:
    def __init__(self, dataset, device):
        self.dataset = dataset
        self.device = device
        self.cvae = CausalVAE().to(device)
        self.optimizer = torch.optim.Adam(self.cvae.parameters(), lr=1e-3)

    def train(self, batch):
        self.cvae.zero_grad()
        x = torch.cat([batch['sector'], batch['traction'], batch['metadata'], batch['match']], dim=-1)
        sector_logits, traction, metadata, match, mu, logvar = self.cvae(x)
        recon_loss = (
            Categorical(logits=sector_logits).log_prob(batch['sector'].argmax(dim=-1)).mean() +
            Normal(traction, 0.1).log_prob(batch['traction']).mean() +
            Normal(metadata, 0.1).log_prob(batch['metadata']).mean() +
            Normal(match, 0.1).log_prob(batch['match']).mean()
        )
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        loss = -recon_loss + kl_loss
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def generate_counterfactual(self, example, intervention):
        x = torch.cat([example['sector'], example['traction'], example['metadata'], example['match']], dim=-1)
        mu, logvar = self.cvae.encode(x)
        z = Normal(mu, logvar.exp().sqrt()).rsample()
        sector_intervention = torch.zeros_like(example['sector'])
        sector_intervention[:, intervention['sector']] = 1
        _, traction, metadata, match = self.cvae.decode(z, sector_intervention)
        return {
            'sector': sector_intervention,
            'traction': traction,
            'metadata': metadata,
            'match': match
        }

# class DynamicCurriculumConstructor:
#     def __init__(self, model, train_dataset, val_dataset=None, config=None, difficulty_scorer=None, device=None):
        # ... existing initialization ...


        

class DynamicCurriculumConstructor:
    """
    Implements dynamic curriculum construction for neural network training.
    
    This component starts with "easy founders" (clear matches) and gradually introduces:
    - Cross-sector founders
    - Founders with missing fields
    - Conflicting founder signals
    
    The curriculum is adjusted dynamically based on training loss velocity.
    """
    
    def __init__(
        self,
        model,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        config: DynamicCurriculumConfig = None,
        difficulty_scorer: Optional[Callable] = None,
        device = None,
        # difficulty_scorer=None
    ):
        """
        Initialize the Dynamic Curriculum Constructor.
        
        Args:
            model: The model being trained
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            config: Configuration for curriculum learning
            difficulty_scorer: Optional function to score example difficulty
            device: Device to use ('cuda' or 'cpu')
        """
        self.model = model
        self.base_train_dataset = train_dataset
        self.val_dataset = val_dataset
        # self.config = config or DynamicCurriculumConfig()
        # self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or DynamicCurriculumConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_size = len(train_dataset)

        # Initialize curriculum state
        initial_mix_ratio = torch.tensor(self.config.mix_ratio, device=self.device)
        initial_difficulties = torch.ones(self.dataset_size, device=self.device) * 0.5
        self.state = torch.cat([initial_mix_ratio, initial_difficulties])

        # Neural ODE for state evolution
        # Continuous Normalizing Flow
        self.cnf = CurriculumCNF(
            num_phases=self.config.num_curriculum_phases, 
            dataset_size=self.dataset_size
            )
        self.global_step = 0
        # Difficulty scoring function
        # self.difficulty_scorer = difficulty_scorer or self._default_difficulty_scorer
        # self.causal_generator = CausalCounterfactualGenerator(train_dataset, self.device)
        # # Compute initial difficulty scores
        # example_difficulties = self._compute_initial_difficulties()
        
        # Create curriculum dataset
        self.curriculum_dataset = CurriculumDataset(
            base_dataset=train_dataset,
            config=self.config,
            # example_difficulties=example_difficulties,
            example_difficulties=initial_difficulties.cpu().numpy(),
            device=self.device
        )
        
        # Metrics tracking
        self.metrics_history = []
        self.global_step = 0
        
        logger.info(f"Initialized DynamicCurriculumConstructor for {len(train_dataset)} training examples")
        if val_dataset:
            logger.info(f"Validation dataset: {len(val_dataset)} examples")
        
        self.wgan_generator = WGANGenerator(input_dim=64, difficulty_dim=1, output_dim=512)  # Adjust output_dim
        self.wgan_discriminator = WGANDiscriminator(input_dim=512)
        self.wgan_optimizer_g = torch.optim.Adam(self.wgan_generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
        self.wgan_optimizer_d = torch.optim.Adam(self.wgan_discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))
        self.wgan_lambda = 10.0

        self.causal_generator = CausalCounterfactualGenerator(train_dataset, self.device)

    def t2_generator(self, batch):
        counterfactual_batch = {}
        for i in range(len(batch['input_ids'])):
            example = {k: v[i:i+1] for k, v in batch.items()}
            intervention = {'sector': np.random.randint(0, 10)}
            counterfactual = self.causal_generator.generate_counterfactual(example, intervention)
            for k, v in counterfactual.items():
                counterfactual_batch.setdefault(k, []).append(v)
        return {k: torch.cat(v, dim=0).to(self.device) for k, v in counterfactual_batch.items()}
    
    def generate_synthetic_examples(self, batch_size, difficulty):
        z = torch.randn(batch_size, 64, device=self.device)
        difficulty = torch.full((batch_size, 1), difficulty, device=self.device)
        with torch.no_grad():
            synthetic = self.wgan_generator(z, difficulty)
        return synthetic

    def train_wgan(self, real_batch, difficulty):
        batch_size = real_batch.size(0)
        z = torch.randn(batch_size, 64, device=self.device)
        difficulty = torch.full((batch_size, 1), difficulty, device=self.device)

        # Train discriminator
        self.wgan_optimizer_d.zero_grad()
        real_score = self.wgan_discriminator(real_batch).mean()
        fake = self.wgan_generator(z, difficulty)
        fake_score = self.wgan_discriminator(fake.detach()).mean()

        # Gradient penalty
        alpha = torch.rand(batch_size, 1, device=self.device)
        interpolates = alpha * real_batch + (1 - alpha) * fake.detach()
        interpolates.requires_grad_(True)
        d_interpolates = self.wgan_discriminator(interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True
        )[0]
        grad_penalty = self.wgan_lambda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        d_loss = fake_score - real_score + grad_penalty
        d_loss.backward()
        self.wgan_optimizer_d.step()

        # Train generator
        self.wgan_optimizer_g.zero_grad()
        fake_score = self.wgan_discriminator(self.wgan_generator(z, difficulty)).mean()
        g_loss = -fake_score
        g_loss.backward()
        self.wgan_optimizer_g.step()

    def t2_generator(self, batch):
        counterfactual_batch = {}
        for i in range(len(batch['input_ids'])):
            example = {k: v[i] for k, v in batch.items()}
            intervention = {'sector': torch.tensor(np.random.randint(0, 10))}
            counterfactual = self.causal_generator.generate_counterfactual(example, intervention)
            for k, v in counterfactual.items():
                counterfactual_batch.setdefault(k, []).append(v)
        return {k: torch.stack(v).to(self.device) for k, v in counterfactual_batch.items()}
    
    def _default_difficulty_scorer(self, example, model=None) -> float:
        """
        Default function to score example difficulty.
        
        Args:
            example: Dataset example
            model: Optional model for prediction-based scoring
            
        Returns:
            Difficulty score between 0.0 and 1.0
        """
        # In a real implementation, this would use meaningful heuristics
        # For founder-VC matching, difficulty could be based on:
        # - Clarity of sector/stage (ambiguous sectors are harder)
        # - Completeness of founder information (missing fields are harder)
        # - Presence of conflicting signals (e.g., climate + crypto)
        
        # For this example, we'll use a simple length-based heuristic
        # assuming that longer descriptions are more complex
        if isinstance(example, dict) and 'input_ids' in example:
            # Count non-padding tokens
            if hasattr(example['input_ids'], 'shape'):
                # Tensor
                mask = example['input_ids'] != 0  # Assuming 0 is padding
                token_count = mask.sum().item()
            else:
                # List
                token_count = sum(1 for t in example['input_ids'] if t != 0)
            
            # Normalize token count to [0, 1]
            # Assume reasonable bounds: 10 tokens (very short) to 500 tokens (very long)
            normalized_count = (token_count - 10) / (500 - 10)
            difficulty = max(0.0, min(1.0, normalized_count))
            return difficulty
        
        # If example format doesn't match expectations, return medium difficulty
        return 0.5
    
    def _compute_initial_difficulties(self) -> np.ndarray:
        """
        Compute initial difficulty scores for all examples.
        
        Returns:
            Array of difficulty scores
        """
        difficulties = []
        
        # Simple progress tracking for large datasets
        total_examples = len(self.base_train_dataset)
        log_interval = max(1, total_examples // 10)
        
        for idx in range(total_examples):
            example = self.base_train_dataset[idx]
            difficulty = self.difficulty_scorer(example)
            difficulties.append(difficulty)
            
            if (idx + 1) % log_interval == 0:
                logger.info(f"Scored {idx + 1}/{total_examples} examples")
        
        # Convert to numpy array
        difficulties = np.array(difficulties)
        
        # Log difficulty distribution
        difficulty_bins = np.linspace(0, 1, 11)
        hist, _ = np.histogram(difficulties, bins=difficulty_bins)
        
        logger.info(f"Difficulty distribution:")
        for i in range(len(hist)):
            logger.info(f"  {difficulty_bins[i]:.1f}-{difficulty_bins[i+1]:.1f}: {hist[i]} examples")
        
        return difficulties
    
    def _compute_loss_based_difficulty(self, example_idx: int, model_loss: float) -> float:
        """
        Compute difficulty score based on model loss.
        
        Args:
            example_idx: Example index
            model_loss: Loss value for this example
            
        Returns:
            Updated difficulty score
        """
        # Assume loss range from 0.0 to 5.0 (adjust based on your task)
        normalized_loss = min(1.0, model_loss / 5.0)
        
        # Blend with existing difficulty using exponential moving average
        current_difficulty = self.curriculum_dataset.example_difficulties[example_idx]
        alpha = 0.1  # Blend factor
        
        return (1 - alpha) * current_difficulty + alpha * normalized_loss
    
    def update_example_difficulties(self, example_indices: List[int], losses: List[float]):
        """
        Update difficulty scores based on model performance.
        
        Args:
            example_indices: List of example indices
            losses: Corresponding loss values
        """
        for idx, loss in zip(example_indices, losses):
            if 0 <= idx < len(self.base_train_dataset):
                new_difficulty = self._compute_loss_based_difficulty(idx, loss)
                self.curriculum_dataset.update_example_difficulty(idx, new_difficulty)
    
    def get_dataloader(self, batch_size: int, num_workers: int = 4) -> DataLoader:
        """
        Get a dataloader for the current curriculum phase.
        
        Args:
            batch_size: Batch size
            num_workers: Number of worker processes
            
        Returns:
            PyTorch DataLoader
        """
        sampler = self.curriculum_dataset.get_sampler()
        
        return DataLoader(
            dataset=self.curriculum_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    
    # def update_curriculum(self, metrics: Dict[str, Any]):
    #     """
    #     Update curriculum state based on training metrics.
        
    #     Args:
    #         metrics: Dictionary of training metrics
    #     """
    #     self.global_step += 1
        
    #     # Update curriculum dataset state
    #     self.curriculum_dataset.update_curriculum_state(metrics)
        
    #     # Record metrics
    #     self.metrics_history.append({
    #         'step': self.global_step,
    #         'phase': self.curriculum_dataset.current_phase,
    #         'mix_ratio': self.curriculum_dataset.current_mix_ratio.copy(),
    #         **metrics
    #     })
    def update_curriculum(self, metrics):
        self.global_step += 1
        metrics_tensor = torch.tensor([
            metrics.get('s_loss', 0.0),
            metrics.get('t1_loss', 0.0),
            metrics.get('t2_loss', 0.0),
            metrics.get('t3_loss', 0.0)
        ], device=self.device)

        # Solve ODE for one step
        t = torch.tensor([self.global_step, self.global_step + 1], device=self.device)
        new_state = torchdiffeq.odeint(
            func=lambda t, s: self.ode(t, s, metrics_tensor),
            y0=self.state,
            t=t,
            method='rk4'
        )[-1]
        # Compute log-likelihood (for monitoring)
        _, divergence = self.cnf(t[-1], new_state, metrics_tensor)
        log_p = Normal(0, 1).log_prob(self.state).sum() - divergence

        # Update state
        self.state = new_state
        self.curriculum_dataset.current_mix_ratio = new_state[:self.config.num_curriculum_phases].tolist()
        self.curriculum_dataset.example_difficulties = new_state[self.config.num_curriculum_phases:].cpu().numpy()

        # Reassign examples to phases
        self.curriculum_dataset.phase_examples = self.curriculum_dataset._assign_examples_to_phases()

        # Logging
        if self.global_step % self.config.log_frequency == 0:
            logger.info(f"Step {self.global_step}: Mix ratio {self.curriculum_dataset.current_mix_ratio}")
        if self.global_step % 100 == 0:
            hardest_phase = self.config.num_curriculum_phases - 1
            hard_indices = self.curriculum_dataset.phase_examples[hardest_phase]
            if hard_indices:
                batch = torch.stack([torch.tensor(self.base_train_dataset[idx]['input_ids']) for idx in hard_indices[:32]])
                self.train_wgan(batch.to(self.device), difficulty=1.0)
        
        # WGAN training for T2/T3
        if self.global_step % 100 == 0:
            hardest_phase = self.config.num_curriculum_phases - 1
            hard_indices = self.curriculum_dataset.phase_examples[hardest_phase]
            if hard_indices:
                batch = torch.stack([torch.tensor(self.base_train_dataset[idx]['input_ids']) for idx in hard_indices[:32]])
                self.train_wgan(batch.to(self.device), difficulty=1.0)

        # Generate synthetic examples for T2/T3
        if metrics.get('t2_loss', 0.0) > 0.5 or metrics.get('t3_loss', 0.0) > 0.5:
            synthetic_batch = self.generate_synthetic_examples(batch_size=32, difficulty=1.0)
            # Add to curriculum_dataset for T2/T3 generators    
    
    def get_curriculum_state(self) -> Dict[str, Any]:
        """
        Get current state of the curriculum.
        
        Returns:
            Dictionary of curriculum state information
        """
        return {
            'current_phase': self.curriculum_dataset.current_phase,
            'phase_step': self.curriculum_dataset.phase_step,
            'global_step': self.global_step,
            'mix_ratio': self.curriculum_dataset.current_mix_ratio,
            'available_examples': sum(len(self.curriculum_dataset.phase_examples[p]) 
                                     for p in range(self.curriculum_dataset.current_phase + 1)),
            'total_examples': len(self.base_train_dataset)
        }
    
    def reset_curriculum(self):
        """Reset curriculum to initial state."""
        self.curriculum_dataset.current_phase = 0
        self.curriculum_dataset.phase_step = 0
        self.curriculum_dataset.current_mix_ratio = self.config.mix_ratio.copy()
        self.curriculum_dataset.plateau_counter = 0
        self.curriculum_dataset.best_loss = float('inf')


# Example integration with STTT Cycle
class IntegratedSTTTCurriculumTrainer:
    """
    Integrates STTT Cycle with Dynamic Curriculum Construction.
    """
    
    def __init__(
        self,
        model,
        optimizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
        sttt_config=None,
        curriculum_config=None,
        device=None
    ):
        """
        Initialize the integrated trainer.
        
        Args:
            model: The model to train
            optimizer: The optimizer to use for training
            train_dataset: Training dataset
            val_dataset: Validation dataset
            sttt_config: Configuration for STTT cycle
            curriculum_config: Configuration for curriculum learning
            device: Device to use ('cuda' or 'cpu')
        """
        from sttt_cycle import STTTConfig, STTTCycle
        
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize curriculum
        self.curriculum_config = curriculum_config or DynamicCurriculumConfig()
        self.curriculum = DynamicCurriculumConstructor(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=self.curriculum_config,
            device=self.device
        )
        
        # Get curriculum dataloader
        self.train_dataloader = self.curriculum.get_dataloader(
            batch_size=self.curriculum_config.phase_duration // 10  # Reasonable batch size
        )
        
        # Initialize baseline validation dataloader
        self.val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=self.curriculum_config.phase_duration // 10,
            shuffle=False
        )
        
        # Initialize STTT cycle
        self.sttt_config = sttt_config or STTTConfig()
        
        # Custom generators for T2 and T3 phases
        def t2_generator(batch):
            # Generate counterfactual examples
            # For this integration, we'll use curriculum's higher difficulty examples
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
            # Generate adversarial examples
            # For now, use the highest difficulty examples from curriculum
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
        self.metrics_history = []
        self.global_step = 0
    
    def train_step(self) -> Dict[str, Any]:
        """
        Execute a single training step with the integrated components.
        
        Returns:
            Dictionary of metrics for this step
        """
        # Execute STTT step
        sttt_metrics = self.sttt_cycle.step()
        
        # Update curriculum based on STTT metrics
        self.curriculum.update_curriculum(sttt_metrics)
        
        # Update curriculum dataloader if phase has changed (less frequent update)
        if self.global_step % 100 == 0:
            self.train_dataloader = self.curriculum.get_dataloader(
                batch_size=self.curriculum_config.phase_duration // 10
            )
            self.sttt_cycle.study_dataloader = self.train_dataloader
            self.sttt_cycle.study_iter = iter(self.train_dataloader)
        
        # Combine metrics
        combined_metrics = {
            'global_step': self.global_step,
            **sttt_metrics,
            'curriculum_phase': self.curriculum.curriculum_dataset.current_phase,
            'curriculum_mix_ratio': self.curriculum.curriculum_dataset.current_mix_ratio.copy()
        }
        
        # Record metrics
        self.metrics_history.append(combined_metrics)
        
        # Update step counter
        self.global_step += 1
        
        return combined_metrics
    
    def train(self, num_steps: int) -> Dict[str, Any]:
        """
        Train the model for a specified number of steps.
        
        Args:
            num_steps: Number of steps to train for
            
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Starting integrated STTT-Curriculum training for {num_steps} steps")
        
        all_metrics = []
        for _ in range(num_steps):
            step_metrics = self.train_step()
            all_metrics.append(step_metrics)
            
            # Logging
            if self.global_step % 10 == 0:
                logger.info(f"Step {self.global_step}: "
                           f"STTT Phase={step_metrics['phase']}, "
                           f"Curriculum Phase={step_metrics['curriculum_phase']}, "
                           f"Loss={step_metrics.get(f'{step_metrics['phase'].lower()}_loss', 'N/A')}")
        
        # Compile final metrics
        final_metrics = {
            'total_steps': num_steps,
            'sttt_metrics': self.sttt_cycle.get_metrics(),
            'curriculum_state': self.curriculum.get_curriculum_state(),
            'metrics_history': self.metrics_history
        }
        
        logger.info(f"Integrated training completed")
        return final_metrics
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current training metrics."""
        return {
            'global_step': self.global_step,
            'sttt_metrics': self.sttt_cycle.get_metrics(),
            'curriculum_state': self.curriculum.get_curriculum_state()
        }


# Example usage:
"""
# Initialize model and optimizer
model = ...
optimizer = ...

# Initialize datasets
train_dataset = ...
val_dataset = ...

# Initialize integrated trainer
from sttt_cycle import STTTConfig

sttt_config = STTTConfig(...)
curriculum_config = DynamicCurriculumConfig(...)

trainer = IntegratedSTTTCurriculumTrainer(
    model=model,
    optimizer=optimizer,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    sttt_config=sttt_config,
    curriculum_config=curriculum_config
)

# Train for 1000 steps
metrics = trainer.train(1000)

# Get current metrics
current_metrics = trainer.get_metrics()
"""
