import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
import time

# Configure logger
logger = logging.getLogger(__name__)

# Quantum computing imports
try:
    import pennylane as qml
    from pennylane import numpy as np
    from pennylane.templates import RandomLayers
    QUANTUM_AVAILABLE = True
except (ImportError, AttributeError) as e:
    QUANTUM_AVAILABLE = False
    logger.warning(f"PennyLane not available or JAX version incompatible: {e}. Quantum annealing will be disabled.")

# Topological data analysis imports
try:
    from ripser import Rips
    from persim import PersistenceImager
    TOPOLOGY_AVAILABLE = True
except ImportError:
    TOPOLOGY_AVAILABLE = False
    logger.warning("Ripser and Persim not available. Topological regularization will be disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LatentSpaceReg")

@dataclass
class LatentSpaceRegConfig:
    """Configuration for Latent Space Regularization."""
    # Core regularization parameters
    l1_penalty_weight: float = 1e-5   
    kl_penalty_weight: float = 1e-5   
    orthogonal_penalty_weight: float = 1e-5
    group_sparsity_weight: float = 1e-5  # NEW: Group sparsity penalty
    hessian_penalty_weight: float = 1e-6  # NEW: Curvature regularization
    spectral_penalty_weight: float = 1e-6  # NEW: Spectral norm regularization weight
    
    # Cross-layer parameters
    cross_layer_enabled: bool = False
    cross_layer_weight: float = 0.05
    
    # Adversarial training parameters
    adversarial_enabled: bool = False
    adversarial_weight: float = 0.05
    adversarial_epsilon: float = 0.1
    
    # Quantum annealing parameters
    use_quantum_annealing: bool = False
    quantum_annealing_weight: float = 0.05
    quantum_circuit_depth: int = 10
    quantum_num_qubits: int = 10
    quantum_annealing_frequency: int = 200
    quantum_learning_rate: float = 1e-3
    quantum_plasticity_enabled: bool = False
    quantum_plasticity_weight: float = 0.05
    
    # Topological regularization parameters
    use_topological_reg: bool = False
    topological_weight: float = 0.05
    target_betti_1: int = 2
    topological_frequency: int = 100
    max_filtration_value: float = 1.0
    persistence_threshold: float = 0.1
    
    # Holographic regularization parameters
    use_holographic_reg: bool = False
    holographic_weight: float = 0.05
    holographic_frequency: int = 50
    phase_learning_rate: float = 1e-3
    reconstruction_weight: float = 1.0
    sparsity_weight: float = 0.1
    
    # Synaptic plasticity parameters
    use_plasticity_reg: bool = False
    plasticity_weight: float = 0.05
    plasticity_learning_rate: float = 1e-4
    plasticity_gamma: float = 0.1
    plasticity_frequency: int = 50
    plasticity_variance_weight: float = 0.1
    
    # NEW: Memory replay parameters
    memory_replay_enabled: bool = False
    memory_replay_weight: float = 0.05
    memory_buffer_size: int = 100
    
    # NEW: Neuromodulation parameters
    neuromodulation_enabled: bool = False
    neuromodulation_weight: float = 0.05
    neuromodulation_beta: float = 0.1
    
    # NEW: Phase transition parameters
    phase_transition_enabled: bool = False
    phase_transition_weight: float = 0.05
    initial_temperature: float = 1.0
    temperature_decay: float = 1e-3
    
    # Chaos regularization parameters
    chaos_enabled: bool = False
    chaos_weight: float = 0.05
    chaos_r: float = 3.8
    
    # Sparsity configuration
    sparsity_target: float = 0.1      
    feature_target_sparsity: float = 0.05  # NEW: Sparsity target for feature level
    channel_target_sparsity: float = 0.2   # NEW: Sparsity target for channel level
    
    # Layer selection
    target_layers: List[str] = field(default_factory=lambda: ["attention", "mlp", "ffn"])
    layer_weights: Dict[str, float] = field(default_factory=dict)
    
    # NEW: Layer-wise adaptation
    layer_adaptive_penalties: bool = True  # Adapt penalties per layer based on sensitivity
    layer_sensitivity_window: int = 20     # Window for computing layer sensitivity
    
    # Dynamic adjustment
    dynamic_penalty_adjustment: bool = True
    min_penalty_weight: float = 1e-8  # Lower minimum for finer control
    max_penalty_weight: float = 1e-2  # Higher maximum for stronger regularization
    
    # NEW: Regularization scheduling
    penalty_schedule: str = "cyclical"  # Options: linear, cosine, cyclical, or adaptive
    cycle_length: int = 1000            # Steps per cycle for cyclical scheduling
    cycle_min_factor: float = 0.1       # Minimum multiplier in cycle
    cycle_max_factor: float = 1.0       # Maximum multiplier in cycle
    
    # Standard scheduling
    warmup_steps: int = 1000        
    cooldown_steps: int = 5000     
    schedule_type: str = "cosine"  # Changed default to cosine
    
    # Monitoring
    activation_history_size: int = 200  # Increased for better statistics
    monitor_frequency: int = 50      
    
    # NEW: Spectral monitoring and regularization
    track_spectral_norm: bool = True    # Track spectral norm of activations
    spectral_norm_frequency: int = 100  # How often to compute spectral norms
    spectral_norm_target: float = 1.0   # Target spectral norm value for regularization
    
    # Integration control
    integration_start_step: int = 1000
    
    # Logging
    log_frequency: int = 100
    verbose: bool = True
    
    # NEW: Adaptive MI-based regularization
    use_mutual_information: bool = True   # Use MI to guide regularization
    mi_update_frequency: int = 200       # Update MI estimates periodically
    mi_estimation_samples: int = 1000    # Samples for MI estimation
    
    # NEW: Layer-wise curriculum parameters
    layer_specific_cycles: bool = True   # Use layer-specific cycle lengths
    layer_cycle_alpha: float = 0.2       # Weight for sensitivity in cycle length adjustment
    
    # NEW: Gradient-based penalty adjustment
    use_gradient_penalty_adjustment: bool = True  # Use gradient-based penalty updates
    penalty_learning_rate: float = 1e-3           # Learning rate for penalty updates 
    penalty_gradient_clip: float = 1e-5           # Clipping value for penalty gradients
    
    # NEW: Adaptive sparsity targets
    use_adaptive_sparsity: bool = True    # Dynamically adjust sparsity targets
    sparsity_sensitivity_weight: float = 0.1  # Weight for sensitivity in sparsity target
    sparsity_mi_weight: float = 0.05      # Weight for MI in sparsity target adjustment
    
    # NEW: Memory-efficient MI estimation  
    mi_mini_batch_size: int = 256         # Mini-batch size for MI estimation
    mi_mini_batch_count: int = 4          # Number of mini-batches for MI estimation
    
    # NEW: Pruning with uncertainty awareness
    prune_uncertainty_beta: float = 0.1    # Weight for variance in importance scores
    gradual_pruning_steps: int = 100       # Steps for gradual pruning implementation
    
    # MI-guided plasticity parameters
    mi_plasticity_enabled: bool = False
    mi_plasticity_weight: float = 0.05
    mi_threshold: float = 0.1
    mi_update_frequency: int = 100
    
    # NEW: Bayesian Synaptic Update parameters
    bayesian_enabled: bool = False
    bayesian_weight: float = 0.05
    bayesian_prior_scale: float = 1.0
    bayesian_kl_weight: float = 0.01
    bayesian_rank: int = 32  # NEW: Rank for low-rank parameterization
    bayesian_adversarial_enabled: bool = False
    bayesian_adversarial_weight: float = 0.05
    
    # NEW: Graph-Based Plasticity parameters
    graph_plasticity_enabled: bool = False
    graph_plasticity_weight: float = 0.05
    graph_hidden_dim: int = 64
    graph_n_layers: int = 2
    graph_edge_sparsity: float = 0.2
    graph_spectral_enabled: bool = False  # NEW: Enable spectral regularization
    graph_spectral_weight: float = 0.05   # NEW: Weight for spectral loss
    graph_spectral_target: float = 1.0    # NEW: Target spectral norm value
    graph_adv_enabled: bool = False  # NEW: Enable adversarial training
    graph_adv_weight: float = 0.05   # NEW: Weight for adversarial loss
    
    # NEW: Temporal Plasticity parameters
    temporal_plasticity_enabled: bool = False
    temporal_plasticity_weight: float = 0.05
    temporal_hidden_dim: int = 64
    temporal_sequence_length: int = 20
    temporal_num_heads: int = 4  # NEW: Number of attention heads for transformer
    temporal_max_sequence_length: int = 50  # NEW: Maximum sequence length
    temporal_min_sequence_length: int = 5   # NEW: Minimum sequence length
    temporal_time_gap_weight: float = 0.1   # NEW: Weight for time gap influence
    

class QuantumAnnealer:
    """Quantum annealing-inspired regularization for latent space optimization."""
    
    def __init__(
        self,
        num_qubits: int,
        circuit_depth: int,
        learning_rate: float,
        device: str = "default.qubit"
    ):
        if not QUANTUM_AVAILABLE:
            logger.warning("PennyLane not available. Using dummy quantum annealer.")
            self.num_qubits = num_qubits
            self.circuit_depth = circuit_depth
            self.learning_rate = learning_rate
            self.dummy_mode = True
            return
            
        self.dummy_mode = False
        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth
        self.learning_rate = learning_rate
        
        # Initialize quantum device
        self.dev = qml.device(device, wires=num_qubits)
        
        # Initialize parameters for the quantum circuit
        self.params = np.random.randn(circuit_depth, num_qubits, 3)
        
        # Define the quantum circuit
        @qml.qnode(self.dev)
        def quantum_circuit(params, activations):
            # Encode activations into quantum state
            for i in range(num_qubits):
                qml.RY(activations[i], wires=i)
            
            # Apply variational layers
            for layer in range(circuit_depth):
                for i in range(num_qubits):
                    qml.RX(params[layer, i, 0], wires=i)
                    qml.RY(params[layer, i, 1], wires=i)
                    qml.RZ(params[layer, i, 2], wires=i)
                
                # Entangling gates
                for i in range(0, num_qubits-1, 2):
                    qml.CNOT(wires=[i, i+1])
            
            # Measure energy
            return qml.expval(qml.PauliZ(0))
        
        self.circuit = quantum_circuit
    
    def compute_energy(self, activations: torch.Tensor) -> torch.Tensor:
        """Compute the quantum energy of the system."""
        if self.dummy_mode:
            # Return a dummy value that mimics energy
            return torch.mean(activations**2) * 0.01
            
        # Normalize activations to [-π, π]
        normalized_acts = torch.clamp(activations, -1, 1) * math.pi
        
        # Convert to numpy for quantum computation
        acts_np = normalized_acts.detach().cpu().numpy()
        
        # Compute energy
        energy = self.circuit(self.params, acts_np)
        
        return torch.tensor(energy, device=activations.device)
    
    def compute_weight_energy(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute quantum energy for synaptic weights."""
        if self.dummy_mode:
            # Return a dummy value that mimics energy
            return torch.mean(weights**2) * 0.01
            
        normalized_weights = torch.clamp(weights, -1, 1) * math.pi
        weights_np = normalized_weights.detach().cpu().numpy()
        
        @qml.qnode(self.dev)
        def weight_circuit(params, weights):
            for i in range(self.num_qubits):
                qml.RY(weights[i], wires=i)
            for layer in range(self.circuit_depth):
                for i in range(self.num_qubits):
                    qml.RX(params[layer, i, 0], wires=i)
                    qml.RY(params[layer, i, 1], wires=i)
                    qml.RZ(params[layer, i, 2], wires=i)
                for i in range(0, self.num_qubits-1, 2):
                    qml.CNOT(wires=[i, i+1])
            return qml.expval(qml.PauliZ(0))
        
        energy = weight_circuit(self.params, weights_np)
        return torch.tensor(energy, device=weights.device)
    
    def update_params(self, gradient: torch.Tensor):
        """Update quantum circuit parameters."""
        if self.dummy_mode:
            return
            
        self.params -= self.learning_rate * gradient.cpu().numpy()


class TopologicalRegularizer:
    """Topological regularization for latent space optimization."""
    
    def __init__(
        self,
        target_betti_1: int,
        max_filtration_value: float,
        persistence_threshold: float
    ):
        if not TOPOLOGY_AVAILABLE:
            raise ImportError("Ripser and Persim are required for topological regularization")
            
        self.target_betti_1 = target_betti_1
        self.max_filtration_value = max_filtration_value
        self.persistence_threshold = persistence_threshold
        
        # Initialize Rips complex calculator
        self.rips = Rips(maxdim=1, thresh=max_filtration_value)
        
        # Initialize persistence imager for visualization
        self.pimgr = PersistenceImager(pixel_size=0.1)
    
    def compute_betti_numbers(self, activations: torch.Tensor) -> Dict[int, int]:
        """Compute Betti numbers from activations."""
        # Convert to numpy and normalize
        acts_np = activations.detach().cpu().numpy()
        acts_np = (acts_np - acts_np.min()) / (acts_np.max() - acts_np.min() + 1e-8)
        
        # Compute persistence diagrams
        diagrams = self.rips.fit_transform(acts_np)
        
        # Count persistent features
        betti_numbers = {}
        
        # Count 0-dimensional features (connected components)
        dgm0 = diagrams[0]
        betti_numbers[0] = len([p for p in dgm0 if p[1] - p[0] > self.persistence_threshold])
        
        # Count 1-dimensional features (holes)
        dgm1 = diagrams[1]
        betti_numbers[1] = len([p for p in dgm1 if p[1] - p[0] > self.persistence_threshold])
        
        return betti_numbers
    
    def compute_topological_loss(self, activations: torch.Tensor) -> torch.Tensor:
        """Compute topological regularization loss."""
        betti_numbers = self.compute_betti_numbers(activations)
        
        # Compute loss as squared difference from target Betti-1 number
        loss = (betti_numbers[1] - self.target_betti_1) ** 2
        
        return torch.tensor(loss, device=activations.device)


class HolographicEncoder:
    """Holographic encoding for latent space regularization."""
    
    def __init__(
        self,
        feature_dim: int,
        learning_rate: float,
        reconstruction_weight: float,
        sparsity_weight: float
    ):
        self.feature_dim = feature_dim
        self.learning_rate = learning_rate
        self.reconstruction_weight = reconstruction_weight
        self.sparsity_weight = sparsity_weight
        
        # Initialize learnable phases
        self.phases = nn.Parameter(torch.randn(feature_dim) * 0.01)
        
        # Initialize basis vectors
        self.basis = torch.eye(feature_dim)
    
    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        """Encode activations into holographic representation."""
        # Apply learned phases
        phase_rotated = activations * torch.exp(1j * self.phases)
        
        # Project onto basis
        holographic = torch.matmul(phase_rotated, self.basis)
        
        return holographic
    
    def decode(self, holographic: torch.Tensor) -> torch.Tensor:
        """Decode holographic representation back to activations."""
        # Inverse projection
        reconstructed = torch.matmul(holographic, self.basis.t())
        
        # Remove phase rotation
        activations = torch.real(reconstructed * torch.exp(-1j * self.phases))
        
        return activations
    
    def compute_loss(self, activations: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute holographic encoding loss."""
        # Encode activations
        holographic = self.encode(activations)
        
        # Decode back
        reconstructed = self.decode(holographic)
        
        # Compute reconstruction loss
        recon_loss = F.mse_loss(reconstructed, activations)
        
        # Compute sparsity loss
        sparsity_loss = torch.norm(holographic, p=1)
        
        # Combine losses
        total_loss = (
            self.reconstruction_weight * recon_loss +
            self.sparsity_weight * sparsity_loss
        )
        
        return total_loss, {
            'reconstruction_loss': recon_loss.item(),
            'sparsity_loss': sparsity_loss.item()
        }
    
    def update_phases(self, gradient: torch.Tensor):
        """Update learned phases."""
        self.phases.data -= self.learning_rate * gradient


class SynapticPlasticityRegularizer:
    """Implements synaptic plasticity-inspired regularization for latent space optimization."""
    
    def __init__(self, feature_dim: int, learning_rate: float, gamma: float, variance_weight: float, config: LatentSpaceRegConfig, layer_name: str = None):
        self.feature_dim = feature_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.variance_weight = variance_weight
        self.config = config
        self.layer_name = layer_name
        
        # Initialize synaptic weights and statistics
        self.weights = torch.randn(feature_dim, feature_dim, requires_grad=True)
        self.activation_mean = torch.zeros(feature_dim)
        self.activation_variance = torch.ones(feature_dim)
        self.update_count = 0
        
        # Initialize MI estimator if enabled
        if self.config.mi_plasticity_enabled:
            self.mi_estimator = LatentMutualInformationEstimator(device=self.weights.device)
        
        # Initialize cross-layer weights
        self.cross_layer_weights = {}  # Dict[layer_pair, weights]
        
        # Initialize memory replay state
        self.memory_buffer = deque(maxlen=self.config.memory_buffer_size)
        
        # Initialize neuromodulation state
        self.modulation_history = deque(maxlen=100)
        self.last_val_loss = None
        
        # Initialize phase transition state
        self.temperature = self.config.initial_temperature
        self.alpha = self.config.temperature_decay
        
        # Initialize quantum annealer if enabled
        if self.config.quantum_plasticity_enabled:
            if not QUANTUM_AVAILABLE:
                logger.warning("PennyLane not available. Quantum plasticity will be disabled.")
                self.quantum_annealer = None
            else:
                self.quantum_annealer = QuantumAnnealer(
                    num_qubits=min(feature_dim, self.config.quantum_num_qubits),
                    circuit_depth=self.config.quantum_circuit_depth,
                    learning_rate=self.config.quantum_learning_rate
                )
        else:
            self.quantum_annealer = None
    
    def store_activations(self, activations: torch.Tensor):
        """Store activations in memory buffer for replay."""
        if activations.dim() > 2:
            activations = activations.view(-1, activations.size(-1))
        self.memory_buffer.append(activations.detach())
    
    def update_modulation(self, val_loss: float) -> float:
        """Update neuromodulation based on validation loss changes."""
        if self.last_val_loss is None:
            modulation = 0.5
        else:
            delta_loss = val_loss - self.last_val_loss
            modulation = torch.sigmoid(torch.tensor(-self.config.neuromodulation_beta * delta_loss)).item()
        self.modulation_history.append(modulation)
        self.last_val_loss = val_loss
        return modulation
    
    def update_temperature(self):
        """Update temperature for phase transition."""
        self.temperature *= math.exp(-self.alpha)
    
    def compute_plasticity_loss(self, activations: torch.Tensor, val_loss: float = None, model_loss_fn: Callable = None, targets: torch.Tensor = None, cross_layer_acts: Dict[str, torch.Tensor] = None) -> torch.Tensor:
        """Compute plasticity loss with memory replay, neuromodulation, phase transition, adversarial training, cross-layer updates, and MI-guided updates."""
        # Store current activations
        self.store_activations(activations)
        
        # Normalize current activations
        normalized_acts = (activations - self.activation_mean) / (self.activation_variance.sqrt() + 1e-6)
        hebbian_updates = torch.matmul(normalized_acts.t(), normalized_acts) / activations.size(0)
        
        # MI-guided updates
        mi_loss = torch.tensor(0.0, device=self.weights.device)
        if self.config.mi_plasticity_enabled and self.update_count % self.config.mi_update_frequency == 0:
            mi_matrix = self.mi_estimator.estimate_mi(activations, self.config.mi_mini_batch_size)
            mi_weight = torch.exp(-0.5 * mi_matrix)
            hebbian_updates *= mi_weight
            mi_loss = torch.sum(torch.relu(mi_matrix - self.config.mi_threshold))
        
        # Adversarial updates
        adv_loss = torch.tensor(0.0, device=self.weights.device)
        if self.config.adversarial_enabled and model_loss_fn is not None and targets is not None:
            activations.requires_grad_(True)
            model_loss = model_loss_fn(activations, targets)
            grad = torch.autograd.grad(model_loss, activations)[0]
            adv_acts = activations + self.config.adversarial_epsilon * torch.sign(grad)
            norm_adv_acts = (adv_acts - self.activation_mean) / (self.activation_variance.sqrt() + 1e-6)
            adv_hebbian = torch.matmul(norm_adv_acts.t(), norm_adv_acts) / adv_acts.size(0)
            hebbian_updates = 0.7 * hebbian_updates + 0.3 * adv_hebbian
            
            # Adversarial loss
            adv_diff = torch.norm(adv_acts - activations, p=2)**2
            kl_adv = F.kl_div(
                F.log_softmax(self.weights, dim=-1),
                F.softmax(self.weights + adv_hebbian, dim=-1),
                reduction='batchmean'
            )
            adv_loss = adv_diff + 0.1 * kl_adv
        
        # Cross-layer updates
        cross_loss = torch.tensor(0.0, device=self.weights.device)
        if self.config.cross_layer_enabled and cross_layer_acts is not None and self.layer_name is not None:
            for other_layer_name, other_acts in cross_layer_acts.items():
                if other_layer_name == self.layer_name:
                    continue  # Skip self
                    
                norm_other_acts = (other_acts - self.activation_mean) / (self.activation_variance.sqrt() + 1e-6)
                cross_hebbian = torch.matmul(normalized_acts.t(), norm_other_acts) / activations.size(0)
                
                # Create or get cross-layer weights
                pair_key = f"{self.layer_name}_{other_layer_name}"
                if pair_key not in self.cross_layer_weights:
                    self.cross_layer_weights[pair_key] = torch.randn(
                        self.feature_dim, 
                        other_acts.size(-1),
                        requires_grad=True,
                        device=self.weights.device
                    )
                
                # Compute cross-layer updates
                cross_updates = self.learning_rate * cross_hebbian * (1 - self.gamma * self.cross_layer_weights[pair_key].abs())
                cross_loss += (
                    torch.norm(cross_updates - self.cross_layer_weights[pair_key], p=2)**2 + 
                    0.1 * torch.var(self.cross_layer_weights[pair_key])
                )
        
        # Apply memory replay if enabled
        replay_updates = torch.zeros_like(self.weights)
        replay_loss = torch.tensor(0.0, device=self.weights.device)
        
        if self.config.memory_replay_enabled and len(self.memory_buffer) > 0:
            # Compute replay updates from memory buffer
            for mem_acts in self.memory_buffer:
                norm_mem = (mem_acts - self.activation_mean) / (self.activation_variance.sqrt() + 1e-6)
                replay_updates += torch.matmul(norm_mem.t(), norm_mem) / mem_acts.size(0)
            replay_updates *= self.learning_rate * (1 - self.gamma * self.weights.abs())
            
            # Compute replay loss
            if len(self.memory_buffer) > 1:
                replay_loss = torch.norm(self.memory_buffer[-1] - self.memory_buffer[-2])**2
            replay_loss += 0.1 * torch.var(self.weights)
        
        # Apply neuromodulation if enabled
        if self.config.neuromodulation_enabled and val_loss is not None:
            modulation = self.update_modulation(val_loss)
            modulated_updates = modulation * self.learning_rate * hebbian_updates * (1 - self.gamma * self.weights.abs())
            neuro_loss = torch.var(torch.tensor(list(self.modulation_history))) + 0.1 * abs(modulation - 0.5)
        else:
            modulated_updates = self.learning_rate * hebbian_updates * (1 - self.gamma * self.weights.abs())
            neuro_loss = torch.tensor(0.0)
        
        # Combine updates
        weight_updates = modulated_updates + 0.5 * replay_updates
        
        # Apply phase transition if enabled
        if self.config.phase_transition_enabled:
            proposed_weights = self.weights + weight_updates
            delta_energy = torch.norm(proposed_weights - self.weights, p=2)**2
            acceptance = torch.exp(-delta_energy / self.temperature)
            mask = (torch.rand_like(self.weights) < acceptance).float()
            weight_updates = mask * proposed_weights + (1 - mask) * self.weights
            phase_loss = torch.mean(torch.abs(weight_updates - self.weights)) + 0.1 * torch.var(weight_updates)
            self.update_temperature()
        else:
            phase_loss = torch.tensor(0.0)
        
        # Compute final loss components
        weight_decay = torch.norm(self.weights, p=2)
        variance_reg = torch.mean(self.activation_variance)
        
        # Combine losses
        plasticity_loss = (
            torch.norm(weight_updates - self.weights, p=2) +
            self.variance_weight * variance_reg +
            0.01 * weight_decay +
            self.config.neuromodulation_weight * neuro_loss +
            self.config.phase_transition_weight * phase_loss +
            self.config.memory_replay_weight * replay_loss +
            self.config.adversarial_weight * adv_loss +
            self.config.cross_layer_weight * cross_loss +
            self.config.mi_plasticity_weight * mi_loss
        )
        
        # Update counter
        self.update_count += 1
        
        return plasticity_loss
    
    def update_statistics(self, activations: torch.Tensor) -> None:
        """Update activation statistics using exponential moving average."""
        batch_mean = activations.mean(dim=0)
        batch_variance = activations.var(dim=0)
        
        # Update statistics
        self.activation_mean = (1 - self.gamma) * self.activation_mean + self.gamma * batch_mean
        self.activation_variance = (1 - self.gamma) * self.activation_variance + self.gamma * batch_variance
        self.update_count += 1
        
    def update_weights(self, gradients: torch.Tensor) -> None:
        """Update synaptic weights based on gradients."""
        with torch.no_grad():
            self.weights -= self.learning_rate * gradients


class BayesianSynapticRegularizer:
    """Bayesian synaptic regularizer using low-rank parameterization."""
    
    def __init__(
        self,
        feature_dim: int,
        rank: int = 32,  # NEW: Rank for low-rank parameterization
        learning_rate: float = 1e-3,
        prior_scale: float = 1.0,
        kl_weight: float = 0.01,
        device = None
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.rank = rank  # NEW: Store rank
        self.learning_rate = learning_rate
        self.prior_scale = prior_scale
        self.kl_weight = kl_weight
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # NEW: Low-rank parameterization
        self.U = nn.Parameter(torch.randn(feature_dim, rank, device=self.device) * 0.01)
        self.V = nn.Parameter(torch.randn(feature_dim, rank, device=self.device) * 0.01)
        self.P = nn.Parameter(torch.ones(feature_dim, rank, device=self.device) * -5)
        self.Q = nn.Parameter(torch.ones(feature_dim, rank, device=self.device) * -5)
        
        # Store prior parameters
        self.prior_logvar = torch.log(torch.tensor(prior_scale**2, device=self.device))
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Track statistics
        self.uncertainty_history = []
        self.kl_history = []
        self.update_count = 0
    
    def sample_weights(self, n_samples: int = 1) -> torch.Tensor:
        """Sample weights from the low-rank parameterized distribution."""
        # Compute mean and log variance using low-rank matrices
        weight_mean = torch.matmul(self.U, self.V.t())
        logvar = torch.matmul(self.P, self.Q.t())
        
        # Sample from distribution
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(n_samples, self.feature_dim, self.feature_dim, device=self.device)
        return weight_mean + std * eps
    
    def compute_kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence between posterior and prior using low-rank parameterization."""
        # Compute mean and log variance
        weight_mean = torch.matmul(self.U, self.V.t())
        logvar = torch.matmul(self.P, self.Q.t())
        
        # Compute KL divergence
        kl_div = 0.5 * (
            torch.exp(logvar) / self.prior_scale**2 +
            weight_mean**2 / self.prior_scale**2 - 1 +
            self.prior_logvar - logvar
        )
        return kl_div.sum()
    
    def compute_uncertainty(self) -> torch.Tensor:
        """Compute uncertainty using low-rank parameterization."""
        logvar = torch.matmul(self.P, self.Q.t())
        return torch.exp(logvar).mean()
    
    def update_from_activations(
        self, 
        activations: torch.Tensor,
        model_loss_fn: Callable = None,
        targets: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Update parameters based on activations using low-rank parameterization."""
        # Sample weights
        weights = self.sample_weights()
        
        # Compute loss
        if model_loss_fn is not None and targets is not None:
            # Apply weights to activations
            weighted_acts = torch.matmul(weights, activations)
            # Compute model loss
            model_loss = model_loss_fn(weighted_acts, targets)
        else:
            model_loss = torch.tensor(0.0, device=self.device)
        
        # Compute KL divergence
        kl_div = self.compute_kl_divergence()
        
        # Total loss
        total_loss = model_loss + self.kl_weight * kl_div
        
        # Update parameters
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Track statistics
        self.uncertainty_history.append(self.compute_uncertainty().item())
        self.kl_history.append(kl_div.item())
        self.update_count += 1
        
        return total_loss, {
            'model_loss': model_loss.item(),
            'kl_divergence': kl_div.item(),
            'uncertainty': self.uncertainty_history[-1]
        }
    
    def update_parameters(self, gradients: Tuple[torch.Tensor, torch.Tensor]):
        """Update parameters using gradients."""
        # Unpack gradients
        grad_U, grad_V = gradients
        
        # Update low-rank matrices
        self.U.data -= self.learning_rate * grad_U
        self.V.data -= self.learning_rate * grad_V
        
        # Update count
        self.update_count += 1


class GraphBasedPlasticityRegularizer:
    """
    Represents the latent space as a graph, with nodes as features and 
    edges as synaptic weights, using graph neural networks to propagate 
    updates, enhancing coherence in representation learning.
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        learning_rate: float = 1e-3,
        edge_sparsity: float = 0.2,
        spectral_enabled: bool = False,  # NEW: Spectral regularization flag
        spectral_weight: float = 0.05,   # NEW: Spectral loss weight
        spectral_target: float = 1.0,    # NEW: Target spectral norm
        adv_enabled: bool = False,  # NEW: Adversarial training flag
        adv_weight: float = 0.05,   # NEW: Adversarial loss weight
        device = None
    ):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.edge_sparsity = edge_sparsity
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Node feature embedding layer
        self.node_embedding = nn.Linear(1, hidden_dim)
        
        # Graph layers
        self.graph_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        
        # Graph readout (aggregation) layer
        self.readout = nn.Linear(hidden_dim, 1)
        
        # Initialize adjacency matrix (graph structure)
        self.adjacency = nn.Parameter(torch.randn(feature_dim, feature_dim, device=self.device))
        self.adjacency_mask = None
        self.update_adjacency_mask()
        
        # Tracking
        self.update_count = 0
        self.connectivity_history = []
        
        # NEW: Spectral regularization parameters
        self.spectral_enabled = spectral_enabled
        self.spectral_weight = spectral_weight
        self.spectral_target = spectral_target
        
        # NEW: Adversarial training parameters
        self.adv_enabled = adv_enabled
        self.adv_weight = adv_weight
    
    def update_adjacency_mask(self):
        """Update sparse adjacency mask based on edge_sparsity."""
        # Create a mask that retains only the top (1-edge_sparsity) edges
        values, _ = torch.sort(self.adjacency.view(-1), descending=True)
        threshold_idx = int(self.feature_dim * self.feature_dim * (1 - self.edge_sparsity))
        threshold = values[min(threshold_idx, len(values)-1)]
        
        self.adjacency_mask = (self.adjacency >= threshold).float()
    
    def graph_propagation(self, node_features: torch.Tensor) -> torch.Tensor:
        """Propagate information through the graph."""
        # Initial node embeddings
        x = self.node_embedding(node_features)
        
        # Apply graph convolution layers
        for layer in self.graph_layers:
            # Message passing: x_i' = Σ_j A_ij * W * x_j + b
            sparse_adj = self.adjacency * self.adjacency_mask
            x = torch.matmul(sparse_adj, layer(x))
            x = F.relu(x)
        
        # Return node embeddings
        return x
    
    def compute_graph_loss(
        self, 
        activations: torch.Tensor,
        model_loss_fn: Callable = None,
        targets: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute graph-based regularization loss with adversarial training."""
        # Get node features [feature_dim, 1]
        node_features = activations.mean(dim=0).unsqueeze(1)
        
        # Get node embeddings through graph propagation
        node_embeddings = self.graph_propagation(node_features)
        
        # Reconstruct original features
        reconstructed = self.readout(node_embeddings).squeeze(-1)
        recon_loss = F.mse_loss(reconstructed, activations.mean(dim=0))
        
        # Adversarial training (NEW)
        adv_loss = torch.tensor(0.0, device=self.device)
        if self.adv_enabled and model_loss_fn is not None and targets is not None:
            # Enable gradient computation for activations
            activations.requires_grad_(True)
            
            # Compute model loss and gradient
            model_loss = model_loss_fn(activations, targets)
            grad = torch.autograd.grad(model_loss, activations)[0]
            
            # Generate adversarial examples
            adv_acts = activations + 0.1 * torch.sign(grad)
            
            # Compute adversarial features and embeddings
            adv_features = adv_acts.mean(dim=0).unsqueeze(1)
            adv_embeddings = self.graph_propagation(adv_features)
            adv_recon = self.readout(adv_embeddings).squeeze(-1)
            adv_recon_loss = F.mse_loss(adv_recon, adv_acts.mean(dim=0))
            
            # Compute spectral adversarial loss
            eigenvalues = torch.linalg.eigvals(self.adjacency * self.adjacency_mask)
            max_eigenvalue = torch.max(torch.abs(eigenvalues.real))
            adv_adj = self.adjacency + 0.1 * torch.matmul(adv_acts.t(), adv_acts)
            adv_eigenvalues = torch.linalg.eigvals(adv_adj * self.adjacency_mask)
            adv_max_eigenvalue = torch.max(torch.abs(adv_eigenvalues.real))
            spectral_adv_loss = (max_eigenvalue - adv_max_eigenvalue)**2
            
            # Combine adversarial losses
            adv_loss = 0.3 * adv_recon_loss + 0.05 * spectral_adv_loss
        
        # Connectivity loss
        connectivity = self.adjacency_mask.mean()
        connectivity_loss = torch.abs(connectivity - (1 - self.edge_sparsity))
        
        # Smoothness loss
        smoothness_loss = torch.tensor(0.0, device=self.device)
        if torch.sum(self.adjacency_mask) > 0:
            node_diff = node_embeddings.unsqueeze(1) - node_embeddings.unsqueeze(0)
            node_diff_norm = torch.norm(node_diff, dim=2)
            smoothness_loss = torch.sum(self.adjacency_mask * node_diff_norm) / (torch.sum(self.adjacency_mask) + 1e-8)
        
        # Spectral regularization
        spectral_loss = torch.tensor(0.0, device=self.device)
        if self.spectral_enabled and torch.sum(self.adjacency_mask) > 0:
            eigenvalues = torch.linalg.eigvals(self.adjacency * self.adjacency_mask)
            max_eigenvalue = torch.max(torch.abs(eigenvalues.real))
            spectral_loss = (max_eigenvalue - self.spectral_target)**2
        
        # Combine all losses
        total_loss = (
            recon_loss + 
            0.1 * connectivity_loss + 
            0.1 * smoothness_loss + 
            self.spectral_weight * spectral_loss +
            (self.adv_weight if self.adv_enabled else 0.0) * adv_loss
        )
        
        # Track connectivity history
        self.connectivity_history.append(connectivity.item())
        
        # Return total loss and components
        return total_loss, {
            'recon_loss': recon_loss.item(),
            'connectivity_loss': connectivity_loss.item(),
            'smoothness_loss': smoothness_loss.item(),
            'spectral_loss': spectral_loss.item() if self.spectral_enabled else 0.0,
            'adv_loss': adv_loss.item() if self.adv_enabled else 0.0,
            'connectivity': connectivity.item(),
            'max_eigenvalue': max_eigenvalue.item() if self.spectral_enabled else 0.0
        }
    
    def update_parameters(self, gradients: torch.Tensor):
        """Update graph parameters based on gradients."""
        with torch.no_grad():
            self.adjacency -= self.learning_rate * gradients
            
            # Periodically update adjacency mask
            if self.update_count % 10 == 0:
                self.update_adjacency_mask()
        
        self.update_count += 1


class TemporalTransformerEncoder(nn.Module):
    """Time-gap aware transformer encoder."""
    
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
    
    def forward(self, src, src_key_padding_mask=None, time_gaps=None):
        """Forward pass with time-gap aware attention."""
        output = src
        
        for layer in self.layers:
            # Compute attention weights
            attn_output, attn_weights = layer.self_attn(output, output, output)
            
            # Adjust attention weights based on time gaps
            if time_gaps is not None:
                # Expand time gaps to match attention weights shape
                time_gaps = time_gaps.unsqueeze(-1).unsqueeze(-1)
                # Apply time gap penalty to attention weights
                attn_weights = attn_weights - torch.log1p(time_gaps)
                # Recompute attention output with adjusted weights
                attn_output = layer.self_attn(output, output, output, attn_weights=attn_weights)[0]
            
            # Apply layer normalization and feed-forward
            output = layer.norm1(output + layer.dropout1(attn_output))
            output = layer.norm2(output + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(output))))))
        
        return output

class TemporalPlasticityRegularizer:
    """Temporal plasticity regularizer with time-gap aware transformer."""
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        sequence_length: int = 20,
        num_heads: int = 4,
        max_sequence_length: int = 50,  # NEW: Maximum sequence length
        min_sequence_length: int = 5,   # NEW: Minimum sequence length
        time_gap_weight: float = 0.1,   # NEW: Weight for time gap influence
        learning_rate: float = 1e-3,
        device = None
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        self.time_gap_weight = time_gap_weight
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize memory buffer
        self.memory_buffer = deque(maxlen=max_sequence_length)
        
        # Initialize transformer components
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = TemporalTransformerEncoder(encoder_layer, num_layers=2)
        
        # Initialize weight prediction head
        self.weight_prediction = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim * feature_dim)
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Track statistics
        self.coherence_history = []
        self.update_count = 0
    
    def update_memory_buffer(self, activations: torch.Tensor, timestamp: float = None):
        """Update memory buffer with new activations and timestamp."""
        if activations.dim() > 2:
            activations = activations.view(-1, activations.size(-1))
        self.memory_buffer.append((activations.mean(dim=0).detach(), timestamp or time.time()))
    
    def get_temporal_sequence(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get temporal sequence and time gaps from memory buffer."""
        if len(self.memory_buffer) < self.min_sequence_length:
            return torch.zeros(1, 1, self.feature_dim, device=self.device), torch.zeros(1, 1, device=self.device)
        
        # Get sequence and timestamps
        seq, timestamps = zip(*self.memory_buffer)
        sequence = torch.stack(seq, dim=0).unsqueeze(0)
        timestamps = torch.tensor(timestamps, device=self.device).unsqueeze(0)
        
        # Compute time gaps
        time_gaps = timestamps[:, 1:] - timestamps[:, :-1]
        time_gaps = torch.cat([torch.zeros_like(time_gaps[:, :1]), time_gaps], dim=1)
        
        return sequence, time_gaps
    
    def predict_weights(self, sequence: torch.Tensor, time_gaps: torch.Tensor) -> torch.Tensor:
        """Predict weights using time-gap aware transformer."""
        # Apply transformer with time gaps
        transformer_out = self.transformer(sequence, time_gaps=time_gaps)
        
        # Get last output and predict weights
        last_out = transformer_out[:, -1, :]
        return self.weight_prediction(last_out).view(self.feature_dim, self.feature_dim)
    
    def compute_temporal_coherence(self, sequence: torch.Tensor, time_gaps: torch.Tensor) -> torch.Tensor:
        """Compute temporal coherence with time gap normalization."""
        if sequence.size(1) < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Compute differences between consecutive states
        diffs = sequence[:, 1:] - sequence[:, :-1]
        
        # Normalize differences by time gaps
        norm_diffs = torch.norm(diffs, dim=2) / (time_gaps[:, 1:] + 1e-8)
        
        # Weight by time gap importance
        weighted_diffs = norm_diffs * torch.exp(-self.time_gap_weight * time_gaps[:, 1:])
        
        return torch.mean(weighted_diffs)
    
    def compute_temporal_loss(self, activations: torch.Tensor, timestamp: float = None) -> Tuple[torch.Tensor, Dict]:
        """Compute temporal regularization loss."""
        # Update memory buffer
        self.update_memory_buffer(activations, timestamp)
        
        # Get temporal sequence and time gaps
        sequence, time_gaps = self.get_temporal_sequence()
        
        # Predict weights
        weights = self.predict_weights(sequence, time_gaps)
        
        # Compute temporal coherence
        coherence = self.compute_temporal_coherence(sequence, time_gaps)
        
        # Track coherence history
        self.coherence_history.append(coherence.item())
        self.update_count += 1
        
        return coherence, {
            'temporal_coherence': coherence.item(),
            'sequence_length': sequence.size(1),
            'time_gap_mean': time_gaps.mean().item()
        }
    
    def update_weights(self, activations: torch.Tensor):
        """Update weights based on temporal patterns."""
        # Get temporal sequence and time gaps
        sequence, time_gaps = self.get_temporal_sequence()
        
        # Predict new weights
        weights = self.predict_weights(sequence, time_gaps)
        
        # Update parameters
        self.optimizer.zero_grad()
        loss = torch.norm(weights - self.weight_prediction.weight.data)
        loss.backward()
        self.optimizer.step()


class LatentSpaceRegularizer:
    """
    Enhanced Latent Space Regularization that enforces structured sparsity, 
    feature disentanglement, and information-theoretic constraints on activations.
    
    This implementation uses multiple coordinated regularization techniques:
    1. L1/L2 sparsity penalties at multiple granularities (neuron, channel, layer)
    2. KL divergence to specific sparsity patterns
    3. Orthogonality constraints for feature disentanglement
    4. Group sparsity for structured weight pruning
    5. Hessian-weighted penalties for curvature-aware regularization
    6. Mutual information minimization between feature subspaces
    7. Bayesian Synaptic Updates for better uncertainty quantification
    8. Graph-Based Plasticity for improved feature coherence
    9. Temporal Plasticity for modeling evolving data patterns
    
    Key enhancement: Adaptive multi-level regularization that targets optimal 
    sparsity at each architectural level based on information flow analysis.
    """
    
    def __init__(
        self,
        model,
        config: LatentSpaceRegConfig = None,
        device = None
    ):
        self.model = model
        self.config = config or LatentSpaceRegConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize quantum annealer if enabled
        if self.config.use_quantum_annealing:
            if not QUANTUM_AVAILABLE:
                logger.warning("PennyLane not available. Quantum annealing will be disabled.")
                self.quantum_annealer = None
            else:
                self.quantum_annealer = QuantumAnnealer(
                    num_qubits=self.config.quantum_num_qubits,
                    circuit_depth=self.config.quantum_circuit_depth,
                    learning_rate=self.config.quantum_learning_rate
                )
        else:
            self.quantum_annealer = None
        
        # Initialize topological regularizer if enabled
        if self.config.use_topological_reg and TOPOLOGY_AVAILABLE:
            self.topological_regularizer = TopologicalRegularizer(
                target_betti_1=self.config.target_betti_1,
                max_filtration_value=self.config.max_filtration_value,
                persistence_threshold=self.config.persistence_threshold
            )
        else:
            self.topological_regularizer = None
        
        # Initialize holographic encoders for each layer
        self.holographic_encoders = {}
        if self.config.use_holographic_reg:
            for name, module in self.model.named_modules():
                if any(target_name.lower() in name.lower() 
                      for target_name in self.config.target_layers):
                    # Get feature dimension
                    if hasattr(module, 'out_features'):
                        feature_dim = module.out_features
                    elif hasattr(module, 'out_channels'):
                        feature_dim = module.out_channels
                    else:
                        continue
                        
                    # Create encoder
                    self.holographic_encoders[name] = HolographicEncoder(
                        feature_dim=feature_dim,
                        learning_rate=self.config.phase_learning_rate,
                        reconstruction_weight=self.config.reconstruction_weight,
                        sparsity_weight=self.config.sparsity_weight
                    ).to(self.device)
        
        # Synaptic plasticity regularizers
        self.plasticity_regularizers = {}
        if self.config.use_plasticity_reg:
            for name, module in self.model.named_modules():
                if any(target_name.lower() in name.lower() 
                      for target_name in self.config.target_layers):
                    # Get feature dimension
                    if hasattr(module, 'out_features'):
                        feature_dim = module.out_features
                    elif hasattr(module, 'out_channels'):
                        feature_dim = module.out_channels
                    else:
                        continue
                        
                    # Create regularizer
                    self.plasticity_regularizers[name] = SynapticPlasticityRegularizer(
                        feature_dim=feature_dim,
                        learning_rate=self.config.plasticity_learning_rate,
                        gamma=self.config.plasticity_gamma,
                        variance_weight=self.config.plasticity_variance_weight,
                        config=self.config,
                        layer_name=name
                    ).to(self.device)
        
        # Activation hooks
        self.hooks = []
        self.layer_activations = {}
        self.activation_gradients = {}  # NEW: Store activation gradients
        
        # Activation statistics with enhanced tracking
        self.activation_stats = defaultdict(lambda: {
            'mean_activation': 0.0,
            'sparsity': 0.0,
            'l1_norm': 0.0,
            'l2_norm': 0.0,  # NEW: Track L2 norm
            'entropy': 0.0,
            'orthogonality': 0.0,
            'spectral_norm': 0.0,  # NEW: Track spectral norm
            'gradient_sensitivity': 0.0,  # NEW: Track gradient sensitivity
            'history': deque(maxlen=self.config.activation_history_size),
            'sensitivity_history': deque(maxlen=self.config.layer_sensitivity_window)
        })
        
        # Current penalties with layer-wise adaptation
        self.current_penalties = {
            'l1': self.config.l1_penalty_weight,
            'kl': self.config.kl_penalty_weight,
            'orthogonal': self.config.orthogonal_penalty_weight,
            'group_sparsity': self.config.group_sparsity_weight,  # NEW
            'hessian': self.config.hessian_penalty_weight  # NEW
        }
        
        # Layer-specific penalties
        self.layer_penalties = defaultdict(lambda: self.current_penalties.copy())
        
        # Register forward and backward hooks
        self._register_hooks()
        
        # NEW: Mutual information estimation state
        if self.config.use_mutual_information:
            self.mi_estimator = LatentMutualInformationEstimator(self.device)
            self.mutual_information = {}
            self.mi_history = defaultdict(list)
        
        # Step counting
        self.global_step = 0
        
        # Metrics tracking
        self.metrics_history = []
        
        # Regularization cycle state
        self.cycle_phase = 0.0  # 0 to 1 phase within cycle
        
        logger.info(f"Initialized Enhanced LatentSpaceRegularizer with {len(self.hooks)} hooks")
        
        # Initialize Bayesian synaptic regularizers
        self.bayesian_regularizers = {}
        if self.config.bayesian_enabled:
            for name, module in self.model.named_modules():
                if any(target_name.lower() in name.lower() 
                      for target_name in self.config.target_layers):
                    # Get feature dimension
                    if hasattr(module, 'out_features'):
                        feature_dim = module.out_features
                    elif hasattr(module, 'out_channels'):
                        feature_dim = module.out_channels
                    else:
                        continue
                        
                    # Create regularizer
                    self.bayesian_regularizers[name] = BayesianSynapticRegularizer(
                        feature_dim=feature_dim,
                        learning_rate=self.config.plasticity_learning_rate,
                        prior_scale=self.config.bayesian_prior_scale,
                        kl_weight=self.config.bayesian_kl_weight,
                        device=self.device
                    )
        
        # Initialize Graph-Based plasticity regularizers
        self.graph_regularizers = {}
        if self.config.graph_plasticity_enabled:
            for name, module in self.model.named_modules():
                if any(target_name.lower() in name.lower() 
                      for target_name in self.config.target_layers):
                    # Get feature dimension
                    if hasattr(module, 'out_features'):
                        feature_dim = module.out_features
                    elif hasattr(module, 'out_channels'):
                        feature_dim = module.out_channels
                    else:
                        continue
                        
                    # Create regularizer
                    self.graph_regularizers[name] = GraphBasedPlasticityRegularizer(
                        feature_dim=feature_dim,
                        hidden_dim=self.config.graph_hidden_dim,
                        n_layers=self.config.graph_n_layers,
                        learning_rate=self.config.plasticity_learning_rate,
                        edge_sparsity=self.config.graph_edge_sparsity,
                        spectral_enabled=self.config.graph_spectral_enabled,
                        spectral_weight=self.config.graph_spectral_weight,
                        spectral_target=self.config.graph_spectral_target,
                        adv_enabled=self.config.graph_adv_enabled,
                        adv_weight=self.config.graph_adv_weight,
                        device=self.device
                    )
        
        # Initialize Temporal plasticity regularizers
        self.temporal_regularizers = {}
        if self.config.temporal_plasticity_enabled:
            for name, module in self.model.named_modules():
                if any(target_name.lower() in name.lower() 
                      for target_name in self.config.target_layers):
                    # Get feature dimension
                    if hasattr(module, 'out_features'):
                        feature_dim = module.out_features
                    elif hasattr(module, 'out_channels'):
                        feature_dim = module.out_channels
                    else:
                        continue
                        
                    # Create regularizer
                    self.temporal_regularizers[name] = TemporalPlasticityRegularizer(
                        feature_dim=feature_dim,
                        hidden_dim=self.config.temporal_hidden_dim,
                        sequence_length=self.config.temporal_sequence_length,
                        learning_rate=self.config.plasticity_learning_rate,
                        max_sequence_length=self.config.temporal_max_sequence_length,
                        min_sequence_length=self.config.temporal_min_sequence_length,
                        time_gap_weight=self.config.temporal_time_gap_weight,
                        device=self.device
                    )
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layers to capture activations and gradients."""
        hook_count = 0
        
        for name, module in self.model.named_modules():
            # Target specific layer types based on config
            should_hook = False
            
            for target_name in self.config.target_layers:
                if target_name.lower() in name.lower():
                    should_hook = True
                    break
            
            if should_hook:
                # Register forward hook
                hook = module.register_forward_hook(
                    lambda mod, inp, out, layer_name=name: self._activation_hook(mod, inp, out, layer_name)
                )
                self.hooks.append(hook)
                
                # Register backward hook for gradient sensitivity
                if self.config.layer_adaptive_penalties:
                    grad_hook = module.register_full_backward_hook(
                        lambda mod, grad_input, grad_output, layer_name=name: self._gradient_hook(mod, grad_input, grad_output, layer_name)
                    )
                    self.hooks.append(grad_hook)
                
                hook_count += 1
                
                # Initialize stats for this layer
                self.activation_stats[name] = {
                    'mean_activation': 0.0,
                    'sparsity': 0.0,
                    'l1_norm': 0.0,
                    'l2_norm': 0.0,  # NEW
                    'entropy': 0.0,
                    'orthogonality': 0.0,
                    'spectral_norm': 0.0,  # NEW
                    'gradient_sensitivity': 0.0,  # NEW
                    'history': deque(maxlen=self.config.activation_history_size),
                    'sensitivity_history': deque(maxlen=self.config.layer_sensitivity_window)
                }
                
                # Set layer weight if not specified
                if name not in self.config.layer_weights:
                    self.config.layer_weights[name] = 1.0
                
                # Initialize layer-specific penalties
                self.layer_penalties[name] = self.current_penalties.copy()
                
                logger.info(f"Registered latent space regularization hook on {name}")
        
        logger.info(f"Registered {hook_count} hooks for latent space regularization")
    
    def _activation_hook(self, module, input_tensor, output_tensor, layer_name):
        """Enhanced hook to capture layer activations with all regularization methods."""
        # Store the activations for regularization
        self.layer_activations[layer_name] = output_tensor
        
        # Apply quantum annealing if enabled
        if (self.quantum_annealer is not None and 
            self.global_step % self.config.quantum_annealing_frequency == 0):
            with torch.no_grad():
                # Reshape activations to match quantum circuit input
                activations = output_tensor.detach()
                if activations.dim() > 2:
                    activations = activations.view(-1, activations.size(-1))
                
                # Pad or truncate to match number of qubits
                if activations.size(-1) > self.config.quantum_num_qubits:
                    activations = activations[:, :self.config.quantum_num_qubits]
                else:
                    padding = torch.zeros(
                        activations.size(0),
                        self.config.quantum_num_qubits - activations.size(-1),
                        device=activations.device
                    )
                    activations = torch.cat([activations, padding], dim=-1)
                
                # Compute quantum energy
                energy = self.quantum_annealer.compute_energy(activations)
                
                # Store energy in stats
                if 'quantum_energy' not in self.activation_stats[layer_name]:
                    self.activation_stats[layer_name]['quantum_energy'] = 0.0
                
                # Update with exponential moving average
                alpha = 0.1
                self.activation_stats[layer_name]['quantum_energy'] = (
                    (1 - alpha) * self.activation_stats[layer_name]['quantum_energy'] +
                    alpha * energy.item()
                )
        
        # Apply topological regularization if enabled
        if (self.topological_regularizer is not None and 
            self.global_step % self.config.topological_frequency == 0):
            with torch.no_grad():
                # Reshape activations
                activations = output_tensor.detach()
                if activations.dim() > 2:
                    activations = activations.view(-1, activations.size(-1))
                
                # Compute Betti numbers
                betti_numbers = self.topological_regularizer.compute_betti_numbers(activations)
                
                # Store in stats
                if 'betti_numbers' not in self.activation_stats[layer_name]:
                    self.activation_stats[layer_name]['betti_numbers'] = {0: 0, 1: 0}
                
                # Update with exponential moving average
                alpha = 0.1
                for dim, count in betti_numbers.items():
                    self.activation_stats[layer_name]['betti_numbers'][dim] = (
                        (1 - alpha) * self.activation_stats[layer_name]['betti_numbers'][dim] +
                        alpha * count
                    )
        
        # Apply holographic regularization if enabled
        if (self.config.use_holographic_reg and 
            layer_name in self.holographic_encoders and
            self.global_step % self.config.holographic_frequency == 0):
            with torch.no_grad():
                # Reshape activations
                activations = output_tensor.detach()
                if activations.dim() > 2:
                    activations = activations.view(-1, activations.size(-1))
                
                # Compute holographic loss
                loss, components = self.holographic_encoders[layer_name].compute_loss(activations)
                
                # Store in stats
                if 'holographic_loss' not in self.activation_stats[layer_name]:
                    self.activation_stats[layer_name]['holographic_loss'] = 0.0
                    self.activation_stats[layer_name]['reconstruction_loss'] = 0.0
                    self.activation_stats[layer_name]['sparsity_loss'] = 0.0
                
                # Update with exponential moving average
                alpha = 0.1
                self.activation_stats[layer_name]['holographic_loss'] = (
                    (1 - alpha) * self.activation_stats[layer_name]['holographic_loss'] +
                    alpha * loss.item()
                )
                self.activation_stats[layer_name]['reconstruction_loss'] = (
                    (1 - alpha) * self.activation_stats[layer_name]['reconstruction_loss'] +
                    alpha * components['reconstruction_loss']
                )
                self.activation_stats[layer_name]['sparsity_loss'] = (
                    (1 - alpha) * self.activation_stats[layer_name]['sparsity_loss'] +
                    alpha * components['sparsity_loss']
                )
        
        # Apply synaptic plasticity regularization if enabled
        if (self.config.use_plasticity_reg and 
            layer_name in self.plasticity_regularizers and
            self.global_step % self.config.plasticity_frequency == 0):
            with torch.no_grad():
                # Get activations
                activations = self.layer_activations[layer_name]
                
                # Compute plasticity loss
                plasticity_loss = self.plasticity_regularizers[layer_name].compute_plasticity_loss(activations)
                
                # Update weights
                self.plasticity_regularizers[layer_name].update_weights(plasticity_loss.grad)
                
                # Update statistics
                self.plasticity_regularizers[layer_name].update_statistics(activations)
        
        # Apply Bayesian synaptic regularization if enabled
        if (self.config.bayesian_enabled and 
            layer_name in self.bayesian_regularizers):
            with torch.no_grad():
                # Get activations
                activations = output_tensor.detach()
                
                # Update Bayesian weights
                loss, stats = self.bayesian_regularizers[layer_name].update_from_activations(activations)
                
                # Store in activation stats
                if 'bayesian_loss' not in self.activation_stats[layer_name]:
                    self.activation_stats[layer_name]['bayesian_loss'] = 0.0
                    self.activation_stats[layer_name]['uncertainty'] = 0.0
                
                # Update with exponential moving average
                alpha = 0.1
                self.activation_stats[layer_name]['bayesian_loss'] = (
                    (1 - alpha) * self.activation_stats[layer_name]['bayesian_loss'] +
                    alpha * loss.item()
                )
                self.activation_stats[layer_name]['uncertainty'] = (
                    (1 - alpha) * self.activation_stats[layer_name]['uncertainty'] +
                    alpha * stats['uncertainty']
                )
        
        # Apply Graph-Based plasticity if enabled
        if (self.config.graph_plasticity_enabled and 
            layer_name in self.graph_regularizers):
            with torch.no_grad():
                # Get activations
                activations = output_tensor.detach()
                
                # Compute graph-based loss
                loss, stats = self.graph_regularizers[layer_name].compute_graph_loss(activations)
                
                # Store in stats
                if 'graph_loss' not in self.activation_stats[layer_name]:
                    self.activation_stats[layer_name]['graph_loss'] = 0.0
                    self.activation_stats[layer_name]['graph_connectivity'] = 0.0
                
                # Update with exponential moving average
                alpha = 0.1
                self.activation_stats[layer_name]['graph_loss'] = (
                    (1 - alpha) * self.activation_stats[layer_name]['graph_loss'] +
                    alpha * loss.item()
                )
                self.activation_stats[layer_name]['graph_connectivity'] = (
                    (1 - alpha) * self.activation_stats[layer_name]['graph_connectivity'] +
                    alpha * stats['connectivity']
                )
        
        # Apply Temporal plasticity if enabled
        if (self.config.temporal_plasticity_enabled and 
            layer_name in self.temporal_regularizers and
            self.global_step % self.config.temporal_update_frequency == 0):
            with torch.no_grad():
                # Get activations
                activations = output_tensor.detach()
                
                # Compute temporal loss
                loss, stats = self.temporal_regularizers[layer_name].compute_temporal_loss(activations)
                
                # Update temporal weights
                self.temporal_regularizers[layer_name].update_weights(activations)
                
                # Store in stats
                if 'temporal_loss' not in self.activation_stats[layer_name]:
                    self.activation_stats[layer_name]['temporal_loss'] = 0.0
                    self.activation_stats[layer_name]['temporal_coherence'] = 0.0
                
                # Update with exponential moving average
                alpha = 0.1
                self.activation_stats[layer_name]['temporal_loss'] = (
                    (1 - alpha) * self.activation_stats[layer_name]['temporal_loss'] +
                    alpha * loss.item()
                )
                self.activation_stats[layer_name]['temporal_coherence'] = (
                    (1 - alpha) * self.activation_stats[layer_name]['temporal_coherence'] +
                    alpha * stats['temporal_coherence']
                )
    
    def _gradient_hook(self, module, grad_input, grad_output, layer_name):
        """NEW: Hook to capture gradient information for sensitivity analysis."""
        if grad_output[0] is not None:
            self.activation_gradients[layer_name] = grad_output[0].detach()
            
            # Compute gradient statistics if we have both activations and gradients
            if layer_name in self.layer_activations and layer_name in self.activation_gradients:
                self._update_gradient_sensitivity(layer_name)
    
    def _update_gradient_sensitivity(self, layer_name):
        """NEW: Update layer sensitivity based on activation-gradient relationship."""
        activations = self.layer_activations[layer_name].detach()
        gradients = self.activation_gradients[layer_name]
        
        # Reshape if needed
        if activations.dim() > 2:
            activations = activations.view(-1, activations.size(-1))
        
        if gradients.dim() > 2:
            gradients = gradients.view(-1, gradients.size(-1))
        
        # Compute activation-gradient alignment
        act_norm = torch.norm(activations, p=2, dim=0) + 1e-8
        grad_norm = torch.norm(gradients, p=2, dim=0) + 1e-8
        
        # Normalize activations and gradients
        act_normalized = activations / act_norm
        grad_normalized = gradients / grad_norm
        
        # Compute sensitivity as alignment between activation and gradient directions
        # High alignment means changes in this direction strongly affect the loss
        alignment = torch.abs(torch.sum(act_normalized * grad_normalized, dim=0))
        
        # Average sensitivity across units
        sensitivity = alignment.mean().item()
        
        # Store in history
        self.activation_stats[layer_name]['sensitivity_history'].append(sensitivity)
        
        # Update running average
        alpha = 0.1  # Update rate
        stats = self.activation_stats[layer_name]
        stats['gradient_sensitivity'] = (1 - alpha) * stats['gradient_sensitivity'] + alpha * sensitivity
        
        # Adjust layer penalties based on sensitivity if adaptive
        if self.config.layer_adaptive_penalties and len(stats['sensitivity_history']) >= 3:
            self._adapt_layer_penalties(layer_name)
    
    def _adapt_layer_penalties(self, layer_name):
        """NEW: Adapt penalties for a specific layer based on its sensitivity."""
        sensitivity = self.activation_stats[layer_name]['gradient_sensitivity']
        
        # Higher sensitivity -> stronger regularization (more impact on learning)
        # Lower sensitivity -> weaker regularization (less impact on learning)
        
        # Compute adaptive factor - normalize sensitivity to reasonable range
        mean_sensitivity = np.mean([
            self.activation_stats[name]['gradient_sensitivity'] 
            for name in self.activation_stats.keys()
        ])
        
        # Adjustment factor: how much this layer differs from average sensitivity
        # Use sigmoid scaling to keep adjustments in reasonable range
        sensitivity_ratio = sensitivity / (mean_sensitivity + 1e-8)
        adjustment_factor = 2.0 / (1.0 + math.exp(-2.0 * sensitivity_ratio)) - 0.5  # Range ~0.5 to ~1.5
        
        # Apply adjustment to layer penalties
        for penalty_type in self.current_penalties:
            base_penalty = self.current_penalties[penalty_type]
            adjusted_penalty = base_penalty * adjustment_factor
            
            # Ensure within bounds
            adjusted_penalty = max(
                self.config.min_penalty_weight,
                min(self.config.max_penalty_weight, adjusted_penalty)
            )
            
            self.layer_penalties[layer_name][penalty_type] = adjusted_penalty
    
    def _update_activation_stats(self, layer_name, activations):
        """
        Enhanced update of statistics for layer activations.
        
        Args:
            layer_name: Name of the layer
            activations: Tensor of activations
        """
        # Handle different tensor shapes
        orig_shape = activations.shape
        if activations.dim() > 2:
            activations = activations.view(-1, activations.size(-1))
        
        # Compute sparsity (proportion of near-zero activations)
        sparsity = (activations.abs() < 1e-5).float().mean().item()
        
        # Compute L1 norm (average)
        l1_norm = activations.abs().mean().item()
        
        # Compute L2 norm (average) - NEW
        l2_norm = torch.norm(activations, p=2, dim=1).mean().item()
        
        # Compute entropy
        act_norm = F.softmax(activations.abs(), dim=-1)
        entropy = -torch.sum(act_norm * torch.log(act_norm + 1e-10)) / activations.size(-1)
        entropy = entropy.mean().item()
        
        # Compute orthogonality measure
        orthogonality = 0.0
        if activations.size(-1) > 1 and activations.size(0) > 1:
            normalized = F.normalize(activations, p=2, dim=-1)
            similarities = torch.matmul(normalized, normalized.transpose(-2, -1))
            
            mask = 1.0 - torch.eye(similarities.size(0), device=similarities.device)
            masked_similarities = similarities * mask
            
            orthogonality = masked_similarities.abs().mean().item()
        
        # Compute spectral norm periodically - NEW
        spectral_norm = 0.0
        if self.config.track_spectral_norm and self.global_step % self.config.spectral_norm_frequency == 0:
            if activations.size(0) > 1 and activations.size(1) > 1:
                # Use power iteration to approximate spectral norm
                u = torch.randn(activations.size(1), 1, device=activations.device)
                u = F.normalize(u, dim=0)
                
                for _ in range(10):  # 10 power iterations
                    v = F.normalize(torch.matmul(activations.t(), torch.matmul(activations, u)), dim=0)
                    u = F.normalize(torch.matmul(activations.t(), torch.matmul(activations, v)), dim=0)
                
                spectral_norm = torch.sqrt(torch.matmul(v.t(), torch.matmul(activations.t(), 
                                                    torch.matmul(activations, u)))).item()
        
        # Update stats with exponential moving average
        alpha = 0.1  # Update rate
        stats = self.activation_stats[layer_name]
        stats['mean_activation'] = (1 - alpha) * stats['mean_activation'] + alpha * activations.mean().item()
        stats['sparsity'] = (1 - alpha) * stats['sparsity'] + alpha * sparsity
        stats['l1_norm'] = (1 - alpha) * stats['l1_norm'] + alpha * l1_norm
        stats['l2_norm'] = (1 - alpha) * stats['l2_norm'] + alpha * l2_norm
        stats['entropy'] = (1 - alpha) * stats['entropy'] + alpha * entropy
        stats['orthogonality'] = (1 - alpha) * stats['orthogonality'] + alpha * orthogonality
        
        if spectral_norm > 0:
            stats['spectral_norm'] = (1 - alpha) * stats['spectral_norm'] + alpha * spectral_norm
        
        # Add to history
        stats['history'].append({
            'step': self.global_step,
            'mean_activation': activations.mean().item(),
            'sparsity': sparsity,
            'l1_norm': l1_norm,
            'l2_norm': l2_norm,
            'entropy': entropy,
            'orthogonality': orthogonality,
            'spectral_norm': spectral_norm if spectral_norm > 0 else stats['spectral_norm']
        })
    
    def _get_regularization_factor(self, layer_name=None) -> float:
        """
        Enhanced regularization scheduling with layer-specific cycles 
        for better curriculum learning.
        
        Args:
            layer_name: Optional name of layer for layer-specific scheduling
            
        Returns:
            Regularization factor between 0.0 and 1.0
        """
        # No regularization before start step
        if self.global_step < self.config.integration_start_step:
            return 0.0
        
        # Layer-specific scheduling for cyclical pattern
        if layer_name is not None and self.config.layer_specific_cycles and self.config.penalty_schedule == "cyclical":
            # Compute layer-specific cycle length based on sensitivity
            base_cycle_length = self.config.cycle_length
            
            if layer_name in self.activation_stats and 'gradient_sensitivity' in self.activation_stats[layer_name]:
                # Get layer sensitivity
                layer_sensitivity = self.activation_stats[layer_name]['gradient_sensitivity']
                
                # Compute mean sensitivity across all layers
                all_sensitivities = [stats['gradient_sensitivity'] 
                                    for name, stats in self.activation_stats.items()]
                mean_sensitivity = sum(all_sensitivities) / max(1, len(all_sensitivities))
                
                # More sensitive layers get longer cycles to allow more exploration
                sensitivity_ratio = layer_sensitivity / (mean_sensitivity + 1e-8)
                alpha = self.config.layer_cycle_alpha  # Weight for sensitivity adjustment
                
                # T_l = T · (1 + α · σ_l/σ̄)
                cycle_length = int(base_cycle_length * (1.0 + alpha * (sensitivity_ratio - 1.0)))
                
                # Ensure reasonable bounds
                cycle_length = max(base_cycle_length // 2, min(base_cycle_length * 2, cycle_length))
            else:
                cycle_length = base_cycle_length
            
            # Compute phase within cycle (0 to 1)
            cycle_progress = ((self.global_step - self.config.integration_start_step) % 
                             cycle_length) / cycle_length
            
            # Store cycle phase for the layer
            if not hasattr(self, 'layer_cycle_phases'):
                self.layer_cycle_phases = {}
            self.layer_cycle_phases[layer_name] = cycle_progress
            
            # Sinusoidal variation between min and max factor
            min_factor = self.config.cycle_min_factor
            max_factor = self.config.cycle_max_factor
            factor = min_factor + (max_factor - min_factor) * 0.5 * (1 + math.cos(math.pi * (1 - cycle_progress)))
            
            return factor
        
        # Handle different schedule types for global scheduling
        if self.config.penalty_schedule == "cyclical":
            # Compute phase within cycle (0 to 1)
            cycle_progress = ((self.global_step - self.config.integration_start_step) % 
                             self.config.cycle_length) / self.config.cycle_length
            self.cycle_phase = cycle_progress
            
            # Sinusoidal variation between min and max factor
            min_factor = self.config.cycle_min_factor
            max_factor = self.config.cycle_max_factor
            factor = min_factor + (max_factor - min_factor) * 0.5 * (1 + math.cos(math.pi * (1 - cycle_progress)))
            
            return factor
        
        # Standard warmup/cooldown scheduling
        warmup_steps = self.config.warmup_steps
        warmup_end = self.config.integration_start_step + warmup_steps
        
        if self.global_step < warmup_end:
            # In warmup phase
            progress = (self.global_step - self.config.integration_start_step) / warmup_steps
            
            if self.config.schedule_type == "linear":
                return progress
            elif self.config.schedule_type == "exponential":
                return 1.0 - math.exp(-5 * progress)
            elif self.config.schedule_type == "cosine":
                return 0.5 * (1 - math.cos(math.pi * progress))
            else:
                return progress  # Default to linear
        
        # After warmup, check for cooldown
        cooldown_start = warmup_end + self.config.cooldown_steps
        
        if self.global_step > cooldown_start:
            # In cooldown phase
            cooldown_progress = (self.global_step - cooldown_start) / warmup_steps
            cooldown_progress = min(1.0, cooldown_progress)
            
            if self.config.schedule_type == "linear":
                return max(0.0, 1.0 - cooldown_progress)
            elif self.config.schedule_type == "exponential":
                return math.exp(-5 * cooldown_progress)
            elif self.config.schedule_type == "cosine":
                return 0.5 * (1 + math.cos(math.pi * cooldown_progress))
            else:
                return max(0.0, 1.0 - cooldown_progress)  # Default to linear
        
        # Between warmup and cooldown, use full regularization
        return 1.0
    
    def _reduce_dimension(self, x: torch.Tensor, target_dim: int = 128) -> torch.Tensor:
        """
        Reduce dimensionality using PCA for more efficient MI estimation.
        
        Args:
            x: Input tensor [batch_size, feature_dim]
            target_dim: Target dimensionality
            
        Returns:
            Reduced tensor [batch_size, target_dim]
        """
        # Center data
        x_mean = x.mean(dim=0, keepdim=True)
        x_centered = x - x_mean
        
        # Compute covariance matrix
        n_samples = x.size(0)
        cov = torch.matmul(x_centered.t(), x_centered) / (n_samples - 1)
        
        # Compute eigenvectors/values
        eigvals, eigvecs = torch.linalg.eigh(cov)
        
        # Sort eigenvectors by eigenvalues in descending order
        _, indices = torch.sort(eigvals, descending=True)
        eigvecs = eigvecs[:, indices]
        
        # Select top eigenvectors
        projection = eigvecs[:, :target_dim]
        
        # Project data
        x_reduced = torch.matmul(x_centered, projection)
        
        return x_reduced

    def compute_regularization_loss(
        self, 
        val_loss: float = None,
        model_loss_fn: Callable = None,
        targets: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute total regularization loss including all methods."""
        # Initialize total loss and components
        reg_loss = torch.tensor(0.0, device=self.device)
        components = {}
        
        # Add quantum annealing loss if enabled
        if self.quantum_annealer is not None:
            quantum_loss = torch.tensor(0.0, device=self.device)
            
            for layer_name, activations in self.layer_activations.items():
                if self.global_step % self.config.quantum_annealing_frequency == 0:
                    # Reshape activations
                    if activations.dim() > 2:
                        activations = activations.view(-1, activations.size(-1))
                    
                    # Pad or truncate
                    if activations.size(-1) > self.config.quantum_num_qubits:
                        activations = activations[:, :self.config.quantum_num_qubits]
                    else:
                        padding = torch.zeros(
                            activations.size(0),
                            self.config.quantum_num_qubits - activations.size(-1),
                            device=activations.device
                        )
                        activations = torch.cat([activations, padding], dim=-1)
                    
                    # Compute quantum energy
                    energy = self.quantum_annealer.compute_energy(activations)
                    quantum_loss += energy
            
            # Scale quantum loss
            quantum_loss *= self.config.quantum_annealing_weight
            
            # Add to total loss
            reg_loss += quantum_loss
            components['quantum_loss'] = quantum_loss.item()
        
        # Add topological regularization loss if enabled
        if self.topological_regularizer is not None:
            topological_loss = torch.tensor(0.0, device=self.device)
            
            for layer_name, activations in self.layer_activations.items():
                if self.global_step % self.config.topological_frequency == 0:
                    # Reshape activations
                    if activations.dim() > 2:
                        activations = activations.view(-1, activations.size(-1))
                    
                    # Compute topological loss
                    loss = self.topological_regularizer.compute_topological_loss(activations)
                    topological_loss += loss
            
            # Scale topological loss
            topological_loss *= self.config.topological_weight
            
            # Add to total loss
            reg_loss += topological_loss
            components['topological_loss'] = topological_loss.item()
        
        # Add holographic regularization loss if enabled
        if self.config.use_holographic_reg:
            holographic_loss = torch.tensor(0.0, device=self.device)
            
            for layer_name, encoder in self.holographic_encoders.items():
                if layer_name in self.layer_activations and self.global_step % self.config.holographic_frequency == 0:
                    # Get activations
                    activations = self.layer_activations[layer_name]
                    
                    # Reshape if needed
                    if activations.dim() > 2:
                        activations = activations.view(-1, activations.size(-1))
                    
                    # Compute holographic loss
                    loss, _ = encoder.compute_loss(activations)
                    holographic_loss += loss
            
            # Scale holographic loss
            holographic_loss *= self.config.holographic_weight
            
            # Add to total loss
            reg_loss += holographic_loss
            components['holographic_loss'] = holographic_loss.item()
        
        # Add synaptic plasticity loss if enabled
        if self.config.use_plasticity_reg:
            plasticity_loss = torch.tensor(0.0, device=self.device)
            
            for layer_name, regularizer in self.plasticity_regularizers.items():
                if layer_name in self.layer_activations and self.global_step % self.config.plasticity_frequency == 0:
                    # Get activations
                    activations = self.layer_activations[layer_name]
                    
                    # Compute plasticity loss with validation loss if available
                    plasticity_loss += self.config.plasticity_weight * regularizer.compute_plasticity_loss(activations, val_loss)
            
            # Add to total loss
            reg_loss += plasticity_loss
            components['plasticity_loss'] = plasticity_loss.item()
        
        # Add Bayesian synaptic loss if enabled
        if self.config.bayesian_enabled:
            bayesian_loss = torch.tensor(0.0, device=self.device)
            
            for layer_name, regularizer in self.bayesian_regularizers.items():
                if layer_name in self.layer_activations:
                    # Get activations
                    activations = self.layer_activations[layer_name]
                    
                    # Compute Bayesian loss with optional adversarial updates
                    loss, _ = regularizer.update_from_activations(
                        activations,
                        model_loss_fn if self.config.bayesian_adversarial_enabled else None,
                        targets if self.config.bayesian_adversarial_enabled else None
                    )
                    bayesian_loss += loss
            
            # Scale Bayesian loss
            bayesian_loss *= self.config.bayesian_weight
            
            # Add to total loss
            reg_loss += bayesian_loss
            components['bayesian_loss'] = bayesian_loss.item()
        
        # Add Graph-Based plasticity loss if enabled
        if self.config.graph_plasticity_enabled:
            graph_loss = torch.tensor(0.0, device=self.device)
            
            for layer_name, regularizer in self.graph_regularizers.items():
                if layer_name in self.layer_activations:
                    # Get activations
                    activations = self.layer_activations[layer_name]
                    
                    # Compute graph loss
                    loss, _ = regularizer.compute_graph_loss(activations)
                    graph_loss += loss
            
            # Scale graph loss
            graph_loss *= self.config.graph_plasticity_weight
            
            # Add to total loss
            reg_loss += graph_loss
            components['graph_loss'] = graph_loss.item()
        
        # Add Temporal plasticity loss if enabled
        if self.config.temporal_plasticity_enabled:
            temporal_loss = torch.tensor(0.0, device=self.device)
            
            for layer_name, regularizer in self.temporal_regularizers.items():
                if layer_name in self.layer_activations:
                    # Get activations
                    activations = self.layer_activations[layer_name]
                    
                    # Compute temporal loss
                    loss, _ = regularizer.compute_temporal_loss(activations)
                    temporal_loss += loss
            
            # Scale temporal loss
            temporal_loss *= self.config.temporal_plasticity_weight
            
            # Add to total loss
            reg_loss += temporal_loss
            components['temporal_loss'] = temporal_loss.item()
        
        return reg_loss, components


# Example integration with training loop
def create_latent_space_regularizer(model: torch.nn.Module) -> Tuple[LatentSpaceRegularizer, LatentSpaceRegConfig]:
    """
    Create a latent space regularizer with balanced default configuration,
    including Bayesian, Graph-Based, and Temporal plasticity.
    
    Args:
        model: The neural network model
        
    Returns:
        Tuple of (regularizer, config)
    """
    # Create config with balanced regularization weights
    config = LatentSpaceRegConfig(
        # Core regularization parameters - balanced for stability
        l1_penalty_weight=5e-6,
        kl_penalty_weight=3e-6,
        orthogonal_penalty_weight=1e-6,
        group_sparsity_weight=2e-6,
        hessian_penalty_weight=5e-7,
        
        # Sparsity targets - set for optimal information capacity
        sparsity_target=0.15,  # 15% overall activation
        feature_target_sparsity=0.10,  # 10% active features
        channel_target_sparsity=0.25,  # 25% active channels
        
        # Target transformer attention and MLP layers
        target_layers=["attention", "mlp", "feed_forward", "output"],
        
        # Enable layer-wise adaptation
        layer_adaptive_penalties=True,
        
        # Use cyclical scheduling for better exploration vs exploitation
        penalty_schedule="cyclical",
        cycle_length=1000,
        
        # Enable MI-based regularization
        use_mutual_information=True,
        
        # Enable Bayesian regularization
        bayesian_enabled=True,
        bayesian_weight=0.03,
        bayesian_prior_scale=1.0,
        bayesian_kl_weight=0.01,
        bayesian_rank=32,  # NEW: Default rank for low-rank parameterization
        bayesian_adversarial_enabled=False,
        bayesian_adversarial_weight=0.05,
        
        # Enable Graph-Based plasticity
        graph_plasticity_enabled=True,
        graph_plasticity_weight=0.03,
        graph_hidden_dim=64,
        graph_n_layers=2,
        graph_edge_sparsity=0.2,
        graph_spectral_enabled=True,  # NEW: Enable spectral regularization
        graph_spectral_weight=0.05,   # NEW: Weight for spectral loss
        graph_spectral_target=1.0,    # NEW: Target spectral norm
        graph_adv_enabled=True,  # NEW: Enable adversarial training
        graph_adv_weight=0.05,   # NEW: Weight for adversarial loss
        
        # Enable Temporal plasticity
        temporal_plasticity_enabled=True,
        temporal_plasticity_weight=0.03,
        temporal_hidden_dim=64,
        temporal_sequence_length=20,
        temporal_num_heads=4,
        temporal_max_sequence_length=50,  # NEW: Maximum sequence length
        temporal_min_sequence_length=5,   # NEW: Minimum sequence length
        temporal_time_gap_weight=0.1,     # NEW: Weight for time gap influence
        
        # ... rest of existing config ...
    )
    
    # Create regularizer
    regularizer = LatentSpaceRegularizer(model, config)
    
    return regularizer, config


# Example integration function
def add_latent_regularization_to_loss(
    model: torch.nn.Module,
    loss_fn: Callable,
    config: LatentSpaceRegConfig = None,
    device = None
) -> Tuple[Callable, LatentSpaceRegularizer]:
    """
    Enhance an existing loss function with latent space regularization.
    
    Args:
        model: Neural network model to regularize
        loss_fn: Original loss function with signature (outputs, targets) -> loss
        config: Optional regularization config, or None to use defaults
        device: Optional device to use for computation
        
    Returns:
        Tuple of (enhanced_loss_function, regularizer_instance)
    """
    # Create regularizer
    regularizer = LatentSpaceRegularizer(model, config, device)
    
    # Define wrapped loss function with regularization
    def regularized_loss_fn(outputs, targets, val_loss_delta=None):
        # Compute original loss
        original_loss = loss_fn(outputs, targets)
        
        # Step the regularizer (computes activation statistics)
        reg_components = regularizer.step(val_loss_delta)
        
        # Get regularization loss
        reg_loss, _ = regularizer.compute_regularization_loss()
        
        # Combine losses
        total_loss = original_loss + reg_loss
        
        # Log combined loss periodically
        if regularizer.global_step % regularizer.config.log_frequency == 0 and regularizer.config.verbose:
            logger.info(f"Total loss at step {regularizer.global_step}: "
                       f"Original={original_loss.item():.4e}, "
                       f"Regularization={reg_loss.item():.4e}, "
                       f"Combined={total_loss.item():.4e}")
        
        return total_loss
    
    return regularized_loss_fn, regularizer


def visualize_regularization_effects(regularizer: LatentSpaceRegularizer, save_path: str = None):
    """
    Generate visualizations of regularization effects on activations.
    
    Args:
        regularizer: LatentSpaceRegularizer instance with history
        save_path: Optional path to save visualizations
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        logger.warning("Matplotlib not available for visualization. Install with 'pip install matplotlib'")
        return
    
    # Skip if no history
    if not regularizer.metrics_history:
        logger.warning("No metrics history available for visualization")
        return
    
    # Extract history data
    steps = [m['global_step'] for m in regularizer.metrics_history]
    
    # Plot overall metrics
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 3, figure=fig)
    
    # Sparsity over time
    ax1 = fig.add_subplot(gs[0, 0])
    if 'avg_sparsity' in regularizer.metrics_history[0]:
        sparsity_values = [m.get('avg_sparsity', 0) for m in regularizer.metrics_history]
        ax1.plot(steps, sparsity_values)
        ax1.set_title('Average Activation Sparsity')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Sparsity Ratio')
    
    # L1 norm over time
    ax2 = fig.add_subplot(gs[0, 1])
    if 'avg_l1_norm' in regularizer.metrics_history[0]:
        l1_values = [m.get('avg_l1_norm', 0) for m in regularizer.metrics_history]
        ax2.plot(steps, l1_values)
        ax2.set_title('Average L1 Norm')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('L1 Norm')
    
    # Orthogonality over time
    ax3 = fig.add_subplot(gs[0, 2])
    if 'avg_orthogonality' in regularizer.metrics_history[0]:
        ortho_values = [m.get('avg_orthogonality', 0) for m in regularizer.metrics_history]
        ax3.plot(steps, ortho_values)
        ax3.set_title('Average Feature Orthogonality')
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Cosine Similarity')
    
    # Regularization loss components
    ax4 = fig.add_subplot(gs[1, :])
    loss_keys = ['l1_loss', 'kl_loss', 'orthogonal_loss', 'group_sparsity_loss', 'hessian_loss', 'mi_loss', 'spectral_loss', 'plasticity_loss', 'bayesian_loss', 'graph_loss', 'temporal_loss']
    for key in loss_keys:
        if key in regularizer.metrics_history[0]:
            values = [m.get(key, 0) for m in regularizer.metrics_history]
            ax4.plot(steps, values, label=key)
    ax4.set_title('Regularization Loss Components')
    ax4.set_xlabel('Steps')
    ax4.set_ylabel('Loss Value')
    ax4.legend()
    
    # Penalties over time
    ax5 = fig.add_subplot(gs[2, :])
    if 'penalties' in regularizer.metrics_history[0]:
        for penalty_type in regularizer.metrics_history[0]['penalties']:
            values = [m['penalties'].get(penalty_type, 0) for m in regularizer.metrics_history]
            ax5.plot(steps, values, label=penalty_type)
        ax5.set_title('Penalty Weights Over Time')
        ax5.set_xlabel('Steps')
        ax5.set_ylabel('Weight Value')
        ax5.set_yscale('log')
        ax5.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved regularization visualization to {save_path}")
    else:
        plt.show()


def auto_tune_regularization(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    n_trials: int = 10,
    trial_steps: int = 100,
    device = None
) -> LatentSpaceRegConfig:
    """
    Auto-tune latent space regularization parameters using validation performance.
    
    Args:
        model: Model to tune regularization for
        dataloader: Training data loader
        val_dataloader: Validation data loader
        loss_fn: Base loss function
        n_trials: Number of trials to run
        trial_steps: Steps per trial
        device: Computation device
        
    Returns:
        Optimized LatentSpaceRegConfig
    """
    try:
        import optuna
    except ImportError:
        logger.error("Optuna not available for hyperparameter tuning. Install with 'pip install optuna'")
        # Return default config
        return LatentSpaceRegConfig()
    
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    def objective(trial):
        # Sample hyperparameters
        config = LatentSpaceRegConfig(
            # Core regularization weights
            l1_penalty_weight=trial.suggest_float('l1_penalty_weight', 1e-7, 1e-4, log=True),
            kl_penalty_weight=trial.suggest_float('kl_penalty_weight', 1e-7, 1e-4, log=True),
            orthogonal_penalty_weight=trial.suggest_float('orthogonal_penalty_weight', 1e-7, 1e-4, log=True),
            group_sparsity_weight=trial.suggest_float('group_sparsity_weight', 1e-7, 1e-4, log=True),
            
            # Sparsity targets
            feature_target_sparsity=trial.suggest_float('feature_target_sparsity', 0.05, 0.3),
            
            # Scheduling
            warmup_steps=trial.suggest_int('warmup_steps', 100, 1000),
            penalty_schedule=trial.suggest_categorical('penalty_schedule', 
                                                     ['linear', 'cosine', 'cyclical']),
        )
        
        # Create regularized loss function
        reg_loss_fn, regularizer = add_latent_regularization_to_loss(
            model, loss_fn, config, device
        )
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        for step in range(trial_steps):
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = reg_loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                
                if batch_idx >= 5:  # Limit batches per step
                    break
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()
                
                # Accuracy for classification tasks
                if outputs.size(1) > 1:
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item()
                    total += targets.size(0)
        
        val_loss = val_loss / len(val_dataloader)
        accuracy = correct / total if total > 0 else 0
        
        # We want to maximize accuracy and minimize validation loss
        return accuracy if total > 0 else -val_loss
    
    # Create and run study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # Get best parameters
    best_params = study.best_params
    logger.info(f"Best regularization parameters: {best_params}")
    
    # Create and return optimized config
    optimized_config = LatentSpaceRegConfig(
        l1_penalty_weight=best_params.get('l1_penalty_weight', 5e-6),
        kl_penalty_weight=best_params.get('kl_penalty_weight', 3e-6),
        orthogonal_penalty_weight=best_params.get('orthogonal_penalty_weight', 1e-6),
        group_sparsity_weight=best_params.get('group_sparsity_weight', 2e-6),
        feature_target_sparsity=best_params.get('feature_target_sparsity', 0.1),
        warmup_steps=best_params.get('warmup_steps', 1000),
        penalty_schedule=best_params.get('penalty_schedule', 'cyclical'),
    )
    
    return optimized_config


def analyze_feature_importance(regularizer: LatentSpaceRegularizer) -> Dict[str, np.ndarray]:
    """
    Analyze feature importance in each layer based on regularization statistics.
    
    Args:
        regularizer: Trained regularizer with history
        
    Returns:
        Dictionary mapping layer names to feature importance arrays
    """
    importance_scores = {}
    
    for layer_name, stats in regularizer.activation_stats.items():
        if 'sensitivity_history' in stats and len(stats['sensitivity_history']) > 0:
            # Get feature dimensions from layer activations
            if layer_name in regularizer.layer_activations:
                activations = regularizer.layer_activations[layer_name]
                
                # Reshape if needed
                if activations.dim() > 2:
                    activations = activations.view(-1, activations.size(-1))
                
                # Get feature dimensions
                num_features = activations.size(-1)
                
                # Initialize importance scores
                importance = np.ones(num_features)
                
                # If we have per-feature gradient sensitivity
                if layer_name in regularizer.activation_gradients:
                    gradients = regularizer.activation_gradients[layer_name]
                    
                    # Reshape if needed
                    if gradients.dim() > 2:
                        gradients = gradients.view(-1, gradients.size(-1))
                    
                    # Check if dimensions match
                    if gradients.size(-1) == num_features:
                        # Compute feature importance as gradient magnitude
                        with torch.no_grad():
                            importance = torch.mean(gradients.abs(), dim=0).cpu().numpy()
                
                importance_scores[layer_name] = importance
    
    return importance_scores


def export_regularization_config(config: LatentSpaceRegConfig, filepath: str):
    """
    Export regularization configuration to a file.
    
    Args:
        config: Configuration to export
        filepath: Path to save the configuration file
    """
    import json
    
    # Convert dataclass to dictionary
    config_dict = {}
    for field_name in config.__dataclass_fields__:
        value = getattr(config, field_name)
        
        # Handle special types
        if isinstance(value, dict):
            # Convert dict values to serializable types
            config_dict[field_name] = {k: float(v) if isinstance(v, (int, float)) else v 
                                     for k, v in value.items()}
        elif isinstance(value, list):
            config_dict[field_name] = value
        else:
            config_dict[field_name] = value
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info(f"Exported regularization configuration to {filepath}")


def import_regularization_config(filepath: str) -> LatentSpaceRegConfig:
    """
    Import regularization configuration from a file.
    
    Args:
        filepath: Path to the configuration file
        
    Returns:
        Imported LatentSpaceRegConfig
    """
    import json
    
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    
    # Create config with loaded values
    config = LatentSpaceRegConfig()
    
    # Set fields from dictionary
    for field_name, value in config_dict.items():
        if hasattr(config, field_name):
            setattr(config, field_name, value)
    
    logger.info(f"Imported regularization configuration from {filepath}")
    return config


class RegularizationScheduler:
    """
    Advanced scheduler for regularization throughout training.
    Provides coordinated scheduling of multiple regularization techniques.
    """
    
    def __init__(
        self, 
        regularizer: LatentSpaceRegularizer,
        total_steps: int,
        schedule_type: str = "adaptive"
    ):
        self.regularizer = regularizer
        self.total_steps = total_steps
        self.schedule_type = schedule_type
        self.current_step = 0
        
        # History tracking
        self.schedule_history = []
        
        # Performance metrics for adaptive scheduling
        self.loss_history = deque(maxlen=100)
        self.val_loss_history = deque(maxlen=10)
        
        logger.info(f"Initialized RegularizationScheduler with {schedule_type} scheduling")
    
    def step(self, training_loss: float = None, val_loss: float = None):
        """
        Update regularization schedule based on current step and metrics.
        
        Args:
            training_loss: Current training loss (optional)
            val_loss: Current validation loss (optional)
        """
        self.current_step += 1
        
        # Update history
        if training_loss is not None:
            self.loss_history.append(training_loss)
        
        if val_loss is not None:
            self.val_loss_history.append(val_loss)
        
        # Calculate progress through training
        progress = self.current_step / self.total_steps
        
        # Apply different scheduling strategies
        if self.schedule_type == "phased":
            self._apply_phased_schedule(progress)
        elif self.schedule_type == "adaptive":
            self._apply_adaptive_schedule()
        elif self.schedule_type == "curriculum":
            self._apply_curriculum_schedule(progress)
        else:
            # For other schedule types, rely on regularizer's internal scheduling
            pass
        
        # Save state to history
        self.schedule_history.append({
            'step': self.current_step,
            'penalties': self.regularizer.current_penalties.copy(),
            'training_loss': training_loss,
            'val_loss': val_loss
        })
    
    def _apply_phased_schedule(self, progress: float):
        """Apply phased regularization schedule based on training progress."""
        # Phase 1: Early training - focus on orthogonality and basic sparsity
        if progress < 0.25:
            self.regularizer.current_penalties['l1'] = 5e-6
            self.regularizer.current_penalties['orthogonal'] = 1e-5  # Higher orthogonality
            self.regularizer.current_penalties['kl'] = 2e-6
            self.regularizer.current_penalties['group_sparsity'] = 1e-6
            self.regularizer.current_penalties['hessian'] = 1e-7
        
        # Phase 2: Mid training - balance all regularizers
        elif progress < 0.6:
            self.regularizer.current_penalties['l1'] = 1e-5
            self.regularizer.current_penalties['orthogonal'] = 5e-6
            self.regularizer.current_penalties['kl'] = 5e-6
            self.regularizer.current_penalties['group_sparsity'] = 3e-6
            self.regularizer.current_penalties['hessian'] = 5e-7
        
        # Phase 3: Late training - focus on structure and pruning
        else:
            self.regularizer.current_penalties['l1'] = 1.5e-5
            self.regularizer.current_penalties['orthogonal'] = 3e-6
            self.regularizer.current_penalties['kl'] = 8e-6
            self.regularizer.current_penalties['group_sparsity'] = 5e-6
            self.regularizer.current_penalties['hessian'] = 1e-6
    
    def _apply_adaptive_schedule(self):
        """Apply adaptive regularization based on loss dynamics."""
        if len(self.loss_history) < 10 or len(self.val_loss_history) < 2:
            return  # Not enough history for adaptation
        
        # Detect overfitting: val_loss increasing while train_loss decreasing
        train_loss_trend = list(self.loss_history)[-10:] 
        is_train_decreasing = train_loss_trend[0] > train_loss_trend[-1]
        
        val_loss_trend = list(self.val_loss_history)
        is_val_increasing = val_loss_trend[0] < val_loss_trend[-1]
        
        # Detect underfitting: both losses stagnating at high values
        is_stagnating = abs(train_loss_trend[0] - train_loss_trend[-1]) / train_loss_trend[0] < 0.01
        
        # Adjust based on detected issues
        if is_train_decreasing and is_val_increasing:
            # Overfitting - increase regularization
            for penalty_type in self.regularizer.current_penalties:
                self.regularizer.current_penalties[penalty_type] *= 1.2
                
            logger.info(f"Adaptive schedule detected overfitting - increasing regularization")
        
        elif is_stagnating:
            # Underfitting - reduce regularization
            for penalty_type in self.regularizer.current_penalties:
                self.regularizer.current_penalties[penalty_type] *= 0.8
                
            logger.info(f"Adaptive schedule detected underfitting - decreasing regularization")
    
    def _apply_curriculum_schedule(self, progress: float):
        """Apply curriculum learning for regularization."""
        # Start with simple regularization and gradually increase complexity
        
        # Start with just L1 sparsity
        if progress < 0.2:
            self.regularizer.current_penalties['l1'] = 5e-6
            self.regularizer.current_penalties['orthogonal'] = 0.0
            self.regularizer.current_penalties['kl'] = 0.0
            self.regularizer.current_penalties['group_sparsity'] = 0.0
            self.regularizer.current_penalties['hessian'] = 0.0
        
        # Add orthogonality
        elif progress < 0.4:
            self.regularizer.current_penalties['l1'] = 5e-6
            self.regularizer.current_penalties['orthogonal'] = 5e-6
            self.regularizer.current_penalties['kl'] = 0.0
            self.regularizer.current_penalties['group_sparsity'] = 0.0 
            self.regularizer.current_penalties['hessian'] = 0.0
        
        # Add KL and group sparsity
        elif progress < 0.6:
            self.regularizer.current_penalties['l1'] = 5e-6
            self.regularizer.current_penalties['orthogonal'] = 5e-6
            self.regularizer.current_penalties['kl'] = 3e-6
            self.regularizer.current_penalties['group_sparsity'] = 2e-6
            self.regularizer.current_penalties['hessian'] = 0.0
        
        # Add everything
        else:
            self.regularizer.current_penalties['l1'] = 5e-6
            self.regularizer.current_penalties['orthogonal'] = 5e-6
            self.regularizer.current_penalties['kl'] = 3e-6
            self.regularizer.current_penalties['group_sparsity'] = 2e-6
            self.regularizer.current_penalties['hessian'] = 5e-7
            
            
def prune_model_based_on_regularization(
    model: torch.nn.Module,
    regularizer: LatentSpaceRegularizer,
    pruning_threshold: float = 0.05,
    importance_threshold: float = 0.1,
    gradual_steps: int = None
) -> Tuple[torch.nn.Module, Dict]:
    """
    Prune model based on feature importance determined by regularization,
    with uncertainty awareness to ensure robust pruning decisions.
    
    Args:
        model: The model to prune
        regularizer: Trained regularizer with statistics
        pruning_threshold: Sparsity threshold for pruning
        importance_threshold: Feature importance threshold
        gradual_steps: Number of steps for gradual pruning (if None, use config value)
        
    Returns:
        Tuple of (pruned_model, pruning_info)
    """
    # Get feature importance scores
    importance_scores = analyze_feature_importance(regularizer)
    
    # Get sparsity statistics and gradient history
    sparsity_stats = {
        layer_name: stats['sparsity'] 
        for layer_name, stats in regularizer.activation_stats.items()
    }
    
    # Set default for gradual pruning steps
    if gradual_steps is None:
        gradual_steps = regularizer.config.gradual_pruning_steps
    
    # Track pruning masks and info
    pruned_model = model
    pruning_info = {}
    pruning_masks = {}
    
    # For each layer with importance scores
    for layer_name, importance in importance_scores.items():
        # Check if layer has enough sparsity for pruning
        if layer_name not in sparsity_stats or sparsity_stats[layer_name] <= pruning_threshold:
            continue
            
        # Find the module
        module = None
        for name, mod in model.named_modules():
            if name == layer_name:
                module = mod
                break
        
        if module is None or not hasattr(module, 'weight'):
            continue
        
        # Calculate uncertainty-aware importance scores
        # Check if we have gradient history for uncertainty estimation
        uncertainty = torch.zeros_like(torch.tensor(importance))
        if (layer_name in regularizer.activation_stats and 
            'sensitivity_history' in regularizer.activation_stats[layer_name] and
            len(regularizer.activation_stats[layer_name]['sensitivity_history']) > 1):
            
            # Extract gradient history
            if hasattr(regularizer.activation_stats[layer_name], 'sensitivity_history'):
                history = list(regularizer.activation_stats[layer_name]['sensitivity_history'])
                if len(history) > 1:
                    # Compute variance of gradient sensitivity as uncertainty measure
                    uncertainty = torch.tensor(np.var(history))
            
        # Incorporate uncertainty: Importance_i(l) = E[|g_l,i|] · exp(-β · Var[|g_l,i|])
        uncertainty_weight = regularizer.config.prune_uncertainty_beta
        uncertainty_factor = torch.exp(-uncertainty_weight * uncertainty)
        
        # More uncertainty means lower pruning score
        robust_importance = torch.tensor(importance) * uncertainty_factor
        
        # Identify low-importance features
        low_importance_mask = robust_importance < importance_threshold
        
        # Skip if nothing to prune
        if not torch.any(low_importance_mask):
            continue
        
        # Create pruning mask for gradual pruning (1 = keep, 0 = prune)
        pruning_mask = torch.ones_like(torch.tensor(importance, dtype=torch.float32))
        pruning_mask[low_importance_mask] = 0.0
        pruning_masks[layer_name] = pruning_mask
        
        # Log pruning info
        pruning_info[layer_name] = {
            'total_features': len(importance),
            'pruned_features': int(torch.sum(low_importance_mask).item()),
            'pruning_ratio': float(torch.sum(low_importance_mask).item()) / len(importance),
            'uncertainty_mean': float(uncertainty.mean().item()) if isinstance(uncertainty, torch.Tensor) else float(uncertainty),
            'gradual_steps': gradual_steps
        }
        
        logger.info(f"Pruning {pruning_info[layer_name]['pruned_features']} features "
                  f"({pruning_info[layer_name]['pruning_ratio']:.2%}) from {layer_name}")
        
        # Apply gradual structured pruning by scaling weights
        # This starts the pruning process - complete pruning happens over gradual_steps
        with torch.no_grad():
            # Initial scaling (1/gradual_steps of the way to full pruning)
            scaling_factor = 1.0 - (1.0 / gradual_steps)
            
            # Handle different layer types
            if isinstance(module, nn.Linear):
                # Create scaling matrix that maintains weights for features to keep
                # and scales down weights for features to remove
                scaling = torch.ones_like(module.weight)
                for i, mask_val in enumerate(pruning_mask):
                    if mask_val == 0.0:  # Feature to be pruned
                        scaling[:, i] = scaling_factor
                
                # Apply scaling
                module.weight.data = module.weight.data * scaling
                
            elif isinstance(module, nn.Conv2d):
                # Create scaling for convolutional layers
                scaling = torch.ones_like(module.weight)
                for i, mask_val in enumerate(pruning_mask):
                    if i < module.weight.size(1) and mask_val == 0.0:  # Channel to be pruned
                        scaling[:, i, :, :] = scaling_factor
                
                # Apply scaling
                module.weight.data = module.weight.data * scaling
    
    logger.info(f"Initiated gradual pruning over {gradual_steps} steps based on latent space regularization")
    
    # Return the model with pruning started and the pruning info
    return pruned_model, pruning_info, pruning_masks


def latent_space_analysis_report(regularizer: LatentSpaceRegularizer, filepath: str = None):
    """
    Generate a comprehensive analysis report on latent space properties.
    
    Args:
        regularizer: Trained regularizer with history
        filepath: Optional path to save report
    """
    # Collect metrics
    metrics = regularizer.get_statistics()
    history = regularizer.metrics_history
    
    # Generate report
    report = ["# Latent Space Analysis Report", ""]
    
    # Overall statistics
    report.append("## Overall Statistics")
    report.append(f"- Total layers tracked: {metrics['num_layers_tracked']}")
    report.append(f"- Global step: {metrics['global_step']}")
    report.append(f"- Current scaling factor: {metrics['scaling_factor']:.4f}")
    report.append("")
    
    # Current penalties
    report.append("## Current Regularization Penalties")
    for penalty_type, value in metrics['penalties'].items():
        report.append(f"- {penalty_type}: {value:.2e}")
    report.append("")
    
    # Layer statistics
    report.append("## Layer-by-Layer Statistics")
    for layer_name, layer_stats in metrics['layers'].items():
        report.append(f"### {layer_name}")
        report.append(f"- Mean activation: {layer_stats['mean_activation']:.4f}")
        report.append(f"- Sparsity: {layer_stats['sparsity']:.4f}")
        report.append(f"- L1 norm: {layer_stats['l1_norm']:.4f}")
        report.append(f"- L2 norm: {layer_stats['l2_norm']:.4f}")
        report.append(f"- Entropy: {layer_stats['entropy']:.4f}")
        report.append(f"- Orthogonality: {layer_stats['orthogonality']:.4f}")
        report.append(f"- Spectral norm: {layer_stats['spectral_norm']:.4f}")
        report.append(f"- Gradient sensitivity: {layer_stats['gradient_sensitivity']:.4f}")
        report.append("")
    
    # Mutual information if available
    if 'avg_mutual_information' in metrics:
        report.append("## Information Theoretic Analysis")
        report.append(f"- Average mutual information: {metrics['avg_mutual_information']:.4f}")
        report.append("")
    
    # Training dynamics if history exists
    if history:
        report.append("## Training Dynamics")
        report.append("- Loss component evolution: See attached visualization")
        report.append("- Penalty adjustment: See attached visualization")
        report.append("")
    
    # Feature importance analysis
    report.append("## Feature Importance Analysis")
    importance_scores = analyze_feature_importance(regularizer)
    for layer_name, importance in importance_scores.items():
        report.append(f"### {layer_name}")
        report.append(f"- Total features: {len(importance)}")
        report.append(f"- Top features: {np.argsort(-importance)[:5].tolist()}")
        report.append(f"- Feature importance range: {importance.min():.4f} - {importance.max():.4f}")
        report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    
    # Detect potential issues
    avg_sparsity = metrics.get('avg_sparsity', 0)
    if avg_sparsity < 0.05:
        report.append("- **Low sparsity detected**: Consider increasing L1 and KL penalties")
    elif avg_sparsity > 0.5:
        report.append("- **Excessive sparsity detected**: Consider reducing L1 and KL penalties")
    
    avg_orthogonality = metrics.get('avg_orthogonality', 0)
    if avg_orthogonality > 0.3:
        report.append("- **High feature correlation detected**: Consider increasing orthogonality penalties")
    
    # Add general recommendations
    report.append("- For improved latent space quality, consider auto-tuning regularization parameters")
    report.append("- Use visualize_regularization_effects() to monitor training dynamics")
    report.append("- Consider model pruning based on feature importance for efficiency")
    
    # Save report
    if filepath:
        with open(filepath, 'w') as f:
            f.write('\n'.join(report))
        logger.info(f"Saved latent space analysis report to {filepath}")
    
    # Return report as string
    return '\n'.join(report)


def setup_advanced_latent_regularization(
    model: torch.nn.Module,
    loss_fn: Callable,
    total_steps: int = 10000,
    auto_tune: bool = False,
    target_layers: List[str] = None,
    schedule_type: str = "curriculum", 
    device = None
) -> Tuple[Callable, RegularizationScheduler]:
    """
    Complete setup for advanced latent space regularization with scheduling.
    
    Args:
        model: Model to regularize
        loss_fn: Base loss function
        total_steps: Total training steps
        auto_tune: Whether to auto-tune parameters
        target_layers: Specific layers to target
        schedule_type: Scheduling approach
        device: Computation device
        
    Returns:
        Tuple of (enhanced_loss_function, scheduler)
    """
    # Create config
    config = LatentSpaceRegConfig()
    
    # Set target layers if specified
    if target_layers:
        config.target_layers = target_layers
    
    # Create regularizer
    reg_loss_fn, regularizer = add_latent_regularization_to_loss(
        model, loss_fn, config, device
    )
    
    # Create scheduler
    scheduler = RegularizationScheduler(
        regularizer,
        total_steps=total_steps,
        schedule_type=schedule_type
    )
    
    # Wrap loss function with scheduler
    def scheduled_reg_loss_fn(outputs, targets, train_loss=None, val_loss=None):
        # Base regularized loss
        loss = reg_loss_fn(outputs, targets)
        
        # Update scheduler
        scheduler.step(
            training_loss=train_loss or loss.item(),
            val_loss=val_loss
        )
        
        return loss
    
    logger.info(f"Setup advanced latent regularization with {schedule_type} scheduling")
    return scheduled_reg_loss_fn, scheduler