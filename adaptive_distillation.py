import torch
import torch.nn.functional as F
import logging
import math
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class AdaptiveDistillationController:
    """
    Implements adaptive distillation with confidence reweighting.
    
    This component adjusts distillation pressure based on teacher confidence:
    - When teacher is confident -> distill strongly
    - When teacher is uncertain -> back off to ground truth
    
    Mathematical formulation:
    L_hybrid = (1 - α_t) · ℓ(f_θ(x), y) + α_t · c(x) · KL(f_θ(x) || p_T(y|x))
    
    Where:
    - α_t is the distillation weight
    - c(x) is the confidence score: c(x) = 1 - H(p_T(y|x))/log(|V|)
    - H(·) is Shannon entropy
    - |V| is vocab/output space size
    """
    def __init__(
        self, 
        student_model,
        teacher_model=None,
        tokenizer=None,
        alpha: float = 0.5,  # Balance between distillation and ground truth
        temperature: float = 2.0,  # Softmax temperature for distillation
        min_confidence: float = 0.2,  # Minimum confidence to apply distillation
        confidence_threshold: float = 0.8,  # Threshold for high confidence
        adaptive_alpha: bool = True,  # Whether to adapt alpha based on training progress
        alpha_schedule: str = 'linear_decay'  # How to schedule alpha over time
    ):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.temperature = temperature
        self.min_confidence = min_confidence
        self.confidence_threshold = confidence_threshold
        self.adaptive_alpha = adaptive_alpha
        self.alpha_schedule = alpha_schedule
        
        # Distillation stats tracking
        self.confidence_history = []
        self.alpha_history = []
        self.total_steps = 0
        self.step_count = 0
        
        # Vocabulary size (for entropy normalization)
        self.vocab_size = self._get_vocab_size()
        
        logger.info(f"Initialized AdaptiveDistillationController with alpha={alpha}, temp={temperature}")
    
    def _get_vocab_size(self) -> int:
        """Get vocabulary size for entropy normalization."""
        if self.tokenizer is not None:
            return len(self.tokenizer)
        
        # Try to get from model config
        for model in [self.student_model, self.teacher_model]:
            if model is not None and hasattr(model, 'config'):
                if hasattr(model.config, 'vocab_size'):
                    return model.config.vocab_size
        
        # Default value
        return 32000  # Common for many tokenizers
    
    def setup(self, total_steps: int) -> None:
        """Set up the controller with total training steps information."""
        self.total_steps = total_steps
        logger.info(f"AdaptiveDistillationController set up with {total_steps} total steps")
    
    def compute_teacher_logits(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute teacher model logits and confidence scores.
        
        Args:
            inputs: Input tensors dictionary
            
        Returns:
            Tuple of (teacher_logits, confidence_scores)
        """
        if self.teacher_model is None:
            raise ValueError("Teacher model is not set")
        
        # Run teacher model
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
        
        # Compute confidence scores
        confidence_scores = self._compute_confidence_scores(teacher_logits)
        
        return teacher_logits, confidence_scores
    
    def _compute_confidence_scores(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence scores based on the entropy of output distribution.
        
        Confidence score:
            c(x) = 1 - H(p_T(y|x))/log(|V|)
        
        Args:
            logits: Logits tensor from teacher model
            
        Returns:
            Tensor of confidence scores in [0, 1]
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Compute entropy of distribution
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=-1)
        
        # Normalize by maximum possible entropy (log of vocab size)
        max_entropy = math.log(self.vocab_size)
        normalized_entropy = entropy / max_entropy
        
        # Confidence score is the complement of normalized entropy
        confidence = 1.0 - normalized_entropy
        
        return confidence
    
    def _get_current_alpha(self) -> float:
        """Get the current distillation weight based on schedule."""
        if not self.adaptive_alpha:
            return self.alpha
        
        # Check if total_steps is set
        if self.total_steps <= 0:
            return self.alpha
        
        # Calculate progress
        progress = min(1.0, self.step_count / self.total_steps)
        
        if self.alpha_schedule == 'constant':
            return self.alpha
        elif self.alpha_schedule == 'linear_decay':
            # Linearly decay alpha over time
            return self.alpha * (1.0 - progress)
        elif self.alpha_schedule == 'cosine_decay':
            # Cosine decay
            return self.alpha * 0.5 * (1.0 + math.cos(math.pi * progress))
        elif self.alpha_schedule == 'sigmoid_rampup':
            # Sigmoid ramp-up (slower at start and end)
            if progress < 0.5:
                # Ramp up
                return self.alpha * (2.0 * progress)**2
            else:
                # Ramp down
                return self.alpha * (1.0 - (2.0 * (progress - 0.5))**2)
        else:
            # Default to constant
            return self.alpha
    
    def distill_step(
    self, 
    inputs: Dict[str, torch.Tensor],
    labels: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Perform a distillation step to compute the hybrid loss with improved numerical stability.
        
        Args:
            inputs: Input tensors dictionary
            labels: Optional ground truth labels
            
        Returns:
            Tuple of (hybrid_loss, distillation_info)
        """
        self.step_count += 1
        current_alpha = self._get_current_alpha()
        self.alpha_history.append(current_alpha)
        
        # If alpha is 0, just use student loss
        if current_alpha == 0:
            student_outputs = self.student_model(**inputs)
            return student_outputs.loss, {'alpha': 0, 'type': 'student_only'}
        
        # If teacher model is not available, return student loss
        if self.teacher_model is None:
            student_outputs = self.student_model(**inputs)
            return student_outputs.loss, {'alpha': 0, 'type': 'no_teacher'}
        
        # Get teacher logits and confidence
        teacher_logits, confidence_scores = self.compute_teacher_logits(inputs)
        
        # Record confidence
        batch_confidence = confidence_scores.mean().item()
        self.confidence_history.append(batch_confidence)
        
        # Run student model
        student_outputs = self.student_model(**inputs)
        student_logits = student_outputs.logits
        
        # Get student loss (if labels provided)
        if 'labels' in inputs or labels is not None:
            student_loss = student_outputs.loss
        else:
            # No supervision loss available
            student_loss = torch.tensor(0.0, device=student_logits.device)
        
        # Computing distillation loss with improved numerical stability
        # Scale logits by temperature
        teacher_logits_temp = teacher_logits / self.temperature
        student_logits_temp = student_logits / self.temperature
        
        # Convert to probabilities - use log_softmax for improved stability
        log_student_probs = F.log_softmax(student_logits_temp, dim=-1)
        teacher_probs = F.softmax(teacher_logits_temp, dim=-1)
        
        # Apply label smoothing to teacher probs to avoid overconfidence
        smoothing = 1e-4
        vocab_size = teacher_probs.size(-1)
        teacher_probs = (1 - smoothing) * teacher_probs + smoothing / vocab_size
        
        # Compute KL divergence with safe reduction
        # First compute per-token KL div
        per_token_kl = teacher_probs * (torch.log(teacher_probs + 1e-10) - log_student_probs)
        
        # Clamp values for stability and sum across vocab dimension
        per_token_kl = torch.clamp(per_token_kl, -100, 100)  # Prevent extreme values
        distillation_loss = per_token_kl.sum(dim=-1)
        
        # Scale distillation loss by temperature²
        distillation_loss = distillation_loss * (self.temperature ** 2)
        
        # Apply confidence weighting with smooth transition
        # Use sigmoid scaling for smoother confidence weighting
        confidence_weight = torch.sigmoid(
            5.0 * (confidence_scores - self.min_confidence) / (1.0 - self.min_confidence)
        )
        
        weighted_distillation_loss = distillation_loss * confidence_weight
        
        # Compute hybrid loss
        # L_hybrid = (1 - α_t) · ℓ(f_θ(x), y) + α_t · c(x) · KL(f_θ(x) || p_T(y|x))
        student_loss_weight = 1.0 - current_alpha
        distillation_loss_weight = current_alpha
        
        hybrid_loss = (
            student_loss_weight * student_loss + 
            distillation_loss_weight * weighted_distillation_loss.mean()
        )
        
        # Prepare distillation info
        distillation_info = {
            'alpha': current_alpha,
            'confidence': batch_confidence,
            'student_loss': student_loss.item(),
            'distillation_loss': distillation_loss.mean().item(),
            'weighted_distillation_loss': weighted_distillation_loss.mean().item(),
            'hybrid_loss': hybrid_loss.item(),
            'type': 'hybrid'
        }
        
        return hybrid_loss, distillation_info
    
    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics."""
        stats = {
            'current_alpha': self._get_current_alpha(),
            'step_count': self.step_count
        }
        
        # Confidence statistics
        if self.confidence_history:
            stats.update({
                'mean_confidence': np.mean(self.confidence_history),
                'min_confidence': np.min(self.confidence_history),
                'max_confidence': np.max(self.confidence_history),
                'recent_confidence': np.mean(self.confidence_history[-min(10, len(self.confidence_history)):])
            })
        
        # Alpha statistics
        if self.alpha_history:
            stats.update({
                'alpha_history': self.alpha_history[-min(10, len(self.alpha_history)):],
                'mean_alpha': np.mean(self.alpha_history)
            })
        
        return stats
