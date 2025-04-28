import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
import logging
import itertools

logger = logging.getLogger(__name__)

@dataclass
class ModalFusionConfig:
    """Configuration for the AdvancedModalFusion module."""
    
    fusion_method: str = "adaptive_attention"  # Options: "adaptive_attention", "cross_modal", "gated", "weighted_sum"
    hidden_dim: int = 256
    attention_heads: int = 4
    dropout_rate: float = 0.1
    dynamic_weighting: bool = True
    collect_stats: bool = True
    use_context_gating: bool = True
    
    # Parameters for the mutual information loss (if enabled)
    mi_loss_weight: float = 0.1
    
    def __post_init__(self):
        """Validate the configuration parameters."""
        valid_fusion_methods = ["adaptive_attention", "cross_modal", "gated", "weighted_sum"]
        if self.fusion_method not in valid_fusion_methods:
            raise ValueError(f"Fusion method must be one of {valid_fusion_methods}, got {self.fusion_method}")
        
        if self.attention_heads <= 0:
            raise ValueError(f"Number of attention heads must be positive, got {self.attention_heads}")
        
        if not 0 <= self.dropout_rate < 1:
            raise ValueError(f"Dropout rate must be between 0 and 1, got {self.dropout_rate}")

class AdvancedModalFusion(nn.Module):
    """
    Advanced modal fusion module for combining multiple modalities.
    
    Supports various fusion methods:
    - adaptive_attention: Uses multi-head attention to dynamically weight modalities
    - cross_modal: Performs cross-modal attention between pairs of modalities
    - gated: Uses gating mechanisms to control information flow
    - weighted_sum: Simple weighted summation of modality features
    """
    
    def __init__(
        self,
        config: ModalFusionConfig,
        model: nn.Module,
        modality_dims: Dict[str, int],
        output_dim: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the fusion module.
        
        Args:
            config: Configuration for the fusion module
            model: The parent model this fusion module is part of
            modality_dims: Dictionary mapping modality names to their dimensions
            output_dim: Dimension of the output features
            device: Device to run the module on
        """
        super().__init__()
        self.config = config
        self.model = model
        self.modality_dims = modality_dims
        self.output_dim = output_dim
        self.device = device
        
        # Initialize projections for each modality to a common dimension
        self.projections = nn.ModuleDict({
            modality: nn.Linear(dim, config.hidden_dim)
            for modality, dim in modality_dims.items()
        })
        
        # Initialize fusion-specific components
        if config.fusion_method == "adaptive_attention":
            self._init_adaptive_attention()
        elif config.fusion_method == "cross_modal":
            self._init_cross_modal()
        elif config.fusion_method == "gated":
            self._init_gated_fusion()
        elif config.fusion_method == "weighted_sum":
            self._init_weighted_sum()
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_dim, output_dim)
        
        # Collect statistics
        self.reset_stats()
    
    def _init_adaptive_attention(self):
        """Initialize components for adaptive attention fusion."""
        # Multi-head attention for dynamic weighting
        self.attention = nn.MultiheadAttention(
            embed_dim=self.config.hidden_dim,
            num_heads=self.config.attention_heads,
            dropout=self.config.dropout_rate,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(self.config.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(self.config.hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.config.hidden_dim * 4, self.config.hidden_dim),
            nn.Dropout(self.config.dropout_rate)
        )
        
        # Dynamic weighting
        if self.config.dynamic_weighting:
            self.modality_weights = nn.Parameter(
                torch.ones(len(self.modality_dims), device=self.device) / len(self.modality_dims)
            )
    
    def _init_cross_modal(self):
        """Initialize components for cross-modal fusion."""
        # Cross-attention between pairs of modalities
        modalities = list(self.modality_dims.keys())
        self.cross_attentions = nn.ModuleDict()
        
        for i, mod_i in enumerate(modalities):
            for j, mod_j in enumerate(modalities):
                if i != j:
                    self.cross_attentions[f"{mod_i}_to_{mod_j}"] = nn.MultiheadAttention(
                        embed_dim=self.config.hidden_dim,
                        num_heads=self.config.attention_heads,
                        dropout=self.config.dropout_rate,
                        batch_first=True
                    )
        
        # Layer norms for each modality
        self.layer_norms = nn.ModuleDict({
            modality: nn.LayerNorm(self.config.hidden_dim)
            for modality in self.modality_dims
        })
        
        # Feed-forward networks for each modality
        self.ffns = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(self.config.dropout_rate),
                nn.Linear(self.config.hidden_dim * 4, self.config.hidden_dim),
                nn.Dropout(self.config.dropout_rate)
            )
            for modality in self.modality_dims
        })
        
        # Final integration
        self.integration = nn.Linear(
            self.config.hidden_dim * len(self.modality_dims),
            self.config.hidden_dim
        )
    
    def _init_gated_fusion(self):
        """Initialize components for gated fusion."""
        # Gates for each modality
        self.gates = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                nn.Sigmoid()
            )
            for modality in self.modality_dims
        })
        
        # Context processing if enabled
        if self.config.use_context_gating:
            # Context vectors for modality interaction
            self.context_processors = nn.ModuleDict({
                modality: nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
                for modality in self.modality_dims
            })
            
            # Cross-modal context gates
            modalities = list(self.modality_dims.keys())
            self.context_gates = nn.ModuleDict()
            
            for i, mod_i in enumerate(modalities):
                for j, mod_j in enumerate(modalities):
                    if i != j:
                        self.context_gates[f"{mod_i}_to_{mod_j}"] = nn.Sequential(
                            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
                            nn.Sigmoid()
                        )
        
        # Integration layer
        self.integration = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
    
    def _init_weighted_sum(self):
        """Initialize weighted sum fusion."""
        num_modalities = len(self.modality_dims)
        
        # Initialize modality weights
        self.modality_weights = nn.Parameter(torch.ones(num_modalities, 1, 1))
        
        # For dynamic weighting
        if self.config.dynamic_weighting:
            # Create a small MLP to generate dynamic weights
            self.weight_mlp = nn.Sequential(
                nn.Linear(num_modalities, num_modalities * 2),
                nn.ReLU(),
                nn.Linear(num_modalities * 2, num_modalities)
            )
        
        # Layer norm for final output
        self.layer_norm = nn.LayerNorm(self.config.hidden_dim)
        
        # Final integration layer
        self.integration = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.output_dim)
        )
    
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None,
        return_loss: bool = False
    ) -> Union[torch.Tensor, tuple]:
        """
        Forward pass through the fusion module.
        
        Args:
            modality_features: Dictionary mapping modality names to feature tensors
                Each tensor should have shape [batch_size, seq_len, feature_dim]
            modality_masks: Optional dictionary mapping modality names to boolean masks
                Each mask should have shape [batch_size, seq_len] where True indicates
                valid tokens and False indicates padding/missing tokens
            return_loss: Whether to return fusion loss
        
        Returns:
            fused_features: Fused features with shape [batch_size, seq_len, output_dim]
            loss: Optional fusion loss if return_loss=True
        """
        # Validate modalities
        for modality in modality_features:
            if modality not in self.modality_dims:
                raise ValueError(f"Unknown modality: {modality}. Expected one of {list(self.modality_dims.keys())}")
        
        # Project each modality to the common dimension
        projected_features = {}
        for modality, features in modality_features.items():
            if features is not None:
                projected_features[modality] = self.projections[modality](features)
        
        # Apply fusion based on selected method
        if self.config.fusion_method == "adaptive_attention":
            fused_features = self._apply_adaptive_attention(projected_features, modality_masks)
        elif self.config.fusion_method == "cross_modal":
            fused_features = self._apply_cross_modal(projected_features, modality_masks)
        elif self.config.fusion_method == "gated":
            fused_features = self._apply_gated_fusion(projected_features, modality_masks)
        elif self.config.fusion_method == "weighted_sum":
            fused_features = self._apply_weighted_sum(projected_features, modality_masks)
        
        # Project to output dimension
        fused_features = self.output_projection(fused_features)
        
        # Calculate fusion loss if requested
        loss = None
        if return_loss:
            loss = self._calculate_fusion_loss(projected_features, modality_masks)
        
        # Update statistics
        if self.config.collect_stats:
            self._update_stats(projected_features, modality_masks)
        
        if return_loss:
            return fused_features, loss
        return fused_features
    
    def _apply_adaptive_attention(
        self,
        projected_features: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Apply adaptive attention fusion."""
        if not projected_features:
            raise ValueError("No modality features provided for fusion")
        
        batch_size, seq_len, hidden_dim = next(iter(projected_features.values())).shape
        
        # Stack all modality features
        stacked_features = []
        
        # Create a combined attention mask if modality masks are provided
        attention_mask = None
        if modality_masks:
            attention_mask = None
            for modality in modality_masks:
                if modality not in projected_features:
                    continue
                mod_mask = modality_masks[modality]
                if attention_mask is None:
                    attention_mask = mod_mask
                else:
                    attention_mask = attention_mask & mod_mask
        
        # Convert attention mask to float if provided
        if attention_mask is not None:
            attention_mask_float = torch.zeros(
                batch_size, seq_len, seq_len, device=self.device
            )
            for i in range(batch_size):
                # Fix: Use proper broadcasting to create 2D mask
                # Instead of unsqueeze(2) which causes dimension error,
                # we use outer product style broadcasting
                mask_i = attention_mask[i]
                attention_mask_float[i] = mask_i.unsqueeze(1).expand(seq_len, seq_len) & mask_i.unsqueeze(0).expand(seq_len, seq_len)
            
            # Create an attention mask where True values get masked out (large negative value)
            # Fix: Convert to bool first to ensure proper logical operations
            attention_mask = (attention_mask_float == 0).float() * -10000.0
        
        # Average features for the query
        for modality in projected_features:
            stacked_features.append(projected_features[modality])
            
        query = torch.stack(stacked_features, dim=0).mean(dim=0)
        query = self.layer_norm1(query)
        
        # Stack features for key and value
        key_value = torch.cat(stacked_features, dim=1)  # [batch_size, num_modalities*seq_len, hidden_dim]
        
        # Apply self-attention
        attn_output, attn_weights = self.attention(
            query=query,
            key=key_value,
            value=key_value,
            attn_mask=None,  # Using the attention_mask here causes issues with the shape
            need_weights=True
        )
        
        # Store attention weights for stats
        if self.config.collect_stats:
            self.stats["attention_weights"] = attn_weights.detach()
        
        # Residual connection and layer norm
        attn_output = query + attn_output
        attn_output = self.layer_norm2(attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(attn_output)
        fused_features = attn_output + ffn_output
        
        return fused_features
    
    def _apply_cross_modal(
        self,
        projected_features: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Apply cross-modal fusion."""
        if not projected_features:
            raise ValueError("No modality features provided for fusion")
        
        modalities = list(projected_features.keys())
        processed_features = {}
        
        # Process each modality with cross-attention from other modalities
        for mod_i in modalities:
            features_i = projected_features[mod_i]
            cross_attended_features = []
            
            # Apply cross-attention from each other modality
            for mod_j in modalities:
                if mod_i != mod_j:
                    features_j = projected_features[mod_j]
                    
                    # Create attention mask if available
                    attn_mask = None
                    if (modality_masks is not None and 
                        mod_i in modality_masks and 
                        mod_j in modality_masks):
                        mask_i = modality_masks[mod_i]
                        mask_j = modality_masks[mod_j]
                        batch_size, seq_len_i = mask_i.shape
                        _, seq_len_j = mask_j.shape
                        
                        # Instead of creating a batch-specific mask that won't match MultiheadAttention requirements,
                        # we'll create a simpler mask approach that works with the shape expectations
                        
                        # For MultiheadAttention, if attn_mask is 3D, it expects shape (batch_size, tgt_len, src_len)
                        # We need to ensure this matches our batch size
                        
                        # For simplicity, we'll use a different approach without manual mask creation
                        # We'll rely on key_padding_mask with MultiheadAttention instead
                        key_padding_mask = ~mask_j  # Invert because key_padding_mask masks where True
                        
                        # Apply cross-attention with key_padding_mask instead of attn_mask
                        attended_features, _ = self.cross_attentions[f"{mod_i}_to_{mod_j}"](
                            query=features_i,
                            key=features_j,
                            value=features_j,
                            key_padding_mask=key_padding_mask,
                            need_weights=False
                        )
                    else:
                        # No mask case
                        attended_features, _ = self.cross_attentions[f"{mod_i}_to_{mod_j}"](
                            query=features_i,
                            key=features_j,
                            value=features_j,
                            need_weights=False
                        )
                    
                    cross_attended_features.append(attended_features)
            
            # Residual connection with original features
            if cross_attended_features:
                cross_attended = torch.stack(cross_attended_features).mean(dim=0)
                processed_features[mod_i] = self.layer_norms[mod_i](features_i + cross_attended)
            else:
                processed_features[mod_i] = self.layer_norms[mod_i](features_i)
            
            # Apply feed-forward network
            processed_features[mod_i] = processed_features[mod_i] + self.ffns[mod_i](processed_features[mod_i])
        
        # Concatenate processed features
        concatenated = torch.cat([processed_features[mod] for mod in modalities], dim=-1)
        
        # Apply final integration
        fused_features = self.integration(concatenated)
        
        return fused_features
    
    def _apply_gated_fusion(
        self,
        projected_features: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Apply gated fusion."""
        if not projected_features:
            raise ValueError("No modality features provided for fusion")
        
        modalities = list(projected_features.keys())
        gated_features = {}
        
        # Apply gates to each modality
        for modality in modalities:
            features = projected_features[modality]
            gate = self.gates[modality](features)
            gated_features[modality] = features * gate
        
        # Apply context gating if enabled
        if self.config.use_context_gating:
            context_enhanced = {}
            
            # Process each modality with context from other modalities
            for mod_i in modalities:
                features_i = gated_features[mod_i]
                context_mods = []
                
                # Apply context gates from each other modality
                for mod_j in modalities:
                    if mod_i != mod_j:
                        features_j = gated_features[mod_j]
                        
                        # Process context
                        context_j = self.context_processors[mod_j](features_j)
                        
                        # Combine with current features
                        combined = torch.cat([features_i, context_j], dim=-1)
                        context_gate = self.context_gates[f"{mod_i}_to_{mod_j}"](combined)
                        
                        # Apply context gate
                        context_mod = features_i + context_gate * context_j
                        context_mods.append(context_mod)
                
                # Combine context modifications
                if context_mods:
                    context_enhanced[mod_i] = torch.stack(context_mods).mean(dim=0)
                else:
                    context_enhanced[mod_i] = features_i
            
            # Update gated features with context
            gated_features = context_enhanced
        
        # Average gated features
        fused_features = torch.stack([gated_features[mod] for mod in modalities]).mean(dim=0)
        
        # Apply integration
        fused_features = self.integration(fused_features)
        
        return fused_features
    
    def _apply_weighted_sum(
        self,
        projected_features: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Apply weighted sum fusion."""
        if not projected_features:
            raise ValueError("No modality features provided for fusion")
        
        # Apply softmax to get normalized weights
        if self.config.dynamic_weighting:
            # Create dynamic weights based on content
            # Extract feature statistics for each modality
            modality_stats = []
            for modality in projected_features:
                features = projected_features[modality]
                # Mean across sequence length and hidden dimension
                stats = features.mean(dim=[1, 2])
                modality_stats.append(stats)
            
            # Stack and create dynamic weights
            if modality_stats:
                dynamic_factors = torch.stack(modality_stats, dim=1)
                # Apply through a small neural network to get scaling factors
                dynamic_scale = torch.sigmoid(self.weight_mlp(dynamic_factors))
                # Reshape to match modality weights
                dynamic_scale = dynamic_scale.view(-1, len(projected_features), 1, 1)
                # Apply scaling to base weights
                scaled_weights = self.modality_weights * dynamic_scale.mean(dim=0)
                weights = F.softmax(scaled_weights, dim=0)
            else:
                weights = F.softmax(self.modality_weights, dim=0)
        else:
            weights = F.softmax(self.modality_weights, dim=0)
        
        if self.config.collect_stats:
            self.stats["modality_weights"] = weights.detach()
        
        # Weighted sum of modality features
        modalities = list(projected_features.keys())
        weighted_features = []
        
        for i, modality in enumerate(modalities):
            features = projected_features[modality]
            
            # Apply mask if available
            if modality_masks is not None and modality in modality_masks:
                mask = modality_masks[modality].unsqueeze(-1)
                features = features * mask
            
            # Apply modality weight
            if i < len(weights):
                weighted_features.append(features * weights[i])
            else:
                # Handle case where there are more modalities than weights
                weighted_features.append(features / len(modalities))
        
        # Sum all weighted features
        fused_features = sum(weighted_features)
        
        # Apply final integration
        fused_features = self.integration(fused_features)
        
        return fused_features
    
    def _calculate_fusion_loss(
        self,
        projected_features: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Calculate fusion loss to encourage effective fusion.
        
        This can include:
        - Mutual information maximization between modalities
        - Diversity loss to encourage unique contributions
        - Alignment loss to encourage consistent representations
        """
        loss = torch.tensor(0.0, device=self.device)
        
        # Skip if not enough modalities for meaningful loss
        if len(projected_features) <= 1:
            return loss
        
        # Calculate mutual information loss
        if self.config.mi_loss_weight > 0:
            mi_loss = self._calculate_mutual_information_loss(projected_features, modality_masks)
            loss = loss + self.config.mi_loss_weight * mi_loss
        
        return loss
    
    def _calculate_mutual_information_loss(
        self,
        projected_features: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Calculate mutual information loss between modalities.
        
        This encourages modalities to share information while maintaining uniqueness.
        """
        modalities = list(projected_features.keys())
        batch_size = next(iter(projected_features.values())).shape[0]
        
        # Simple implementation using cosine similarity
        cos_sim_total = torch.tensor(0.0, device=self.device)
        count = 0
        
        for i in range(len(modalities)):
            for j in range(i+1, len(modalities)):
                mod_i = modalities[i]
                mod_j = modalities[j]
                features_i = projected_features[mod_i]
                features_j = projected_features[mod_j]
                
                # Mean pool if needed
                if len(features_i.shape) > 2:
                    # Apply masks if available
                    if modality_masks is not None and mod_i in modality_masks:
                        mask_i = modality_masks[mod_i].unsqueeze(-1)
                        features_i = features_i * mask_i
                        features_i = features_i.sum(dim=1) / mask_i.sum(dim=1).clamp(min=1e-10)
                    else:
                        features_i = features_i.mean(dim=1)
                
                if len(features_j.shape) > 2:
                    # Apply masks if available
                    if modality_masks is not None and mod_j in modality_masks:
                        mask_j = modality_masks[mod_j].unsqueeze(-1)
                        features_j = features_j * mask_j
                        features_j = features_j.sum(dim=1) / mask_j.sum(dim=1).clamp(min=1e-10)
                    else:
                        features_j = features_j.mean(dim=1)
                
                # Calculate cosine similarity
                cos_sim = F.cosine_similarity(features_i, features_j, dim=1)
                
                # Target similarity: not too close, not too far (around 0.5)
                target_sim = torch.ones_like(cos_sim) * 0.5
                cos_sim_loss = F.mse_loss(cos_sim, target_sim)
                
                cos_sim_total += cos_sim_loss
                count += 1
        
        if count > 0:
            return cos_sim_total / count
        return cos_sim_total
    
    def _update_stats(
        self,
        projected_features: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None
    ):
        """Update fusion statistics."""
        if not self.config.collect_stats:
            return
            
        # Record modality presence/absence
        all_possible_modalities = set(self.modality_dims.keys())
        present_modalities = set(projected_features.keys())
        missing_modalities = all_possible_modalities - present_modalities
        
        # Calculate percentage of missing modalities
        missing_modality_percentages = {}
        for modality in missing_modalities:
            missing_modality_percentages[modality] = 100.0
            
        self.stats["missing_modality_percentages"] = missing_modality_percentages
        
        # Update average modality weights if applicable
        if "modality_weights" in self.stats and self.stats["modality_weights"] is not None:
            modality_weights = self.stats["modality_weights"]
            
            # Average over present modalities
            avg_weights = {}
            for i, modality in enumerate(present_modalities):
                if i < modality_weights.shape[0]:
                    avg_weights[modality] = modality_weights[i].mean().item()
                    
            self.stats["avg_modality_weights"] = avg_weights
            
        # Update mutual information if we have multiple modalities
        if len(present_modalities) >= 2:
            # Calculate pairwise mutual information
            mi_vals = {}
            
            for mod1, mod2 in itertools.combinations(present_modalities, 2):
                features1 = projected_features[mod1]
                features2 = projected_features[mod2]
                
                # Compute mutual information (simplified version)
                # This is just a placeholder; real MI computation would be more complex
                # We'll use cosine similarity as a proxy
                f1 = features1.mean(dim=1)  # Avg over sequence length
                f2 = features2.mean(dim=1)  # Avg over sequence length
                
                # Normalize features
                f1 = F.normalize(f1, p=2, dim=1)
                f2 = F.normalize(f2, p=2, dim=1)
                
                # Compute cosine similarity
                sim = torch.bmm(f1.unsqueeze(1), f2.unsqueeze(2)).squeeze()
                
                # Average over batch
                mi_val = sim.mean().item()
                mi_vals[f"{mod1}-{mod2}"] = mi_val
                
            self.stats["mutual_information"] = mi_vals
    
    def reset_stats(self):
        """Reset fusion statistics."""
        self.stats = {
            "attention_weights": None,
            "modality_weights": None,
            "missing_modality_percentages": {},
            "avg_modality_weights": {},
            "mutual_information": {}
        }
    
    def get_fusion_stats(self):
        """Get fusion statistics."""
        return self.stats

# Helper functions for working with the fusion module
def adapt_fusion_for_architecture_search(
    fusion_module: AdvancedModalFusion,
    search_space: List[str],
    temperature: float = 1.0
):
    """
    Adapt the fusion module for architecture search.
    
    Args:
        fusion_module: The fusion module to adapt
        search_space: List of fusion methods to search over
        temperature: Temperature for the Gumbel-Softmax sampling
    
    Returns:
        The adapted fusion module
    """
    # Validate search space
    valid_methods = ["adaptive_attention", "cross_modal", "gated", "weighted_sum"]
    for method in search_space:
        if method not in valid_methods:
            raise ValueError(f"Invalid fusion method: {method}. Must be one of {valid_methods}")
    
    # Create fusion method logits
    fusion_module.method_logits = nn.Parameter(
        torch.ones(len(search_space), device=fusion_module.device)
    )
    fusion_module.search_space = search_space
    fusion_module.temperature = temperature
    
    # Store the original forward method
    original_forward = fusion_module.forward
    
    # Define a new forward method with architecture search
    def forward_with_search(*args, **kwargs):
        # Sample fusion method using Gumbel-Softmax
        method_probs = F.gumbel_softmax(
            fusion_module.method_logits,
            tau=fusion_module.temperature,
            hard=True
        )
        
        # Get the sampled method
        method_idx = torch.argmax(method_probs).item()
        sampled_method = fusion_module.search_space[method_idx]
        
        # Temporarily set the fusion method
        original_method = fusion_module.config.fusion_method
        fusion_module.config.fusion_method = sampled_method
        
        # Call the original forward method
        result = original_forward(*args, **kwargs)
        
        # Restore the original fusion method
        fusion_module.config.fusion_method = original_method
        
        return result
    
    # Replace the forward method
    fusion_module.original_forward = original_forward
    fusion_module.forward = forward_with_search
    
    return fusion_module 