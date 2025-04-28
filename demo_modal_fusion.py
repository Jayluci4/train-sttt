import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from advanced_modal_fusion import AdvancedModalFusion, ModalFusionConfig

def create_synthetic_data(batch_size=4, seq_len=20):
    """Create synthetic multimodal data for demonstration."""
    # Text features (simulating word embeddings)
    text_dim = 64
    text = torch.randn(batch_size, seq_len, text_dim)
    
    # Vision features (simulating image region features)
    vision_dim = 128
    vision = torch.randn(batch_size, seq_len, vision_dim)
    
    # Audio features (simulating audio frames)
    audio_dim = 96
    audio = torch.randn(batch_size, seq_len, audio_dim)
    
    # Create attention masks (1 = attend, 0 = mask)
    # In this demo, we'll simulate some missing modalities
    text_mask = torch.ones(batch_size, seq_len)
    vision_mask = torch.ones(batch_size, seq_len)
    audio_mask = torch.ones(batch_size, seq_len)
    
    # Randomly mask some positions
    vision_mask[:, 5:10] = 0  # Vision missing in positions 5-9
    audio_mask[:, 15:] = 0    # Audio missing in positions 15+
    
    inputs = {
        "text": text,
        "vision": vision,
        "audio": audio
    }
    
    masks = {
        "text": text_mask,
        "vision": vision_mask,
        "audio": audio_mask
    }
    
    modality_dims = {
        "text": text_dim,
        "vision": vision_dim,
        "audio": audio_dim
    }
    
    return inputs, masks, modality_dims

def visualize_fusion_weights(fusion_module, inputs, masks):
    """Visualize the weights assigned to each modality."""
    # Forward pass to compute fusion weights
    _ = fusion_module.forward(inputs, masks)
    stats = fusion_module.get_fusion_stats()
    
    if "modality_weights" not in stats:
        print("No modality weights available in fusion stats")
        return
    
    weights = stats["modality_weights"]
    
    # Convert to numpy for visualization
    weight_data = {}
    for modality, w in weights.items():
        if isinstance(w, torch.Tensor):
            weight_data[modality] = w.detach().cpu().numpy().mean(axis=0)
        else:
            weight_data[modality] = np.array(w)
    
    # Plot the weights
    plt.figure(figsize=(12, 5))
    for modality, w in weight_data.items():
        if len(w.shape) > 1:
            # If weights are per-position, take the average
            w = w.mean(axis=1)
        plt.plot(w, label=modality)
    
    plt.title("Modality Importance Weights")
    plt.xlabel("Sequence Position")
    plt.ylabel("Weight")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("modality_weights.png")
    plt.close()
    
    print(f"Modality weights visualization saved to modality_weights.png")

def visualize_mutual_information(fusion_module, inputs):
    """Visualize the mutual information between modalities."""
    # Forward pass to compute MI
    _, loss = fusion_module.forward(inputs, return_loss=True)
    stats = fusion_module.get_fusion_stats()
    
    if "mi_scores" not in stats:
        print("No mutual information scores available in fusion stats")
        return
    
    mi_scores = stats["mi_scores"]
    
    # Create pairwise MI matrix
    modalities = list(inputs.keys())
    mi_matrix = np.zeros((len(modalities), len(modalities)))
    
    for i, mod1 in enumerate(modalities):
        for j, mod2 in enumerate(modalities):
            key = f"{mod1}_{mod2}"
            if key in mi_scores:
                mi_value = mi_scores[key]
                if isinstance(mi_value, torch.Tensor):
                    mi_value = mi_value.item()
                mi_matrix[i, j] = mi_value
    
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        mi_matrix, 
        annot=True, 
        xticklabels=modalities, 
        yticklabels=modalities,
        cmap="YlGnBu"
    )
    plt.title("Mutual Information Between Modalities")
    plt.tight_layout()
    plt.savefig("mutual_information.png")
    plt.close()
    
    print(f"Mutual information visualization saved to mutual_information.png")
    
def demo_fusion_methods():
    """Demonstrate and compare different fusion methods."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create synthetic data
    inputs, masks, modality_dims = create_synthetic_data()
    
    # Move data to device
    for key in inputs:
        inputs[key] = inputs[key].to(device)
        masks[key] = masks[key].to(device)
    
    # Define dummy model (not used directly)
    dummy_model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU()
    )
    
    # Define fusion methods to test
    fusion_methods = ["adaptive_attention", "cross_modal", "gated", "weighted_sum"]
    
    results = {}
    
    for method in fusion_methods:
        print(f"\nTesting fusion method: {method}")
        
        # Create fusion configuration
        config = ModalFusionConfig(
            fusion_method=method,
            attention_heads=4,
            hidden_dim=128,
            dropout_rate=0.1,
            use_residual=True,
            activation_fn="gelu",
            normalize_features=True,
            dynamic_weighting=True,
            use_context_gating=True
        )
        
        # Create fusion module
        fusion_module = AdvancedModalFusion(
            config=config,
            model=dummy_model,
            modality_dims=modality_dims,
            device=device
        )
        
        # Forward pass with loss
        fused_features, loss = fusion_module.forward(inputs, masks, return_loss=True)
        
        # Store results
        results[method] = {
            "fused_features": fused_features,
            "loss": loss.item(),
            "fusion_module": fusion_module
        }
        
        print(f"  Loss: {loss.item():.4f}")
        
        # Visualize weights for this method
        visualize_fusion_weights(fusion_module, inputs, masks)
        
        # Visualize mutual information
        visualize_mutual_information(fusion_module, inputs)
    
    # Compare losses across methods
    plt.figure(figsize=(10, 5))
    methods = list(results.keys())
    losses = [results[m]["loss"] for m in methods]
    
    plt.bar(methods, losses)
    plt.title("Mutual Information Loss by Fusion Method")
    plt.ylabel("Loss Value (lower is better)")
    plt.grid(True, alpha=0.3)
    plt.savefig("fusion_method_comparison.png")
    plt.close()
    
    print("\nFusion method comparison saved to fusion_method_comparison.png")
    
if __name__ == "__main__":
    demo_fusion_methods() 