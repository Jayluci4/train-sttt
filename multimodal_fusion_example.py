import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from advanced_modal_fusion import AdvancedModalFusion, ModalFusionConfig

# Set random seed for reproducibility
torch.manual_seed(42)

class SimpleTextEncoder(nn.Module):
    """Simple encoder for text modality."""
    def __init__(self, vocab_size=1000, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x)
        output, _ = self.encoder(embedded)
        return self.output_proj(output)


class SimpleImageEncoder(nn.Module):
    """Simple encoder for image modality."""
    def __init__(self, input_channels=3, hidden_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten(start_dim=1, end_dim=2)  # Preserve sequence dim
        self.output_proj = nn.Linear(128 * 4, hidden_dim)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, channels, height, width]
        batch_size, seq_len = x.shape[0], x.shape[1]
        x = x.view(batch_size * seq_len, x.shape[2], x.shape[3], x.shape[4])
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = self.flatten(x.view(batch_size, seq_len, x.shape[1], -1))
        return self.output_proj(x)


class SimpleAudioEncoder(nn.Module):
    """Simple encoder for audio modality."""
    def __init__(self, input_dim=40, hidden_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.output_proj = nn.Linear(128, hidden_dim)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, audio_features, audio_length]
        batch_size, seq_len = x.shape[0], x.shape[1]
        x = x.view(batch_size * seq_len, x.shape[2], x.shape[3])
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # Average over time dimension
        x = x.mean(dim=2)
        x = x.view(batch_size, seq_len, -1)
        return self.output_proj(x)


class SimpleMultimodalModel(nn.Module):
    """
    Simple multimodal model using the AdvancedModalFusion module.
    
    This model encodes text, image, and audio modalities separately,
    then fuses them using the AdvancedModalFusion module.
    """
    def __init__(
        self,
        vocab_size=1000,
        embed_dim=128,
        hidden_dim=256,
        output_dim=128,
        fusion_method="adaptive_attention"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Modality encoders
        self.text_encoder = SimpleTextEncoder(vocab_size, embed_dim, hidden_dim)
        self.image_encoder = SimpleImageEncoder(input_channels=3, hidden_dim=hidden_dim)
        self.audio_encoder = SimpleAudioEncoder(input_dim=40, hidden_dim=hidden_dim)
        
        # Modality dimensions
        modality_dims = {
            "text": hidden_dim,
            "image": hidden_dim,
            "audio": hidden_dim
        }
        
        # Fusion configuration
        fusion_config = ModalFusionConfig(
            fusion_method=fusion_method,
            hidden_dim=hidden_dim,
            attention_heads=4,
            dropout_rate=0.1,
            dynamic_weighting=True,
            collect_stats=True,
            use_context_gating=(fusion_method == "gated"),
            mi_loss_weight=0.1
        )
        
        # Fusion module
        self.fusion_module = AdvancedModalFusion(
            config=fusion_config,
            model=self,
            modality_dims=modality_dims,
            output_dim=hidden_dim
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, batch, return_loss=False):
        """
        Forward pass through the model.
        
        Args:
            batch: Dictionary containing input modalities:
                - "text": Text input of shape [batch_size, seq_len]
                - "image": Image input of shape [batch_size, seq_len, channels, height, width]
                - "audio": Audio input of shape [batch_size, seq_len, audio_features, audio_length]
            return_loss: Whether to return fusion loss
        
        Returns:
            output: Model output of shape [batch_size, seq_len, output_dim]
            loss: Optional fusion loss if return_loss=True
        """
        # Process each modality
        modality_features = {}
        modality_masks = {}
        
        if "text" in batch and batch["text"] is not None:
            text_input = batch["text"]
            modality_features["text"] = self.text_encoder(text_input)
            # Create mask for padding (assuming 0 is pad token)
            modality_masks["text"] = (text_input != 0)
        
        if "image" in batch and batch["image"] is not None:
            image_input = batch["image"]
            modality_features["image"] = self.image_encoder(image_input)
            # All image tokens are valid
            modality_masks["image"] = torch.ones(
                image_input.shape[0], image_input.shape[1], dtype=torch.bool,
                device=image_input.device
            )
        
        if "audio" in batch and batch["audio"] is not None:
            audio_input = batch["audio"]
            modality_features["audio"] = self.audio_encoder(audio_input)
            # All audio tokens are valid
            modality_masks["audio"] = torch.ones(
                audio_input.shape[0], audio_input.shape[1], dtype=torch.bool,
                device=audio_input.device
            )
        
        # Apply fusion
        if return_loss:
            fused_features, fusion_loss = self.fusion_module(
                modality_features, modality_masks, return_loss=True
            )
            # Apply output projection
            output = self.output_proj(fused_features)
            return output, fusion_loss
        else:
            fused_features = self.fusion_module(modality_features, modality_masks)
            # Apply output projection
            output = self.output_proj(fused_features)
            return output


def generate_synthetic_data(batch_size=32, seq_len=10, device="cuda"):
    """Generate synthetic data for multimodal model."""
    # Text data: integers representing tokens
    text = torch.randint(1, 1000, (batch_size, seq_len), device=device)
    # Add some padding
    pad_mask = torch.rand(batch_size, seq_len, device=device) > 0.8
    text = text.masked_fill(pad_mask, 0)
    
    # Image data: random tensors representing images
    image = torch.rand(batch_size, seq_len, 3, 32, 32, device=device)
    
    # Audio data: random tensors representing audio spectrograms
    audio = torch.rand(batch_size, seq_len, 40, 64, device=device)
    
    # Target: random labels
    target = torch.randint(0, 10, (batch_size, seq_len), device=device)
    
    return {
        "text": text,
        "image": image,
        "audio": audio,
        "target": target
    }


def train_example(
    model,
    num_epochs=5,
    batch_size=32,
    learning_rate=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """Train the model on synthetic data."""
    print(f"Training model with fusion method: {model.fusion_module.config.fusion_method}")
    
    # Move model to device
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        # Generate synthetic data
        batch = generate_synthetic_data(batch_size=batch_size, device=device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output, fusion_loss = model(batch, return_loss=True)
        
        # Calculate classification loss (cross entropy)
        target = batch["target"]
        output = output.view(-1, output.shape[-1])
        target = target.view(-1)
        
        # For simplicity, convert output to have 10 classes
        logits = torch.nn.Linear(model.hidden_dim, 10).to(device)(output)
        class_loss = F.cross_entropy(logits, target)
        
        # Combine losses
        total_loss = class_loss + fusion_loss
        
        # Backward pass
        total_loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Print progress
        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Class Loss: {class_loss.item():.4f}, "
                  f"Fusion Loss: {fusion_loss.item():.4f}")
        
        # Get fusion stats
        if epoch == num_epochs - 1:
            stats = model.fusion_module.get_fusion_stats()
            print("\nFusion Stats:")
            for stat_name, stat_values in stats.items():
                if isinstance(stat_values, dict):
                    print(f"  {stat_name}:")
                    for key, value in stat_values.items():
                        if hasattr(value, "item"):
                            print(f"    {key}: {value.item():.4f}")
                        else:
                            print(f"    {key}: {value}")
                elif stat_values is not None and hasattr(stat_values, "shape"):
                    print(f"  {stat_name}: tensor of shape {stat_values.shape}")
    
    return model


def compare_fusion_methods(
    hidden_dim=256,
    output_dim=128,
    num_epochs=5,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """Compare different fusion methods."""
    fusion_methods = [
        "adaptive_attention",
        "cross_modal",
        "gated",
        "weighted_sum"
    ]
    
    results = {}
    
    for method in fusion_methods:
        print(f"\n=== Testing fusion method: {method} ===")
        
        # Create model
        model = SimpleMultimodalModel(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            fusion_method=method
        )
        
        # Train model
        model = train_example(
            model=model,
            num_epochs=num_epochs,
            device=device
        )
        
        # Evaluate model
        eval_batch = generate_synthetic_data(batch_size=64, device=device)
        with torch.no_grad():
            output, fusion_loss = model(eval_batch, return_loss=True)
            
            # Calculate classification loss
            target = eval_batch["target"]
            output = output.view(-1, output.shape[-1])
            target = target.view(-1)
            
            # For simplicity, convert output to have 10 classes
            logits = torch.nn.Linear(model.hidden_dim, 10).to(device)(output)
            class_loss = F.cross_entropy(logits, target)
            
            # Calculate accuracy
            predictions = logits.argmax(dim=1)
            correct = (predictions == target).float().mean()
            
            # Store results
            results[method] = {
                "class_loss": class_loss.item(),
                "fusion_loss": fusion_loss.item(),
                "accuracy": correct.item()
            }
    
    # Print comparison
    print("\n=== Fusion Method Comparison ===")
    print(f"{'Method':<20} {'Class Loss':<12} {'Fusion Loss':<12} {'Accuracy':<12}")
    print("-" * 60)
    for method, metrics in results.items():
        print(f"{method:<20} {metrics['class_loss']:<12.4f} {metrics['fusion_loss']:<12.4f} {metrics['accuracy']:<12.4f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    methods = list(results.keys())
    accuracies = [results[method]["accuracy"] for method in methods]
    plt.bar(methods, accuracies)
    plt.title("Accuracy by Fusion Method")
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    class_losses = [results[method]["class_loss"] for method in methods]
    fusion_losses = [results[method]["fusion_loss"] for method in methods]
    x = np.arange(len(methods))
    width = 0.35
    plt.bar(x - width/2, class_losses, width, label="Class Loss")
    plt.bar(x + width/2, fusion_losses, width, label="Fusion Loss")
    plt.title("Losses by Fusion Method")
    plt.xticks(x, methods, rotation=45)
    plt.ylabel("Loss")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("fusion_method_comparison.png")
    print("Comparison plot saved as 'fusion_method_comparison.png'")


if __name__ == "__main__":
    # Use CPU if running in environment without GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create and train a simple model
    model = SimpleMultimodalModel(
        hidden_dim=256,
        output_dim=128,
        fusion_method="adaptive_attention"
    )
    
    print("\n=== Training simple model ===")
    train_example(model, num_epochs=5, device=device)
    
    # Compare different fusion methods
    print("\n=== Comparing fusion methods ===")
    compare_fusion_methods(num_epochs=3, device=device) 