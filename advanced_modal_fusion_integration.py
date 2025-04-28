import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torchvision.transforms as transforms
from advanced_modal_fusion import AdvancedModalFusion, ModalFusionConfig

class MultimodalTransformer(nn.Module):
    """
    A multimodal transformer model that integrates text, image, and audio modalities
    using the AdvancedModalFusion module.
    """
    
    def __init__(
        self,
        text_model_name="bert-base-uncased",
        image_model_name="google/vit-base-patch16-224",
        audio_model_name="microsoft/wavlm-base-plus",
        fusion_method="adaptive_attention",
        num_labels=2,
        hidden_dim=256,
        dropout_rate=0.1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        self.device = device
        
        # Initialize text encoder
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        
        # Initialize image encoder
        self.image_encoder = AutoModel.from_pretrained(image_model_name)
        self.image_processor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Initialize audio encoder
        self.audio_encoder = AutoModel.from_pretrained(audio_model_name)
        
        # Get modality dimensions
        self.modality_dims = {
            "text": self.text_encoder.config.hidden_size,
            "image": self.image_encoder.config.hidden_size,
            "audio": self.audio_encoder.config.hidden_size
        }
        
        # Create fusion config
        self.fusion_config = ModalFusionConfig(
            fusion_method=fusion_method,
            hidden_dim=hidden_dim,
            attention_heads=8,
            dropout_rate=dropout_rate,
            dynamic_weighting=True,
            collect_stats=True,
            use_context_gating=True
        )
        
        # Initialize fusion module
        self.fusion_module = AdvancedModalFusion(
            config=self.fusion_config,
            model=self,
            modality_dims=self.modality_dims,
            output_dim=hidden_dim,
            device=device
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_labels)
        )
        
    def encode_text(self, text_inputs):
        """Encode text inputs using the text encoder."""
        if isinstance(text_inputs[0], str):
            # Tokenize text if input is a list of strings
            encoded_text = self.text_tokenizer(
                text_inputs,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)
            
            # Get text features
            text_outputs = self.text_encoder(
                input_ids=encoded_text.input_ids,
                attention_mask=encoded_text.attention_mask,
                return_dict=True
            )
            
            # Return sequence output and attention mask
            return text_outputs.last_hidden_state, encoded_text.attention_mask
        else:
            # Assume input is already tokenized
            return self.text_encoder(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                return_dict=True
            ).last_hidden_state, text_inputs["attention_mask"]
    
    def encode_image(self, image_inputs):
        """Encode image inputs using the image encoder."""
        if isinstance(image_inputs[0], Image.Image):
            # Process images if input is a list of PIL images
            processed_images = torch.stack([
                self.image_processor(img) for img in image_inputs
            ]).to(self.device)
            
            # Get image features
            image_outputs = self.image_encoder(
                processed_images,
                return_dict=True
            )
            
            # Return sequence output and attention mask (all 1s for images)
            batch_size = processed_images.size(0)
            seq_len = image_outputs.last_hidden_state.size(1)
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.device)
            return image_outputs.last_hidden_state, attention_mask
        else:
            # Assume input is already processed
            return self.image_encoder(
                image_inputs["pixel_values"],
                return_dict=True
            ).last_hidden_state, image_inputs.get("attention_mask", 
                torch.ones(image_inputs["pixel_values"].size(0), 
                          self.image_encoder.config.image_size // self.image_encoder.config.patch_size, 
                          dtype=torch.bool, device=self.device))
    
    def encode_audio(self, audio_inputs):
        """Encode audio inputs using the audio encoder."""
        if audio_inputs is None:
            return None, None
            
        # Get audio features
        audio_outputs = self.audio_encoder(
            audio_inputs["input_values"],
            attention_mask=audio_inputs.get("attention_mask", None),
            return_dict=True
        )
        
        # Return sequence output and attention mask
        return audio_outputs.last_hidden_state, audio_inputs.get("attention_mask", 
            torch.ones(audio_outputs.last_hidden_state.size(0), 
                      audio_outputs.last_hidden_state.size(1), 
                      dtype=torch.bool, device=self.device))
    
    def forward(self, batch, return_loss=False):
        """
        Forward pass through the multimodal transformer.
        
        Args:
            batch: Dictionary containing modality inputs:
                - text: Text inputs (tokenized or list of strings)
                - image: Image inputs (processed or list of PIL images)
                - audio: Audio inputs (processed or None)
            return_loss: Whether to return fusion loss
            
        Returns:
            logits: Classification logits
            loss: Optional fusion loss if return_loss=True
        """
        # Encode each modality
        modality_features = {}
        modality_masks = {}
        
        # Text encoding
        if "text" in batch and batch["text"] is not None:
            modality_features["text"], modality_masks["text"] = self.encode_text(batch["text"])
        
        # Image encoding
        if "image" in batch and batch["image"] is not None:
            modality_features["image"], modality_masks["image"] = self.encode_image(batch["image"])
        
        # Audio encoding
        if "audio" in batch and batch["audio"] is not None:
            modality_features["audio"], modality_masks["audio"] = self.encode_audio(batch["audio"])
        
        # Apply fusion
        if return_loss:
            fused_features, fusion_loss = self.fusion_module(
                modality_features, modality_masks, return_loss=True
            )
        else:
            fused_features = self.fusion_module(modality_features, modality_masks)
        
        # Apply classification head to [CLS] token or mean-pooled representation
        if modality_features.get("text") is not None:
            # Use [CLS] token from text if available
            pooled_output = fused_features[:, 0]
        else:
            # Otherwise use mean pooling
            pooled_output = fused_features.mean(dim=1)
        
        # Get classification logits
        logits = self.classifier(pooled_output)
        
        if return_loss:
            return logits, fusion_loss
        return logits
    
    def get_fusion_stats(self):
        """Get fusion statistics."""
        return self.fusion_module.get_fusion_stats()
    
    def reset_fusion_stats(self):
        """Reset fusion statistics."""
        self.fusion_module.reset_stats()


def example_usage():
    """Example demonstrating how to use the MultimodalTransformer."""
    # Initialize model
    model = MultimodalTransformer(
        fusion_method="adaptive_attention",
        num_labels=3,  # For a 3-class classification task
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Sample batch (assuming preprocessing is done separately)
    batch = {
        "text": ["This is a sample text input", "Another example text"],
        "image": [Image.new('RGB', (224, 224)), Image.new('RGB', (224, 224))],
        "audio": None  # No audio in this example
    }
    
    # Forward pass
    logits = model(batch)
    print(f"Output logits shape: {logits.shape}")
    
    # Get fusion statistics
    fusion_stats = model.get_fusion_stats()
    print("Fusion Statistics:")
    for key, value in fusion_stats.items():
        print(f"  {key}: {value}")
    
    # Forward pass with loss
    logits, fusion_loss = model(batch, return_loss=True)
    print(f"Fusion loss: {fusion_loss.item()}")
    
    # Reset fusion statistics
    model.reset_fusion_stats()
    
    return model


if __name__ == "__main__":
    example_usage() 