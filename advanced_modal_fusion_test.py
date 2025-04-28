import torch
import unittest
from advanced_modal_fusion import AdvancedModalFusion, ModalFusionConfig

class TestAdvancedModalFusion(unittest.TestCase):
    def setUp(self):
        # Set up common test parameters
        self.batch_size = 4
        self.seq_len = 10
        self.text_dim = 768
        self.image_dim = 1024
        self.audio_dim = 512
        self.hidden_dim = 256
        self.output_dim = 256
        
        # Create modality dimensions dictionary
        self.modality_dims = {
            "text": self.text_dim,
            "image": self.image_dim,
            "audio": self.audio_dim
        }
        
        # Create input tensors
        self.inputs = {
            "text": torch.randn(self.batch_size, self.seq_len, self.text_dim),
            "image": torch.randn(self.batch_size, self.seq_len, self.image_dim),
            "audio": torch.randn(self.batch_size, self.seq_len, self.audio_dim)
        }
        
        # Create attention masks (1 = keep, 0 = mask)
        self.masks = {
            "text": torch.ones(self.batch_size, self.seq_len, dtype=torch.bool),
            "image": torch.ones(self.batch_size, self.seq_len, dtype=torch.bool),
            "audio": torch.ones(self.batch_size, self.seq_len, dtype=torch.bool)
        }
        
        # Mask some positions for testing
        self.masks["text"][:, 8:] = False
        self.masks["image"][:, 5:7] = False
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Mock model for testing
        self.model = torch.nn.Module()
    
    def test_adaptive_attention_fusion(self):
        """Test the adaptive attention fusion method."""
        config = ModalFusionConfig(
            fusion_method="adaptive_attention",
            attention_heads=4,
            hidden_dim=self.hidden_dim,
            dropout_rate=0.1
        )
        
        fusion_module = AdvancedModalFusion(
            config=config,
            model=self.model,
            modality_dims=self.modality_dims,
            output_dim=self.output_dim,
            device=self.device
        ).to(self.device)
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in self.inputs.items()}
        masks = {k: v.to(self.device) for k, v in self.masks.items()}
        
        # Forward pass
        output = fusion_module(inputs, masks)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.output_dim))
        
        # Test with return_loss=True
        output, loss = fusion_module(inputs, masks, return_loss=True)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.output_dim))
        self.assertIsInstance(loss, torch.Tensor)
        
        # Test stats collection
        stats = fusion_module.get_fusion_stats()
        self.assertIn("avg_modality_weights", stats)
        self.assertIn("mutual_information", stats)
        
    def test_cross_modal_fusion(self):
        """Test the cross-modal fusion method."""
        config = ModalFusionConfig(
            fusion_method="cross_modal",
            attention_heads=4,
            hidden_dim=self.hidden_dim,
            dropout_rate=0.1
        )
        
        fusion_module = AdvancedModalFusion(
            config=config,
            model=self.model,
            modality_dims=self.modality_dims,
            output_dim=self.output_dim,
            device=self.device
        ).to(self.device)
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in self.inputs.items()}
        masks = {k: v.to(self.device) for k, v in self.masks.items()}
        
        # Forward pass
        output = fusion_module(inputs, masks)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.output_dim))
    
    def test_gated_fusion(self):
        """Test the gated fusion method."""
        config = ModalFusionConfig(
            fusion_method="gated",
            hidden_dim=self.hidden_dim,
            dropout_rate=0.1
        )
        
        fusion_module = AdvancedModalFusion(
            config=config,
            model=self.model,
            modality_dims=self.modality_dims,
            output_dim=self.output_dim,
            device=self.device
        ).to(self.device)
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in self.inputs.items()}
        masks = {k: v.to(self.device) for k, v in self.masks.items()}
        
        # Forward pass
        output = fusion_module(inputs, masks)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.output_dim))
    
    def test_weighted_sum_fusion(self):
        """Test the weighted sum fusion method."""
        config = ModalFusionConfig(
            fusion_method="weighted_sum",
            hidden_dim=self.hidden_dim,
            dropout_rate=0.1
        )
        
        fusion_module = AdvancedModalFusion(
            config=config,
            model=self.model,
            modality_dims=self.modality_dims,
            output_dim=self.output_dim,
            device=self.device
        ).to(self.device)
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in self.inputs.items()}
        masks = {k: v.to(self.device) for k, v in self.masks.items()}
        
        # Forward pass
        output = fusion_module(inputs, masks)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.output_dim))
    
    def test_missing_modalities(self):
        """Test fusion with missing modalities."""
        config = ModalFusionConfig(
            fusion_method="adaptive_attention",
            hidden_dim=self.hidden_dim,
            dropout_rate=0.1
        )
        
        fusion_module = AdvancedModalFusion(
            config=config,
            model=self.model,
            modality_dims=self.modality_dims,
            output_dim=self.output_dim,
            device=self.device
        ).to(self.device)
        
        # Create inputs with missing modality
        incomplete_inputs = {
            "text": self.inputs["text"].to(self.device),
            "image": self.inputs["image"].to(self.device)
        }
        
        # Only include masks for the modalities that are present
        masks = {
            "text": self.masks["text"].to(self.device),
            "image": self.masks["image"].to(self.device)
        }
        
        # Forward pass
        output = fusion_module(incomplete_inputs, masks)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.output_dim))
        
        # Check stats to ensure they record the missing modality
        stats = fusion_module.get_fusion_stats()
        # Check if we're tracking missing modalities
        if "missing_modality_percentages" in stats:
            self.assertEqual(stats["missing_modality_percentages"].get("audio", 0), 100)
    
    def test_dynamic_weighting(self):
        """Test dynamic weighting."""
        # Test with dynamic weighting enabled
        config = ModalFusionConfig(
            fusion_method="weighted_sum",
            hidden_dim=self.hidden_dim,
            dynamic_weighting=True
        )
        
        fusion_module = AdvancedModalFusion(
            config=config,
            model=self.model,
            modality_dims=self.modality_dims,
            output_dim=self.output_dim,
            device=self.device
        ).to(self.device)
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in self.inputs.items()}
        masks = {k: v.to(self.device) for k, v in self.masks.items()}
        
        # Forward pass
        output_with_dynamic = fusion_module(inputs, masks)
        
        # Now test with dynamic weighting disabled
        config.dynamic_weighting = False
        fusion_module.config = config
        
        output_without_dynamic = fusion_module(inputs, masks)
        
        # The outputs should be different
        self.assertFalse(torch.allclose(output_with_dynamic, output_without_dynamic, atol=1e-5))
    
    def test_mutual_information_computation(self):
        """Test mutual information computation."""
        config = ModalFusionConfig(
            fusion_method="adaptive_attention",
            hidden_dim=self.hidden_dim,
            collect_stats=True
        )
        
        fusion_module = AdvancedModalFusion(
            config=config,
            model=self.model,
            modality_dims=self.modality_dims,
            output_dim=self.output_dim,
            device=self.device
        ).to(self.device)
        
        # Reset stats
        fusion_module.reset_stats()
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in self.inputs.items()}
        masks = {k: v.to(self.device) for k, v in self.masks.items()}
        
        # Forward pass
        fusion_module(inputs, masks)
        
        # Check mutual information stats
        stats = fusion_module.get_fusion_stats()
        self.assertIn("mutual_information", stats)
        # Check if at least one modality pair has MI computed
        self.assertGreater(len(stats["mutual_information"]), 0)
    
    def test_context_gating(self):
        """Test context gating."""
        # Test with context gating enabled
        config = ModalFusionConfig(
            fusion_method="weighted_sum",
            hidden_dim=self.hidden_dim,
            use_context_gating=True
        )
        
        fusion_module = AdvancedModalFusion(
            config=config,
            model=self.model,
            modality_dims=self.modality_dims,
            output_dim=self.output_dim,
            device=self.device
        ).to(self.device)
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in self.inputs.items()}
        masks = {k: v.to(self.device) for k, v in self.masks.items()}
        
        # Forward pass
        output_with_gating = fusion_module(inputs, masks)
        
        # Now test with context gating disabled
        config.use_context_gating = False
        fusion_module.config = config
        
        # Re-initialize to disable the gating module
        fusion_module = AdvancedModalFusion(
            config=config,
            model=self.model,
            modality_dims=self.modality_dims,
            output_dim=self.output_dim,
            device=self.device
        ).to(self.device)
        
        output_without_gating = fusion_module(inputs, masks)
        
        # The outputs should be different in general
        # Note: This test might be flaky since weights are randomly initialized
        differences = (output_with_gating - output_without_gating).abs().sum().item()
        self.assertGreater(differences, 0)

if __name__ == "__main__":
    unittest.main() 