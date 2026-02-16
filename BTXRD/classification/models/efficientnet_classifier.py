"""
EfficientNet-B0 Classifier for Bone Tumor Classification
========================================================
Standalone classification model with freeze/unfreeze capability.
"""

import torch
import torch.nn as nn
from torchvision import models


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B0 based classifier for tumor classification.
    
    Supports two-phase training:
    - Phase 1: Freeze backbone, train only classifier head
    - Phase 2: Unfreeze all, fine-tune end-to-end
    """
    
    def __init__(self, num_classes=9, pretrained=True, dropout=0.3):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate before final FC layer
        """
        super(EfficientNetClassifier, self).__init__()
        
        # Load pretrained EfficientNet-B0
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_b0(weights=weights)
        else:
            self.backbone = models.efficientnet_b0(weights=None)
        
        # Get feature dimension
        in_features = self.backbone.classifier[1].in_features
        
        # Replace classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
        self.num_classes = num_classes
        self._frozen = False
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, 224, 224]
        
        Returns:
            Class logits [B, num_classes]
        """
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze backbone parameters (for phase 1 training)."""
        # Freeze all except classifier
        for name, param in self.backbone.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        
        self._frozen = True
        print("✓ Backbone frozen - only training classifier head")
    
    def unfreeze_backbone(self):
        """Unfreeze all parameters (for phase 2 fine-tuning)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        
        self._frozen = False
        print("✓ Backbone unfrozen - training end-to-end")
    
    def is_frozen(self):
        """Check if backbone is frozen."""
        return self._frozen
    
    def get_trainable_parameters(self):
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_parameters(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_efficientnet_classifier(num_classes=9, pretrained=True, dropout=0.3):
    """
    Factory function to create EfficientNet classifier.
    
    Args:
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        dropout: Dropout rate
    
    Returns:
        EfficientNetClassifier model
    """
    model = EfficientNetClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )
    
    print(f"EfficientNet-B0 Classifier created:")
    print(f"  Total parameters: {model.get_total_parameters():,}")
    print(f"  Trainable parameters: {model.get_trainable_parameters():,}")
    
    return model


if __name__ == "__main__":
    # Test model
    print("Testing EfficientNet-B0 Classifier...")
    
    model = create_efficientnet_classifier(num_classes=9, pretrained=True)
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test freeze/unfreeze
    print("\nTesting freeze/unfreeze:")
    model.freeze_backbone()
    print(f"  Trainable params (frozen): {model.get_trainable_parameters():,}")
    
    model.unfreeze_backbone()
    print(f"  Trainable params (unfrozen): {model.get_trainable_parameters():,}")
    
    print("\n✓ Model test passed!")
