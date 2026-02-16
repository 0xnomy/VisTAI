"""
ConvNeXt Teacher Model for Bone Tumor Classification
Large capacity model for knowledge distillation
"""

import torch
import torch.nn as nn
import timm


class ConvNeXtTeacher(nn.Module):
    """
    ConvNeXt-Small/Base model as teacher for knowledge distillation.
    Modernized ResNet with inverted bottlenecks and layer normalization.
    """
    
    def __init__(self, num_classes=9, model_size='small', pretrained=True, dropout=0.3):
        """
        Args:
            num_classes: Number of output classes
            model_size: 'tiny', 'small', 'base', 'large'
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate before classifier
        """
        super().__init__()
        
        # Model size mapping
        model_names = {
            'tiny': 'convnext_tiny.fb_in22k_ft_in1k',
            'small': 'convnext_small.fb_in22k_ft_in1k', 
            'base': 'convnext_base.fb_in22k_ft_in1k',
            'large': 'convnext_large.fb_in22k_ft_in1k'
        }
        
        if model_size not in model_names:
            raise ValueError(f"model_size must be one of {list(model_names.keys())}")
        
        self.model_size = model_size
        model_name = model_names[model_size]
        
        # Load pretrained ConvNeXt
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # We'll add custom pooling
        )
        
        # Get feature dimension
        if model_size == 'tiny':
            self.feature_dim = 768
        elif model_size == 'small':
            self.feature_dim = 768
        elif model_size == 'base':
            self.feature_dim = 1024
        else:  # large
            self.feature_dim = 1536
        
        # Custom classifier head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(self.feature_dim, num_classes)
        
        # Initialize classifier weights
        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)
    
    def forward(self, x, return_features=False):
        """
        Forward pass with optional feature extraction.
        
        Args:
            x: Input tensor [B, 3, H, W]
            return_features: If True, return intermediate features for distillation
        
        Returns:
            logits: Classification logits [B, num_classes]
            features: (optional) Feature maps for distillation
        """
        # Extract features
        features = self.backbone(x)  # [B, C, H, W]
        
        # Global pooling
        pooled = self.global_pool(features)  # [B, C, 1, 1]
        pooled = torch.flatten(pooled, 1)    # [B, C]
        
        # Classifier
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        
        if return_features:
            return logits, {
                'feature_maps': features,
                'pooled_features': pooled
            }
        
        return logits
    
    def freeze_backbone(self):
        """Freeze all backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def get_num_params(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_convnext_teacher(num_classes=9, model_size='small', pretrained=True, dropout=0.3):
    """
    Factory function to create ConvNeXt teacher model.
    
    Args:
        num_classes: Number of output classes
        model_size: 'tiny' (28M), 'small' (50M), 'base' (89M), 'large' (198M)
        pretrained: Use ImageNet-22K → ImageNet-1K pretrained weights
        dropout: Dropout rate
    
    Returns:
        ConvNeXt model
    """
    model = ConvNeXtTeacher(
        num_classes=num_classes,
        model_size=model_size,
        pretrained=pretrained,
        dropout=dropout
    )
    
    print(f"ConvNeXt-{model_size.capitalize()} Teacher Model")
    print(f"Total parameters: {model.get_num_params():,}")
    print(f"Trainable parameters: {model.get_num_trainable_params():,}")
    
    return model


if __name__ == '__main__':
    # Test model creation
    print("Testing ConvNeXt Teacher Models:\n")
    
    for size in ['tiny', 'small']:
        print(f"\n{'='*50}")
        model = create_convnext_teacher(num_classes=9, model_size=size, pretrained=False)
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        logits = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {logits.shape}")
        
        # Test feature extraction
        logits, features = model(x, return_features=True)
        print(f"Feature maps shape: {features['feature_maps'].shape}")
        print(f"Pooled features shape: {features['pooled_features'].shape}")
