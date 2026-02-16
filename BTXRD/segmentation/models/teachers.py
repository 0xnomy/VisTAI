"""
Segmentation Teacher Models
===========================
UNet++, DeepLabV3+, and MAnet teacher models using segmentation-models-pytorch.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class UNetPlusPlusTeacher(nn.Module):
    """
    UNet++ (Nested UNet) teacher model for segmentation.
    Enhanced skip connections for better feature propagation.
    """
    
    def __init__(
        self, 
        encoder_name: str = 'resnet50',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 1,
        encoder_depth: int = 5,
        decoder_channels: tuple = (256, 128, 64, 32, 16)
    ):
        super().__init__()
        
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.classes = classes
        
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels
        )
        
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.segmentation_head = self.model.segmentation_head
    
    def forward(self, x, return_features=False):
        if return_features:
            encoder_features = self.encoder(x)
            decoder_output = self.decoder(*encoder_features)
            mask = self.segmentation_head(decoder_output)
            return mask, {'encoder': encoder_features, 'decoder': decoder_output}
        return self.model(x)
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DeepLabV3PlusTeacher(nn.Module):
    """
    DeepLabV3+ teacher model.
    Uses atrous spatial pyramid pooling (ASPP).
    """
    
    def __init__(
        self,
        encoder_name: str = 'resnet50',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 1,
        encoder_depth: int = 5,
        encoder_output_stride: int = 16
    ):
        super().__init__()
        
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.classes = classes
        
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
            encoder_depth=encoder_depth,
            encoder_output_stride=encoder_output_stride
        )
        
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.segmentation_head = self.model.segmentation_head
    
    def forward(self, x, return_features=False):
        if return_features:
            encoder_features = self.encoder(x)
            decoder_output = self.decoder(*encoder_features)
            mask = self.segmentation_head(decoder_output)
            return mask, {'encoder': encoder_features, 'decoder': decoder_output}
        return self.model(x)
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MAnetTeacher(nn.Module):
    """
    MAnet (Multi-scale Attention Network) teacher model.
    Uses attention mechanisms to focus on important regions.
    """
    
    def __init__(
        self,
        encoder_name: str = 'efficientnet-b4',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 1,
        encoder_depth: int = 5,
        decoder_channels: tuple = (256, 128, 64, 32, 16),
        decoder_pab_channels: int = 64
    ):
        super().__init__()
        
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.classes = classes
        
        self.model = smp.MAnet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            decoder_pab_channels=decoder_pab_channels
        )
        
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.segmentation_head = self.model.segmentation_head
    
    def forward(self, x, return_features=False):
        if return_features:
            encoder_features = self.encoder(x)
            decoder_output = self.decoder(*encoder_features)
            mask = self.segmentation_head(decoder_output)
            return mask, {'encoder': encoder_features, 'decoder': decoder_output}
        return self.model(x)
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Factory functions
def create_unetplusplus_teacher(encoder_name='resnet50', encoder_weights='imagenet', **kwargs):
    """Factory function for UNet++ teacher."""
    model = UNetPlusPlusTeacher(encoder_name=encoder_name, encoder_weights=encoder_weights, **kwargs)
    print(f"UNet++ Teacher ({encoder_name})")
    print(f"  Total parameters: {model.get_num_params():,}")
    return model


def create_deeplabv3plus_teacher(encoder_name='resnet50', encoder_weights='imagenet', **kwargs):
    """Factory function for DeepLabV3+ teacher."""
    model = DeepLabV3PlusTeacher(encoder_name=encoder_name, encoder_weights=encoder_weights, **kwargs)
    print(f"DeepLabV3+ Teacher ({encoder_name})")
    print(f"  Total parameters: {model.get_num_params():,}")
    return model


def create_manet_teacher(encoder_name='efficientnet-b4', encoder_weights='imagenet', **kwargs):
    """Factory function for MAnet teacher."""
    model = MAnetTeacher(encoder_name=encoder_name, encoder_weights=encoder_weights, **kwargs)
    print(f"MAnet Teacher ({encoder_name})")
    print(f"  Total parameters: {model.get_num_params():,}")
    return model


if __name__ == '__main__':
    print("Testing Segmentation Teacher Models...\n")
    
    x = torch.randn(2, 3, 256, 256)
    
    # Test UNet++
    print("=" * 50)
    model = create_unetplusplus_teacher(encoder_weights=None)
    output = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    
    # Test DeepLabV3+
    print("\n" + "=" * 50)
    model = create_deeplabv3plus_teacher(encoder_weights=None)
    output = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    
    # Test MAnet
    print("\n" + "=" * 50)
    model = create_manet_teacher(encoder_weights=None)
    output = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    
    print("\n✓ All tests passed!")
