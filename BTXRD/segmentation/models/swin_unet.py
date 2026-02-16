"""
Swin-UNet: Swin Transformer + UNet Decoder
==========================================
State-of-the-art transformer-based segmentation model for medical imaging.

Architecture:
- Encoder: Swin Transformer (Small variant, pretrained on ImageNet-22K)
- Decoder: UNet-style progressive upsampling with skip connections
- Deep supervision: Multi-scale auxiliary outputs for better training
"""

import torch
import torch.nn as nn
import timm


class DecoderBlock(nn.Module):
    """UNet-style decoder block with skip connections"""
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def forward(self, x, skip):
        """
        Args:
            x: Upsampled features from previous decoder stage
            skip: Skip connection from encoder
        """
        x = torch.cat([x, skip], dim=1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class SwinUNet(nn.Module):
    """
    Swin-UNet: Swin Transformer + UNet Decoder
    
    Features:
    - Hierarchical Swin Transformer encoder (4 stages)
    - UNet decoder with skip connections
    - Deep supervision (optional)
    - ImageNet-22K pretrained weights
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        img_size: int = 224,
        embed_dim: int = 96,
        deep_supervision: bool = True,
        pretrained: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        self.deep_supervision = deep_supervision
        
        # Swin Transformer Encoder (Small variant)
        # Select appropriate Swin variant based on image size
        if img_size == 224:
            model_name = 'swin_small_patch4_window7_224'
        elif img_size == 384:
            model_name = 'swin_base_patch4_window12_384'  # Use base for 384 (better for larger images)
        else:
            # For other sizes, use 224 variant and hope timm handles it
            model_name = 'swin_small_patch4_window7_224'
            print(f"⚠️  Warning: Using Swin-Small-224 for img_size={img_size}. May cause issues.")
        
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3)  # Extract all 4 stages
        )
        
        # Get feature dimensions from encoder
        # Swin-Small: [96, 192, 384, 768] channels at strides [4, 8, 16, 32]
        feature_info = self.encoder.feature_info
        enc_channels = [info['num_chs'] for info in feature_info]  # [96, 192, 384, 768]
        
        # Decoder channels (progressively decreasing)
        dec_channels = [512, 256, 128, 64]
        
        # Bottleneck: Process highest-level features
        self.bottleneck = nn.Sequential(
            nn.Conv2d(enc_channels[3], dec_channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dec_channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # Decoder blocks (4 stages, bottom-up)
        self.decoder4 = DecoderBlock(dec_channels[0], enc_channels[2], dec_channels[1])  # 512+384 -> 256
        self.decoder3 = DecoderBlock(dec_channels[1], enc_channels[1], dec_channels[2])  # 256+192 -> 128
        self.decoder2 = DecoderBlock(dec_channels[2], enc_channels[0], dec_channels[3])  # 128+96 -> 64
        
        # Final upsampling to original resolution
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(dec_channels[3], dec_channels[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dec_channels[3]),
            nn.ReLU(inplace=True)
        )
        
        # Segmentation head (output logits)
        self.segmentation_head = nn.Conv2d(dec_channels[3], num_classes, kernel_size=1)
        
        # Deep supervision heads (auxiliary outputs for training)
        if deep_supervision:
            self.aux_head1 = nn.Conv2d(dec_channels[1], num_classes, kernel_size=1)  # 1/8 resolution
            self.aux_head2 = nn.Conv2d(dec_channels[2], num_classes, kernel_size=1)  # 1/4 resolution
            self.aux_head3 = nn.Conv2d(dec_channels[3], num_classes, kernel_size=1)  # 1/2 resolution
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize decoder weights (encoder already pretrained)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m not in self.encoder.modules():  # Don't reinit encoder
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                if m not in self.encoder.modules():
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_features=False):
        """
        Args:
            x: Input tensor (B, 3, H, W)
            return_features: Return intermediate features (for KD)
        
        Returns:
            - Training mode with deep_supervision: tuple of (main_output, [aux1, aux2, aux3])
            - Inference mode: main_output only (B, 1, H, W)
            - If return_features: (output, feature_dict)
        """
        input_size = x.shape[2:]
        
        # Encoder (4 feature maps at different resolutions)
        encoder_features = self.encoder(x)
        
        # Swin Transformer returns features in NHWC format, convert to NCHW
        encoder_features = [f.permute(0, 3, 1, 2).contiguous() if f.dim() == 4 and f.shape[-1] in [96, 192, 384, 768] else f 
                           for f in encoder_features]
        
        # encoder_features[0]: (B, 96, H/4, W/4)
        # encoder_features[1]: (B, 192, H/8, W/8)
        # encoder_features[2]: (B, 384, H/16, W/16)
        # encoder_features[3]: (B, 768, H/32, W/32)
        
        # Bottleneck
        x = self.bottleneck(encoder_features[3])  # (B, 512, H/32, W/32)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # (B, 512, H/16, W/16)
        
        # Decoder stage 4: 1/16 -> 1/8
        x = self.decoder4(x, encoder_features[2])  # (B, 256, H/16, W/16)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # (B, 256, H/8, W/8)
        decoder4_out = x
        
        # Decoder stage 3: 1/8 -> 1/4
        x = self.decoder3(x, encoder_features[1])  # (B, 128, H/8, W/8)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # (B, 128, H/4, W/4)
        decoder3_out = x
        
        # Decoder stage 2: 1/4 -> 1/2
        x = self.decoder2(x, encoder_features[0])  # (B, 64, H/4, W/4)
        decoder2_out = x
        
        # Final upsampling: 1/4 -> 1/1
        x = self.final_upsample(x)  # (B, 64, H/2, W/2)
        
        # Main output
        main_output = self.segmentation_head(x)  # (B, 1, H/2, W/2)
        main_output = nn.functional.interpolate(main_output, size=input_size, mode='bilinear', align_corners=False)
        
        # Return features for knowledge distillation
        if return_features:
            features = {
                'encoder': encoder_features,
                'decoder': [decoder4_out, decoder3_out, decoder2_out, x]
            }
            return main_output, features
        
        # Deep supervision during training
        if self.training and self.deep_supervision:
            aux1 = self.aux_head1(decoder4_out)  # 1/8 resolution
            aux1 = nn.functional.interpolate(aux1, size=input_size, mode='bilinear', align_corners=False)
            
            aux2 = self.aux_head2(decoder3_out)  # 1/4 resolution
            aux2 = nn.functional.interpolate(aux2, size=input_size, mode='bilinear', align_corners=False)
            
            aux3 = self.aux_head3(decoder2_out)  # 1/2 resolution
            aux3 = nn.functional.interpolate(aux3, size=input_size, mode='bilinear', align_corners=False)
            
            return main_output, [aux1, aux2, aux3]
        
        return main_output
    
    def freeze_encoder(self):
        """Freeze encoder for two-phase training (Phase 1)"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print(f"✓ Frozen Swin encoder ({self.get_num_trainable_params():,} trainable params)")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder for fine-tuning (Phase 2)"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print(f"✓ Unfrozen Swin encoder ({self.get_num_trainable_params():,} trainable params)")
    
    def get_num_params(self):
        """Total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self):
        """Number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_swin_unet(
    num_classes: int = 1,
    img_size: int = 224,
    deep_supervision: bool = True,
    pretrained: bool = True
):
    """
    Factory function to create Swin-UNet model
    
    Args:
        num_classes: Number of output classes (1 for binary segmentation)
        img_size: Input image size (224 or 384)
        deep_supervision: Enable multi-scale supervision
        pretrained: Use ImageNet-22K pretrained weights
    
    Returns:
        SwinUNet model
    """
    model = SwinUNet(
        num_classes=num_classes,
        img_size=img_size,
        deep_supervision=deep_supervision,
        pretrained=pretrained
    )
    return model


if __name__ == '__main__':
    # Test model
    model = create_swin_unet(num_classes=1, img_size=224, deep_supervision=True, pretrained=False)
    print(f"Total parameters: {model.get_num_params():,}")
    print(f"Trainable parameters: {model.get_num_trainable_params():,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    
    # Training mode
    model.train()
    output = model(x)
    if isinstance(output, tuple):
        main, aux = output
        print(f"Training - Main output: {main.shape}, Aux outputs: {len(aux)}")
    
    # Inference mode
    model.eval()
    with torch.no_grad():
        output = model(x)
        print(f"Inference output: {output.shape}")
    
    # Test freeze/unfreeze
    model.freeze_encoder()
    model.unfreeze_encoder()
