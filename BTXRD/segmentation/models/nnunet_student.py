"""
nnU-Net-style Student Model
===========================
High-capacity U-Net with deep supervision for binary tumor segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double convolution block with InstanceNorm and LeakyReLU.
    nnU-Net style: Conv → InstanceNorm → LeakyReLU → Conv → InstanceNorm → LeakyReLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downsampling block: MaxPool → DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Upsampling block with skip connections"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class NNUNetStudent(nn.Module):
    """
    nnU-Net-style high-capacity U-Net for student model.
    5-stage encoder with channels [32, 64, 128, 256, 512]
    Supports deep supervision during training for better gradients.
    
    ~7.8M parameters
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1, deep_supervision: bool = True):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # Encoder: 5 stages
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        
        # Decoder: 4 upsampling stages
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)
        
        # Final output head
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)
        
        # Deep supervision heads
        if self.deep_supervision:
            self.ds1 = nn.Conv2d(256, out_channels, kernel_size=1)  # 1/8 resolution
            self.ds2 = nn.Conv2d(128, out_channels, kernel_size=1)  # 1/4 resolution
            self.ds3 = nn.Conv2d(64, out_channels, kernel_size=1)   # 1/2 resolution
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)      # 32 channels, full resolution
        x2 = self.down1(x1)   # 64 channels, 1/2
        x3 = self.down2(x2)   # 128 channels, 1/4
        x4 = self.down3(x3)   # 256 channels, 1/8
        x5 = self.down4(x4)   # 512 channels, 1/16
        
        # Decoder with skip connections
        d4 = self.up1(x5, x4)  # 256 channels, 1/8
        d3 = self.up2(d4, x3)  # 128 channels, 1/4
        d2 = self.up3(d3, x2)  # 64 channels, 1/2
        d1 = self.up4(d2, x1)  # 32 channels, full resolution
        
        # Final output (logits, no sigmoid)
        out = self.outc(d1)
        
        # Deep supervision outputs (for training only)
        if self.training and self.deep_supervision:
            ds_out1 = self.ds1(d4)  # 1/8 resolution
            ds_out2 = self.ds2(d3)  # 1/4 resolution
            ds_out3 = self.ds3(d2)  # 1/2 resolution
            return out, ds_out3, ds_out2, ds_out1
        
        return out
    
    def get_num_params(self):
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_nnunet_student(in_channels: int = 3, out_channels: int = 1, deep_supervision: bool = True, **kwargs):
    """Factory function for nnU-Net student model."""
    model = NNUNetStudent(in_channels=in_channels, out_channels=out_channels, deep_supervision=deep_supervision)
    print(f"nnU-Net Student Model")
    print(f"  Total parameters: {model.get_num_params():,}")
    return model


if __name__ == "__main__":
    print("Testing nnU-Net Student Model...")
    
    model = create_nnunet_student()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224).to(device)
    
    model.train()
    outputs = model(x)
    print(f"\nTraining mode (deep supervision):")
    print(f"  Input: {x.shape}")
    if isinstance(outputs, tuple):
        for i, o in enumerate(outputs):
            print(f"  Output {i}: {o.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    print(f"\nEval mode:")
    print(f"  Output: {output.shape}")
    
    print("\n✓ Test passed!")
