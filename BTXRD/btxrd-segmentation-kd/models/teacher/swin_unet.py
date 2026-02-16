import torch
import torch.nn as nn
import timm


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x, skip):
        x = torch.cat([x, skip], dim=1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class SwinUNet(nn.Module):
    def __init__(self, num_classes=1, img_size=224, deep_supervision=True, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.deep_supervision = deep_supervision
        
        if img_size == 224:
            model_name = 'swin_small_patch4_window7_224'
        elif img_size == 384:
            model_name = 'swin_base_patch4_window12_384'
        else:
            model_name = 'swin_small_patch4_window7_224'
        
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )
        
        feature_info = self.encoder.feature_info
        enc_channels = [info['num_chs'] for info in feature_info]
        dec_channels = [512, 256, 128, 64]
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(enc_channels[3], dec_channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dec_channels[0]),
            nn.ReLU(inplace=True)
        )
        
        self.decoder4 = DecoderBlock(dec_channels[0], enc_channels[2], dec_channels[1])
        self.decoder3 = DecoderBlock(dec_channels[1], enc_channels[1], dec_channels[2])
        self.decoder2 = DecoderBlock(dec_channels[2], enc_channels[0], dec_channels[3])
        
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(dec_channels[3], dec_channels[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dec_channels[3]),
            nn.ReLU(inplace=True)
        )
        
        self.segmentation_head = nn.Conv2d(dec_channels[3], num_classes, kernel_size=1)
        
        if deep_supervision:
            self.aux_head1 = nn.Conv2d(dec_channels[1], num_classes, kernel_size=1)
            self.aux_head2 = nn.Conv2d(dec_channels[2], num_classes, kernel_size=1)
            self.aux_head3 = nn.Conv2d(dec_channels[3], num_classes, kernel_size=1)
    
    def forward(self, x, return_features=False):
        input_size = x.shape[2:]
        encoder_features = self.encoder(x)
        encoder_features = [f.permute(0, 3, 1, 2).contiguous() if f.dim() == 4 and f.shape[-1] in [96, 192, 384, 768] else f 
                           for f in encoder_features]
        
        x = self.bottleneck(encoder_features[3])
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        x = self.decoder4(x, encoder_features[2])
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        decoder4_out = x
        
        x = self.decoder3(x, encoder_features[1])
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        decoder3_out = x
        
        x = self.decoder2(x, encoder_features[0])
        decoder2_out = x
        
        x = self.final_upsample(x)
        
        main_output = self.segmentation_head(x)
        main_output = nn.functional.interpolate(main_output, size=input_size, mode='bilinear', align_corners=False)
        
        if return_features:
            features = {
                'decoder': [decoder4_out, decoder3_out, decoder2_out]
            }
            return main_output, features
        
        if self.training and self.deep_supervision:
            aux1 = self.aux_head1(decoder4_out)
            aux1 = nn.functional.interpolate(aux1, size=input_size, mode='bilinear', align_corners=False)
            aux2 = self.aux_head2(decoder3_out)
            aux2 = nn.functional.interpolate(aux2, size=input_size, mode='bilinear', align_corners=False)
            aux3 = self.aux_head3(decoder2_out)
            aux3 = nn.functional.interpolate(aux3, size=input_size, mode='bilinear', align_corners=False)
            return main_output, [aux1, aux2, aux3]
        
        return main_output
