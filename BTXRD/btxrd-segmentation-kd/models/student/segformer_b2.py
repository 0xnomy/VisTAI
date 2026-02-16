import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerConfig


class SegFormerB2(nn.Module):
    def __init__(self, num_classes=1, image_size=224, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        
        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b2-finetuned-ade-512-512",
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            config = SegformerConfig.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
            config.num_labels = num_classes
            self.model = SegformerForSemanticSegmentation(config)
        
        self.decoder_channels = [128, 256, 512, 512]
    
    def forward(self, x, return_features=False):
        input_size = x.shape[2:]
        outputs = self.model(x, output_hidden_states=True, return_dict=True)
        logits = outputs.logits
        logits = nn.functional.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        
        if return_features:
            # SegFormer encoder outputs 4 stages: hidden_states[0-3]
            # Extract the last 3 stages to match teacher's 3 decoder features
            hidden_states = outputs.hidden_states
            decoder_features = []
            for i in [1, 2, 3]:
                feat = hidden_states[i]
                # Check if feat is in sequence format (B, N, C) or spatial format (B, C, H, W)
                if len(feat.shape) == 3:
                    # Sequence format: reshape to spatial
                    B, N, C = feat.shape
                    H = W = int(N ** 0.5)
                    feat = feat.permute(0, 2, 1).reshape(B, C, H, W)
                # else: already in spatial format (B, C, H, W)
                decoder_features.append(feat)
            
            features = {'decoder': decoder_features}
            return logits, features
        
        return logits
    
    def freeze_encoder(self):
        for name, param in self.model.named_parameters():
            if 'segformer.encoder' in name:
                param.requires_grad = False
    
    def unfreeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = True
