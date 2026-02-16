import torch
import torch.nn as nn
import timm


class ConvNeXtTinyStudent(nn.Module):
    def __init__(self, num_classes=9, pretrained=True):
        super().__init__()
        self.model = timm.create_model('convnext_tiny', pretrained=pretrained, num_classes=num_classes)
        self.feature_dim = self.model.head.fc.in_features
        
    def forward(self, x, return_features=False):
        features = self.model.forward_features(x)
        features = self.model.head.global_pool(features)
        features = self.model.head.norm(features)
        features = self.model.head.flatten(features)
        logits = self.model.head.fc(features)
        
        if return_features:
            return logits, features
        return logits
    
    def freeze_backbone(self):
        for name, param in self.model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
    
    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True
