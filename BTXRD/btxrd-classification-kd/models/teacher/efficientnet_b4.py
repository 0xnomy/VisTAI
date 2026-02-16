import torch
import torch.nn as nn
import timm


class EfficientNetB4Teacher(nn.Module):
    def __init__(self, num_classes=9, pretrained=True):
        super().__init__()
        self.model = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=num_classes)
        self.feature_dim = self.model.classifier.in_features
        
    def forward(self, x, return_features=False):
        features = self.model.forward_features(x)
        features = self.model.global_pool(features)
        features = features.flatten(1)
        logits = self.model.classifier(features)
        
        if return_features:
            return logits, features
        return logits
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
