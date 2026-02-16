import torch
import torch.nn as nn


class FeatureAdapter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        return self.adapter(x)


class FeatureAdapterModule(nn.Module):
    def __init__(self, student_channels, teacher_channels):
        super().__init__()
        self.adapters = nn.ModuleList([
            FeatureAdapter(s_ch, t_ch) 
            for s_ch, t_ch in zip(student_channels, teacher_channels)
        ])
    
    def forward(self, student_features):
        adapted_features = []
        for feat, adapter in zip(student_features, self.adapters):
            adapted_features.append(adapter(feat))
        return adapted_features
