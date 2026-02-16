import torch
import torch.nn as nn
import torch.nn.functional as F


class ResponseDistillationLoss(nn.Module):
    def __init__(self, temperature=4.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, student_logits, teacher_logits):
        # Use BCE with logits for AMP compatibility
        student_scaled = student_logits / self.temperature
        teacher_scaled = teacher_logits / self.temperature
        loss = F.binary_cross_entropy_with_logits(
            student_scaled, 
            torch.sigmoid(teacher_scaled).detach()
        )
        return loss * (self.temperature ** 2)


class FeatureDistillationLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, student_features, teacher_features):
        total_loss = 0
        for s_feat, t_feat in zip(student_features, teacher_features):
            if s_feat.shape != t_feat.shape:
                s_feat = F.interpolate(s_feat, size=t_feat.shape[2:], mode='bilinear', align_corners=False)
            s_feat_norm = F.normalize(s_feat, p=2, dim=1)
            t_feat_norm = F.normalize(t_feat.detach(), p=2, dim=1)
            loss = F.mse_loss(s_feat_norm, t_feat_norm)
            total_loss += loss
        return total_loss / len(student_features)


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, response_weight=0.3, feature_weight=0.3, task_weight=0.4):
        super().__init__()
        self.response_weight = response_weight
        self.feature_weight = feature_weight
        self.task_weight = task_weight
        
        self.response_loss = ResponseDistillationLoss(temperature)
        self.feature_loss = FeatureDistillationLoss()
    
    def forward(self, student_logits, student_features, teacher_logits, teacher_features, task_loss):
        response_loss = self.response_loss(student_logits, teacher_logits)
        feature_loss = self.feature_loss(student_features, teacher_features)
        
        total_loss = (
            self.response_weight * response_loss +
            self.feature_weight * feature_loss +
            self.task_weight * task_loss
        )
        
        return total_loss, {
            'response': response_loss.item(),
            'feature': feature_loss.item(),
            'task': task_loss.item(),
            'total': total_loss.item()
        }
