import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=5.0, alpha_kd=0.5, alpha_feature=0.1, alpha_ce=0.4):
        super().__init__()
        self.temperature = temperature
        self.alpha_kd = alpha_kd
        self.alpha_feature = alpha_feature
        self.alpha_ce = alpha_ce
    
    def forward(self, student_logits, teacher_logits, student_features, teacher_features, 
                targets, ce_loss):
        T = self.temperature
        
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction='batchmean'
        ) * (T * T)
        
        student_features_norm = F.normalize(student_features, p=2, dim=1)
        teacher_features_norm = F.normalize(teacher_features, p=2, dim=1)
        feature_loss = F.mse_loss(student_features_norm, teacher_features_norm)
        
        total_loss = (self.alpha_kd * kd_loss + 
                     self.alpha_feature * feature_loss + 
                     self.alpha_ce * ce_loss)
        
        loss_dict = {
            'kd_loss': kd_loss.item(),
            'feature_loss': feature_loss.item(),
            'ce_loss': ce_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
