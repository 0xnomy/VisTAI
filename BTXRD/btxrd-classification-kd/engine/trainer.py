import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import random


class KDTrainer:
    def __init__(self, teacher, student, projection, kd_loss_fn, ce_loss_fn, 
                 optimizer, scheduler, device, amp=True):
        self.teacher = teacher
        self.student = student
        self.projection = projection
        self.kd_loss_fn = kd_loss_fn
        self.ce_loss_fn = ce_loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.amp = amp
        self.scaler = torch.amp.GradScaler('cuda') if amp else None
        
        self.teacher.eval()
        self.teacher.freeze()
    
    def train_epoch(self, dataloader, epoch, cutmix_prob=0.5, mixup_prob=0.5, mixup_alpha=0.2):
        self.student.train()
        total_loss = 0
        loss_components = {'kd_loss': 0, 'feature_loss': 0, 'ce_loss': 0}
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            r = random.random()
            if r < cutmix_prob:
                images, targets_a, targets_b, lam = cutmix(images, targets)
                mixed = True
            elif r < cutmix_prob + mixup_prob:
                images, targets_a, targets_b, lam = mixup(images, targets, mixup_alpha)
                mixed = True
            else:
                mixed = False
            
            self.optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=self.amp):
                with torch.no_grad():
                    teacher_logits, teacher_features = self.teacher(images, return_features=True)
                
                student_logits, student_features = self.student(images, return_features=True)
                student_features_proj = self.projection(student_features)
                
                if mixed:
                    ce_loss = lam * self.ce_loss_fn(student_logits, targets_a) + \
                             (1 - lam) * self.ce_loss_fn(student_logits, targets_b)
                else:
                    ce_loss = self.ce_loss_fn(student_logits, targets)
                
                loss, loss_dict = self.kd_loss_fn(
                    student_logits, teacher_logits,
                    student_features_proj, teacher_features,
                    targets, ce_loss
                )
            
            if self.amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            for k, v in loss_dict.items():
                if k != 'total_loss':
                    loss_components[k] += v
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        avg_components = {k: v / len(dataloader) for k, v in loss_components.items()}
        
        return avg_loss, avg_components


def cutmix(images, targets, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size).to(images.device)
    
    _, _, h, w = images.size()
    cx = np.random.uniform(0, w)
    cy = np.random.uniform(0, h)
    cut_w = w * np.sqrt(1 - lam)
    cut_h = h * np.sqrt(1 - lam)
    
    x1 = int(np.clip(cx - cut_w / 2, 0, w))
    x2 = int(np.clip(cx + cut_w / 2, 0, w))
    y1 = int(np.clip(cy - cut_h / 2, 0, h))
    y2 = int(np.clip(cy + cut_h / 2, 0, h))
    
    images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
    
    return images, targets, targets[index], lam


def mixup(images, targets, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size).to(images.device)
    
    mixed_images = lam * images + (1 - lam) * images[index]
    
    return mixed_images, targets, targets[index], lam
