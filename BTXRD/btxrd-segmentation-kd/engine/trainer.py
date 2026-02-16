import torch
from tqdm import tqdm
from utils.metrics import calculate_metrics


class KDTrainer:
    def __init__(self, teacher, student, adapters, optimizer, scheduler, kd_loss_fn, task_loss_fn, device, amp=True):
        self.teacher = teacher
        self.student = student
        self.adapters = adapters
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.kd_loss_fn = kd_loss_fn
        self.task_loss_fn = task_loss_fn
        self.device = device
        self.amp = amp
        self.scaler = torch.cuda.amp.GradScaler() if amp else None
        
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def train_epoch(self, dataloader, epoch):
        self.student.train()
        total_loss = 0
        metrics_sum = {'response': 0, 'feature': 0, 'task': 0}
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.amp):
                with torch.no_grad():
                    teacher_logits, teacher_features = self.teacher(images, return_features=True)
                
                student_logits, student_features = self.student(images, return_features=True)
                
                adapted_student_features = self.adapters(student_features['decoder'])
                
                task_loss = self.task_loss_fn(student_logits, masks)
                
                loss, loss_dict = self.kd_loss_fn(
                    student_logits,
                    adapted_student_features,
                    teacher_logits,
                    teacher_features['decoder'],
                    task_loss
                )
            
            self.optimizer.zero_grad()
            if self.amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            for key in metrics_sum:
                metrics_sum[key] += loss_dict[key]
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        avg_metrics = {k: v / len(dataloader) for k, v in metrics_sum.items()}
        
        return avg_loss, avg_metrics
