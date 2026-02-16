import torch
from tqdm import tqdm
from utils.metrics import compute_metrics


class Evaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def evaluate(self, dataloader, num_classes):
        self.model.eval()
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc='Evaluating'):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                logits = self.model(images)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())
                all_probs.append(probs.cpu())
        
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_probs = torch.cat(all_probs)
        
        metrics = compute_metrics(all_preds, all_targets, all_probs, num_classes)
        
        return metrics
