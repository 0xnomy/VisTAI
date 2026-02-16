import torch
from tqdm import tqdm
from utils.metrics import calculate_metrics


class Evaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def evaluate(self, dataloader):
        self.model.eval()
        all_dice = []
        all_iou = []
        all_pixel_acc = []
        all_sensitivity = []
        all_specificity = []
        
        with torch.no_grad():
            for images, masks in tqdm(dataloader, desc='Evaluating'):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                logits = self.model(images)
                preds = torch.sigmoid(logits) > 0.5
                
                metrics = calculate_metrics(preds, masks)
                all_dice.append(metrics['dice'])
                all_iou.append(metrics['iou'])
                all_pixel_acc.append(metrics['pixel_acc'])
                all_sensitivity.append(metrics['sensitivity'])
                all_specificity.append(metrics['specificity'])
        
        return {
            'dice': sum(all_dice) / len(all_dice),
            'iou': sum(all_iou) / len(all_iou),
            'pixel_acc': sum(all_pixel_acc) / len(all_pixel_acc),
            'sensitivity': sum(all_sensitivity) / len(all_sensitivity),
            'specificity': sum(all_specificity) / len(all_specificity)
        }
