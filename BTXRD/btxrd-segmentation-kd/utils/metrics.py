import torch


def calculate_metrics(pred, target):
    pred = pred.float()
    target = target.float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2. * intersection + 1e-7) / (union + 1e-7)
    iou = (intersection + 1e-7) / (pred.sum() + target.sum() - intersection + 1e-7)
    
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    correct = (pred_flat == target_flat).sum()
    pixel_acc = correct.float() / target_flat.numel()
    
    tp = (pred_flat * target_flat).sum()
    fn = ((1 - pred_flat) * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum()
    
    sensitivity = (tp + 1e-7) / (tp + fn + 1e-7)
    specificity = (tn + 1e-7) / (tn + fp + 1e-7)
    
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'pixel_acc': pixel_acc.item(),
        'sensitivity': sensitivity.item(),
        'specificity': specificity.item()
    }
