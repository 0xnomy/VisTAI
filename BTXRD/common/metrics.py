"""Evaluation metrics for classification and segmentation."""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def compute_dice_score(pred_mask, true_mask, threshold=0.5, smooth=1e-6):
    """
    Compute Dice coefficient for binary segmentation.
    
    Args:
        pred_mask: Predicted mask logits or probabilities [B, 1, H, W] or [1, H, W]
        true_mask: Ground truth binary mask [B, 1, H, W] or [1, H, W]
        threshold: Threshold for binarization
        smooth: Smoothing constant
    
    Returns:
        Dice score (float)
    """
    # Apply sigmoid if logits
    if pred_mask.min() < 0 or pred_mask.max() > 1:
        pred_mask = torch.sigmoid(pred_mask)
    
    pred_binary = (pred_mask > threshold).float()
    
    intersection = (pred_binary * true_mask).sum()
    dice = (2.0 * intersection + smooth) / (
        pred_binary.sum() + true_mask.sum() + smooth
    )
    
    return dice.item()


def compute_iou(pred_mask, true_mask, threshold=0.5, smooth=1e-6):
    """
    Compute Intersection over Union (IoU) for binary segmentation.
    
    Args:
        pred_mask: Predicted mask logits or probabilities [B, 1, H, W] or [1, H, W]
        true_mask: Ground truth binary mask [B, 1, H, W] or [1, H, W]
        threshold: Threshold for binarization
        smooth: Smoothing constant
    
    Returns:
        IoU score (float)
    """
    # Apply sigmoid if logits
    if pred_mask.min() < 0 or pred_mask.max() > 1:
        pred_mask = torch.sigmoid(pred_mask)
    
    pred_binary = (pred_mask > threshold).float()
    
    intersection = (pred_binary * true_mask).sum()
    union = pred_binary.sum() + true_mask.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


def compute_pixel_accuracy(pred_mask, true_mask, threshold=0.5):
    """
    Compute pixel-wise accuracy for binary segmentation.
    
    Args:
        pred_mask: Predicted mask logits or probabilities [B, 1, H, W] or [1, H, W]
        true_mask: Ground truth binary mask [B, 1, H, W] or [1, H, W]
        threshold: Threshold for binarization
    
    Returns:
        Pixel accuracy (float)
    """
    # Apply sigmoid if logits
    if pred_mask.min() < 0 or pred_mask.max() > 1:
        pred_mask = torch.sigmoid(pred_mask)
    
    pred_binary = (pred_mask > threshold).float()
    
    correct = (pred_binary == true_mask).sum()
    total = true_mask.numel()
    accuracy = correct.float() / total
    
    return accuracy.item()


def compute_classification_metrics(y_true, y_pred, class_names=None):
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels (numpy array or list)
        y_pred: Predicted labels (numpy array or list)
        class_names: List of class names (optional)
    
    Returns:
        Dictionary with metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Extract macro averaged metrics
    f1 = report_dict.get('macro avg', {}).get('f1-score', 0.0)
    precision = report_dict.get('macro avg', {}).get('precision', 0.0)
    recall = report_dict.get('macro avg', {}).get('recall', 0.0)
    
    return {
        'accuracy': accuracy * 100.0,  # Convert to percentage
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm,
        'classification_report': report_dict
    }


def compute_segmentation_metrics_batch(pred_masks, true_masks, threshold=0.5):
    """
    Compute segmentation metrics for a batch (per-sample average).
    
    Args:
        pred_masks: Predicted masks [B, 1, H, W]
        true_masks: Ground truth masks [B, 1, H, W]
        threshold: Binarization threshold
    
    Returns:
        Dictionary with mean metrics
    """
    batch_size = pred_masks.size(0)
    
    dice_scores = []
    iou_scores = []
    pixel_accs = []
    
    for i in range(batch_size):
        dice = compute_dice_score(pred_masks[i:i+1], true_masks[i:i+1], threshold)
        iou = compute_iou(pred_masks[i:i+1], true_masks[i:i+1], threshold)
        pix_acc = compute_pixel_accuracy(pred_masks[i:i+1], true_masks[i:i+1], threshold)
        
        dice_scores.append(dice)
        iou_scores.append(iou)
        pixel_accs.append(pix_acc)
    
    return {
        'dice': np.mean(dice_scores),
        'iou': np.mean(iou_scores),
        'pixel_accuracy': np.mean(pixel_accs)
    }
