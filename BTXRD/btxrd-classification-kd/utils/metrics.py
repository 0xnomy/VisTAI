import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix


def compute_metrics(preds, targets, probs, num_classes):
    preds_np = preds.numpy()
    targets_np = targets.numpy()
    probs_np = probs.numpy()
    
    accuracy = accuracy_score(targets_np, preds_np)
    
    top3_accuracy = top_k_accuracy(probs_np, targets_np, k=3)
    
    macro_f1 = f1_score(targets_np, preds_np, average='macro', zero_division=0)
    weighted_f1 = f1_score(targets_np, preds_np, average='weighted', zero_division=0)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        targets_np, preds_np, average=None, zero_division=0
    )
    
    cm = confusion_matrix(targets_np, preds_np, labels=list(range(num_classes)))
    
    metrics = {
        'accuracy': accuracy,
        'top3_accuracy': top3_accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist(),
        'per_class_f1': f1.tolist(),
        'per_class_support': support.tolist(),
        'confusion_matrix': cm.tolist()
    }
    
    return metrics


def top_k_accuracy(probs, targets, k=3):
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
    correct = np.array([targets[i] in top_k_preds[i] for i in range(len(targets))])
    return correct.mean()
