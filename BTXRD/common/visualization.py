"""Visualization utilities for training results and predictions."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import torch


def plot_training_curves(history, save_path=None, title='Training Curves'):
    """
    Plot training and validation curves.
    
    Args:
        history: Dictionary with training history
                 e.g., {'train_loss': [...], 'val_loss': [...], ...}
        save_path: Path to save plot (optional)
        title: Plot title
    """
    metrics = list(history.keys())
    n_metrics = len([m for m in metrics if 'train' in m])
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]
    
    epochs = range(1, len(history[metrics[0]]) + 1)
    
    idx = 0
    for metric_name in metrics:
        if 'train' in metric_name:
            val_metric = metric_name.replace('train', 'val')
            
            axes[idx].plot(epochs, history[metric_name], label=f'Train', marker='o', markersize=3)
            if val_metric in history:
                axes[idx].plot(epochs, history[val_metric], label=f'Val', marker='s', markersize=3)
            
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric_name.replace('train_', '').replace('_', ' ').title())
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
            idx += 1
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Training curves saved to {save_path}")
    
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path=None, title='Confusion Matrix'):
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix array [num_classes, num_classes]
        class_names: List of class names
        save_path: Path to save plot (optional)
        title: Plot title
    """
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
    
    plt.close()


def visualize_segmentation(image, true_mask, pred_mask, save_path=None, 
                          title='Segmentation Result'):
    """
    Visualize segmentation result.
    
    Args:
        image: Input image tensor [3, H, W] (denormalized, 0-1 range)
        true_mask: Ground truth mask [1, H, W] or [H, W]
        pred_mask: Predicted mask [1, H, W] or [H, W]
        save_path: Path to save visualization (optional)
        title: Plot title
    """
    # Convert tensors to numpy
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.cpu().numpy()
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    
    # Handle dimensions
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    
    if true_mask.ndim == 3:
        true_mask = true_mask[0]
    if pred_mask.ndim == 3:
        pred_mask = pred_mask[0]
    
    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Input Image', fontsize=11)
    axes[0].axis('off')
    
    # Ground truth mask
    axes[1].imshow(image)
    axes[1].imshow(true_mask, alpha=0.5, cmap='Reds')
    axes[1].set_title('Ground Truth Mask', fontsize=11)
    axes[1].axis('off')
    
    # Predicted mask
    axes[2].imshow(image)
    axes[2].imshow(pred_mask, alpha=0.5, cmap='Blues')
    axes[2].set_title('Predicted Mask', fontsize=11)
    axes[2].axis('off')
    
    # Overlay (both masks)
    axes[3].imshow(image)
    axes[3].imshow(true_mask, alpha=0.3, cmap='Reds', label='Ground Truth')
    axes[3].imshow(pred_mask, alpha=0.3, cmap='Blues', label='Prediction')
    axes[3].set_title('Overlay', fontsize=11)
    axes[3].axis('off')
    
    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    
    plt.close()


def plot_classification_samples(images, labels, predictions, class_names, 
                                save_path=None, num_samples=16):
    """
    Visualize classification predictions in a grid.
    
    Args:
        images: List or tensor of images [N, 3, H, W]
        labels: List or tensor of true labels [N]
        predictions: List or tensor of predicted labels [N]
        class_names: List of class names
        save_path: Path to save visualization (optional)
        num_samples: Number of samples to display
    """
    num_samples = min(num_samples, len(images))
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for idx in range(num_samples):
        img = images[idx]
        true_label = labels[idx]
        pred_label = predictions[idx]
        
        # Convert tensor to numpy
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        # Transpose if needed
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        
        # Display
        axes[idx].imshow(img)
        
        # Title with color coding
        is_correct = true_label == pred_label
        color = 'green' if is_correct else 'red'
        
        title = f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}"
        axes[idx].set_title(title, fontsize=9, color=color)
        axes[idx].axis('off')
    
    # Hide extra subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Classification Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"✓ Classification samples saved to {save_path}")
    
    plt.close()
