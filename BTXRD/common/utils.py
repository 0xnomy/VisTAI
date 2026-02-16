"""Utility functions for model training and evaluation."""

import torch
import random
import numpy as np
from pathlib import Path
import json


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path, is_best=False):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        metrics: Dictionary of metrics
        save_path: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.parent / f'best_{save_path.name}'
        torch.save(checkpoint, best_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cuda'):
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        model: PyTorch model
        optimizer: Optimizer (optional)
        scheduler: Learning rate scheduler (optional)
        device: Device to load to
    
    Returns:
        Dictionary with epoch and metrics
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {})
    }


def freeze_model(model):
    """
    Freeze all parameters in a model.
    
    Args:
        model: PyTorch model or module
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model):
    """
    Unfreeze all parameters in a model.
    
    Args:
        model: PyTorch model or module
    """
    for param in model.parameters():
        param.requires_grad = True


def get_learning_rate(optimizer):
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: PyTorch optimizer
    
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_metrics_to_json(metrics, save_path):
    """
    Save metrics dictionary to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save JSON
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python native types
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    metrics_serializable = convert_to_serializable(metrics)
    
    with open(save_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)


class EMA:
    """Exponential Moving Average for model weights."""
    
    def __init__(self, model, decay=0.999):
        """
        Args:
            model: PyTorch model
            decay: Decay rate for moving average
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Register parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow weights with current model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        """Apply shadow weights to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
