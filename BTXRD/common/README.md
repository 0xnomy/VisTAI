# Common Utilities Module

Shared utilities for both classification and segmentation pipelines.

## Modules

### losses.py
Custom loss functions for both tasks:
- **FocalLoss**: Handles class imbalance (γ=2.0, class weighting)
- **DiceLoss**: Region-based segmentation loss
- **CombinedSegmentationLoss**: Dice (0.6) + BCE (0.4)
- **BoundaryLoss**: Sobel edge detection for boundary refinement

### metrics.py
Evaluation metrics:
- **Segmentation**: Dice score, IoU, pixel accuracy (per-sample averaging)
- **Classification**: Accuracy, confusion matrix, classification report

### transforms.py
Data augmentation pipelines:
- **Classification**: Strong augmentation (rotation, flip, color jitter, blur, CLAHE)
- **Segmentation**: Synchronized image+mask transforms
- **Denormalization**: For visualization (ImageNet stats)

### utils.py
Training utilities:
- **Reproducibility**: `set_seed()` for deterministic training
- **Model inspection**: `count_parameters()` for model size
- **Checkpoints**: `save_checkpoint()`, `load_checkpoint()`
- **Transfer learning**: `freeze_model()`, `unfreeze_model()`
- **Exponential Moving Average (EMA)**: Smooth model weights (decay=0.999)
- **AverageMeter**: Track metrics during training

### visualization.py
Visualization tools:
- **Training curves**: Loss/accuracy over epochs
- **Confusion matrix**: Heatmap with annotations
- **Segmentation**: 4-panel display (image, mask, prediction, overlay)
- **Classification samples**: Grid of predictions with labels

### config_loader.py
Configuration management:
- **YAML loader**: Load training configs with validation
- **Config class**: Dot notation access (`config.training.batch_size`)

## Usage

```python
from common.losses import FocalLoss, CombinedSegmentationLoss
from common.metrics import compute_dice_score, compute_classification_metrics
from common.transforms import get_classification_transforms
from common.utils import set_seed, freeze_model, EMA
from common.visualization import plot_training_curves
from common.config_loader import load_config

# Set reproducibility
set_seed(42)

# Load config
config = load_config('configs/training.yaml')
lr = config.training.optimizer.lr  # Dot notation

# Get transforms
train_transforms = get_classification_transforms(phase='train')

# Initialize loss
criterion = FocalLoss(gamma=2.0, class_weights=weights)

# Track metrics
from common.utils import AverageMeter
loss_meter = AverageMeter()
loss_meter.update(loss.item(), batch_size)

# EMA smoothing
ema = EMA(model, decay=0.999)
# After each optimizer step:
ema.update()

# Freeze/unfreeze for transfer learning
freeze_model(model)  # Freeze all layers
unfreeze_model(model)  # Unfreeze all

# Visualize results
plot_training_curves(history, save_path='curves.png')
```

## Design Principles

1. **Modularity**: Each utility is independent and reusable
2. **Consistency**: Same API patterns across modules
3. **Flexibility**: Configurable parameters with sensible defaults
4. **Documentation**: Comprehensive docstrings with examples
