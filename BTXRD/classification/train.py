"""
Classification Training
=======================
Train classification models (EfficientNet-B0 student or ConvNeXt teacher).

Usage:
    # Train student model (EfficientNet-B0)
    python train.py --model efficientnet_b0 --epochs 30

    # Train teacher model (ConvNeXt-Small)
    python train.py --model convnext_small --epochs 75 --two-phase
    
    # Train with custom settings
    python train.py --model efficientnet_b0 --epochs 50 --batch-size 16 --lr 0.0005
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# Project imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from classification.models import build_model, list_models
from classification.datasets.classification_dataset import create_classification_dataloaders
from common.losses import FocalLoss
from common.metrics import compute_classification_metrics
from common.utils import save_checkpoint, AverageMeter


# =============================================================================
# Training Functions
# =============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    """Train for one epoch."""
    model.train()
    
    epoch_loss = AverageMeter()
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Train')
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            logits = model(images)
            loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Accuracy
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        epoch_loss.update(loss.item(), labels.size(0))
        
        pbar.set_postfix({
            'loss': f'{epoch_loss.avg:.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return epoch_loss.avg, 100.0 * correct / total


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    
    val_loss = AverageMeter()
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Val'):
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            _, predicted = torch.max(logits, 1)
            
            val_loss.update(loss.item(), labels.size(0))
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    metrics = compute_classification_metrics(all_labels, all_predictions)
    metrics['loss'] = val_loss.avg
    
    return metrics


# =============================================================================
# Main Training Function
# =============================================================================

def train(
    model_name: str = 'efficientnet_b0',
    num_epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    freeze_epochs: int = 10,
    output_dir: str = None
):
    """
    Train a classification model with two-phase strategy for pretrained models.
    
    Phase 1 (if pretrained backbone exists):
        - Freeze backbone, train classifier head only
        - Duration: freeze_epochs (default 10)
        - Learning rate: learning_rate * 5 (higher for head training)
    
    Phase 2:
        - Unfreeze all layers, fine-tune end-to-end
        - Duration: remaining epochs
        - Learning rate: learning_rate (default 0.001)
    
    Args:
        model_name: Model name ('efficientnet_b0', 'convnext_small', etc.)
        num_epochs: Total number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate for Phase 2 (fine-tuning)
        freeze_epochs: Number of epochs to freeze backbone (Phase 1). Set to 0 to disable.
        output_dir: Output directory for checkpoints
    """
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load label encoding
    with open(project_root / 'label_encoding.json', 'r') as f:
        label_info = json.load(f)
    
    num_classes = label_info['num_classes']
    idx_to_label = {int(v): k for k, v in label_info['label_to_idx'].items()}
    
    # Data paths
    train_csv = project_root / 'augmented_classification_data/augmented_train.csv'
    val_csv = project_root / 'augmented_classification_data/augmented_val.csv'
    test_csv = project_root / 'augmented_classification_data/augmented_test.csv'
    
    # Create dataloaders
    print(f"\n📊 Loading data...")
    dataloaders = create_classification_dataloaders(
        str(train_csv), str(val_csv), str(test_csv),
        label_info['label_to_idx'],
        batch_size=batch_size,
        num_workers=4
    )
    
    # Build model
    print(f"\n🏗️  Building model: {model_name}")
    model = build_model(model_name, num_classes=num_classes, pretrained=True)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {num_params:,}")
    
    # Check if model has pretrained backbone capability
    has_pretrained_backbone = hasattr(model, 'freeze_backbone') and hasattr(model, 'unfreeze_backbone')
    
    # Two-phase training setup
    if has_pretrained_backbone and freeze_epochs > 0:
        print(f"\n🔒 Two-Phase Training Strategy:")
        print(f"   Phase 1: Freeze backbone, train head only ({freeze_epochs} epochs, LR={learning_rate * 5})")
        print(f"   Phase 2: Unfreeze all, fine-tune end-to-end ({num_epochs - freeze_epochs} epochs, LR={learning_rate})")
        model.freeze_backbone()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Trainable parameters (Phase 1): {trainable_params:,} / {num_params:,}")
    else:
        print(f"\n🔓 Single-Phase Training:")
        print(f"   Model has no pretrained backbone or freeze_epochs=0")
        print(f"   Training end-to-end from start")
        freeze_epochs = 0
    
    # Compute class weights
    import pandas as pd
    train_df = pd.read_csv(train_csv)
    train_labels = np.array([label_info['label_to_idx'][l] for l in train_df['labels']])
    class_counts = np.bincount(train_labels, minlength=num_classes)
    class_weights = np.sqrt(len(train_labels) / (num_classes * class_counts + 1e-6))
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"\n📈 Class distribution:")
    for i in range(num_classes):
        print(f"   {idx_to_label[i]}: {class_counts[i]} samples (weight: {class_weights[i]:.2f})")
    
    # Loss and scaler
    criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.1)
    scaler = GradScaler('cuda')
    
    # Optimizer (Phase 1 learning rate)
    phase1_lr = learning_rate * 5 if has_pretrained_backbone and freeze_epochs > 0 else learning_rate
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=phase1_lr,
        weight_decay=0.01
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Output directory
    if output_dir is None:
        output_dir = project_root / 'classification/outputs' / model_name
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output: {output_dir}")
    
    # Training state
    best_acc = 0.0
    patience_counter = 0
    early_stop_patience = 10
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting Training: {num_epochs} epochs")
    print(f"{'='*60}")
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # Phase transition: unfreeze backbone after freeze_epochs
        if has_pretrained_backbone and epoch == freeze_epochs + 1:
            print(f"\n{'='*60}")
            print(f"🔓 Phase 2: Unfreezing backbone for fine-tuning")
            print(f"{'='*60}")
            
            # Unfreeze backbone
            model.unfreeze_backbone()
            
            # Reconfigure optimizer with lower learning rate
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs - freeze_epochs, eta_min=1e-6
            )
            
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"   Trainable parameters (Phase 2): {trainable_params:,} / {num_params:,}")
            print(f"   New learning rate: {learning_rate}")
            print()
        
        train_loss, train_acc = train_one_epoch(
            model, dataloaders['train'], criterion, optimizer, device, scaler
        )
        
        val_metrics = validate(model, dataloaders['val'], criterion, device)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
        print(f"Val   - F1: {val_metrics['f1']:.4f}")
        
        # Save latest
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            {'val_accuracy': val_metrics['accuracy'], 'val_f1': val_metrics['f1']},
            output_dir / 'checkpoint_latest.pth'
        )
        
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'val_accuracy': val_metrics['accuracy'], 'val_f1': val_metrics['f1']},
                output_dir / 'checkpoint_best.pth'
            )
            print(f"✓ Best model saved! Acc: {val_metrics['accuracy']:.2f}%")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{early_stop_patience}")
        
        if patience_counter >= early_stop_patience:
            print(f"\n⚠️  Early stopping at epoch {epoch}")
            break
    
    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Training complete! Best Accuracy: {best_acc:.2f}%")
    return best_acc


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Classification Model')
    parser.add_argument('--model', type=str, default='efficientnet_b0',
                        help='Model name (efficientnet_b0, convnext_small, etc.)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for Phase 2 (fine-tuning)')
    parser.add_argument('--freeze-epochs', type=int, default=10,
                        help='Number of epochs to freeze backbone (Phase 1). Set to 0 to disable.')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for checkpoints')
    parser.add_argument('--list-models', action='store_true',
                        help='List available models')
    
    args = parser.parse_args()
    
    if args.list_models:
        list_models()
    else:
        train(
            model_name=args.model,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            freeze_epochs=args.freeze_epochs,
            output_dir=args.output_dir
        )
