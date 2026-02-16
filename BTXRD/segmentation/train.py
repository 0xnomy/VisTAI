"""
Segmentation Training
=====================
Train segmentation models (nnU-Net student or teacher models).

Usage:
    # Train student model (nnU-Net)
    python train.py --model nnunet --epochs 100

    # Train teacher model (UNet++ with ResNet50)
    python train.py --model unetplusplus_resnet50 --epochs 100
    
    # Train with custom settings
    python train.py --model nnunet --epochs 150 --batch-size 4 --lr 0.0005
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

# Project imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from segmentation.models import build_model, list_models
from segmentation.datasets.segmentation_dataset import create_segmentation_dataloaders
from common.utils import save_checkpoint, AverageMeter
from common.metrics import compute_dice_score, compute_iou


# =============================================================================
# Loss Function
# =============================================================================

class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss with deep supervision support."""
    
    def __init__(self, dice_weight: float = 0.7, bce_weight: float = 0.3, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
    
    def _compute_loss(self, pred_logits, target_mask):
        """Compute Dice + BCE loss for single output."""
        # BCE loss
        bce_loss = self.bce(pred_logits, target_mask)
        
        # Dice loss
        pred_sigmoid = torch.sigmoid(pred_logits)
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target_mask.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        dice_loss = 1.0 - dice_score
        
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss
    
    def forward(self, outputs, target_mask):
        """Handle deep supervision outputs."""
        if isinstance(outputs, tuple):
            # outputs = (main_output, [aux1, aux2, aux3])
            main_output = outputs[0]
            aux_outputs = outputs[1] if isinstance(outputs[1], list) else outputs[1:]
            
            # Main output loss (highest weight)
            main_loss = self._compute_loss(main_output, target_mask)
            
            # Deep supervision losses
            ds_weights = [0.15, 0.15, 0.1]  # Weights for aux outputs
            ds_loss = 0
            
            for ds_out, weight in zip(aux_outputs, ds_weights):
                target_resized = F.interpolate(target_mask, size=ds_out.shape[2:], 
                                               mode='bilinear', align_corners=False)
                ds_loss += weight * self._compute_loss(ds_out, target_resized)
            
            return 0.6 * main_loss + 0.4 * ds_loss
        
        return self._compute_loss(outputs, target_mask)


# =============================================================================
# Training Functions
# =============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Train for one epoch."""
    model.train()
    
    epoch_loss = AverageMeter()
    epoch_dice = AverageMeter()
    epoch_iou = AverageMeter()
    
    pbar = tqdm(dataloader, desc='Train')
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Extract main output for metrics
        pred_logits = outputs[0] if isinstance(outputs, tuple) else outputs
        
        with torch.no_grad():
            pred_sigmoid = torch.sigmoid(pred_logits)
            pred_binary = (pred_sigmoid > 0.5).float()
            
            dice = compute_dice_score(pred_binary, masks)
            iou = compute_iou(pred_binary, masks)
            
            epoch_loss.update(loss.item(), images.size(0))
            epoch_dice.update(dice, images.size(0))
            epoch_iou.update(iou, images.size(0))
        
        pbar.set_postfix({
            'loss': f'{epoch_loss.avg:.4f}',
            'dice': f'{epoch_dice.avg:.4f}',
            'iou': f'{epoch_iou.avg:.4f}'
        })
    
    return epoch_loss.avg, epoch_dice.avg, epoch_iou.avg


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    
    val_loss = AverageMeter()
    val_dice = AverageMeter()
    val_iou = AverageMeter()
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Val'):
            images = images.to(device)
            masks = masks.to(device)
            
            pred_logits = model(images)
            loss = criterion(pred_logits, masks)
            
            pred_sigmoid = torch.sigmoid(pred_logits)
            pred_binary = (pred_sigmoid > 0.5).float()
            
            dice = compute_dice_score(pred_binary, masks)
            iou = compute_iou(pred_binary, masks)
            
            val_loss.update(loss.item(), images.size(0))
            val_dice.update(dice, images.size(0))
            val_iou.update(iou, images.size(0))
    
    return val_loss.avg, val_dice.avg, val_iou.avg


# =============================================================================
# Main Training Function
# =============================================================================

def train(
    model_name: str = 'deeplabv3plus_resnet101',
    num_epochs: int = 100,
    batch_size: int = 6,
    learning_rate: float = 0.001,
    image_size: int = 320,
    output_dir: str = None,
    freeze_epochs: int = 12
):
    """
    Train a segmentation model with two-phase strategy for pretrained models.
    
    Phase 1 (if pretrained encoder exists):
        - Freeze encoder, train decoder only
        - Duration: freeze_epochs (default 12)
        - Learning rate: learning_rate (default 0.001)
    
    Phase 2:
        - Unfreeze all layers, fine-tune end-to-end
        - Duration: remaining epochs
        - Learning rate: learning_rate / 10 (default 0.0001)
    
    Args:
        model_name: Model name ('nnunet', 'deeplabv3plus_resnet101', etc.)
        num_epochs: Total number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate for Phase 1
        image_size: Input image size
        output_dir: Output directory for checkpoints
        freeze_epochs: Number of epochs to freeze encoder (Phase 1)
    """
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Data paths (CSVs are in segmentation folder)
    seg_dir = project_root / 'segmentation'
    train_csv = seg_dir / 'segmentation_train.csv'
    val_csv = seg_dir / 'segmentation_val.csv'
    test_csv = seg_dir / 'segmentation_test.csv'
    
    # Create dataloaders
    print(f"\n📊 Loading data...")
    dataloaders = create_segmentation_dataloaders(
        str(train_csv), str(val_csv), str(test_csv),
        batch_size=batch_size,
        image_size=image_size
    )
    
    # Build model
    print(f"\n🏗️  Building model: {model_name}")
    
    # Enable deep supervision for nnunet
    model_kwargs = {}
    if model_name == 'nnunet':
        model_kwargs['deep_supervision'] = True
    
    model = build_model(model_name, pretrained=True, **model_kwargs)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {num_params:,}")
    
    # Check if model has pretrained encoder capability
    has_pretrained_encoder = hasattr(model, 'freeze_encoder') and hasattr(model, 'unfreeze_encoder')
    
    # Two-phase training setup
    if has_pretrained_encoder and freeze_epochs > 0:
        print(f"\n🔒 Two-Phase Training Strategy:")
        print(f"   Phase 1: Freeze encoder, train decoder only ({freeze_epochs} epochs, LR={learning_rate})")
        print(f"   Phase 2: Unfreeze all, fine-tune end-to-end ({num_epochs - freeze_epochs} epochs, LR={learning_rate/10})")
        model.freeze_encoder()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Trainable parameters (Phase 1): {trainable_params:,} / {num_params:,}")
    else:
        print(f"\n🔓 Single-Phase Training:")
        print(f"   Model has no pretrained encoder or freeze_epochs=0")
        print(f"   Training end-to-end from start")
        freeze_epochs = 0
    
    # Loss and scaler (Dice-focused for medical segmentation)
    criterion = DiceBCELoss(dice_weight=0.7, bce_weight=0.3)
    scaler = GradScaler('cuda')
    
    # Optimizer and scheduler (Phase 1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-7
    )
    
    # Output directory
    if output_dir is None:
        output_dir = project_root / 'segmentation/outputs' / model_name
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output: {output_dir}")
    
    # Training state
    best_dice = 0.0
    patience_counter = 0
    early_stop_patience = 22  # Increased from 15 to 22
    history = {'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice': []}
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting Training: {num_epochs} epochs")
    print(f"{'='*60}")
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # Phase transition: unfreeze encoder after freeze_epochs
        if has_pretrained_encoder and epoch == freeze_epochs + 1:
            print(f"\n{'='*60}")
            print(f"🔓 Phase 2: Unfreezing encoder for fine-tuning")
            print(f"{'='*60}")
            
            # Unfreeze encoder
            model.unfreeze_encoder()
            
            # Reconfigure optimizer with lower learning rate
            new_lr = learning_rate / 10
            optimizer = optim.AdamW(model.parameters(), lr=new_lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-7
            )
            
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"   Trainable parameters (Phase 2): {trainable_params:,} / {num_params:,}")
            print(f"   New learning rate: {new_lr}")
            print()
        
        train_loss, train_dice, train_iou = train_one_epoch(
            model, dataloaders['train'], criterion, optimizer, scaler, device
        )
        
        val_loss, val_dice, val_iou = validate(
            model, dataloaders['val'], criterion, device
        )
        
        scheduler.step(val_dice)
        
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        
        # Save latest
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            {'val_dice': val_dice, 'val_iou': val_iou},
            output_dir / 'checkpoint_latest.pth'
        )
        
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            best_checkpoint_path = output_dir / 'checkpoint_best.pth'
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'val_dice': val_dice, 'val_iou': val_iou},
                best_checkpoint_path
            )
            # Verify checkpoint was saved
            if best_checkpoint_path.exists():
                size_mb = best_checkpoint_path.stat().st_size / (1024**2)
                print(f"✓ Best model saved! Dice: {val_dice:.4f} ({size_mb:.1f} MB)")
            else:
                print(f"⚠️  WARNING: Checkpoint save failed at {best_checkpoint_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{early_stop_patience}")
        
        if patience_counter >= early_stop_patience:
            print(f"\n⚠️  Early stopping at epoch {epoch}")
            break
    
    # Save history
    history_path = output_dir / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Final verification of saved checkpoints
    print(f"\n{'='*60}")
    print(f"Training Complete Summary")
    print(f"{'='*60}")
    print(f"Best Validation Dice: {best_dice:.4f}")
    
    best_checkpoint = output_dir / 'checkpoint_best.pth'
    latest_checkpoint = output_dir / 'checkpoint_latest.pth'
    
    if best_checkpoint.exists():
        size_mb = best_checkpoint.stat().st_size / (1024**2)
        print(f"✓ Best checkpoint saved: {best_checkpoint} ({size_mb:.1f} MB)")
    else:
        print(f"✗ Best checkpoint NOT found: {best_checkpoint}")
    
    if latest_checkpoint.exists():
        size_mb = latest_checkpoint.stat().st_size / (1024**2)
        print(f"✓ Latest checkpoint saved: {latest_checkpoint} ({size_mb:.1f} MB)")
    else:
        print(f"✗ Latest checkpoint NOT found: {latest_checkpoint}")
    
    if history_path.exists():
        print(f"✓ Training history saved: {history_path}")
    else:
        print(f"✗ Training history NOT found: {history_path}")
    
    print(f"{'='*60}\n")
    
    return best_dice


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Segmentation Model')
    parser.add_argument('--model', type=str, default='deeplabv3plus_resnet101',
                        help='Model name (deeplabv3plus_resnet101, unetplusplus_resnet50, etc.)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=6,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate (Phase 1 for pretrained models)')
    parser.add_argument('--image-size', type=int, default=320,
                        help='Input image size')
    parser.add_argument('--freeze-epochs', type=int, default=12,
                        help='Number of epochs to freeze encoder (Phase 1). Set to 0 to disable.')
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
            image_size=args.image_size,
            output_dir=args.output_dir,
            freeze_epochs=args.freeze_epochs
        )
