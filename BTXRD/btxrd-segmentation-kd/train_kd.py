import os
import argparse
import yaml
import logging
import torch
from torch.utils.data import DataLoader

from models.teacher.swin_unet import SwinUNet
from models.student.segformer_b2 import SegFormerB2
from models.adapters import FeatureAdapterModule
from datasets.segmentation_dataset import SegmentationDataset
from losses.dice_bce import DiceBCELoss
from losses.kd_losses import KnowledgeDistillationLoss
from engine.trainer import KDTrainer
from engine.evaluator import Evaluator
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint, load_checkpoint

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_dataloaders(config):
    train_dataset = SegmentationDataset(
        csv_path=config['data']['train_csv'],
        image_size=config['data']['image_size'],
        augment=True,
        aug_config=config['augmentation']
    )
    
    val_dataset = SegmentationDataset(
        csv_path=config['data']['val_csv'],
        image_size=config['data']['image_size'],
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['phase1']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['phase1']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader


def train_phase(phase_name, config, phase_config, teacher, student, adapters, 
                train_loader, val_loader, device, best_dice, patience_counter):
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting {phase_name}")
    logger.info(f"{'='*60}")
    
    if phase_config['freeze_encoder']:
        student.freeze_encoder()
        logger.info("Encoder frozen")
    else:
        student.unfreeze_encoder()
        logger.info("Encoder unfrozen")
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, list(student.parameters()) + list(adapters.parameters())),
        lr=phase_config['lr'],
        weight_decay=config['training']['optimizer']['weight_decay'],
        betas=config['training']['optimizer']['betas']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=phase_config['epochs'],
        eta_min=config['training']['scheduler']['min_lr']
    )
    
    task_loss_fn = DiceBCELoss()
    kd_loss_fn = KnowledgeDistillationLoss(
        temperature=config['distillation']['temperature'],
        response_weight=config['distillation']['response_weight'],
        feature_weight=config['distillation']['feature_weight'],
        task_weight=config['distillation']['task_weight']
    )
    
    trainer = KDTrainer(
        teacher=teacher,
        student=student,
        adapters=adapters,
        optimizer=optimizer,
        scheduler=scheduler,
        kd_loss_fn=kd_loss_fn,
        task_loss_fn=task_loss_fn,
        device=device,
        amp=config['training']['amp']
    )
    
    evaluator = Evaluator(student, device)
    
    save_dir = config['training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(1, phase_config['epochs'] + 1):
        train_loss, train_metrics = trainer.train_epoch(train_loader, epoch)
        
        logger.info(f"\nEpoch {epoch}/{phase_config['epochs']}")
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"  Response: {train_metrics['response']:.4f}")
        logger.info(f"  Feature: {train_metrics['feature']:.4f}")
        logger.info(f"  Task: {train_metrics['task']:.4f}")
        
        if epoch % config['validation']['interval'] == 0:
            val_metrics = evaluator.evaluate(val_loader)
            logger.info(f"Val Dice: {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f}")
            
            if val_metrics['dice'] > best_dice:
                best_dice = val_metrics['dice']
                patience_counter = 0
                save_path = os.path.join(save_dir, 'best_model.pth')
                save_checkpoint(student, optimizer, scheduler, epoch, val_metrics, save_path)
                logger.info(f"✓ Saved best model (Dice: {best_dice:.4f})")
            else:
                patience_counter += 1
                logger.info(f"No improvement ({patience_counter}/{config['training']['early_stopping_patience']})")
            
            if patience_counter >= config['training']['early_stopping_patience']:
                logger.info("Early stopping triggered")
                break
        
        scheduler.step()
    
    return best_dice, patience_counter


def main():
    parser = argparse.ArgumentParser(description='Knowledge Distillation Training')
    parser.add_argument('--config', type=str, default='configs/kd_config.yaml')
    parser.add_argument('--teacher-checkpoint', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    config = load_config(args.config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    logger.info("Loading teacher model...")
    teacher = SwinUNet(
        num_classes=config['teacher']['num_classes'],
        img_size=config['teacher']['image_size'],
        deep_supervision=config['teacher']['deep_supervision'],
        pretrained=False
    )
    load_checkpoint(teacher, args.teacher_checkpoint)
    teacher = teacher.to(device)
    teacher.eval()
    logger.info("Teacher loaded and frozen")
    
    logger.info("Creating student model...")
    student = SegFormerB2(
        num_classes=config['student']['num_classes'],
        image_size=config['student']['image_size'],
        pretrained=config['student']['pretrained']
    )
    student = student.to(device)
    
    logger.info("Creating feature adapters...")
    teacher_channels = [256, 128, 64]  # Swin-UNet decoder channels
    student_channels = [128, 320, 512]  # SegFormer-B2 encoder stages 1,2,3 channels
    adapters = FeatureAdapterModule(student_channels, teacher_channels)
    adapters = adapters.to(device)
    
    logger.info("Loading datasets...")
    train_loader, val_loader = create_dataloaders(config)
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    best_dice = 0.0
    patience_counter = 0
    
    best_dice, patience_counter = train_phase(
        "Phase 1: Decoder Warm-up",
        config,
        config['training']['phase1'],
        teacher,
        student,
        adapters,
        train_loader,
        val_loader,
        device,
        best_dice,
        patience_counter
    )
    
    best_dice, patience_counter = train_phase(
        "Phase 2: Full Knowledge Distillation",
        config,
        config['training']['phase2'],
        teacher,
        student,
        adapters,
        train_loader,
        val_loader,
        device,
        best_dice,
        patience_counter
    )
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training Complete!")
    logger.info(f"Best Dice: {best_dice:.4f}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
