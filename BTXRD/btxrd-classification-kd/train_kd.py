import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.teacher.efficientnet_b4 import EfficientNetB4Teacher
from models.student.convnext_tiny import ConvNeXtTinyStudent
from models.projection import ProjectionHead
from datasets.classification_dataset import ClassificationDataset, get_transforms
from losses.kd_loss import KnowledgeDistillationLoss
from losses.label_smoothing_ce import LabelSmoothingCrossEntropy
from engine.trainer import KDTrainer
from engine.evaluator import Evaluator
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import setup_logger
from utils.seed import set_seed


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_dataloaders(config):
    train_transform = get_transforms(config, 'train')
    val_transform = get_transforms(config, 'val')
    
    train_dataset = ClassificationDataset(
        config['dataset']['data_root'], 
        split='train', 
        transform=train_transform
    )
    
    val_dataset = ClassificationDataset(
        config['dataset']['data_root'], 
        split='val', 
        transform=val_transform,
        class_names=train_dataset.class_names
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=config['dataset']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        num_workers=config['dataset']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.class_names, train_dataset.class_weights


def train_phase(config, teacher, student, projection, train_loader, val_loader, 
                device, logger, phase_name, epochs, lr, freeze_backbone=False):
    
    if freeze_backbone:
        student.freeze_backbone()
        logger.info("Student backbone frozen")
    else:
        student.unfreeze_all()
        logger.info("Student fully unfrozen")
    
    class_weights = train_loader.dataset.class_weights.to(device)
    
    ce_loss_fn = LabelSmoothingCrossEntropy(
        smoothing=config['label_smoothing'],
        weight=class_weights
    )
    
    kd_loss_fn = KnowledgeDistillationLoss(
        temperature=config['kd_loss']['temperature'],
        alpha_kd=config['kd_loss']['alpha_kd'],
        alpha_feature=config['kd_loss']['alpha_feature'],
        alpha_ce=config['kd_loss']['alpha_ce']
    )
    
    params = [p for p in student.parameters() if p.requires_grad]
    params += list(projection.parameters())
    
    optimizer = torch.optim.AdamW(
        params,
        lr=lr,
        weight_decay=config['training']['weight_decay'],
        betas=config['optimizer']['betas'],
        eps=config['optimizer']['eps']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=config['scheduler']['min_lr']
    )
    
    trainer = KDTrainer(
        teacher, student, projection, kd_loss_fn, ce_loss_fn,
        optimizer, scheduler, device, amp=config['mixed_precision']
    )
    
    evaluator = Evaluator(student, device)
    
    best_metric = 0.0
    patience_counter = 0
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting {phase_name}")
    logger.info(f"{'='*60}")
    
    for epoch in range(1, epochs + 1):
        train_loss, loss_components = trainer.train_epoch(
            train_loader, epoch,
            cutmix_prob=config['augmentation']['train']['cutmix_prob'],
            mixup_prob=config['augmentation']['train']['mixup_prob'],
            mixup_alpha=config['augmentation']['train']['mixup_alpha']
        )
        
        logger.info(f"Epoch {epoch}/{epochs}")
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"  KD: {loss_components['kd_loss']:.4f}")
        logger.info(f"  Feature: {loss_components['feature_loss']:.4f}")
        logger.info(f"  CE: {loss_components['ce_loss']:.4f}")
        
        metrics = evaluator.evaluate(val_loader, config['dataset']['num_classes'])
        
        logger.info(f"Val Accuracy: {metrics['accuracy']:.4f} | "
                   f"Top-3: {metrics['top3_accuracy']:.4f}")
        logger.info(f"Val Macro F1: {metrics['macro_f1']:.4f} | "
                   f"Weighted F1: {metrics['weighted_f1']:.4f}")
        
        current_metric = metrics[config['save_best_metric']]
        
        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            
            save_checkpoint(
                student, optimizer, scheduler, epoch, metrics,
                os.path.join(config['output_dir'], 'best_model.pth')
            )
            logger.info(f"[BEST] Saved best model ({config['save_best_metric']}: {best_metric:.4f})")
        else:
            patience_counter += 1
            logger.info(f"No improvement ({patience_counter}/{config['training']['early_stopping_patience']})")
        
        save_checkpoint(
            student, optimizer, scheduler, epoch, metrics,
            os.path.join(config['output_dir'], 'latest_model.pth')
        )
        
        if patience_counter >= config['training']['early_stopping_patience']:
            logger.info("Early stopping triggered")
            break
        
        scheduler.step()
    
    return best_metric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/kd_config.yaml')
    parser.add_argument('--teacher-checkpoint', type=str, required=True)
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(config['seed'])
    
    os.makedirs(config['output_dir'], exist_ok=True)
    logger = setup_logger(os.path.join(config['output_dir'], 'training.log'))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    logger.info("Loading teacher model...")
    teacher = EfficientNetB4Teacher(
        num_classes=config['dataset']['num_classes'],
        pretrained=config['teacher']['pretrained']
    )
    load_checkpoint(teacher, args.teacher_checkpoint, map_location=device)
    teacher = teacher.to(device)
    teacher.freeze()
    logger.info(f"Teacher loaded and frozen (feature dim: {teacher.feature_dim})")
    
    logger.info("Creating student model...")
    student = ConvNeXtTinyStudent(
        num_classes=config['dataset']['num_classes'],
        pretrained=config['student']['pretrained']
    )
    student = student.to(device)
    logger.info(f"Student created (feature dim: {student.feature_dim})")
    
    logger.info("Creating projection head...")
    projection = ProjectionHead(
        student_dim=config['projection']['student_dim'],
        teacher_dim=config['projection']['teacher_dim'],
        hidden_dim=config['projection']['hidden_dim']
    )
    projection = projection.to(device)
    logger.info("Projection head created")
    
    logger.info("Loading datasets...")
    train_loader, val_loader, class_names, class_weights = create_dataloaders(config)
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info(f"Classes: {class_names}")
    
    best_phase1 = train_phase(
        config, teacher, student, projection, train_loader, val_loader,
        device, logger, "Phase 1: Decoder Warm-up",
        config['training']['phase1_epochs'],
        config['training']['phase1_lr'],
        freeze_backbone=True
    )
    
    logger.info(f"\nPhase 1 complete. Best {config['save_best_metric']}: {best_phase1:.4f}")
    
    best_phase2 = train_phase(
        config, teacher, student, projection, train_loader, val_loader,
        device, logger, "Phase 2: Full Fine-Tuning",
        config['training']['phase2_epochs'],
        config['training']['phase2_lr'],
        freeze_backbone=False
    )
    
    logger.info(f"\n{'='*60}")
    logger.info("Training Complete!")
    logger.info(f"Phase 1 Best {config['save_best_metric']}: {best_phase1:.4f}")
    logger.info(f"Phase 2 Best {config['save_best_metric']}: {best_phase2:.4f}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
