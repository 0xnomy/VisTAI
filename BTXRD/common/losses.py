"""Loss functions for classification and segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in classification.
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection"
    """
    
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        """
        Args:
            alpha: Class weights tensor [num_classes]
            gamma: Focusing parameter (default: 2.0)
            label_smoothing: Label smoothing factor (default: 0.0)
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.ce = nn.CrossEntropyLoss(weight=alpha, label_smoothing=label_smoothing, reduction='none')
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, num_classes] logits
            targets: [B] class indices
        """
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """Dice Loss for segmentation."""
    
    def __init__(self, smooth=1.0):
        """
        Args:
            smooth: Smoothing constant to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, 1, H, W] logits
            targets: [B, 1, H, W] binary masks
        """
        inputs = torch.sigmoid(inputs)
        
        # Flatten spatial dimensions
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        
        intersection = (inputs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            inputs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        
        return 1 - dice.mean()


class CombinedSegmentationLoss(nn.Module):
    """
    Combined Dice + BCE loss for segmentation.
    Provides both region-based (Dice) and pixel-based (BCE) supervision.
    """
    
    def __init__(self, dice_weight=0.6, bce_weight=0.4, smooth=1.0):
        """
        Args:
            dice_weight: Weight for Dice loss
            bce_weight: Weight for BCE loss
            smooth: Smoothing constant for Dice
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss(smooth=smooth)
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, 1, H, W] logits
            targets: [B, 1, H, W] binary masks
        """
        dice_loss = self.dice(inputs, targets)
        bce_loss = self.bce(inputs, targets)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss using Sobel edge detection.
    Helps improve segmentation at tumor boundaries.
    """
    
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        
        # Sobel kernels for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, 1, H, W] predicted logits
            targets: [B, 1, H, W] ground truth masks
        """
        inputs = torch.sigmoid(inputs)
        
        # Compute edges
        pred_edges_x = F.conv2d(inputs, self.sobel_x, padding=1)
        pred_edges_y = F.conv2d(inputs, self.sobel_y, padding=1)
        pred_edges = torch.sqrt(pred_edges_x ** 2 + pred_edges_y ** 2 + 1e-6)
        
        target_edges_x = F.conv2d(targets, self.sobel_x, padding=1)
        target_edges_y = F.conv2d(targets, self.sobel_y, padding=1)
        target_edges = torch.sqrt(target_edges_x ** 2 + target_edges_y ** 2 + 1e-6)
        
        # MSE loss on edges
        boundary_loss = F.mse_loss(pred_edges, target_edges)
        
        return self.weight * boundary_loss
