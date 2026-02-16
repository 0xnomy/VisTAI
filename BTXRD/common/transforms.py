"""Data augmentation and transformation pipelines."""

import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import random
from PIL import Image


def get_classification_transforms(phase='train', image_size=224):
    """
    Get image transforms for classification.
    
    Args:
        phase: 'train', 'val', or 'test'
        image_size: Target image size (default: 224)
    
    Returns:
        torchvision.transforms.Compose object
    """
    if phase == 'train':
        # Strong augmentation for training
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, 
                                 saturation=0.4, hue=0.15),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), 
                                   scale=(0.9, 1.1), shear=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # No augmentation for val/test
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def get_segmentation_transforms(phase='train', image_size=224):
    """
    Get synchronized transforms for segmentation (image + mask).
    
    Note: This returns a function that takes (image, mask) and applies
    synchronized random transforms.
    
    Args:
        phase: 'train', 'val', or 'test'
        image_size: Target image size (default: 224)
    
    Returns:
        Function that takes (image, mask) PIL images
    """
    
    def transform_train(image, mask):
        """Apply synchronized random transforms for training."""
        
        # Resize
        image = image.resize((image_size, image_size), Image.BILINEAR)
        mask = mask.resize((image_size, image_size), Image.NEAREST)
        
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        # Random rotation
        if random.random() > 0.3:
            angle = random.uniform(-30, 30)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
        
        # Random crop and resize
        if random.random() > 0.3:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image, scale=(0.8, 1.0), ratio=(0.9, 1.1)
            )
            image = TF.resized_crop(image, i, j, h, w, 
                                   (image_size, image_size), 
                                   interpolation=Image.BILINEAR)
            mask = TF.resized_crop(mask, i, j, h, w, 
                                  (image_size, image_size), 
                                  interpolation=Image.NEAREST)
        
        # Color jitter (only on image)
        if random.random() > 0.3:
            image = transforms.ColorJitter(
                brightness=0.4, contrast=0.4, 
                saturation=0.4, hue=0.15
            )(image)
        
        # Convert to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        
        # Normalize image (not mask)
        image = TF.normalize(image, 
                           mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
        
        # Binarize mask
        mask = (mask > 0.5).float()
        
        return image, mask
    
    def transform_val_test(image, mask):
        """Apply only resize and normalization for val/test."""
        
        # Resize
        image = image.resize((image_size, image_size), Image.BILINEAR)
        mask = mask.resize((image_size, image_size), Image.NEAREST)
        
        # Convert to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        
        # Normalize image (not mask)
        image = TF.normalize(image,
                           mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
        
        # Binarize mask
        mask = (mask > 0.5).float()
        
        return image, mask
    
    if phase == 'train':
        return transform_train
    else:
        return transform_val_test


def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize a normalized image tensor for visualization.
    
    Args:
        tensor: Normalized image tensor [C, H, W] or [B, C, H, W]
        mean: Mean used for normalization
        std: Std used for normalization
    
    Returns:
        Denormalized tensor in [0, 1] range
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    if tensor.dim() == 4:  # Batch
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    return tensor
