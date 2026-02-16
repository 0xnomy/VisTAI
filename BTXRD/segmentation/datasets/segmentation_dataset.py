"""
PyTorch Dataset for tumor segmentation.
Loads images and corresponding binary masks for training/validation/testing.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random


class SegmentationDataset(Dataset):
    # Dataset for loading images and binary segmentation masks
    
    def __init__(self, csv_path, image_size=384, augment=False):
        self.data = pd.read_csv(csv_path)
        self.image_size = image_size
        self.augment = augment
        
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image and mask
        image = Image.open(row['image_path']).convert('RGB')
        mask = Image.open(row['mask_path']).convert('L')
        
        # Resize
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        
        # Apply synchronized augmentation
        if self.augment:
            image, mask = self._apply_augmentation(image, mask)
        
        # Convert to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        
        # Normalize image
        image = self.normalize(image)
        
        # Binarize mask - lower threshold to preserve sparse regions
        mask = (mask > 0.1).float()
        
        return image, mask
    
    def _apply_augmentation(self, image, mask):
        # Synchronized augmentation for image and mask
        
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-20, 20)
            image = TF.rotate(image, angle, fill=0)
            mask = TF.rotate(mask, angle, fill=0)
        
        return image, mask


def create_segmentation_dataloaders(train_csv, val_csv, test_csv=None, 
                                   batch_size=16, num_workers=0, image_size=384):
    # Create dataloaders for segmentation training
    
    # Datasets
    train_dataset = SegmentationDataset(train_csv, image_size=image_size, augment=True)
    val_dataset = SegmentationDataset(val_csv, image_size=image_size, augment=False)
    
    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    loaders = {'train': train_loader, 'val': val_loader}
    
    if test_csv:
        test_dataset = SegmentationDataset(test_csv, image_size=image_size, augment=False)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        loaders['test'] = test_loader
    
    return loaders


if __name__ == "__main__":
    # Test dataset loading
    print("Testing Segmentation Dataset...")
    
    train_csv = r"c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\segmentation\segmentation_train.csv"
    val_csv = r"c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\segmentation\segmentation_val.csv"
    test_csv = r"c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\segmentation\segmentation_test.csv"
    
    # Create dataloaders
    loaders = create_segmentation_dataloaders(
        train_csv, val_csv, test_csv,
        batch_size=4, num_workers=0, image_size=224
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(loaders['train'].dataset)} samples")
    print(f"  Val:   {len(loaders['val'].dataset)} samples")
    print(f"  Test:  {len(loaders['test'].dataset)} samples")
    
    # Test loading a batch
    print("\nTesting batch loading...")
    for images, masks in loaders['train']:
        print(f"  Image batch shape: {images.shape}")
        print(f"  Mask batch shape:  {masks.shape}")
        print(f"  Image dtype: {images.dtype}, range: [{images.min():.2f}, {images.max():.2f}]")
        print(f"  Mask dtype:  {masks.dtype}, range: [{masks.min():.2f}, {masks.max():.2f}]")
        print(f"  Mask unique values: {torch.unique(masks)}")
        break
    
    print("\n✓ Dataset loading successful!")
