import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SegmentationDataset(Dataset):
    def __init__(self, csv_path, image_size=224, augment=False, aug_config=None):
        self.df = pd.read_csv(csv_path)
        self.image_size = image_size
        self.augment = augment
        
        if augment and aug_config:
            self.transform = self._get_augmentation(aug_config)
        else:
            self.transform = self._get_val_transform()
    
    def _get_augmentation(self, config):
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.HorizontalFlip(p=config.get('horizontal_flip', 0.5)),
            A.VerticalFlip(p=config.get('vertical_flip', 0.5)),
            A.Rotate(limit=config.get('rotation_limit', 15), p=0.5),
            A.RandomBrightnessContrast(p=config.get('brightness_contrast', 0.3)),
            A.GaussianBlur(p=config.get('gaussian_blur', 0.2)),
            A.ElasticTransform(alpha=120, sigma=6, p=config.get('elastic_transform', 0.3)),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def _get_val_transform(self):
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']
        mask_path = row['mask_path']
        
        image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))
        mask = (mask > 127).astype(np.float32)
        
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask'].unsqueeze(0)
        
        return image, mask
