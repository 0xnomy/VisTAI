import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as T


class ClassificationDataset(Dataset):
    def __init__(self, data_root, split='train', transform=None, class_names=None):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        
        split_dir = os.path.join(data_root, split)
        
        if class_names is None:
            self.class_names = sorted([d for d in os.listdir(split_dir) 
                                      if os.path.isdir(os.path.join(split_dir, d))])
        else:
            self.class_names = class_names
        
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        
        self.samples = []
        for class_name in self.class_names:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    label = self.class_to_idx[class_name]
                    self.samples.append((img_path, label))
        
        self.compute_class_weights()
    
    def compute_class_weights(self):
        labels = [label for _, label in self.samples]
        class_counts = np.bincount(labels, minlength=len(self.class_names))
        total = len(labels)
        self.class_weights = torch.FloatTensor([total / (len(self.class_names) * count) 
                                                 if count > 0 else 0.0 
                                                 for count in class_counts])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(config, split='train'):
    image_size = config['dataset']['image_size']
    
    if split == 'train':
        aug_cfg = config['augmentation']['train']
        transforms = T.Compose([
            T.RandomResizedCrop(image_size, scale=tuple(aug_cfg['crop_scale'])),
            T.RandomHorizontalFlip() if aug_cfg['horizontal_flip'] else Identity(),
            T.RandomVerticalFlip() if aug_cfg.get('vertical_flip', False) else Identity(),
            T.RandomRotation(aug_cfg['rotation']),
            T.ColorJitter(*aug_cfg['color_jitter']),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        aug_cfg = config['augmentation']['val']
        transforms = T.Compose([
            T.Resize(aug_cfg['resize']),
            T.CenterCrop(aug_cfg['center_crop']),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transforms


class Identity(nn.Module):
    def forward(self, x):
        return x

