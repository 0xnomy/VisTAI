"""
Classification Dataset Loader
==============================
Loads augmented or original classification data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from pathlib import Path
import sys

# Add common module to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'common'))
from transforms import get_classification_transforms


class ClassificationDataset(Dataset):
    """
    Dataset for tumor classification.
    Loads images and their corresponding class labels.
    """
    
    def __init__(self, csv_path, label_encoding, transform=None, phase='train'):
        """
        Args:
            csv_path: Path to CSV file with columns: image_path, labels
            label_encoding: Dictionary mapping class names to indices
            transform: torchvision transforms (optional, will use default if None)
            phase: 'train', 'val', or 'test'
        """
        self.df = pd.read_csv(csv_path)
        self.label_encoding = label_encoding
        self.phase = phase
        
        # Use provided transform or get default
        if transform is None:
            self.transform = get_classification_transforms(phase=phase)
        else:
            self.transform = transform
        
        print(f"Loaded {len(self.df)} samples for {phase}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Transformed image tensor [3, H, W]
            label: Class label (integer)
        """
        row = self.df.iloc[idx]
        
        # Load image
        image_path = row['image_path']
        image = Image.open(image_path).convert('RGB')
        
        # Get label
        label_name = row['labels']
        label = self.label_encoding[label_name]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_classification_dataloaders(
    train_csv, 
    val_csv, 
    test_csv, 
    label_encoding,
    batch_size=32,
    num_workers=4,
    pin_memory=True
):
    """
    Create dataloaders for classification training.
    
    Args:
        train_csv: Path to training CSV
        val_csv: Path to validation CSV
        test_csv: Path to test CSV
        label_encoding: Dictionary mapping class names to indices
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    
    # Create datasets
    train_dataset = ClassificationDataset(
        train_csv, label_encoding, 
        transform=None,  # Will use default
        phase='train'
    )
    
    val_dataset = ClassificationDataset(
        val_csv, label_encoding,
        transform=None,
        phase='val'
    )
    
    test_dataset = ClassificationDataset(
        test_csv, label_encoding,
        transform=None,
        phase='test'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


if __name__ == "__main__":
    import json
    
    print("Testing Classification Dataset...")
    
    # Paths (adjust as needed)
    base_path = Path(r"c:\Users\Nauman\Desktop\vistai\FYP\BTXRD")
    
    # Try augmented data first, fall back to original
    augmented_path = base_path / "augmented_classification_data"
    if (augmented_path / "augmented_train.csv").exists():
        train_csv = augmented_path / "augmented_train.csv"
        val_csv = augmented_path / "augmented_val.csv"
        test_csv = augmented_path / "augmented_test.csv"
        print("Using augmented dataset")
    else:
        train_csv = base_path / "segmentation_train.csv"
        val_csv = base_path / "segmentation_val.csv"
        test_csv = base_path / "segmentation_test.csv"
        print("Using original dataset")
    
    # Load label encoding
    with open(base_path / "label_encoding.json", 'r') as f:
        encoding = json.load(f)
        label_to_idx = encoding['label_to_idx']
    
    # Create dataloaders
    loaders = create_classification_dataloaders(
        train_csv, val_csv, test_csv,
        label_to_idx,
        batch_size=16,
        num_workers=0  # Use 0 for testing
    )
    
    # Test loading a batch
    print("\nTesting batch loading...")
    images, labels = next(iter(loaders['train']))
    
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Unique labels in batch: {torch.unique(labels).tolist()}")
    
    print("\n✓ Dataset test passed!")
