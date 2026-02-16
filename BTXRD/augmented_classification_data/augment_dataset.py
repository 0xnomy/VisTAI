"""
Dataset Augmentation Script for Classification
==============================================
Creates augmented versions of the classification dataset using:
- Geometric/Spatial: rotation, flipping, scaling, shifting, shearing
- Photometric: brightness, contrast, saturation, hue, noise, blur

Generates multiple augmented versions per image to significantly increase dataset size.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random
import json


class DatasetAugmenter:
    """Augment dataset with geometric and photometric transforms."""
    
    def __init__(self, num_augmentations_per_image=5):
        """
        Args:
            num_augmentations_per_image: How many augmented versions to create per image
        """
        self.num_augmentations = num_augmentations_per_image
    
    def geometric_augmentation(self, image):
        """Apply random geometric transformations."""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Random rotation (-30 to +30 degrees)
        if random.random() > 0.3:
            angle = random.uniform(-30, 30)
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_array = cv2.warpAffine(img_array, rotation_matrix, (w, h), 
                                       borderMode=cv2.BORDER_REFLECT)
        
        # Random horizontal flip
        if random.random() > 0.5:
            img_array = cv2.flip(img_array, 1)
        
        # Random vertical flip
        if random.random() > 0.5:
            img_array = cv2.flip(img_array, 0)
        
        # Random affine transform (scaling + shifting + shearing)
        if random.random() > 0.3:
            # Scale: 0.85 to 1.15
            scale = random.uniform(0.85, 1.15)
            # Translation: -10% to +10%
            tx = random.uniform(-0.1, 0.1) * w
            ty = random.uniform(-0.1, 0.1) * h
            # Shear: -15 to +15 degrees
            shear = random.uniform(-15, 15)
            
            M = np.float32([
                [scale, np.tan(np.deg2rad(shear)), tx],
                [0, scale, ty]
            ])
            img_array = cv2.warpAffine(img_array, M, (w, h), 
                                       borderMode=cv2.BORDER_REFLECT)
        
        # Random perspective transform
        if random.random() > 0.5:
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            offset = int(0.1 * min(h, w))
            pts2 = pts1 + np.random.randint(-offset, offset, pts1.shape).astype(np.float32)
            M = cv2.getPerspectiveTransform(pts1, pts2)
            img_array = cv2.warpPerspective(img_array, M, (w, h), 
                                           borderMode=cv2.BORDER_REFLECT)
        
        # Random crop and resize (0.8 to 1.0 scale)
        if random.random() > 0.3:
            crop_scale = random.uniform(0.8, 1.0)
            new_h, new_w = int(h * crop_scale), int(w * crop_scale)
            
            top = random.randint(0, h - new_h)
            left = random.randint(0, w - new_w)
            
            img_array = img_array[top:top+new_h, left:left+new_w]
            img_array = cv2.resize(img_array, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return Image.fromarray(img_array)
    
    def photometric_augmentation(self, image):
        """Apply random photometric (color/lighting) transformations."""
        
        # Brightness adjustment (0.7 to 1.3)
        if random.random() > 0.3:
            enhancer = ImageEnhance.Brightness(image)
            factor = random.uniform(0.7, 1.3)
            image = enhancer.enhance(factor)
        
        # Contrast adjustment (0.7 to 1.3)
        if random.random() > 0.3:
            enhancer = ImageEnhance.Contrast(image)
            factor = random.uniform(0.7, 1.3)
            image = enhancer.enhance(factor)
        
        # Saturation adjustment (0.7 to 1.3)
        if random.random() > 0.3:
            enhancer = ImageEnhance.Color(image)
            factor = random.uniform(0.7, 1.3)
            image = enhancer.enhance(factor)
        
        # Sharpness adjustment (0.5 to 2.0)
        if random.random() > 0.5:
            enhancer = ImageEnhance.Sharpness(image)
            factor = random.uniform(0.5, 2.0)
            image = enhancer.enhance(factor)
        
        # Convert to numpy for additional transforms
        img_array = np.array(image).astype(np.float32)
        
        # Add Gaussian noise
        if random.random() > 0.4:
            noise_std = random.uniform(5, 15)
            noise = np.random.normal(0, noise_std, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255)
        
        # Add salt & pepper noise
        if random.random() > 0.7:
            prob = random.uniform(0.001, 0.005)
            # Salt
            salt = np.random.random(img_array.shape[:2]) < prob / 2
            img_array[salt] = 255
            # Pepper
            pepper = np.random.random(img_array.shape[:2]) < prob / 2
            img_array[pepper] = 0
        
        # Gaussian blur
        if random.random() > 0.5:
            kernel_size = random.choice([3, 5])
            sigma = random.uniform(0.5, 2.0)
            img_array = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), sigma)
        
        # Gamma correction
        if random.random() > 0.5:
            gamma = random.uniform(0.7, 1.3)
            img_array = np.power(img_array / 255.0, gamma) * 255.0
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if random.random() > 0.6:
            img_array = img_array.astype(np.uint8)
            # Apply to each channel
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def augment_image(self, image):
        """Apply both geometric and photometric augmentation."""
        # Apply geometric first
        image = self.geometric_augmentation(image)
        # Then photometric
        image = self.photometric_augmentation(image)
        return image
    
    def augment_dataset(self, original_csv, images_dir, output_dir):
        """
        Augment entire dataset.
        
        Args:
            original_csv: Path to original CSV (train/val/test)
            images_dir: Directory containing original images
            output_dir: Directory to save augmented images
        
        Returns:
            DataFrame with augmented dataset info
        """
        # Read original dataset
        df = pd.read_csv(original_csv)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each class
        for label in df['labels'].unique():
            (output_dir / label).mkdir(exist_ok=True)
        
        augmented_records = []
        
        print(f"\nAugmenting dataset: {Path(original_csv).name}")
        print(f"Original samples: {len(df)}")
        print(f"Augmentations per image: {self.num_augmentations}")
        print(f"Expected total: {len(df) * (1 + self.num_augmentations)}")
        print("=" * 70)
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting"):
            image_path = row['image_path']
            label = row['labels']
            
            # Load original image
            image = Image.open(image_path).convert('RGB')
            
            # Save original image to new location
            orig_filename = Path(image_path).stem + "_orig.png"
            orig_save_path = output_dir / label / orig_filename
            image.save(orig_save_path)
            
            augmented_records.append({
                'image_path': str(orig_save_path),
                'labels': label,
                'augmented': False
            })
            
            # Generate augmented versions
            for aug_idx in range(self.num_augmentations):
                aug_image = self.augment_image(image.copy())
                
                # Save augmented image
                aug_filename = Path(image_path).stem + f"_aug{aug_idx+1}.png"
                aug_save_path = output_dir / label / aug_filename
                aug_image.save(aug_save_path)
                
                augmented_records.append({
                    'image_path': str(aug_save_path),
                    'labels': label,
                    'augmented': True
                })
        
        # Create augmented dataframe
        aug_df = pd.DataFrame(augmented_records)
        
        print(f"\n✓ Augmentation complete!")
        print(f"  Original: {len(df)}")
        print(f"  Augmented: {len(aug_df)}")
        print(f"  Increase: {len(aug_df) / len(df):.2f}x")
        
        return aug_df


def main():
    """Main augmentation pipeline."""
    
    base_path = Path(r"c:\Users\Nauman\Desktop\vistai\FYP\BTXRD")
    
    # Paths
    train_csv = base_path / "segmentation_train.csv"
    val_csv = base_path / "segmentation_val.csv"
    test_csv = base_path / "segmentation_test.csv"
    
    images_dir = base_path / "images_resized"
    output_base = base_path / "augmented_classification_data"
    
    # Output directories
    train_output = output_base / "train"
    val_output = output_base / "val"
    test_output = output_base / "test"
    
    # Load label encoding
    with open(base_path / "label_encoding.json", 'r') as f:
        label_encoding = json.load(f)
    
    print("=" * 70)
    print("DATASET AUGMENTATION FOR CLASSIFICATION")
    print("=" * 70)
    
    # Create augmenter
    # Train: 5 augmentations per image (6x total)
    # Val: 2 augmentations per image (3x total)
    # Test: Keep original only (no augmentation)
    
    train_augmenter = DatasetAugmenter(num_augmentations_per_image=5)
    val_augmenter = DatasetAugmenter(num_augmentations_per_image=2)
    
    # Augment training set
    print("\n[1/3] Training Set")
    train_aug_df = train_augmenter.augment_dataset(
        train_csv, images_dir, train_output
    )
    train_aug_csv = output_base / "augmented_train.csv"
    train_aug_df.to_csv(train_aug_csv, index=False)
    print(f"  → Saved to {train_aug_csv}")
    
    # Augment validation set
    print("\n[2/3] Validation Set")
    val_aug_df = val_augmenter.augment_dataset(
        val_csv, images_dir, val_output
    )
    val_aug_csv = output_base / "augmented_val.csv"
    val_aug_df.to_csv(val_aug_csv, index=False)
    print(f"  → Saved to {val_aug_csv}")
    
    # Copy test set (no augmentation for fair evaluation)
    print("\n[3/3] Test Set (No Augmentation)")
    test_df = pd.read_csv(test_csv)
    test_output.mkdir(parents=True, exist_ok=True)
    
    test_records = []
    for label in test_df['labels'].unique():
        (test_output / label).mkdir(exist_ok=True)
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Copying"):
        image_path = row['image_path']
        label = row['labels']
        
        image = Image.open(image_path).convert('RGB')
        filename = Path(image_path).name
        save_path = test_output / label / filename
        image.save(save_path)
        
        test_records.append({
            'image_path': str(save_path),
            'labels': label,
            'augmented': False
        })
    
    test_aug_df = pd.DataFrame(test_records)
    test_aug_csv = output_base / "augmented_test.csv"
    test_aug_df.to_csv(test_aug_csv, index=False)
    print(f"  → Saved to {test_aug_csv}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("AUGMENTATION SUMMARY")
    print("=" * 70)
    print(f"\nTraining Set:")
    print(f"  Original: 1,493 images")
    print(f"  Augmented: {len(train_aug_df):,} images ({len(train_aug_df)/1493:.1f}x)")
    print(f"  Location: {train_output}")
    
    print(f"\nValidation Set:")
    print(f"  Original: 187 images")
    print(f"  Augmented: {len(val_aug_df):,} images ({len(val_aug_df)/187:.1f}x)")
    print(f"  Location: {val_output}")
    
    print(f"\nTest Set:")
    print(f"  Original: 187 images")
    print(f"  Augmented: {len(test_aug_df):,} images (1.0x - no augmentation)")
    print(f"  Location: {test_output}")
    
    print(f"\n✓ Total augmented dataset: {len(train_aug_df) + len(val_aug_df) + len(test_aug_df):,} images")
    
    # Print class distribution
    print("\n" + "=" * 70)
    print("CLASS DISTRIBUTION (Training Set)")
    print("=" * 70)
    class_counts = train_aug_df['labels'].value_counts().sort_index()
    for label, count in class_counts.items():
        print(f"  {label:30s}: {count:4d} images")
    
    # Save label encoding to augmented folder
    with open(output_base / "label_encoding.json", 'w') as f:
        json.dump(label_encoding, f, indent=2)
    
    print("\n" + "=" * 70)
    print("✓ Dataset augmentation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
