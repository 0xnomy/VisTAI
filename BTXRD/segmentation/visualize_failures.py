"""
Visualize Failed Segmentation Cases
====================================
Show actual images, ground truth masks, and predictions side-by-side
to understand WHY the model is failing (without medical knowledge).
"""

import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from segmentation.models import build_model
from segmentation.datasets.segmentation_dataset import SegmentationDataset

def load_model(checkpoint_path, device='cuda'):
    """Load trained Swin-UNet model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Model was trained with deep_supervision=True
    model = build_model('swin_unet', pretrained=False, img_size=224, deep_supervision=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def predict_single_image(model, image, device='cuda'):
    """Get model prediction for a single image"""
    image = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        # Handle deep supervision (multiple outputs)
        if isinstance(output, (tuple, list)):
            output = output[0]
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
    
    return pred_mask

def visualize_failure_cases(failed_image_ids, csv_path, checkpoint_path, num_samples=10, output_dir='failure_analysis'):
    """Visualize failed predictions"""
    
    print(f"Loading model from {checkpoint_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(checkpoint_path, device)
    
    # Load dataset
    print(f"Loading validation dataset...")
    dataset = SegmentationDataset(csv_path, image_size=224, augment=False)
    
    # Get indices of failed images
    df = pd.read_csv(csv_path)
    failed_indices = []
    for img_id in failed_image_ids[:num_samples]:
        idx = df[df['image_path'].str.contains(img_id)].index
        if len(idx) > 0:
            failed_indices.append(idx[0])
    
    if not failed_indices:
        print("No failed images found!")
        return
    
    print(f"\nVisualizing {len(failed_indices)} failed cases...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create figure
    n_samples = len(failed_indices)
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    
    # Handle single sample case
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(failed_indices):
        # Get data
        image, mask = dataset[idx]
        image_path = df.iloc[idx]['image_path']
        image_id = Path(image_path).stem
        
        # Get prediction
        pred_mask = predict_single_image(model, image, device)
        
        # Convert to numpy for visualization
        image_np = image.permute(1, 2, 0).cpu().numpy()
        mask_np = mask.squeeze().cpu().numpy()
        
        # Denormalize image (assuming ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = image_np * std + mean
        image_np = np.clip(image_np, 0, 1)
        
        # Plot
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f'{image_id}\nOriginal Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].set_title(f'Ground Truth\nPixels: {int(mask_np.sum())}')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Prediction (prob)\nMax: {pred_mask.max():.3f}')
        axes[i, 2].axis('off')
        
        # Overlay
        overlay = image_np.copy()
        # Ground truth in green
        overlay[mask_np > 0.5] = [0, 1, 0]
        # Prediction in red
        overlay[pred_mask > 0.5] = [1, 0, 0]
        # Overlap in yellow
        overlap = (mask_np > 0.5) & (pred_mask > 0.5)
        overlay[overlap] = [1, 1, 0]
        
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title(f'Overlay\nGreen=GT, Red=Pred, Yellow=Match')
        axes[i, 3].axis('off')
        
        # Calculate Dice for this image
        pred_binary = (pred_mask > 0.5).astype(np.float32)
        intersection = (mask_np * pred_binary).sum()
        dice = (2 * intersection) / (mask_np.sum() + pred_binary.sum() + 1e-8)
        
        print(f"  {image_id}: GT pixels={int(mask_np.sum())}, Pred pixels={int(pred_binary.sum())}, Dice={dice:.4f}")
    
    plt.tight_layout()
    output_file = output_path / f'failed_cases_visualization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved visualization: {output_file}")
    plt.close()

def main():
    # Failed images from full validation
    failed_images = [
        'IMG000554', 'IMG000284', 'IMG001557', 'IMG001656', 'IMG000724',
        'IMG000262', 'IMG000295', 'IMG000237', 'IMG000352', 'IMG001190',
        'IMG000536', 'IMG000144', 'IMG001806', 'IMG000385', 'IMG000546',
        'IMG001252', 'IMG001068', 'IMG001394', 'IMG000988', 'IMG000973',
    ]
    
    csv_path = 'segmentation_val.csv'
    checkpoint_path = 'outputs/swin_unet_teacher/checkpoint_best.pth'
    
    print("="*70)
    print("Visualizing Failed Segmentation Cases")
    print("="*70)
    
    visualize_failure_cases(
        failed_images,
        csv_path,
        checkpoint_path,
        num_samples=10,
        output_dir='failure_analysis'
    )

if __name__ == '__main__':
    main()
