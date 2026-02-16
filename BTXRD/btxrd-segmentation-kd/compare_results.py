import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def visualize_comparison(image_path, mask_path, pred_path, output_path):
    """Create a side-by-side comparison visualization"""
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    pred = Image.open(pred_path).convert('L')
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original X-ray', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Ground truth mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth Mask', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(pred, cmap='gray')
    axes[2].set_title('Student Model Prediction', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Overlay
    axes[3].imshow(image)
    pred_array = np.array(pred) / 255.0
    axes[3].imshow(pred_array, cmap='jet', alpha=0.4)
    axes[3].set_title('Prediction Overlay', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved comparison to {output_path}')


if __name__ == '__main__':
    # Define test cases
    test_cases = [
        {
            'name': 'other_bt',
            'image': r'c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\images_resized\IMG000845.jpeg',
            'mask': r'c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\segmentation_masks\IMG000845_mask.png',
            'pred': 'inference_results/IMG000845_pred.png',
        },
        {
            'name': 'osteosarcoma',
            'image': r'c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\images_resized\IMG001375.jpeg',
            'mask': r'c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\segmentation_masks\IMG001375_mask.png',
            'pred': 'inference_results/IMG001375_osteosarcoma_pred.png',
        },
        {
            'name': 'osteochondroma',
            'image': r'c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\images_resized\IMG001120.jpeg',
            'mask': r'c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\segmentation_masks\IMG001120_mask.png',
            'pred': 'inference_results/IMG001120_osteochondroma_pred.png',
        },
        {
            'name': 'bone_cyst',
            'image': r'c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\images_resized\IMG000691.jpeg',
            'mask': r'c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\segmentation_masks\IMG000691_mask.png',
            'pred': 'inference_results/IMG000691_bone_cyst_pred.png',
        },
    ]
    
    for case in test_cases:
        output_path = f'inference_results/comparison_{case["name"]}.png'
        visualize_comparison(case['image'], case['mask'], case['pred'], output_path)
