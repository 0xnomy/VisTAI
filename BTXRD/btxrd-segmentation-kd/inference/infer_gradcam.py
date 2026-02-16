"""
Segmentation Inference with Grad-CAM Visualization
Runs inference on the entire validation dataset and generates Grad-CAM visualizations.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.student.segformer_b2 import SegFormerB2
from utils.checkpoint import load_checkpoint


class SegmentationGradCAM:
    """Grad-CAM for segmentation models"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.hook_handles.append(target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(target_layer.register_full_backward_hook(backward_hook))
    
    def generate(self, image_tensor, mask_size):
        """Generate Grad-CAM for segmentation"""
        self.model.eval()
        
        output = self.model(image_tensor)
        self.model.zero_grad()
        
        # For segmentation, we take the mean of the output as the target
        target = output.mean()
        target.backward()
        
        if self.gradients is None or self.activations is None:
            return np.zeros((mask_size, mask_size), dtype=np.float32)
        
        gradients = self.gradients.cpu().numpy()[0]
        activations = self.activations.cpu().numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        
        cam = cv2.resize(cam, (mask_size, mask_size))
        return cam
    
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()


def calculate_dice_score(pred_mask, gt_mask):
    """Calculate Dice coefficient"""
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = (2.0 * intersection) / union
    return dice


def calculate_iou(pred_mask, gt_mask):
    """Calculate IoU (Intersection over Union)"""
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return iou


def save_segmentation_visualization(image_np, gt_mask, pred_mask, cam, dice, iou, output_path):
    """Save 6-panel visualization with Grad-CAM"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title('Original X-ray', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Ground truth mask
    axes[0, 1].imshow(image_np)
    axes[0, 1].imshow(gt_mask, alpha=0.5, cmap='Reds')
    axes[0, 1].set_title('Ground Truth Mask', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Predicted mask
    axes[0, 2].imshow(image_np)
    axes[0, 2].imshow(pred_mask, alpha=0.5, cmap='Blues')
    axes[0, 2].set_title(f'Predicted Mask\nDice: {dice:.3f}, IoU: {iou:.3f}', 
                         fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Grad-CAM heatmap
    axes[1, 0].imshow(cam, cmap='jet')
    axes[1, 0].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Grad-CAM overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Grad-CAM Overlay', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Comparison (GT vs Pred)
    comparison = np.zeros_like(image_np)
    # True Positives: Green
    tp = np.logical_and(gt_mask > 0, pred_mask > 0)
    comparison[tp] = [0, 255, 0]
    # False Positives: Blue
    fp = np.logical_and(gt_mask == 0, pred_mask > 0)
    comparison[fp] = [0, 0, 255]
    # False Negatives: Red
    fn = np.logical_and(gt_mask > 0, pred_mask == 0)
    comparison[fn] = [255, 0, 0]
    
    axes[1, 2].imshow(image_np)
    axes[1, 2].imshow(comparison, alpha=0.5)
    axes[1, 2].set_title('Comparison\nGreen: TP, Blue: FP, Red: FN', 
                         fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'Segmentation Results - Dice: {dice:.3f}, IoU: {iou:.3f}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def preprocess_image(image_path, image_size):
    """Load and preprocess image"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    image = image.resize((image_size, image_size))
    
    image_array = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = (image_array - mean) / std
    
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).float()
    
    return image_tensor, image, np.array(image)


def preprocess_mask(mask_path, image_size):
    """Load and preprocess mask"""
    mask = Image.open(mask_path).convert('L')
    mask = mask.resize((image_size, image_size), Image.NEAREST)
    mask_array = np.array(mask)
    mask_binary = (mask_array > 127).astype(np.uint8)
    return mask_binary


def infer_with_gradcam(model, csv_path, output_dir, device, image_size=224, 
                      threshold=0.5, save_vis=True, max_vis=50):
    """Run inference with Grad-CAM on entire validation dataset"""
    
    # Read validation data
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} validation samples")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    if save_vis:
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
    
    # Setup Grad-CAM - target the last layer of the decoder
    target_layer = None
    for name, module in model.named_modules():
        if 'decode_head' in name and isinstance(module, torch.nn.Conv2d):
            target_layer = module
    
    if target_layer is None:
        # Fallback to any conv layer in the model
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
    
    model.eval()
    
    results = []
    total_dice = 0
    total_iou = 0
    total = 0
    vis_count = 0
    
    print(f"\nRunning inference with Grad-CAM...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        image_path = row['image_path']
        mask_path = row['mask_path']
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found: {mask_path}")
            continue
        
        # Load and preprocess
        image_tensor, image_pil, image_np = preprocess_image(image_path, image_size)
        gt_mask = preprocess_mask(mask_path, image_size)
        
        image_tensor = image_tensor.to(device)
        
        # Inference
        with torch.no_grad():
            logits = model(image_tensor)
            prob = torch.sigmoid(logits)
            pred_mask = (prob > threshold).cpu().numpy()[0, 0].astype(np.uint8)
        
        # Calculate metrics
        dice = calculate_dice_score(pred_mask, gt_mask)
        iou = calculate_iou(pred_mask, gt_mask)
        
        total_dice += dice
        total_iou += iou
        total += 1
        
        # Generate Grad-CAM and save visualization
        if target_layer is not None and save_vis and vis_count < max_vis:
            gradcam = SegmentationGradCAM(model, target_layer)
            cam = gradcam.generate(image_tensor, image_size)
            gradcam.remove_hooks()
            
            # Save visualization
            img_filename = os.path.basename(image_path).replace('.jpeg', '').replace('.jpg', '').replace('.png', '')
            vis_path = os.path.join(vis_dir, f"{img_filename}_segmentation_gradcam.png")
            save_segmentation_visualization(image_np, gt_mask, pred_mask, cam, 
                                          dice, iou, vis_path)
            vis_count += 1
        
        # Record results
        results.append({
            'image_path': image_path,
            'image_filename': os.path.basename(image_path),
            'mask_path': mask_path,
            'dice_score': dice,
            'iou': iou,
            'tumor_type': row.get('labels', 'unknown')
        })
    
    # Calculate overall metrics
    avg_dice = total_dice / total if total > 0 else 0
    avg_iou = total_iou / total if total > 0 else 0
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(output_dir, 'segmentation_results.csv')
    results_df.to_csv(results_csv, index=False)
    
    # Per-class metrics
    if 'tumor_type' in results_df.columns:
        class_metrics = {}
        for tumor_type in results_df['tumor_type'].unique():
            type_df = results_df[results_df['tumor_type'] == tumor_type]
            class_metrics[tumor_type] = {
                'count': len(type_df),
                'avg_dice': type_df['dice_score'].mean(),
                'avg_iou': type_df['iou'].mean()
            }
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"SEGMENTATION INFERENCE COMPLETE")
    print(f"{'='*70}")
    print(f"Total samples: {total}")
    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")
    
    if 'tumor_type' in results_df.columns:
        print(f"\nPer-class results:")
        for tumor_type, metrics in class_metrics.items():
            print(f"  {tumor_type}: Dice={metrics['avg_dice']:.4f}, IoU={metrics['avg_iou']:.4f} (n={metrics['count']})")
    
    print(f"\nResults saved to: {results_csv}")
    if save_vis:
        print(f"Visualizations saved to: {vis_dir} ({vis_count} samples)")
    print(f"{'='*70}")
    
    return results_df, avg_dice, avg_iou


def main():
    parser = argparse.ArgumentParser(description='Segmentation inference with Grad-CAM')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--csv', type=str, required=True, help='Path to validation CSV file')
    parser.add_argument('--output-dir', type=str, default='segmentation_results_gradcam',
                       help='Output directory for results')
    parser.add_argument('--image-size', type=int, default=224, help='Input image size')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold')
    parser.add_argument('--max-vis', type=int, default=50,
                       help='Maximum number of visualizations to save')
    parser.add_argument('--no-vis', action='store_true',
                       help='Disable visualization saving (faster)')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = SegFormerB2(num_classes=1, image_size=args.image_size, pretrained=False)
    load_checkpoint(model, args.checkpoint)
    model = model.to(device)
    print("Model loaded successfully")
    
    # Run inference
    infer_with_gradcam(
        model=model,
        csv_path=args.csv,
        output_dir=args.output_dir,
        device=device,
        image_size=args.image_size,
        threshold=args.threshold,
        save_vis=not args.no_vis,
        max_vis=args.max_vis
    )


if __name__ == '__main__':
    main()
