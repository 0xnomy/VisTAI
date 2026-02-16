"""
Segmentation Inference
======================
Run inference with any segmentation model (student or teacher).

Usage:
    python inference.py --model nnunet --checkpoint outputs/nnunet/checkpoint_best.pth
    python inference.py --model unetplusplus_resnet50 --checkpoint outputs/teacher/checkpoint_best.pth --visualize
"""

import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Project imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from segmentation.models import build_model, list_models
from common.transforms import get_segmentation_transforms
from common.metrics import compute_dice_score, compute_iou
from common.gradcam import GradCAM


class SegmentationInference:
    """Handles inference for segmentation models."""
    
    def __init__(self, model_name: str, checkpoint_path: str, device: str = None):
        """
        Initialize inference engine.
        
        Args:
            model_name: Name of model ('nnunet', 'unetplusplus_resnet50', etc.)
            checkpoint_path: Path to model checkpoint
            device: Device to use (auto-detect if None)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        # Build model
        print(f"\n📦 Loading {model_name} model...")
        self.model = build_model(model_name, pretrained=False)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Print checkpoint info
        if 'epoch' in checkpoint:
            print(f"   Loaded from epoch {checkpoint['epoch']}")
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            if 'val_dice' in metrics:
                print(f"   Validation Dice: {metrics['val_dice']:.4f}")
            if 'val_iou' in metrics:
                print(f"   Validation IoU: {metrics['val_iou']:.4f}")
        
        # Get transforms
        self.transform = get_segmentation_transforms(phase='test')
        
        # Setup GradCAM - use last encoder layer for UNet-style models
        self._setup_gradcam()
    
    def _setup_gradcam(self):
        """Setup GradCAM for the model."""
        # Find suitable target layer for GradCAM
        if hasattr(self.model, 'down4'):
            # NNUNet style models
            self.gradcam = GradCAM(self.model, self.model.down4)
        elif hasattr(self.model, 'encoder'):
            # Teacher models with encoder - check encoder type
            if hasattr(self.model.encoder, 'layer4'):
                # ResNet encoders
                self.gradcam = GradCAM(self.model, self.model.encoder.layer4)
            elif hasattr(self.model.encoder, 'blocks'):
                # EfficientNet encoders - use last block
                self.gradcam = GradCAM(self.model, self.model.encoder.blocks[-1])
            else:
                self.gradcam = None
        else:
            self.gradcam = None
    
    def _generate_gradcam(self, image_tensor):
        """Generate GradCAM heatmap."""
        if self.gradcam is None:
            return None
        
        self.model.train()  # Need gradient computation
        output = self.model(image_tensor)
        if isinstance(output, tuple):
            output = output[0]
        
        # Backward pass for GradCAM
        self.model.zero_grad()
        output.mean().backward()
        
        # Generate heatmap
        gradients = self.gradcam.gradients
        activations = self.gradcam.activations
        
        if gradients is None or activations is None:
            self.model.eval()
            return None
        
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        self.model.eval()
        return cam
    
    def predict(self, image_path: str, threshold: float = 0.5, use_gradcam: bool = False):
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to image
            threshold: Threshold for binary prediction
            use_gradcam: Generate GradCAM visualization
        
        Returns:
            dict with prediction results
        """
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (W, H)
        
        # Transform needs a dummy mask for test phase
        dummy_mask = Image.new('L', image.size, 0)
        image_tensor, _ = self.transform(image, dummy_mask)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Generate GradCAM if requested
        gradcam_map = None
        if use_gradcam:
            gradcam_map = self._generate_gradcam(image_tensor)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(image_tensor)
            
            # Handle deep supervision output
            if isinstance(output, tuple):
                output = output[0]
            
            # Sigmoid and threshold
            prob = torch.sigmoid(output)
            pred_mask = (prob > threshold).float()
        
        # Convert to numpy
        prob_np = prob.squeeze().cpu().numpy()
        mask_np = pred_mask.squeeze().cpu().numpy()
        
        result = {
            'image_path': image_path,
            'probability_map': prob_np,
            'binary_mask': mask_np,
            'original_size': original_size
        }
        
        if gradcam_map is not None:
            result['gradcam'] = gradcam_map
        
        return result
    
    def predict_with_gt(self, image_path: str, mask_path: str, threshold: float = 0.5, use_gradcam: bool = False):
        """
        Run inference with ground truth comparison.
        
        Args:
            image_path: Path to image
            mask_path: Path to ground truth mask
            threshold: Threshold for binary prediction
            use_gradcam: Generate GradCAM visualization
        
        Returns:
            dict with prediction results and metrics
        """
        result = self.predict(image_path, threshold, use_gradcam)
        
        # Load ground truth
        gt_mask = Image.open(mask_path).convert('L')
        gt_mask = np.array(gt_mask) / 255.0
        
        # Resize prediction to match GT if needed
        pred_mask = result['binary_mask']
        if pred_mask.shape != gt_mask.shape:
            pred_tensor = torch.tensor(pred_mask).unsqueeze(0).unsqueeze(0)
            pred_tensor = F.interpolate(pred_tensor, size=gt_mask.shape, mode='nearest')
            pred_mask = pred_tensor.squeeze().numpy()
        
        # Compute metrics
        pred_tensor = torch.tensor(pred_mask).unsqueeze(0).unsqueeze(0)
        gt_tensor = torch.tensor(gt_mask).unsqueeze(0).unsqueeze(0)
        
        dice = compute_dice_score(pred_tensor, gt_tensor)
        iou = compute_iou(pred_tensor, gt_tensor)
        
        result['gt_mask'] = gt_mask
        result['dice'] = dice
        result['iou'] = iou
        
        return result
    
    def predict_batch(self, csv_path: str, num_samples: int = 10, use_gradcam: bool = False):
        """
        Run inference on multiple images from CSV.
        
        Args:
            csv_path: Path to CSV with image_path and mask_path columns
            num_samples: Number of samples to process
            use_gradcam: Generate GradCAM visualizations
        
        Returns:
            list of prediction results with average metrics
        """
        df = pd.read_csv(csv_path)
        sample_df = df.sample(n=min(num_samples, len(df)), random_state=42).reset_index(drop=True)
        
        results = []
        total_dice = 0
        total_iou = 0
        
        for _, row in sample_df.iterrows():
            result = self.predict_with_gt(row['image_path'], row['mask_path'], use_gradcam=use_gradcam)
            results.append(result)
            total_dice += result['dice']
            total_iou += result['iou']
        
        avg_dice = total_dice / len(results)
        avg_iou = total_iou / len(results)
        
        return results, avg_dice, avg_iou
    
    def visualize_predictions(self, results: list, output_path: str = None):
        """
        Visualize predictions in a grid.
        
        Args:
            results: List of prediction results
            output_path: Optional path to save the figure
        """
        n = len(results)
        has_gradcam = 'gradcam' in results[0] if results else False
        n_cols = 5 if has_gradcam else 4
        fig, axes = plt.subplots(n, n_cols, figsize=(4 * n_cols, 4 * n))
        
        if n == 1:
            axes = axes.reshape(1, -1)
        
        for idx, result in enumerate(results):
            # Load original image
            img = Image.open(result['image_path']).convert('RGB')
            
            # Image
            axes[idx, 0].imshow(img)
            axes[idx, 0].set_title(f"Input: {Path(result['image_path']).name}")
            axes[idx, 0].axis('off')
            
            # Ground truth
            if 'gt_mask' in result:
                axes[idx, 1].imshow(result['gt_mask'], cmap='gray')
                axes[idx, 1].set_title("Ground Truth")
            else:
                axes[idx, 1].axis('off')
            axes[idx, 1].axis('off')
            
            # Prediction
            axes[idx, 2].imshow(result['binary_mask'], cmap='gray')
            title = "Prediction"
            if 'dice' in result:
                title += f"\nDice: {result['dice']:.4f}"
            axes[idx, 2].set_title(title)
            axes[idx, 2].axis('off')
            
            # Overlay
            img_array = np.array(img)
            overlay = img_array.copy()
            mask = result['binary_mask']
            
            # Resize mask to match image
            if mask.shape != img_array.shape[:2]:
                mask = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize(
                    (img_array.shape[1], img_array.shape[0]), Image.NEAREST)) / 255.0
            
            # Red overlay for prediction
            overlay[mask > 0.5] = [255, 100, 100]
            blended = (0.6 * img_array + 0.4 * overlay).astype(np.uint8)
            
            axes[idx, 3].imshow(blended)
            axes[idx, 3].set_title("Overlay")
            axes[idx, 3].axis('off')
            
            # GradCAM if available
            if has_gradcam and 'gradcam' in result:
                cam = result['gradcam']
                # Resize to match image
                if cam.shape != img_array.shape[:2]:
                    cam = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(
                        (img_array.shape[1], img_array.shape[0]), Image.BILINEAR)) / 255.0
                
                # Apply colormap
                import matplotlib.cm as cm
                heatmap = cm.jet(cam)[:, :, :3]
                heatmap = (heatmap * 255).astype(np.uint8)
                
                # Overlay on image
                gradcam_overlay = (0.5 * img_array + 0.5 * heatmap).astype(np.uint8)
                axes[idx, 4].imshow(gradcam_overlay)
                axes[idx, 4].set_title("GradCAM")
                axes[idx, 4].axis('off')
        
        # Overall title
        if results and 'dice' in results[0]:
            avg_dice = np.mean([r['dice'] for r in results])
            avg_iou = np.mean([r['iou'] for r in results])
            plt.suptitle(f"Model: {self.model_name} | Avg Dice: {avg_dice:.4f} | Avg IoU: {avg_iou:.4f}",
                        fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved visualization to {output_path}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Segmentation Inference')
    parser.add_argument('--model', type=str, default='nnunet',
                        help='Model name (nnunet, unetplusplus_resnet50, etc.)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                        help='Single image path for inference')
    parser.add_argument('--mask', type=str, default=None,
                        help='Ground truth mask path (optional)')
    parser.add_argument('--csv', type=str, default=None,
                        help='CSV file with test images')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to process from CSV')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary prediction')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize predictions')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save visualization')
    parser.add_argument('--list-models', action='store_true',
                        help='List available models')
    
    args = parser.parse_args()
    
    if args.list_models:
        list_models()
        return
    
    # Initialize inference engine
    inference = SegmentationInference(
        model_name=args.model,
        checkpoint_path=args.checkpoint
    )
    
    # Single image inference
    if args.image:
        if args.mask:
            result = inference.predict_with_gt(args.image, args.mask, args.threshold)
            print(f"\n🔍 Results for {Path(args.image).name}:")
            print(f"   Dice Score: {result['dice']:.4f}")
            print(f"   IoU Score: {result['iou']:.4f}")
        else:
            result = inference.predict(args.image, args.threshold)
            print(f"\n🔍 Prediction for {Path(args.image).name}:")
            print(f"   Mask shape: {result['binary_mask'].shape}")
            print(f"   Tumor coverage: {result['binary_mask'].mean() * 100:.2f}%")
        
        if args.visualize:
            inference.visualize_predictions([result], args.output)
        return
    
    # Batch inference from CSV
    csv_path = args.csv or (project_root / 'segmentation/segmentation_test.csv')
    print(f"\n📊 Running batch inference on {csv_path}...")
    
    results, avg_dice, avg_iou = inference.predict_batch(str(csv_path), num_samples=args.num_samples)
    
    print(f"\n✅ Results ({len(results)} samples):")
    print(f"   Average Dice: {avg_dice:.4f}")
    print(f"   Average IoU: {avg_iou:.4f}")
    
    print("\n   Details:")
    for i, r in enumerate(results, 1):
        name = Path(r['image_path']).stem
        print(f"   {i:2}. {name:30} Dice: {r['dice']:.4f}  IoU: {r['iou']:.4f}")
    
    if args.visualize:
        output_path = args.output or f"inference_results_{args.model}.png"
        inference.visualize_predictions(results, output_path)


if __name__ == '__main__':
    main()
