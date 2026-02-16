import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(__file__))
from models.student.segformer_b2 import SegFormerB2
from utils.checkpoint import load_checkpoint


class SegmentationGradCAM:
    """Grad-CAM for segmentation models"""
    
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []
        self.activations = []
        self.hooks = []
        
        # Register hooks for multiple layers
        for layer in target_layers:
            self.hooks.append(layer.register_forward_hook(self._save_activation))
            self.hooks.append(layer.register_full_backward_hook(self._save_gradient))
    
    def _save_activation(self, module, input, output):
        self.activations.append(output.detach())
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0].detach())
    
    def generate(self, image_tensor):
        """Generate Grad-CAM heatmap"""
        self.model.eval()
        self.gradients = []
        self.activations = []
        
        # Forward pass
        output = self.model(image_tensor)
        
        # Backward pass - use output as target
        self.model.zero_grad()
        target = torch.sigmoid(output).sum()
        target.backward()
        
        # Generate CAM from the last captured layer
        if len(self.gradients) == 0 or len(self.activations) == 0:
            # Fallback: create empty heatmap
            return np.zeros((image_tensor.shape[2], image_tensor.shape[3]), dtype=np.float32)
        
        gradients = self.gradients[-1].cpu().numpy()[0]
        activations = self.activations[-1].cpu().numpy()[0]
        
        # Global average pooling on gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU and normalize
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Resize to input size
        cam = cv2.resize(cam, (image_tensor.shape[3], image_tensor.shape[2]))
        
        return cam
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


def preprocess_image(image_path, image_size=224):
    """Load and preprocess image"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    image_resized = image.resize((image_size, image_size))
    
    # Normalize
    image_array = np.array(image_resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_normalized = (image_array - mean) / std
    
    # To tensor
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0).float()
    
    return image_tensor, image_resized, original_size


def apply_colormap(cam, image_np):
    """Apply jet colormap to CAM and overlay on image"""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)
    return heatmap, overlay


def inference_with_visualization(model, image_path, mask_path, output_path, device, threshold=0.5, image_size=224):
    """Run inference and create comprehensive visualization"""
    
    # Preprocess
    image_tensor, image_resized, original_size = preprocess_image(image_path, image_size)
    image_tensor = image_tensor.to(device)
    image_np = np.array(image_resized)
    
    # Get target layers for Grad-CAM (use decoder layers)
    target_layers = []
    for name, module in model.model.named_modules():
        if 'decode_head' in name and isinstance(module, torch.nn.Conv2d):
            target_layers.append(module)
            if len(target_layers) >= 1:  # Use first decoder conv layer
                break
    
    # Generate prediction
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        prob = torch.sigmoid(logits)
        mask_pred = (prob > threshold).cpu().numpy()[0, 0]
        prob_np = prob.cpu().numpy()[0, 0]
    
    # Generate Grad-CAM
    if len(target_layers) > 0:
        gradcam = SegmentationGradCAM(model, target_layers)
        cam = gradcam.generate(image_tensor)
        gradcam.remove_hooks()
    else:
        cam = np.zeros_like(prob_np)
    
    # Load ground truth
    mask_gt = np.array(Image.open(mask_path).convert('L').resize((image_size, image_size))) / 255.0
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1
    # Original X-ray
    axes[0, 0].imshow(image_resized)
    axes[0, 0].set_title('Original X-ray', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Ground truth mask
    axes[0, 1].imshow(mask_gt, cmap='gray')
    axes[0, 1].set_title('Ground Truth Mask', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Student prediction
    axes[0, 2].imshow(mask_pred, cmap='gray')
    axes[0, 2].set_title('Student Model Prediction', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2
    # Prediction overlay
    axes[1, 0].imshow(image_resized)
    axes[1, 0].imshow(mask_pred, cmap='jet', alpha=0.4)
    axes[1, 0].set_title('Prediction Overlay', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Grad-CAM heatmap
    axes[1, 1].imshow(cam, cmap='jet')
    axes[1, 1].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Grad-CAM overlay
    heatmap, cam_overlay = apply_colormap(cam, image_np)
    axes[1, 2].imshow(cam_overlay)
    axes[1, 2].set_title('Grad-CAM Overlay', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    
    # Add metrics text
    # Compute Dice score
    intersection = np.sum(mask_gt * mask_pred)
    dice = (2. * intersection) / (np.sum(mask_gt) + np.sum(mask_pred) + 1e-8)
    
    # Compute IoU
    union = np.sum(np.logical_or(mask_gt, mask_pred))
    iou = intersection / (union + 1e-8)
    
    fig.suptitle(f'Dice: {dice:.4f} | IoU: {iou:.4f}', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return dice, iou


def main():
    # Configuration
    checkpoint_path = 'outputs/kd_student/best_model.pth'
    image_size = 224
    threshold = 0.5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print("Loading student model...")
    model = SegFormerB2(num_classes=1, image_size=image_size, pretrained=False)
    load_checkpoint(model, checkpoint_path)
    model = model.to(device)
    print(f"Model loaded on {device}")
    
    # Select 10 test images from validation set
    test_cases = [
        {
            'name': 'other_bt_1',
            'image': r'c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\images_resized\IMG000845.jpeg',
            'mask': r'c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\segmentation_masks\IMG000845_mask.png',
        },
        {
            'name': 'osteosarcoma_1',
            'image': r'c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\images_resized\IMG001375.jpeg',
            'mask': r'c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\segmentation_masks\IMG001375_mask.png',
        },
        {
            'name': 'osteochondroma_1',
            'image': r'c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\images_resized\IMG001120.jpeg',
            'mask': r'c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\segmentation_masks\IMG001120_mask.png',
        },
        {
            'name': 'bone_cyst_1',
            'image': r'c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\images_resized\IMG000691.jpeg',
            'mask': r'c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\segmentation_masks\IMG000691_mask.png',
        },
        {
            'name': 'multiple_osteochondromas_1',
            'image': r'c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\images_resized\IMG000973.jpeg',
            'mask': r'c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\segmentation_masks\IMG000973_mask.png',
        },
        {
            'name': 'synovial_osteochondroma_1',
            'image': r'c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\images_resized\IMG001190.jpeg',
            'mask': r'c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\segmentation_masks\IMG001190_mask.png',
        },
        {
            'name': 'other_mt_1',
            'image': r'c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\images_resized\IMG000020.jpeg',
            'mask': r'c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\segmentation_masks\IMG000020_mask.png',
        },
        {
            'name': 'bone_cyst_2',
            'image': r'c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\images_resized\IMG000683.jpeg',
            'mask': r'c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\segmentation_masks\IMG000683_mask.png',
        },
        {
            'name': 'osteosarcoma_2',
            'image': r'c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\images_resized\IMG000168.jpeg',
            'mask': r'c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\segmentation_masks\IMG000168_mask.png',
        },
        {
            'name': 'osteochondroma_2',
            'image': r'c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\images_resized\IMG000707.jpeg',
            'mask': r'c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\segmentation_masks\IMG000707_mask.png',
        },
    ]
    
    # Create output directory
    os.makedirs('gradcam_results', exist_ok=True)
    
    # Process each image
    results = []
    for i, case in enumerate(test_cases, 1):
        print(f"\n[{i}/10] Processing {case['name']}...")
        output_path = f"gradcam_results/{case['name']}_visualization.png"
        
        try:
            dice, iou = inference_with_visualization(
                model, case['image'], case['mask'], output_path,
                device, threshold, image_size
            )
            results.append({
                'name': case['name'],
                'dice': dice,
                'iou': iou,
                'output': output_path
            })
            print(f"  ✓ Dice: {dice:.4f}, IoU: {iou:.4f}")
            print(f"  ✓ Saved to {output_path}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("INFERENCE COMPLETE - Summary")
    print("="*60)
    avg_dice = np.mean([r['dice'] for r in results])
    avg_iou = np.mean([r['iou'] for r in results])
    print(f"Average Dice: {avg_dice:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"\nTotal images processed: {len(results)}/10")
    print(f"Output directory: gradcam_results/")
    print("="*60)


if __name__ == '__main__':
    main()
