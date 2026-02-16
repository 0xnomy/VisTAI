"""
Classification Inference with Grad-CAM Visualization
Runs inference on the entire test dataset and generates Grad-CAM visualizations.
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
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.student.convnext_tiny import ConvNeXtTinyStudent
from utils.checkpoint import load_checkpoint


class GradCAM:
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
    
    def generate(self, image_tensor, target_class):
        self.model.eval()
        
        output = self.model(image_tensor)
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()
        
        if self.gradients is None or self.activations is None:
            return np.zeros((image_tensor.shape[2], image_tensor.shape[3]), dtype=np.float32)
        
        gradients = self.gradients.cpu().numpy()[0]
        activations = self.activations.cpu().numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        
        cam = cv2.resize(cam, (image_tensor.shape[3], image_tensor.shape[2]))
        return cam
    
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()


def get_transform(image_size=384):
    return T.Compose([
        T.Resize(416),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def create_gradcam_overlay(image_np, cam):
    """Create heatmap and overlay visualization"""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)
    return heatmap, overlay


def save_visualization(image_np, cam, pred_class, confidence, ground_truth, 
                       class_names, top5_probs, top5_indices, output_path):
    """Save 4-panel visualization with Grad-CAM"""
    heatmap, overlay = create_gradcam_overlay(image_np, cam)
    correct = pred_class == ground_truth
    
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
    
    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image_np)
    ax1.set_title('Original X-ray', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Grad-CAM heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(cam, cmap='jet')
    ax2.set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Overlay
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(overlay)
    ax3.set_title('Grad-CAM Overlay', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Prediction summary
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis('off')
    pred_text = f"Predicted: {class_names[pred_class]}\n"
    pred_text += f"Confidence: {confidence:.2%}\n\n"
    pred_text += f"Ground Truth: {class_names[ground_truth]}\n"
    pred_text += f"Status: {'CORRECT' if correct else 'INCORRECT'}"
    ax4.text(0.1, 0.5, pred_text, fontsize=13, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen' if correct else 'lightcoral', alpha=0.8))
    ax4.set_title('Prediction Summary', fontsize=14, fontweight='bold')
    
    # Top-5 predictions
    ax5 = fig.add_subplot(gs[1, :])
    colors = ['green' if i == ground_truth else 'skyblue' for i in top5_indices]
    ax5.barh(range(len(top5_indices)), top5_probs.numpy(), color=colors)
    ax5.set_yticks(range(len(top5_indices)))
    ax5.set_yticklabels([class_names[i] for i in top5_indices])
    ax5.set_xlabel('Probability', fontsize=12)
    ax5.set_title('Top-5 Predictions', fontsize=14, fontweight='bold')
    ax5.set_xlim(0, 1)
    
    for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
        ax5.text(prob + 0.02, i, f'{prob:.2%}', va='center', fontsize=10)
    
    title = f"Classification: {class_names[pred_class]} ({confidence:.1%})"
    if correct:
        title += " [CORRECT]"
    else:
        title += " [INCORRECT]"
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def infer_with_gradcam(model, csv_path, output_dir, device, class_names, image_size=384, 
                      save_vis=True, max_vis=50):
    """Run inference with Grad-CAM on entire test dataset"""
    
    # Read test data
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} test samples")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    if save_vis:
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
    
    # Setup Grad-CAM
    target_layer = None
    for name, module in model.model.named_modules():
        if 'stages.3' in name and isinstance(module, torch.nn.Conv2d):
            target_layer = module
    
    if target_layer is None:
        for name, module in model.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
    
    transform = get_transform(image_size)
    model.eval()
    
    results = []
    correct = 0
    total = 0
    vis_count = 0
    
    # Create label to index mapping
    label_to_idx = {name: i for i, name in enumerate(class_names)}
    
    print(f"\nRunning inference with Grad-CAM...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        image_path = row['image_path']
        ground_truth_label = row['labels']
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        ground_truth_idx = label_to_idx.get(ground_truth_label, -1)
        if ground_truth_idx == -1:
            print(f"Warning: Unknown label: {ground_truth_label}")
            continue
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_resized = image.resize((image_size, image_size))
        image_np = np.array(image_resized)
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            logits = model(image_tensor)
            probs = F.softmax(logits, dim=1)[0]
            pred_class = torch.argmax(probs).item()
            confidence = probs[pred_class].item()
        
        # Move to CPU for further processing
        probs = probs.cpu()
        
        # Generate Grad-CAM
        if target_layer is not None and save_vis and vis_count < max_vis:
            gradcam = GradCAM(model, target_layer)
            cam = gradcam.generate(image_tensor, pred_class)
            gradcam.remove_hooks()
            
            # Get top-5 predictions
            top5_probs, top5_indices = torch.topk(probs, min(5, len(class_names)))
            
            # Save visualization
            img_filename = os.path.basename(image_path).replace('.jpeg', '').replace('.jpg', '').replace('.png', '')
            vis_path = os.path.join(vis_dir, f"{img_filename}_gradcam.png")
            save_visualization(image_np, cam, pred_class, confidence, ground_truth_idx,
                             class_names, top5_probs, top5_indices, vis_path)
            vis_count += 1
        else:
            cam = None
        
        # Record results
        is_correct = pred_class == ground_truth_idx
        correct += int(is_correct)
        total += 1
        
        results.append({
            'image_path': image_path,
            'image_filename': os.path.basename(image_path),
            'ground_truth': ground_truth_label,
            'predicted': class_names[pred_class],
            'confidence': confidence,
            'correct': is_correct
        })
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(output_dir, 'inference_results.csv')
    results_df.to_csv(results_csv, index=False)
    
    # Per-class accuracy
    class_results = {}
    for cls in class_names:
        cls_df = results_df[results_df['ground_truth'] == cls]
        if len(cls_df) > 0:
            cls_acc = (cls_df['correct'].sum() / len(cls_df)) * 100
            class_results[cls] = {
                'total': len(cls_df),
                'correct': cls_df['correct'].sum(),
                'accuracy': cls_acc
            }
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"INFERENCE COMPLETE")
    print(f"{'='*70}")
    print(f"Total samples: {total}")
    print(f"Overall accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"\nPer-class results:")
    for cls, metrics in class_results.items():
        print(f"  {cls}: {metrics['accuracy']:.1f}% ({metrics['correct']}/{metrics['total']})")
    print(f"\nResults saved to: {results_csv}")
    if save_vis:
        print(f"Visualizations saved to: {vis_dir} ({vis_count} samples)")
    print(f"{'='*70}")
    
    return results_df, accuracy


def main():
    parser = argparse.ArgumentParser(description='Classification inference with Grad-CAM')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--csv', type=str, required=True, help='Path to test CSV file')
    parser.add_argument('--output-dir', type=str, default='inference_results_gradcam', 
                       help='Output directory for results')
    parser.add_argument('--num-classes', type=int, default=9, help='Number of classes')
    parser.add_argument('--image-size', type=int, default=384, help='Input image size')
    parser.add_argument('--max-vis', type=int, default=50, 
                       help='Maximum number of visualizations to save')
    parser.add_argument('--no-vis', action='store_true', 
                       help='Disable visualization saving (faster)')
    parser.add_argument('--class-names', type=str, nargs='+',
                       default=['giant cell tumor', 'multiple osteochondromas', 'osteochondroma',
                               'osteofibroma', 'osteosarcoma', 'other bt', 'other mt',
                               'simple bone cyst', 'synovial osteochondroma'],
                       help='Class names')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = ConvNeXtTinyStudent(num_classes=args.num_classes, pretrained=False)
    load_checkpoint(model, args.checkpoint, map_location=device)
    model = model.to(device)
    print("Model loaded successfully")
    
    # Run inference
    infer_with_gradcam(
        model=model,
        csv_path=args.csv,
        output_dir=args.output_dir,
        device=device,
        class_names=args.class_names,
        image_size=args.image_size,
        save_vis=not args.no_vis,
        max_vis=args.max_vis
    )


if __name__ == '__main__':
    main()
