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
import torchvision.transforms as T

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
        
        self.hook_handles.append(
            target_layer.register_forward_hook(self._save_activation)
        )
        self.hook_handles.append(
            target_layer.register_full_backward_hook(self._save_gradient)
        )
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, image_tensor, target_class):
        self.model.eval()
        
        image_tensor.requires_grad = True
        output = self.model(image_tensor)
        
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()
        
        if self.gradients is None or self.activations is None:
            return np.zeros((image_tensor.shape[2], image_tensor.shape[3]), dtype=np.float32)
        
        # Move to CPU for numpy conversion
        gradients = self.gradients.cpu().numpy()[0]
        activations = self.activations.cpu().numpy()[0]
        
        # Compute weights as mean of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of activation maps
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


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean


def apply_colormap(cam, image_np):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)
    return heatmap, overlay


def inference_with_visualization(model, image_path, ground_truth_label, output_path, 
                                 class_names, device, image_size=384):
    transform = get_transform(image_size)
    
    image = Image.open(image_path).convert('RGB')
    image_resized = image.resize((image_size, image_size))
    image_np = np.array(image_resized)
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)[0]
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()
    
    # Move to CPU for numpy operations
    probs = probs.cpu()
    
    target_layer = None
    for name, module in model.model.named_modules():
        if 'stages.3' in name and isinstance(module, torch.nn.Conv2d):
            target_layer = module
    
    if target_layer is None:
        for name, module in model.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
    
    if target_layer is not None:
        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate(image_tensor, pred_class)
        gradcam.remove_hooks()
    else:
        cam = np.zeros((image_size, image_size), dtype=np.float32)
    
    top5_probs, top5_indices = torch.topk(probs, min(5, len(class_names)))
    
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image_resized)
    ax1.set_title('Original X-ray', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(cam, cmap='jet')
    ax2.set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    heatmap, cam_overlay = apply_colormap(cam, image_np)
    ax3.imshow(cam_overlay)
    ax3.set_title('Grad-CAM Overlay', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis('off')
    pred_text = f"Predicted: {class_names[pred_class]}\n"
    pred_text += f"Confidence: {confidence:.2%}\n\n"
    pred_text += f"Ground Truth: {class_names[ground_truth_label]}\n"
    correct = pred_class == ground_truth_label
    pred_text += f"Status: {'CORRECT' if correct else 'INCORRECT'}"
    ax4.text(0.1, 0.5, pred_text, fontsize=13, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen' if correct else 'lightcoral', alpha=0.8))
    ax4.set_title('Prediction Summary', fontsize=14, fontweight='bold')
    
    ax5 = fig.add_subplot(gs[1, :])
    colors = ['green' if i == ground_truth_label else 'skyblue' for i in top5_indices]
    bars = ax5.barh(range(len(top5_indices)), top5_probs.numpy(), color=colors)
    ax5.set_yticks(range(len(top5_indices)))
    ax5.set_yticklabels([class_names[i] for i in top5_indices])
    ax5.set_xlabel('Probability', fontsize=12)
    ax5.set_title('Top-5 Predictions', fontsize=14, fontweight='bold')
    ax5.set_xlim(0, 1)
    
    for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
        ax5.text(prob + 0.02, i, f'{prob:.2%}', va='center', fontsize=10)
    
    title = f"Classification Result: {class_names[pred_class]} ({confidence:.1%})"
    if correct:
        title += " ✓"
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'predicted_class': class_names[pred_class],
        'predicted_idx': pred_class,
        'confidence': confidence,
        'ground_truth': class_names[ground_truth_label],
        'correct': correct,
        'top5_classes': [class_names[i] for i in top5_indices],
        'top5_probs': top5_probs.cpu().numpy().tolist()
    }


def main():
    checkpoint_path = 'outputs/kd_student/best_model.pth'
    image_size = 384
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    class_names = [
        'giant cell tumor',
        'multiple osteochondromas', 
        'osteochondroma',
        'osteofibroma',
        'osteosarcoma',
        'other bt',
        'other mt',
        'simple bone cyst',
        'synovial osteochondroma'
    ]
    
    print("Loading student model...")
    model = ConvNeXtTinyStudent(num_classes=len(class_names), pretrained=False)
    load_checkpoint(model, checkpoint_path, map_location=device)
    model = model.to(device)
    print(f"Model loaded on {device}")
    
    test_cases = [
        {'image': r'C:\Users\Nauman\Desktop\vistai\FYP\BTXRD\augmented_classification_data\test\giant cell tumor\IMG000466.jpeg', 'label': 0, 'name': 'giant_cell_tumor_1'},
        {'image': r'C:\Users\Nauman\Desktop\vistai\FYP\BTXRD\augmented_classification_data\test\multiple osteochondromas\IMG000298.jpeg', 'label': 1, 'name': 'multiple_osteochondromas_1'},
        {'image': r'C:\Users\Nauman\Desktop\vistai\FYP\BTXRD\augmented_classification_data\test\osteochondroma\IMG000266.jpeg', 'label': 2, 'name': 'osteochondroma_1'},
        {'image': r'C:\Users\Nauman\Desktop\vistai\FYP\BTXRD\augmented_classification_data\test\osteofibroma\IMG000413.jpeg', 'label': 3, 'name': 'osteofibroma_1'},
        {'image': r'C:\Users\Nauman\Desktop\vistai\FYP\BTXRD\augmented_classification_data\test\osteosarcoma\IMG000013.jpeg', 'label': 4, 'name': 'osteosarcoma_1'},
        {'image': r'C:\Users\Nauman\Desktop\vistai\FYP\BTXRD\augmented_classification_data\test\other bt\IMG000588.jpeg', 'label': 5, 'name': 'other_bt_1'},
        {'image': r'C:\Users\Nauman\Desktop\vistai\FYP\BTXRD\augmented_classification_data\test\other mt\IMG000012.jpeg', 'label': 6, 'name': 'other_mt_1'},
        {'image': r'C:\Users\Nauman\Desktop\vistai\FYP\BTXRD\augmented_classification_data\test\simple bone cyst\IMG000288.jpeg', 'label': 7, 'name': 'simple_bone_cyst_1'},
        {'image': r'C:\Users\Nauman\Desktop\vistai\FYP\BTXRD\augmented_classification_data\test\synovial osteochondroma\IMG001131.jpeg', 'label': 8, 'name': 'synovial_osteochondroma_1'},
        {'image': r'C:\Users\Nauman\Desktop\vistai\FYP\BTXRD\augmented_classification_data\test\giant cell tumor\IMG000781.jpeg', 'label': 0, 'name': 'giant_cell_tumor_2'},
    ]
    
    os.makedirs('classification_results', exist_ok=True)
    
    results = []
    correct_count = 0
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Processing {case['name']}...")
        
        if not os.path.exists(case['image']):
            print(f"  Warning: Image not found, skipping...")
            continue
        
        output_path = f"classification_results/{case['name']}_visualization.png"
        
        try:
            result = inference_with_visualization(
                model, case['image'], case['label'], output_path,
                class_names, device, image_size
            )
            
            results.append({
                'name': case['name'],
                'predicted': result['predicted_class'],
                'ground_truth': result['ground_truth'],
                'confidence': result['confidence'],
                'correct': result['correct']
            })
            
            if result['correct']:
                correct_count += 1
            
            status = "CORRECT" if result['correct'] else "INCORRECT"
            print(f"  Predicted: {result['predicted_class']} ({result['confidence']:.2%})")
            print(f"  Ground Truth: {result['ground_truth']}")
            print(f"  Status: {status}")
            print(f"  Saved to {output_path}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "="*70)
    print("INFERENCE COMPLETE - Summary")
    print("="*70)
    
    if results:
        accuracy = correct_count / len(results)
        print(f"Test Accuracy: {accuracy:.2%} ({correct_count}/{len(results)})")
        print(f"\nPer-Class Results:")
        
        for r in results:
            status_icon = "✓" if r['correct'] else "✗"
            print(f"  {status_icon} {r['name']}: {r['predicted']} ({r['confidence']:.1%})")
        
        print(f"\nTotal images processed: {len(results)}/{len(test_cases)}")
        print(f"Output directory: classification_results/")
    else:
        print("No images were processed successfully.")
    
    print("="*70)


if __name__ == '__main__':
    main()
