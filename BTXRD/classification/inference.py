"""
Classification Inference
========================
Run inference with any classification model (student or teacher).

Usage:
    python inference.py --model efficientnet_b0 --checkpoint outputs/efficientnet_b0/checkpoints/best_model.pth
    python inference.py --model convnext_small --checkpoint outputs/convnext_small_teacher/checkpoint_best.pth --visualize
"""

import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import sys

# Project imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from classification.models import build_model, list_models
from common.transforms import get_classification_transforms


class ClassificationInference:
    """Handles inference for classification models."""
    
    def __init__(self, model_name: str, checkpoint_path: str, device: str = None):
        """
        Initialize inference engine.
        
        Args:
            model_name: Name of model ('efficientnet_b0', 'convnext_small', etc.)
            checkpoint_path: Path to model checkpoint
            device: Device to use (auto-detect if None)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        # Load label encoding
        self.label_encoding_path = project_root / 'label_encoding.json'
        with open(self.label_encoding_path, 'r') as f:
            encoding = json.load(f)
            self.label_to_idx = encoding['label_to_idx']
            self.idx_to_label = {int(k): v for k, v in encoding['idx_to_label'].items()}
            self.num_classes = encoding['num_classes']
        
        # Build model
        print(f"\n📦 Loading {model_name} model...")
        self.model = build_model(model_name, num_classes=self.num_classes, pretrained=False)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Print checkpoint info
        if 'epoch' in checkpoint:
            print(f"   Loaded from epoch {checkpoint['epoch']}")
        if 'metrics' in checkpoint and 'val_accuracy' in checkpoint['metrics']:
            print(f"   Validation accuracy: {checkpoint['metrics']['val_accuracy']:.2f}%")
        
        # Get transforms
        self.transform = get_classification_transforms(phase='test')
    
    def predict(self, image_path: str, top_k: int = 3):
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to image
            top_k: Number of top predictions to return
        
        Returns:
            dict with prediction results
        """
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(image_tensor)
            probs = F.softmax(output, dim=1)
            
            # Top-k predictions
            top_probs, top_indices = torch.topk(probs[0], top_k)
            
            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                predictions.append({
                    'class': self.idx_to_label[idx.item()],
                    'confidence': prob.item() * 100
                })
        
        return {
            'image_path': image_path,
            'predicted_class': predictions[0]['class'],
            'confidence': predictions[0]['confidence'],
            'top_k': predictions
        }
    
    def predict_batch(self, csv_path: str, num_samples: int = 10):
        """
        Run inference on multiple images from CSV.
        
        Args:
            csv_path: Path to CSV with image_path and labels columns
            num_samples: Number of samples to process
        
        Returns:
            list of prediction results with accuracy
        """
        df = pd.read_csv(csv_path)
        
        # Sample stratified by class
        samples = []
        for class_name in df['labels'].unique():
            class_df = df[df['labels'] == class_name]
            n = min(num_samples // len(df['labels'].unique()) + 1, len(class_df))
            samples.append(class_df.sample(n=n, random_state=42))
        
        sample_df = pd.concat(samples).head(num_samples).reset_index(drop=True)
        
        results = []
        correct = 0
        
        for _, row in sample_df.iterrows():
            pred = self.predict(row['image_path'])
            pred['true_label'] = row['labels']
            pred['correct'] = pred['predicted_class'] == row['labels']
            if pred['correct']:
                correct += 1
            results.append(pred)
        
        accuracy = 100.0 * correct / len(results)
        return results, accuracy
    
    def visualize_predictions(self, results: list, output_path: str = None):
        """
        Visualize predictions in a grid.
        
        Args:
            results: List of prediction results from predict_batch
            output_path: Optional path to save the figure
        """
        n = len(results)
        cols = min(5, n)
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        axes = [ax for row in ([axes] if rows == 1 else axes) for ax in (row if hasattr(row, '__iter__') else [row])]
        
        for idx, (ax, result) in enumerate(zip(axes, results)):
            # Load and display image
            img = Image.open(result['image_path']).convert('RGB')
            ax.imshow(img)
            ax.axis('off')
            
            # Color based on correctness
            color = 'green' if result['correct'] else 'red'
            
            # Title
            title = f"True: {result['true_label']}\n"
            title += f"Pred: {result['predicted_class']} ({result['confidence']:.1f}%)"
            ax.set_title(title, fontsize=10, color=color)
            
            # Border
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
        
        # Hide empty axes
        for ax in axes[len(results):]:
            ax.axis('off')
        
        plt.suptitle(f"Model: {self.model_name} | Accuracy: {sum(r['correct'] for r in results)}/{len(results)}", 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved visualization to {output_path}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Classification Inference')
    parser.add_argument('--model', type=str, default='efficientnet_b0',
                        help='Model name (efficientnet_b0, convnext_small, etc.)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                        help='Single image path for inference')
    parser.add_argument('--csv', type=str, default=None,
                        help='CSV file with test images')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples to process from CSV')
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
    inference = ClassificationInference(
        model_name=args.model,
        checkpoint_path=args.checkpoint
    )
    
    # Single image inference
    if args.image:
        result = inference.predict(args.image)
        print(f"\n🔍 Prediction for {Path(args.image).name}:")
        print(f"   Class: {result['predicted_class']}")
        print(f"   Confidence: {result['confidence']:.2f}%")
        print(f"   Top-3:")
        for i, pred in enumerate(result['top_k'], 1):
            print(f"      {i}. {pred['class']}: {pred['confidence']:.2f}%")
        return
    
    # Batch inference from CSV
    csv_path = args.csv or (project_root / 'augmented_classification_data/augmented_test.csv')
    print(f"\n📊 Running batch inference on {csv_path}...")
    
    results, accuracy = inference.predict_batch(str(csv_path), num_samples=args.num_samples)
    
    print(f"\n✅ Results ({len(results)} samples):")
    print(f"   Accuracy: {accuracy:.2f}%")
    print(f"   Correct: {sum(r['correct'] for r in results)}/{len(results)}")
    
    # Print individual results
    print("\n   Details:")
    for i, r in enumerate(results, 1):
        status = "✓" if r['correct'] else "✗"
        print(f"   {i:2}. {status} True: {r['true_label']:25} Pred: {r['predicted_class']:25} ({r['confidence']:.1f}%)")
    
    if args.visualize:
        output_path = args.output or f"inference_results_{args.model}.png"
        inference.visualize_predictions(results, output_path)


if __name__ == '__main__':
    main()
