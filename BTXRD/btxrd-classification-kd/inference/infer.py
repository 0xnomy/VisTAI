import os
import sys
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.student.convnext_tiny import ConvNeXtTinyStudent
from utils.checkpoint import load_checkpoint


def get_inference_transform(image_size=384):
    return T.Compose([
        T.Resize(416),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def infer_single(model, image_path, transform, device, class_names):
    model.eval()
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()
    
    return {
        'predicted_class': class_names[pred_class],
        'predicted_idx': pred_class,
        'confidence': confidence,
        'probabilities': {class_names[i]: probs[i].item() for i in range(len(class_names))}
    }


def infer_batch(model, image_dir, transform, device, class_names, output_csv=None):
    results = []
    
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        result = infer_single(model, img_path, transform, device, class_names)
        result['image_file'] = img_file
        results.append(result)
        print(f"{img_file}: {result['predicted_class']} ({result['confidence']:.4f})")
    
    if output_csv:
        df = pd.DataFrame([{
            'image_file': r['image_file'],
            'predicted_class': r['predicted_class'],
            'confidence': r['confidence']
        } for r in results])
        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--image-dir', type=str, help='Directory of images')
    parser.add_argument('--output-csv', type=str, default='predictions.csv')
    parser.add_argument('--num-classes', type=int, default=9)
    parser.add_argument('--image-size', type=int, default=384)
    parser.add_argument('--class-names', type=str, nargs='+', 
                       default=['giant cell tumor', 'multiple osteochondromas', 'osteochondroma',
                               'osteofibroma', 'osteosarcoma', 'other bt', 'other mt', 
                               'simple bone cyst', 'synovial osteochondroma'])
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ConvNeXtTinyStudent(num_classes=args.num_classes, pretrained=False)
    load_checkpoint(model, args.checkpoint, map_location=device)
    model = model.to(device)
    
    transform = get_inference_transform(args.image_size)
    
    if args.image:
        result = infer_single(model, args.image, transform, device, args.class_names)
        print(f"Predicted: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("\nAll probabilities:")
        for cls, prob in result['probabilities'].items():
            print(f"  {cls}: {prob:.4f}")
    
    elif args.image_dir:
        infer_batch(model, args.image_dir, transform, device, args.class_names, args.output_csv)
    
    else:
        print("Please provide either --image or --image-dir")


if __name__ == '__main__':
    main()
