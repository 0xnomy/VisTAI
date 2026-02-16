import os
import argparse
import numpy as np
from PIL import Image
import torch
from models.student.segformer_b2 import SegFormerB2
from utils.checkpoint import load_checkpoint


def infer_single_image(model, image_path, device, threshold=0.5, image_size=224):
    model.eval()
    
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    image = image.resize((image_size, image_size))
    
    image_array = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = (image_array - mean) / std
    
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        logits = model(image_tensor)
        prob = torch.sigmoid(logits)
        mask = (prob > threshold).cpu().numpy()[0, 0]
    
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    mask_pil = mask_pil.resize(original_size, Image.NEAREST)
    
    return mask_pil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--output', type=str, default='output_mask.png')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    model = SegFormerB2(num_classes=1, image_size=args.image_size, pretrained=False)
    load_checkpoint(model, args.checkpoint)
    model = model.to(device)
    
    mask = infer_single_image(model, args.image, device, args.threshold, args.image_size)
    mask.save(args.output)
    print(f'Saved mask to {args.output}')


if __name__ == '__main__':
    main()
