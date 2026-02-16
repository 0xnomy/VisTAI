# Grad-CAM visualization for both classification and segmentation models

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


class GradCAM:
    # Grad-CAM implementation for any model
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, image_tensor, target_class=None):
        # Forward pass
        self.model.eval()
        output = self.model(image_tensor)
        
        # Backward pass
        self.model.zero_grad()
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Generate heatmap
        gradients = self.gradients.cpu().numpy()[0]
        activations = self.activations.cpu().numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (image_tensor.shape[3], image_tensor.shape[2]))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        return cam
    
    def visualize(self, image, cam, alpha=0.5):
        # Overlay heatmap on image
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy().transpose(1, 2, 0)
            image = (image * 255).astype(np.uint8)
        
        overlay = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        return overlay


def apply_gradcam_classification(model, image_tensor, target_layer_name='features.8'):
    # Apply Grad-CAM to EfficientNet classification model
    target_layer = dict([*model.named_modules()])[target_layer_name]
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(image_tensor)
    return cam


def apply_gradcam_segmentation(model, image_tensor):
    # Apply Grad-CAM to MobileNetV2-UNet segmentation model
    # Use encoder4 as target layer for best visualization
    gradcam = GradCAM(model, model.encoder4)
    cam = gradcam.generate(image_tensor)
    return cam
