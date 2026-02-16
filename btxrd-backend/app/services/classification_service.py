"""
Classification Service – wraps the ConvNeXt-Tiny KD student model.
Loads the checkpoint from BTXRD/combined_inference/models/ and exposes
a simple `predict(image)` interface.  Also provides Grad-CAM.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from app.config import Settings

# ── Label map ──────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "giant cell tumor",
    "multiple osteochondromas",
    "osteochondroma",
    "osteofibroma",
    "osteosarcoma",
    "other bt",
    "other mt",
    "simple bone cyst",
    "synovial osteochondroma",
]

MALIGNANCY_MAP = {
    "giant cell tumor": "benign (locally aggressive)",
    "multiple osteochondromas": "benign",
    "osteochondroma": "benign",
    "osteofibroma": "benign",
    "osteosarcoma": "malignant",
    "other bt": "benign",
    "other mt": "malignant",
    "simple bone cyst": "benign",
    "synovial osteochondroma": "benign",
}


# ── Self-contained model definition ───────────────────────────────────────
class _ConvNeXtTinyStudent(nn.Module):
    def __init__(self, num_classes: int = 9):
        super().__init__()
        import timm
        self.model = timm.create_model(
            "convnext_tiny", pretrained=False, num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ── Grad-CAM ──────────────────────────────────────────────────────────────
class _GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations = self.gradients = None
        self._hooks = [
            target_layer.register_forward_hook(self._fwd_hook),
            target_layer.register_full_backward_hook(self._bwd_hook),
        ]

    def _fwd_hook(self, m, i, o):
        self.activations = o.detach()

    def _bwd_hook(self, m, gi, go):
        self.gradients = go[0].detach()

    @torch.enable_grad()
    def __call__(self, tensor: torch.Tensor, target_class: int) -> np.ndarray:
        self.model.zero_grad()
        out = self.model(tensor)
        score = out[0, target_class]
        score.backward()
        if self.activations is None or self.gradients is None:
            return np.zeros((tensor.shape[2], tensor.shape[3]), np.float32)
        w = self.gradients.cpu().numpy()[0].mean(axis=(1, 2))
        a = self.activations.cpu().numpy()[0]
        cam = np.maximum((w[:, None, None] * a).sum(0), 0)
        if cam.max() > 0:
            cam /= cam.max()
        cam = cv2.resize(cam, (tensor.shape[3], tensor.shape[2]))
        return cam

    def remove(self):
        for h in self._hooks:
            h.remove()


# ── Public service ─────────────────────────────────────────────────────────
class ClassificationService:
    """Thread-safe, singleton classification service."""

    def __init__(self, settings: Settings):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = settings.cls_image_size
        self.model = self._load(settings.cls_checkpoint)
        self.transform = T.Compose([
            T.Resize(416),
            T.CenterCrop(self.image_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        # Find target layer for Grad-CAM (last Conv2d in stage 3)
        self._cam_layer = self._find_cam_layer()

    # ── private helpers ────────────────────────────────────────────────────
    def _load(self, ckpt_path: str) -> nn.Module:
        model = _ConvNeXtTinyStudent(num_classes=9)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        sd = {
            (k.replace("backbone.", "") if k.startswith("backbone.") else k): v
            for k, v in sd.items()
        }
        model.load_state_dict(sd, strict=False)
        return model.to(self.device).eval()

    def _find_cam_layer(self) -> nn.Module:
        layer = None
        for name, mod in self.model.named_modules():
            if "stages.3" in name and isinstance(mod, nn.Conv2d):
                layer = mod
        if layer is None:
            for _, mod in self.model.named_modules():
                if isinstance(mod, nn.Conv2d):
                    layer = mod
        return layer

    # ── public API ─────────────────────────────────────────────────────────
    def predict(self, image: Image.Image) -> dict:
        """
        Run classification on a PIL Image.

        Returns dict with keys:
          top_class, confidence, malignancy,
          probabilities (dict class→float),
          top5 (list of {class, probability})
        """
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(probs.argmax())
        top5_idx = probs.argsort()[::-1][:5]

        return {
            "top_class": CLASS_NAMES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "malignancy": MALIGNANCY_MAP.get(CLASS_NAMES[pred_idx], "unknown"),
            "probabilities": {
                CLASS_NAMES[i]: round(float(probs[i]), 5) for i in range(len(CLASS_NAMES))
            },
            "top5": [
                {"class": CLASS_NAMES[int(i)], "probability": round(float(probs[int(i)]), 5)}
                for i in top5_idx
            ],
        }

    def grad_cam(self, image: Image.Image) -> np.ndarray:
        """Return H×W Grad-CAM heatmap in [0, 1] for the top predicted class."""
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            pred_idx = int(logits.argmax(dim=1).item())

        gc = _GradCAM(self.model, self._cam_layer)
        cam = gc(tensor, pred_idx)
        gc.remove()
        return cam

    def grad_cam_overlay(self, image: Image.Image, size: int = 384) -> np.ndarray:
        """Return an RGB overlay (numpy uint8) of Grad-CAM on the image."""
        cam = self.grad_cam(image)
        img_np = np.array(image.convert("RGB").resize((size, size)))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_PLASMA)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)
        return overlay
