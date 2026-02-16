"""
Segmentation Service – wraps the SegFormer-B2 KD student model.
Loads checkpoint from BTXRD/combined_inference/models/ and exposes
`predict(image)` / `grad_cam(image)` interfaces.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from app.config import Settings


# ── Self-contained model definition ───────────────────────────────────────
class _SegFormerB2Student(nn.Module):
    def __init__(self, num_classes: int = 1, image_size: int = 224):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation, SegformerConfig

        config = SegformerConfig.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512"
        )
        config.num_labels = num_classes
        self.model = SegformerForSemanticSegmentation(config)
        self.image_size = image_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]
        out = self.model(x, return_dict=True)
        logits = F.interpolate(
            out.logits, size=input_size, mode="bilinear", align_corners=False
        )
        return logits


# ── Grad-CAM ──────────────────────────────────────────────────────────────
class _GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations = self.gradients = None
        self._hooks = [
            target_layer.register_forward_hook(self._fwd),
            target_layer.register_full_backward_hook(self._bwd),
        ]

    def _fwd(self, m, i, o):
        self.activations = o.detach()

    def _bwd(self, m, gi, go):
        self.gradients = go[0].detach()

    @torch.enable_grad()
    def __call__(self, tensor: torch.Tensor) -> np.ndarray:
        self.model.zero_grad()
        out = self.model(tensor)
        scalar = out.mean()
        scalar.backward()
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


# ── Helpers ────────────────────────────────────────────────────────────────
def _preprocess(image: Image.Image, size: int = 224) -> tuple[torch.Tensor, np.ndarray]:
    img = image.convert("RGB").resize((size, size))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    return tensor, np.array(img)


def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    inter = (p & g).sum()
    total = p.sum() + g.sum()
    return float(2.0 * inter / total) if total > 0 else (1.0 if inter == 0 else 0.0)


def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    inter = (p & g).sum()
    union = (p | g).sum()
    return float(inter / union) if union > 0 else (1.0 if inter == 0 else 0.0)


# ── Public service ─────────────────────────────────────────────────────────
class SegmentationService:
    """Thread-safe, singleton segmentation service."""

    def __init__(self, settings: Settings):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = settings.seg_image_size
        self.threshold = settings.seg_threshold
        self.model = self._load(settings.seg_checkpoint)
        self._cam_layer = self._find_cam_layer()

    def _load(self, ckpt_path: str) -> nn.Module:
        model = _SegFormerB2Student(num_classes=1, image_size=self.image_size)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        return model.to(self.device).eval()

    def _find_cam_layer(self) -> nn.Module:
        layer = None
        for name, mod in self.model.named_modules():
            if "decode_head" in name and isinstance(mod, nn.Conv2d):
                layer = mod
        if layer is None:
            for _, mod in self.model.named_modules():
                if isinstance(mod, nn.Conv2d):
                    layer = mod
        return layer

    # ── public API ─────────────────────────────────────────────────────────
    def predict(self, image: Image.Image) -> dict:
        """
        Run segmentation on a PIL Image.

        Returns dict with keys:
          mask (np uint8 H×W, 0 or 1),
          tumor_coverage (float percentage),
          image_size (int)
        """
        tensor, _ = _preprocess(image, self.image_size)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            prob = torch.sigmoid(logits)
            mask = (prob > self.threshold).cpu().numpy()[0, 0].astype(np.uint8)

        coverage = float(mask.sum()) / float(mask.size) * 100.0
        return {
            "mask": mask,
            "tumor_coverage": round(coverage, 2),
            "image_size": self.image_size,
        }

    def grad_cam(self, image: Image.Image) -> np.ndarray:
        """Return H×W Grad-CAM heatmap in [0,1]."""
        tensor, _ = _preprocess(image, self.image_size)
        tensor = tensor.to(self.device)
        gc = _GradCAM(self.model, self._cam_layer)
        cam = gc(tensor)
        gc.remove()
        return cam

    def grad_cam_overlay(self, image: Image.Image) -> np.ndarray:
        """Return an RGB overlay (numpy uint8) of Grad-CAM on the image."""
        cam = self.grad_cam(image)
        size = self.image_size
        img_np = np.array(image.convert("RGB").resize((size, size)))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_VIRIDIS)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)
        return overlay

    def make_mask_overlay(self, image: Image.Image, mask: np.ndarray,
                          color: tuple = (46, 134, 171), alpha: float = 0.5) -> np.ndarray:
        """Overlay binary mask on image with a given color."""
        size = self.image_size
        img_np = np.array(image.convert("RGB").resize((size, size))).astype(np.float32)
        mask_bool = mask > 0
        img_np[mask_bool] = (
            img_np[mask_bool] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
        )
        return img_np.astype(np.uint8)
