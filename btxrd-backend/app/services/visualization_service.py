"""
Visualization Service – generates overlay images, saves them to disk,
and returns file paths / URLs the frontend can load.
"""

import os
import uuid
import numpy as np
import cv2
from PIL import Image

from app.config import get_settings


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_image_array(arr: np.ndarray, subdir: str = "results", prefix: str = "vis") -> str:
    """Save a numpy RGB array as PNG. Returns the relative URL path."""
    settings = get_settings()
    out_dir = os.path.join(settings.resolved_upload_dir, subdir)
    _ensure_dir(out_dir)
    fname = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    fpath = os.path.join(out_dir, fname)
    img = Image.fromarray(arr.astype(np.uint8))
    img.save(fpath)
    return f"/files/{subdir}/{fname}"


def save_mask_png(mask: np.ndarray, subdir: str = "results", prefix: str = "mask") -> str:
    """Save a binary 0/1 mask as a PNG. Returns relative URL path."""
    settings = get_settings()
    out_dir = os.path.join(settings.resolved_upload_dir, subdir)
    _ensure_dir(out_dir)
    fname = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    fpath = os.path.join(out_dir, fname)
    Image.fromarray((mask * 255).astype(np.uint8)).save(fpath)
    return f"/files/{subdir}/{fname}"


def create_mask_overlay(
    image: Image.Image,
    mask: np.ndarray,
    size: int = 224,
    color: tuple = (46, 134, 171),
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay a binary mask on an image with the given color."""
    img_np = np.array(image.convert("RGB").resize((size, size))).astype(np.float32)
    mask_resized = cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    mask_bool = mask_resized > 0
    img_np[mask_bool] = img_np[mask_bool] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    return img_np.astype(np.uint8)


def create_gradcam_overlay(
    image: Image.Image,
    cam: np.ndarray,
    size: int = 224,
    colormap: int = cv2.COLORMAP_PLASMA,
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on an image."""
    img_np = np.array(image.convert("RGB").resize((size, size)))
    cam_resized = cv2.resize(cam, (size, size))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)
    return overlay
