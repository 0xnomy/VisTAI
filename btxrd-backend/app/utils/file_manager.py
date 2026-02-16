"""
File manager – handles upload storage, path resolution, cleanup.
"""

import os
import uuid
from PIL import Image

from app.config import get_settings


def save_upload(file_bytes: bytes, original_filename: str) -> tuple[str, str, str]:
    """
    Save uploaded bytes to disk.
    Returns (image_id, saved_filename, relative_url).
    """
    settings = get_settings()
    upload_dir = settings.resolved_upload_dir
    os.makedirs(upload_dir, exist_ok=True)

    ext = os.path.splitext(original_filename)[1].lower() or ".png"
    image_id = uuid.uuid4().hex
    saved_name = f"{image_id}{ext}"
    fpath = os.path.join(upload_dir, saved_name)

    with open(fpath, "wb") as f:
        f.write(file_bytes)

    return image_id, saved_name, f"/files/{saved_name}"


def get_upload_path(image_id: str) -> str | None:
    """Resolve an image_id back to its file path on disk."""
    settings = get_settings()
    upload_dir = settings.resolved_upload_dir
    for fname in os.listdir(upload_dir):
        if fname.startswith(image_id) and not os.path.isdir(os.path.join(upload_dir, fname)):
            return os.path.join(upload_dir, fname)
    return None


def load_image(image_id: str) -> Image.Image:
    """Load a previously uploaded image as PIL Image."""
    path = get_upload_path(image_id)
    if path is None:
        raise FileNotFoundError(f"Image {image_id} not found in uploads.")
    return Image.open(path).convert("RGB")
