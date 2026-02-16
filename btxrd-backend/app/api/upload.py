"""
Upload API – POST /api/upload
Accepts an X-ray image and returns an image_id for subsequent calls.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.schemas.upload import UploadResponse
from app.utils.file_manager import save_upload
from app.config import get_settings

router = APIRouter()

ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}


@router.post("/upload", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """
    Upload a bone-tumor X-ray image.

    Returns an `image_id` that must be passed to /api/inference, /api/chat, etc.
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Allowed: {', '.join(ALLOWED_TYPES)}",
        )

    settings = get_settings()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    contents = await file.read()

    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(contents) / 1e6:.1f} MB). "
                   f"Max allowed: {settings.max_upload_size_mb} MB.",
        )

    image_id, filename, url = save_upload(contents, file.filename or "upload.png")

    return UploadResponse(image_id=image_id, filename=filename, url=url)
