"""Pydantic schemas for image upload."""

from pydantic import BaseModel


class UploadResponse(BaseModel):
    image_id: str
    filename: str
    url: str
