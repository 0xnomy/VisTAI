"""Pydantic schemas for inference requests / responses."""

from __future__ import annotations
from pydantic import BaseModel
from typing import Optional


class InferenceRequest(BaseModel):
    image_id: str
    intent: str = "full"  # "classification" | "segmentation" | "full"


class ClassificationResult(BaseModel):
    top_class: str
    confidence: float
    malignancy: str
    probabilities: dict[str, float]
    top5: list[dict]


class SegmentationResult(BaseModel):
    mask_url: str
    overlay_url: str
    gradcam_url: str
    tumor_coverage: float


class InferenceResponse(BaseModel):
    image_id: str
    classification: Optional[ClassificationResult] = None
    segmentation: Optional[SegmentationResult] = None
    cls_gradcam_url: Optional[str] = None
    original_url: str
