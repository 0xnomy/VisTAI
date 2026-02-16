"""
Inference API – POST /api/inference
Runs classification and/or segmentation on a previously uploaded image.
"""

from fastapi import APIRouter, HTTPException
import cv2

from app.schemas.inference import InferenceRequest, InferenceResponse, ClassificationResult, SegmentationResult
from app.utils.file_manager import load_image
from app.services import visualization_service as vis_svc
from app import main as app_main  # to access singletons

router = APIRouter()


@router.post("/inference", response_model=InferenceResponse)
async def run_inference(req: InferenceRequest):
    """
    Run AI inference on a previously uploaded X-ray image.

    **intent** options:
    - `"classification"` – tumor type prediction only
    - `"segmentation"` – tumor region mask only
    - `"full"` – both classification + segmentation (default)
    """
    # Validate models are loaded
    if app_main.cls_service is None or app_main.seg_service is None:
        raise HTTPException(status_code=503, detail="Models are still loading. Try again shortly.")

    # Load the image
    try:
        image = load_image(req.image_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Image {req.image_id} not found.")

    intent = req.intent.lower()
    cls_result = None
    seg_result = None
    cls_gradcam_url = None

    # ── Classification ─────────────────────────────────────────────────────
    if intent in ("classification", "full"):
        cls_raw = app_main.cls_service.predict(image)
        cls_result = ClassificationResult(**cls_raw)

        # Grad-CAM overlay
        cls_cam_overlay = app_main.cls_service.grad_cam_overlay(image, size=384)
        cls_gradcam_url = vis_svc.save_image_array(cls_cam_overlay, subdir="results", prefix="cls_gradcam")

    # ── Segmentation ───────────────────────────────────────────────────────
    if intent in ("segmentation", "full"):
        seg_raw = app_main.seg_service.predict(image)
        mask = seg_raw["mask"]

        # Save mask PNG
        mask_url = vis_svc.save_mask_png(mask, subdir="results", prefix="mask")

        # Save overlay
        overlay_arr = vis_svc.create_mask_overlay(image, mask, size=224)
        overlay_url = vis_svc.save_image_array(overlay_arr, subdir="results", prefix="seg_overlay")

        # Grad-CAM
        seg_cam = app_main.seg_service.grad_cam(image)
        seg_cam_overlay = vis_svc.create_gradcam_overlay(image, seg_cam, size=224, colormap=cv2.COLORMAP_VIRIDIS)
        seg_gradcam_url = vis_svc.save_image_array(seg_cam_overlay, subdir="results", prefix="seg_gradcam")

        seg_result = SegmentationResult(
            mask_url=mask_url,
            overlay_url=overlay_url,
            gradcam_url=seg_gradcam_url,
            tumor_coverage=seg_raw["tumor_coverage"],
        )

    # Original image URL
    original_url = f"/files/{req.image_id}"
    # Find actual extension
    from app.utils.file_manager import get_upload_path
    import os
    upath = get_upload_path(req.image_id)
    if upath:
        original_url = f"/files/{os.path.basename(upath)}"

    return InferenceResponse(
        image_id=req.image_id,
        classification=cls_result,
        segmentation=seg_result,
        cls_gradcam_url=cls_gradcam_url,
        original_url=original_url,
    )
