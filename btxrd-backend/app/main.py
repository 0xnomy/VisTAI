"""
BTXRD Backend – FastAPI Application Entry Point
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.config import get_settings
from app.services.classification_service import ClassificationService
from app.services.segmentation_service import SegmentationService
from app.api import upload, inference, chat, report


# ── Shared singletons (loaded once at startup) ────────────────────────────
cls_service: ClassificationService | None = None
seg_service: SegmentationService | None = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load heavy ML models once on startup, release on shutdown."""
    global cls_service, seg_service
    settings = get_settings()

    print("⏳  Loading classification model …")
    cls_service = ClassificationService(settings)
    print("⏳  Loading segmentation model …")
    seg_service = SegmentationService(settings)
    print("✅  Models loaded and ready.\n")

    # Ensure upload & output dirs exist
    os.makedirs(settings.resolved_upload_dir, exist_ok=True)
    os.makedirs(os.path.join(settings.resolved_upload_dir, "results"), exist_ok=True)

    yield  # ← app runs here

    # Cleanup
    cls_service = None
    seg_service = None
    print("🛑  Models unloaded.")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="BTXRD – Bone Tumor X-Ray Detection API",
        description="Classification, segmentation, Grad-CAM, chat, and report generation for bone tumor X-rays.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # ── CORS ───────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ────────────────────────────────────────────────────────────
    app.include_router(upload.router, prefix="/api", tags=["Upload"])
    app.include_router(inference.router, prefix="/api", tags=["Inference"])
    app.include_router(chat.router, prefix="/api", tags=["Chat"])
    app.include_router(report.router, prefix="/api", tags=["Report"])

    # ── Serve uploaded/generated files ─────────────────────────────────────
    os.makedirs(settings.resolved_upload_dir, exist_ok=True)
    app.mount(
        "/files",
        StaticFiles(directory=settings.resolved_upload_dir),
        name="files",
    )

    @app.get("/health")
    async def health():
        return {"status": "ok", "models_loaded": cls_service is not None}

    return app


app = create_app()
