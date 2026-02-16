"""
BTXRD Backend – Application Configuration
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── Paths ──────────────────────────────────────────────────────────────
    btxrd_project_root: str = os.getenv(
        "BTXRD_PROJECT_ROOT",
        str(Path(__file__).resolve().parents[2] / "BTXRD"),
    )

    # ── Server ─────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # ── LLM ────────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    local_llm_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # ── Inference ──────────────────────────────────────────────────────────
    cls_image_size: int = 384
    seg_image_size: int = 224
    seg_threshold: float = 0.5

    # ── Uploads ────────────────────────────────────────────────────────────
    upload_dir: str = ""
    max_upload_size_mb: int = 20

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def cls_checkpoint(self) -> str:
        return os.path.join(
            self.btxrd_project_root,
            "combined_inference", "models", "classification_student.pth",
        )

    @property
    def seg_checkpoint(self) -> str:
        return os.path.join(
            self.btxrd_project_root,
            "combined_inference", "models", "segmentation_student.pth",
        )

    @property
    def resolved_upload_dir(self) -> str:
        if self.upload_dir:
            return self.upload_dir
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "uploads",
        )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
