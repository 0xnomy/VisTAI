"""
Report API – POST /api/report  &  GET /api/report/pdf/{filename}
Generates a structured radiology-style report from inference results,
including a downloadable PDF with embedded images.
"""

import os
import uuid
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import get_settings
from app.schemas.report import ReportRequest, ReportResponse
from app.services.llm_service import LLMService
from app.services.pdf_service import generate_pdf

logger = logging.getLogger(__name__)
router = APIRouter()

_llm: LLMService | None = None


def _get_llm() -> LLMService:
    global _llm
    if _llm is None:
        _llm = LLMService()
    return _llm


@router.post("/report", response_model=ReportResponse)
async def generate_report(req: ReportRequest):
    """
    Generate a professional radiology-style report based on inference results.
    Returns the markdown report text AND a URL to download the PDF.
    """
    llm = _get_llm()
    settings = get_settings()

    # 1. Generate report text via LLM
    try:
        report_text = await llm.generate_report(req.analysis)
    except Exception as e:
        logger.error("Report text generation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Report generation error: {str(e)}")

    # 2. Generate PDF with embedded images
    pdf_url = None
    try:
        pdf_bytes = generate_pdf(
            report_text=report_text,
            analysis=req.analysis,
            upload_dir=settings.resolved_upload_dir,
            case_id=req.image_id or "N/A",
        )

        # Save PDF to results directory
        results_dir = os.path.join(settings.resolved_upload_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        pdf_filename = f"report_{uuid.uuid4().hex[:8]}.pdf"
        pdf_path = os.path.join(results_dir, pdf_filename)

        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        pdf_url = f"/files/results/{pdf_filename}"
        logger.info("PDF report saved: %s", pdf_path)

    except Exception as e:
        logger.error("PDF generation failed: %s", e, exc_info=True)
        # Don't fail the whole request — still return the text report
        pdf_url = None

    return ReportResponse(report=report_text, pdf_url=pdf_url)
