"""Pydantic schemas for report generation."""

from pydantic import BaseModel


class ReportRequest(BaseModel):
    image_id: str
    analysis: dict  # full inference result


class ReportResponse(BaseModel):
    report: str
    pdf_url: str | None = None
