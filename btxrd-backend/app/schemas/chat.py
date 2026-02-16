"""Pydantic schemas for chat endpoint."""

from __future__ import annotations
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    image_id: str
    message: str
    analysis: dict  # full inference result to ground the LLM
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    reply: str
