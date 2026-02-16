"""
Chat API – POST /api/chat
LLM-powered Q&A grounded in the inference results.
Supports streaming via SSE and non-streaming JSON.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_service import LLMService

router = APIRouter()

# Lazy singleton
_llm: LLMService | None = None


def _get_llm() -> LLMService:
    global _llm
    if _llm is None:
        _llm = LLMService()
    return _llm


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Send a message about the X-ray analysis and get an LLM response.

    The `analysis` field must contain the full inference result so the
    LLM can ground its response in actual predictions.
    """
    llm = _get_llm()
    history = [{"role": m.role, "content": m.content} for m in req.history]

    try:
        reply = await llm.chat(req.message, req.analysis, history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    return ChatResponse(reply=reply)


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Streaming chat endpoint – returns Server-Sent Events (SSE).
    The frontend should use EventSource or fetch with streaming.
    """
    llm = _get_llm()
    history = [{"role": m.role, "content": m.content} for m in req.history]

    async def event_generator():
        try:
            async for token in llm.chat_stream(req.message, req.analysis, history):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
