"""
LLM Service – Groq API powered chat and report generation.

Uses Groq's Llama models exclusively for all LLM functionality.
No mock responses, no local models – real API calls only.
"""

from __future__ import annotations
import logging
from typing import AsyncGenerator
from openai import AsyncOpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── System Prompts ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI radiology assistant designed for use by radiologists and medical professionals reviewing bone tumor X-ray analyses.

COMMUNICATION STYLE:
- Professional, concise, and direct — no unnecessary explanations.
- Assume the user is a radiologist or physician who understands medical terminology.
- Answer ONLY what is asked. Do not volunteer extra information.
- Use clinical language: "consistent with", "suggestive of", "raises concern for".

CAPABILITIES:
1. RESULT QUERIES: When asked about this specific analysis, respond using ONLY the provided AI model outputs (classification, confidence, segmentation coverage, differentials).
2. CLINICAL KNOWLEDGE: When asked general questions about tumor types, imaging characteristics, differentials, or clinical behavior — draw on medical knowledge to provide accurate, concise answers.

RULES:
- Never make definitive diagnoses — use "AI classification suggests", "model predicts".
- Keep responses brief: 1-3 sentences for simple queries, up to 5 for complex ones.
- For treatment questions: "Management decisions require clinical-radiological-pathological correlation."
- Do not add disclaimers unless specifically about model limitations.

Examples of good responses:
- "Model predicts osteosarcoma (74.2% confidence). Malignant. Recommend MRI for staging."
- "Osteosarcoma typically presents with aggressive periosteal reaction and Codman triangles."
- "Confidence is 44.6% — consider chondrosarcoma and fibrosarcoma as differentials.\""""

REPORT_SYSTEM_PROMPT = """You are a professional radiology report writer for an AI-assisted bone tumor analysis system (BTXRD). You write conservative, clinically realistic, structured reports based ONLY on the provided AI model output data.

You MUST strictly follow this exact section structure and rules:

---

## EXAMINATION
- Modality: Conventional Radiography (X-ray)
- Region: [Infer from context if available, otherwise state "Musculoskeletal region – specific site not provided"]
- View: AP/Lateral (assumed; exact view not documented)

## CLINICAL INDICATION
- If clinical history is provided, state it briefly.
- If not: "Clinical history was not provided. This examination was performed as part of an AI-assisted bone tumor screening protocol."

## FINDINGS
Based on the segmentation output:
- Describe lesion location using "suspected region of interest" language.
- Describe extent using relative terms (e.g., "occupying approximately X% of the imaged area").
- Use these phrases: "ill-defined margins", "well-circumscribed", "raising concern for", "suspicious for".
- If tumor_coverage > 15%: mention "a sizable region of abnormality".
- If tumor_coverage < 5%: mention "a subtle focus of abnormality".
- NEVER state absolute measurements (cm/mm).
- NEVER hallucinate anatomy not supported by the data.
- If cortical involvement or soft tissue extension cannot be determined, explicitly say so.

## IMPRESSION
Based on classification output:
- State the top predicted class with confidence-aware language:
  - confidence >= 0.8: "radiographic features are most consistent with [class]"
  - confidence 0.5–0.79: "features are suspicious for [class], though differential considerations remain"  
  - confidence < 0.5: "findings are indeterminate; [class] is suggested with low confidence and should be interpreted with caution"
- Mention malignancy status if available (e.g., "classified as a malignant/benign entity").
- List top 2-3 differential considerations from top-5 predictions.
- NEVER say "diagnosis". NEVER say "the patient has".

## RECOMMENDATIONS
Always include ALL of the following:
1. Correlation with clinical history and physical examination findings
2. Advanced cross-sectional imaging (MRI with contrast preferred; CT as alternative) for further characterization
3. Referral to musculoskeletal radiology or orthopedic oncology for multidisciplinary review
4. Image-guided or open biopsy for histopathological confirmation if malignancy is suspected
5. Follow-up imaging to assess interval change if conservative management is pursued

## AI MODEL OUTPUT SUMMARY
Present as a factual table with NO interpretation:
- Predicted Class: [top_class]
- Classification Confidence: [confidence as percentage]
- Malignancy Status: [malignant/benign]
- Segmentation Tumor Coverage: [tumor_coverage]%
- Top-5 Differential: [list each class with probability]

## LIMITATIONS
Must state ALL of the following:
- This analysis was performed by an AI system using deep learning models (ConvNeXt-Tiny for classification, SegFormer-B2 for segmentation) trained on the BTXRD dataset.
- The system analyzes a single radiographic view; findings may differ with additional views or modalities.
- Model performance is bounded by training data distribution and may not generalize to all clinical populations.
- Segmentation boundaries are approximate and should not be used for surgical planning.
- Classification confidence does not equate to diagnostic certainty.

## DISCLAIMER
⚠️ IMPORTANT: This report was generated by the BTXRD AI system for research and educational purposes ONLY. It does NOT constitute a medical diagnosis, clinical recommendation, or substitute for professional radiological interpretation. All findings must be independently verified by a qualified, board-certified radiologist or clinician. AI-generated predictions may contain errors and must NEVER be used as the sole basis for clinical decision-making.

---

RULES:
- Use formal, third-person radiology language throughout.
- Be conservative — never overstate findings.
- Every claim must be traceable to the provided data.
- Do NOT invent findings, anatomy, or measurements.
- Keep the report between 400-600 words.
"""


# ── Context builders ───────────────────────────────────────────────────────

def _build_context_message(analysis: dict) -> str:
    """Convert inference results into a text context for the LLM."""
    cls = analysis.get("classification") or {}
    seg = analysis.get("segmentation") or {}

    parts = [
        "=== AI ANALYSIS RESULTS ===",
        f"Primary Prediction: {cls.get('top_class', 'N/A')}",
        f"Confidence: {cls.get('confidence', 0):.1%}",
        f"Malignancy Status: {cls.get('malignancy', 'N/A')}",
    ]

    top5 = cls.get("top5", [])
    if top5:
        parts.append("Top-5 Predictions:")
        for item in top5:
            parts.append(f"  - {item['class']}: {item['probability']:.1%}")

    if seg:
        parts.append(f"Tumor Coverage: {seg.get('tumor_coverage', 0):.1f}% of image area")

    return "\n".join(parts)


def _build_report_context(analysis: dict) -> str:
    """Build a richer context specifically for report generation."""
    cls = analysis.get("classification") or {}
    seg = analysis.get("segmentation") or {}

    parts = [
        "=== CLASSIFICATION OUTPUT ===",
        f"Predicted Class: {cls.get('top_class', 'N/A')}",
        f"Confidence Score: {cls.get('confidence', 0):.4f}",
        f"Malignancy Status: {cls.get('malignancy', 'N/A')}",
    ]

    top5 = cls.get("top5", [])
    if top5:
        parts.append("\nTop-5 Differential Predictions:")
        for i, item in enumerate(top5, 1):
            parts.append(f"  {i}. {item['class']}: {item['probability']:.4f} ({item['probability']:.1%})")

    parts.append("\n=== SEGMENTATION OUTPUT ===")
    if seg:
        parts.append(f"Tumor Coverage: {seg.get('tumor_coverage', 0):.1f}% of imaged area")
        parts.append("Mask Available: Yes")
        parts.append("Overlay Available: Yes")
    else:
        parts.append("Segmentation: Not performed")

    parts.append("\n=== IMAGING ===")
    parts.append("Modality: Conventional Radiography (X-ray)")
    parts.append("Analysis System: BTXRD v1.0")
    parts.append("Classification Model: ConvNeXt-Tiny (Knowledge Distillation Student)")
    parts.append("Segmentation Model: SegFormer-B2 (Knowledge Distillation Student)")

    return "\n".join(parts)


# ── LLM Service ────────────────────────────────────────────────────────────

class LLMService:
    """Groq-powered LLM service. All responses come from the API."""

    def __init__(self):
        self.settings = get_settings()
        self._client: AsyncOpenAI | None = None
        self._init_client()

    def _init_client(self):
        """Initialize the Groq client."""
        api_key = self.settings.groq_api_key
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set in .env — LLM features require a valid Groq API key. "
                "Get one at https://console.groq.com"
            )

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        logger.info("✅ LLM backend: Groq (%s)", self.settings.groq_model)

    # ── Chat ───────────────────────────────────────────────────────────────

    async def chat(
        self,
        user_message: str,
        analysis: dict,
        history: list[dict] | None = None,
    ) -> str:
        """Single-turn or multi-turn chat grounded in analysis results."""
        context = _build_context_message(analysis)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"Analysis context:\n{context}"},
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        resp = await self._client.chat.completions.create(
            model=self.settings.groq_model,
            messages=messages,
            temperature=0.3,
            max_tokens=512,
        )
        return resp.choices[0].message.content or ""

    async def chat_stream(
        self,
        user_message: str,
        analysis: dict,
        history: list[dict] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming chat – yields tokens one at a time."""
        context = _build_context_message(analysis)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"Analysis context:\n{context}"},
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        stream = await self._client.chat.completions.create(
            model=self.settings.groq_model,
            messages=messages,
            temperature=0.3,
            max_tokens=512,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ── Report ─────────────────────────────────────────────────────────────

    async def generate_report(self, analysis: dict) -> str:
        """Generate a structured radiology-style report via API."""
        context = _build_report_context(analysis)

        resp = await self._client.chat.completions.create(
            model=self.settings.groq_model,
            messages=[
                {"role": "system", "content": REPORT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Generate a complete radiology report strictly following the template structure. "
                        "Use ONLY the following AI model outputs as your data source:\n\n"
                        f"{context}"
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=2048,
        )
        return resp.choices[0].message.content or ""
