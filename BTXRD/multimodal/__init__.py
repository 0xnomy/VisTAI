"""
Multimodal Module for VLM-powered Medical Image Chat
=====================================================

This module adds vision-language understanding and chat capabilities
to the existing bone tumor classification/segmentation pipeline.

CRITICAL: This module does NOT replace or retrain existing CNN models.
It provides a reasoning layer on top of frozen CNN predictions.

Components:
-----------
- vlm/: Vision-Language Model (CLIP) for image-text encoding
- llm/: Language Model (Phi-2/TinyLlama) for grounded chat
- retrieval/: Similarity search over medical facts
- pipeline.py: Orchestrator connecting all components
"""

__version__ = "1.0.0"
__author__ = "BTXRD Team"
