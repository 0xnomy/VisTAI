"""Vision-Language Model module"""
from .clip_encoder import CLIPEncoder, encode_image, encode_text
from .prompt_builder import build_medical_prompt
from .embedding_utils import combine_embeddings, compute_similarity

__all__ = [
    'CLIPEncoder',
    'encode_image',
    'encode_text',
    'build_medical_prompt',
    'combine_embeddings',
    'compute_similarity'
]
