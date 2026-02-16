"""
CLIP-based Vision-Language Encoder
===================================

Uses OpenCLIP as a FROZEN encoder to create aligned image-text embeddings.
This is NOT used for classification - only for creating a multimodal representation space.

Design Rationale:
-----------------
1. Use pretrained CLIP (RN50 or ViT-B/32) without fine-tuning
2. Encode X-ray images into 512-dim or 768-dim space
3. Encode structured medical facts into same space
4. Enable similarity-based retrieval and grounding

The VLM does NOT make diagnostic decisions - it enriches representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
import open_clip


class CLIPEncoder:
    """
    Frozen CLIP encoder for medical image-text alignment.
    
    This encoder is used to create embeddings, NOT for classification.
    It provides a shared semantic space for images and text.
    """
    
    def __init__(self, model_name='RN50', pretrained='openai', device=None):
        """
        Initialize CLIP encoder.
        
        Args:
            model_name: CLIP architecture ('RN50', 'ViT-B-32', 'ViT-B-16')
            pretrained: Checkpoint name ('openai', 'laion400m_e32', etc.)
            device: torch device
            
        Note: All parameters are frozen by default.
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pretrained CLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained
        )
        
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # Move to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Get embedding dimension
        self.embed_dim = self.model.text_projection.shape[1]
        
        print(f"✓ CLIP Encoder initialized: {model_name}")
        print(f"  Embedding dimension: {self.embed_dim}")
        print(f"  All parameters frozen (no fine-tuning)")
        print(f"  Device: {self.device}")
    
    def encode_image(self, image_input):
        """
        Encode image into CLIP embedding space.
        
        Args:
            image_input: PIL Image, torch.Tensor, or path string
            
        Returns:
            image_embedding: [embed_dim] normalized vector
        """
        # Handle different input types
        if isinstance(image_input, str) or isinstance(image_input, Path):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input
        elif isinstance(image_input, torch.Tensor):
            # Already preprocessed
            image_tensor = image_input.to(self.device)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        # Preprocess if needed
        if not isinstance(image_input, torch.Tensor):
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Encode
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            # Normalize to unit sphere
            image_features = F.normalize(image_features, dim=-1)
        
        return image_features.squeeze(0).cpu().numpy()
    
    def encode_text(self, text):
        """
        Encode text into CLIP embedding space.
        
        Args:
            text: String or list of strings
            
        Returns:
            text_embedding: [embed_dim] or [N, embed_dim] normalized vectors
        """
        # Tokenize
        if isinstance(text, str):
            text = [text]
        
        text_tokens = self.tokenizer(text).to(self.device)
        
        # Encode
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            # Normalize
            text_features = F.normalize(text_features, dim=-1)
        
        result = text_features.cpu().numpy()
        return result[0] if len(text) == 1 else result
    
    def compute_similarity(self, image_emb, text_emb):
        """
        Compute cosine similarity between image and text embeddings.
        
        Args:
            image_emb: [embed_dim] vector
            text_emb: [embed_dim] vector
            
        Returns:
            similarity: scalar in [-1, 1]
        """
        if isinstance(image_emb, np.ndarray):
            image_emb = torch.from_numpy(image_emb)
        if isinstance(text_emb, np.ndarray):
            text_emb = torch.from_numpy(text_emb)
        
        # Cosine similarity (already normalized)
        similarity = (image_emb * text_emb).sum().item()
        return similarity


# Convenience functions
def encode_image(encoder, image_input):
    """Encode image using CLIP encoder."""
    return encoder.encode_image(image_input)


def encode_text(encoder, text):
    """Encode text using CLIP encoder."""
    return encoder.encode_text(text)


def create_clip_encoder(model_name='RN50', device=None):
    """
    Factory function to create CLIP encoder.
    
    Recommended models for edge deployment:
    - 'RN50': 38M params, 512-dim embeddings
    - 'RN50x4': Better accuracy but 4x slower
    - 'ViT-B-32': 88M params, 512-dim, good balance
    
    For production edge deployment, use RN50.
    """
    return CLIPEncoder(model_name=model_name, device=device)


if __name__ == "__main__":
    # Test CLIP encoder
    print("\n" + "="*70)
    print("Testing CLIP Encoder")
    print("="*70)
    
    # Create encoder
    encoder = create_clip_encoder(model_name='RN50')
    
    # Test text encoding
    medical_texts = [
        "An X-ray showing osteosarcoma tumor in the bone",
        "A benign bone cyst visible in radiograph",
        "Malignant tumor with irregular boundaries"
    ]
    
    print("\nEncoding medical texts...")
    for text in medical_texts:
        emb = encoder.encode_text(text)
        print(f"  '{text[:40]}...' → [{emb.shape[0]}-dim vector]")
    
    # Test similarity
    text_emb1 = encoder.encode_text(medical_texts[0])
    text_emb2 = encoder.encode_text(medical_texts[1])
    text_emb3 = encoder.encode_text(medical_texts[2])
    
    sim_12 = encoder.compute_similarity(text_emb1, text_emb2)
    sim_13 = encoder.compute_similarity(text_emb1, text_emb3)
    
    print(f"\nSemantic Similarity:")
    print(f"  Text 1 vs Text 2 (different tumors): {sim_12:.3f}")
    print(f"  Text 1 vs Text 3 (both malignant): {sim_13:.3f}")
    
    print("\n✓ CLIP Encoder test passed")
    print("="*70)
