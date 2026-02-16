"""
Embedding Utilities for VLM Operations
=======================================

Utilities for combining, comparing, and manipulating CLIP embeddings.
"""

import numpy as np
import torch
import torch.nn.functional as F


def combine_embeddings(image_emb, text_emb, method='concat'):
    """
    Combine image and text embeddings into unified representation.
    
    Args:
        image_emb: [D] image embedding vector
        text_emb: [D] text embedding vector
        method: 'concat', 'average', 'weighted'
        
    Returns:
        combined_emb: Combined embedding vector
    """
    if isinstance(image_emb, np.ndarray):
        image_emb = torch.from_numpy(image_emb)
    if isinstance(text_emb, np.ndarray):
        text_emb = torch.from_numpy(text_emb)
    
    if method == 'concat':
        # Simple concatenation [2D]
        combined = torch.cat([image_emb, text_emb], dim=-1)
    
    elif method == 'average':
        # Average (maintains dimensionality) [D]
        combined = (image_emb + text_emb) / 2.0
        combined = F.normalize(combined, dim=-1)
    
    elif method == 'weighted':
        # Weighted combination (70% image, 30% text for medical images)
        combined = 0.7 * image_emb + 0.3 * text_emb
        combined = F.normalize(combined, dim=-1)
    
    else:
        raise ValueError(f"Unknown combination method: {method}")
    
    return combined.numpy()


def compute_similarity(emb1, emb2, metric='cosine'):
    """
    Compute similarity between two embeddings.
    
    Args:
        emb1: [D] embedding vector
        emb2: [D] embedding vector
        metric: 'cosine', 'euclidean', 'dot'
        
    Returns:
        similarity: Scalar similarity score
    """
    if isinstance(emb1, np.ndarray):
        emb1 = torch.from_numpy(emb1)
    if isinstance(emb2, np.ndarray):
        emb2 = torch.from_numpy(emb2)
    
    if metric == 'cosine':
        # Cosine similarity [-1, 1]
        emb1_norm = F.normalize(emb1, dim=-1)
        emb2_norm = F.normalize(emb2, dim=-1)
        sim = (emb1_norm * emb2_norm).sum().item()
    
    elif metric == 'dot':
        # Dot product (for normalized embeddings, same as cosine)
        sim = (emb1 * emb2).sum().item()
    
    elif metric == 'euclidean':
        # Negative euclidean distance (higher = more similar)
        dist = torch.norm(emb1 - emb2).item()
        sim = -dist
    
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")
    
    return sim


def batch_similarity(query_emb, database_embs, metric='cosine', top_k=5):
    """
    Compute similarity between query and database of embeddings.
    
    Args:
        query_emb: [D] query embedding
        database_embs: [N, D] database embeddings
        metric: Similarity metric
        top_k: Number of top matches to return
        
    Returns:
        top_indices: [top_k] indices of most similar embeddings
        top_scores: [top_k] similarity scores
    """
    if isinstance(query_emb, np.ndarray):
        query_emb = torch.from_numpy(query_emb)
    if isinstance(database_embs, np.ndarray):
        database_embs = torch.from_numpy(database_embs)
    
    # Normalize for cosine similarity
    query_norm = F.normalize(query_emb.unsqueeze(0), dim=-1)
    database_norm = F.normalize(database_embs, dim=-1)
    
    # Compute similarities
    if metric == 'cosine' or metric == 'dot':
        similarities = torch.matmul(query_norm, database_norm.T).squeeze(0)
    elif metric == 'euclidean':
        # Negative distances
        dists = torch.cdist(query_norm.unsqueeze(0), database_norm).squeeze(0)
        similarities = -dists
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Get top-k
    top_scores, top_indices = torch.topk(similarities, min(top_k, len(similarities)))
    
    return top_indices.numpy(), top_scores.numpy()


def normalize_embedding(emb):
    """
    Normalize embedding to unit length.
    
    Args:
        emb: [D] or [N, D] embedding(s)
        
    Returns:
        normalized_emb: Same shape, unit normalized
    """
    if isinstance(emb, np.ndarray):
        emb = torch.from_numpy(emb)
    
    emb_norm = F.normalize(emb, dim=-1)
    
    return emb_norm.numpy()


def embedding_statistics(embeddings):
    """
    Compute statistics over a collection of embeddings.
    
    Args:
        embeddings: [N, D] array of embeddings
        
    Returns:
        stats: Dictionary with mean, std, etc.
    """
    if isinstance(embeddings, list):
        embeddings = np.stack(embeddings)
    
    return {
        'mean': embeddings.mean(axis=0),
        'std': embeddings.std(axis=0),
        'l2_norm_mean': np.linalg.norm(embeddings, axis=1).mean(),
        'l2_norm_std': np.linalg.norm(embeddings, axis=1).std(),
        'dimensionality': embeddings.shape[1],
        'num_samples': embeddings.shape[0]
    }


if __name__ == "__main__":
    # Test embedding utilities
    print("\n" + "="*70)
    print("Testing Embedding Utilities")
    print("="*70)
    
    # Create mock embeddings
    np.random.seed(42)
    image_emb = np.random.randn(512).astype(np.float32)
    text_emb = np.random.randn(512).astype(np.float32)
    
    # Normalize
    image_emb = normalize_embedding(image_emb)
    text_emb = normalize_embedding(text_emb)
    
    print(f"\nImage embedding shape: {image_emb.shape}")
    print(f"Text embedding shape: {text_emb.shape}")
    
    # Test combination methods
    print("\nCombination methods:")
    for method in ['concat', 'average', 'weighted']:
        combined = combine_embeddings(image_emb, text_emb, method=method)
        print(f"  {method}: shape={combined.shape}")
    
    # Test similarity
    print("\nSimilarity metrics:")
    for metric in ['cosine', 'dot', 'euclidean']:
        sim = compute_similarity(image_emb, text_emb, metric=metric)
        print(f"  {metric}: {sim:.4f}")
    
    # Test batch similarity
    print("\nBatch similarity (top-3):")
    database = np.random.randn(10, 512).astype(np.float32)
    database = normalize_embedding(database)
    
    top_idx, top_scores = batch_similarity(image_emb, database, top_k=3)
    for idx, score in zip(top_idx, top_scores):
        print(f"  Rank {idx}: similarity={score:.4f}")
    
    print("\n✓ Embedding utilities test passed")
    print("="*70)
