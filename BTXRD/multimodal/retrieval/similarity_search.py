"""
Similarity Search Engine
========================

Performs similarity search over historical cases using VLM embeddings.
Enables retrieval-augmented generation for the LLM.

Design:
-------
1. Index historical cases with VLM embeddings
2. Given new image, find K most similar historical cases
3. Provide these as context to LLM for richer responses
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pickle


class SimilaritySearchEngine:
    """
    Similarity search over VLM embeddings.
    """
    
    def __init__(self, embedding_dim: int = 512):
        """
        Initialize search engine.
        
        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.database_embeddings = None
        self.database_metadata = []
        self.indexed = False
        
        print(f"✓ Similarity search engine initialized (dim={embedding_dim})")
    
    def index_cases(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict]
    ):
        """
        Index historical cases with embeddings.
        
        Args:
            embeddings: [N, D] array of case embeddings
            metadata: List of N metadata dictionaries
        """
        if embeddings.shape[0] != len(metadata):
            raise ValueError("Number of embeddings must match metadata length")
        
        self.database_embeddings = embeddings
        self.database_metadata = metadata
        self.indexed = True
        
        print(f"✓ Indexed {len(metadata)} cases")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_class: Optional[str] = None
    ) -> Tuple[List[int], List[float], List[Dict]]:
        """
        Search for K most similar cases.
        
        Args:
            query_embedding: [D] query embedding
            k: Number of results to return
            filter_class: Optional tumor class to filter by
            
        Returns:
            indices: List of K indices in database
            scores: List of K similarity scores
            metadata: List of K metadata dicts
        """
        if not self.indexed:
            raise RuntimeError("Search engine not indexed. Call index_cases() first.")
        
        # Normalize embeddings
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        db_norm = self.database_embeddings / (np.linalg.norm(self.database_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute cosine similarities
        similarities = np.dot(db_norm, query_norm)
        
        # Apply class filter if specified
        if filter_class:
            valid_indices = [
                i for i, meta in enumerate(self.database_metadata)
                if meta.get('tumor_class') == filter_class
            ]
            
            if not valid_indices:
                return [], [], []
            
            filtered_similarities = similarities[valid_indices]
            top_k_filtered = np.argsort(filtered_similarities)[-k:][::-1]
            
            indices = [valid_indices[i] for i in top_k_filtered]
            scores = filtered_similarities[top_k_filtered].tolist()
        else:
            # Get top-k globally
            top_k = np.argsort(similarities)[-k:][::-1]
            indices = top_k.tolist()
            scores = similarities[top_k].tolist()
        
        # Retrieve metadata
        metadata = [self.database_metadata[i] for i in indices]
        
        return indices, scores, metadata
    
    def search_by_tumor_class(
        self,
        tumor_class: str,
        k: int = 5
    ) -> Tuple[List[int], List[Dict]]:
        """
        Get examples of a specific tumor class.
        
        Args:
            tumor_class: Tumor type to search for
            k: Number of examples
            
        Returns:
            indices: List of indices
            metadata: List of metadata
        """
        if not self.indexed:
            raise RuntimeError("Search engine not indexed.")
        
        matching_indices = [
            i for i, meta in enumerate(self.database_metadata)
            if meta.get('tumor_class') == tumor_class
        ]
        
        if not matching_indices:
            return [], []
        
        # Sample k random examples
        selected = np.random.choice(
            matching_indices,
            size=min(k, len(matching_indices)),
            replace=False
        )
        
        indices = selected.tolist()
        metadata = [self.database_metadata[i] for i in indices]
        
        return indices, metadata
    
    def save_index(self, save_path: str):
        """
        Save indexed database to disk.
        
        Args:
            save_path: Path to save file
        """
        if not self.indexed:
            raise RuntimeError("Nothing to save. Index first.")
        
        data = {
            'embeddings': self.database_embeddings,
            'metadata': self.database_metadata,
            'embedding_dim': self.embedding_dim
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Index saved to {save_path}")
    
    def load_index(self, load_path: str):
        """
        Load indexed database from disk.
        
        Args:
            load_path: Path to index file
        """
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.database_embeddings = data['embeddings']
        self.database_metadata = data['metadata']
        self.embedding_dim = data['embedding_dim']
        self.indexed = True
        
        print(f"✓ Index loaded from {load_path}")
        print(f"  {len(self.database_metadata)} cases indexed")


def create_dummy_index(
    num_cases: int = 100,
    embedding_dim: int = 512,
    num_classes: int = 9
) -> SimilaritySearchEngine:
    """
    Create dummy index for testing.
    
    Args:
        num_cases: Number of cases to generate
        embedding_dim: Embedding dimension
        num_classes: Number of tumor classes
        
    Returns:
        Indexed search engine
    """
    # Generate random embeddings
    embeddings = np.random.randn(num_cases, embedding_dim).astype(np.float32)
    
    # Normalize
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Generate metadata
    tumor_classes = [
        'osteosarcoma', 'osteochondroma', 'giant cell tumor',
        'simple bone cyst', 'osteofibroma', 'multiple osteochondromas',
        'synovial osteochondroma', 'other bt', 'other mt'
    ]
    
    metadata = []
    for i in range(num_cases):
        meta = {
            'case_id': f'case_{i:04d}',
            'tumor_class': tumor_classes[i % num_classes],
            'confidence': 0.5 + 0.5 * np.random.rand()
        }
        metadata.append(meta)
    
    # Create and index
    engine = SimilaritySearchEngine(embedding_dim=embedding_dim)
    engine.index_cases(embeddings, metadata)
    
    return engine


if __name__ == "__main__":
    # Test similarity search
    print("\n" + "="*70)
    print("Testing Similarity Search Engine")
    print("="*70)
    
    # Create dummy index
    print("\nCreating dummy index with 50 cases...")
    engine = create_dummy_index(num_cases=50, embedding_dim=512)
    
    # Create query
    query_emb = np.random.randn(512).astype(np.float32)
    query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    
    # Search
    print("\nSearching for top-5 similar cases...")
    indices, scores, metadata = engine.search(query_emb, k=5)
    
    print("\nTop 5 results:")
    for idx, score, meta in zip(indices, scores, metadata):
        print(f"  Rank {idx}: {meta['tumor_class']} (similarity={score:.3f})")
    
    # Search with filter
    print("\nSearching for osteosarcoma cases only...")
    indices, scores, metadata = engine.search(
        query_emb,
        k=3,
        filter_class='osteosarcoma'
    )
    
    print(f"Found {len(indices)} osteosarcoma cases:")
    for idx, score, meta in zip(indices, scores, metadata):
        print(f"  Case {idx}: confidence={meta['confidence']:.2f}, similarity={score:.3f}")
    
    # Test save/load
    print("\nTesting save/load...")
    test_path = 'test_index.pkl'
    engine.save_index(test_path)
    
    new_engine = SimilaritySearchEngine(embedding_dim=512)
    new_engine.load_index(test_path)
    
    # Cleanup
    Path(test_path).unlink()
    
    print("\n✓ Similarity search test passed")
    print("="*70)
