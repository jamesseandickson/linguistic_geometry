"""
Sentence Transformer Encoder - Free, runs locally.

Uses the sentence-transformers library with models from HuggingFace.
No API key required, runs on CPU or GPU.

Popular models:
- all-MiniLM-L6-v2: Fast, 384 dimensions (default)
- all-mpnet-base-v2: Better quality, 768 dimensions
- paraphrase-multilingual-MiniLM-L12-v2: Multilingual, 384 dimensions
"""

from typing import List, Union
import numpy as np
from .base import BaseEncoder


class SentenceTransformerEncoder(BaseEncoder):
    """
    Encoder using sentence-transformers library.
    
    Free and open-source, runs locally without API keys.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """
        Initialize sentence transformer encoder.
        
        Args:
            model_name: HuggingFace model name (default: all-MiniLM-L6-v2)
            device: Device to run on ('cpu', 'cuda', or None for auto)
        
        Examples:
            >>> encoder = SentenceTransformerEncoder()  # Uses default model
            >>> encoder = SentenceTransformerEncoder("all-mpnet-base-v2")  # Better quality
        """
        super().__init__(model_name)
        
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model = SentenceTransformer(model_name, device=device)
        self._embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32, 
               show_progress_bar: bool = False, **kwargs) -> np.ndarray:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Single string or list of strings
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            **kwargs: Additional arguments passed to model.encode()
        
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        # Handle single string
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            **kwargs
        )
        
        return embeddings
    
    @property
    def embedding_dim(self) -> int:
        """Return embedding dimensionality."""
        return self._embedding_dim


# Convenience functions for common models
def get_fast_encoder() -> SentenceTransformerEncoder:
    """Get a fast, lightweight encoder (384D)."""
    return SentenceTransformerEncoder("all-MiniLM-L6-v2")


def get_quality_encoder() -> SentenceTransformerEncoder:
    """Get a high-quality encoder (768D)."""
    return SentenceTransformerEncoder("all-mpnet-base-v2")


def get_multilingual_encoder() -> SentenceTransformerEncoder:
    """Get a multilingual encoder (384D)."""
    return SentenceTransformerEncoder("paraphrase-multilingual-MiniLM-L12-v2")
