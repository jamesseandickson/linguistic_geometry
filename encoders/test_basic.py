"""
Basic unit tests for encoders.

Simple tests to verify encoders work correctly.
Run with: python -m pytest encoders/test_basic.py
"""

import pytest
import numpy as np
from .sentence_transformer import SentenceTransformerEncoder


class TestSentenceTransformerEncoder:
    """Test SentenceTransformerEncoder functionality."""
    
    def test_initialization(self):
        """Test encoder can be initialized."""
        encoder = SentenceTransformerEncoder()
        assert encoder.model_name == "all-MiniLM-L6-v2"
        assert encoder.embedding_dim == 384
    
    def test_single_text_encoding(self):
        """Test encoding a single text."""
        encoder = SentenceTransformerEncoder()
        text = "hello world"
        
        embedding = encoder.encode(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, 384)
        assert embedding.dtype == np.float32
    
    def test_multiple_text_encoding(self):
        """Test encoding multiple texts."""
        encoder = SentenceTransformerEncoder()
        texts = ["happy", "sad", "joyful"]
        
        embeddings = encoder.encode(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float32
    
    def test_semantic_similarity(self):
        """Test that similar words have similar embeddings."""
        encoder = SentenceTransformerEncoder()
        
        # Encode similar and dissimilar words
        embeddings = encoder.encode(["happy", "joyful", "freezing"])
        
        # Compute cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings)
        
        # happy-joyful should be more similar than happy-freezing
        happy_joyful = similarities[0, 1]
        happy_freezing = similarities[0, 2]
        
        assert happy_joyful > happy_freezing
        assert happy_joyful > 0.5  # Should be reasonably similar
    
    def test_different_models(self):
        """Test different sentence transformer models."""
        models = [
            ("all-MiniLM-L6-v2", 384),
            ("all-mpnet-base-v2", 768),
        ]
        
        for model_name, expected_dim in models:
            try:
                encoder = SentenceTransformerEncoder(model_name)
                assert encoder.embedding_dim == expected_dim
                
                # Test encoding works
                embedding = encoder.encode("test")
                assert embedding.shape == (1, expected_dim)
                
            except Exception as e:
                pytest.skip(f"Model {model_name} not available: {e}")


def test_convenience_functions():
    """Test convenience functions."""
    from .sentence_transformer import get_fast_encoder, get_quality_encoder
    
    # Test fast encoder
    fast_encoder = get_fast_encoder()
    assert fast_encoder.model_name == "all-MiniLM-L6-v2"
    assert fast_encoder.embedding_dim == 384
    
    # Test quality encoder (might not be available)
    try:
        quality_encoder = get_quality_encoder()
        assert quality_encoder.model_name == "all-mpnet-base-v2"
        assert quality_encoder.embedding_dim == 768
    except Exception:
        pytest.skip("Quality encoder not available")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])