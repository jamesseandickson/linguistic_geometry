"""
Encoders - Convert text to embeddings using various models.

Available encoders:
- SentenceTransformerEncoder: Free, runs locally (sentence-transformers)
- OpenAIEncoder: Requires API key (text-embedding-3-small/large)
"""

from .base import BaseEncoder
from .sentence_transformer import SentenceTransformerEncoder

__all__ = [
    'BaseEncoder',
    'SentenceTransformerEncoder',
]
