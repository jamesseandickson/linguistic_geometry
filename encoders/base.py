"""
Base encoder interface.
"""

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np


class BaseEncoder(ABC):
    """Abstract base class for all encoders."""
    
    def __init__(self, model_name: str):
        """
        Initialize encoder.
        
        Args:
            model_name: Name/identifier of the model to use
        """
        self.model_name = model_name
    
    @abstractmethod
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Single string or list of strings to encode
            **kwargs: Additional encoder-specific parameters
        
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimensionality of embeddings produced by this encoder."""
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model_name})"
