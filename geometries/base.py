"""
Base class for all geometric structures.

Defines the interface that all geometries must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class Geometry(ABC):
    """
    Abstract base class for geometric structures.
    
    Each geometry represents a hypothesis about how linguistic concepts
    are organized in embedding space.
    """
    
    def __init__(self, dim: int, **kwargs):
        """
        Initialize geometry.
        
        Args:
            dim: Dimensionality of the geometry
            **kwargs: Geometry-specific parameters
        """
        self.dim = dim
        self.params = kwargs
        self._fitted = False
    
    @abstractmethod
    def fit(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Fit the geometry to a set of embeddings.
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
        
        Returns:
            Dictionary of fit metrics (e.g., distortion, reconstruction_error)
        """
        pass
    
    @abstractmethod
    def project(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Project embeddings into this geometry.
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
        
        Returns:
            Projected embeddings of shape (n_samples, self.dim)
        """
        pass
    
    @abstractmethod
    def reconstruct(self, projected: np.ndarray) -> np.ndarray:
        """
        Reconstruct original embeddings from projected form.
        
        Args:
            projected: Array of shape (n_samples, self.dim)
        
        Returns:
            Reconstructed embeddings of shape (n_samples, embedding_dim)
        """
        pass
    
    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute distance between two points in this geometry.
        
        Default: Euclidean distance. Override for non-Euclidean geometries.
        
        Args:
            x: Point of shape (self.dim,)
            y: Point of shape (self.dim,)
        
        Returns:
            Distance between x and y
        """
        return np.linalg.norm(x - y)
    
    def compute_distortion(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Compute distortion between original and reconstructed embeddings.
        
        Args:
            original: Original embeddings of shape (n_samples, embedding_dim)
            reconstructed: Reconstructed embeddings of shape (n_samples, embedding_dim)
        
        Returns:
            Mean squared error
        """
        return float(np.mean((original - reconstructed) ** 2))
    
    def __repr__(self):
        return f"{self.__class__.__name__}(dim={self.dim})"
