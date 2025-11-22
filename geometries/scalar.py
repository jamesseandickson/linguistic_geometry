"""
Scalar Geometry (ℝ) - 1D intensity representation.

From semantic_atlas.md:
"Single number. No direction, just 'how much'."

Good for:
- Sentiment intensity: good → excellent, bad → terrible
- Certainty: maybe → definitely
- Formality: informal → formal (if context fixed)
"""

import numpy as np
from typing import Dict
from .base import Geometry


class ScalarGeometry(Geometry):
    """
    1-dimensional scalar representation.
    
    Projects embeddings onto a single axis of maximum variance.
    Useful for testing whether a concept cluster is fundamentally 1D.
    """
    
    def __init__(self, dim: int = 1, **kwargs):
        """
        Initialize scalar geometry.
        
        Args:
            dim: Must be 1 for scalar geometry
            **kwargs: Additional parameters
        """
        if dim != 1:
            raise ValueError("ScalarGeometry must have dim=1")
        super().__init__(dim, **kwargs)
        self.projection_vector = None
        self.mean = None
    
    def fit(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Fit scalar geometry by finding principal component.
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
        
        Returns:
            Dictionary with:
                - variance_explained: How much variance the 1D projection captures
                - reconstruction_error: MSE after projection and reconstruction
        """
        # Center the data
        self.mean = np.mean(embeddings, axis=0)
        centered = embeddings - self.mean
        
        # Find first principal component
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Get largest eigenvector
        idx = np.argmax(eigenvalues)
        self.projection_vector = eigenvectors[:, idx]
        
        # Compute metrics
        total_variance = np.sum(eigenvalues)
        variance_explained = eigenvalues[idx] / total_variance
        
        self._fitted = True
        
        # Test reconstruction
        projected = self.project(embeddings)
        reconstructed = self.reconstruct(projected)
        reconstruction_error = self.compute_distortion(embeddings, reconstructed)
        
        return {
            'variance_explained': float(variance_explained),
            'reconstruction_error': float(reconstruction_error),
            'compression_ratio': embeddings.shape[1] / self.dim
        }
    
    def project(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Project embeddings onto scalar axis.
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
        
        Returns:
            Scalar values of shape (n_samples, 1)
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before project()")
        
        centered = embeddings - self.mean
        scalars = np.dot(centered, self.projection_vector)
        return scalars.reshape(-1, 1)
    
    def reconstruct(self, projected: np.ndarray) -> np.ndarray:
        """
        Reconstruct embeddings from scalar projection.
        
        Args:
            projected: Array of shape (n_samples, 1)
        
        Returns:
            Reconstructed embeddings of shape (n_samples, embedding_dim)
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before reconstruct()")
        
        scalars = projected.flatten()
        reconstructed = self.mean + np.outer(scalars, self.projection_vector)
        return reconstructed
    
    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute distance between two scalar values.
        
        Args:
            x: Scalar of shape (1,)
            y: Scalar of shape (1,)
        
        Returns:
            Absolute difference
        """
        return float(np.abs(x - y))
