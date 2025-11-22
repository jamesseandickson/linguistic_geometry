"""
Euclidean Geometry (ℝⁿ) - Standard vector embeddings.

From semantic_atlas.md:
"An arrow: magnitude + direction. A point in multi-dimensional semantic space."

Good for:
- Multi-dimensional static meanings
- Embeddings of "happy", "sad", "angry"
- "polite" vs "rude" in a politeness–stance–formality subspace
- "doctor", "lawyer", "teacher" in a profession manifold

This is the BASELINE - what LLMs already use.
"""

import numpy as np
from typing import Dict
from .base import Geometry


class EuclideanGeometry(Geometry):
    """
    Standard Euclidean vector space (ℝⁿ).
    
    Uses PCA for dimensionality reduction if target dim < embedding dim.
    This is the baseline geometry that LLMs already use.
    """
    
    def __init__(self, dim: int, **kwargs):
        """
        Initialize Euclidean geometry.
        
        Args:
            dim: Target dimensionality
            **kwargs: Additional parameters
        """
        super().__init__(dim, **kwargs)
        self.projection_matrix = None
        self.mean = None
    
    def fit(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Fit Euclidean geometry using PCA.
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
        
        Returns:
            Dictionary with:
                - variance_explained: Fraction of variance captured
                - reconstruction_error: MSE after projection and reconstruction
                - compression_ratio: Original dim / target dim
        """
        n_samples, embedding_dim = embeddings.shape
        
        # Center the data
        self.mean = np.mean(embeddings, axis=0)
        centered = embeddings - self.mean
        
        if self.dim >= embedding_dim:
            # No compression needed - use identity
            self.projection_matrix = np.eye(embedding_dim)
            variance_explained = 1.0
            reconstruction_error = 0.0
        else:
            # Use PCA for compression
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # Sort by eigenvalue (descending)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Keep top k components
            self.projection_matrix = eigenvectors[:, :self.dim]
            
            # Compute variance explained
            total_variance = np.sum(eigenvalues)
            explained_variance = np.sum(eigenvalues[:self.dim])
            variance_explained = explained_variance / total_variance
            
            self._fitted = True
            
            # Test reconstruction
            projected = self.project(embeddings)
            reconstructed = self.reconstruct(projected)
            reconstruction_error = self.compute_distortion(embeddings, reconstructed)
        
        return {
            'variance_explained': float(variance_explained),
            'reconstruction_error': float(reconstruction_error),
            'compression_ratio': embedding_dim / self.dim
        }
    
    def project(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Project embeddings into lower-dimensional Euclidean space.
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
        
        Returns:
            Projected embeddings of shape (n_samples, self.dim)
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before project()")
        
        centered = embeddings - self.mean
        projected = np.dot(centered, self.projection_matrix)
        return projected
    
    def reconstruct(self, projected: np.ndarray) -> np.ndarray:
        """
        Reconstruct embeddings from projected form.
        
        Args:
            projected: Array of shape (n_samples, self.dim)
        
        Returns:
            Reconstructed embeddings of shape (n_samples, embedding_dim)
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before reconstruct()")
        
        reconstructed = self.mean + np.dot(projected, self.projection_matrix.T)
        return reconstructed
    
    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Euclidean distance between two points.
        
        Args:
            x: Point of shape (self.dim,)
            y: Point of shape (self.dim,)
        
        Returns:
            Euclidean distance
        """
        return float(np.linalg.norm(x - y))
