"""
Hyperbolic Geometry - Poincaré Ball Model (ℍⁿ).

From semantic_atlas.md:
"Hyperbolic space has negative curvature - parallel lines diverge.
Good for hierarchies, trees, taxonomies."

Good for:
- Hierarchical concepts: "animal" → "mammal" → "dog" → "poodle"
- Social structures: "authority", "subordinate", "peer"
- Abstract taxonomies: "emotion" → "positive" → "joy"
- Asymmetric relationships with depth

Key properties:
- Distance grows exponentially from origin
- More "room" near the boundary (good for tree-like structures)
- Natural representation of hierarchy and entailment
"""

import numpy as np
from typing import Dict
from .base import Geometry


class HyperbolicGeometry(Geometry):
    """
    Hyperbolic geometry using the Poincaré ball model.
    
    The Poincaré ball is the unit ball {x ∈ ℝⁿ : ||x|| < 1} with
    the Riemannian metric ds² = 4/(1-||x||²)² dx².
    
    This creates negative curvature - ideal for hierarchical structures.
    """
    
    def __init__(self, dim: int, curvature: float = 1.0, **kwargs):
        """
        Initialize hyperbolic geometry.
        
        Args:
            dim: Target dimensionality
            curvature: Curvature parameter (default: 1.0, higher = more curved)
            **kwargs: Additional parameters
        """
        super().__init__(dim, **kwargs)
        self.curvature = curvature
        self.projection_matrix = None
        self.mean = None
        self.scale = None
    
    def fit(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Fit hyperbolic geometry to embeddings.
        
        Strategy:
        1. Use PCA to reduce to target dimensionality (Euclidean step)
        2. Map to Poincaré ball using exponential map from origin
        3. Optimize to preserve pairwise distances in hyperbolic metric
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
        
        Returns:
            Dictionary with:
                - variance_explained: Fraction of variance captured by PCA
                - reconstruction_error: MSE after round-trip
                - compression_ratio: Original dim / target dim
                - hyperbolic_distortion: Distance preservation in hyperbolic metric
        """
        n_samples, embedding_dim = embeddings.shape
        
        # Step 1: PCA projection to target dimensionality
        self.mean = np.mean(embeddings, axis=0)
        centered = embeddings - self.mean
        
        if self.dim >= embedding_dim:
            self.projection_matrix = np.eye(embedding_dim)
            variance_explained = 1.0
        else:
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
        
        # Step 2: Determine scaling to map into Poincaré ball
        # Project all points and find max norm
        euclidean_projected = np.dot(centered, self.projection_matrix)
        max_norm = np.max(np.linalg.norm(euclidean_projected, axis=1))
        
        # Scale so all points fit comfortably in ball (leave margin)
        self.scale = 0.9 / (max_norm + 1e-8)
        
        self._fitted = True
        
        # Test reconstruction
        projected = self.project(embeddings)
        reconstructed = self.reconstruct(projected)
        reconstruction_error = self.compute_distortion(embeddings, reconstructed)
        
        # Compute hyperbolic distortion
        hyperbolic_distortion = self._compute_hyperbolic_distortion(embeddings, projected)
        
        return {
            'variance_explained': float(variance_explained),
            'reconstruction_error': float(reconstruction_error),
            'compression_ratio': embedding_dim / self.dim,
            'hyperbolic_distortion': float(hyperbolic_distortion),
            'scale_factor': float(self.scale)
        }
    
    def project(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Project embeddings into Poincaré ball.
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
        
        Returns:
            Points in Poincaré ball of shape (n_samples, self.dim)
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before project()")
        
        # PCA projection
        centered = embeddings - self.mean
        euclidean_proj = np.dot(centered, self.projection_matrix)
        
        # Scale to fit in ball
        scaled = euclidean_proj * self.scale
        
        # Apply exponential map to move to Poincaré ball
        # For small values, exp_map ≈ tanh(||x||) * x/||x||
        norms = np.linalg.norm(scaled, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        
        # Exponential map from origin
        hyperbolic = np.tanh(norms * np.sqrt(self.curvature)) * scaled / norms
        
        return hyperbolic
    
    def reconstruct(self, projected: np.ndarray) -> np.ndarray:
        """
        Reconstruct embeddings from Poincaré ball.
        
        Args:
            projected: Points in Poincaré ball of shape (n_samples, self.dim)
        
        Returns:
            Reconstructed embeddings of shape (n_samples, embedding_dim)
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before reconstruct()")
        
        # Inverse exponential map (logarithmic map)
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        
        # Log map back to tangent space
        euclidean = np.arctanh(np.clip(norms, -0.999, 0.999)) / np.sqrt(self.curvature) * projected / norms
        
        # Unscale
        unscaled = euclidean / self.scale
        
        # Inverse PCA projection
        reconstructed = self.mean + np.dot(unscaled, self.projection_matrix.T)
        
        return reconstructed
    
    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute hyperbolic distance in Poincaré ball.
        
        The Poincaré distance formula:
        d(x,y) = arcosh(1 + 2||x-y||²/((1-||x||²)(1-||y||²)))
        
        Args:
            x: Point in Poincaré ball of shape (self.dim,)
            y: Point in Poincaré ball of shape (self.dim,)
        
        Returns:
            Hyperbolic distance
        """
        # Compute norms
        norm_x_sq = np.sum(x**2)
        norm_y_sq = np.sum(y**2)
        norm_diff_sq = np.sum((x - y)**2)
        
        # Poincaré distance formula
        numerator = 2 * norm_diff_sq
        denominator = (1 - norm_x_sq) * (1 - norm_y_sq)
        
        # Avoid numerical issues
        denominator = max(denominator, 1e-8)
        
        distance = np.arccosh(1 + numerator / denominator)
        return float(distance / np.sqrt(self.curvature))
    
    def _compute_hyperbolic_distortion(self, embeddings: np.ndarray, projected: np.ndarray) -> float:
        """
        Measure how well hyperbolic distances preserve original Euclidean distances.
        
        Args:
            embeddings: Original embeddings
            projected: Projected points in Poincaré ball
        
        Returns:
            Mean relative distance distortion
        """
        n = len(embeddings)
        if n < 2:
            return 0.0
        
        # Sample pairs to avoid O(n²) computation
        n_pairs = min(100, n * (n - 1) // 2)
        distortions = []
        
        for _ in range(n_pairs):
            i, j = np.random.choice(n, size=2, replace=False)
            
            # Original Euclidean distance
            euclidean_dist = np.linalg.norm(embeddings[i] - embeddings[j])
            
            # Hyperbolic distance
            hyperbolic_dist = self.distance(projected[i], projected[j])
            
            # Relative distortion
            if euclidean_dist > 1e-8:
                distortion = abs(hyperbolic_dist - euclidean_dist) / euclidean_dist
                distortions.append(distortion)
        
        return np.mean(distortions) if distortions else 0.0
