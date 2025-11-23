"""
Spinor Geometry - Complex Phase/Rotation Space.

From semantic_atlas.md:
"Spinors capture rotational symmetry and phase. Good for cyclic, periodic,
or polar concepts that have orientation and can 'rotate back' to themselves."

Good for:
- Temporal cycles: "morning" → "afternoon" → "evening" → "night" → "morning"
- Tense/aspect: past → present → future (with aspectual phases)
- Polarity: positive ↔ negative (with gradations)
- Modal cycles: certainty → possibility → impossibility
- Emotional valence with arousal (2D phase space)

Key properties:
- Represents points as complex numbers (magnitude + phase)
- Natural periodicity: rotation by 2π returns to same state
- Can capture both "how much" (magnitude) and "what kind" (phase)
- Supports smooth interpolation around cycles
"""

import numpy as np
from typing import Dict
from .base import Geometry


class SpinorGeometry(Geometry):
    """
    Spinor geometry using complex-valued representations.
    
    Each dimension is represented as a complex number z = r·e^(iθ),
    where r is magnitude and θ is phase angle.
    
    This naturally captures cyclic and rotational structure in language.
    """
    
    def __init__(self, dim: int, n_phases: int = 1, **kwargs):
        """
        Initialize spinor geometry.
        
        Args:
            dim: Target dimensionality (number of complex dimensions)
            n_phases: Number of independent phase dimensions (default: 1)
                     - 1: single phase (e.g., simple cycles)
                     - 2: two independent phases (e.g., valence + arousal)
                     - dim: fully independent phases per dimension
            **kwargs: Additional parameters
        """
        super().__init__(dim, **kwargs)
        self.n_phases = min(n_phases, dim)  # Can't have more phases than dimensions
        self.projection_matrix = None
        self.mean = None
        self.phase_matrix = None  # Maps to phase space
        self.magnitude_scale = None
    
    def fit(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Fit spinor geometry to embeddings.
        
        Strategy:
        1. PCA to reduce to target dimensionality
        2. Decompose into magnitude and phase components
        3. Learn phase structure from data (detect cyclicity)
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
        
        Returns:
            Dictionary with:
                - variance_explained: Fraction of variance captured
                - reconstruction_error: MSE after round-trip
                - compression_ratio: Original dim / target dim
                - phase_coherence: How well phases align with cycles
                - magnitude_range: Range of magnitudes
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
        
        # Step 2: Project to reduced space
        euclidean_projected = np.dot(centered, self.projection_matrix)
        
        # Step 3: Learn phase structure
        # For each phase dimension, find the circular component
        self.phase_matrix = np.zeros((self.dim, self.n_phases))
        
        for phase_idx in range(self.n_phases):
            # Use pairs of dimensions to create phase
            if phase_idx * 2 + 1 < self.dim:
                # Create rotation matrix for this phase
                self.phase_matrix[phase_idx * 2, phase_idx] = 1.0
                self.phase_matrix[phase_idx * 2 + 1, phase_idx] = 1.0
            else:
                # Use remaining dimensions
                self.phase_matrix[phase_idx, phase_idx] = 1.0
        
        # Step 4: Compute magnitude scaling
        magnitudes = np.linalg.norm(euclidean_projected, axis=1)
        self.magnitude_scale = np.mean(magnitudes) if len(magnitudes) > 0 else 1.0
        
        self._fitted = True
        
        # Test reconstruction
        projected = self.project(embeddings)
        reconstructed = self.reconstruct(projected)
        reconstruction_error = self.compute_distortion(embeddings, reconstructed)
        
        # Compute phase coherence
        phase_coherence = self._compute_phase_coherence(projected)
        
        return {
            'variance_explained': float(variance_explained),
            'reconstruction_error': float(reconstruction_error),
            'compression_ratio': embedding_dim / self.dim,
            'phase_coherence': float(phase_coherence),
            'magnitude_range': float(np.max(magnitudes) - np.min(magnitudes)),
            'n_phases': self.n_phases
        }
    
    def project(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Project embeddings into spinor (complex) space.
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
        
        Returns:
            Complex array of shape (n_samples, self.dim) with magnitude and phase
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before project()")
        
        # PCA projection to target dimensionality
        centered = embeddings - self.mean
        euclidean_proj = np.dot(centered, self.projection_matrix)
        # euclidean_proj shape: (n_samples, self.dim)
        
        # Convert to complex representation
        # Each pair of real dimensions becomes one complex dimension
        n_samples = len(embeddings)
        complex_proj = np.zeros((n_samples, self.dim), dtype=complex)
        
        for i in range(self.dim):
            if i * 2 + 1 < self.dim:
                # Use pair of dimensions for real and imaginary parts
                real_part = euclidean_proj[:, i * 2]
                imag_part = euclidean_proj[:, i * 2 + 1]
                complex_proj[:, i] = real_part + 1j * imag_part
            else:
                # Use single dimension (real-valued)
                idx = min(i, self.dim - 1)
                complex_proj[:, i] = euclidean_proj[:, idx]
        
        return complex_proj
    
    def reconstruct(self, projected: np.ndarray) -> np.ndarray:
        """
        Reconstruct embeddings from spinor representation.
        
        Args:
            projected: Complex array of shape (n_samples, self.dim)
        
        Returns:
            Reconstructed embeddings of shape (n_samples, embedding_dim)
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before reconstruct()")
        
        # Convert complex back to real (inverse of project operation)
        n_samples = len(projected)
        target_dim = self.projection_matrix.shape[1]  # This is self.dim
        euclidean_proj = np.zeros((n_samples, target_dim))
        
        for i in range(self.dim):
            if i * 2 + 1 < target_dim:
                # Split complex into real and imaginary
                euclidean_proj[:, i * 2] = np.real(projected[:, i])
                euclidean_proj[:, i * 2 + 1] = np.imag(projected[:, i])
            else:
                # Use real part only
                idx = min(i, target_dim - 1)
                euclidean_proj[:, idx] = np.real(projected[:, i])
        
        # Inverse PCA projection
        reconstructed = self.mean + np.dot(euclidean_proj, self.projection_matrix.T)
        
        return reconstructed
    
    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute distance in spinor space.
        
        Uses a combination of magnitude difference and phase difference:
        d(x,y) = sqrt(|mag(x) - mag(y)|² + |phase(x) - phase(y)|²)
        
        Args:
            x: Complex point of shape (self.dim,)
            y: Complex point of shape (self.dim,)
        
        Returns:
            Spinor distance
        """
        # Magnitude distance
        mag_x = np.abs(x)
        mag_y = np.abs(y)
        mag_dist = np.linalg.norm(mag_x - mag_y)
        
        # Phase distance (accounting for periodicity)
        phase_x = np.angle(x)
        phase_y = np.angle(y)
        
        # Compute shortest angular distance (wrap around 2π)
        phase_diff = np.abs(phase_x - phase_y)
        phase_diff = np.minimum(phase_diff, 2 * np.pi - phase_diff)
        phase_dist = np.linalg.norm(phase_diff)
        
        # Combined distance
        return float(np.sqrt(mag_dist**2 + phase_dist**2))
    
    def get_phases(self, projected: np.ndarray) -> np.ndarray:
        """
        Extract phase angles from spinor representation.
        
        Args:
            projected: Complex array of shape (n_samples, self.dim)
        
        Returns:
            Phase angles of shape (n_samples, self.dim) in radians
        """
        return np.angle(projected)
    
    def get_magnitudes(self, projected: np.ndarray) -> np.ndarray:
        """
        Extract magnitudes from spinor representation.
        
        Args:
            projected: Complex array of shape (n_samples, self.dim)
        
        Returns:
            Magnitudes of shape (n_samples, self.dim)
        """
        return np.abs(projected)
    
    def rotate(self, projected: np.ndarray, angle: float, dim: int = 0) -> np.ndarray:
        """
        Rotate points in spinor space by a given angle.
        
        This demonstrates the natural rotational symmetry of spinor geometry.
        
        Args:
            projected: Complex array of shape (n_samples, self.dim)
            angle: Rotation angle in radians
            dim: Which dimension to rotate (default: 0)
        
        Returns:
            Rotated complex array
        """
        rotated = projected.copy()
        rotation = np.exp(1j * angle)
        rotated[:, dim] *= rotation
        return rotated
    
    def _compute_phase_coherence(self, projected: np.ndarray) -> float:
        """
        Measure how coherent the phases are (do they cluster or spread uniformly?).
        
        Higher coherence suggests the data has natural cyclic structure.
        
        Args:
            projected: Complex array of shape (n_samples, self.dim)
        
        Returns:
            Phase coherence score (0 = uniform, 1 = perfectly aligned)
        """
        phases = np.angle(projected)
        
        # Compute mean resultant length (circular statistics)
        # For each dimension, compute how concentrated the phases are
        coherences = []
        
        for dim in range(self.dim):
            # Convert phases to unit vectors
            unit_vectors = np.exp(1j * phases[:, dim])
            
            # Mean resultant length
            mean_vector = np.mean(unit_vectors)
            coherence = np.abs(mean_vector)
            coherences.append(coherence)
        
        return np.mean(coherences)
