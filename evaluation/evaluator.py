"""
Geometry Evaluation Framework

Evaluates all geometries on all concept categories with train/test validation.
Measures reconstruction error, entropy, and compression efficiency.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split


class GeometryEvaluator:
    """
    Evaluates geometries with train/test split validation.
    
    Process:
    1. Split embeddings 80/20 into train/test
    2. Fit geometry on train set only
    3. Measure reconstruction on both train and test
    4. Calculate metrics (error, entropy, compression)
    5. Check for overfitting (train vs test gap)
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize evaluator.
        
        Args:
            test_size: Fraction of data for test set (default: 0.2 = 80/20 split)
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
    
    def evaluate_geometry(
        self,
        geometry,
        embeddings: np.ndarray,
        concept_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single geometry with train/test split.
        
        Args:
            geometry: Geometry instance (must have fit/project/reconstruct methods)
            embeddings: Array of shape (n_samples, embedding_dim)
            concept_names: Optional list of concept names for tracking
        
        Returns:
            Dictionary with:
                - train_error: Reconstruction error on training set
                - test_error: Reconstruction error on test set
                - overfit_gap: Difference between test and train error
                - compression_ratio: Original dim / target dim
                - entropy: Estimated entropy of geometry parameters
                - fit_metrics: Metrics returned by geometry.fit()
        """
        n_samples = len(embeddings)
        
        # Handle small datasets
        if n_samples < 5:
            # Too small for train/test split, use all data
            return self._evaluate_no_split(geometry, embeddings)
        
        # Split into train/test
        indices = np.arange(n_samples)
        train_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        train_embeddings = embeddings[train_idx]
        test_embeddings = embeddings[test_idx]
        
        # Fit on training data only
        fit_metrics = geometry.fit(train_embeddings)
        
        # Evaluate on training set
        train_projected = geometry.project(train_embeddings)
        train_reconstructed = geometry.reconstruct(train_projected)
        train_error = np.mean((train_embeddings - train_reconstructed) ** 2)
        
        # Evaluate on test set
        test_projected = geometry.project(test_embeddings)
        test_reconstructed = geometry.reconstruct(test_projected)
        test_error = np.mean((test_embeddings - test_reconstructed) ** 2)
        
        # Calculate overfitting gap
        overfit_gap = test_error - train_error
        
        # Estimate entropy (simplified: based on number of parameters)
        entropy = self._estimate_entropy(geometry, embeddings.shape[1])
        
        return {
            'train_error': float(train_error),
            'test_error': float(test_error),
            'overfit_gap': float(overfit_gap),
            'compression_ratio': fit_metrics.get('compression_ratio', 1.0),
            'entropy': float(entropy),
            'fit_metrics': fit_metrics,
            'n_train': len(train_idx),
            'n_test': len(test_idx)
        }
    
    def _evaluate_no_split(self, geometry, embeddings: np.ndarray) -> Dict[str, Any]:
        """Evaluate without train/test split (for small datasets)."""
        fit_metrics = geometry.fit(embeddings)
        
        projected = geometry.project(embeddings)
        reconstructed = geometry.reconstruct(projected)
        error = np.mean((embeddings - reconstructed) ** 2)
        
        entropy = self._estimate_entropy(geometry, embeddings.shape[1])
        
        return {
            'train_error': float(error),
            'test_error': float(error),
            'overfit_gap': 0.0,
            'compression_ratio': fit_metrics.get('compression_ratio', 1.0),
            'entropy': float(entropy),
            'fit_metrics': fit_metrics,
            'n_train': len(embeddings),
            'n_test': 0
        }
    
    def _estimate_entropy(self, geometry, original_dim: int) -> float:
        """
        Estimate entropy of geometry representation.
        
        Simplified entropy calculation based on:
        - Number of parameters in the geometry
        - Compression ratio
        - Information content
        
        Lower entropy = more efficient representation
        """
        # Get target dimension
        target_dim = geometry.dim
        
        # Base entropy: bits needed to represent compressed space
        base_entropy = target_dim * np.log2(original_dim)
        
        # Adjust for geometry type
        geometry_name = geometry.__class__.__name__
        
        if geometry_name == 'ScalarGeometry':
            # 1D is most compressed
            entropy = base_entropy * 0.5
        elif geometry_name == 'SpinorGeometry':
            # Complex numbers need 2x storage
            entropy = base_entropy * 1.5
        elif geometry_name == 'HyperbolicGeometry':
            # Hyperbolic needs extra parameters (curvature, scale)
            entropy = base_entropy * 1.2
        else:
            # Euclidean is baseline
            entropy = base_entropy
        
        return entropy
    
    def compare_geometries(
        self,
        geometries: Dict[str, Any],
        embeddings: np.ndarray,
        category_name: str = "concepts"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple geometries on the same embeddings.
        
        Args:
            geometries: Dict mapping geometry names to geometry instances
            embeddings: Array of shape (n_samples, embedding_dim)
            category_name: Name of concept category (for reporting)
        
        Returns:
            Dict mapping geometry names to evaluation results
        """
        results = {}
        
        for geom_name, geometry in geometries.items():
            try:
                results[geom_name] = self.evaluate_geometry(geometry, embeddings)
                results[geom_name]['success'] = True
            except Exception as e:
                results[geom_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def find_best_geometry(
        self,
        results: Dict[str, Dict[str, Any]],
        metric: str = 'test_error'
    ) -> Tuple[str, float]:
        """
        Find the best performing geometry based on a metric.
        
        Args:
            results: Results from compare_geometries()
            metric: Metric to optimize ('test_error', 'entropy', 'overfit_gap')
        
        Returns:
            Tuple of (best_geometry_name, best_metric_value)
        """
        valid_results = {
            name: res for name, res in results.items()
            if res.get('success', False) and metric in res
        }
        
        if not valid_results:
            return None, None
        
        # Lower is better for all our metrics
        best_name = min(valid_results.keys(), key=lambda k: valid_results[k][metric])
        best_value = valid_results[best_name][metric]
        
        return best_name, best_value
