"""
Unit tests for basic geometries.

Tests the API and functionality of ScalarGeometry and EuclideanGeometry
using small, controlled examples (not synthetic semantic data).
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from geometries.scalar import ScalarGeometry
from geometries.euclidean import EuclideanGeometry


def test_scalar_geometry_api():
    """Test ScalarGeometry API works correctly."""
    print("Testing ScalarGeometry API...")
    
    # Small test data: 3 points in 4D space
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0], 
        [3.0, 0.0, 0.0, 0.0]
    ])
    
    # Fit scalar geometry
    geom = ScalarGeometry(dim=1)
    metrics = geom.fit(embeddings)
    
    # Test API
    assert 'variance_explained' in metrics
    assert 'reconstruction_error' in metrics
    assert 'compression_ratio' in metrics
    assert 0 <= metrics['variance_explained'] <= 1
    assert metrics['compression_ratio'] == 4.0  # 4D → 1D
    
    # Test projection/reconstruction
    projected = geom.project(embeddings)
    reconstructed = geom.reconstruct(projected)
    
    assert projected.shape == (3, 1)
    assert reconstructed.shape == (3, 4)
    
    print("  ✓ ScalarGeometry API correct")


def test_euclidean_geometry_api():
    """Test EuclideanGeometry API works correctly."""
    print("Testing EuclideanGeometry API...")
    
    # Small test data: 4 points in 6D space
    np.random.seed(42)  # For reproducible results
    embeddings = np.random.randn(4, 6)
    
    # Fit Euclidean geometry
    geom = EuclideanGeometry(dim=3)
    metrics = geom.fit(embeddings)
    
    # Test API
    assert 'variance_explained' in metrics
    assert 'reconstruction_error' in metrics
    assert 'compression_ratio' in metrics
    assert 0 <= metrics['variance_explained'] <= 1
    assert metrics['compression_ratio'] == 2.0  # 6D → 3D
    
    # Test projection/reconstruction
    projected = geom.project(embeddings)
    reconstructed = geom.reconstruct(projected)
    
    assert projected.shape == (4, 3)
    assert reconstructed.shape == (4, 6)
    
    print("  ✓ EuclideanGeometry API correct")


def test_distance_computation():
    """Test distance computation methods."""
    print("Testing distance computation...")
    
    # Test scalar distance
    scalar_geom = ScalarGeometry(dim=1)
    x = np.array([1.0])
    y = np.array([4.0])
    dist = scalar_geom.distance(x, y)
    assert np.isclose(dist, 3.0), f"Expected 3.0, got {dist}"
    
    # Test Euclidean distance
    euclidean_geom = EuclideanGeometry(dim=2)
    x = np.array([0.0, 0.0])
    y = np.array([3.0, 4.0])
    dist = euclidean_geom.distance(x, y)
    assert np.isclose(dist, 5.0), f"Expected 5.0, got {dist}"  # 3-4-5 triangle
    
    print("  ✓ Distance computation correct")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("Testing edge cases...")
    
    # Test with minimal data
    embeddings = np.array([[1.0, 2.0]])  # Single point
    
    geom = ScalarGeometry(dim=1)
    metrics = geom.fit(embeddings)
    
    # Should handle gracefully
    assert 'variance_explained' in metrics
    
    # Test dimension validation
    try:
        geom = ScalarGeometry(dim=0)
        assert False, "Should raise error for dim=0"
    except ValueError:
        pass  # Expected
    
    print("  ✓ Edge cases handled correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("Geometry Unit Tests")
    print("=" * 60)
    print()
    
    test_scalar_geometry_api()
    test_euclidean_geometry_api()
    test_distance_computation()
    test_edge_cases()
    
    print()
    print("=" * 60)
    print("✓ All unit tests passed!")
    print("=" * 60)
    print()
    print("Next: Use these geometries with real embeddings from encoders/")