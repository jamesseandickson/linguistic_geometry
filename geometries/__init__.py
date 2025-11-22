"""
Geometric structures for linguistic meaning.

Each geometry represents a hypothesis about how linguistic concepts
are structured in embedding space.
"""

from .base import Geometry
from .scalar import ScalarGeometry
from .euclidean import EuclideanGeometry

__all__ = [
    "Geometry",
    "ScalarGeometry",
    "EuclideanGeometry",
]
