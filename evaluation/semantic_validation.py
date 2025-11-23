"""
Quick Semantic Validation Tests

Tests whether geometries reveal expected semantic structures:
- Spinor: Are time concepts arranged cyclically?
- Hyperbolic: Are social concepts hierarchical?
- Scalar: Are valence concepts on a line?

This is a sanity check to see if there's ANY signal before building full evaluation.
"""

import numpy as np
from typing import Dict, List, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_spinor_cyclicity(
    concepts: List[str],
    spinor_projected: np.ndarray,
    expected_order: List[str] = None
) -> Dict[str, float]:
    """
    Test if concepts are arranged cyclically in Spinor space.
    
    Args:
        concepts: List of concept names (e.g., ["morning", "afternoon", "evening", "night"])
        spinor_projected: Spinor projections (complex array)
        expected_order: Expected cyclic order (if known)
    
    Returns:
        Dict with:
            - phase_variance: How evenly distributed are phases? (0=perfect circle, 1=random)
            - circular_correlation: Correlation with expected order (if provided)
            - rotation_compositionality: Can we traverse by rotation?
    """
    from geometries.spinor import SpinorGeometry
    
    # Extract phases from first complex dimension
    phases = np.angle(spinor_projected[:, 0])
    
    # Normalize phases to [0, 2π]
    phases = (phases + 2*np.pi) % (2*np.pi)
    
    # 1. Measure phase distribution uniformity
    # Perfect circle: phases evenly spaced
    n = len(phases)
    expected_spacing = 2*np.pi / n
    
    # Sort phases and compute gaps
    sorted_phases = np.sort(phases)
    gaps = np.diff(sorted_phases)
    gaps = np.append(gaps, 2*np.pi - sorted_phases[-1] + sorted_phases[0])  # Wrap around
    
    # Variance in gaps (0 = perfectly uniform)
    gap_variance = np.var(gaps) / expected_spacing**2
    
    # 2. Circular correlation with expected order (if provided)
    circular_correlation = None
    if expected_order is not None:
        # Map concepts to their expected positions
        concept_to_idx = {c: i for i, c in enumerate(concepts)}
        expected_phases = np.array([2*np.pi * concept_to_idx[c] / n for c in expected_order])
        
        # Compute circular correlation
        # Convert to unit vectors and compute mean resultant length
        actual_vectors = np.exp(1j * phases)
        expected_vectors = np.exp(1j * expected_phases)
        
        # Correlation = |mean(actual * conj(expected))|
        correlation = np.abs(np.mean(actual_vectors * np.conj(expected_vectors)))
        circular_correlation = float(correlation)
    
    # 3. Test rotation compositionality
    # If we rotate by expected_spacing, do we get the next concept?
    rotation_score = 0.0
    for i in range(n):
        # Rotate concept i by expected_spacing
        rotated_phase = (phases[i] + expected_spacing) % (2*np.pi)
        
        # Find nearest concept
        distances = np.abs(phases - rotated_phase)
        # Handle wrap-around
        distances = np.minimum(distances, 2*np.pi - distances)
        nearest_idx = np.argmin(distances)
        
        # Check if nearest is the "next" concept (cyclically)
        expected_next_idx = (i + 1) % n
        if nearest_idx == expected_next_idx:
            rotation_score += 1.0
    
    rotation_score /= n
    
    return {
        'phase_variance': float(gap_variance),
        'circular_correlation': circular_correlation,
        'rotation_compositionality': float(rotation_score),
        'phases': phases.tolist(),
        'concepts': concepts
    }


def test_hyperbolic_hierarchy(
    concepts: List[str],
    hyperbolic_projected: np.ndarray,
    expected_levels: Dict[str, int] = None
) -> Dict[str, float]:
    """
    Test if concepts show hierarchical structure in Hyperbolic space.
    
    Args:
        concepts: List of concept names
        hyperbolic_projected: Hyperbolic projections (in Poincaré ball)
        expected_levels: Dict mapping concepts to hierarchy levels (0=root, 1=child, etc.)
    
    Returns:
        Dict with:
            - distance_growth_rate: Do distances grow exponentially with depth?
            - tree_likeness: How tree-like is the structure?
            - level_correlation: Correlation with expected levels (if provided)
    """
    # 1. Measure distance from origin (proxy for hierarchy depth)
    norms = np.linalg.norm(hyperbolic_projected, axis=1)
    
    # 2. Check if distances grow exponentially
    # In hyperbolic space, points at different hierarchy levels should have
    # exponentially growing distances from origin
    sorted_norms = np.sort(norms)
    
    # Fit exponential: y = a * e^(b*x)
    # Take log: log(y) = log(a) + b*x
    x = np.arange(len(sorted_norms))
    log_norms = np.log(sorted_norms + 1e-8)  # Avoid log(0)
    
    # Linear regression on log scale
    coeffs = np.polyfit(x, log_norms, 1)
    growth_rate = float(coeffs[0])  # Slope = exponential growth rate
    
    # R² for exponential fit
    predicted = coeffs[0] * x + coeffs[1]
    ss_res = np.sum((log_norms - predicted)**2)
    ss_tot = np.sum((log_norms - np.mean(log_norms))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # 3. Measure tree-likeness
    # In a tree, most pairwise distances should be large (different branches)
    # Only nearby concepts should be close
    n = len(concepts)
    if n > 1:
        from geometries.hyperbolic import HyperbolicGeometry
        geom = HyperbolicGeometry(dim=hyperbolic_projected.shape[1])
        geom._fitted = True  # Hack to use distance function
        
        distances = []
        for i in range(n):
            for j in range(i+1, n):
                dist = geom.distance(hyperbolic_projected[i], hyperbolic_projected[j])
                distances.append(dist)
        
        # Tree-likeness: high variance in distances (some very close, most far)
        tree_likeness = float(np.std(distances) / (np.mean(distances) + 1e-8))
    else:
        tree_likeness = 0.0
    
    # 4. Correlation with expected levels (if provided)
    level_correlation = None
    if expected_levels is not None:
        expected = np.array([expected_levels[c] for c in concepts])
        # Correlation between norm and expected level
        level_correlation = float(np.corrcoef(norms, expected)[0, 1])
    
    return {
        'distance_growth_rate': growth_rate,
        'exponential_fit_r2': float(r_squared),
        'tree_likeness': tree_likeness,
        'level_correlation': level_correlation,
        'norms': norms.tolist(),
        'concepts': concepts
    }


def test_scalar_linearity(
    concepts: List[str],
    scalar_projected: np.ndarray,
    expected_order: List[str] = None
) -> Dict[str, float]:
    """
    Test if concepts lie on a line in Scalar space.
    
    Args:
        concepts: List of concept names
        scalar_projected: Scalar projections (1D array)
        expected_order: Expected linear order (if known)
    
    Returns:
        Dict with:
            - linearity: How well do concepts lie on a line? (always 1.0 for 1D)
            - order_correlation: Correlation with expected order (if provided)
            - polarity_separation: Are opposites at extremes?
    """
    # Extract 1D values
    values = scalar_projected.flatten()
    
    # 1. Linearity (trivially 1.0 for 1D)
    linearity = 1.0
    
    # 2. Order correlation with expected order
    order_correlation = None
    if expected_order is not None:
        concept_to_idx = {c: i for i, c in enumerate(concepts)}
        expected_positions = np.array([concept_to_idx[c] for c in expected_order])
        
        # Rank correlation (Spearman)
        from scipy.stats import spearmanr
        order_correlation = float(spearmanr(values, expected_positions)[0])
    
    # 3. Polarity separation
    # Check if extreme values correspond to opposite concepts
    min_idx = np.argmin(values)
    max_idx = np.argmax(values)
    
    polarity_separation = float(values[max_idx] - values[min_idx])
    
    return {
        'linearity': linearity,
        'order_correlation': order_correlation,
        'polarity_separation': polarity_separation,
        'values': values.tolist(),
        'concepts': concepts,
        'min_concept': concepts[min_idx],
        'max_concept': concepts[max_idx]
    }


def run_semantic_validation():
    """Run quick semantic validation tests."""
    from corpora.loader import load_corpus
    from encoders.sentence_transformer import SentenceTransformerEncoder
    from geometries.spinor import SpinorGeometry
    from geometries.hyperbolic import HyperbolicGeometry
    from geometries.scalar import ScalarGeometry
    
    print("\n" + "="*80)
    print("  SEMANTIC VALIDATION: Quick Structure Tests")
    print("="*80)
    print("\nTesting if geometries reveal expected semantic structures...")
    
    # Load corpus
    corpus = load_corpus("semantic_concepts_v0")
    
    # Initialize encoder
    encoder = SentenceTransformerEncoder(model_name='all-MiniLM-L6-v2')
    
    # Test 1: Spinor on Time Concepts
    print("\n" + "-"*80)
    print("TEST 1: Spinor Geometry on Time Concepts")
    print("-"*80)
    
    time_concepts = ["morning", "afternoon", "evening", "night"]
    print(f"Concepts: {time_concepts}")
    print("Hypothesis: Should be arranged cyclically (phases evenly distributed)")
    
    time_embeddings = encoder.encode(time_concepts)
    spinor = SpinorGeometry(dim=2, n_phases=1)
    spinor.fit(time_embeddings)
    spinor_proj = spinor.project(time_embeddings)
    
    spinor_results = test_spinor_cyclicity(
        time_concepts,
        spinor_proj,
        expected_order=time_concepts
    )
    
    print(f"\nResults:")
    print(f"  Phase variance: {spinor_results['phase_variance']:.4f} (0=perfect circle)")
    print(f"  Circular correlation: {spinor_results['circular_correlation']:.4f} (1=perfect match)")
    print(f"  Rotation compositionality: {spinor_results['rotation_compositionality']:.4f} (1=perfect)")
    print(f"\n  Phases: {[f'{p:.2f}' for p in spinor_results['phases']]}")
    
    if spinor_results['phase_variance'] < 0.5:
        print("  ✅ STRONG CYCLIC SIGNAL DETECTED!")
    elif spinor_results['phase_variance'] < 1.0:
        print("  ⚠️  WEAK CYCLIC SIGNAL")
    else:
        print("  ❌ NO CYCLIC STRUCTURE")
    
    # Test 2: Hyperbolic on Social Hierarchy
    print("\n" + "-"*80)
    print("TEST 2: Hyperbolic Geometry on Social Hierarchy")
    print("-"*80)
    
    social_concepts = ["leader", "manager", "worker", "subordinate"]
    expected_levels = {"leader": 0, "manager": 1, "worker": 2, "subordinate": 3}
    print(f"Concepts: {social_concepts}")
    print("Hypothesis: Should show hierarchical structure (exponential distance growth)")
    
    social_embeddings = encoder.encode(social_concepts)
    hyperbolic = HyperbolicGeometry(dim=4, curvature=1.0)
    hyperbolic.fit(social_embeddings)
    hyperbolic_proj = hyperbolic.project(social_embeddings)
    
    hyperbolic_results = test_hyperbolic_hierarchy(
        social_concepts,
        hyperbolic_proj,
        expected_levels=expected_levels
    )
    
    print(f"\nResults:")
    print(f"  Distance growth rate: {hyperbolic_results['distance_growth_rate']:.4f} (>0=exponential)")
    print(f"  Exponential fit R²: {hyperbolic_results['exponential_fit_r2']:.4f} (1=perfect)")
    print(f"  Tree-likeness: {hyperbolic_results['tree_likeness']:.4f} (higher=more tree-like)")
    print(f"  Level correlation: {hyperbolic_results['level_correlation']:.4f} (1=perfect)")
    print(f"\n  Norms: {[f'{n:.3f}' for n in hyperbolic_results['norms']]}")
    
    if hyperbolic_results['level_correlation'] > 0.7:
        print("  ✅ STRONG HIERARCHICAL SIGNAL DETECTED!")
    elif hyperbolic_results['level_correlation'] > 0.4:
        print("  ⚠️  WEAK HIERARCHICAL SIGNAL")
    else:
        print("  ❌ NO HIERARCHICAL STRUCTURE")
    
    # Test 3: Scalar on Valence
    print("\n" + "-"*80)
    print("TEST 3: Scalar Geometry on Valence Concepts")
    print("-"*80)
    
    valence_concepts = ["terrible", "bad", "neutral", "good", "excellent"]
    print(f"Concepts: {valence_concepts}")
    print("Hypothesis: Should lie on a line (negative → positive)")
    
    valence_embeddings = encoder.encode(valence_concepts)
    scalar = ScalarGeometry(dim=1)
    scalar.fit(valence_embeddings)
    scalar_proj = scalar.project(valence_embeddings)
    
    scalar_results = test_scalar_linearity(
        valence_concepts,
        scalar_proj,
        expected_order=valence_concepts
    )
    
    print(f"\nResults:")
    print(f"  Order correlation: {scalar_results['order_correlation']:.4f} (1=perfect order)")
    print(f"  Polarity separation: {scalar_results['polarity_separation']:.4f}")
    print(f"  Min concept: '{scalar_results['min_concept']}'")
    print(f"  Max concept: '{scalar_results['max_concept']}'")
    print(f"\n  Values: {[f'{v:.3f}' for v in scalar_results['values']]}")
    
    if scalar_results['order_correlation'] > 0.8:
        print("  ✅ STRONG LINEAR STRUCTURE DETECTED!")
    elif scalar_results['order_correlation'] > 0.5:
        print("  ⚠️  WEAK LINEAR STRUCTURE")
    else:
        print("  ❌ NO LINEAR STRUCTURE")
    
    # Summary
    print("\n" + "="*80)
    print("  SUMMARY")
    print("="*80)
    
    signals_detected = []
    if spinor_results['phase_variance'] < 0.5:
        signals_detected.append("Spinor captures time cycles")
    if hyperbolic_results['level_correlation'] > 0.7:
        signals_detected.append("Hyperbolic captures social hierarchy")
    if scalar_results['order_correlation'] > 0.8:
        signals_detected.append("Scalar captures valence order")
    
    if signals_detected:
        print("\n✅ SEMANTIC SIGNALS DETECTED:")
        for signal in signals_detected:
            print(f"  • {signal}")
        print("\n→ Geometries DO reveal semantic structure!")
        print("→ Worth implementing full semantic evaluation (Option B)")
    else:
        print("\n❌ NO STRONG SEMANTIC SIGNALS DETECTED")
        print("→ Either:")
        print("  1. Embeddings don't capture these structures")
        print("  2. Need better test concepts")
        print("  3. Need different encoder")
        print("→ Consider testing with different encoder or concepts")


if __name__ == "__main__":
    try:
        run_semantic_validation()
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
