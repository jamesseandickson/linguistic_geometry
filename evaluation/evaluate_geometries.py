"""
Geometry Pattern Detection

Tests whether different geometries reveal latent semantic structure in embeddings.
Does NOT test reconstruction error (circular logic).
DOES test whether geometric patterns match semantic patterns.

For each geometry:
- Spinor: Tests for cyclic/rotational patterns
- Hyperbolic: Tests for hierarchical/tree patterns  
- Scalar: Tests for linear/polar patterns
- Euclidean: Baseline cluster cohesion

Key insight: We're discovering structure, not approximating Euclidean space.
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

from corpora.loader import load_corpus
from encoders.sentence_transformer import SentenceTransformerEncoder
from geometries.scalar import ScalarGeometry
from geometries.euclidean import EuclideanGeometry
from geometries.spinor import SpinorGeometry
from geometries.hyperbolic import HyperbolicGeometry


# ============================================================================
# PATTERN DETECTION FUNCTIONS
# ============================================================================

def detect_spinor_cycles(
    concepts: List[str],
    spinor_projected: np.ndarray
) -> Dict[str, float]:
    """
    Detect cyclic patterns in Spinor space.
    
    Measures:
    1. Phase uniformity - are phases evenly distributed?
    2. Rotation compositionality - can we traverse by rotation?
    3. Phase coherence - do similar concepts have similar phases?
    """
    # Extract phases from first complex dimension
    phases = np.angle(spinor_projected[:, 0])
    phases = (phases + 2*np.pi) % (2*np.pi)  # Normalize to [0, 2π]
    
    n = len(phases)
    
    # 1. Phase uniformity (0 = perfectly uniform circle)
    sorted_phases = np.sort(phases)
    gaps = np.diff(sorted_phases)
    gaps = np.append(gaps, 2*np.pi - sorted_phases[-1] + sorted_phases[0])
    expected_gap = 2*np.pi / n
    uniformity = 1.0 - (np.std(gaps) / expected_gap)  # 1 = perfect, 0 = random
    
    # 2. Rotation compositionality
    # Can we reach the next concept by rotating by 2π/n?
    rotation_score = 0.0
    for i in range(n):
        rotated = (phases[i] + 2*np.pi/n) % (2*np.pi)
        distances = np.minimum(np.abs(phases - rotated), 2*np.pi - np.abs(phases - rotated))
        nearest = np.argmin(distances)
        if nearest == (i + 1) % n or nearest == (i - 1) % n:
            rotation_score += 1.0
    rotation_score /= n
    
    # 3. Phase coherence (inverse of circular variance)
    mean_phase = np.angle(np.mean(np.exp(1j * phases)))
    circular_variance = 1 - np.abs(np.mean(np.exp(1j * (phases - mean_phase))))
    coherence = 1.0 - circular_variance
    
    return {
        'phase_uniformity': float(uniformity),
        'rotation_compositionality': float(rotation_score),
        'phase_coherence': float(coherence),
        'cyclic_score': float((uniformity + rotation_score + coherence) / 3),
        'phases': phases.tolist()
    }


def detect_hyperbolic_hierarchy(
    concepts: List[str],
    hyperbolic_projected: np.ndarray,
    geometry: HyperbolicGeometry
) -> Dict[str, float]:
    """
    Detect hierarchical patterns in Hyperbolic space.
    
    Measures:
    1. Distance growth - do distances grow exponentially?
    2. Tree structure - are most distances large (different branches)?
    3. Depth stratification - are there clear depth levels?
    """
    n = len(concepts)
    
    # 1. Distance from origin (proxy for depth)
    norms = np.linalg.norm(hyperbolic_projected, axis=1)
    
    # Exponential growth test
    sorted_norms = np.sort(norms)
    x = np.arange(len(sorted_norms))
    log_norms = np.log(sorted_norms + 1e-8)
    coeffs = np.polyfit(x, log_norms, 1)
    growth_rate = coeffs[0]
    
    # R² for exponential fit
    predicted = coeffs[0] * x + coeffs[1]
    ss_res = np.sum((log_norms - predicted)**2)
    ss_tot = np.sum((log_norms - np.mean(log_norms))**2)
    exponential_fit = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # 2. Tree structure (high variance in pairwise distances)
    distances = []
    for i in range(n):
        for j in range(i+1, n):
            dist = geometry.distance(hyperbolic_projected[i], hyperbolic_projected[j])
            distances.append(dist)
    
    if len(distances) > 0:
        tree_score = np.std(distances) / (np.mean(distances) + 1e-8)
    else:
        tree_score = 0.0
    
    # 3. Depth stratification (are norms clustered into levels?)
    # Use coefficient of variation
    depth_stratification = np.std(norms) / (np.mean(norms) + 1e-8)
    
    return {
        'exponential_growth': float(max(0, min(1, exponential_fit))),
        'tree_structure': float(min(1.0, tree_score / 2.0)),  # Normalize
        'depth_stratification': float(min(1.0, depth_stratification)),
        'hierarchical_score': float((exponential_fit + min(1.0, tree_score/2.0)) / 2),
        'norms': norms.tolist()
    }


def detect_scalar_polarity(
    concepts: List[str],
    scalar_projected: np.ndarray
) -> Dict[str, float]:
    """
    Detect linear/polar patterns in Scalar space.
    
    Measures:
    1. Spread - are concepts spread across the line?
    2. Polarity - are extremes meaningful?
    3. Monotonicity - is there a clear ordering?
    """
    values = scalar_projected.flatten()
    n = len(values)
    
    # 1. Spread (normalized standard deviation)
    spread = np.std(values) / (np.abs(np.mean(values)) + 1e-8)
    spread = min(1.0, spread)
    
    # 2. Polarity (distance between extremes relative to mean distance)
    min_val, max_val = np.min(values), np.max(values)
    polarity = (max_val - min_val) / (np.mean(np.abs(values)) + 1e-8)
    polarity = min(1.0, polarity / 4.0)  # Normalize
    
    # 3. Monotonicity (check if there's a clear ordering)
    # Sort values and check if adjacent concepts are semantically related
    sorted_indices = np.argsort(values)
    
    # Simple monotonicity: variance in sorted positions
    expected_positions = np.arange(n)
    actual_positions = np.argsort(sorted_indices)
    monotonicity = 1.0 - (np.std(actual_positions - expected_positions) / n)
    
    return {
        'spread': float(spread),
        'polarity': float(polarity),
        'monotonicity': float(monotonicity),
        'linear_score': float((spread + polarity) / 2),
        'values': values.tolist(),
        'min_concept': concepts[np.argmin(values)],
        'max_concept': concepts[np.argmax(values)]
    }


def measure_euclidean_clustering(
    concepts: List[str],
    euclidean_projected: np.ndarray
) -> Dict[str, float]:
    """
    Measure cluster quality in Euclidean space (baseline).
    
    Measures:
    1. Compactness - how tight are the concepts?
    2. Separation - average pairwise distance
    3. Isotropy - is the space evenly distributed?
    """
    n = len(concepts)
    
    # 1. Compactness (inverse of average distance to centroid)
    centroid = np.mean(euclidean_projected, axis=0)
    distances_to_centroid = np.linalg.norm(euclidean_projected - centroid, axis=1)
    compactness = 1.0 / (np.mean(distances_to_centroid) + 1e-8)
    compactness = min(1.0, compactness / 10.0)  # Normalize
    
    # 2. Average pairwise distance
    pairwise_dists = []
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(euclidean_projected[i] - euclidean_projected[j])
            pairwise_dists.append(dist)
    
    avg_distance = np.mean(pairwise_dists) if pairwise_dists else 0.0
    
    # 3. Isotropy (how evenly spread in all directions)
    # Compute variance in each dimension
    dim_variances = np.var(euclidean_projected, axis=0)
    isotropy = 1.0 - (np.std(dim_variances) / (np.mean(dim_variances) + 1e-8))
    
    return {
        'compactness': float(compactness),
        'avg_distance': float(avg_distance),
        'isotropy': float(isotropy),
        'cluster_score': float((compactness + isotropy) / 2)
    }


# ============================================================================
# EVALUATION RUNNER
# ============================================================================

def evaluate_category(
    category_name: str,
    concepts: List[str],
    embeddings: np.ndarray,
    target_dim: int = 4
) -> Dict[str, Dict]:
    """Evaluate all geometries on a single category."""
    
    results = {}
    
    # 1. Spinor - test for cycles
    spinor = SpinorGeometry(dim=target_dim, n_phases=1)
    spinor.fit(embeddings)
    spinor_proj = spinor.project(embeddings)
    results['Spinor'] = detect_spinor_cycles(concepts, spinor_proj)
    
    # 2. Hyperbolic - test for hierarchy
    hyperbolic = HyperbolicGeometry(dim=target_dim, curvature=1.0)
    hyperbolic.fit(embeddings)
    hyperbolic_proj = hyperbolic.project(embeddings)
    results['Hyperbolic'] = detect_hyperbolic_hierarchy(concepts, hyperbolic_proj, hyperbolic)
    
    # 3. Scalar - test for polarity
    scalar = ScalarGeometry(dim=1)
    scalar.fit(embeddings)
    scalar_proj = scalar.project(embeddings)
    results['Scalar'] = detect_scalar_polarity(concepts, scalar_proj)
    
    # 4. Euclidean - baseline clustering
    euclidean = EuclideanGeometry(dim=target_dim)
    euclidean.fit(embeddings)
    euclidean_proj = euclidean.project(embeddings)
    results['Euclidean'] = measure_euclidean_clustering(concepts, euclidean_proj)
    
    return results


def print_results(category_name: str, results: Dict[str, Dict], concepts: List[str]):
    """Print results for a category."""
    print(f"\n{'='*80}")
    print(f"  {category_name.upper()} ({len(concepts)} concepts)")
    print(f"{'='*80}")
    print(f"Sample: {', '.join(concepts[:5])}...")
    
    # Spinor
    spinor = results['Spinor']
    print(f"\n🌀 SPINOR (Cyclic Patterns):")
    print(f"  Phase uniformity:        {spinor['phase_uniformity']:.3f}")
    print(f"  Rotation compositionality: {spinor['rotation_compositionality']:.3f}")
    print(f"  Phase coherence:         {spinor['phase_coherence']:.3f}")
    print(f"  → Cyclic score:          {spinor['cyclic_score']:.3f} ", end="")
    if spinor['cyclic_score'] > 0.7:
        print("✅ STRONG CYCLIC STRUCTURE")
    elif spinor['cyclic_score'] > 0.4:
        print("⚠️  WEAK CYCLIC STRUCTURE")
    else:
        print("❌ NO CYCLIC STRUCTURE")
    
    # Hyperbolic
    hyper = results['Hyperbolic']
    print(f"\n🌳 HYPERBOLIC (Hierarchical Patterns):")
    print(f"  Exponential growth:      {hyper['exponential_growth']:.3f}")
    print(f"  Tree structure:          {hyper['tree_structure']:.3f}")
    print(f"  Depth stratification:    {hyper['depth_stratification']:.3f}")
    print(f"  → Hierarchical score:    {hyper['hierarchical_score']:.3f} ", end="")
    if hyper['hierarchical_score'] > 0.7:
        print("✅ STRONG HIERARCHICAL STRUCTURE")
    elif hyper['hierarchical_score'] > 0.4:
        print("⚠️  WEAK HIERARCHICAL STRUCTURE")
    else:
        print("❌ NO HIERARCHICAL STRUCTURE")
    
    # Scalar
    scalar = results['Scalar']
    print(f"\n📏 SCALAR (Linear/Polar Patterns):")
    print(f"  Spread:                  {scalar['spread']:.3f}")
    print(f"  Polarity:                {scalar['polarity']:.3f}")
    print(f"  Monotonicity:            {scalar['monotonicity']:.3f}")
    print(f"  → Linear score:          {scalar['linear_score']:.3f} ", end="")
    if scalar['linear_score'] > 0.7:
        print("✅ STRONG LINEAR STRUCTURE")
    elif scalar['linear_score'] > 0.4:
        print("⚠️  WEAK LINEAR STRUCTURE")
    else:
        print("❌ NO LINEAR STRUCTURE")
    print(f"  Extremes: '{scalar['min_concept']}' ↔ '{scalar['max_concept']}'")
    
    # Euclidean
    euclidean = results['Euclidean']
    print(f"\n📐 EUCLIDEAN (Baseline Clustering):")
    print(f"  Compactness:             {euclidean['compactness']:.3f}")
    print(f"  Isotropy:                {euclidean['isotropy']:.3f}")
    print(f"  → Cluster score:         {euclidean['cluster_score']:.3f}")


def print_summary(all_results: Dict[str, Dict]):
    """Print overall summary."""
    print(f"\n\n{'='*80}")
    print("  SUMMARY: Geometric Patterns Detected")
    print(f"{'='*80}\n")
    
    print(f"{'Category':<20} {'Cyclic':<10} {'Hierarchical':<15} {'Linear':<10} {'Best Fit':<15}")
    print("-" * 80)
    
    for category, results in all_results.items():
        cyclic = results['Spinor']['cyclic_score']
        hierarchical = results['Hyperbolic']['hierarchical_score']
        linear = results['Scalar']['linear_score']
        
        # Determine best fit
        scores = {
            'Cyclic': cyclic,
            'Hierarchical': hierarchical,
            'Linear': linear
        }
        best_fit = max(scores.keys(), key=lambda k: scores[k])
        best_score = scores[best_fit]
        
        # Format scores with indicators
        def fmt(score):
            if score > 0.7:
                return f"{score:.3f} ✅"
            elif score > 0.4:
                return f"{score:.3f} ⚠️ "
            else:
                return f"{score:.3f}   "
        
        print(f"{category:<20} {fmt(cyclic):<10} {fmt(hierarchical):<15} {fmt(linear):<10} {best_fit:<15}")
    
    print("\n" + "="*80)
    print("  KEY FINDINGS")
    print("="*80 + "\n")
    
    # Count strong signals
    strong_cyclic = sum(1 for r in all_results.values() if r['Spinor']['cyclic_score'] > 0.7)
    strong_hierarchical = sum(1 for r in all_results.values() if r['Hyperbolic']['hierarchical_score'] > 0.7)
    strong_linear = sum(1 for r in all_results.values() if r['Scalar']['linear_score'] > 0.7)
    
    print(f"Strong Signals Detected:")
    print(f"  🌀 Cyclic patterns:        {strong_cyclic} categories")
    print(f"  🌳 Hierarchical patterns:  {strong_hierarchical} categories")
    print(f"  📏 Linear patterns:        {strong_linear} categories")
    
    if strong_cyclic + strong_hierarchical + strong_linear > 0:
        print(f"\n✅ SUCCESS: Geometries reveal semantic structure!")
        print(f"   Different geometries capture different patterns.")
    else:
        print(f"\n⚠️  WEAK SIGNALS: No strong geometric patterns detected.")
        print(f"   Consider: different encoder, better concepts, or more data.")


def main():
    """Run geometric pattern detection."""
    print("\n" + "="*80)
    print("  GEOMETRIC PATTERN DETECTION")
    print("="*80)
    print("\nTesting whether geometries reveal latent semantic structure...")
    print("NOT testing reconstruction error (that's circular logic).")
    print("TESTING whether geometric patterns match semantic patterns.\n")
    
    # Load corpus
    print("📚 Loading semantic concepts...")
    corpus = load_corpus("semantic_concepts_v0")
    
    # Group by subdomain
    categories = {}
    for cluster in corpus.clusters:
        subdomain = cluster.subdomain
        if subdomain not in categories:
            categories[subdomain] = []
        categories[subdomain].extend(cluster.expressions)
    
    print(f"Loaded {len(categories)} categories: {', '.join(categories.keys())}\n")
    
    # Initialize encoder
    print("🔤 Initializing encoder...")
    encoder = SentenceTransformerEncoder(model_name='all-MiniLM-L6-v2')
    print(f"Using: {encoder.model_name}\n")
    
    # Evaluate each category
    all_results = {}
    
    for category_name, concepts in categories.items():
        # Encode
        embeddings = encoder.encode(concepts)
        
        # Evaluate
        results = evaluate_category(category_name, concepts, embeddings, target_dim=4)
        all_results[category_name] = results
        
        # Print
        print_results(category_name, results, concepts)
    
    # Print summary
    print_summary(all_results)
    
    print("\n" + "="*80)
    print("  INTERPRETATION GUIDE")
    print("="*80 + "\n")
    print("Score > 0.7:  Strong geometric pattern detected ✅")
    print("Score 0.4-0.7: Weak pattern (might be real or noise) ⚠️")
    print("Score < 0.4:  No pattern detected ❌")
    print("\nWhat this means:")
    print("  • Cyclic score: Concepts arranged in a circle (time, cycles)")
    print("  • Hierarchical score: Tree-like structure (taxonomies, ranks)")
    print("  • Linear score: Concepts on a line (valence, intensity)")
    print("\nThis tests the PROJECT HYPOTHESIS: Do different geometries")
    print("reveal different semantic structures already latent in embeddings?")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
