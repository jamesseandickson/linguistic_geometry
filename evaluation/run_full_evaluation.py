"""
Full Evaluation: All Geometries × All Categories

Tests all 4 geometries on all 6 semantic concept categories.
Outputs a clean, scannable comparison report.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from corpora.loader import load_corpus
from encoders.sentence_transformer import SentenceTransformerEncoder
from geometries.scalar import ScalarGeometry
from geometries.euclidean import EuclideanGeometry
from geometries.spinor import SpinorGeometry
from geometries.hyperbolic import HyperbolicGeometry
from evaluation.evaluator import GeometryEvaluator


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_category_results(category_name: str, results: dict, best_geom: str):
    """Print results for a single category."""
    print(f"\n📊 {category_name.upper()}")
    print("-" * 80)
    print(f"{'Geometry':<15} {'Test Error':<12} {'Train Error':<12} {'Overfit':<10} {'Entropy':<10} {'Compress':<10}")
    print("-" * 80)
    
    for geom_name in ['Scalar', 'Euclidean', 'Spinor', 'Hyperbolic']:
        if geom_name not in results or not results[geom_name].get('success', False):
            print(f"{geom_name:<15} {'FAILED':<12}")
            continue
        
        res = results[geom_name]
        winner = "✅" if geom_name == best_geom else "  "
        
        print(f"{geom_name:<15} "
              f"{res['test_error']:<12.6f} "
              f"{res['train_error']:<12.6f} "
              f"{res['overfit_gap']:<10.6f} "
              f"{res['entropy']:<10.2f} "
              f"{res['compression_ratio']:<10.1f}x {winner}")


def print_summary_table(all_results: dict):
    """Print final summary table showing winners per category."""
    print_header("SUMMARY: Best Geometry per Category")
    
    print(f"\n{'Category':<15} {'Winner':<15} {'Test Error':<12} {'Why?':<40}")
    print("-" * 80)
    
    insights = {
        'emotions': 'Valence structure (positive/negative)',
        'time': 'Cyclic/sequential patterns',
        'space': 'Dimensional relationships',
        'modality': 'Certainty gradations',
        'social': 'Hierarchical relationships',
        'abstract': 'Complex conceptual structure'
    }
    
    for category, results in all_results.items():
        if not results:
            continue
        
        # Find best geometry
        valid_results = {k: v for k, v in results.items() if v.get('success', False)}
        if not valid_results:
            continue
        
        best_geom = min(valid_results.keys(), key=lambda k: valid_results[k]['test_error'])
        best_error = valid_results[best_geom]['test_error']
        
        insight = insights.get(category, 'Unknown structure')
        
        print(f"{category:<15} {best_geom:<15} {best_error:<12.6f} {insight:<40}")


def main():
    """Run full evaluation."""
    print_header("LINGUISTIC GEOMETRY: FULL EVALUATION")
    print("\nTesting all geometries on all semantic concept categories")
    print("Train/Test Split: 80/20")
    print("Target Dimension: 4 (for fair comparison)")
    
    # Load corpus
    print("\n📚 Loading semantic concepts...")
    corpus = load_corpus("semantic_concepts_v0")
    
    # Group clusters by subdomain (emotions, time, space, etc.)
    categories = {}
    for cluster in corpus.clusters:
        subdomain = cluster.subdomain
        if subdomain not in categories:
            categories[subdomain] = []
        categories[subdomain].extend(cluster.expressions)
    
    print(f"Loaded {len(categories)} categories: {', '.join(categories.keys())}")
    
    # Initialize encoder
    print("\n🔤 Initializing encoder...")
    encoder = SentenceTransformerEncoder(model_name='all-MiniLM-L6-v2')
    print(f"Using: {encoder.model_name}")
    
    # Initialize evaluator
    evaluator = GeometryEvaluator(test_size=0.2, random_state=42)
    
    # Target dimension for all geometries (for fair comparison)
    target_dim = 4
    
    # Store all results
    all_results = {}
    
    # Evaluate each category
    for category, concepts in categories.items():
        print(f"\n\n{'='*80}")
        print(f"Evaluating: {category.upper()}")
        print(f"{'='*80}")
        
        print(f"Concepts: {len(concepts)} items")
        print(f"Sample: {', '.join(concepts[:5])}...")
        
        # Encode concepts
        print("Encoding...")
        embeddings = encoder.encode(concepts)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Initialize all geometries
        geometries = {
            'Scalar': ScalarGeometry(dim=1),  # 1D is natural for scalar
            'Euclidean': EuclideanGeometry(dim=target_dim),
            'Spinor': SpinorGeometry(dim=target_dim, n_phases=1),
            'Hyperbolic': HyperbolicGeometry(dim=target_dim, curvature=1.0)
        }
        
        # Evaluate all geometries
        print("\nEvaluating geometries...")
        results = evaluator.compare_geometries(geometries, embeddings, category)
        
        # Find best
        best_geom, best_error = evaluator.find_best_geometry(results, metric='test_error')
        
        # Store results
        all_results[category] = results
        
        # Print results
        print_category_results(category, results, best_geom)
        
        # Print special insights for this category
        if best_geom == 'Spinor':
            spinor_res = results['Spinor']
            if 'fit_metrics' in spinor_res and 'phase_coherence' in spinor_res['fit_metrics']:
                coherence = spinor_res['fit_metrics']['phase_coherence']
                print(f"\n  💫 Spinor detected phase coherence: {coherence:.3f} (cyclic structure!)")
        
        if best_geom == 'Hyperbolic':
            hyper_res = results['Hyperbolic']
            if 'fit_metrics' in hyper_res and 'hyperbolic_distortion' in hyper_res['fit_metrics']:
                distortion = hyper_res['fit_metrics']['hyperbolic_distortion']
                print(f"\n  🌀 Hyperbolic distortion: {distortion:.3f} (hierarchical structure!)")
        
        if best_geom == 'Scalar':
            print(f"\n  📏 Scalar geometry wins - concepts lie on a simple line!")
    
    # Print final summary
    print("\n\n")
    print_summary_table(all_results)
    
    # Print key insights
    print_header("KEY INSIGHTS")
    
    # Count geometry wins
    wins = {}
    for category, results in all_results.items():
        valid_results = {k: v for k, v in results.items() if v.get('success', False)}
        if valid_results:
            best = min(valid_results.keys(), key=lambda k: valid_results[k]['test_error'])
            wins[best] = wins.get(best, 0) + 1
    
    print("\n🏆 Geometry Performance:")
    for geom in ['Scalar', 'Euclidean', 'Spinor', 'Hyperbolic']:
        count = wins.get(geom, 0)
        bar = "█" * count
        print(f"  {geom:<15} {bar} ({count} categories)")
    
    print("\n📈 What This Tells Us:")
    print("  • Lower test error = better semantic structure capture")
    print("  • Small overfit gap = geometry generalizes well")
    print("  • Lower entropy = more efficient representation")
    print("  • Different geometries excel at different semantic structures")
    
    print("\n✅ Evaluation complete! Check results above.")
    print("\nNext steps:")
    print("  1. Examine which geometries work best for which concepts")
    print("  2. Investigate why (phase coherence, hierarchy, etc.)")
    print("  3. Document findings in semantic_atlas.md")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
