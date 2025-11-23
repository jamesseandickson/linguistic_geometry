"""
Contextual Geometry Test

Tests whether geometric patterns persist when concepts appear in sentences.

Key Question: Does "morning" maintain its cyclic phase when embedded in 
"I wake up in the morning" vs. isolated "morning"?

If YES: Geometry is fundamental to how LLMs represent meaning
If NO: Geometry is artifact of decontextualized embeddings
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from encoders.sentence_transformer import SentenceTransformerEncoder
from geometries.spinor import SpinorGeometry
from geometries.scalar import ScalarGeometry


def test_time_cycle_persistence():
    """Test if time concepts maintain cyclic structure in context."""
    print("🕐 Testing Time Cycle Persistence")
    print("="*60)
    
    # Initialize encoder
    encoder = SentenceTransformerEncoder(model_name='all-MiniLM-L6-v2')
    
    # Time concepts and contextualized sentences
    time_data = {
        'morning': [
            "I wake up in the morning feeling refreshed",
            "The morning sun shines through my window",
            "Good morning everyone at the meeting"
        ],
        'afternoon': [
            "I have lunch in the afternoon with colleagues",
            "The afternoon was spent working on projects", 
            "See you this afternoon for coffee"
        ],
        'evening': [
            "I relax in the evening after a long day",
            "The evening sky was painted with colors",
            "Good evening friends and family"
        ],
        'night': [
            "I sleep peacefully at night",
            "The night was quiet and dark",
            "Good night everyone, sweet dreams"
        ]
    }
    
    # Test 1: Isolated concepts (baseline)
    print("\n1. ISOLATED CONCEPTS (Baseline)")
    print("-" * 40)
    
    isolated_concepts = list(time_data.keys())
    isolated_embeddings = encoder.encode(isolated_concepts)
    
    # Project to Spinor
    spinor = SpinorGeometry(dim=4)
    spinor.fit(isolated_embeddings)
    isolated_projected = spinor.project(isolated_embeddings)
    
    # Extract phases
    isolated_phases = np.angle(isolated_projected[:, 0])
    isolated_phases = (isolated_phases + 2*np.pi) % (2*np.pi)
    
    print("Isolated concept phases:")
    for concept, phase in zip(isolated_concepts, isolated_phases):
        print(f"  {concept:10s}: {phase:.3f} rad ({phase*180/np.pi:.1f}°)")
    
    # Compute cyclic score for isolated
    from evaluation.evaluate_geometries import detect_spinor_cycles
    isolated_results = detect_spinor_cycles(isolated_concepts, isolated_projected)
    print(f"\nIsolated cyclic score: {isolated_results['cyclic_score']:.3f}")
    
    # Test 2: Contextualized embeddings
    print("\n2. CONTEXTUALIZED EMBEDDINGS")
    print("-" * 40)
    
    # Extract contextualized embeddings (average across contexts)
    contextualized_embeddings = []
    for concept, sentences in time_data.items():
        # Get embeddings for all sentences containing this concept
        sentence_embeddings = encoder.encode(sentences)
        # Average across contexts
        avg_embedding = np.mean(sentence_embeddings, axis=0)
        contextualized_embeddings.append(avg_embedding)
    
    contextualized_embeddings = np.array(contextualized_embeddings)
    
    # Project to Spinor (using same fitted geometry)
    contextualized_projected = spinor.project(contextualized_embeddings)
    
    # Extract phases
    contextualized_phases = np.angle(contextualized_projected[:, 0])
    contextualized_phases = (contextualized_phases + 2*np.pi) % (2*np.pi)
    
    print("Contextualized concept phases:")
    for concept, phase in zip(isolated_concepts, contextualized_phases):
        print(f"  {concept:10s}: {phase:.3f} rad ({phase*180/np.pi:.1f}°)")
    
    # Compute cyclic score for contextualized
    contextualized_results = detect_spinor_cycles(isolated_concepts, contextualized_projected)
    print(f"\nContextualized cyclic score: {contextualized_results['cyclic_score']:.3f}")
    
    # Test 3: Phase consistency
    print("\n3. PHASE CONSISTENCY ANALYSIS")
    print("-" * 40)
    
    # Compute phase differences
    phase_diffs = []
    for i, concept in enumerate(isolated_concepts):
        diff = abs(isolated_phases[i] - contextualized_phases[i])
        # Handle wraparound
        diff = min(diff, 2*np.pi - diff)
        phase_diffs.append(diff)
        print(f"  {concept:10s}: {diff:.3f} rad ({diff*180/np.pi:.1f}° difference)")
    
    # Overall consistency score (1 = identical phases, 0 = random)
    mean_diff = np.mean(phase_diffs)
    consistency = 1.0 - (mean_diff / np.pi)  # Normalize by max possible difference
    
    print(f"\nPhase consistency: {consistency:.3f}")
    
    # Assessment
    print(f"\n4. ASSESSMENT")
    print("-" * 40)
    
    if contextualized_results['cyclic_score'] > 0.7 and consistency > 0.7:
        print(f"✅ SUCCESS: Cyclic structure PERSISTS in context!")
        print(f"   → Time concepts maintain geometric relationships")
        print(f"   → Spinor geometry captures fundamental structure")
    elif contextualized_results['cyclic_score'] > 0.4:
        print(f"⚠️  PARTIAL: Cyclic structure weakened but present")
        print(f"   → Some geometric signal survives context")
        print(f"   → May be useful for specific applications")
    else:
        print(f"❌ FAILURE: Cyclic structure lost in context")
        print(f"   → Geometry is artifact of isolated embeddings")
        print(f"   → Need different approach for contextual tasks")
    
    return {
        'isolated_score': isolated_results['cyclic_score'],
        'contextualized_score': contextualized_results['cyclic_score'],
        'consistency': consistency
    }


def test_valence_scale_persistence():
    """Test if valence concepts maintain linear structure in context."""
    print("\n\n📊 Testing Valence Scale Persistence")
    print("="*60)
    
    # Initialize encoder
    encoder = SentenceTransformerEncoder(model_name='all-MiniLM-L6-v2')
    
    # Valence concepts and contextualized sentences
    valence_data = {
        'terrible': [
            "This movie was absolutely terrible",
            "I had a terrible day at work",
            "The weather is terrible today"
        ],
        'bad': [
            "That was a bad decision to make",
            "I feel bad about what happened",
            "The food tastes bad"
        ],
        'neutral': [
            "The meeting was neutral and uneventful",
            "I have neutral feelings about this",
            "The response was neutral"
        ],
        'good': [
            "This is a good book to read",
            "I feel good about the results",
            "Good job on the presentation"
        ],
        'excellent': [
            "The performance was excellent tonight",
            "This is excellent work you've done",
            "Excellent choice for dinner"
        ]
    }
    
    # Test 1: Isolated concepts
    print("\n1. ISOLATED CONCEPTS (Baseline)")
    print("-" * 40)
    
    isolated_concepts = list(valence_data.keys())
    isolated_embeddings = encoder.encode(isolated_concepts)
    
    # Project to Scalar
    scalar = ScalarGeometry(dim=1)
    scalar.fit(isolated_embeddings)
    isolated_projected = scalar.project(isolated_embeddings)
    
    print("Isolated concept values:")
    for concept, value in zip(isolated_concepts, isolated_projected.flatten()):
        print(f"  {concept:10s}: {value:.3f}")
    
    # Compute linear score
    from evaluation.evaluate_geometries import detect_scalar_polarity
    isolated_results = detect_scalar_polarity(isolated_concepts, isolated_projected)
    print(f"\nIsolated linear score: {isolated_results['linear_score']:.3f}")
    
    # Test 2: Contextualized embeddings
    print("\n2. CONTEXTUALIZED EMBEDDINGS")
    print("-" * 40)
    
    # Extract contextualized embeddings
    contextualized_embeddings = []
    for concept, sentences in valence_data.items():
        sentence_embeddings = encoder.encode(sentences)
        avg_embedding = np.mean(sentence_embeddings, axis=0)
        contextualized_embeddings.append(avg_embedding)
    
    contextualized_embeddings = np.array(contextualized_embeddings)
    
    # Project to Scalar (using same fitted geometry)
    contextualized_projected = scalar.project(contextualized_embeddings)
    
    print("Contextualized concept values:")
    for concept, value in zip(isolated_concepts, contextualized_projected.flatten()):
        print(f"  {concept:10s}: {value:.3f}")
    
    # Compute linear score
    contextualized_results = detect_scalar_polarity(isolated_concepts, contextualized_projected)
    print(f"\nContextualized linear score: {contextualized_results['linear_score']:.3f}")
    
    # Test 3: Value correlation
    isolated_values = isolated_projected.flatten()
    contextualized_values = contextualized_projected.flatten()
    
    correlation = np.corrcoef(isolated_values, contextualized_values)[0, 1]
    print(f"\nValue correlation: {correlation:.3f}")
    
    # Assessment
    print(f"\n3. ASSESSMENT")
    print("-" * 40)
    
    if contextualized_results['linear_score'] > 0.7 and correlation > 0.7:
        print(f"✅ SUCCESS: Linear structure PERSISTS in context!")
        print(f"   → Valence concepts maintain scalar relationships")
        print(f"   → Scalar geometry captures fundamental structure")
    elif contextualized_results['linear_score'] > 0.4:
        print(f"⚠️  PARTIAL: Linear structure weakened but present")
        print(f"   → Some scalar signal survives context")
    else:
        print(f"❌ FAILURE: Linear structure lost in context")
        print(f"   → Scalar geometry doesn't persist")
    
    return {
        'isolated_score': isolated_results['linear_score'],
        'contextualized_score': contextualized_results['linear_score'],
        'correlation': correlation
    }


def main():
    """Run contextual geometry persistence tests."""
    print("🧪 CONTEXTUAL GEOMETRY PERSISTENCE TESTS")
    print("="*80)
    print("\nTesting whether geometric patterns persist when concepts")
    print("appear in sentences vs. isolated embeddings.\n")
    
    print("This answers the critical question:")
    print("💡 Is geometry fundamental to LLM representations,")
    print("   or just an artifact of decontextualized embeddings?\n")
    
    try:
        # Test 1: Time cycles
        time_results = test_time_cycle_persistence()
        
        # Test 2: Valence scales  
        valence_results = test_valence_scale_persistence()
        
        # Overall summary
        print("\n\n" + "="*80)
        print("  FINAL RESULTS")
        print("="*80)
        
        print(f"\n🕐 TIME CYCLES:")
        print(f"   Isolated score:        {time_results['isolated_score']:.3f}")
        print(f"   Contextualized score:  {time_results['contextualized_score']:.3f}")
        print(f"   Phase consistency:     {time_results['consistency']:.3f}")
        
        print(f"\n📊 VALENCE SCALES:")
        print(f"   Isolated score:        {valence_results['isolated_score']:.3f}")
        print(f"   Contextualized score:  {valence_results['contextualized_score']:.3f}")
        print(f"   Value correlation:     {valence_results['correlation']:.3f}")
        
        # Overall assessment
        strong_signals = 0
        if time_results['contextualized_score'] > 0.7 and time_results['consistency'] > 0.7:
            strong_signals += 1
        if valence_results['contextualized_score'] > 0.7 and valence_results['correlation'] > 0.7:
            strong_signals += 1
        
        print(f"\n" + "="*80)
        print("  CONCLUSION")
        print("="*80)
        
        if strong_signals == 2:
            print(f"\n🎉 BREAKTHROUGH: Geometry is FUNDAMENTAL!")
            print(f"   ✅ Both cyclic and linear patterns persist in context")
            print(f"   ✅ LLMs naturally encode geometric semantic structure")
            print(f"   ✅ Ready to build geometry-aware applications")
            
            print(f"\n🚀 NEXT STEPS:")
            print(f"   1. Test temporal reasoning tasks")
            print(f"   2. Build sentiment analysis with scalar geometry")
            print(f"   3. Create interpretability tools")
            print(f"   4. Test with other encoders (OpenAI, BERT)")
            
        elif strong_signals == 1:
            print(f"\n⚠️  PARTIAL SUCCESS: Some geometry persists")
            print(f"   → One pattern type is robust to context")
            print(f"   → Worth investigating further")
            print(f"   → May be concept-type dependent")
            
        else:
            print(f"\n🤔 CONTEXT DISRUPTS GEOMETRY:")
            print(f"   → Geometric patterns are weakened by context")
            print(f"   → May still be useful for concept-level tasks")
            print(f"   → Need different approach for sentence-level tasks")
        
        return {
            'time': time_results,
            'valence': valence_results,
            'strong_signals': strong_signals
        }
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()