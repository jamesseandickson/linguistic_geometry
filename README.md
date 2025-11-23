# 💡 Linguistic Geometry  
### Mapping Language Into Structured Mathematical Spaces

Linguistic Geometry explores how **linguistic meaning**, encoded as vectors in large language models, can be explained, compressed, and transformed using **geometric and algebraic structures**.

The project treats each geometry as a *hypothesis* about how linguistic meaning behaves.

This repository contains the early prototype of that system.

---

## ✨ Vision

Different aspects of language might align better with different geometric forms:

- **Line-like** → simple valence or scalar judgments  
- **Hyperbolic** → hierarchical structure  
- **Cyclic/phase-like** → tense, polarity, aspectual cycles  
- **Fibre-like** → pragmatics with internal state per context  
- **Sheaf-like** → discourse with local vs global consistency  

Transformers approximate all of this in **Euclidean tensors**, which may distort some behaviours.  
The aim is to discover:

> Which mathematical structures most accurately capture the geometry already latent in LLM embeddings?

And:

> Can we compress these structures into lower-entropy representations without losing meaning?

---

## 🎯 Goals

1. **Create an atlas of linguistic geometries**  
   Compare Euclidean, hyperbolic, spinor, fibre, and sheaf-like spaces.

2. **Probe LLM tensor geometry**  
   Measure cluster structures, curvature signals, separability, cyclicity, and compositionality.

3. **Evaluate compression**  
   Can a geometry represent a concept set more efficiently than raw tensors?

4. **Design tasks and metrics**  
   Structural probes, pairwise relations, cluster cohesion, cross-concept separation, reconstruction error, entropy.

5. **Build geometry-aware corpora**  
   Concept sets that stimulate different types of geometric behaviour.

6. **Support multiple encoders**  
   Compare semantic geometry across different LLM embedding models.

7. **Move toward a “Linguistic Geometry Atlas”**  
   Summaries of how different geometries behave across domains.

---

## 📁 Project Structure

```
linguistic_geometry/
│
├── corpora/
│   ├── semantic_concepts_v0.yml    # ✅ concept sets (emotion, time, space, etc.)
│   └── loader.py                   # ✅ corpus loader with train/test split
│
├── encoders/
│   ├── base.py                     # ✅ encoder interface
│   ├── sentence_transformer.py    # ✅ sentence-transformers implementation
│   └── test_basic.py               # ✅ encoder tests
│
├── geometries/
│   ├── base.py                     # ✅ geometry interface
│   ├── euclidean.py                # ✅ standard euclidean space
│   ├── scalar.py                   # ✅ 1D line geometry
│   ├── hyperbolic.py               # 🔜 hierarchical structure (planned)
│   ├── spinor.py                   # 🔜 phase/rotation geometry (planned)
│   └── sheaf.py                    # 🔮 discourse consistency (future)
│
├── evaluation/
│   ├── train_test_split.py         # 🔜 80/20 split of concept embeddings
│   ├── geometry_evaluator.py      # 🔜 fit on train, validate on test
│   ├── metrics.py                  # 🔜 reconstruction error, entropy, compression
│   └── results_tracker.py          # 🔜 per-geometry, per-category results
│
├── notebooks/
│   ├── 01_corpus_exploration       # ✅ corpus analysis
│   ├── 02_encoder_exploration      # ✅ encoder testing
│   └── 03_geometry_comparison      # 🔜 train/test geometry evaluation
│
├── semantic_atlas.md               # ✅ research notes & findings
├── requirements.txt                # ✅ dependencies
└── README.md                       # ✅ this file

```

**Legend:**
- ✅ Implemented
- 🔜 Planned (near-term)
- 🔮 Future exploration

---

## 🚀 Getting Started

1. Load the concept corpus:
   ```python
   from linguistic_geometry.corpora.loader import load_corpus
   corpus = load_corpus("semantic_concepts_v0")
   ```

2. Select an encoder:

   ```python
   from linguistic_geometry.encoders.llm_openai import OpenAIEncoder
   encoder = OpenAIEncoder(model="your-model-here")
   ```

3. Choose a geometry:

   ```python
   from linguistic_geometry.geometries.euclidean import EuclideanGeometry
   geom = EuclideanGeometry(dim=128)
   ```

4. Run an experiment:

   ```python
   from linguistic_geometry.core.experiment import run_experiment
   result = run_experiment(
       corpus_name="semantic_concepts_v0",
       encoder_name="openai_default",
       geometry_name="euclidean",
       task_name="concept_cluster"
   )
   ```

5. Inspect metrics, distortion, cluster cohesion, etc.

---

## 🧪 Current State

* Initial corpus: `semantic_concepts_v0`
* Core scaffolding implemented (registry, tasks, metrics)
* Euclidean geometry available
* Hyperbolic and spinor geometries planned
* Compression experiments planned
* Early notebooks for exploration

---

## 🧱 Roadmap (Short)

* Spinor and hyperbolic geometry modules
* Sheaf and fibre stubs
* Compression task and metrics
* Multi-encoder comparisons
* “Geometry Atlas” visualisations
* Expanded corpora across more domains

---

## 📜 License & Contribution

This project is research-oriented and exploratory.
Contributions of new geometries, corpora, or tasks are welcome.

---

## 🌌 Why This Project Exists

Language might not be “flat”.
Meanings could bend, branch, loop, and glue together.
The goal of Linguistic Geometry is to make hidden structures visible via agnostic geometry — and to see whether we can encode them in simpler, lower-entropy forms.

A geometric view of language opens the door to new compression methods, new architectures, and new ways of understanding model behaviour.