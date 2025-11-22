# %% [markdown]
# # 📚 Corpus Exploration
#
# This notebook explores the semantic concept corpus and demonstrates the loader functionality.
#
# **Goal:** Understand what concepts we have and which domains might map to different geometries.

# %% [markdown]
# ## 1. Load the Corpus

# %%
import sys
from pathlib import Path

# Add project root to path
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))

from corpora.loader import load_corpus

# %%
# Load the corpus
corpus = load_corpus("semantic_concepts_v0")

print(f"Loaded: {corpus.corpus_id}")
print(f"Version: {corpus.version}")
print(f"Language: {corpus.language}")
print(f"Total clusters: {len(corpus)}")
print(f"Total expressions: {len(corpus.get_all_expressions())}")

# %% [markdown]
# ## 2. Explore Domains
#
# What semantic domains do we have?

# %%
print("Available domains:")
print(sorted(corpus.domains))

# %%
# Domain breakdown
print("\nDomain statistics:")
print("-" * 60)
for domain in sorted(corpus.domains):
    clusters = corpus.filter_by_domain(domain)
    total_expr = sum(len(c) for c in clusters)
    print(f"{domain:20s} | {len(clusters):2d} clusters | {total_expr:3d} expressions")

# %% [markdown]
# ## 3. Examine Specific Domains
#
# Let's look at domains that might have interesting geometric properties.

# %% [markdown]
# ### Emotion (Potential Spinor/Polarity Candidate)

# %%
emotion_clusters = corpus.filter_by_domain("emotion")

print(f"Emotion domain: {len(emotion_clusters)} clusters\n")
for cluster in emotion_clusters:
    print(f"Subdomain: {cluster.subdomain}")
    print(f"Concept: {cluster.concept}")
    print(f"Size: {len(cluster)} expressions")
    print(f"Examples: {cluster.expressions[:10]}")
    print()

# %% [markdown]
# ### Social Structure (Potential Hyperbolic/Hierarchy Candidate)

# %%
social_clusters = corpus.filter_by_domain("social_structure")

print(f"Social structure domain: {len(social_clusters)} clusters\n")
for cluster in social_clusters:
    print(f"Subdomain: {cluster.subdomain}")
    print(f"Concept: {cluster.concept}")
    print(f"Size: {len(cluster)} expressions")
    print(f"Examples: {cluster.expressions[:10]}")
    print()

# %% [markdown]
# ### Space (Potential Rotation/Axis Candidate)

# %%
space_clusters = corpus.filter_by_domain("space")

print(f"Space domain: {len(space_clusters)} clusters\n")
for cluster in space_clusters:
    print(f"Subdomain: {cluster.subdomain}")
    print(f"Concept: {cluster.concept}")
    print(f"Size: {len(cluster)} expressions")
    print(f"Examples: {cluster.expressions[:10]}")
    print()

# %% [markdown]
# ### Time (Potential Cyclic/Phase Candidate)

# %%
time_clusters = corpus.filter_by_domain("time")

print(f"Time domain: {len(time_clusters)} clusters\n")
for cluster in time_clusters:
    print(f"Subdomain: {cluster.subdomain}")
    print(f"Concept: {cluster.concept}")
    print(f"Size: {len(cluster)} expressions")
    print(f"Examples: {cluster.expressions[:10]}")
    print()

# %% [markdown]
# ## 4. Cluster Size Distribution
#
# How are expressions distributed across clusters?

# %%
cluster_sizes = [len(c) for c in corpus.clusters]

print(f"Cluster size statistics:")
print(f"  Min: {min(cluster_sizes)}")
print(f"  Max: {max(cluster_sizes)}")
print(f"  Mean: {sum(cluster_sizes) / len(cluster_sizes):.1f}")
print(f"  Total expressions: {sum(cluster_sizes)}")

# %% [markdown]
# ## 5. All Clusters Overview

# %%
print("All clusters:")
print("=" * 80)
for i, cluster in enumerate(corpus.clusters, 1):
    print(f"{i:2d}. {cluster.domain:20s} / {cluster.subdomain:25s} ({len(cluster):2d} expr)")
    print(f"    {cluster.concept}")
    print(f"    Examples: {', '.join(cluster.expressions[:5])}...")
    print()

# %% [markdown]
# ## 6. Next Steps
#
# Now that we understand the corpus structure, we can:
#
# 1. **Test different geometries** on different domains
# 2. **Measure compression** (can we represent these concepts in lower dimensions?)
# 3. **Compare distortion** across geometries
# 4. **Identify which geometry fits which domain best**
#
# ### Hypotheses to Test:
#
# - **Emotion** → Spinor/SU(2) (polarity: happy ↔ sad)
# - **Social structure** → Hyperbolic (hierarchy: employee → manager → CEO)
# - **Space** → Rotations/SO(3) (axes: left ↔ right, up ↔ down)
# - **Time** → Cyclic/Phase (tense cycles, aspect)
# - **Modality** → ? (to be discovered)
#
# The loader is **geometry-agnostic** — experiments will discover the truth.
