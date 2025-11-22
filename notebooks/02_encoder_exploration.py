# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Encoder Exploration
# 
# This notebook explores different text encoders for the Linguistic Geometry project.
# 
# **What we'll test:**
# 1. Basic encoding functionality
# 2. Semantic similarity patterns
# 3. Corpus encoding performance
# 4. Model comparisons
# 
# **Goal:** Understand how different encoders represent semantic concepts as numerical embeddings.

# %% [markdown]
# ## Setup

# %%
import sys
from pathlib import Path

# Add project root to path
project_root = Path.cwd().parent if 'notebooks' in str(Path.cwd()) else Path.cwd()
sys.path.insert(0, str(project_root))

from encoders import SentenceTransformerEncoder
from corpora.loader import load_corpus

# %% [markdown]
# ## Test 1: Basic Encoding Functionality
# 
# Let's start by testing that our encoder can convert text to embeddings.

# %%
print("=" * 80)
print("Testing Basic Encoding")
print("=" * 80)
print()

# Initialize encoder
print("Loading encoder (all-MiniLM-L6-v2)...")
encoder = SentenceTransformerEncoder()
print(f"✓ Encoder loaded: {encoder}")
print(f"  Embedding dimension: {encoder.embedding_dim}D")
print()

# Test single string
text = "hello world"
embedding = encoder.encode(text)
print(f"Single text: '{text}'")
print(f"  Embedding shape: {embedding.shape}")
print(f"  First 5 values: {embedding[0, :5]}")
print()

# Test multiple strings
texts = ["happy", "sad", "joyful", "miserable"]
embeddings = encoder.encode(texts)
print(f"Multiple texts: {texts}")
print(f"  Embeddings shape: {embeddings.shape}")
print()

# %% [markdown]
# ## Test 2: Semantic Similarity
# 
# Let's test that semantically similar words have similar embeddings.
# We'll compute cosine similarities between word embeddings.

# %%
print("=" * 80)
print("Testing Semantic Similarity")
print("=" * 80)
print()

encoder = SentenceTransformerEncoder()

# Encode words
words = ["happy", "joyful", "sad", "freezing", "hot"]
embeddings = encoder.encode(words)

# Compute cosine similarities
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(embeddings)

print("Cosine similarities:")
print(f"{'':12s}", end="")
for word in words:
    print(f"{word:10s}", end="")
print()

for i, word in enumerate(words):
    print(f"{word:12s}", end="")
    for j in range(len(words)):
        print(f"{similarities[i, j]:10.3f}", end="")
    print()
print()

# Check specific pairs
happy_joyful = similarities[0, 1]
happy_sad = similarities[0, 2]
happy_freezing = similarities[0, 3]

print("Expected patterns:")
print(f"  happy ↔ joyful:   {happy_joyful:.3f} (should be high)")
print(f"  happy ↔ sad:      {happy_sad:.3f} (should be medium)")
print(f"  happy ↔ freezing: {happy_freezing:.3f} (should be low)")
print()

# %% [markdown]
# ## Test 3: Corpus Encoding
# 
# Now let's test encoding a concept cluster from our semantic corpus.

# %%
print("=" * 80)
print("Testing Corpus Encoding")
print("=" * 80)
print()

# Load corpus
corpus = load_corpus("semantic_concepts_v0")
print(f"Loaded corpus: {corpus.corpus_id}")
print(f"Total clusters: {len(corpus)}")
print()

# Get first cluster
cluster = corpus.clusters[0]
print(f"Test cluster: {cluster.domain}/{cluster.subdomain}")
print(f"Concept: {cluster.concept}")
print(f"Size: {len(cluster)} expressions")
print(f"Examples: {', '.join(cluster.expressions[:5])}")
print()

# Encode cluster
encoder = SentenceTransformerEncoder()
embeddings = encoder.encode(cluster.expressions, show_progress_bar=True)

print(f"✓ Encoded {len(cluster)} expressions")
print(f"  Embeddings shape: {embeddings.shape}")
print(f"  Memory: {embeddings.nbytes / 1024:.1f} KB")
print()

# %% [markdown]
# ## Test 4: Model Comparison
# 
# Let's compare different sentence-transformer models to see their characteristics.

# %%
print("=" * 80)
print("Comparing Different Models")
print("=" * 80)
print()

models = [
    ("all-MiniLM-L6-v2", "Fast & lightweight"),
    ("all-mpnet-base-v2", "High quality"),
]

test_text = "The quick brown fox jumps over the lazy dog"

for model_name, description in models:
    print(f"Model: {model_name}")
    print(f"  Description: {description}")
    
    try:
        encoder = SentenceTransformerEncoder(model_name)
        embedding = encoder.encode(test_text)
        
        print(f"  ✓ Dimension: {encoder.embedding_dim}D")
        print(f"  ✓ Shape: {embedding.shape}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print()

# %% [markdown]
# ## Summary
# 
# This notebook demonstrated:
# 
# 1. **Basic Encoding** - Converting text to numerical embeddings
# 2. **Semantic Similarity** - Similar words have similar embeddings
# 3. **Corpus Integration** - Encoding concept clusters from our semantic corpus
# 4. **Model Comparison** - Different models have different characteristics
# 
# **Next Steps:**
# - Connect these encoders to the geometry fitting system
# - Test which geometric structures best explain semantic concepts
# - Discover if emotions, temperatures, etc. have intrinsic geometric patterns
# 
# **Key Insight:** We now have a working system to convert semantic concepts into numerical representations that can be analyzed for geometric structure!

# %%
print("🎉 Encoder exploration complete!")
print("Ready to discover geometric patterns in language!")