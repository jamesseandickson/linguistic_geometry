# Encoders

Convert text to embeddings using various models.

## Available Encoders

### SentenceTransformerEncoder (Free, Local)

Uses the `sentence-transformers` library with models from HuggingFace.
- ✅ **Free** - No API keys required
- ✅ **Local** - Runs on your machine (CPU or GPU)
- ✅ **Fast** - Optimized for semantic similarity tasks

**Popular Models:**
- `all-MiniLM-L6-v2` - Fast & lightweight (384D) - **Default**
- `all-mpnet-base-v2` - High quality (768D)
- `paraphrase-multilingual-MiniLM-L12-v2` - Multilingual (384D)

**Usage:**
```python
from encoders import SentenceTransformerEncoder

# Use default model (all-MiniLM-L6-v2)
encoder = SentenceTransformerEncoder()

# Or specify a different model
encoder = SentenceTransformerEncoder("all-mpnet-base-v2")

# Encode text
embeddings = encoder.encode(["happy", "sad", "joyful"])
print(embeddings.shape)  # (3, 384)
```

**Convenience Functions:**
```python
from encoders.sentence_transformer import (
    get_fast_encoder,      # all-MiniLM-L6-v2 (384D)
    get_quality_encoder,   # all-mpnet-base-v2 (768D)
    get_multilingual_encoder  # multilingual (384D)
)

encoder = get_fast_encoder()
```

## Testing

Run the test suite:
```bash
python -m encoders.test_encoders
```

This will:
1. Test basic encoding functionality
2. Verify semantic similarity (similar words → similar embeddings)
3. Encode a concept cluster from the corpus
4. Compare different models

## Adding New Encoders

To add a new encoder (e.g., OpenAI, Cohere, etc.):

1. Create a new file in `encoders/` (e.g., `openai_encoder.py`)
2. Inherit from `BaseEncoder`
3. Implement `encode()` and `embedding_dim` property
4. Add to `__init__.py`

Example:
```python
from .base import BaseEncoder
import numpy as np

class MyEncoder(BaseEncoder):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        # Initialize your model here
    
    def encode(self, texts, **kwargs) -> np.ndarray:
        # Implement encoding logic
        pass
    
    @property
    def embedding_dim(self) -> int:
        return 1536  # or whatever your model uses
```

## Model Comparison

| Model | Dimensions | Speed | Quality | Free? |
|-------|-----------|-------|---------|-------|
| all-MiniLM-L6-v2 | 384 | ⚡⚡⚡ | ⭐⭐ | ✅ |
| all-mpnet-base-v2 | 768 | ⚡⚡ | ⭐⭐⭐ | ✅ |
| OpenAI text-embedding-3-small | 1536 | ⚡⚡⚡ | ⭐⭐⭐ | ❌ (API key) |
| OpenAI text-embedding-3-large | 3072 | ⚡⚡ | ⭐⭐⭐⭐ | ❌ (API key) |

## Next Steps

Once encoders are working, you can:
1. Use them with the `geometries/` system to fit different geometric structures
2. Run experiments via `core/experiment.py`
3. Compare how different encoders represent semantic concepts
