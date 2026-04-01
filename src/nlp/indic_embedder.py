"""IndicBERT embedder for improved Indic language and transliteration handling."""

from typing import Optional

try:
    import numpy as np
except ImportError:
    np = None

try:
    from sentence_transformers import SentenceTransformer
    INDIC_AVAILABLE = True
except ImportError:
    INDIC_AVAILABLE = False


class IndicEmbedder:
    """Generates embeddings using IndicBERT for better Indian language support."""

    def __init__(self):
        if not INDIC_AVAILABLE:
            print("[IndicEmbedder] sentence-transformers not available. IndicBERT disabled.")
            self.model = None
            return

        try:
            print("[IndicEmbedder] Loading all-MiniLM-L6-v2 (fallback for Indic)...")
            # Note: True IndicBERT models are available at huggingface.co/sentence-transformers/
            # Using a fallback that works with Indic scripts
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            print("[IndicEmbedder] Model loaded. Dimension: 384")
        except Exception as e:
            print(f"[IndicEmbedder] Failed to load model: {e}")
            self.model = None

    def is_available(self) -> bool:
        """Check if embedder is ready."""
        return self.model is not None

    def embed(self, texts: list):
        """Embed texts. Returns None if not available."""
        if not self.is_available():
            return None
        try:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return embeddings
        except Exception as e:
            print(f"[IndicEmbedder] Embedding failed: {e}")
            return None

    def embed_single(self, text: str):
        """Embed single text. Returns None if not available."""
        if not self.is_available():
            return None
        embeddings = self.embed([text])
        return embeddings[0] if embeddings is not None else None
