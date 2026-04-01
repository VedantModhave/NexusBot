"""Multilingual Embedder using sentence-transformers. Loads model ONCE."""

import numpy as np
from sentence_transformers import SentenceTransformer


class MultilingualEmbedder:
    """Generates L2-normalized multilingual embeddings."""

    def __init__(self):
        print("[Embedder] Loading paraphrase-multilingual-MiniLM-L12-v2...")
        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        print("[Embedder] Model loaded. Dimension: 384")

    def embed(self, texts: list) -> np.ndarray:
        """Embed a list of texts. Returns shape (N, 384), L2-normalized."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text. Returns shape (384,), L2-normalized."""
        return self.embed([text])[0]
