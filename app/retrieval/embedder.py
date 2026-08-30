"""
ClauseIQ — Embedder
Wraps Sentence Transformers to embed text chunks.
Uses all-MiniLM-L6-v2: fast, 384-dim, strong semantic similarity.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    """
    Lightweight wrapper around Sentence Transformers.
    Singleton-style: load once, embed many times.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"  → Embedding dimension: {self.dimension}")

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """
        Embed a list of strings.

        Args:
            texts:      List of text strings to embed
            batch_size: Processing batch size (tune for your RAM)

        Returns:
            np.ndarray of shape (len(texts), self.dimension), dtype float32
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # for cosine similarity
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Returns:
            np.ndarray of shape (1, self.dimension), dtype float32
        """
        vec = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.astype(np.float32)
