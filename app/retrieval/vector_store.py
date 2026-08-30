"""
ClauseIQ — Vector Store (FAISS)
Manages the FAISS index and the chunk metadata store.

Design:
  - FAISS IndexFlatIP (inner product on L2-normalized vectors = cosine similarity)
  - Chunk metadata (text, document, page, doc_type) stored in a parallel list
  - Supports save/load for persistence between sessions
  - Supports metadata filtering (doc_type, document name) pre-retrieval

Note: For an interview, you can explain:
  "I used IndexFlatIP with normalized vectors, which gives cosine similarity.
   In production I'd switch to IndexIVFFlat for ANN (approximate nearest neighbor)
   which trades a small accuracy hit for O(log n) instead of O(n) search."
"""

import json
import pickle
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    """
    FAISS-backed vector store with metadata filtering support.
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: Embedding dimension (e.g. 384 for all-MiniLM-L6-v2)
        """
        self.dimension = dimension
        # IndexFlatIP: exact cosine similarity (no approximation)
        self.index = faiss.IndexFlatIP(dimension)
        # Parallel metadata store: index i → chunk dict
        self.chunks: list[dict] = []

    def add_chunks(self, chunks: list[dict], embeddings: np.ndarray) -> None:
        """
        Add chunks and their embeddings to the store.

        Args:
            chunks:     List of chunk dicts (must be parallel to embeddings)
            embeddings: np.ndarray of shape (len(chunks), dimension), float32
        """
        assert len(chunks) == embeddings.shape[0], (
            f"Mismatch: {len(chunks)} chunks vs {embeddings.shape[0]} embeddings"
        )
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        print(f"Vector store: {self.index.ntotal} total chunks indexed")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        filter_doc_type: str | None = None,
        filter_document: str | None = None,
    ) -> list[dict]:
        """
        Retrieve top-k most similar chunks for a query embedding.

        Args:
            query_embedding:  Shape (1, dimension), float32
            top_k:            Number of chunks to return
            filter_doc_type:  If set, only return chunks with this doc_type
            filter_document:  If set, only return chunks from this document

        Returns:
            List of chunk dicts with an added 'score' key (cosine similarity).
        """
        # If filtering, we over-retrieve and post-filter
        fetch_k = top_k * 5 if (filter_doc_type or filter_document) else top_k
        fetch_k = min(fetch_k, self.index.ntotal)  # can't fetch more than we have

        scores, indices = self.index.search(query_embedding, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]

            # Apply metadata filters
            if filter_doc_type and chunk.get("doc_type") != filter_doc_type:
                continue
            if filter_document and chunk.get("document") != filter_document:
                continue

            results.append({**chunk, "score": float(score)})

            if len(results) == top_k:
                break

        return results

    # ─── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """
        Persist the FAISS index and chunk metadata to disk.

        Creates:
            {path}.faiss  — FAISS binary index
            {path}.meta   — pickled chunk metadata
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path) + ".faiss")

        with open(str(path) + ".meta", "wb") as f:
            pickle.dump(
                {"chunks": self.chunks, "dimension": self.dimension}, f
            )

        print(f"Vector store saved: {path}.faiss + {path}.meta")

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        """
        Load a previously saved vector store from disk.
        """
        path = Path(path)
        index = faiss.read_index(str(path) + ".faiss")

        with open(str(path) + ".meta", "rb") as f:
            meta = pickle.load(f)

        store = cls(dimension=meta["dimension"])
        store.index = index
        store.chunks = meta["chunks"]

        print(
            f"Vector store loaded: {store.index.ntotal} chunks "
            f"(dim={store.dimension})"
        )
        return store

    @property
    def total_chunks(self) -> int:
        return self.index.ntotal
