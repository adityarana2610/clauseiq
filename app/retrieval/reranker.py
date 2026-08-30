"""
ClauseIQ — Cross-Encoder Reranker
Takes the top-K retrieved chunks and re-scores them with a cross-encoder.

Why two stages?
  - Bi-encoder (FAISS): fast vector search, optimized for recall
  - Cross-encoder (reranker): slower but more accurate — reads query + chunk together
  - Two-stage = best of both: high recall + high precision

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS MARCO passage ranking
  - Works well for Q&A / document retrieval
  - ~66MB, runs on CPU in ~50ms per batch
"""

from sentence_transformers import CrossEncoder

_DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """
    Cross-encoder reranker for two-stage retrieval.
    """

    def __init__(self, model_name: str = _DEFAULT_RERANKER_MODEL):
        print(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Rerank retrieved chunks using cross-encoder scores.

        Args:
            query:   The user's question
            chunks:  List of chunk dicts (as returned by VectorStore.search)
            top_k:   Number of chunks to return after reranking

        Returns:
            Top-k chunks sorted by reranker score (descending),
            each with an added 'rerank_score' key.
        """
        if not chunks:
            return []

        # Build (query, passage) pairs for cross-encoder
        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self.model.predict(pairs)

        # Attach reranker scores
        reranked = [
            {**chunk, "rerank_score": float(score)}
            for chunk, score in zip(chunks, scores)
        ]

        # Sort descending by reranker score
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:top_k]
