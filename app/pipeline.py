"""
ClauseIQ — RAG Pipeline
Orchestrates the full pipeline:
  ingest → embed → store → retrieve → rerank → generate → cite

This is the core class you'll reference in interviews.
"""

from pathlib import Path

from app.ingestion.pdf_parser import extract_all_pdfs
from app.ingestion.cleaner import clean_pages
from app.ingestion.chunker import chunk_pages
from app.retrieval.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.reranker import Reranker
from app.generation.llm import MistralLLM


class RAGPipeline:
    """
    End-to-end RAG pipeline for insurance policy Q&A.

    Usage:
        pipeline = RAGPipeline()
        pipeline.ingest(pdf_dir="data/raw_pdfs")
        pipeline.save("data/faiss_index")

        # Later session:
        pipeline = RAGPipeline.load("data/faiss_index")
        result = pipeline.ask("What is the deductible for flood damage?")
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        mistral_model: str = "mistral-small-latest",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        retrieval_top_k: int = 20,
        reranker_top_k: int = 5,
        use_reranker: bool = True,
        mistral_api_key: str | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_top_k = retrieval_top_k
        self.reranker_top_k = reranker_top_k
        self.use_reranker = use_reranker

        self.embedder = Embedder(model_name=embedding_model)
        self.vector_store = VectorStore(dimension=self.embedder.dimension)
        self.reranker = Reranker() if use_reranker else None
        self.llm = MistralLLM(api_key=mistral_api_key, model=mistral_model)

    # ─── Ingestion ────────────────────────────────────────────────────────────

    def ingest(
        self,
        pdf_dir: str | Path,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> dict:
        """
        Full ingestion: PDF → clean → chunk → embed → index.

        Args:
            pdf_dir:       Directory containing PDF files
            chunk_size:    Override default chunk size (for experiments)
            chunk_overlap: Override default overlap (for experiments)

        Returns:
            Ingestion summary dict.
        """
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap

        print("=" * 60)
        print("ClauseIQ — Starting Ingestion Pipeline")
        print("=" * 60)

        # Step 1: Parse PDFs
        print("\n[1/4] Parsing PDFs...")
        pages = extract_all_pdfs(pdf_dir)

        # Step 2: Clean text
        print("\n[2/4] Cleaning text...")
        pages = clean_pages(pages)

        # Step 3: Chunk
        print(f"\n[3/4] Chunking (size={chunk_size}, overlap={chunk_overlap})...")
        chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Step 4: Embed + index
        print("\n[4/4] Embedding + indexing...")
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed(texts)
        self.vector_store.add_chunks(chunks, embeddings)

        summary = {
            "num_pdfs": len(set(c["document"] for c in chunks)),
            "num_pages_after_cleaning": len(pages),
            "num_chunks": len(chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }
        print(f"\nIngestion complete: {summary}")
        return summary

    # ─── Query ────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        filter_doc_type: str | None = None,
        filter_document: str | None = None,
    ) -> list[dict]:
        """
        Retrieve (and optionally rerank) chunks for a query.
        Use this for retrieval-only evaluation (no LLM call).
        """
        query_embedding = self.embedder.embed_query(query)

        chunks = self.vector_store.search(
            query_embedding,
            top_k=self.retrieval_top_k,
            filter_doc_type=filter_doc_type,
            filter_document=filter_document,
        )

        if self.use_reranker and self.reranker and chunks:
            chunks = self.reranker.rerank(query, chunks, top_k=self.reranker_top_k)

        return chunks

    def ask(
        self,
        question: str,
        filter_doc_type: str | None = None,
        filter_document: str | None = None,
    ) -> dict:
        """
        Full RAG answer: retrieve → generate.

        Returns:
            dict with keys:
                - question: str
                - answer: str
                - sources: list of {document, page, doc_type, score/rerank_score}
                - model: str
                - num_chunks_used: int
        """
        chunks = self.retrieve(
            question,
            filter_doc_type=filter_doc_type,
            filter_document=filter_document,
        )

        if not chunks:
            return {
                "question": question,
                "answer": "I cannot find any relevant information in the provided documents.",
                "sources": [],
                "model": self.llm.model,
                "num_chunks_used": 0,
            }

        result = self.llm.generate(question=question, chunks=chunks)

        sources = [
            {
                "document": c["document"],
                "page": c["page"],
                "doc_type": c["doc_type"],
                "score": c.get("rerank_score", c.get("score", 0.0)),
            }
            for c in chunks
        ]

        return {
            "question": question,
            "answer": result["answer"],
            "sources": sources,
            "model": result["model"],
            "num_chunks_used": len(chunks),
        }

    # ─── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save the vector store to disk."""
        self.vector_store.save(path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        mistral_api_key: str | None = None,
        mistral_model: str = "mistral-small-latest",
        use_reranker: bool = True,
    ) -> "RAGPipeline":
        """Load a previously saved pipeline from disk."""
        store = VectorStore.load(path)
        pipeline = cls(
            mistral_api_key=mistral_api_key,
            mistral_model=mistral_model,
            use_reranker=use_reranker,
        )
        pipeline.vector_store = store
        return pipeline

    @property
    def num_documents(self) -> int:
        docs = set(c["document"] for c in self.vector_store.chunks)
        return len(docs)

    @property
    def num_chunks(self) -> int:
        return self.vector_store.total_chunks
