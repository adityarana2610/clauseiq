"""
ClauseIQ — Ingestion Runner
Ingests all PDFs from raw_pdfs_v2/, builds the FAISS index, and
runs a sanity-check query to verify retrieval works.

Usage:
    python scripts/run_ingestion.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline import RAGPipeline

PDF_DIR = "raw_pdfs_v2"
INDEX_PATH = "data/faiss_index"


def main():
    print("=" * 60)
    print("ClauseIQ — Ingestion Runner")
    print("=" * 60)

    # Initialize pipeline (no LLM needed for ingestion)
    pipeline = RAGPipeline(
        chunk_size=800,
        chunk_overlap=100,
        use_reranker=False,  # don't load reranker for ingestion
    )

    # Run ingestion
    summary = pipeline.ingest(pdf_dir=PDF_DIR)

    # Save index
    pipeline.save(INDEX_PATH)
    print(f"\nIndex saved to: {INDEX_PATH}")

    # Sanity check — retrieve for a known fact
    print("\n" + "=" * 60)
    print("SANITY CHECK — Retrieval Test")
    print("=" * 60)

    test_questions = [
        "What is the dwelling coverage limit for the standard homeowners policy?",
        "What is the flood damage deductible?",
        "Are cyber attacks covered?",
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        chunks = pipeline.retrieve(q)
        if chunks:
            top = chunks[0]
            print(f"  Top hit: {top['document']} — page {top['page']} (score: {top['score']:.4f})")
            print(f"  Snippet: {top['text'][:150]}...")
        else:
            print("  No results!")

    print("\n" + "=" * 60)
    print("Ingestion complete!")
    print(f"  PDFs: {summary['num_pdfs']}")
    print(f"  Pages: {summary['num_pages_after_cleaning']}")
    print(f"  Chunks: {summary['num_chunks']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
