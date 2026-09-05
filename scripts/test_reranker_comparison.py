"""
ClauseIQ — Side-by-Side Reranker Tester
Accepts a question via command-line argument OR interactive prompt:
  python scripts/test_reranker_comparison.py "What is the dwelling coverage limit for the premium homeowners policy?"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline import RAGPipeline
from app.retrieval.vector_store import VectorStore


def run_comparison(pipeline, query: str):
    print("\n" + "=" * 65, flush=True)
    print(f"QUESTION: {query}", flush=True)
    print("=" * 65, flush=True)

    # 1. Vector Search (Bi-encoder only)
    q_emb = pipeline.embedder.embed_query(query)
    vector_candidates = pipeline.vector_store.search(q_emb, top_k=20)
    base_top5 = vector_candidates[:5]

    # 2. Rerank (Cross-encoder)
    reranked_top5 = pipeline.reranker.rerank(query, vector_candidates, top_k=5)

    print("\n" + "-" * 65, flush=True)
    print("STAGE 1: BI-ENCODER VECTOR SEARCH (Top-5 by Cosine Similarity)", flush=True)
    print("-" * 65, flush=True)
    for rank, c in enumerate(base_top5, 1):
        snippet = c["text"].replace("\n", " ")[:85] + "..."
        print(f"  Rank #{rank} | Score: {c['score']:.4f} | {c['document']} (p.{c['page']})", flush=True)
        print(f"         \"{snippet}\"", flush=True)

    print("\n" + "-" * 65, flush=True)
    print("STAGE 2: CROSS-ENCODER RERANKER (Top-5 Re-scored & Re-ordered)", flush=True)
    print("-" * 65, flush=True)
    for rank, c in enumerate(reranked_top5, 1):
        prev_rank = "was not in top-5"
        for orig_idx, orig_c in enumerate(base_top5, 1):
            if orig_c["document"] == c["document"] and orig_c["page"] == c["page"]:
                prev_rank = f"was Rank #{orig_idx}"
                break

        snippet = c["text"].replace("\n", " ")[:85] + "..."
        print(f"  Rank #{rank} | Score: {c['rerank_score']:+.4f} ({prev_rank}) | {c['document']} (p.{c['page']})", flush=True)
        print(f"         \"{snippet}\"", flush=True)

    # Compare top-1
    base_top1 = f"{base_top5[0]['document']}:p{base_top5[0]['page']}"
    rerank_top1 = f"{reranked_top5[0]['document']}:p{reranked_top5[0]['page']}"
    if base_top1 == rerank_top1:
        print(f"\n[Observation] Both placed '{rerank_top1}' at Rank #1. Reranker refined positions 2-5.", flush=True)
    else:
        print(f"\n[Observation] RANK #1 SWAPPED! Bi-encoder chose '{base_top1}', but Cross-Encoder promoted '{rerank_top1}' to #1!", flush=True)


def main():
    print("=" * 65, flush=True)
    print("ClauseIQ — Interactive Reranker Comparison Tester", flush=True)
    print("=" * 65, flush=True)
    print("Initializing embedding model, reranker, and vector database...", flush=True)

    pipeline = RAGPipeline(
        chunk_size=800,
        chunk_overlap=100,
        use_reranker=True,
        skip_llm=True,
    )

    if Path("data/faiss_index.faiss").exists():
        pipeline.vector_store = VectorStore.load("data/faiss_index")
    else:
        pipeline.ingest("raw_pdfs_v2", chunk_size=800, chunk_overlap=100)
        pipeline.save("data/faiss_index")

    print("\n[OK] Ready! Type ANY insurance question below.", flush=True)
    print("Type 'exit' or 'quit' to stop.\n", flush=True)

    while True:
        try:
            print("Ask a question:")
            query = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break

        run_comparison(pipeline, query)
        print("\n" + "=" * 65 + "\n", flush=True)


if __name__ == "__main__":
    main()
