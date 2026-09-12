"""
ClauseIQ — Interactive Full-Pipeline Tester (Phase 3)
Allows asking arbitrary questions against the ingested policy index,
displaying:
  1. Retrieved chunks (before reranking)
  2. Reranked chunks (with cross-encoder scores)
  3. Grounded LLM answer with source citations (via Gemini)
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.pipeline import RAGPipeline


def main():
    print("=" * 65)
    print(" ClauseIQ — Interactive RAG Pipeline Tester (Phase 3)")
    print("=" * 65)
    print("Loading pipeline and vector index...")

    index_path = Path("data/faiss_index")
    if not (Path("data/faiss_index.faiss").exists() and Path("data/faiss_index.meta").exists()):
        print("Error: Vector index not found at data/faiss_index. Please run ingestion first.")
        sys.exit(1)

    pipeline = RAGPipeline.load("data/faiss_index", use_reranker=True)
    print("\nPipeline ready! (Type 'exit' or 'quit' to stop)\n")

    while True:
        try:
            query = input("Ask a question: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting. Goodbye!")
                break

            print("\nSearching and generating answer...")
            result = pipeline.ask(query)

            print("\n" + "─" * 60)
            print("ANSWER:")
            print(result["answer"])
            print("─" * 60)

            print("SOURCES USED (Top reranked chunks):")
            for i, s in enumerate(result["sources"], 1):
                doc = s.get("document", "Unknown")
                page = s.get("page", "?")
                score = s.get("score", 0.0)
                doc_type = s.get("doc_type", "")
                print(f"  [{i}] {doc} (p. {page}) [{doc_type}] — Rerank Score: {score:.3f}")

            print("─" * 60 + "\n")

        except KeyboardInterrupt:
            print("\nExiting. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
