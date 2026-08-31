"""
ClauseIQ — Interactive Retrieval Tester
Run this script to interactively test the vector database.
Type a question, and it will return the top 3 chunks retrieved from the PDFs.

Usage:
    python scripts/interactive_retrieval.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline import RAGPipeline

INDEX_PATH = "data/faiss_index"

def main():
    print("=" * 60)
    print("ClauseIQ — Interactive Retrieval Tester")
    print("=" * 60)

    print("Loading vector database (FAISS)...")
    try:
        # We load with chunk_size 500 since that was our best performing config
        pipeline = RAGPipeline.load(
            INDEX_PATH,
            chunk_size=500, 
            chunk_overlap=50, 
            use_reranker=True,
            skip_llm=True  # Skip Mistral since we're only testing retrieval
        )
        print("[OK] Database loaded successfully.")
    except Exception as e:
        print(f"[FAIL] Failed to load database: {e}")
        print("Make sure you've run 'python scripts/run_ingestion.py' first.")
        sys.exit(1)

    print("\nType 'quit' or 'exit' to stop.\n")

    while True:
        try:
            query = input("\nAsk a question about the insurance policies:\n> ")
            if query.lower().strip() in ['quit', 'exit', 'q']:
                break
            
            if not query.strip():
                continue

            print("\nSearching...")
            # Get top 3 results
            results = pipeline.retrieve(query)[:3]

            if not results:
                print("No results found.")
                continue

            print("\n--- TOP RESULTS ---")
            for i, res in enumerate(results, 1):
                doc = res.get('document', 'Unknown')
                page = res.get('page', 'Unknown')
                score = res.get('score', 0.0)
                text = res.get('text', '').replace('\n', ' ').strip()
                
                print(f"\n[{i}] {doc} (Page {page}) - Score: {score:.4f}")
                # Print a truncated snippet
                snippet = text if len(text) < 250 else text[:250] + "..."
                print(f"    \"{snippet}\"")
                
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            import traceback
            print(f"\nError occurred:")
            traceback.print_exc()

    print("\nGoodbye!")

if __name__ == "__main__":
    main()
