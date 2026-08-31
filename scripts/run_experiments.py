"""
ClauseIQ — Chunking Experiments
Runs full ingestion + retrieval eval for multiple chunk configs.
Produces the comparison table required by the roadmap.

Usage:
    python scripts/run_experiments.py
    python scripts/run_experiments.py --with-reranker   # also test reranker
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline import RAGPipeline
from app.evaluation.metrics import compute_retrieval_metrics

PDF_DIR = "raw_pdfs_v2"
EVAL_SET_PATH = "data/eval_set.json"
RESULTS_PATH = "experiments/results.md"

# Experiment configs: (label, chunk_size, chunk_overlap)
EXPERIMENTS = [
    ("chunk_500", 500, 50),
    ("chunk_800", 800, 100),
    ("chunk_1200", 1200, 150),
]


def run_single_experiment(
    label: str,
    chunk_size: int,
    chunk_overlap: int,
    eval_set: list[dict],
    use_reranker: bool = False,
    k: int = 5,
) -> dict:
    """Run a single ingestion + eval cycle."""
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT: {label} (chunk={chunk_size}, overlap={chunk_overlap}, reranker={use_reranker})")
    print(f"{'=' * 60}")

    start = time.time()

    # Build pipeline
    pipeline = RAGPipeline(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        use_reranker=use_reranker,
    )

    # Ingest
    summary = pipeline.ingest(pdf_dir=PDF_DIR, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Eval — retrieval only
    results = []
    answerable = [q for q in eval_set if q.get("is_answerable", True)]
    unanswerable = [q for q in eval_set if not q.get("is_answerable", True)]

    print(f"\nEvaluating {len(answerable)} answerable + {len(unanswerable)} unanswerable questions...")

    for item in eval_set:
        retrieved = pipeline.retrieve(item["question"])
        results.append({
            "question": item["question"],
            "expected_document": item.get("expected_document", ""),
            "expected_page": item.get("expected_page", -1),
            "is_answerable": item.get("is_answerable", True),
            "retrieved_chunks": retrieved,
        })

    metrics = compute_retrieval_metrics(results, k=k)
    elapsed = time.time() - start

    # Print per-question breakdown for answerable
    print(f"\n{'-' * 60}")
    print(f"PER-QUESTION RESULTS (answerable only):")
    print(f"{'-' * 60}")
    hits = 0
    misses = []
    for item, result in zip(eval_set, results):
        if not item.get("is_answerable", True):
            continue
        retrieved = result["retrieved_chunks"][:k]
        found = any(
            c.get("document") == item.get("expected_document")
            and c.get("page") == item.get("expected_page")
            for c in retrieved
        )
        status = "HIT" if found else "MISS"
        if found:
            hits += 1
        else:
            misses.append(item["id"])
        top_doc = retrieved[0]["document"] if retrieved else "—"
        top_page = retrieved[0]["page"] if retrieved else "—"
        print(f"  {item['id']}: {status} (expected {item.get('expected_document')}:p{item.get('expected_page')}, "
              f"got {top_doc}:p{top_page})")

    if misses:
        print(f"\n  MISSES: {', '.join(misses)}")

    return {
        "label": label,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "use_reranker": use_reranker,
        "num_chunks": summary["num_chunks"],
        "recall_at_5": metrics.get(f"recall_at_{k}", 0.0),
        "hit_rate": metrics.get("hit_rate", 0.0),
        "elapsed_seconds": round(elapsed, 1),
    }


def append_results_to_md(results: list[dict], path: str):
    """Write a proper results table to the markdown file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# ClauseIQ — Experiment Results\n",
        "\nGenerated automatically by `scripts/run_experiments.py`\n",
        "\n---\n",
        "\n## Retrieval Experiments\n",
        "\n| Config | Chunk Size | Overlap | Reranker | Chunks | Recall@5 | Hit Rate | Time |\n",
        "|---|---|---|---|---|---|---|---|\n",
    ]

    for r in results:
        lines.append(
            f"| {r['label']} | {r['chunk_size']} | {r['chunk_overlap']} | "
            f"{'Yes' if r['use_reranker'] else 'No'} | {r['num_chunks']} | "
            f"{r['recall_at_5']:.1%} | {r['hit_rate']:.1%} | {r['elapsed_seconds']}s |\n"
        )

    # Best config analysis
    best = max(results, key=lambda x: x["recall_at_5"])
    lines.append(f"\n---\n")
    lines.append(f"\n## Key Findings\n\n")
    lines.append(f"- **Best chunk size**: {best['chunk_size']} tokens\n")
    lines.append(f"- **Best Recall@5**: {best['recall_at_5']:.1%}\n")
    lines.append(f"- **Best Hit Rate**: {best['hit_rate']:.1%}\n")

    # Reranker comparison if available
    no_rerank = [r for r in results if not r["use_reranker"] and r["chunk_size"] == best["chunk_size"]]
    with_rerank = [r for r in results if r["use_reranker"] and r["chunk_size"] == best["chunk_size"]]
    if no_rerank and with_rerank:
        lines.append(
            f"- **Reranking improvement**: Recall@5 from "
            f"{no_rerank[0]['recall_at_5']:.1%} → {with_rerank[0]['recall_at_5']:.1%}\n"
        )

    path.write_text("".join(lines), encoding="utf-8")
    print(f"\nResults written to: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-reranker", action="store_true",
                        help="Also run best config with reranker")
    parser.add_argument("--k", type=int, default=5, help="Top-k for eval")
    args = parser.parse_args()

    # Load eval set
    eval_set = json.loads(Path(EVAL_SET_PATH).read_text(encoding="utf-8"))
    answerable = [q for q in eval_set if q.get("is_answerable", True)]
    unanswerable = [q for q in eval_set if not q.get("is_answerable", True)]
    print(f"Loaded eval set: {len(eval_set)} questions ({len(answerable)} answerable, {len(unanswerable)} unanswerable)")

    all_results = []

    # Run chunking experiments (without reranker)
    for label, chunk_size, overlap in EXPERIMENTS:
        result = run_single_experiment(
            label=label,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            eval_set=eval_set,
            use_reranker=False,
            k=args.k,
        )
        all_results.append(result)
        print(f"\n>>> {label}: Recall@5={result['recall_at_5']:.1%}, Hit Rate={result['hit_rate']:.1%}")

    # Optionally run best config with reranker
    if args.with_reranker:
        best = max(all_results, key=lambda x: x["recall_at_5"])
        result = run_single_experiment(
            label=f"{best['label']}_reranked",
            chunk_size=best["chunk_size"],
            chunk_overlap=best["chunk_overlap"],
            eval_set=eval_set,
            use_reranker=True,
            k=args.k,
        )
        all_results.append(result)
        print(f"\n>>> {result['label']}: Recall@5={result['recall_at_5']:.1%}, Hit Rate={result['hit_rate']:.1%}")

    # Write results
    append_results_to_md(all_results, RESULTS_PATH)

    # Final summary
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
    print(f"{'Config':<25} {'Chunks':<8} {'Recall@5':<10} {'Hit Rate':<10}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['label']:<25} {r['num_chunks']:<8} {r['recall_at_5']:<10.1%} {r['hit_rate']:<10.1%}")


if __name__ == "__main__":
    main()
