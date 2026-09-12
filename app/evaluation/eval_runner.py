"""
ClauseIQ — Evaluation Runner
Runs the eval set against the pipeline and reports metrics.

Usage:
    python -m app.evaluation.eval_runner \
        --index_path data/faiss_index \
        --eval_set data/eval_set.json \
        --chunk_size 800 \
        --use_reranker

Output:
    - Prints results table
    - Appends row to experiments/results.md
"""

import argparse
import json
import time
from pathlib import Path

from app.pipeline import RAGPipeline
from app.evaluation.metrics import compute_retrieval_metrics, compute_refusal_rate


def run_eval(
    pipeline: RAGPipeline,
    eval_set: list[dict],
    k: int = 5,
    run_llm: bool = False,
) -> dict:
    """
    Run the eval set through the pipeline.

    Args:
        pipeline:  Loaded RAGPipeline
        eval_set:  List of eval question dicts
        k:         Top-k for retrieval metrics
        run_llm:   If True, also call the LLM and measure refusal rate

    Returns:
        Full results dict with all metrics.
    """
    results = []
    print(f"\nRunning eval on {len(eval_set)} questions (top-{k})...")

    for i, item in enumerate(eval_set, 1):
        question = item["question"]
        is_answerable = item.get("is_answerable", True)

        # Retrieval only
        retrieved = pipeline.retrieve(question)

        result = {
            "question": question,
            "expected_document": item.get("expected_document", ""),
            "expected_page": item.get("expected_page", -1),
            "is_answerable": is_answerable,
            "retrieved_chunks": retrieved,
        }

        # Optional LLM generation
        if run_llm:
            llm_result = pipeline.ask(question)
            result["answer"] = llm_result["answer"]

        results.append(result)

        if i % 10 == 0:
            print(f"  Progress: {i}/{len(eval_set)}")

    retrieval_metrics = compute_retrieval_metrics(results, k=k)
    refusal_metrics = compute_refusal_rate(results) if run_llm else {}

    all_metrics = {**retrieval_metrics, **refusal_metrics}

    print("\n" + "=" * 50)
    print("EVAL RESULTS")
    print("=" * 50)
    for key, val in all_metrics.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.1%}")
        else:
            print(f"  {key}: {val}")

    return {"metrics": all_metrics, "results": results}


def append_to_results_table(
    results_path: str | Path,
    config_label: str,
    metrics: dict,
) -> None:
    """
    Append a row to the experiments/results.md Markdown table.
    Creates the file with header if it doesn't exist.
    """
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "| Config | Chunk Size | Reranker | "
        "Recall@5 | Hit Rate | Refusal Rate |\n"
        "|---|---|---|---|---|---|\n"
    )

    row = (
        f"| {config_label} | "
        f"{metrics.get('chunk_size', '—')} | "
        f"{metrics.get('use_reranker', '—')} | "
        f"{metrics.get('recall_at_5', 0):.1%} | "
        f"{metrics.get('hit_rate', 0):.1%} | "
        f"{metrics.get('refusal_rate', '—')} |\n"
    )

    if not results_path.exists():
        results_path.write_text(
            "# ClauseIQ — Experiment Results\n\n" + header + row
        )
    else:
        with open(results_path, "a") as f:
            f.write(row)

    print(f"\nResults appended to: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ClauseIQ eval set")
    parser.add_argument("--index_path", default="data/faiss_index")
    parser.add_argument("--eval_set", default="data/eval_set.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--run_llm", action="store_true")
    parser.add_argument("--label", default="experiment")
    args = parser.parse_args()

    # Load pipeline
    pipeline = RAGPipeline.load(args.index_path)

    # Load eval set
    eval_set = json.loads(Path(args.eval_set).read_text())
    print(f"Loaded eval set: {len(eval_set)} questions")

    # Run
    output = run_eval(pipeline, eval_set, k=args.k, run_llm=args.run_llm)

    # Log to results table
    append_to_results_table(
        "experiments/results.md",
        config_label=args.label,
        metrics=output["metrics"],
    )
