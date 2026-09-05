"""
ClauseIQ — Reranker + Refusal + Category Experiments
Runs all remaining Phase 2 evaluation experiments in one pass:

  1. chunk_800_reranked  (where baseline is 97.5%)
  2. chunk_1200_reranked (where baseline is 97.5%)
  3. Retrieval-level refusal rate on 10 unanswerable questions
  4. Recall@5 breakdown by question_type category

Usage:
    python scripts/run_phase2_eval.py
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline import RAGPipeline
from app.evaluation.metrics import compute_retrieval_metrics

PDF_DIR = "raw_pdfs_v2"
EVAL_SET_PATH = "data/eval_set.json"
RESULTS_PATH = "experiments/results.md"


# ============================================================
# HELPERS
# ============================================================

def run_single_experiment(label, chunk_size, chunk_overlap, eval_set, use_reranker=False, k=5):
    """Run a single ingestion + retrieval eval cycle. Returns result dict."""
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT: {label} (chunk={chunk_size}, overlap={chunk_overlap}, reranker={use_reranker})")
    print(f"{'=' * 60}")

    start = time.time()

    pipeline = RAGPipeline(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        use_reranker=use_reranker,
        skip_llm=True,
    )
    summary = pipeline.ingest(pdf_dir=PDF_DIR, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    results = []
    for item in eval_set:
        retrieved = pipeline.retrieve(item["question"])
        results.append({
            "id": item["id"],
            "question": item["question"],
            "expected_document": item.get("expected_document", ""),
            "expected_page": item.get("expected_page", -1),
            "is_answerable": item.get("is_answerable", True),
            "category": item.get("category", "unknown"),
            "question_type": item.get("question_type", "unknown"),
            "retrieved_chunks": retrieved,
        })

    metrics = compute_retrieval_metrics(results, k=k)
    elapsed = time.time() - start

    # Per-question breakdown
    print(f"\n{'-' * 60}")
    print("PER-QUESTION RESULTS (answerable only):")
    print(f"{'-' * 60}")
    hits = 0
    misses = []
    for r in results:
        if not r["is_answerable"]:
            continue
        retrieved = r["retrieved_chunks"][:k]
        found = any(
            c.get("document") == r["expected_document"]
            and c.get("page") == r["expected_page"]
            for c in retrieved
        )
        status = "HIT" if found else "MISS"
        if found:
            hits += 1
        else:
            misses.append(r["id"])
        top_doc = retrieved[0]["document"] if retrieved else "-"
        top_page = retrieved[0]["page"] if retrieved else "-"
        print(f"  {r['id']}: {status} (expected {r['expected_document']}:p{r['expected_page']}, "
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
        "raw_results": results,
    }


def compute_refusal_metrics(results, score_threshold=0.40):
    """
    Retrieval-level refusal rate for unanswerable questions.
    If the top-1 retrieved chunk has a score below the threshold,
    we consider the system would effectively 'refuse' (low confidence).
    """
    unanswerable = [r for r in results if not r["is_answerable"]]
    if not unanswerable:
        return {"refusal_rate": None, "details": []}

    details = []
    correct_refusals = 0
    for r in unanswerable:
        chunks = r["retrieved_chunks"]
        top_score = chunks[0]["score"] if chunks else 0.0
        top_doc = chunks[0]["document"] if chunks else "-"
        # Low score = system has nothing relevant = would refuse
        would_refuse = top_score < score_threshold
        if would_refuse:
            correct_refusals += 1
        details.append({
            "id": r["id"],
            "question": r["question"],
            "top_score": round(top_score, 4),
            "top_doc": top_doc,
            "would_refuse": would_refuse,
        })

    return {
        "refusal_rate": correct_refusals / len(unanswerable),
        "correct_refusals": correct_refusals,
        "total_unanswerable": len(unanswerable),
        "details": details,
    }


def compute_category_breakdown(results, k=5):
    """Compute Recall@5 grouped by question_type."""
    by_type = defaultdict(list)
    for r in results:
        if not r["is_answerable"]:
            continue
        by_type[r["question_type"]].append(r)

    breakdown = {}
    for qtype, items in sorted(by_type.items()):
        hits = sum(
            1 for r in items
            if any(
                c.get("document") == r["expected_document"]
                and c.get("page") == r["expected_page"]
                for c in r["retrieved_chunks"][:k]
            )
        )
        breakdown[qtype] = {
            "count": len(items),
            "hits": hits,
            "recall_at_5": hits / len(items) if items else 0.0,
        }
    return breakdown


def write_final_results(
    baseline_rows, reranker_rows, refusal_metrics, category_breakdown, path
):
    """Write the final comprehensive results.md."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# ClauseIQ - Experiment Results\n\n")
    lines.append("Generated automatically by `scripts/run_phase2_eval.py`\n\n")

    # ── Section 1: Retrieval Experiments ──
    lines.append("---\n\n")
    lines.append("## Retrieval Experiments\n\n")
    lines.append("| Config | Chunk Size | Overlap | Reranker | Chunks | Recall@5 | Hit Rate | Time |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")

    all_rows = baseline_rows + reranker_rows
    for r in all_rows:
        lines.append(
            f"| {r['label']} | {r['chunk_size']} | {r['chunk_overlap']} | "
            f"{'Yes' if r['use_reranker'] else 'No'} | {r['num_chunks']} | "
            f"{r['recall_at_5']:.1%} | {r['hit_rate']:.1%} | {r['elapsed_seconds']}s |\n"
        )

    # ── Section 2: Refusal Rate ──
    lines.append("\n---\n\n")
    lines.append("## Hallucination / Refusal Rate Experiments\n\n")
    lines.append("Measured at the retrieval level: if the top-1 chunk's cosine similarity score\n")
    lines.append("is below 0.40, the system has low confidence and would effectively refuse to answer.\n")
    lines.append("This is a retrieval-level proxy; true LLM-based refusal will be tested in Phase 3.\n\n")

    lines.append("| Config | Unanswerable Qs | Correct Refusals | Refusal Rate |\n")
    lines.append("|---|---|---|---|\n")
    lines.append(
        f"| best_config | {refusal_metrics['total_unanswerable']} | "
        f"{refusal_metrics['correct_refusals']} | "
        f"{refusal_metrics['refusal_rate']:.1%} |\n"
    )

    lines.append("\n### Per-Question Detail\n\n")
    lines.append("| Question ID | Question | Top-1 Score | Top-1 Doc | Would Refuse |\n")
    lines.append("|---|---|---|---|---|\n")
    for d in refusal_metrics["details"]:
        q_short = d["question"][:60] + "..." if len(d["question"]) > 60 else d["question"]
        lines.append(
            f"| {d['id']} | {q_short} | {d['top_score']:.4f} | "
            f"{d['top_doc']} | {'Yes' if d['would_refuse'] else 'No'} |\n"
        )

    # ── Section 3: Category Breakdown ──
    lines.append("\n---\n\n")
    lines.append("## Recall@5 by Question Category\n\n")
    lines.append("Breakdown by question difficulty/type to verify high recall isn't masking\n")
    lines.append("weaknesses in harder question categories.\n\n")
    lines.append("| Question Type | Count | Hits | Recall@5 |\n")
    lines.append("|---|---|---|---|\n")
    total_count = 0
    total_hits = 0
    for qtype, data in sorted(category_breakdown.items()):
        lines.append(
            f"| {qtype} | {data['count']} | {data['hits']} | {data['recall_at_5']:.1%} |\n"
        )
        total_count += data["count"]
        total_hits += data["hits"]
    lines.append(f"| **Overall** | **{total_count}** | **{total_hits}** | **{total_hits/total_count:.1%}** |\n")

    # ── Key Findings ──
    lines.append("\n---\n\n")
    lines.append("## Key Findings\n\n")

    best_overall = max(all_rows, key=lambda x: x["recall_at_5"])
    lines.append(f"- **Best chunk size**: {best_overall['chunk_size']} tokens - Recall@5 of {best_overall['recall_at_5']:.1%}\n")

    # Find meaningful reranker delta (on chunk_800)
    base_800 = next((r for r in all_rows if r["label"] == "chunk_800"), None)
    rerank_800 = next((r for r in all_rows if r["label"] == "chunk_800_reranked"), None)
    if base_800 and rerank_800:
        lines.append(
            f"- **Reranking improvement**: Recall@5 from "
            f"{base_800['recall_at_5']:.1%} to {rerank_800['recall_at_5']:.1%} on chunk_800\n"
        )

    lines.append(
        f"- **Refusal rate**: {refusal_metrics['refusal_rate']:.1%} "
        f"({refusal_metrics['correct_refusals']}/{refusal_metrics['total_unanswerable']} "
        f"unanswerable questions correctly flagged as low-confidence)\n"
    )

    # Pick one example refused question
    refused = [d for d in refusal_metrics["details"] if d["would_refuse"]]
    if refused:
        example = refused[0]
        lines.append(
            f"- **Example correct refusal**: \"{example['question']}\" "
            f"(top-1 score: {example['top_score']:.4f})\n"
        )

    path.write_text("".join(lines), encoding="utf-8")
    print(f"\nResults written to: {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    # Load eval set
    eval_set = json.loads(Path(EVAL_SET_PATH).read_text(encoding="utf-8"))
    answerable = [q for q in eval_set if q.get("is_answerable", True)]
    unanswerable = [q for q in eval_set if not q.get("is_answerable", True)]
    print(f"Loaded eval set: {len(eval_set)} questions "
          f"({len(answerable)} answerable, {len(unanswerable)} unanswerable)")

    # ── Step 1: Run baseline experiments (no reranker) ──
    print("\n\n" + "#" * 60)
    print("# STEP 1: BASELINE EXPERIMENTS (no reranker)")
    print("#" * 60)

    baseline_configs = [
        ("chunk_500", 500, 50),
        ("chunk_800", 800, 100),
        ("chunk_1200", 1200, 150),
    ]
    baseline_rows = []
    for label, cs, co in baseline_configs:
        result = run_single_experiment(label, cs, co, eval_set, use_reranker=False)
        baseline_rows.append(result)
        print(f"\n>>> {label}: Recall@5={result['recall_at_5']:.1%}")

    # ── Step 2: Reranker experiments on chunk_800 and chunk_1200 ──
    print("\n\n" + "#" * 60)
    print("# STEP 2: RERANKER EXPERIMENTS (chunk_800 + chunk_1200)")
    print("#" * 60)

    reranker_configs = [
        ("chunk_800_reranked", 800, 100),
        ("chunk_1200_reranked", 1200, 150),
    ]
    reranker_rows = []
    for label, cs, co in reranker_configs:
        result = run_single_experiment(label, cs, co, eval_set, use_reranker=True)
        reranker_rows.append(result)
        print(f"\n>>> {label}: Recall@5={result['recall_at_5']:.1%}")

    # ── Step 3: Refusal rate on unanswerable questions ──
    # Use the best overall config's raw results
    print("\n\n" + "#" * 60)
    print("# STEP 3: REFUSAL RATE (unanswerable questions)")
    print("#" * 60)

    all_results_rows = baseline_rows + reranker_rows
    best_row = max(all_results_rows, key=lambda x: x["recall_at_5"])
    print(f"\nUsing best config: {best_row['label']}")

    refusal_metrics = compute_refusal_metrics(best_row["raw_results"])
    print(f"\nRefusal Rate: {refusal_metrics['refusal_rate']:.1%} "
          f"({refusal_metrics['correct_refusals']}/{refusal_metrics['total_unanswerable']})")
    print("\nPer-question detail:")
    for d in refusal_metrics["details"]:
        status = "REFUSE" if d["would_refuse"] else "PASS-THROUGH"
        print(f"  {d['id']}: {status} (top-1 score={d['top_score']:.4f}, doc={d['top_doc']})")
        print(f"    Q: {d['question']}")

    # ── Step 4: Category breakdown ──
    print("\n\n" + "#" * 60)
    print("# STEP 4: RECALL@5 BY QUESTION CATEGORY")
    print("#" * 60)

    category_breakdown = compute_category_breakdown(best_row["raw_results"])
    print(f"\n{'Question Type':<20} {'Count':<8} {'Hits':<8} {'Recall@5':<10}")
    print("-" * 50)
    for qtype, data in sorted(category_breakdown.items()):
        print(f"{qtype:<20} {data['count']:<8} {data['hits']:<8} {data['recall_at_5']:<10.1%}")

    # ── Step 5: Write final results.md ──
    print("\n\n" + "#" * 60)
    print("# STEP 5: WRITING FINAL RESULTS")
    print("#" * 60)

    # Strip raw_results before passing (not needed for markdown)
    clean_baseline = [{k: v for k, v in r.items() if k != "raw_results"} for r in baseline_rows]
    clean_reranker = [{k: v for k, v in r.items() if k != "raw_results"} for r in reranker_rows]

    write_final_results(clean_baseline, clean_reranker, refusal_metrics, category_breakdown, RESULTS_PATH)

    # Final summary
    print("\n" + "=" * 60)
    print("ALL PHASE 2 EXPERIMENTS COMPLETE")
    print("=" * 60)
    print(f"{'Config':<25} {'Chunks':<8} {'Recall@5':<10} {'Hit Rate':<10}")
    print("-" * 60)
    for r in all_results_rows:
        print(f"{r['label']:<25} {r['num_chunks']:<8} {r['recall_at_5']:<10.1%} {r['hit_rate']:<10.1%}")


if __name__ == "__main__":
    main()
