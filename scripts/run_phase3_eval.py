"""
ClauseIQ — Phase 3 Evaluation Script: LLM Generation & Refusal Benchmarking
Evaluates full pipeline generation with Gemini on the 50-question evaluation set.

Key Focus:
  1. True LLM Refusal Rate on 10 unanswerable questions
     (verifies hallucination guardrails)
  2. Answer generation & citation presence on 40 answerable questions
  3. Token usage and latency statistics
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.pipeline import RAGPipeline
from app.evaluation.metrics import compute_refusal_rate


def run_evaluation(
    mode: str = "unanswerable",
    limit: int | None = None,
    delay: float = 4.0,
    index_path: str = "data/faiss_index",
    eval_set_path: str = "data/eval_set.json",
    output_path: str = "experiments/phase3_results.json",
):
    print("=" * 65)
    print(f" ClauseIQ — Phase 3 Evaluation [Mode: {mode.upper()}]")
    print("=" * 65)

    # Load eval set
    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    if mode == "unanswerable":
        questions = [q for q in eval_set if not q.get("is_answerable", True)]
    elif mode == "answerable":
        questions = [q for q in eval_set if q.get("is_answerable", True)]
    else:  # all
        questions = eval_set

    if limit:
        questions = questions[:limit]

    print(f"Total questions to evaluate: {len(questions)}")
    print(f"Estimated API calls: {len(questions)}")
    print(f"Inter-call delay: {delay}s (to respect free-tier 15 RPM)\n")

    # Load pipeline
    print("Loading RAG pipeline...")
    pipeline = RAGPipeline.load(index_path, use_reranker=True)
    print(f"Pipeline ready with {pipeline.num_chunks} chunks.\n")

    results = []
    total_latency = 0.0

    print("=" * 65)
    for idx, q in enumerate(questions, 1):
        qid = q.get("id", f"Q_{idx}")
        question_text = q["question"]
        is_answerable = q.get("is_answerable", True)

        print(f"[{idx}/{len(questions)}] ({qid}) [Answerable={is_answerable}]")
        print(f"  Q: {question_text}")

        t0 = time.time()
        res = None
        for attempt in range(3):
            try:
                res = pipeline.ask(question_text)
                break
            except Exception as e:
                if attempt < 2:
                    print(f"    [Retry {attempt + 1}/3 after transient error: {e}] - waiting 10s...")
                    time.sleep(10)
                else:
                    print(f"    [Failed all 3 attempts: {e}]")
                    break

        latency = time.time() - t0
        total_latency += latency

        if res is None:
            results.append({
                "id": qid,
                "question": question_text,
                "is_answerable": is_answerable,
                "error": "Failed to generate answer (quota or connection issue)",
            })
            # If we encountered a fatal quota exhaustion, don't waste time on remaining queries
            print(f"  ERROR: Could not get response for {qid}.\n")
            continue

        try:
            answer = res.get("answer", "")
            sources = res.get("sources", [])

            # Check refusal phrases
            is_refusal = any(
                phrase in answer.lower()
                for phrase in [
                    "cannot find",
                    "not in the",
                    "not mentioned",
                    "no information",
                    "don't have",
                    "do not have",
                    "unable to find",
                ]
            )

            # Check citation presence (mentions document or page)
            expected_doc = q.get("expected_document", "")
            has_doc_citation = expected_doc.lower().replace(".pdf", "") in answer.lower() if expected_doc else False

            status_str = "REFUSED" if is_refusal else "ANSWERED"
            print(f"  A: {answer[:120]}..." if len(answer) > 120 else f"  A: {answer}")
            print(f"  -> Status: {status_str} | Latency: {latency:.2f}s\n")

            results.append({
                "id": qid,
                "question": question_text,
                "is_answerable": is_answerable,
                "expected_document": q.get("expected_document"),
                "expected_page": q.get("expected_page"),
                "answer": answer,
                "is_refusal": is_refusal,
                "has_doc_citation": has_doc_citation,
                "sources": sources,
                "latency_seconds": round(latency, 2),
            })

        except Exception as e:
            print(f"  ERROR processing response: {e}\n")
            results.append({
                "id": qid,
                "question": question_text,
                "is_answerable": is_answerable,
                "error": str(e),
            })

        # Save progress incrementally
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, indent=2)

        # Polite rate limiting between API calls
        if idx < len(questions) and delay > 0:
            time.sleep(delay)

    print("=" * 65)
    print(" EVALUATION SUMMARY")
    print("=" * 65)

    unans = [r for r in results if not r.get("is_answerable", True) and "error" not in r]
    ans = [r for r in results if r.get("is_answerable", True) and "error" not in r]

    summary = {
        "mode": mode,
        "total_evaluated": len(results),
        "avg_latency": round(total_latency / max(1, len(results)), 2),
    }

    if unans:
        correct_refusals = sum(1 for r in unans if r.get("is_refusal", False))
        refusal_rate = (correct_refusals / len(unans)) * 100
        summary["unanswerable_total"] = len(unans)
        summary["correct_refusals"] = correct_refusals
        summary["refusal_rate_pct"] = round(refusal_rate, 1)
        print(f"Unanswerable Questions Evaluated : {len(unans)}")
        print(f"Correctly Refused               : {correct_refusals}/{len(unans)}")
        print(f"True LLM Refusal Rate           : {refusal_rate:.1f}%")

    if ans:
        answered_count = sum(1 for r in ans if not r.get("is_refusal", False))
        citations_count = sum(1 for r in ans if r.get("has_doc_citation", False))
        summary["answerable_total"] = len(ans)
        summary["answered_count"] = answered_count
        summary["citations_count"] = citations_count
        print(f"Answerable Questions Evaluated   : {len(ans)}")
        print(f"Answered (Non-refusal)          : {answered_count}/{len(ans)}")
        print(f"Explicit Doc Citations in Text  : {citations_count}/{len(ans)}")

    print(f"Average Latency per Query        : {summary['avg_latency']}s")
    print("=" * 65)

    # Save output
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"Full results saved to: {output_path}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Phase 3 LLM Generation Evaluation")
    parser.add_argument(
        "--mode",
        choices=["unanswerable", "answerable", "all"],
        default="unanswerable",
        help="Which subset of the eval set to test (default: unanswerable - 10 calls)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to evaluate",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=4.0,
        help="Seconds delay between calls (default: 4.0 for 15 RPM free tier)",
    )
    args = parser.parse_args()

    run_evaluation(mode=args.mode, limit=args.limit, delay=args.delay)


if __name__ == "__main__":
    main()
