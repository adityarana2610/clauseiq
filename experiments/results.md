# ClauseIQ — Experiment Results

This file tracks all chunking and retrieval experiments.
Run `python -m app.evaluation.eval_runner` to add rows automatically.

---

## Retrieval Experiments

| Config | Chunk Size | Overlap | Reranker | Recall@5 | Hit Rate | Notes |
|---|---|---|---|---|---|---|
| baseline_500 | 500 | 50 | No | — | — | *Run eval to fill* |
| baseline_800 | 800 | 100 | No | — | — | *Run eval to fill* |
| baseline_1200 | 1200 | 150 | No | — | — | *Run eval to fill* |
| best_no_rerank | 800 | 100 | No | — | — | *Run eval to fill* |
| best_with_rerank | 800 | 100 | Yes | — | — | *Run eval to fill* |

---

## Hallucination / Refusal Experiments

| Prompt Version | Unanswerable Qs | Refusal Rate | Notes |
|---|---|---|---|
| v1_baseline | 10 | — | *Run eval with --run_llm to fill* |
| v2_tuned | 10 | — | *After prompt tuning* |

---

## Key Findings

*(Fill these in after running experiments — these are your interview answers)*

- **Best chunk size**: ___
- **Why**: ___
- **Reranking improvement**: Recall@5 from ___% → ___%
- **Hallucination rate before tuning**: ___%
- **Hallucination rate after tuning**: ___%
- **Example of correct refusal**: 
  > Q: "..." → A: "I cannot find this information in the provided documents."
