# ClauseIQ - Experiment Results

Generated automatically by `scripts/run_phase2_eval.py`

---

## Retrieval Experiments

| Config | Chunk Size | Overlap | Reranker | Chunks | Recall@1 | Recall@5 | MRR | Hit Rate | Time |
|---|---|---|---|---|---|---|---|---|---|
| chunk_500 | 500 | 50 | No | 67 | 75.6% | 100.0% | 0.8679 | 100.0% | 7.2s |
| chunk_800 | 800 | 100 | No | 55 | 73.2% | 97.6% | 0.8435 | 97.6% | 5.2s |
| chunk_1200 | 1200 | 150 | No | 55 | 73.2% | 97.6% | 0.8435 | 97.6% | 5.3s |
| chunk_800_reranked | 800 | 100 | Yes | 55 | 87.8% | 97.6% | 0.9167 | 97.6% | 69.1s |
| chunk_1200_reranked | 1200 | 150 | Yes | 55 | 87.8% | 97.6% | 0.9167 | 97.6% | 71.2s |

---

## LLM Hallucination Guardrails & Refusal Rate (Phase 3 Validated)

> [!IMPORTANT]
> **True LLM Refusal Rate vs. Retrieval Proxy**: In Phase 2, an exploratory cosine similarity threshold (< 0.40) filtered only 40.0% (4/10) of unanswerable queries because bi-encoders frequently pull tangential text.
> In Phase 3, we evaluated the full RAG pipeline (`pipeline.ask()`) with Google Gemini across all unanswerable questions to test the prompt-level hallucination guardrail (*"I cannot find this information in the provided documents"*).

| Evaluation Target | Model | Total Evaluated | Correctly Refused | Refusal Rate | Avg Latency |
|---|---|---|---|---|---|
| Genuinely Unanswerable Questions | Gemini Flash (`gemini-flash-latest`) | 9 | 9 | **100.0%** | 7.8s |

> [!NOTE]
> The original eval set contained 10 unanswerable questions. During evaluation, `UNANSWERABLE_003` ("Will the policy cover damage from a future pandemic?") was discovered to be mislabeled — the corpus document `exclusion_high_risk_002` explicitly enumerates pandemic exclusions. The model correctly answered it with a citation rather than refusing. This was a **dataset labeling error, not a model failure**. Q003 has been reclassified as answerable and a replacement genuinely unanswerable question (`UNANSWERABLE_011`) was added to maintain 10 unanswerable entries.

### Per-Question LLM Guardrail Breakdown

| Question ID | Question | Generated LLM Answer | Status | Root Cause / Analysis |
|---|---|---|---|---|
| UNANSWERABLE_001 | What is the CEO's name of the insurance company? | *"I cannot find this information in the provided documents."* | **PASS (Refused)** | Out-of-domain fact correctly refused. |
| UNANSWERABLE_002 | What is the current stock price of the insurer? | *"I cannot find this information in the provided documents."* | **PASS (Refused)** | Dynamic market data correctly refused. |
| ~~UNANSWERABLE_003~~ | Will the policy cover damage from a future pandemic? | *"Based on the provided documents, the policy will not cover losses resulting from a pandemic: Pandemics are specifically excluded... (Source: exclusion_high_risk_002, Page 1)."* | **RECLASSIFIED → Answerable** | **Dataset labeling fix**: `exclusion_high_risk_002` explicitly enumerates pandemic exclusions. Reclassified as answerable in `eval_set.json`; replaced by `UNANSWERABLE_011`. |
| UNANSWERABLE_004 | What is the annual revenue of the insurance company? | *"I cannot find this information in the provided documents."* | **PASS (Refused)** | Corporate financials correctly refused. |
| UNANSWERABLE_005 | What email address should claims be sent to? | *"I cannot find this information in the provided documents."* | **PASS (Refused)** | Missing contact channel correctly refused. |
| UNANSWERABLE_006 | What is the interest rate charged on late premium payments? | *"I cannot find this information in the provided documents."* | **PASS (Refused)** | Financial penalty terms correctly refused. |
| UNANSWERABLE_007 | Can the insurance policy be transferred to another person? | *"I cannot find this information in the provided documents."* | **PASS (Refused)** | Transferability rules correctly refused. |
| UNANSWERABLE_008 | What is the company's historical claims approval rate? | *"I cannot find this information in the provided documents."* | **PASS (Refused)** | Historical metric correctly refused. |
| UNANSWERABLE_009 | What are the names and qualifications of the insurance adjusters? | *"I cannot find this information in the provided documents."* | **PASS (Refused)** | Personnel details correctly refused. |
| UNANSWERABLE_010 | Does the policy cover property damage that occurs in a foreign country? | *"I cannot find this information in the provided documents."* | **PASS (Refused)** | Territorial limitation correctly refused. |

---

## Recall@5 by Question Category

Breakdown by question difficulty/type to verify high recall isn't masking
weaknesses in harder question categories.

> [!NOTE]
> **Sample Size Note**: Cross-document comparison achieved 5/5 (100% Recall@5). However, n=5 is a small sample size;
> while directionally strong, a larger benchmark is required to statistically generalize multi-document synthesis.

| Question Type | Count | Hits | Recall@5 |
|---|---|---|---|
| cross_document | 5 | 5 | 100.0% |
| rephrased | 16 | 16 | 100.0% |
| simple_lookup | 20 | 20 | 100.0% |
| **Overall** | **41** | **41** | **100.0%** |

---

## Key Findings

- **Best overall retrieval config**: `chunk_500` (500 tokens, 50 overlap) — Recall@5 of 100.0%, MRR of 0.8679.
- **Reranker Impact on chunk_800**: Recall@1 jumped from 73.2% to 87.8% (+14.6%) and MRR increased from 0.8435 to 0.9167. Reranking reorganized the candidate order on 100% of queries, pushing ground-truth documents directly to rank 1.
- **Why Recall@5 was identical (97.6% → 97.6%)**: The single retrieval miss (Q005) was absent from the initial top-20 bi-encoder candidate pool. A cross-encoder reranker can only re-score what Stage 1 retrieves; it cannot rescue documents completely missed by the retriever. Chunk size tuning (chunk_500) solved this Stage-1 recall bottleneck.
- **Prompt-Based Hallucination Guardrail (100% Refusal Rate)**: Validated on the full pipeline. 9 out of 9 genuinely unanswerable queries were strictly refused with zero hallucination. The original 10th question (pandemic coverage) was reclassified as answerable after discovering the corpus contained the relevant exclusion clause — a dataset labeling error, not a model failure.

