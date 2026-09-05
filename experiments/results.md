# ClauseIQ - Experiment Results

Generated automatically by `scripts/run_phase2_eval.py`

---

## Retrieval Experiments

| Config | Chunk Size | Overlap | Reranker | Chunks | Recall@1 | Recall@5 | MRR | Hit Rate | Time |
|---|---|---|---|---|---|---|---|---|---|
| chunk_500 | 500 | 50 | No | 67 | 75.0% | 100.0% | 0.8646 | 100.0% | 7.2s |
| chunk_800 | 800 | 100 | No | 55 | 72.5% | 97.5% | 0.8396 | 97.5% | 5.2s |
| chunk_1200 | 1200 | 150 | No | 55 | 72.5% | 97.5% | 0.8396 | 97.5% | 5.3s |
| chunk_800_reranked | 800 | 100 | Yes | 55 | 87.5% | 97.5% | 0.9146 | 97.5% | 69.1s |
| chunk_1200_reranked | 1200 | 150 | Yes | 55 | 87.5% | 97.5% | 0.9146 | 97.5% | 71.2s |

---

## Retrieval-Stage Confidence Threshold (Phase 2 Exploratory Proxy)

> [!WARNING]
> **Not LLM Refusal**: This metric tests a retrieval cosine-similarity threshold (< 0.40 score = low confidence).
> It is **NOT** the prompt-based LLM refusal rate. The true hallucination guardrail test ("I cannot find this information...")
> will be evaluated in Phase 3 by running unanswerable questions through `pipeline.ask()` with Mistral.

| Config | Unanswerable Qs | Sub-Threshold Chunks | Proxy Filter Rate |
|---|---|---|---|
| best_config | 10 | 4 | 40.0% |

### Per-Question Retrieval Confidence Detail

| Question ID | Question | Top-1 Score | Top-1 Doc | Would Flag Low-Conf |
|---|---|---|---|---|
| UNANSWERABLE_001 | What is the CEO's name of the insurance company? | 0.3982 | policy_home_premium_002 | Yes |
| UNANSWERABLE_002 | What is the current stock price of the insurer? | 0.3716 | policy_home_premium_002 | Yes |
| UNANSWERABLE_003 | Will the policy cover damage from a future pandemic? | 0.5001 | exclusion_high_risk_002 | No |
| UNANSWERABLE_004 | What is the annual revenue of the insurance company? | 0.4303 | policy_health_individual_001 | No |
| UNANSWERABLE_005 | What email address should claims be sent to? | 0.3617 | policy_motor_basic_001 | Yes |
| UNANSWERABLE_006 | What is the interest rate charged on late premium payments? | 0.3566 | exclusion_high_risk_002 | Yes |
| UNANSWERABLE_007 | Can the insurance policy be transferred to another person? | 0.5079 | policy_motor_basic_001 | No |
| UNANSWERABLE_008 | What is the company's historical claims approval rate? | 0.4285 | claim_process_timeline_002 | No |
| UNANSWERABLE_009 | What are the names and qualifications of the insurance adjus... | 0.4737 | policy_motor_basic_001 | No |
| UNANSWERABLE_010 | Does the policy cover property damage that occurs in a forei... | 0.4902 | exclusion_general_001 | No |

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
| simple_lookup | 19 | 19 | 100.0% |
| **Overall** | **40** | **40** | **100.0%** |

---

## Key Findings

- **Best overall config**: `chunk_500` (500 tokens, 50 overlap) -- Recall@5 of 100.0%, MRR of 0.8646
- **Reranker Impact on chunk_800**: Recall@1 jumped from 72.5% to 87.5% (+15.0%) and MRR increased from 0.8396 to 0.9146. Reranking reorganized the candidate order on 100% of queries, pushing ground-truth documents directly to rank 1.
- **Why Recall@5 was identical (97.5% -> 97.5%)**: The single retrieval miss (Q005) was absent from the initial top-20 bi-encoder candidate pool. A cross-encoder reranker can only re-score what Stage 1 retrieves; it cannot rescue documents completely missed by the retriever. Chunk size tuning (chunk_500) solved this Stage-1 recall bottleneck.
- **Exploratory Retrieval Confidence Filter**: 40.0% (4/10) of unanswerable queries fell below the 0.40 cosine similarity threshold. Because embedding models often find tangential text, prompt-based LLM guardrails in Phase 3 are required for robust hallucination prevention.
