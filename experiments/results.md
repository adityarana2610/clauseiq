# ClauseIQ - Experiment Results

Generated automatically by `scripts/run_phase2_eval.py`

---

## Retrieval Experiments

| Config | Chunk Size | Overlap | Reranker | Chunks | Recall@5 | Hit Rate | Time |
|---|---|---|---|---|---|---|---|
| chunk_500 | 500 | 50 | No | 67 | 100.0% | 100.0% | 8.2s |
| chunk_800 | 800 | 100 | No | 55 | 97.5% | 97.5% | 5.4s |
| chunk_1200 | 1200 | 150 | No | 55 | 97.5% | 97.5% | 5.1s |
| chunk_800_reranked | 800 | 100 | Yes | 55 | 97.5% | 97.5% | 69.5s |
| chunk_1200_reranked | 1200 | 150 | Yes | 55 | 97.5% | 97.5% | 69.9s |

---

## Hallucination / Refusal Rate Experiments

Measured at the retrieval level: if the top-1 chunk's cosine similarity score
is below 0.40, the system has low confidence and would effectively refuse to answer.
This is a retrieval-level proxy; true LLM-based refusal will be tested in Phase 3.

| Config | Unanswerable Qs | Correct Refusals | Refusal Rate |
|---|---|---|---|
| best_config | 10 | 4 | 40.0% |

### Per-Question Detail

| Question ID | Question | Top-1 Score | Top-1 Doc | Would Refuse |
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

| Question Type | Count | Hits | Recall@5 |
|---|---|---|---|
| cross_document | 5 | 5 | 100.0% |
| rephrased | 16 | 16 | 100.0% |
| simple_lookup | 19 | 19 | 100.0% |
| **Overall** | **40** | **40** | **100.0%** |

---

## Key Findings

- **Best chunk size**: 500 tokens (with 50-token overlap) -- perfect 100.0% Recall@5 across all 40 answerable questions
- **Reranking did not help on chunk_800/1200**: Recall stayed at 97.5% even with the cross-encoder reranker. The single miss (Q005: "cancellation notice period") is a **recall problem**, not a ranking problem -- the correct chunk (page 3) simply isn't retrieved in the top-20 candidates, so reranking can't fix it. Smaller chunks (500) solve this by giving page 3 its own chunk.
- **Retrieval-level refusal rate: 40%** (4/10 unanswerable questions flagged as low-confidence). This is expected to be low -- vector search always finds *something* similar, even for irrelevant questions. True refusal relies on the LLM's prompt instructions (Phase 3). Questions like "Will the policy cover a future pandemic?" score 0.50 because the exclusion docs discuss related topics.
- **Category breakdown confirms scores are legitimate**: simple_lookup (19/19), rephrased (16/16), and cross_document (5/5) all hit 100% on chunk_500. The high recall is not inflated by easy questions masking failures on harder ones.
- **Example correct refusal**: "What is the CEO's name of the insurance company?" (top-1 score: 0.3982 -- system correctly has low confidence)
