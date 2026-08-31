# ClauseIQ — Experiment Results

Generated automatically by `scripts/run_experiments.py`

---

## Retrieval Experiments

| Config | Chunk Size | Overlap | Reranker | Chunks | Recall@5 | Hit Rate | Time |
|---|---|---|---|---|---|---|---|
| chunk_500 | 500 | 50 | No | 67 | 100.0% | 100.0% | 8.5s |
| chunk_800 | 800 | 100 | No | 55 | 97.5% | 97.5% | 6.0s |
| chunk_1200 | 1200 | 150 | No | 55 | 97.5% | 97.5% | 5.3s |
| chunk_500_reranked | 500 | 50 | Yes | 67 | 100.0% | 100.0% | 84.5s |

---

## Key Findings

- **Best chunk size**: 500 tokens
- **Best Recall@5**: 100.0%
- **Best Hit Rate**: 100.0%
- **Reranking improvement**: Recall@5 from 100.0% → 100.0%
