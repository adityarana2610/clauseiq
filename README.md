# ClauseIQ — Insurance Policy RAG System

An enterprise-grade Retrieval-Augmented Generation (RAG) system for insurance policy Q&A. Upload insurance policy PDFs, ask natural language questions, and get grounded, cited answers backed by real document excerpts — not hallucinated responses.

Built to be fully defensible in a technical interview: every design decision has a measurable outcome, and every experiment produces a number.

**Status:** Phase 1 (core pipeline), Phase 2 (data + retrieval evaluation), and Phase 3 (LLM generation, citations & hallucination guardrails) complete.

---

## Problem Statement

Insurance organizations maintain large volumes of policy, claims, and exclusion documentation, making manual information retrieval slow and error-prone. Customers and support staff need fast, accurate answers with traceable sources — not a black-box chatbot that might invent policy details.

## Solution

ClauseIQ retrieves the most relevant policy clauses for a given question, reranks them for precision, and generates an answer strictly grounded in that retrieved context — with document and page citations for every fact. If the answer isn't in the documents, the system is designed to say so instead of guessing.

---

## Architecture

```
PDF Files
   │
   ▼
PyMuPDF Parser → Cleaner → Token-Based Chunker
   │
   ▼
Sentence Transformer Embeddings (all-MiniLM-L6-v2)
   │
   ▼
FAISS Vector Store (IndexFlatIP)
   │
   ▼
User Query
   │
   ▼
Query Embedding → FAISS Top-20 Retrieval → Cross-Encoder Reranker → Top-5
   │
   ▼
Prompt (strict grounding + citation rules) → Google Gemini Flash LLM
   │
   ▼
Answer + Source Citations (document + page)
```

Two-stage retrieval: a fast bi-encoder (FAISS) narrows the corpus to 20 candidates, then a cross-encoder reranker re-scores those 20 for precision before the top 5 go to the LLM.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.13 |
| Backend | FastAPI |
| UI | Streamlit |
| PDF Parsing | PyMuPDF |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2, 384-dim) |
| Vector DB | FAISS (IndexFlatIP) |
| Reranker | Cross-encoder (ms-marco-MiniLM) |
| LLM | Google Gemini API (`gemini-flash-latest` with `gemini-flash-lite` resilience) |
| Containerization | Docker + Docker Compose |

---

## RAG Pipeline

1. **Ingestion** — PyMuPDF extracts text per page, preserving document name, page number, and doc type (auto-detected from filename prefix: `policy_`, `endorsement_`, `claim_`, `exclusion_`)
2. **Cleaning** — strips headers, footers, page numbers, and normalizes whitespace
3. **Chunking** — token-based splitting (tiktoken, model-aligned) with configurable size and overlap
4. **Embedding** — each chunk converted to a 384-dim vector via Sentence Transformers
5. **Retrieval** — FAISS cosine similarity search returns top-20 candidates
6. **Reranking** — cross-encoder re-scores the 20 candidates for precision, returns top-5
7. **Generation** — top-5 chunks + strict grounding prompt sent to Google Gemini Flash; answer includes per-fact citations

---

## Evaluation

A 51-question evaluation set (41 answerable, 10 deliberately unanswerable) was built against a synthetic corpus of 12 insurance documents (home/motor/health policies, endorsements, claims guidelines, exclusions) **before** any pipeline tuning began — every experiment below is measured against this fixed set.

### Retrieval Experiments

| Config | Chunk Size | Overlap | Reranker | Chunks | Recall@1 | Recall@5 | MRR | Hit Rate | Time |
|---|---|---|---|---|---|---|---|---|---|
| chunk_500 | 500 | 50 | No | 67 | 75.6% | 100.0% | 0.8679 | 100.0% | 7.2s |
| chunk_800 | 800 | 100 | No | 55 | 73.2% | 97.6% | 0.8435 | 97.6% | 5.2s |
| chunk_1200 | 1200 | 150 | No | 55 | 73.2% | 97.6% | 0.8435 | 97.6% | 5.3s |
| chunk_800_reranked | 800 | 100 | Yes | 55 | 87.8% | 97.6% | 0.9167 | 97.6% | 69.1s |
| chunk_1200_reranked | 1200 | 150 | Yes | 55 | 87.8% | 97.6% | 0.9167 | 97.6% | 71.2s |

**No single config dominates — it's a trade-off:**
- `chunk_500` maximizes **Recall@5** (100%) — best for retrieval completeness
- `chunk_800_reranked` maximizes **Recall@1 / MRR** (87.8%, 0.9167) — best for top-result precision, at ~13x the latency (69s vs 5s)

**Why Recall@5 stayed flat after reranking (97.6% → 97.6%):** the single retrieval miss never appeared in the initial top-20 candidate pool from Stage 1. A reranker can only reorder what the retriever surfaces — it can't rescue a document the retriever missed entirely. Chunk size tuning (`chunk_500`), not reranking, solved that Stage-1 bottleneck.

**Latency trade-off:** reranking adds roughly 13x latency for a modest top-1 precision gain — worth it in a workflow where precision matters more than response time, not worth it for a real-time chat UI.

### Recall@5 by Question Category

| Question Type | Count | Hits | Recall@5 |
|---|---|---|---|
| cross_document | 5 | 5 | 100.0% |
| rephrased | 16 | 16 | 100.0% |
| simple_lookup | 20 | 20 | 100.0% |
| **Overall** | **41** | **41** | **100.0%** |

*Note: cross_document (comparison) questions are only n=5 — directionally strong, but too small a sample to generalize confidently without further testing.*

### Hallucination / Refusal Rate (Phase 3 Validated)

| Metric | Target | Result | Status |
|---|---|---|---|
| **True LLM Refusal Rate** | 9 Genuinely Unanswerable Questions | **100.0% (9/9)** | **Validated** |
| Exploratory Retrieval Proxy | Cosine Similarity < 0.40 | 40.0% (4/10) | Phase 2 Baseline |

**Key Interview Insight:**
- In Phase 2, a cosine similarity confidence threshold (< 0.40) only caught 40% of unanswerable questions because bi-encoders frequently pull tangentially related policy text.
- In Phase 3, evaluating through the full end-to-end `pipeline.ask()` with Gemini yielded a **100% (9/9)** true refusal rate (*"I cannot find this information in the provided documents"*).
- The original eval set had 10 unanswerable questions. One (`UNANSWERABLE_003`, pandemic coverage) was discovered to be mislabeled — the corpus document `exclusion_high_risk_002` explicitly contains the relevant exclusion clause. The model correctly answered it with a grounded citation. This was a **dataset labeling error, not a model failure**; the question has been reclassified as answerable.

---

## Limitations

- Evaluated against a synthetic document set rather than real-world insurance filings; retrieval quality on messier, non-uniform real PDFs (scanned pages, complex tables) is untested
- No hybrid (keyword + semantic) search — pure vector retrieval may miss exact-term matches on jargon or policy codes
- No multi-turn conversation memory — each question is treated independently
- FAISS `IndexFlatIP` is exact search (O(n)); fine at this corpus size, would need `IndexIVFFlat` or a managed vector DB (Qdrant) at millions of vectors
- No authentication or document-level access control
- Cross-document comparison questions tested on a small sample (n=5)
- **Gemini API resilience**: `gemini-flash-latest` intermittently returns 503 (server overload). The pipeline includes an automatic fallback to `gemini-flash-lite-latest`, which may produce slightly shorter or less nuanced answers than the primary model

## Future Work

- Hybrid search (BM25 + vector) for better handling of domain-specific terminology
- Query rewriting for multi-turn, conversational follow-up questions
- Document-level access control / authentication
- OCR support for scanned policy documents
- Multimodal RAG for tables and embedded images within PDFs

---

## Setup

```bash
git clone https://github.com/adityarana2610/clauseiq.git
cd clauseiq
pip install -r requirements.txt
cp .env.example .env   # add your GOOGLE_API_KEY
```

**Ingest documents:**
```python
from app.pipeline import RAGPipeline

pipeline = RAGPipeline(chunk_size=800, use_reranker=True)
pipeline.ingest('data/raw_pdfs')
pipeline.save('data/faiss_index')
```

**Ask a question:**
```python
pipeline = RAGPipeline.load('data/faiss_index')
result = pipeline.ask('What is the flood damage deductible?')
```

**Run the API:**
```bash
uvicorn app.api.main:app --reload --port 8000
# docs at http://localhost:8000/docs
```

**Run the UI:**
```bash
streamlit run ui/streamlit_app.py
```

**Run with Docker:**
```bash
docker-compose -f docker/docker-compose.yml up --build
```

**Run evaluation:**
```bash
python -m app.evaluation.eval_runner \
  --index_path data/faiss_index \
  --eval_set data/eval_set.json \
  --k 5 --run_llm --label best_with_rerank
```

---

## Project Structure

```
clauseiq/
├── app/
│   ├── ingestion/       # PDF parsing, cleaning, chunking
│   ├── retrieval/       # Embeddings, FAISS store, reranker
│   ├── generation/      # Prompts, LLM wrapper
│   ├── evaluation/      # Metrics, eval runner
│   ├── api/             # FastAPI endpoints
│   └── pipeline.py      # End-to-end orchestrator
├── ui/                  # Streamlit app
├── data/                # Raw PDFs, eval set
├── experiments/         # Results tables
├── docker/
└── requirements.txt
```
