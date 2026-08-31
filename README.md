# ClauseIQ — Insurance Policy RAG System

An enterprise-grade Retrieval-Augmented Generation (RAG) system for insurance policy Q&A. Upload insurance policy PDFs, ask natural language questions, and get grounded, cited answers backed by real document excerpts — not hallucinated responses.

Built to be fully defensible in a technical interview: every design decision has a measurable outcome, and every experiment produces a number.

---

## Problem Statement

Insurance organizations maintain large volumes of policy, claims, and exclusion documentation, making manual information retrieval slow and error-prone. Customers and support staff need fast, accurate answers with traceable sources — not a black-box chatbot that might invent policy details.

## Solution

ClauseIQ retrieves the most relevant policy clauses for a given question, reranks them for precision, and generates an answer strictly grounded in that retrieved context — with document and page citations for every fact. If the answer isn't in the documents, the system says so instead of guessing.

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
Prompt (strict grounding + citation rules) → Mistral LLM
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
| LLM | Mistral API (mistral-small) |
| Containerization | Docker + Docker Compose |

---

## RAG Pipeline

1. **Ingestion** — PyMuPDF extracts text per page, preserving document name, page number, and doc type (auto-detected from filename prefix: `policy_`, `endorsement_`, `claim_`, `exclusion_`)
2. **Cleaning** — strips headers, footers, page numbers, and normalizes whitespace
3. **Chunking** — token-based splitting (tiktoken, model-aligned) with configurable size and overlap
4. **Embedding** — each chunk converted to a 384-dim vector via Sentence Transformers
5. **Retrieval** — FAISS cosine similarity search returns top-20 candidates
6. **Reranking** — cross-encoder re-scores the 20 candidates for precision, returns top-5
7. **Generation** — top-5 chunks + strict grounding prompt sent to Mistral; answer includes per-fact citations

---

## Evaluation

A 50-question evaluation set (40 answerable, 10 deliberately unanswerable) was built against a synthetic corpus of 12 insurance documents (home/motor/health policies, endorsements, claims guidelines, exclusions) before any pipeline tuning began — every experiment below is measured against this fixed set.

**Metrics tracked:** Recall@k, Hit Rate, Refusal Rate (hallucination control)

| Metric | Result |
|---|---|
| Recall@5 (baseline, 800-token chunks, no reranker) | _fill in_ |
| Recall@5 (with reranker) | _fill in_ |
| Hallucination rate (before prompt tuning) | _fill in_ |
| Hallucination rate (after prompt tuning) | _fill in_ |

## Experiments

Chunk size and reranker impact were measured against the fixed eval set:

| Configuration | Chunk Size | Reranker | Recall@5 | Hit Rate |
|---|---|---|---|---|
| baseline_500 | 500 | No | _fill in_ | _fill in_ |
| baseline_800 | 800 | No | _fill in_ | _fill in_ |
| baseline_1200 | 1200 | No | _fill in_ | _fill in_ |
| best_no_rerank | 800 | No | _fill in_ | _fill in_ |
| best_with_rerank | 800 | Yes | _fill in_ | _fill in_ |

---

## Limitations

- Evaluated against a synthetic document set rather than real-world insurance filings; retrieval quality on messier, non-uniform real PDFs (scanned pages, complex tables) is untested
- No hybrid (keyword + semantic) search — pure vector retrieval may miss exact-term matches on jargon or policy codes
- No multi-turn conversation memory — each question is treated independently
- FAISS `IndexFlatIP` is exact search (O(n)); fine at this corpus size, would need `IndexIVFFlat` or a managed vector DB (Qdrant) at millions of vectors
- No authentication or document-level access control

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
cp .env.example .env   # add your Mistral API key
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
