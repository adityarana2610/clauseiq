# Enterprise Insurance Knowledge Assistant — Roadmap

**Goal:** A RAG system you can defend line-by-line in an interview, with real measured numbers — not a feature checklist.

**Core principle:** Build the eval set early. Every phase after that gets judged against it, not vibes.

---

## Stack

| Component | Choice |
|---|---|
| Language | Python |
| Backend | FastAPI |
| UI | Streamlit |
| PDF parsing | PyMuPDF |
| Embeddings | Sentence Transformers (start local, swap to API model later if time) |
| Vector DB | FAISS → Qdrant (only if time permits) |
| LLM | OpenAI API (or any capable API model) |
| Reranker | Cross-encoder (e.g. `ms-marco-MiniLM`) |
| Containerization | Docker |

---

## Week 1 — Foundation + Eval Set First

| Day | Task |
|---|---|
| 1 | RAG concepts: embeddings, vector search, cosine similarity, chunking, top-k, hallucination. Checkpoint: explain "why not just paste the PDF into GPT?" |
| 2 | Collect 10–15 insurance docs (real or synthetic — synthetic is fine, you control ground truth) |
| 3 | **Build the 50-question eval set now**: `question, expected_answer, expected_document, expected_page`. Include ~10 unanswerable questions. |
| 4 | PDF ingestion with PyMuPDF → store `{text, page, document, document_type}` |
| 5 | Text cleaning (whitespace, headers/footers, empty pages) |
| 6–7 | Chunking v1 (800 tokens / 100 overlap) → embeddings → FAISS index |

**Why eval set first:** every phase from here has a number attached to it instead of "seems fine."

---

## Week 2 — Retrieval Quality

| Day | Task |
|---|---|
| 8 | Run your 50 questions through retrieval only (no LLM yet). Measure Recall@k, hit rate. |
| 9–10 | Run 2 chunking experiments (e.g. 500 vs 800 vs 1200 tokens) against the eval set. Log results in a table. |
| 11 | Add basic LLM generation: retrieved chunks → prompt → answer. Enforce "don't answer outside context." |
| 12 | Add citations (document + page per chunk) |
| 13–14 | Add reranker (top 20 → rerank → top 5). Re-run eval set, compare with/without reranker. |

**Milestone check:** you should have a table like:
| Config | Retrieval Recall | Answer Accuracy |
|---|---|---|
| 500-tok, no rerank | X% | Y% |
| 800-tok, no rerank | X% | Y% |
| 800-tok + rerank | X% | Y% |

---

## Week 3 — Hallucination + Application Layer

| Day | Task |
|---|---|
| 15 | Run the 10 unanswerable questions through the system. Measure hallucination rate. Tune prompt until refusal rate is high. |
| 16 | Metadata filtering (policy type, document type) — lets you show "structured filter + semantic search" |
| 17–18 | FastAPI backend: `/upload`, `/ask`, `/documents`, `/health`. Move RAG logic out of notebook into `app/` structure. |
| 19–20 | Streamlit UI: upload, ask, show answer + sources |
| 21 | Docker: containerize FastAPI + Streamlit + vector store |

---

## Week 4 — Polish + Documentation

| Day | Task |
|---|---|
| 22 | Query rewriting for follow-up questions (conversation history → standalone query) — *stretch, do if on schedule* |
| 23–24 | Clean up experiment tables, finalize eval numbers, write up hallucination results |
| 25 | Deploy (any cloud/container host) — doesn't need to be fancy, just live |
| 26–28 | README: problem statement, architecture diagram, tech stack, pipeline explanation, real eval results, experiments table, limitations, future work |

---

## What to put under "Future Work" in your README (things you can say you're actively extending)

- Hybrid search (BM25 + vector)
- Query rewriting / multi-turn conversation memory
- Document-level access control / auth
- OCR for scanned documents
- Multimodal RAG (tables, images in PDFs)

This is legitimate to mention as in-progress — it shows scope awareness without inflating what's actually built and tested.

---

## Non-negotiables for the interview story

By the end you must be able to say, with numbers:
1. "I tested chunk sizes X, Y, Z — Y performed best because \_\_\_"
2. "Reranking improved accuracy from X% to Y%"
3. "My hallucination rate was X% before prompt tuning, Y% after"
4. "Here's a question my system correctly refused to answer"

These four sentences are worth more than any additional feature.
