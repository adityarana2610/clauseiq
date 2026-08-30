"""
ClauseIQ — FastAPI Backend
Endpoints:
  GET  /health       → system health check
  GET  /documents    → list indexed documents
  POST /upload       → ingest a PDF
  POST /ask          → ask a question

Run with:
    uvicorn app.api.main:app --reload --port 8000
"""

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.pipeline import RAGPipeline


# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ClauseIQ API",
    description="Insurance Policy RAG System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (lazy-loaded)
_pipeline: RAGPipeline | None = None
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        # Try loading existing index first
        if Path(FAISS_INDEX_PATH + ".faiss").exists():
            _pipeline = RAGPipeline.load(FAISS_INDEX_PATH)
        else:
            # Fresh pipeline (no documents yet)
            _pipeline = RAGPipeline()
    return _pipeline


# ─── Schemas ──────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    filter_doc_type: str | None = None
    filter_document: str | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]
    model: str
    num_chunks_used: int


class HealthResponse(BaseModel):
    status: str
    num_documents: int
    num_chunks: int


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    """System health check — returns document and chunk counts."""
    pipeline = get_pipeline()
    return {
        "status": "ok",
        "num_documents": pipeline.num_documents,
        "num_chunks": pipeline.num_chunks,
    }


@app.get("/documents")
def list_documents():
    """List all indexed documents with metadata."""
    pipeline = get_pipeline()
    # Summarize by document
    doc_summary = {}
    for chunk in pipeline.vector_store.chunks:
        doc = chunk["document"]
        if doc not in doc_summary:
            doc_summary[doc] = {
                "document": doc,
                "doc_type": chunk["doc_type"],
                "pages": set(),
                "num_chunks": 0,
            }
        doc_summary[doc]["pages"].add(chunk["page"])
        doc_summary[doc]["num_chunks"] += 1

    # Convert sets to sorted lists for JSON serialization
    result = []
    for doc_info in doc_summary.values():
        result.append(
            {
                **doc_info,
                "pages": sorted(doc_info["pages"]),
                "num_pages": len(doc_info["pages"]),
            }
        )

    return {"documents": result, "total": len(result)}


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    doc_type: str = Form(default="unknown"),
):
    """
    Upload and ingest a PDF into the vector store.
    The file is parsed, chunked, embedded, and added to FAISS.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pipeline = get_pipeline()

    # Save uploaded file to a temp directory and ingest
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / file.filename
        pdf_path.write_bytes(await file.read())

        summary = pipeline.ingest(pdf_dir=tmpdir)
        pipeline.save(FAISS_INDEX_PATH)

    return {
        "message": f"Successfully ingested {file.filename}",
        "summary": summary,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Ask a question about the indexed insurance documents.
    Returns an answer with citations (document + page).
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    pipeline = get_pipeline()

    if pipeline.num_chunks == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents have been indexed yet. Upload PDFs via /upload first.",
        )

    result = pipeline.ask(
        question=request.question,
        filter_doc_type=request.filter_doc_type,
        filter_document=request.filter_document,
    )
    return result
