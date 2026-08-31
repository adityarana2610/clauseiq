"""
ClauseIQ — PDF Parser
Uses PyMuPDF to extract text and metadata from insurance policy PDFs.
Preserves page numbers and document-level metadata for citation.
"""

from pathlib import Path
from typing import Optional
import fitz  # PyMuPDF


def extract_pages(pdf_path: str | Path) -> list[dict]:
    """
    Extract all pages from a PDF as a list of page dicts.

    Returns:
        List of dicts with keys:
            - text: str        (raw extracted text)
            - page: int        (1-indexed page number)
            - document: str    (filename without extension)
            - document_path: str (absolute path)
            - doc_type: str    ("policy" | "endorsement" | "claim" | "unknown")
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc_type = _infer_doc_type(pdf_path.name)
    pages = []

    with fitz.open(str(pdf_path)) as pdf:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text("text")

            pages.append(
                {
                    "text": text,
                    "page": page_num + 1,  # 1-indexed
                    "document": pdf_path.stem,
                    "document_path": str(pdf_path.resolve()),
                    "doc_type": doc_type,
                }
            )

    return pages


def extract_all_pdfs(pdf_dir: str | Path) -> list[dict]:
    """
    Extract all PDFs from a directory.

    Returns:
        Flat list of page dicts across all documents.
    """
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    all_pages = []
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in: {pdf_dir}")

    print(f"Found {len(pdf_files)} PDF(s) in {pdf_dir}")
    for pdf_path in pdf_files:
        try:
            pages = extract_pages(pdf_path)
            all_pages.extend(pages)
            print(f"  [OK] {pdf_path.name}: {len(pages)} pages extracted")
        except Exception as e:
            print(f"  [FAIL] {pdf_path.name}: Failed — {e}")

    print(f"\nTotal pages extracted: {len(all_pages)}")
    return all_pages


def _infer_doc_type(filename: str) -> str:
    """
    Infer document type from filename conventions.
    Expects files like: policy_home_001.pdf, endorsement_flood.pdf, etc.
    """
    filename_lower = filename.lower()
    if "policy" in filename_lower:
        return "policy"
    elif "endorsement" in filename_lower:
        return "endorsement"
    elif "claim" in filename_lower:
        return "claim"
    elif "exclusion" in filename_lower:
        return "exclusion"
    else:
        return "unknown"
