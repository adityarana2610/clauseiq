"""
ClauseIQ — Text Chunker
Splits cleaned page text into overlapping token-based chunks.
Preserves source metadata (document, page) on each chunk for citations.

Design notes:
  - Token counting via tiktoken (cl100k_base — same tokenizer as GPT-4 / Mistral approx)
  - Chunks respect sentence boundaries where possible (split on ". ")
  - chunk_size and chunk_overlap are configurable for experiments
"""

import tiktoken

# Default tokenizer — close enough approximation for Mistral
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[dict]:
    """
    Chunk a list of cleaned page dicts into fixed-size token chunks.

    Args:
        pages:         List of page dicts (from cleaner.clean_pages)
        chunk_size:    Max tokens per chunk
        chunk_overlap: Token overlap between consecutive chunks

    Returns:
        List of chunk dicts with keys:
            - text: str
            - tokens: int
            - chunk_id: str          (e.g. "policy_home_001_p3_c0")
            - document: str
            - page: int
            - doc_type: str
    """
    chunks = []

    for page in pages:
        page_chunks = _chunk_text(
            text=page["text"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for idx, (chunk_text, token_count) in enumerate(page_chunks):
            chunk_id = f"{page['document']}_p{page['page']}_c{idx}"
            chunks.append(
                {
                    "text": chunk_text,
                    "tokens": token_count,
                    "chunk_id": chunk_id,
                    "document": page["document"],
                    "page": page["page"],
                    "doc_type": page["doc_type"],
                }
            )

    print(
        f"Chunking complete: {len(chunks)} chunks from {len(pages)} pages "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


# ─── Helpers ──────────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """Return the number of tokens in a string."""
    return len(_TOKENIZER.encode(text))


def _chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, int]]:
    """
    Split a single page's text into overlapping chunks.

    Returns:
        List of (chunk_text, token_count) tuples.
    """
    tokens = _TOKENIZER.encode(text)
    total = len(tokens)

    if total <= chunk_size:
        # Page fits in a single chunk
        return [(text, total)]

    chunks = []
    start = 0

    while start < total:
        end = min(start + chunk_size, total)
        chunk_tokens = tokens[start:end]
        chunk_text = _TOKENIZER.decode(chunk_tokens)
        chunks.append((chunk_text, len(chunk_tokens)))

        if end == total:
            break

        # Advance by (chunk_size - overlap), never go backward
        start = max(start + 1, end - chunk_overlap)

    return chunks
