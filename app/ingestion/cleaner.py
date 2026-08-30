"""
ClauseIQ — Text Cleaner
Removes noise from raw PDF-extracted text:
  - Repeated whitespace / blank lines
  - Common header/footer patterns (page numbers, document titles repeating)
  - Very short / empty pages
"""

import re


# Minimum character count for a page to be considered non-empty
MIN_PAGE_LENGTH = 50

# Regex patterns for common insurance PDF noise
_HEADER_FOOTER_PATTERNS = [
    r"^\s*Page\s+\d+\s+of\s+\d+\s*$",          # "Page 1 of 10"
    r"^\s*\d+\s*$",                               # standalone page numbers
    r"^\s*Confidential\s*$",                      # "Confidential" stamp
    r"^\s*DRAFT\s*$",                             # draft watermarks
    r"^\s*[-─═]{3,}\s*$",                        # divider lines
]
_HEADER_FOOTER_RE = re.compile(
    "|".join(f"(?:{p})" for p in _HEADER_FOOTER_PATTERNS),
    re.IGNORECASE | re.MULTILINE,
)


def clean_page(page_dict: dict) -> dict:
    """
    Clean text from a single page dict (as returned by pdf_parser).
    Returns the same dict with 'text' replaced by cleaned text.
    """
    text = page_dict["text"]
    text = _remove_headers_footers(text)
    text = _normalize_whitespace(text)
    return {**page_dict, "text": text}


def clean_pages(pages: list[dict]) -> list[dict]:
    """
    Clean and filter a list of page dicts.
    Removes empty/very-short pages after cleaning.
    """
    cleaned = []
    removed = 0

    for page in pages:
        page = clean_page(page)
        if len(page["text"].strip()) >= MIN_PAGE_LENGTH:
            cleaned.append(page)
        else:
            removed += 1

    if removed:
        print(f"Removed {removed} empty/short page(s) after cleaning")

    return cleaned


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _remove_headers_footers(text: str) -> str:
    """Remove lines that match common header/footer patterns."""
    lines = text.splitlines()
    filtered = [
        line for line in lines
        if not _HEADER_FOOTER_RE.fullmatch(line.strip())
    ]
    return "\n".join(filtered)


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple blank lines to a single blank line and strip edges."""
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normalize spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
