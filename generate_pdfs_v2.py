"""
generate_pdfs_v2.py
-------------------
Enhanced PDF generation with proper visual formatting:
  - Bold, larger section headers with spacing
  - Paragraph breaks between numbered / bullet items
  - Indented bullet points with bullet character
  - 1.3-1.4x line spacing (leading)
  - 1-inch margins on all sides
  - Centered bold title on first page of each document
"""

import re
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, PageBreak, Spacer, HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.colors import HexColor
from pypdf import PdfReader

# ── Constants ───────────────────────────────────────────────────────────────

SRC_FILE = Path(__file__).parent / "synthetic_insurance_policies_clean.txt"
OUT_DIR = Path(__file__).parent / "raw_pdfs_v2"
DOC_DELIM = re.compile(r"^=== DOCUMENT:\s*(.+?)\s*===$")
PAGE_DELIM = re.compile(r"^---\s*PAGE\s+\d+\s*---$")


# ── Styles ──────────────────────────────────────────────────────────────────

def make_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "DocTitle",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "DocSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            alignment=TA_CENTER,
            spaceAfter=3,
            textColor=HexColor("#555555"),
        ),
        "section": ParagraphStyle(
            "SectionHeader",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            spaceBefore=14,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=15.4,  # ~1.4x body font
            spaceAfter=8,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=15.4,
            leftIndent=28,
            bulletIndent=14,
            spaceAfter=4,
        ),
        "numbered": ParagraphStyle(
            "Numbered",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=15.4,
            leftIndent=20,
            firstLineIndent=-20,
            spaceAfter=6,
        ),
    }


# ── Document Parsing (unchanged from v1) ────────────────────────────────────

def parse_documents(filepath: Path) -> list[tuple[str, list[str]]]:
    """Split source file into (doc_name, [page1_text, page2_text, ...])."""
    raw = filepath.read_text(encoding="utf-8")
    docs: list[tuple[str, list[str]]] = []
    name = None
    pages: list[str] = []
    buf: list[str] = []

    for line in raw.splitlines():
        m = DOC_DELIM.match(line.strip())
        if m:
            if name is not None:
                if buf:
                    pages.append("\n".join(buf).strip())
                docs.append((name, [p for p in pages if p]))
            name, pages, buf = m.group(1).strip(), [], []
            continue
        if PAGE_DELIM.match(line.strip()):
            pages.append("\n".join(buf).strip())
            buf = []
            continue
        buf.append(line)

    if name is not None:
        if buf:
            pages.append("\n".join(buf).strip())
        docs.append((name, [p for p in pages if p]))

    return docs


# ── XML escaping for reportlab Paragraph ────────────────────────────────────

def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ── Text Segmentation ──────────────────────────────────────────────────────

def segment_page(text: str, is_first_page: bool = False) -> list[tuple[str, str]]:
    """
    Parse one page of raw text into structured segments:
      ('title', ...), ('subtitle', ...), ('section', ...),
      ('body', ...), ('bullet', ...), ('numbered', ...)
    """
    segs: list[tuple[str, str]] = []
    rem = text.strip()

    # ── Title & subtitles (first page only) ──────────────────────────────
    if is_first_page:
        # Extract POLICY TYPE / DOCUMENT TYPE as the document title
        m = re.match(
            r"((?:POLICY|DOCUMENT)\s+TYPE:\s*.+?)"
            r"(?=\s+(?:POLICY IDENTIFIER|COVERAGE ADD-ON|APPLICABILITY))",
            rem,
        )
        if m:
            segs.append(("title", m.group(1).strip()))
            rem = rem[m.end() :].strip()

        # Extract subtitle fields (POLICY IDENTIFIER, COVERAGE ADD-ON, etc.)
        for kw in ("POLICY IDENTIFIER", "COVERAGE ADD-ON", "APPLICABILITY"):
            m = re.match(
                rf"({re.escape(kw)}:\s*.+?)"
                rf"(?=\s+(?:SECTION|POLICY IDENTIFIER|COVERAGE ADD-ON|APPLICABILITY))",
                rem,
            )
            if m:
                segs.append(("subtitle", m.group(1).strip()))
                rem = rem[m.end() :].strip()

    # ── All-caps continuation header (split section header from prev page) ─
    if not is_first_page:
        m = re.match(r"^((?:[A-Z][-A-Z/]{1,}\s+){3,})", rem)
        if m:
            segs.append(("section", m.group(1).strip()))
            rem = rem[m.end() :].strip()

    # ── Split on SECTION headers ─────────────────────────────────────────
    rem = re.sub(r"(?<=\S)\s+(SECTION(?:\s+\d+)?:)", r"\n\n\1", rem)

    for block in re.split(r"\n\n+", rem):
        block = block.strip()
        if not block:
            continue

        # Bare "SECTION:" at end of page (header split across pages)
        if block == "SECTION:":
            segs.append(("section", "SECTION:"))
            continue

        # Try matching a section header at the start of the block
        sm = re.match(
            r"(SECTION\s*\d*:\s*(?:[A-Z][-A-Z/]+(?:\s+|$))+)(.*)",
            block,
            re.DOTALL,
        )
        if sm:
            segs.append(("section", sm.group(1).strip()))
            body = sm.group(2).strip()
            if body:
                segs.extend(_parse_body(body))
        else:
            segs.extend(_parse_body(block))

    return segs


def _parse_body(text: str) -> list[tuple[str, str]]:
    """
    Split body text into body / bullet / numbered segments by detecting:
      - Numbered definitions:  1. "Term" ...
      - Numbered conditions:   1. UPPERCASE LABEL: ...
      - Bullet items:          - Capitalized text ...
    """
    segs: list[tuple[str, str]] = []
    t = text

    # Insert line breaks before numbered items (definition-style with quotes)
    t = re.sub(r'(?<=\S)\s+(\d{1,2}\.\s+")', r"\n\1", t)
    # Insert line breaks before numbered items (condition-style with UPPERCASE)
    t = re.sub(r"(?<=\S)\s+(\d{1,2}\.\s+[A-Z]{2})", r"\n\1", t)
    # Insert line breaks before bullet items
    t = re.sub(r"(?<=\S)\s+(- [A-Z])", r"\n\1", t)

    for line in t.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("- ") and len(line) > 2 and line[2].isupper():
            segs.append(("bullet", line[2:]))  # strip the "- " prefix
        elif re.match(r"\d{1,2}\.\s", line):
            segs.append(("numbered", line))
        else:
            segs.append(("body", line))

    return segs


# ── PDF Generation ──────────────────────────────────────────────────────────

def build_pdf(
    doc_name: str, pages: list[str], out_path: Path, styles: dict
) -> None:
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=LETTER,
        leftMargin=1 * inch,
        rightMargin=1 * inch,
        topMargin=1 * inch,
        bottomMargin=1 * inch,
    )
    story: list = []

    for page_idx, page_text in enumerate(pages):
        segments = segment_page(page_text, is_first_page=(page_idx == 0))
        prev_type = None

        for stype, stext in segments:
            safe = _esc(stext)

            # ── Separator line after title / subtitle block ──
            if (
                prev_type in ("title", "subtitle")
                and stype not in ("title", "subtitle")
            ):
                story.append(Spacer(1, 4))
                story.append(
                    HRFlowable(
                        width="100%",
                        thickness=0.5,
                        color=HexColor("#BBBBBB"),
                        spaceAfter=10,
                    )
                )

            # ── Render each segment type ──
            if stype == "title":
                story.append(Paragraph(safe, styles["title"]))

            elif stype == "subtitle":
                story.append(Paragraph(safe, styles["subtitle"]))

            elif stype == "section":
                story.append(Paragraph(safe, styles["section"]))

            elif stype == "bullet":
                story.append(
                    Paragraph(safe, styles["bullet"], bulletText="\u2022")
                )

            elif stype == "numbered":
                story.append(Paragraph(safe, styles["numbered"]))

            else:  # body
                story.append(Paragraph(safe, styles["body"]))

            prev_type = stype

        # Insert page break between pages (not after the last one)
        if page_idx < len(pages) - 1:
            story.append(PageBreak())

    doc.build(story)


# ── Verification ────────────────────────────────────────────────────────────

def verify_pdf(path: Path) -> int:
    return len(PdfReader(str(path)).pages)


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    styles = make_styles()

    print(f"Source:  {SRC_FILE}")
    print(f"Output: {OUT_DIR}\n")

    documents = parse_documents(SRC_FILE)
    print(f"Parsed {len(documents)} documents.\n")

    results: list[tuple[str, int, int]] = []
    mismatches: list[tuple[str, int, int, str | None]] = []

    print(f"{'Document':<45} {'Markers':>8} {'Pages':>8}  Status")
    print("-" * 80)

    for doc_name, pages in documents:
        marker_count = len(pages)
        out_path = OUT_DIR / f"{doc_name}.pdf"

        try:
            build_pdf(doc_name, pages, out_path, styles)
            actual_pages = verify_pdf(out_path)
        except Exception as e:
            print(f"  !! ERROR {doc_name}: {e}")
            mismatches.append((doc_name, marker_count, -1, str(e)))
            continue

        if actual_pages == marker_count:
            status = "OK"
            note = ""
        elif actual_pages > marker_count:
            status = "OK"
            note = f"  (formatting expanded to +{actual_pages - marker_count} pages)"
        else:
            status = "!! MISMATCH"
            note = f"  (fewer pages than markers!)"
            mismatches.append((doc_name, marker_count, actual_pages, None))

        print(f"  {doc_name:<43} {marker_count:>8} {actual_pages:>8}  {status}{note}")
        results.append((doc_name, marker_count, actual_pages))

    # ── Summary ──
    print("\n" + "=" * 80)
    total_pages = sum(r[2] for r in results)
    print(f"TOTAL: {len(results)} PDFs generated  |  {total_pages} total pages  |  Output: {OUT_DIR}")

    if mismatches:
        print(f"\n!!  {len(mismatches)} DOCUMENT(S) WITH PAGE-COUNT ISSUE:")
        for n, e, a, err in mismatches:
            if err:
                print(f"   - {n}: ERROR - {err}")
            else:
                print(f"   - {n}: expected {e}, got {a}")
    else:
        print("\n[OK]  All documents generated successfully!")


if __name__ == "__main__":
    main()
