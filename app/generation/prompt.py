"""
ClauseIQ — Prompt Templates
Keeps all prompts in one place — easy to iterate during tuning.

Interview note: prompt engineering for hallucination control is a first-class
design decision. You can explain:
  - Why we explicitly say "only use context below"
  - Why we tell the model to say "I don't know" vs. making things up
  - How the UNANSWERABLE_THRESHOLD test validates the refusal behavior
"""

SYSTEM_PROMPT = """You are ClauseIQ, an expert assistant for insurance policy documents.

Your job is to answer questions accurately based ONLY on the provided policy excerpts.

Rules you MUST follow:
1. Only use information from the provided context sections below.
2. If the answer is not in the context, say: "I cannot find this information in the provided documents."
3. Do NOT hallucinate, guess, or use general insurance knowledge not present in the context.
4. Always cite your source: mention the document name and page number for each fact you state.
5. Be concise but complete.
"""

ANSWER_PROMPT_TEMPLATE = """{system}

---
CONTEXT:
{context_block}
---

QUESTION: {question}

ANSWER (cite document and page for each fact):"""


def build_context_block(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into the context block for the prompt.
    Each chunk shows its source (document + page) and text.
    """
    lines = []
    for i, chunk in enumerate(chunks, 1):
        doc = chunk.get("document", chunk.get("source_doc", "Policy Document"))
        page = chunk.get("page", "?")
        header = f"[Source {i}: {doc} — Page {page}]"
        lines.append(f"{header}\n{chunk.get('text', '').strip()}")
    return "\n\n".join(lines)



def build_answer_prompt(question: str, chunks: list[dict]) -> str:
    """Build the full prompt to send to LLM."""
    context_block = build_context_block(chunks)
    return ANSWER_PROMPT_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        context_block=context_block,
        question=question,
    )
