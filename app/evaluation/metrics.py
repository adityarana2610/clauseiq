"""
ClauseIQ — Evaluation Metrics
Computes retrieval and answer quality metrics against the eval set.

Metrics:
  - Recall@k     : Was the expected document/page in the top-k retrieved chunks?
  - Hit Rate     : Fraction of questions with at least 1 relevant chunk in top-k
  - Refusal Rate : Fraction of unanswerable questions where the model refused to answer
"""

import re


def recall_at_k(
    retrieved_chunks: list[dict],
    expected_document: str,
    expected_page: int,
    k: int = 5,
) -> bool:
    """
    Check if the expected source appears in the top-k retrieved chunks.

    Returns:
        True if the expected (document, page) is in the top-k chunks.
    """
    for chunk in retrieved_chunks[:k]:
        if (
            chunk.get("document") == expected_document
            and chunk.get("page") == expected_page
        ):
            return True
    return False


def compute_retrieval_metrics(results: list[dict], k: int = 5) -> dict:
    """
    Compute retrieval metrics over a list of evaluation results.

    Args:
        results: List of dicts, each with:
            - retrieved_chunks: list[dict]
            - expected_document: str
            - expected_page: int
            - is_answerable: bool  (False for unanswerable questions)
        k: Top-k to evaluate at

    Returns:
        {
            "recall_at_k": float,
            "hit_rate": float,
            "num_questions": int,
            "num_answerable": int,
        }
    """
    answerable = [r for r in results if r.get("is_answerable", True)]
    if not answerable:
        return {"recall_at_k": 0.0, "hit_rate": 0.0, "num_questions": 0, "num_answerable": 0}

    hits = sum(
        1 for r in answerable
        if recall_at_k(r["retrieved_chunks"], r["expected_document"], r["expected_page"], k=k)
    )

    return {
        f"recall_at_{k}": hits / len(answerable),
        "hit_rate": hits / len(answerable),
        "num_questions": len(results),
        "num_answerable": len(answerable),
    }


def compute_refusal_rate(results: list[dict]) -> dict:
    """
    Compute hallucination/refusal rate on unanswerable questions.

    A refusal is detected if the answer contains the phrase
    "cannot find" or "not in the" (matching our prompt template).

    Args:
        results: List of dicts with:
            - answer: str
            - is_answerable: bool

    Returns:
        {
            "refusal_rate": float,   # higher is better (we WANT refusals)
            "num_unanswerable": int,
        }
    """
    _REFUSAL_PHRASES = [
        "cannot find",
        "not in the",
        "not mentioned",
        "no information",
        "don't have",
        "do not have",
        "unable to find",
    ]

    unanswerable = [r for r in results if not r.get("is_answerable", True)]
    if not unanswerable:
        return {"refusal_rate": None, "num_unanswerable": 0}

    refusals = 0
    for r in unanswerable:
        answer_lower = r.get("answer", "").lower()
        if any(phrase in answer_lower for phrase in _REFUSAL_PHRASES):
            refusals += 1

    return {
        "refusal_rate": refusals / len(unanswerable),
        "num_unanswerable": len(unanswerable),
        "num_refused": refusals,
    }
