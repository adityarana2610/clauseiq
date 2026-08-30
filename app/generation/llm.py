"""
ClauseIQ — Mistral LLM Wrapper
Handles communication with Mistral API for answer generation.
"""

import os
from mistralai import Mistral
from app.generation.prompt import build_answer_prompt


class MistralLLM:
    """
    Thin wrapper around Mistral API for RAG answer generation.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "mistral-small-latest",
    ):
        self.api_key = api_key or os.environ["MISTRAL_API_KEY"]
        self.model = model
        self.client = Mistral(api_key=self.api_key)
        print(f"Mistral LLM initialized: {self.model}")

    def generate(
        self,
        question: str,
        chunks: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> dict:
        """
        Generate an answer for a question given retrieved chunks.

        Args:
            question:    The user's question
            chunks:      Retrieved + reranked chunks
            temperature: Low temp (0.1) for factual answers; higher for creative
            max_tokens:  Max tokens in the response

        Returns:
            dict with keys:
                - answer: str
                - model: str
                - prompt_tokens: int
                - completion_tokens: int
        """
        prompt = build_answer_prompt(question=question, chunks=chunks)

        response = self.client.chat.complete(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        answer = response.choices[0].message.content.strip()
        usage = response.usage

        return {
            "answer": answer,
            "model": self.model,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
        }
