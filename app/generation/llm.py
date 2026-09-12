"""
ClauseIQ — Gemini LLM Wrapper
Handles communication with Google Gemini API for answer generation.

Uses Gemini Flash (gemini-3.6-flash) — fast, high context, grounded Q&A.
"""

import os
from google import genai
from google.genai import types
from app.generation.prompt import build_answer_prompt


class GeminiLLM:
    """
    Thin wrapper around Google Gemini API for RAG answer generation.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-flash-latest",
    ):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Set it in your .env file or environment."
            )
        self.model = model
        self._client = genai.Client(api_key=self.api_key)
        print(f"Gemini LLM initialized: {self.model}")

    def generate(
        self,
        question: str,
        chunks: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 2048,
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

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        used_model = self.model
        try:
            response = self._client.models.generate_content(
                model=used_model,
                contents=prompt,
                config=config,
            )
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                fallback_model = "gemini-flash-lite-latest" if used_model != "gemini-flash-lite-latest" else "gemini-flash-latest"
                print(f"  [Notice: 503 on {used_model}, falling back to {fallback_model}]")
                used_model = fallback_model
                response = self._client.models.generate_content(
                    model=used_model,
                    contents=prompt,
                    config=config,
                )
            else:
                raise e

        answer = response.text.strip() if response.text else ""

        # Extract token counts from usage metadata
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0

        return {
            "answer": answer,
            "model": self.model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }


