import os
import time
from dataclasses import dataclass
from typing import Protocol

from app.rag import RankedChunk, synthesize_answer


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    answer_mode: str
    model: str | None
    latency_ms: float
    notes: list[str]


class AnswerGenerator(Protocol):
    def generate(self, question: str, ranked_chunks: list[RankedChunk]) -> GenerationResult:
        pass


class LocalAnswerGenerator:
    def generate(self, question: str, ranked_chunks: list[RankedChunk]) -> GenerationResult:
        start = time.perf_counter()
        answer = synthesize_answer(question, ranked_chunks)
        return GenerationResult(
            answer=answer,
            answer_mode="local",
            model=None,
            latency_ms=(time.perf_counter() - start) * 1000,
            notes=[],
        )


class OpenAIAnswerGenerator:
    def __init__(self, model: str = "gpt-4.1", client: object | None = None) -> None:
        self.model = model
        self._client = client

    def generate(self, question: str, ranked_chunks: list[RankedChunk]) -> GenerationResult:
        start = time.perf_counter()
        response = self.client.responses.create(
            model=self.model,
            input=build_grounded_prompt(question, ranked_chunks),
        )
        return GenerationResult(
            answer=response.output_text,
            answer_mode="openai",
            model=self.model,
            latency_ms=(time.perf_counter() - start) * 1000,
            notes=[],
        )

    @property
    def client(self) -> object:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install OpenAI support with: pip install -e .[openai]") from exc
            self._client = OpenAI()
        return self._client


def select_answer_generator(
    answer_mode: str,
    model: str,
    openai_client: object | None = None,
) -> AnswerGenerator:
    if answer_mode == "local":
        return LocalAnswerGenerator()
    if answer_mode == "openai":
        return OpenAIAnswerGenerator(model=model, client=openai_client)
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIAnswerGenerator(model=model, client=openai_client)
    return LocalAnswerGenerator()


def generate_with_fallback(
    question: str,
    ranked_chunks: list[RankedChunk],
    answer_mode: str,
    model: str,
    openai_client: object | None = None,
) -> GenerationResult:
    generator = select_answer_generator(answer_mode, model, openai_client)
    try:
        return generator.generate(question, ranked_chunks)
    except RuntimeError as exc:
        if answer_mode == "openai":
            raise
        fallback = LocalAnswerGenerator().generate(question, ranked_chunks)
        return GenerationResult(
            answer=fallback.answer,
            answer_mode=fallback.answer_mode,
            model=fallback.model,
            latency_ms=fallback.latency_ms,
            notes=[f"OpenAI generation unavailable; used local fallback: {exc}"],
        )


def build_grounded_prompt(question: str, ranked_chunks: list[RankedChunk]) -> str:
    context = "\n\n".join(
        (
            f"[{item.chunk.document_id}:{item.chunk.chunk_id}]\n"
            f"Title: {item.chunk.title}\n"
            f"Text: {item.chunk.text}"
        )
        for item in ranked_chunks
    )
    if not context:
        context = "No retrieved mortgage research context was found."

    return (
        "You are a mortgage research copilot. Answer only from the provided context. "
        "If the context is insufficient, say so. Include inline citations using the exact "
        "[document_id:chunk_id] markers from the context.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}"
    )
