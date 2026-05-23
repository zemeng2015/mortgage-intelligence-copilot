from dataclasses import dataclass
import re


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "with",
}


@dataclass(frozen=True)
class Chunk:
    document_id: str
    title: str
    chunk_id: str
    text: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class RankedChunk:
    chunk: Chunk
    score: float


def chunk_text(
    document_id: str,
    title: str,
    text: str,
    metadata: dict[str, str] | None = None,
    chunk_size: int = 1200,
    overlap: int = 120,
) -> list[Chunk]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    for index, start in enumerate(range(0, len(text), step)):
        chunk_body = text[start : start + chunk_size].strip()
        if not chunk_body:
            continue
        chunks.append(
            Chunk(
                document_id=document_id,
                title=title,
                chunk_id=f"{document_id}-{index:04d}",
                text=chunk_body,
                metadata=metadata or {},
            )
        )
    return chunks


class InMemoryRagStore:
    def __init__(self) -> None:
        self._chunks_by_document: dict[str, list[Chunk]] = {}

    def ingest(
        self,
        document_id: str,
        title: str,
        text: str,
        metadata: dict[str, str] | None = None,
    ) -> list[Chunk]:
        chunks = chunk_text(document_id, title, text, metadata)
        self._chunks_by_document[document_id] = chunks
        return chunks

    def list_documents(self) -> list[dict[str, object]]:
        return [
            {
                "document_id": document_id,
                "title": chunks[0].title,
                "chunk_count": len(chunks),
                "metadata": chunks[0].metadata,
            }
            for document_id, chunks in sorted(self._chunks_by_document.items())
            if chunks
        ]

    def search(
        self,
        query: str,
        top_k: int = 3,
        filters: dict[str, str] | None = None,
    ) -> list[RankedChunk]:
        query_terms = tokenize(query)
        ranked: list[RankedChunk] = []

        for chunks in self._chunks_by_document.values():
            for chunk in chunks:
                if filters and not metadata_matches(chunk.metadata, filters):
                    continue
                score = score_chunk(query_terms, chunk)
                if score > 0:
                    ranked.append(RankedChunk(chunk=chunk, score=score))

        return sorted(ranked, key=lambda item: item.score, reverse=True)[:top_k]


def metadata_matches(metadata: dict[str, str], filters: dict[str, str]) -> bool:
    return all(metadata.get(key) == value for key, value in filters.items())


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9-]+", text.lower())
        if token not in STOPWORDS and len(token) > 2
    }


def score_chunk(query_terms: set[str], chunk: Chunk) -> float:
    if not query_terms:
        return 0.0
    chunk_terms = tokenize(chunk.text)
    overlap = query_terms & chunk_terms
    return len(overlap) / len(query_terms)


def synthesize_answer(question: str, ranked_chunks: list[RankedChunk]) -> str:
    if not ranked_chunks:
        return (
            "I could not find enough grounded mortgage research context to answer this "
            f"question: {question}"
        )

    strongest = ranked_chunks[0].chunk
    excerpt = make_excerpt(strongest.text, max_length=280)
    return (
        f"Based on {strongest.title}, {excerpt} "
        f"[{strongest.document_id}:{strongest.chunk_id}]"
    )


def make_excerpt(text: str, max_length: int = 240) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3].rstrip() + "..."
