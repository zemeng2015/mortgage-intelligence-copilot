import time
import uuid

from fastapi import FastAPI
from fastapi import HTTPException

from app.llm import generate_with_fallback
from app.models import (
    AskRequest,
    AskResponse,
    Citation,
    DocumentIngestRequest,
    DocumentIngestResponse,
    DocumentSummary,
    RetrievedChunkTrace,
    RunTrace,
)
from app.rag import InMemoryRagStore, make_excerpt
from app.tracing import InMemoryRunStore

app = FastAPI(title="Mortgage Intelligence Copilot")
rag_store = InMemoryRagStore()
run_store = InMemoryRunStore()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents", response_model=DocumentIngestResponse)
def ingest_document(request: DocumentIngestRequest) -> DocumentIngestResponse:
    chunks = rag_store.ingest(
        document_id=request.document_id,
        title=request.title,
        text=request.text,
        metadata=request.metadata,
    )
    return DocumentIngestResponse(
        document_id=request.document_id,
        title=request.title,
        chunk_count=len(chunks),
    )


@app.get("/documents", response_model=list[DocumentSummary])
def list_documents() -> list[DocumentSummary]:
    return [DocumentSummary.model_validate(document) for document in rag_store.list_documents()]


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    run_id = str(uuid.uuid4())
    start = time.perf_counter()
    retrieval_start = time.perf_counter()
    ranked_chunks = rag_store.search(
        query=request.question,
        top_k=request.top_k,
        filters=request.filters,
    )
    retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

    generation = generate_with_fallback(
        question=request.question,
        ranked_chunks=ranked_chunks,
        answer_mode=request.answer_mode,
        model=request.model,
    )

    citations = [
        Citation(
            document_id=item.chunk.document_id,
            title=item.chunk.title,
            chunk_id=item.chunk.chunk_id,
            excerpt=make_excerpt(item.chunk.text),
            score=item.score,
        )
        for item in ranked_chunks
    ]
    retrieved_chunk_traces = [
        RetrievedChunkTrace(
            document_id=item.chunk.document_id,
            title=item.chunk.title,
            chunk_id=item.chunk.chunk_id,
            excerpt=make_excerpt(item.chunk.text),
            score=item.score,
        )
        for item in ranked_chunks
    ]
    total_latency_ms = (time.perf_counter() - start) * 1000
    response = AskResponse(
        run_id=run_id,
        answer=generation.answer,
        citations=citations,
        confidence=ranked_chunks[0].score if ranked_chunks else 0.0,
        retrieval_query=request.question,
        answer_mode=generation.answer_mode,
        model=generation.model,
    )
    run_store.save(
        RunTrace(
            run_id=run_id,
            question=request.question,
            answer_mode=generation.answer_mode,
            model=generation.model,
            retrieval_query=request.question,
            retrieved_chunks=retrieved_chunk_traces,
            answer=generation.answer,
            latency_ms=total_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation.latency_ms,
            notes=generation.notes,
        )
    )
    return response


@app.get("/runs/{run_id}", response_model=RunTrace)
def get_run(run_id: str) -> RunTrace:
    trace = run_store.get(run_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return trace
