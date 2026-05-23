from pydantic import BaseModel, Field


class DocumentIngestRequest(BaseModel):
    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=20)
    metadata: dict[str, str] = Field(default_factory=dict)


class DocumentIngestResponse(BaseModel):
    document_id: str
    title: str
    chunk_count: int


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    filters: dict[str, str] = Field(default_factory=dict)
    top_k: int = Field(default=3, ge=1, le=10)
    answer_mode: str = Field(default="auto", pattern="^(auto|local|openai)$")
    model: str = "gpt-4.1"


class Citation(BaseModel):
    document_id: str
    title: str
    chunk_id: str
    excerpt: str
    score: float


class AskResponse(BaseModel):
    run_id: str
    answer: str
    citations: list[Citation]
    confidence: float
    retrieval_query: str
    answer_mode: str
    model: str | None = None


class RetrievedChunkTrace(BaseModel):
    document_id: str
    title: str
    chunk_id: str
    score: float
    excerpt: str


class RunTrace(BaseModel):
    run_id: str
    question: str
    answer_mode: str
    model: str | None = None
    retrieval_query: str
    retrieved_chunks: list[RetrievedChunkTrace]
    answer: str
    latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    notes: list[str] = Field(default_factory=list)


class DocumentSummary(BaseModel):
    document_id: str
    title: str
    chunk_count: int
    metadata: dict[str, str] = Field(default_factory=dict)
