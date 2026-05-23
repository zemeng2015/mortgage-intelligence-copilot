# Mortgage Intelligence Copilot

Production-style RAG and agent workflow system for mortgage, housing, and financial research documents.

## Why This Project

This project is designed to show applied AI engineering strength: retrieval, citations, tool calling, workflow orchestration, evaluation, and cloud-ready backend design. It maps naturally to enterprise AI roles that need Python, AWS, data pipelines, and reliable LLM systems.

## Target Resume Bullet

Built a production-style mortgage research copilot using FastAPI, AWS-style workflow orchestration, document ingestion, vector retrieval, and citation-grounded LLM responses over financial documents.

## Core Capabilities

- Upload and ingest mortgage or housing research documents.
- Parse documents into chunks with metadata.
- Retrieve relevant chunks with a local lexical retriever.
- Answer user questions with grounded citations.
- Run agent tools such as document search, scenario comparison, and report generation.
- Track latency, token usage, retrieval quality, and citation correctness.

## Current Implementation

This first version is a local-first RAG service that does not require an external
LLM or vector database. It focuses on the backend mechanics that make a copilot
credible: ingestion, chunking, retrieval, citations, confidence scoring, and API
tests.

- `POST /documents` ingests a mortgage or housing research document.
- `GET /documents` lists ingested documents and chunk counts.
- `POST /ask` retrieves matching chunks and returns a citation-grounded answer.
- `GET /runs/{run_id}` returns retrieval and answer-generation trace details.
- Metadata filters support scoped retrieval, such as region or report type.
- `answer_mode` supports `local`, `openai`, and `auto`.
- Unit tests cover health checks, ingestion, retrieval, filtering, citations, LLM
  provider wiring, and run traces.

## Suggested Stack

- Backend: FastAPI, Pydantic, SQLAlchemy
- AI: OpenAI API or AWS Bedrock, LangGraph, sentence-transformers
- Storage: Postgres + pgvector, S3-compatible object storage
- Workflow: AWS Step Functions design, local orchestrator for development
- Observability: structured logs, eval reports, cost tracking

## Milestones

1. Build local document ingestion and chunking.
2. Add vector search with citation-grounded RAG.
3. Add tool-calling agent workflow.
4. Add evaluation dataset and regression runner.
5. Add cloud deployment notes for AWS.

## Local Development

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
python -m pytest
python -m ruff check .
```

OpenAI support is optional:

```bash
pip install -e ".[openai]"
```

## API Example

Ingest a document:

```powershell
$body = @{
  document_id = "sample-housing-report"
  title = "Sample Housing Market Report"
  text = [string](Get-Content .\samples\housing_report.txt -Raw)
  metadata = @{ region = "us"; report_type = "market" }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri http://127.0.0.1:8000/documents `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

Ask a grounded question:

```powershell
$body = @{
  question = "What happened to borrower risk and delinquency?"
  filters = @{ region = "us" }
  answer_mode = "local"
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri http://127.0.0.1:8000/ask `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

Use OpenAI generation after setting `OPENAI_API_KEY`:

```powershell
$env:OPENAI_API_KEY = "your_api_key"

$body = @{
  question = "What happened to borrower risk and delinquency?"
  filters = @{ region = "us" }
  answer_mode = "openai"
  model = "gpt-4.1"
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri http://127.0.0.1:8000/ask `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

Inspect a run trace:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/runs/{run_id}
```

## Next Milestones

1. Replace lexical retrieval with embeddings and pgvector.
2. Add citation validation against retrieved chunks.
3. Connect the eval runner from `llm-eval-observability`.
4. Add Bedrock provider support.
5. Add AWS deployment notes with S3, ECS/Lambda, and Postgres.
