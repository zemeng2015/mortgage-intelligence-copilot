from app.rag import InMemoryRagStore, chunk_text, synthesize_answer


def test_chunk_text_adds_document_metadata() -> None:
    chunks = chunk_text(
        document_id="doc-1",
        title="Housing Report",
        text="Delinquency declined while borrower risk improved.",
        metadata={"region": "us"},
    )

    assert chunks[0].document_id == "doc-1"
    assert chunks[0].title == "Housing Report"
    assert chunks[0].metadata == {"region": "us"}


def test_store_search_ranks_matching_chunks() -> None:
    store = InMemoryRagStore()
    store.ingest(
        document_id="risk-report",
        title="Risk Report",
        text="Borrower risk improved because mortgage delinquency declined.",
    )
    store.ingest(
        document_id="rate-report",
        title="Rate Report",
        text="Mortgage rates increased during the quarter.",
    )

    results = store.search("How did borrower risk and delinquency change?")

    assert results[0].chunk.document_id == "risk-report"
    assert results[0].score > 0


def test_store_search_applies_metadata_filters() -> None:
    store = InMemoryRagStore()
    store.ingest(
        document_id="west-report",
        title="West Report",
        text="Mortgage delinquency declined in the western region.",
        metadata={"region": "west"},
    )
    store.ingest(
        document_id="east-report",
        title="East Report",
        text="Mortgage delinquency increased in the eastern region.",
        metadata={"region": "east"},
    )

    results = store.search("mortgage delinquency", filters={"region": "east"})

    assert [result.chunk.document_id for result in results] == ["east-report"]


def test_synthesize_answer_includes_citation_marker() -> None:
    store = InMemoryRagStore()
    store.ingest(
        document_id="risk-report",
        title="Risk Report",
        text="Borrower risk improved because mortgage delinquency declined.",
    )

    answer = synthesize_answer("What happened to borrower risk?", store.search("borrower risk"))

    assert "Risk Report" in answer
    assert "[risk-report:risk-report-0000]" in answer
