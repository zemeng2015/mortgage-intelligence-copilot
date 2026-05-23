from fastapi.testclient import TestClient

from app.main import app


def test_ingest_document_and_ask_returns_citations() -> None:
    client = TestClient(app)

    ingest_response = client.post(
        "/documents",
        json={
            "document_id": "sample-housing-report",
            "title": "Sample Housing Market Report",
            "text": (
                "Borrower risk improved in the second quarter because mortgage "
                "delinquency declined and refinance activity stabilized."
            ),
            "metadata": {"region": "us"},
        },
    )

    assert ingest_response.status_code == 200
    assert ingest_response.json()["chunk_count"] == 1

    ask_response = client.post(
        "/ask",
        json={
            "question": "What happened to borrower risk and delinquency?",
            "filters": {"region": "us"},
        },
    )

    body = ask_response.json()
    assert ask_response.status_code == 200
    assert body["run_id"]
    assert body["answer_mode"] == "local"
    assert body["confidence"] > 0
    assert body["citations"][0]["document_id"] == "sample-housing-report"
    assert "Borrower risk improved" in body["answer"]

    run_response = client.get(f"/runs/{body['run_id']}")
    run_body = run_response.json()

    assert run_response.status_code == 200
    assert run_body["run_id"] == body["run_id"]
    assert run_body["retrieved_chunks"][0]["document_id"] == "sample-housing-report"
    assert run_body["retrieval_latency_ms"] >= 0
    assert run_body["generation_latency_ms"] >= 0
