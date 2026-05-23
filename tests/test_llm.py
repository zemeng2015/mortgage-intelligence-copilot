from app.llm import OpenAIAnswerGenerator, build_grounded_prompt, select_answer_generator
from app.rag import InMemoryRagStore


def test_select_answer_generator_uses_local_when_requested() -> None:
    generator = select_answer_generator(answer_mode="local", model="test-model")

    assert generator.__class__.__name__ == "LocalAnswerGenerator"


def test_build_grounded_prompt_includes_citation_markers() -> None:
    store = InMemoryRagStore()
    store.ingest(
        document_id="doc-1",
        title="Risk Report",
        text="Borrower risk improved because delinquency declined.",
    )
    ranked_chunks = store.search("borrower risk")

    prompt = build_grounded_prompt("What changed?", ranked_chunks)

    assert "[doc-1:doc-1-0000]" in prompt
    assert "Answer only from the provided context" in prompt


def test_openai_answer_generator_uses_responses_api_client() -> None:
    class FakeResponse:
        output_text = "Borrower risk improved [doc-1:doc-1-0000]."

    class FakeResponses:
        def create(self, model: str, input: str) -> FakeResponse:
            assert model == "test-model"
            assert "What changed?" in input
            return FakeResponse()

    class FakeClient:
        responses = FakeResponses()

    store = InMemoryRagStore()
    store.ingest(
        document_id="doc-1",
        title="Risk Report",
        text="Borrower risk improved because delinquency declined.",
    )

    result = OpenAIAnswerGenerator(model="test-model", client=FakeClient()).generate(
        "What changed?",
        store.search("borrower risk"),
    )

    assert result.answer_mode == "openai"
    assert result.model == "test-model"
    assert "[doc-1:doc-1-0000]" in result.answer
