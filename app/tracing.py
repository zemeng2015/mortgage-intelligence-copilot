from app.models import RunTrace


class InMemoryRunStore:
    def __init__(self) -> None:
        self._runs: dict[str, RunTrace] = {}

    def save(self, trace: RunTrace) -> None:
        self._runs[trace.run_id] = trace

    def get(self, run_id: str) -> RunTrace | None:
        return self._runs.get(run_id)
