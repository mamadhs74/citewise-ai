from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    source: str
    chunk_id: int
    text: str
    score: float

    @property
    def citation(self) -> str:
        return f"[{self.source}#chunk-{self.chunk_id}]"
