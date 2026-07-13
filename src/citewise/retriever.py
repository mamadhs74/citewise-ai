from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import chunk_text
from .models import SearchResult


class EvidenceRetriever:
    def __init__(
        self,
        *,
        chunk_size: int = 120,
        overlap: int = 25,
        min_score: float = 0.01,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_score = min_score
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._chunks: list[tuple[str, int, str]] = []

    @property
    def is_ready(self) -> bool:
        return self._vectorizer is not None and bool(self._chunks)

    def fit(self, documents: Mapping[str, str]) -> int:
        chunks: list[tuple[str, int, str]] = []

        for source, text in documents.items():
            for chunk_id, chunk in enumerate(
                chunk_text(
                    text,
                    chunk_size=self.chunk_size,
                    overlap=self.overlap,
                ),
                start=1,
            ):
                chunks.append((source, chunk_id, chunk))

        if not chunks:
            raise ValueError("No non-empty document chunks were provided")

        corpus = [text for _, _, text in chunks]
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(corpus)

        self._chunks = chunks
        self._vectorizer = vectorizer
        self._matrix = matrix
        return len(chunks)

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        if not self.is_ready:
            raise RuntimeError("Retriever must be fitted before search")
        if not query.strip():
            return []
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        assert self._vectorizer is not None
        query_vector = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self._matrix).ravel()
        ranked_indices = np.argsort(scores)[::-1]

        results: list[SearchResult] = []
        for index in ranked_indices:
            score = float(scores[index])
            if score < self.min_score:
                continue
            source, chunk_id, text = self._chunks[int(index)]
            results.append(
                SearchResult(
                    source=source,
                    chunk_id=chunk_id,
                    text=text,
                    score=score,
                )
            )
            if len(results) == top_k:
                break
        return results
