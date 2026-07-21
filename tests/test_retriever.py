import pytest

from citewise.retriever import EvidenceRetriever


@pytest.fixture
def retriever() -> EvidenceRetriever:
    engine = EvidenceRetriever(chunk_size=30, overlap=5, min_score=0.0)
    engine.fit(
        {
            "retrieval.txt": (
                "Retrieval systems rank relevant passages for a user query. "
                "Evaluation should measure whether useful evidence is returned."
            ),
            "vision.txt": (
                "Computer vision systems analyze images, objects, and visual scenes."
            ),
        }
    )
    return engine


def test_search_returns_relevant_source_first(retriever: EvidenceRetriever) -> None:
    results = retriever.search("How do we evaluate retrieval evidence?", top_k=2)

    assert results
    assert results[0].source == "retrieval.txt"
    assert results[0].citation.startswith("[retrieval.txt#chunk-")


def test_empty_query_returns_no_results(retriever: EvidenceRetriever) -> None:
    assert retriever.search("   ") == []


def test_search_requires_fit() -> None:
    with pytest.raises(RuntimeError):
        EvidenceRetriever().search("query")


def test_available_sources_are_sorted(
    retriever: EvidenceRetriever,
) -> None:
    assert retriever.available_sources == (
        "retrieval.txt",
        "vision.txt",
    )


def test_search_can_filter_by_source(
    retriever: EvidenceRetriever,
) -> None:
    results = retriever.search(
        "systems analyze evidence and images",
        top_k=5,
        sources={"vision.txt"},
    )

    assert results
    assert all(result.source == "vision.txt" for result in results)


def test_empty_source_filter_returns_no_results(
    retriever: EvidenceRetriever,
) -> None:
    results = retriever.search(
        "retrieval evidence",
        sources=set(),
    )

    assert results == []
