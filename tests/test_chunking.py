import pytest

from citewise.chunking import chunk_text


def test_chunk_text_uses_overlap() -> None:
    text = " ".join(f"word{i}" for i in range(10))
    chunks = chunk_text(text, chunk_size=6, overlap=2)

    assert chunks == [
        "word0 word1 word2 word3 word4 word5",
        "word4 word5 word6 word7 word8 word9",
    ]


def test_chunk_text_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=5, overlap=5)
