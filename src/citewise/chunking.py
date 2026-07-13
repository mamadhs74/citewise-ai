from __future__ import annotations

import re
from pathlib import Path


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 25) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = normalize_whitespace(text).split()
    if not words:
        return []

    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        if start + chunk_size >= len(words):
            break
    return chunks


def load_text_documents(directory: Path) -> dict[str, str]:
    if not directory.exists():
        raise FileNotFoundError(f"Document directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Expected a directory: {directory}")

    documents: dict[str, str] = {}
    for path in sorted(directory.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents[path.name] = text
    return documents
