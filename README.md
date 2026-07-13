# CiteWise AI

CiteWise AI is a local, evidence-first document search application. It indexes text files, ranks relevant passages for a question, and displays an inspectable citation for every result.

## Why this project

Modern AI assistants need retrieval, grounding, evaluation, and transparent evidence. This first version deliberately stops before answer generation: it makes retrieval behavior visible and testable before adding an LLM.

## Features

- Local `.txt` document ingestion
- Overlapping passage chunking
- TF-IDF unigram and bigram retrieval
- Cosine-similarity ranking
- Source and chunk citations
- Streamlit interface
- Unit tests for chunking and retrieval

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
streamlit run app.py
```

On Windows PowerShell, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Test

```bash
pytest -q
```

## Roadmap

1. Add PDF ingestion with page-level citations.
2. Add a labeled retrieval evaluation dataset.
3. Add hybrid lexical and embedding retrieval.
4. Add optional local LLM answer generation constrained to retrieved evidence.
5. Add confidence and abstention behavior.
