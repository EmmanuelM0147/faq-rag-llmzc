# FAQ RAG Assistant

A retrieval-augmented generation (RAG) pipeline that answers questions from a structured FAQ knowledge base. Documents are retrieved with BM25-style search, then synthesized into natural-language answers with Google Gemini.

## How it works

1. **Ingest** — Load FAQ documents from a public JSON API
2. **Retrieve** — Search by question, section, and answer text (in-memory or SQLite)
3. **Generate** — Send retrieved context to Gemini with a fixed system instruction

## Features

- FAQ ingestion from a public API
- In-memory search with `minsearch`
- Persistent SQLite search with `sqlitesearch` (`faq.db`)
- Reusable `RAGBase` helper with Gemini integration and automatic model fallback
- Jupyter notebooks for exploration and ingestion

## Requirements

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- A [Google AI API key](https://ai.google.dev/)

## Setup

1. Clone the repository:

```bash
git clone https://github.com/EmmanuelM0147/faq-rag-llmzc.git
cd faq-rag-llmzc
```

2. Install dependencies:

```bash
python -m uv sync
```

If `uv` is not on your PATH:

```bash
pip install uv
python -m uv sync
```

3. Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_api_key_here
```

4. Start Jupyter:

```bash
python -m uv run jupyter notebook
```

## Project structure

| File | Description |
|------|-------------|
| `ingest.py` | Load FAQ data and build a `minsearch` index |
| `rag_helper.py` | `RAGBase` class — search, prompt building, and Gemini calls |
| `notebook.ipynb` | End-to-end RAG walkthrough |
| `rag_cleaned.ipynb` | Streamlined RAG workflow |
| `rag_ingest.ipynb` | Build an in-memory search index |
| `persistent_rag_ingest.ipynb` | Ingest FAQs into SQLite (`faq.db`) |
| `sqlite-ingest.ipynb` | SQLite ingest with progress output |
| `persinsent_rag.ipynb` | Query the persistent SQLite index |

## Usage

### In-memory RAG

```python
import os
from dotenv import load_dotenv
from google import genai

from ingest import load_faq_data, build_index
from rag_helper import RAGBase

load_dotenv()
genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

documents = load_faq_data()
index = build_index(documents)

assistant = RAGBase(index, genai_client)
print(assistant.rag("I just discovered the course. Can I join now?"))
```

### Persistent SQLite RAG

1. Run `persistent_rag_ingest.ipynb` to build `faq.db`.
2. Query with `persinsent_rag.ipynb`:

```python
from sqlitesearch import TextSearchIndex

sqlite_index = TextSearchIndex(
    text_fields=["question", "section", "answer"],
    keyword_fields=["course"],
    db_path="faq.db",
)

assistant = RAGBase(sqlite_index, genai_client)
print(assistant.rag("How do I get a certificate?"))
```

## Models

By default, `RAGBase` uses `gemini-2.5-flash-lite` and falls back to:

- `gemini-3.1-flash-lite`
- `gemma-4-26b-a4b-it`

Override the model when creating the assistant:

```python
assistant = RAGBase(index, genai_client, model="gemini-3.1-flash-lite")
```

## Notes

- Loading all FAQ data takes ~30–60 seconds (network requests to the data source).
- `faq.db`, `.env`, and `.venv` are gitignored — create them locally.
- Free-tier Gemini API quotas may cause `429` errors; the helper retries and tries fallback models automatically.

## Acknowledgments

FAQ data from [DataTalks.Club](https://datatalks.club/faq/). Built while studying LLM engineering and retrieval-augmented generation.
