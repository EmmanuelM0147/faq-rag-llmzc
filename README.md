# FAQ RAG — LLM Zoomcamp

Retrieval-augmented generation (RAG) over the [DataTalks.Club FAQ](https://datatalks.club/faq/) for **LLM Zoomcamp**. The project loads course FAQ documents, searches them with BM25-style retrieval, and answers questions using **Google Gemini**.

Built as part of the LLM Zoomcamp course.

## Features

- FAQ ingestion from the DataTalks.Club API
- In-memory search with `minsearch`
- Persistent SQLite search with `sqlitesearch` (`faq.db`)
- Reusable `RAGBase` helper with Gemini integration and model fallback
- Jupyter notebooks covering the full RAG workflow

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
| `ingest.py` | Load FAQ data from DataTalks.Club and build a `minsearch` index |
| `rag_helper.py` | `RAGBase` class — search, prompt building, and Gemini calls |
| `notebook.ipynb` | Main course notebook: RAG from scratch with Gemini |
| `rag_cleaned.ipynb` | Simplified RAG workflow |
| `rag_ingest.ipynb` | Build an in-memory search index |
| `persistent_rag_ingest.ipynb` | Ingest LLM Zoomcamp FAQs into SQLite (`faq.db`) |
| `sqlite-ingest.ipynb` | SQLite ingest with progress output |
| `persinsent_rag.ipynb` | RAG using the persistent SQLite index |

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
2. Use the index in `persinsent_rag.ipynb`:

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

- Loading all FAQ data takes ~30–60 seconds (network requests to DataTalks.Club).
- `faq.db`, `.env`, and `.venv` are gitignored — create them locally.
- Free-tier Gemini API quotas may cause `429` errors; the helper retries and tries fallback models automatically.

## License

Course homework / learning project.
