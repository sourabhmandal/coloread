# Coloread
Read any book 5× faster with automated colorful highlighted texts.

## Overview

**coloread** is a FastAPI application that ingests a PDF via an API endpoint and returns an annotated version of the document with the most important passages highlighted in yellow — powered by an agentic AI workflow.

![Coloread demo](assets/coloread_demo.gif)

### How it works

1. **Upload** – Send a PDF file to `POST /api/v1/pdf/highlight`.
2. **Extract** – [opendataloader-pdf](https://pypi.org/project/opendataloader-pdf/) converts the PDF to plain text.
3. **Analyse** – A [LangChain](https://python.langchain.com/) agent (backed by an OpenAI LLM) reads the text and identifies the most important phrases and sentences.
4. **Annotate** – [PyMuPDF](https://pymupdf.readthedocs.io/) searches for each phrase in the original PDF and adds a yellow highlight annotation.
5. **Download** – The highlighted PDF is returned as a file download.

## Requirements

- Python 3.12+
- An OpenAI-compatible LLM endpoint (default: `http://127.0.0.1:8080/v1`)

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/sourabhmandal/coloread.git
cd coloread

# 2. Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and set OPENAI_BASE_URL / OPENAI_MODEL
```

## Running the server

```bash
uvicorn app.main:app --reload
```

## Running a Local LLM
We use Gemma 4 2B model for development (gemma-4-E2B-it-Q4_K_M.gguf). Here is a example

```bash
./build/bin/llama-server -m ~/.lmstudio/models/lmstudio-community/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_K_M.gguf -ngl 999 --port 8080
```

## API reference
Interactive docs (Swagger UI) are at `http://localhost:8000/docs`.

| Method | API | Description |
|--------|----------|---------|
| `POST` | `/api/v1/assistant/highlight` | send a pdf file and this api highlight important sections from the pdf |

## Running tests

```bash
pytest tests/ -v
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_BASE_URL` | `http://127.0.0.1:8080/v1` | OpenAI-compatible chat completions endpoint base URL. |
| `OPENAI_API_KEY` | *(optional)* | API key used by providers that require auth. |
| `OPENAI_MODEL` | `gemma-4-E2B-it-GGUF` | Model used by the highlight agent. |
| `MAX_UPLOAD_SIZE_MB` | `20` | Maximum accepted PDF size in megabytes. |

## Project structure

```
coloread/
├── app/
│   ├── main.py                              # entrypoint
│   ├── assistant/
│   │   └── highlight_agent.py               # Apply highlights via PyMuPDF
│   ├── core/
│   │   ├── router.py                        # CRUD API endpoint
│   │   ├── pdf_extractor.py                 # LangChain agent to identify highlights
│   │   └── pdf_annotator.py                 # Apply highlights via PyMuPDF
│   │   └── schemas.py                       # pydantic schemas for strict typing
├── tests/
│   └── test_pdf_router.py                   # Pytest test suite
├── requirements.txt
├── .env.example
└── README.md
```
