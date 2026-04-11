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
- An **Github PAT (Token)** (Support for other api keys will be added later)

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
# Edit .env and set your GITHUB_TOKEN
```

## Running the server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.  
Interactive docs (Swagger UI) are at `http://localhost:8000/docs`.

## API reference

| Method | API | Description |
|--------|----------|---------|
| `POST` | `/api/v1/pdf/highlight` | send a pdf file and this api highlight important sections from the pdf |

## Running tests

```bash
pytest tests/ -v
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GITHUB_TOKEN` | *(required)* | Your OpenAI API key. |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model used by the highlight agent. |
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
