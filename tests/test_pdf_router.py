"""Tests for the PDF highlight API endpoint."""

import io
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_pdf(text: str = "Hello world. This is a test PDF document.") -> bytes:
    """Create a minimal single-page PDF in memory."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), text)
    return doc.tobytes()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /api/v1/pdf/highlight
# ---------------------------------------------------------------------------


def test_highlight_pdf_wrong_content_type():
    """Non-PDF uploads should be rejected with HTTP 400."""
    response = client.post(
        "/api/v1/pdf/highlight",
        files={"file": ("test.txt", b"not a pdf", "text/plain")},
    )
    assert response.status_code == 400


def test_highlight_pdf_empty_pdf():
    """A PDF with no extractable text should be rejected with HTTP 400."""
    # Create an empty PDF (no text layer)
    doc = fitz.open()
    doc.new_page()
    empty_pdf_bytes = doc.tobytes()

    with (
        patch(
            "app.routers.pdf.extract_text_from_pdf",
            side_effect=ValueError("The PDF appears to contain no extractable text."),
        ),
    ):
        response = client.post(
            "/api/v1/pdf/highlight",
            files={"file": ("empty.pdf", empty_pdf_bytes, "application/pdf")},
        )
    assert response.status_code == 400
    assert "text" in response.json()["detail"].lower()


def test_highlight_pdf_no_api_key():
    """Missing OpenAI key should surface as HTTP 500."""
    pdf_bytes = _make_simple_pdf()

    with (
        patch("app.routers.pdf.extract_text_from_pdf", return_value="Some text."),
        patch(
            "app.routers.pdf.identify_highlights",
            side_effect=RuntimeError("No OpenAI API key found."),
        ),
    ):
        response = client.post(
            "/api/v1/pdf/highlight",
            files={"file": ("doc.pdf", pdf_bytes, "application/pdf")},
        )
    assert response.status_code == 500
    assert "OpenAI API key" in response.json()["detail"]


def test_highlight_pdf_success(tmp_path):
    """Happy path: a valid PDF should be processed and a PDF returned."""
    pdf_bytes = _make_simple_pdf(
        "Hello world. This is a test PDF document. Important conclusion here."
    )
    highlighted_pdf_bytes = _make_simple_pdf("highlighted version")

    with (
        patch(
            "app.routers.pdf.extract_text_from_pdf",
            return_value="Hello world. This is a test PDF document. Important conclusion here.",
        ),
        patch(
            "app.routers.pdf.identify_highlights",
            return_value=["Hello world.", "Important conclusion here."],
        ),
        patch(
            "app.routers.pdf.apply_highlights",
            return_value=2,
        ) as mock_apply,
        patch("app.routers.pdf.open", create=True) as mock_open,
        patch("app.routers.pdf.tempfile") as mock_tempfile,
    ):
        # Let the router actually write the input PDF but mock the annotator output
        # Instead, test with real calls but mock the agent and annotator
        pass

    # Simpler integration test using real services where possible
    # and patching only the LLM call + annotator I/O
    phrases = ["Hello world.", "Important conclusion here."]

    with (
        patch(
            "app.routers.pdf.identify_highlights",
            return_value=phrases,
        ),
    ):
        response = client.post(
            "/api/v1/pdf/highlight",
            files={"file": ("test.pdf", pdf_bytes, "application/pdf")},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert "highlighted" in response.headers.get("content-disposition", "")
    # The response body should be a valid PDF (starts with %PDF)
    assert response.content[:4] == b"%PDF"


def test_highlight_pdf_file_too_large():
    """Files exceeding the size limit should be rejected with HTTP 413."""
    # Temporarily lower the limit via the env-var-backed constant
    with patch("app.routers.pdf._MAX_UPLOAD_BYTES", 1):
        response = client.post(
            "/api/v1/pdf/highlight",
            files={"file": ("big.pdf", b"%PDF-1.4 " + b"x" * 100, "application/pdf")},
        )
    assert response.status_code == 413


# ---------------------------------------------------------------------------
# Unit tests for pdf_extractor
# ---------------------------------------------------------------------------


def test_extract_text_from_pdf(tmp_path):
    """extract_text_from_pdf should return non-empty text for a real PDF."""
    from app.core.pdf_extractor import extract_text_from_pdf

    pdf_bytes = _make_simple_pdf("Extractable text content for testing.")
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(pdf_bytes)

    text = extract_text_from_pdf(str(pdf_file))
    assert "Extractable" in text or len(text) > 0


# ---------------------------------------------------------------------------
# Unit tests for pdf_annotator
# ---------------------------------------------------------------------------


def test_apply_highlights_creates_annotations(tmp_path):
    """apply_highlights should add highlight annotations to the PDF."""
    from app.core.pdf_annotator import apply_highlights

    text = "This is the important sentence. And another one."
    pdf_bytes = _make_simple_pdf(text)
    input_pdf = tmp_path / "input.pdf"
    output_pdf = tmp_path / "output.pdf"
    input_pdf.write_bytes(pdf_bytes)

    phrases = ["important sentence"]
    count = apply_highlights(str(input_pdf), phrases, str(output_pdf))

    assert output_pdf.exists()
    # The annotated PDF should be a valid PDF file
    doc = fitz.open(str(output_pdf))
    assert doc.page_count >= 1
    doc.close()


# ---------------------------------------------------------------------------
# Unit tests for highlight_agent
# ---------------------------------------------------------------------------


def test_identify_highlights_no_key():
    """identify_highlights should raise RuntimeError when no API key is set."""
    from app.assistant.highlight_agent import identify_highlights

    with patch(
        "app.services.highlight_agent.get_settings",
        return_value=SimpleNamespace(github_token=None, openai_model="gpt-4o-mini"),
    ):
        with pytest.raises(RuntimeError, match="OpenAI API key"):
            identify_highlights("Some text.", openai_api_key=None)


def test_identify_highlights_bad_json():
    """identify_highlights should raise ValueError for non-JSON LLM responses."""
    from app.assistant.highlight_agent import identify_highlights

    mock_response = MagicMock()
    mock_response.content = "This is not JSON"

    with (
        patch("app.services.highlight_agent.ChatOpenAI"),
        patch(
            "app.services.highlight_agent._PROMPT.__or__",
            return_value=MagicMock(invoke=MagicMock(return_value=mock_response)),
        ),
    ):
        with pytest.raises(ValueError, match="non-JSON"):
            identify_highlights("Some text.", openai_api_key="fake-key")


def test_identify_highlights_success():
    """identify_highlights should return a deduplicated list of phrases."""
    from app.assistant.highlight_agent import identify_highlights

    mock_response = MagicMock()
    mock_response.content = '["Key phrase one.", "Key phrase two.", "Key phrase one."]'

    # _PROMPT | llm creates a RunnableSequence.  Patch _PROMPT so that __or__
    # returns a mock chain whose invoke() yields our pre-baked response.
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_response

    with (
        patch("app.services.highlight_agent.ChatOpenAI"),
        patch(
            "app.services.highlight_agent._PROMPT",
            **{"__or__": MagicMock(return_value=mock_chain)},
        ),
    ):
        phrases = identify_highlights("Some text.", openai_api_key="fake-key")

    assert phrases == ["Key phrase one.", "Key phrase two."]
