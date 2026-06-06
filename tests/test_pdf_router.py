"""Tests for the PDF highlight API endpoint and keyphrase extraction."""

from unittest.mock import patch

import fitz
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _make_simple_pdf(text: str) -> bytes:
    """Create a minimal single-page PDF in memory."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), text)
    return doc.tobytes()


def test_health_endpoints():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    response = client.get("/api/v1/assistant/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_highlight_pdf_rejects_non_pdf():
    response = client.post(
        "/api/v1/assistant/highlight",
        files={"file": ("test.txt", b"not a pdf", "text/plain")},
    )

    assert response.status_code == 400


def test_highlight_pdf_rejects_empty_pdf():
    empty_pdf = _make_simple_pdf("")

    with patch(
        "app.core.router.extract_text_from_pdf",
        side_effect=ValueError("The PDF appears to contain no extractable text."),
    ):
        response = client.post(
            "/api/v1/assistant/highlight",
            files={"file": ("empty.pdf", empty_pdf, "application/pdf")},
        )

    assert response.status_code == 400
    assert "text" in response.json()["detail"].lower()


def test_highlight_pdf_success():
    pdf_text = (
        "Introduction paragraph with overview and context. "
        "Therefore, the retention rate improved significantly. "
        "Supporting detail about methodology and data collection. "
        "In conclusion, the approach works because the evaluation was repeated."
    )
    pdf_bytes = _make_simple_pdf(pdf_text)

    with (
        patch("app.core.router.extract_text_from_pdf", return_value=pdf_text),
        patch(
            "app.core.router.identify_highlights",
            return_value=[
                "Introduction paragraph",
                "retention rate improved significantly",
                "Supporting detail",
                "data collection",
                "approach works",
            ],
        ),
    ):
        response = client.post(
            "/api/v1/assistant/highlight",
            files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert "highlighted" in response.headers.get("content-disposition", "")
    assert response.content[:4] == b"%PDF"


def test_highlight_pdf_file_too_large():
    with patch("app.core.router._MAX_UPLOAD_BYTES", 1):
        response = client.post(
            "/api/v1/assistant/highlight",
            files={"file": ("big.pdf", b"%PDF-1.4 " + b"x" * 100, "application/pdf")},
        )

    assert response.status_code == 413


def test_identify_highlights_prefers_keyphrases_and_biases():
    from app.assistant.highlight_agent import identify_highlights

    text = (
        "Introduction paragraph with overview and context. "
        "Therefore, the retention rate improved significantly. "
        "Supporting detail about methodology and data collection. "
        "In conclusion, the approach works because the evaluation was repeated."
    )

    phrases = identify_highlights(text)

    assert len(phrases) >= 5
    assert "Introduction paragraph" in phrases
    assert "retention rate improved significantly" in phrases
    assert "approach works" in phrases
    assert all(phrase in text for phrase in phrases)
    assert all(len(phrase.split()) > 1 for phrase in phrases)


def test_identify_highlights_empty_text():
    from app.assistant.highlight_agent import identify_highlights

    assert identify_highlights("   ") == []