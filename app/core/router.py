"""PDF ingestion and highlighting router.

Endpoint
--------
POST /api/v1/pdf/highlight
    Accept a PDF file upload, run the agentic highlight workflow, and return
    an annotated PDF file with the important passages highlighted in yellow.
"""

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from app.settings import get_settings
from app.assistant.highlight_agent import identify_highlights
from app.core.pdf_annotator import apply_highlights
from app.core.pdf_extractor import extract_text_from_pdf

router = APIRouter(prefix="/pdf", tags=["pdf"])

_MAX_UPLOAD_BYTES = get_settings().max_upload_size_mb * 1024 * 1024


@router.post(
    "/highlight",
    summary="Ingest a PDF and return it with important text highlighted",
    response_description="The annotated PDF file with yellow highlights.",
    responses={
        200: {"content": {"application/pdf": {}}, "description": "Highlighted PDF."},
        400: {"description": "Invalid or unprocessable PDF."},
        413: {"description": "File exceeds the maximum allowed size."},
        500: {"description": "Internal server error during processing."},
    },
)
async def highlight_pdf(file: UploadFile) -> FileResponse:
    """Upload a PDF and receive a highlighted version back.

    The endpoint:
    1. Reads the uploaded PDF.
    2. Extracts its text content using **opendataloader-pdf**.
    3. Sends the text to a **LangChain** agent that identifies the most
       important phrases.
    4. Applies yellow highlight annotations to those phrases using PyMuPDF.
    5. Returns the annotated PDF as a file download.
    """
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted (content-type: application/pdf).",
        )

    content = await file.read()

    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size {len(content) / 1024 / 1024:.1f} MB exceeds the "
                f"{_MAX_UPLOAD_BYTES // 1024 // 1024} MB limit."
            ),
        )

    original_filename = Path(file.filename or "document.pdf").stem

    with tempfile.TemporaryDirectory() as work_dir:
        input_pdf = str(Path(work_dir) / f"{original_filename}.pdf")
        output_pdf = str(Path(work_dir) / f"{original_filename}_highlighted.pdf")

        with open(input_pdf, "wb") as fh:
            fh.write(content)

        try:
            text = extract_text_from_pdf(input_pdf)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        try:
            phrases = identify_highlights(text)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Highlight agent returned an unexpected response: {exc}",
            ) from exc

        apply_highlights(input_pdf, phrases, output_pdf)

        # Read the annotated PDF into memory so we can return it after the
        # temporary directory is cleaned up.
        with open(output_pdf, "rb") as fh:
            annotated_bytes = fh.read()

    # Write the final file to a stable temp location for FileResponse
    final_path = tempfile.mktemp(suffix=".pdf")
    with open(final_path, "wb") as fh:
        fh.write(annotated_bytes)

    return FileResponse(
        path=final_path,
        media_type="application/pdf",
        filename=f"{original_filename}_highlighted.pdf",
        background=None,
    )
