"""PDF text extraction using opendataloader-pdf.

The ``opendataloader_pdf.convert`` function converts a PDF file on disk to one
or more output formats (JSON, text, markdown, …).  We request the ``text``
format so we get a plain-text representation of the document content which we
can then feed to the LangChain highlight agent.
"""

import json
import os
import tempfile
from pathlib import Path

from opendataloader_pdf import convert


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract plain text from a PDF file using opendataloader-pdf.

    Args:
        pdf_path: Absolute path to the PDF file on disk.

    Returns:
        The extracted text content of the PDF.

    Raises:
        ValueError: If the PDF could not be parsed or yields no text.
    """
    with tempfile.TemporaryDirectory() as output_dir:
        convert(
            input_path=pdf_path,
            output_dir=output_dir,
            format="text",
            quiet=True,
        )

        pdf_stem = Path(pdf_path).stem
        txt_file = Path(output_dir) / f"{pdf_stem}.txt"

        if not txt_file.exists():
            # Fall back: look for any .txt file in the output directory
            txt_files = list(Path(output_dir).glob("*.txt"))
            if not txt_files:
                raise ValueError(
                    "opendataloader-pdf produced no text output for the provided PDF."
                )
            txt_file = txt_files[0]

        text = txt_file.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError("The PDF appears to contain no extractable text.")

    return text
