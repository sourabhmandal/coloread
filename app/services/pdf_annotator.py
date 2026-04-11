"""PDF annotation – apply highlight annotations using PyMuPDF.

Given a list of verbatim phrases returned by the highlight agent, this module
searches for those phrases in every page of a PDF document and adds a yellow
highlight annotation wherever a match is found.
"""

import fitz  # PyMuPDF


def apply_highlights(pdf_path: str, phrases: list[str], output_path: str) -> int:
    """Search *phrases* in the PDF and apply highlight annotations.

    Args:
        pdf_path: Path to the original (input) PDF file.
        phrases: List of verbatim text phrases to highlight.
        output_path: Path where the annotated PDF will be saved.

    Returns:
        The total number of highlight annotations that were applied.
    """
    doc = fitz.open(pdf_path)
    total = 0

    for page in doc:
        for phrase in phrases:
            if not phrase.strip():
                continue
            # search_for returns a list of Rect objects for each occurrence
            rects = page.search_for(phrase)
            for rect in rects:
                highlight = page.add_highlight_annot(rect)
                highlight.update()
                total += 1

    doc.save(output_path)
    doc.close()

    return total
