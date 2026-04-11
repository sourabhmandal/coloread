"""Pydantic schemas for the Coloread API."""

from pydantic import BaseModel, Field


class HighlightResponse(BaseModel):
    """Metadata returned alongside the highlighted PDF download."""

    filename: str = Field(..., description="Name of the highlighted PDF file.")
    phrases_highlighted: list[str] = Field(
        ...,
        description="List of text phrases that were highlighted in the document.",
    )
    total_highlights: int = Field(
        ..., description="Total number of highlight annotations applied."
    )


class ErrorResponse(BaseModel):
    """Standard error response body."""

    detail: str = Field(..., description="Human-readable error message.")
