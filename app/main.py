"""Coloread – application entry point."""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.core import router

app = FastAPI(
    title="Coloread",
    description=(
        "Read any book 5× faster with automated colorful highlighted texts. "
        "Upload a PDF and receive a version with the most important passages "
        "highlighted by an AI agent."
    ),
    version="0.1.0",
)

app.include_router(router.router, prefix="/api/v1")


@app.get("/health", tags=["health"])
async def health() -> JSONResponse:
    """Return a simple health-check response."""
    return JSONResponse({"status": "ok"})
