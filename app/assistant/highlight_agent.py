"""Agentic highlight workflow using LangChain.

This module implements a LangChain-based agent that reads the text extracted
from a PDF and returns a list of important phrases / sentences that should be
highlighted.  The agent is intentionally kept stateless: a fresh call to
``identify_highlights`` produces an independent result and does not rely on
conversation history.
"""

import json
from typing import Optional

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import get_settings

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert reading assistant.  Your job is to identify the most \
important phrases, sentences, or short passages in a document that a reader \
should highlight for quick review.

Rules:
- Return ONLY a JSON array of strings.  Each string must be an exact verbatim \
  substring of the provided text.
- Select between 5 and 20 phrases (fewer for short documents, more for long ones).
- Prefer complete sentences or meaningful clauses over single words.
- Focus on key facts, definitions, conclusions, and critical arguments.
- Do NOT include any explanations, markdown fences, or keys — just the raw \
  JSON array.

Example output:
["First key phrase.", "Second important sentence.", "A critical conclusion."]
"""

_HUMAN_PROMPT = """\
Document text:

{text}

Identify the most important phrases to highlight.
"""

_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _SYSTEM_PROMPT), ("human", _HUMAN_PROMPT)]
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def identify_highlights(
    text: str,
    openai_api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> list[str]:
    """Use a LangChain LLM agent to identify phrases worth highlighting.

    Args:
        text: Plain-text content of the PDF.
        openai_api_key: OpenAI API key.  Falls back to the ``GITHUB_TOKEN``
            environment variable when not provided.
        model: OpenAI model name.  Falls back to ``OPENAI_MODEL`` env var,
            then to ``gpt-4o-mini``.

    Returns:
        A list of verbatim text phrases that should be highlighted.

    Raises:
        ValueError: If the LLM returns a response that cannot be parsed as a
            JSON array of strings.
        RuntimeError: If no OpenAI API key is available.
    """
    settings = get_settings()

    model_name = model or settings.openai_model

    llm = ChatOpenAI(
        model=model_name,
        api_key=settings.github_token,
        base_url="https://models.inference.ai.azure.com",
        temperature=0,
    )

    chain = _PROMPT | llm

    # Truncate very long documents to avoid exceeding context windows.
    # gpt-4o-mini supports ~128k tokens; 200 k chars ≈ 50 k tokens (safe margin).
    max_chars = 200_000
    truncated_text = text[:max_chars]

    response = chain.invoke({"text": truncated_text})
    response_content = response.content
    raw = (
        response_content.strip()
        if isinstance(response_content, str)
        else json.dumps(response_content)
    )

    # Strip accidental markdown fences
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        phrases = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned a non-JSON response: {raw[:200]!r}"
        ) from exc

    if not isinstance(phrases, list) or not all(
        isinstance(p, str) for p in phrases
    ):
        raise ValueError(
            "LLM response is not a JSON array of strings.  "
            f"Got: {raw[:200]!r}"
        )

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for phrase in phrases:
        if phrase and phrase not in seen:
            seen.add(phrase)
            unique.append(phrase)

    return unique
