"""Backward-compatible settings exports.

Prefer importing from `app.config` in new code.
"""

from app.config import Settings, get_settings

__all__ = ["Settings", "get_settings"]
