"""Shared LLM client for all API-based modules.

Provides a singleton OpenAI-compatible client configured via:
1. Explicit params (api_url, api_key, model)
2. YAML config (pipeline.llm section)
3. Environment variables (DQ_API_BASE_URL, DQ_API_KEY, DQ_MODEL)
4. Fallback env vars (OPENAI_BASE_URL, OPENAI_API_KEY)
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_client_instance = None
_default_model: str | None = None


def get_llm_config() -> dict[str, str | None]:
    """Get LLM config from environment variables."""
    return {
        "api_url": os.environ.get("DQ_API_BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
        "api_key": os.environ.get("DQ_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        "model": os.environ.get("DQ_MODEL"),
    }


def get_client(
    api_url: str | None = None,
    api_key: str | None = None,
) -> Any:
    """Get or create a shared OpenAI client.

    Args:
        api_url: Override base URL. Falls back to env vars.
        api_key: Override API key. Falls back to env vars.

    Returns:
        openai.OpenAI client instance, or None if openai not installed.
    """
    global _client_instance

    try:
        import openai
    except ImportError:
        logger.warning("openai not installed. Install with: uv add openai")
        return None

    env = get_llm_config()
    url = api_url or env["api_url"]
    key = api_key or env["api_key"]

    if not key:
        logger.warning("No API key configured. Set DQ_API_KEY or OPENAI_API_KEY env var.")
        return None

    # Create new client if params differ or no client exists
    if _client_instance is not None:
        if (getattr(_client_instance, "_custom_url", None) == url and
                getattr(_client_instance, "_custom_key", None) == key):
            return _client_instance

    kwargs: dict[str, Any] = {"api_key": key}
    if url:
        kwargs["base_url"] = url

    _client_instance = openai.OpenAI(**kwargs)
    _client_instance._custom_url = url  # type: ignore
    _client_instance._custom_key = key  # type: ignore
    logger.info("Created LLM client: %s", url or "default OpenAI")
    return _client_instance


def get_default_model() -> str:
    """Get default model name from env or fallback."""
    env = get_llm_config()
    return env["model"] or "gpt-4o-mini"


def reset_client():
    """Reset the singleton client (for testing)."""
    global _client_instance
    _client_instance = None
