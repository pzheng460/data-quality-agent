"""Shared LLM client for all API-based modules.

Provides a singleton OpenAI-compatible client configured via (in priority order):
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
_yaml_config: dict[str, Any] | None = None


def set_config_from_yaml(llm_config) -> None:
    """Store LLM config loaded from YAML for later use.

    Args:
        llm_config: LLMConfig dataclass instance from config.py
    """
    global _yaml_config
    _yaml_config = {
        "api_url": llm_config.api_url,
        "api_key": llm_config.api_key,
        "model": llm_config.model,
    }


def get_llm_config() -> dict[str, str | None]:
    """Get LLM config from YAML → env vars (in priority order)."""
    yaml_url = _yaml_config.get("api_url") if _yaml_config else None
    yaml_key = _yaml_config.get("api_key") if _yaml_config else None
    yaml_model = _yaml_config.get("model") if _yaml_config else None

    return {
        "api_url": yaml_url or os.environ.get("DQ_API_BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
        "api_key": yaml_key or os.environ.get("DQ_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        "model": yaml_model or os.environ.get("DQ_MODEL"),
    }


def get_client(
    api_url: str | None = None,
    api_key: str | None = None,
) -> Any:
    """Get or create a shared OpenAI client.

    Priority: explicit params > YAML config > env vars.

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
        logger.warning("No API key configured. Set DQ_API_KEY or OPENAI_API_KEY env var, or add llm.api_key in YAML config.")
        return None

    # Reuse existing client if params match
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
    """Get default model name from explicit config / YAML / env / fallback."""
    env = get_llm_config()
    return env["model"] or "gpt-4o-mini"


def reset_client():
    """Reset the singleton client and YAML config (for testing)."""
    global _client_instance, _yaml_config
    _client_instance = None
    _yaml_config = None
