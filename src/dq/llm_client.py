"""Shared LLM client for all API-based modules.

Supports two backends:
- "anthropic" (default): Anthropic Messages API (/v1/messages)
- "openai": OpenAI-compatible Chat Completions API (/v1/chat/completions)

Configuration priority: CLI args > configs/llm.yaml > env vars > defaults.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_client_instance = None
_yaml_config: dict[str, Any] | None = None


def set_config_from_yaml(llm_config) -> None:
    """Store LLM config loaded from YAML for later use."""
    global _yaml_config
    _yaml_config = {
        "api_url": llm_config.api_url,
        "api_key": llm_config.api_key,
        "model": llm_config.model,
        "backend": getattr(llm_config, "backend", "anthropic"),
    }


def get_llm_config() -> dict[str, str | None]:
    """Get LLM config from YAML → env vars (in priority order)."""
    yaml_url = _yaml_config.get("api_url") if _yaml_config else None
    yaml_key = _yaml_config.get("api_key") if _yaml_config else None
    yaml_model = _yaml_config.get("model") if _yaml_config else None
    yaml_backend = _yaml_config.get("backend") if _yaml_config else None

    return {
        "api_url": yaml_url or os.environ.get("DQ_API_BASE_URL") or os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
        "api_key": yaml_key or os.environ.get("DQ_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        "model": yaml_model or os.environ.get("DQ_MODEL"),
        "backend": yaml_backend or os.environ.get("DQ_LLM_BACKEND") or "anthropic",
    }


def get_client(
    api_url: str | None = None,
    api_key: str | None = None,
) -> Any:
    """Get or create a shared LLM client.

    Returns an AnthropicClient or OpenAIClient wrapper depending on backend config.
    """
    global _client_instance

    env = get_llm_config()
    url = api_url or env["api_url"]
    key = api_key or env["api_key"]
    backend = env["backend"]

    if not key:
        logger.warning("No API key configured. Set api_key in configs/llm.yaml, or DQ_API_KEY env var.")
        return None

    # Reuse existing client if params match
    if _client_instance is not None:
        if (getattr(_client_instance, "_custom_url", None) == url and
                getattr(_client_instance, "_custom_key", None) == key and
                getattr(_client_instance, "_custom_backend", None) == backend):
            return _client_instance

    if backend == "openai":
        _client_instance = _create_openai_client(url, key)
    else:
        _client_instance = _create_anthropic_client(url, key)

    if _client_instance is not None:
        _client_instance._custom_url = url  # type: ignore
        _client_instance._custom_key = key  # type: ignore
        _client_instance._custom_backend = backend  # type: ignore
        logger.info("Created %s LLM client: %s", backend, url or "default")

    return _client_instance


def _create_anthropic_client(url: str | None, key: str) -> Any:
    """Create Anthropic client."""
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic not installed. Install with: uv add anthropic")
        return None

    kwargs: dict[str, Any] = {"api_key": key}
    if url:
        kwargs["base_url"] = url
    return anthropic.Anthropic(**kwargs)


def _create_openai_client(url: str | None, key: str) -> Any:
    """Create OpenAI-compatible client."""
    try:
        import openai
    except ImportError:
        logger.warning("openai not installed. Install with: uv add openai")
        return None

    kwargs: dict[str, Any] = {"api_key": key}
    if url:
        kwargs["base_url"] = url
    return openai.OpenAI(**kwargs)


def get_backend() -> str:
    """Get current backend type ('anthropic' or 'openai')."""
    env = get_llm_config()
    return env["backend"] or "anthropic"


def get_default_model() -> str:
    """Get default model name."""
    env = get_llm_config()
    if env["model"]:
        return env["model"]
    backend = env["backend"] or "anthropic"
    if backend == "anthropic":
        return "claude-sonnet-4-20250514"
    return "gpt-4o-mini"


def reset_client():
    """Reset the singleton client and YAML config (for testing)."""
    global _client_instance, _yaml_config
    _client_instance = None
    _yaml_config = None
