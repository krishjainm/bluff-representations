"""
Shared OpenAI SDK client construction.

Supports direct OpenAI (default) and compatible proxies such as **OpenRouter**
via environment variables (no code changes needed).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import openai


def resolved_openai_base_url(api_key: Optional[str] = None) -> Optional[str]:
    explicit = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENROUTER_BASE_URL")
    if explicit:
        return explicit.strip()
    key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
    # OpenRouter keys are rejected by api.openai.com unless base_url is set.
    if key.startswith("sk-or-v1"):
        return "https://openrouter.ai/api/v1"
    return None


def openai_base_url() -> Optional[str]:
    return resolved_openai_base_url(None)


def default_chat_model_for_base(base: Optional[str]) -> str:
    """OpenRouter expects provider-prefixed IDs (e.g. openai/gpt-4o)."""
    override = os.environ.get("OPENAI_CHAT_MODEL")
    if override:
        return override.strip()
    if base and "openrouter" in base.lower():
        return "openai/gpt-4o"
    return "gpt-4o"


def default_judge_model(api_key: Optional[str] = None) -> str:
    override = os.environ.get("OPENAI_JUDGE_MODEL")
    if override:
        return override.strip()
    base = resolved_openai_base_url(api_key)
    if base and "openrouter" in base.lower():
        return "openai/gpt-4o-mini"
    return "gpt-4o-mini"


def make_openai_client(api_key: str) -> openai.OpenAI:
    kwargs: Dict[str, Any] = {"api_key": api_key}
    base = resolved_openai_base_url(api_key)
    if base:
        kwargs["base_url"] = base.rstrip("/")
        headers: Dict[str, str] = {}
        ref = os.environ.get("OPENROUTER_HTTP_REFERER")
        if ref:
            headers["HTTP-Referer"] = ref
        title = os.environ.get("OPENROUTER_APP_NAME")
        if title:
            headers["X-Title"] = title
        if headers:
            kwargs["default_headers"] = headers
    return openai.OpenAI(**kwargs)
