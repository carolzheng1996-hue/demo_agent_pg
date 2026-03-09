from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

try:
    from .LLM import get_llm
except ImportError:
    from LLM import get_llm


def _should_use_outside() -> bool:
    return bool(os.getenv("OUT_OPENAI_API_KEY") or os.getenv("OUT_OPENAI_API_BASE"))


def _extract_text(response: Any) -> Optional[str]:
    if response is None:
        return None
    content = getattr(response, "content", None)
    if isinstance(content, str):
        text = content.strip()
        return text or None
    if isinstance(response, str):
        text = response.strip()
        return text or None
    return None


def _parse_json_text(raw: str) -> Optional[Dict[str, Any]]:
    text = raw.strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def invoke_text(
    system_prompt: str,
    user_prompt: str,
    model_name: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> Optional[str]:
    try:
        model = get_llm(model_name or "gpt-oss-120b", is_outside=_should_use_outside())
        response = model.invoke(f"{system_prompt}\n\n{user_prompt}")
        return _extract_text(response)
    except Exception:
        return None


def invoke_json(
    system_prompt: str,
    user_prompt: str,
    model_name: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    try:
        model = get_llm(model_name or "gpt-oss-120b", is_outside=_should_use_outside())
        response = model.invoke(
            f"{system_prompt}\n\nReturn exactly one JSON object.\n\n{user_prompt}"
        )
        text = _extract_text(response)
        if text is None:
            return None
        return _parse_json_text(text)
    except Exception:
        return None
