from __future__ import annotations

import json
from typing import Any, Dict, Optional

from config import build_api_config


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


class LLMClient:
    def __init__(self, api_config: Optional[Dict[str, Optional[str]]] = None):
        cfg = build_api_config(api_config or {})
        self.provider = str(cfg.get("provider") or "openai").lower()
        self.base_url = cfg.get("base_url")
        self.api_key = cfg.get("api_key")
        self.model = cfg.get("model") or "gpt-4o-mini"
        self._client = None

    @property
    def enabled(self) -> bool:
        # Treat any OpenAI-compatible endpoint as enabled when key is provided.
        return bool(self.api_key)

    def _get_client(self):
        if not self.enabled:
            return None
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except Exception:
            return None

        kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
        return self._client

    def complete_text(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 400,
        temperature: float = 0.0,
    ) -> Optional[str]:
        client = self._get_client()
        if client is None:
            return None
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = resp.choices[0].message.content if resp.choices else None
            if isinstance(content, str):
                text = content.strip()
                return text or None
        except Exception:
            return None
        return None

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        client = self._get_client()
        if client is None:
            return None

        # First try strict JSON mode; fallback to text + parse for broader compatibility.
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content if resp.choices else None
            if isinstance(content, str):
                parsed = _parse_json_text(content)
                if parsed is not None:
                    return parsed
        except Exception:
            pass

        plain = self.complete_text(
            system_prompt=system_prompt + "\nAlways respond with a single JSON object.",
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if plain is None:
            return None
        return _parse_json_text(plain)
