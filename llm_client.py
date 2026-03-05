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
        self.trust_env_proxy = str(cfg.get("trust_env_proxy") or "false").lower() in {"1", "true", "yes", "on"}
        self.timeout_seconds = float(cfg.get("timeout_seconds") or 60.0)
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
        try:
            import httpx

            kwargs["http_client"] = httpx.Client(trust_env=self.trust_env_proxy, timeout=self.timeout_seconds)
        except Exception:
            pass
        self._client = OpenAI(**kwargs)
        return self._client

    def _chat_create(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        response_format: Optional[Dict[str, str]] = None,
    ):
        client = self._get_client()
        if client is None:
            return None

        req: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            req["response_format"] = response_format

        attempts: list[Dict[str, Any]] = [dict(req)]

        # Some OpenAI-compatible gateways prefer max_completion_tokens.
        req_alt = dict(req)
        req_alt.pop("max_tokens", None)
        req_alt["max_completion_tokens"] = max_tokens
        attempts.append(req_alt)

        # Some gateways reject response_format or temperature; keep a minimal fallback.
        req_min = dict(req_alt)
        req_min.pop("response_format", None)
        req_min.pop("temperature", None)
        attempts.append(req_min)

        for payload in attempts:
            try:
                return client.chat.completions.create(**payload)
            except Exception:
                continue
        return None

    def complete_text(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 400,
        temperature: float = 0.0,
    ) -> Optional[str]:
        resp = self._chat_create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if resp is None:
            return None
        content = resp.choices[0].message.content if resp.choices else None
        if isinstance(content, str):
            text = content.strip()
            return text or None
        return None

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        # First try strict JSON mode; fallback to text + parse for broader compatibility.
        resp = self._chat_create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        if resp is not None:
            content = resp.choices[0].message.content if resp.choices else None
            if isinstance(content, str):
                parsed = _parse_json_text(content)
                if parsed is not None:
                    return parsed

        plain = self.complete_text(
            system_prompt=system_prompt + "\nAlways respond with a single JSON object.",
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if plain is None:
            return None
        return _parse_json_text(plain)
