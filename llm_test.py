from __future__ import annotations

import argparse
import sys
from typing import Optional

from config import build_api_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test OpenAI-compatible LLM API key/base/model")
    parser.add_argument("--api-key", default=None, help="API key (or env: API_KEY/OPENAI_API_KEY/AGENT_API_KEY)")
    parser.add_argument(
        "--api-base",
        "--api-base-url",
        dest="api_base",
        default=None,
        help="API base URL (or env: API_BASE/OPENAI_BASE_URL/OPENAI_API_BASE/AGENT_API_BASE_URL)",
    )
    parser.add_argument("--model", default=None, help="Model name (or env: AGENT_MODEL)")
    parser.add_argument("--message", default="请仅回复: OK", help="Test user message")
    parser.add_argument("--skip-model-probe", action="store_true", help="Skip /models probe")
    return parser.parse_args()


def _chat_create_compatible(client, model: str, message: str):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": message},
        ],
        "temperature": 0.0,
        "max_tokens": 64,
    }
    attempts = [dict(payload)]
    alt = dict(payload)
    alt.pop("max_tokens", None)
    alt["max_completion_tokens"] = 64
    attempts.append(alt)
    minimal = dict(alt)
    minimal.pop("temperature", None)
    attempts.append(minimal)

    last_exc: Optional[Exception] = None
    for req in attempts:
        try:
            return client.chat.completions.create(**req), None
        except Exception as exc:  # pragma: no cover
            last_exc = exc
    return None, last_exc


def run_test(api_key: str, api_base: Optional[str], model: str, message: str, trust_env_proxy: bool, skip_model_probe: bool) -> int:
    try:
        import httpx
        from openai import APIStatusError, OpenAI
    except Exception as exc:
        print(f"[FAIL] openai SDK 不可用: {exc}")
        return 2

    kwargs = {"api_key": api_key}
    if api_base:
        kwargs["base_url"] = api_base
    kwargs["http_client"] = httpx.Client(trust_env=trust_env_proxy, timeout=60.0)

    try:
        client = OpenAI(**kwargs)
        print(f"[INFO] base_url={api_base or '(default)'}")
        print(f"[INFO] model={model}")
        print(f"[INFO] trust_env_proxy={trust_env_proxy}")
        if not skip_model_probe:
            try:
                listed = client.models.list()
                model_ids = [m.id for m in getattr(listed, "data", []) if getattr(m, "id", None)]
                print(f"[INFO] models_count={len(model_ids)}")
                if model_ids and model not in model_ids:
                    print("[WARN] 当前 model 不在 /models 列表中，可能导致 400/404。")
                    print("[INFO] 示例可用模型:", ", ".join(model_ids[:8]))
            except Exception as exc:
                print(f"[WARN] /models 探测失败: {exc}")
        resp, err = _chat_create_compatible(client, model=model, message=message)
        if err is not None:
            raise err
    except Exception as exc:
        print(f"[FAIL] 请求失败: {exc}")
        try:
            if isinstance(exc, APIStatusError):
                body = exc.response.text if exc.response is not None else ""
                code = exc.status_code
                print(f"[FAIL] status_code={code}")
                if body:
                    print(f"[FAIL] response_body={body[:400]}")
        except Exception:
            pass
        print("[HINT] 若看到连接 127.0.0.1 之类错误，通常是系统代理变量导致；可在 .env 里设置 AGENT_TRUST_ENV_PROXY=false。")
        return 1

    content = ""
    if resp.choices and resp.choices[0].message and resp.choices[0].message.content:
        content = resp.choices[0].message.content.strip()
    print("[OK] API 调用成功")
    print(f"model: {resp.model}")
    print(f"reply: {content or '(empty)'}")
    return 0


def main() -> int:
    args = parse_args()
    api_cfg = build_api_config(
        {
            "api_key": args.api_key,
            "base_url": args.api_base,
            "model": args.model,
        }
    )
    api_key = api_cfg.get("api_key")
    api_base = api_cfg.get("base_url")
    model = api_cfg.get("model") or "gpt-4o-mini"
    trust_env_proxy = str(api_cfg.get("trust_env_proxy") or "false").lower() in {"1", "true", "yes", "on"}

    if not api_key:
        print("[FAIL] 缺少 API key。请通过 --api-key 或环境变量 API_KEY/OPENAI_API_KEY/AGENT_API_KEY 提供。")
        return 2

    return run_test(
        api_key=api_key,
        api_base=api_base,
        model=model,
        message=args.message,
        trust_env_proxy=trust_env_proxy,
        skip_model_probe=args.skip_model_probe,
    )


if __name__ == "__main__":
    sys.exit(main())
