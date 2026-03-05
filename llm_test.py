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
    return parser.parse_args()


def run_test(api_key: str, api_base: Optional[str], model: str, message: str) -> int:
    try:
        from openai import OpenAI
    except Exception as exc:
        print(f"[FAIL] openai SDK 不可用: {exc}")
        return 2

    kwargs = {"api_key": api_key}
    if api_base:
        kwargs["base_url"] = api_base

    try:
        client = OpenAI(**kwargs)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": message},
            ],
            temperature=0.0,
            max_tokens=64,
        )
    except Exception as exc:
        print(f"[FAIL] 请求失败: {exc}")
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

    if not api_key:
        print("[FAIL] 缺少 API key。请通过 --api-key 或环境变量 API_KEY/OPENAI_API_KEY/AGENT_API_KEY 提供。")
        return 2

    return run_test(api_key=api_key, api_base=api_base, model=model, message=args.message)


if __name__ == "__main__":
    sys.exit(main())
