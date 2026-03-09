from __future__ import annotations

import argparse
import sys
from importlib import import_module
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test project LLM bootstrap via 5_DGagent/LLM")
    parser.add_argument("--message", default="请仅回复: OK", help="Test user message")
    return parser.parse_args()

def run_test(message: str) -> int:
    try:
        get_llm = import_module("5_DGagent.LLM").get_llm
    except Exception as exc:
        print(f"[FAIL] 无法加载 5_DGagent/LLM: {exc}")
        return 2

    try:
        use_outside = bool(os.getenv("OUT_OPENAI_API_KEY") or os.getenv("OUT_OPENAI_API_BASE"))
        model = get_llm(is_outside=use_outside)
        resp = model.invoke(message)
    except Exception as exc:
        print(f"[FAIL] 请求失败: {exc}")
        print("[HINT] 请检查 5_DGagent/LLM/.env、5_DGagent/.env 或仓库根目录 .env 是否提供了可用模型配置。")
        return 1

    content = getattr(resp, "content", None) or str(resp)
    print("[OK] LLM 调用成功")
    print(f"reply: {str(content).strip() or '(empty)'}")
    return 0


def main() -> int:
    args = parse_args()
    return run_test(message=args.message)


if __name__ == "__main__":
    sys.exit(main())
