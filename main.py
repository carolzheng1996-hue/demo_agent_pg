from __future__ import annotations

import argparse
import json

from config import build_api_config, ensure_directories
from global_state import GlobalState
from orchestrator import Orchestrator
from task_manager import TaskManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time-series multi-agent CLI")
    parser.add_argument("--query", required=True, help="用户任务，例如：针对etth数据集构建时序预测模型")
    parser.add_argument("--dataset-name", default="etth", help="数据集名")
    parser.add_argument("--dataset-path", default=None, help="CSV 路径，例如 data/ETTh1.csv")
    parser.add_argument("--print-state", action="store_true", help="打印完整 global state")
    parser.add_argument("--api-provider", default=None, help="LLM provider，默认 openai")
    parser.add_argument("--api-key", default=None, help="LLM API Key（可替代环境变量）")
    parser.add_argument("--api-base-url", "--api-base", dest="api_base_url", default=None, help="LLM API Base URL（可选）")
    parser.add_argument("--api-model", default=None, help="LLM 模型名，例如 gpt-4o-mini")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()

    api_config = build_api_config(
        {
            "provider": args.api_provider,
            "api_key": args.api_key,
            "base_url": args.api_base_url,
            "model": args.api_model,
        }
    )
    state = GlobalState(initial={"api_config": api_config})
    task_manager = TaskManager()
    orchestrator = Orchestrator(state=state, task_manager=task_manager)

    plan, _ = orchestrator.run(
        user_query=args.query,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
    )

    snapshot = state.snapshot()
    print("Plan:", " -> ".join(plan))
    print("Summary:\n", snapshot.get("summary_text", "(empty)"))
    if snapshot.get("report_path"):
        print("Report:", snapshot["report_path"])

    if args.print_state:
        print(json.dumps(snapshot, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
