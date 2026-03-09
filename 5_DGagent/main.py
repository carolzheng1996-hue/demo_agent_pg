from __future__ import annotations

import argparse
import json

from .config import ensure_directories
from .orchestrator import DGOrchestrator
from .state import DGGlobalState
from .task_manager import DGTaskManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DGAgent data analysis CLI")
    parser.add_argument("--query", required=True, help="用户任务描述")
    parser.add_argument("--dataset-path", required=True, help="输入数据 CSV 路径")
    parser.add_argument("--dataset-name", default="custom", help="数据集名称")
    parser.add_argument("--approve-generated-code", action="store_true", help="确认执行 data_reading subagent 生成的数据读取代码")
    parser.add_argument("--train-ratio", type=float, default=None, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=None, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=None, help="测试集比例")
    parser.add_argument("--print-state", action="store_true", help="打印完整 state")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    state = DGGlobalState(
        initial={
            "approve_generated_code": bool(args.approve_generated_code),
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
        }
    )
    task_manager = DGTaskManager()
    orchestrator = DGOrchestrator(state=state, task_manager=task_manager)
    plan, _ = orchestrator.run(user_query=args.query, dataset_path=args.dataset_path, dataset_name=args.dataset_name)
    snapshot = state.snapshot()

    print("Plan:", " -> ".join(plan))
    print("Teams:", ", ".join(snapshot.get("selected_teams", [])))
    if snapshot.get("awaiting_user_confirmation"):
        proposal = snapshot.get("data_reader_proposal", {})
        print("Awaiting confirmation for generated code.")
        print(proposal.get("generated_code", ""))
    else:
        print("Summary:\n", snapshot.get("summary_text", "(empty)"))
        if snapshot.get("report_path"):
            print("Report:", snapshot["report_path"])

    if args.print_state:
        print(json.dumps(snapshot, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
