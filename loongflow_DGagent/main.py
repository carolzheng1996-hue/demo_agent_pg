from __future__ import annotations

import argparse
import json
from typing import Dict, Optional

try:
    from .config import DEFAULT_MAX_ITERATIONS, MAX_ITERATIONS_CAP, ensure_directories
    from .orchestrator import DGOrchestrator
    from .state import DGGlobalState
    from .task_manager import DGTaskManager
    from .tools import regenerate_code_with_feedback
except ImportError:
    from config import DEFAULT_MAX_ITERATIONS, MAX_ITERATIONS_CAP, ensure_directories
    from orchestrator import DGOrchestrator
    from state import DGGlobalState
    from task_manager import DGTaskManager
    from tools import regenerate_code_with_feedback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DGAgent data analysis CLI")
    parser.add_argument("--query", required=True, help="用户任务描述")
    parser.add_argument("--dataset-path", required=True, help="输入数据 CSV 路径")
    parser.add_argument("--dataset-name", default="custom", help="数据集名称")
    parser.add_argument("--target-col", default="", help="显式指定目标列，多个列用逗号分隔")
    parser.add_argument("--approve-generated-code", action="store_true", help="确认执行 data_formatter subagent 生成的数据处理代码")
    parser.add_argument("--train-ratio", type=float, default=None, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=None, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=None, help="测试集比例")
    parser.add_argument("--use-custom-processing", action="store_true", help="启用自定义数据处理说明")
    parser.add_argument("--custom-processing-steps", default="", help="用户自定义数据处理步骤描述")
    parser.add_argument("--input-length", type=int, default=None, help="输入窗口长度")
    parser.add_argument("--output-length", type=int, default=None, help="输出窗口长度")
    parser.add_argument("--time-increment", type=int, default=None, help="滑窗步长")
    parser.add_argument(
        "--normalization-policy",
        choices=["auto", "force_on", "force_off"],
        default="auto",
        help="标准化策略：自动判断/强制开启/强制关闭",
    )
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS, help="自动优化最大轮数，最大不超过10")
    parser.add_argument("--print-state", action="store_true", help="打印完整 state")
    return parser.parse_args()


def _approval_proposal(state: DGGlobalState, subagent: str) -> Dict:
    key_map = {
        "data_formatter": "data_formatter_proposal",
        "split_strategy": "split_strategy_proposal",
    }
    proposal_key = key_map.get(subagent, f"{subagent}_proposal")
    proposal = state.read(proposal_key, {})
    return proposal if isinstance(proposal, dict) else {}


def _prompt_user_confirmation(subagent: str, proposal: Dict) -> str:
    print(f"Awaiting confirmation for generated code from `{subagent}`.")
    print(proposal.get("generated_code", ""))
    while True:
        answer = input("Choose [y] confirm, [m] modify, [n] cancel: ").strip().lower()
        if answer in {"y", "yes"}:
            return "confirm"
        if answer in {"m", "modify"}:
            return "modify"
        if answer in {"", "n", "no"}:
            return "cancel"
        print("Please answer with 'y', 'm', or 'n'.")


def _run_with_confirmation_loop(
    orchestrator: DGOrchestrator,
    state: DGGlobalState,
    query: str,
    dataset_path: str,
    dataset_name: str,
) -> tuple[list[str], Optional[Dict]]:
    outputs: Optional[Dict] = None
    plan: list[str] = []
    while True:
        plan, outputs = orchestrator.run(user_query=query, dataset_path=dataset_path, dataset_name=dataset_name)
        snapshot = state.snapshot()
        awaiting = snapshot.get("awaiting_user_confirmation")
        if not awaiting:
            return plan, outputs

        subagent = str(awaiting.get("subagent") or "")
        proposal = _approval_proposal(state, subagent)
        decision = _prompt_user_confirmation(subagent, proposal)
        if decision == "cancel":
            print("Execution stopped because generated code was not approved.")
            return plan, outputs
        if decision == "modify":
            feedback = input("Describe what is wrong with the generated code: ").strip() or "The generated code needs correction."
            rewritten = regenerate_code_with_feedback(
                subagent=subagent,
                original_code=str(proposal.get("generated_code", "")),
                feedback=feedback,
                context=proposal.get("context", {}) if isinstance(proposal.get("context"), dict) else {},
            )
            if not rewritten:
                print("Failed to regenerate code from feedback. Keeping the current proposal.")
                continue
            proposal["generated_code"] = rewritten
            feedback_history = list(proposal.get("feedback_history", []))
            feedback_history.append({"feedback": feedback, "updated_at": "cli"})
            proposal["feedback_history"] = feedback_history
            state.write(f"{subagent}_proposal", proposal)
            modified_proposals = dict(state.read("modified_proposals", {}) or {})
            modified_proposals[subagent] = rewritten
            state.write("modified_proposals", modified_proposals)
            continue

        approved_codes = dict(state.read("approved_generated_codes", {}) or {})
        approved_codes[subagent] = proposal.get("generated_code", "")
        state.update(
            {
                "approved_generated_codes": approved_codes,
                "awaiting_user_confirmation": None,
            }
        )


def main() -> None:
    args = parse_args()
    ensure_directories()
    state = DGGlobalState(
        load_existing=False,
        initial={
            "approve_generated_code": bool(args.approve_generated_code),
            "target_col": str(args.target_col or "").strip(),
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "use_custom_processing": bool(args.use_custom_processing),
            "custom_processing_steps": str(args.custom_processing_steps or "").strip(),
            "input_length": args.input_length,
            "output_length": args.output_length,
            "time_increment": args.time_increment,
            "normalization_policy": args.normalization_policy,
            "max_iterations": max(1, min(int(args.max_iterations), MAX_ITERATIONS_CAP)),
        }
    )
    task_manager = DGTaskManager()
    orchestrator = DGOrchestrator(state=state, task_manager=task_manager)
    plan, _ = _run_with_confirmation_loop(
        orchestrator=orchestrator,
        state=state,
        query=args.query,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
    )
    snapshot = state.snapshot()

    print("Plan:", " -> ".join(plan))
    print("Teams:", ", ".join(snapshot.get("selected_teams", [])))
    if snapshot.get("awaiting_user_confirmation"):
        subagent = str(snapshot.get("awaiting_user_confirmation", {}).get("subagent") or "")
        proposal = _approval_proposal(state, subagent)
        print(f"Awaiting confirmation for generated code from `{subagent}`.")
        print(proposal.get("generated_code", ""))
    else:
        print("Summary:\n", snapshot.get("summary_text", "(empty)"))
        if snapshot.get("report_path"):
            print("Report:", snapshot["report_path"])

    if args.print_state:
        print(json.dumps(snapshot, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
