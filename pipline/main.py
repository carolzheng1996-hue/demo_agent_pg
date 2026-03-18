from __future__ import annotations

import argparse
import json

try:
    from .config import DEFAULT_MAX_ITERATIONS, MAX_ITERATIONS_CAP, ensure_directories
    from .orchestrator import DGOrchestrator
    from .state import DGGlobalState
    from .task_manager import DGTaskManager
except ImportError:
    from config import DEFAULT_MAX_ITERATIONS, MAX_ITERATIONS_CAP, ensure_directories
    from orchestrator import DGOrchestrator
    from state import DGGlobalState
    from task_manager import DGTaskManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic time-series forecasting pipeline")
    parser.add_argument("--query", required=True, help="用户任务描述")
    parser.add_argument("--dataset-path", required=True, help="输入按 station=<unit> 分区的目录路径")
    parser.add_argument("--dataset-name", default="custom", help="数据集名称")
    parser.add_argument("--unit", default="", help="站点 ID，多个站点用逗号分隔；为空时默认读取目录下全部站点")
    parser.add_argument(
        "--formatter-unit",
        default="",
        help="data_formatter 阶段要处理的站点 ID，多个站点用逗号分隔；为空时默认使用前序阶段全部站点",
    )
    parser.add_argument(
        "--start-stage",
        default="data_reading",
        choices=[
            "data_reading",
            "data_formatter",
            "data_analysis",
            "feature_engineering",
            "split_strategy",
            "datanorm",
            "preprocess",
            "model_selection",
            "model_training",
            "model_integration",
            "evaluator",
            "summary",
        ],
        help="流程起始阶段",
    )
    parser.add_argument(
        "--end-stage",
        default="summary",
        choices=[
            "data_reading",
            "data_formatter",
            "data_analysis",
            "feature_engineering",
            "split_strategy",
            "datanorm",
            "preprocess",
            "model_selection",
            "model_training",
            "model_integration",
            "evaluator",
            "summary",
        ],
        help="流程结束阶段",
    )
    parser.add_argument("--skip-split", action="store_true", help="跳过训练/验证/测试切分")
    parser.add_argument("--target-col", default="", help="显式指定目标列，多个列用逗号分隔")
    parser.add_argument("--input-feature-cols", default="", help="显式指定模型输入列，多个列用逗号分隔")
    parser.add_argument(
        "--split-method",
        default="global_last_k",
        choices=["station_last_k", "station_month_last_k", "global_last_k", "fixed_date", "leave_stations_out"],
        help="数据集切分方法",
    )
    parser.add_argument("--split-cutoff-date", default="", help="固定日期切分时使用，格式如 2023-03-01")
    parser.add_argument("--split-test-units", default="", help="留站切分时使用，多个站点用逗号分隔")
    parser.add_argument("--train-ratio", type=float, default=None, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=None, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=None, help="测试集比例")
    parser.add_argument("--input-length", type=int, default=None, help="输入窗口长度")
    parser.add_argument("--output-length", type=int, default=None, help="输出窗口长度")
    parser.add_argument("--time-increment", type=int, default=None, help="滑窗步长")
    parser.add_argument(
        "--normalization-policy",
        choices=["auto", "off", "zscore", "minmax"],
        default="auto",
        help="标准化策略：自动判断/关闭/zscore/minmax",
    )
    parser.add_argument("--use-system-random", dest="use_system_random", action="store_true", default=True, help="特征工程迭代时使用 SystemRandom 随机选择策略")
    parser.add_argument("--disable-system-random", dest="use_system_random", action="store_false", help="关闭 SystemRandom，改为确定性种子策略")
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS, help="自动优化最大轮数，最大不超过10")
    parser.add_argument("--print-state", action="store_true", help="打印完整 state")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    state = DGGlobalState(
        load_existing=False,
        initial={
            "unit": str(args.unit or "").strip(),
            "formatter_unit": str(args.formatter_unit or "").strip(),
            "start_stage": str(args.start_stage or "data_reading").strip(),
            "end_stage": str(args.end_stage or "summary").strip(),
            "skip_split": bool(args.skip_split),
            "target_col": str(args.target_col or "").strip(),
            "input_feature_cols": str(args.input_feature_cols or "").strip(),
            "split_method": str(args.split_method or "global_last_k").strip(),
            "split_cutoff_date": str(args.split_cutoff_date or "").strip(),
            "split_test_units": str(args.split_test_units or "").strip(),
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "input_length": args.input_length,
            "output_length": args.output_length,
            "time_increment": args.time_increment,
            "normalization_policy": args.normalization_policy,
            "use_system_random": bool(args.use_system_random),
            "max_iterations": max(1, min(int(args.max_iterations), MAX_ITERATIONS_CAP)),
        },
    )
    task_manager = DGTaskManager()
    orchestrator = DGOrchestrator(state=state, task_manager=task_manager)
    plan, _ = orchestrator.run(
        user_query=args.query,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
    )
    snapshot = state.snapshot()

    print("Plan:", " -> ".join(plan))
    print("Teams:", ", ".join(snapshot.get("selected_teams", [])))
    print("Summary:\n", snapshot.get("summary_text", "(empty)"))
    if snapshot.get("report_path"):
        print("Report:", snapshot["report_path"])

    if args.print_state:
        print(json.dumps(snapshot, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
