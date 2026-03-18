from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .config import (
        build_runtime_state,
        CONFIG_ALL_FILE,
        ensure_directories,
        runtime_defaults,
    )
    from .orchestrator import DGOrchestrator
    from .state import DGGlobalState
    from .task_manager import DGTaskManager
except ImportError:
    from config import build_runtime_state, CONFIG_ALL_FILE, ensure_directories, runtime_defaults
    from orchestrator import DGOrchestrator
    from state import DGGlobalState
    from task_manager import DGTaskManager


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config-all", default=str(CONFIG_ALL_FILE))
    pre_args, _ = pre_parser.parse_known_args()
    defaults = runtime_defaults(Path(str(pre_args.config_all)).expanduser())
    parser = argparse.ArgumentParser(description="Deterministic time-series forecasting pipeline")
    parser.add_argument("--config-all", default=str(CONFIG_ALL_FILE), help="统一运行配置 JSON 路径")
    parser.add_argument("--query", required=True, help="用户任务描述")
    parser.add_argument("--dataset-path", required=True, help="输入按 station=<unit> 分区的目录路径")
    parser.add_argument("--dataset-name", default=defaults["dataset_name"], help="数据集名称")
    parser.add_argument("--unit", default=defaults["unit"], help="站点 ID，多个站点用逗号分隔；为空时默认读取目录下全部站点")
    parser.add_argument(
        "--formatter-unit",
        default=defaults["formatter_unit"],
        help="data_formatter 阶段要处理的站点 ID，多个站点用逗号分隔；为空时默认使用前序阶段全部站点",
    )
    parser.add_argument(
        "--start-stage",
        default=defaults["start_stage"],
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
        default=defaults["end_stage"],
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
    parser.add_argument("--enable-split", dest="enable_split", action="store_true", default=bool(defaults["enable_split"]), help="启用训练/验证切分")
    parser.add_argument("--disable-split", dest="enable_split", action="store_false", help="关闭训练/验证切分")
    parser.add_argument("--enable-feature-engineering", dest="enable_feature_engineering", action="store_true", default=bool(defaults["enable_feature_engineering"]), help="启用特征工程")
    parser.add_argument("--disable-feature-engineering", dest="enable_feature_engineering", action="store_false", help="关闭特征工程，预处理将直接使用 formatted 数据")
    parser.add_argument("--target-col", default=defaults["target_col"], help="显式指定目标列，多个列用逗号分隔")
    parser.add_argument("--input-feature-cols", default=defaults["input_feature_cols"], help="显式指定模型输入列，多个列用逗号分隔")
    parser.add_argument(
        "--split-method",
        default=defaults["split_method"],
        choices=["station_last_k", "station_month_last_k", "global_last_k", "fixed_date", "leave_stations_out"],
        help="数据集切分方法",
    )
    parser.add_argument("--split-cutoff-date", default=defaults["split_cutoff_date"], help="固定日期切分时使用，格式如 2023-03-01")
    parser.add_argument("--split-test-units", default=defaults["split_test_units"], help="留站切分时使用，多个站点用逗号分隔")
    parser.add_argument("--train-ratio", type=float, default=defaults["train_ratio"], help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=defaults["val_ratio"], help="验证集比例")
    parser.add_argument("--input-length", type=int, default=defaults["input_length"], help="输入窗口长度")
    parser.add_argument("--output-length", type=int, default=defaults["output_length"], help="输出窗口长度")
    parser.add_argument("--points-per-day", type=int, default=defaults["points_per_day"], help="每天的采样点数，用于 DS 序列缺失填补")
    parser.add_argument("--enable-normalization", dest="enable_normalization", action="store_true", default=bool(defaults["enable_normalization"]), help="启用标准化")
    parser.add_argument("--disable-normalization", dest="enable_normalization", action="store_false", help="关闭标准化，仅做预处理清洗")
    parser.add_argument(
        "--normalization-policy",
        choices=["auto", "off", "zscore", "minmax"],
        default=defaults["normalization_policy"],
        help="标准化策略：自动判断/关闭/zscore/minmax",
    )
    parser.add_argument("--use-system-random", dest="use_system_random", action="store_true", default=bool(defaults["use_system_random"]), help="特征工程迭代时使用 SystemRandom 随机选择策略")
    parser.add_argument("--disable-system-random", dest="use_system_random", action="store_false", help="关闭 SystemRandom，改为确定性种子策略")
    parser.add_argument("--max-iterations", type=int, default=defaults["max_iterations"], help="自动优化最大轮数，最大不超过10")
    parser.add_argument("--print-state", action="store_true", help="打印完整 state")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    config_path = Path(str(args.config_all)).expanduser()
    initial_state = build_runtime_state(
        {
            "dataset_name": args.dataset_name,
            "unit": args.unit,
            "formatter_unit": args.formatter_unit,
            "start_stage": args.start_stage,
            "end_stage": args.end_stage,
            "enable_split": args.enable_split,
            "skip_split": not args.enable_split,
            "enable_feature_engineering": args.enable_feature_engineering,
            "target_col": args.target_col,
            "input_feature_cols": args.input_feature_cols,
            "split_method": args.split_method,
            "split_cutoff_date": args.split_cutoff_date,
            "split_test_units": args.split_test_units,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "input_length": args.input_length,
            "output_length": args.output_length,
            "points_per_day": args.points_per_day,
            "enable_normalization": args.enable_normalization,
            "normalization_policy": args.normalization_policy,
            "use_system_random": args.use_system_random,
            "max_iterations": args.max_iterations,
        },
        config_path=config_path,
    )
    initial_state.update(
        {
            "runtime_config_path": str(config_path),
            "runtime_config": runtime_defaults(config_path),
        }
    )
    state = DGGlobalState(
        load_existing=False,
        initial=initial_state,
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
