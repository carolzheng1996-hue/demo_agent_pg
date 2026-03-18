from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    from .agent_loop import agent_loop, run_subagent
    from .config import MAX_ITERATIONS_CAP, PLAN_FILE
    from .state import DGGlobalState
    from .subagents import SUBAGENT_REGISTRY
    from .task_manager import DGTaskManager
    from .teams import TEAM_REGISTRY
    from .tools import ensure_task_context, mean_ensemble, prepare_iteration_artifacts
except ImportError:
    from agent_loop import agent_loop, run_subagent
    from config import MAX_ITERATIONS_CAP, PLAN_FILE
    from state import DGGlobalState
    from subagents import SUBAGENT_REGISTRY
    from task_manager import DGTaskManager
    from teams import TEAM_REGISTRY
    from tools import ensure_task_context, mean_ensemble, prepare_iteration_artifacts


class DGOrchestrator:
    TASK_PLAN_FILE = "task_plan.json"
    FORECAST_STAGE_ORDER = [
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
    ]
    DATA_PROCESSING_STAGE_ORDER = [
        "data_reading",
        "data_formatter",
        "data_analysis",
        "feature_engineering",
        "split_strategy",
        "datanorm",
        "preprocess",
        "summary",
    ]
    ANALYSIS_STAGE_ORDER = [
        "data_reading",
        "data_formatter",
        "data_analysis",
        "summary",
    ]
    MODEL_STAGES = {
        "model_selection",
        "model_training",
        "model_integration",
        "evaluator",
    }

    def __init__(self, state: DGGlobalState, task_manager: DGTaskManager):
        self.state = state
        self.task_manager = task_manager

    @staticmethod
    def _analyze_intent(user_query: str) -> str:
        query = str(user_query or "").strip().lower()
        analysis_kw = ["统计", "分析", "概览", "distribution", "analysis", "summary"]
        processing_kw = ["处理", "预处理", "清洗", "格式化", "切分", "特征工程", "ods", "ds", "split", "preprocess", "feature"]
        forecast_kw = ["预测", "建模", "训练", "forecast", "model", "train"]
        if any(keyword in query for keyword in forecast_kw):
            return "build_forecast_model"
        if any(keyword in query for keyword in processing_kw):
            return "data_processing"
        if any(keyword in query for keyword in analysis_kw):
            return "analysis_only"
        return "build_forecast_model"

    @classmethod
    def _stage_order_for_intent(cls, intent: str, start_stage: str, end_stage: str) -> List[str]:
        if intent == "build_forecast_model":
            return cls.FORECAST_STAGE_ORDER
        if start_stage in {"model_selection", "model_training", "model_integration", "evaluator"} or end_stage in {
            "model_selection",
            "model_training",
            "model_integration",
            "evaluator",
        }:
            return cls.FORECAST_STAGE_ORDER
        if intent == "data_processing":
            return cls.DATA_PROCESSING_STAGE_ORDER
        return cls.ANALYSIS_STAGE_ORDER

    @classmethod
    def _slice_plan(cls, stage_order: List[str], start_stage: str, end_stage: str, skip_split: bool) -> List[str]:
        if start_stage not in stage_order or end_stage not in stage_order:
            raise ValueError(f"Unsupported stage range: {start_stage} -> {end_stage}")
        start_index = stage_order.index(start_stage)
        end_index = stage_order.index(end_stage)
        if start_index > end_index:
            raise ValueError(f"start_stage must not be after end_stage: {start_stage} -> {end_stage}")

        plan = stage_order[start_index : end_index + 1]
        if skip_split:
            plan = [step for step in plan if step not in {"split_strategy", "model_integration", "evaluator"}]
        return plan

    @classmethod
    def _canonical_plan(cls, intent: str, start_stage: str, end_stage: str, skip_split: bool) -> Tuple[List[str], Dict]:
        stage_order = cls._stage_order_for_intent(intent, start_stage, end_stage)
        plan = cls._slice_plan(stage_order, start_stage, end_stage, skip_split)
        requires_modeling = any(step in cls.MODEL_STAGES for step in plan)
        if requires_modeling:
            task_type = "forecast_modeling"
        elif any(step in plan for step in {"data_reading", "data_formatter", "data_analysis", "feature_engineering", "preprocess"}):
            task_type = "data_processing"
        else:
            task_type = "analysis_only"
        teams = [
            team_name
            for team_name, payload in TEAM_REGISTRY.items()
            if any(step in plan for step in payload["subagents"])
        ]
        return plan, {
            "task_type": task_type,
            "requires_modeling": requires_modeling,
            "requires_split": ("split_strategy" in plan) and not skip_split,
            "teams": teams,
            "reason": "deterministic_pipeline",
            "plan_source": "rule_based",
            "start_stage": start_stage,
            "end_stage": end_stage,
            "skip_split": skip_split,
        }

    @staticmethod
    def _pre_steps(plan: List[str], requires_modeling: bool) -> List[str]:
        static_steps = ["data_reading", "data_formatter", "data_analysis", "split_strategy", "datanorm"]
        if not requires_modeling:
            static_steps.extend(["feature_engineering", "preprocess"])
        return [step for step in plan if step in static_steps]

    @staticmethod
    def _iterative_steps(plan: List[str], requires_modeling: bool) -> List[str]:
        iterative = ["model_selection", "model_training", "model_integration", "evaluator", "summary"]
        if requires_modeling:
            iterative = ["feature_engineering", "preprocess"] + iterative
        return [step for step in plan if step in iterative]

    @staticmethod
    def _parse_name_list(raw: str) -> List[str]:
        return [item.strip() for item in str(raw or "").split(",") if item.strip()]

    @staticmethod
    def _resolve_ds_path(dataset_path: str) -> Path:
        candidate = Path(str(dataset_path))
        if candidate.is_file():
            return candidate
        for nested in [
            candidate / "ds_dataset.parquet",
            candidate / "data_reading_ds_dataset.parquet",
            candidate / "data" / "data_reading_ds_dataset.parquet",
        ]:
            if nested.exists():
                return nested
        raise FileNotFoundError(f"Cannot resolve ds_dataset.parquet from {dataset_path}")

    @staticmethod
    def _resolve_formatted_paths(dataset_path: str) -> Dict[str, str]:
        candidate = Path(str(dataset_path))
        if candidate.is_file():
            station_name = candidate.stem.replace("formatted_dataset_", "") or "default"
            return {station_name: str(candidate)}

        station_paths: Dict[str, str] = {}
        search_dirs = [candidate, candidate / "data" / "formatted"]
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for path in sorted(search_dir.glob("formatted_dataset_*.parquet")):
                station_name = path.stem.replace("formatted_dataset_", "")
                station_paths[station_name] = str(path)
            if station_paths:
                return station_paths
        raise FileNotFoundError(f"No formatted_dataset_*.parquet found under {dataset_path}")

    @staticmethod
    def _resolve_preprocess_paths(dataset_path: str) -> Dict[str, str]:
        candidate = Path(str(dataset_path))
        if candidate.is_file():
            raise ValueError("Preprocess bootstrap requires a directory containing train/val/test parquet files.")

        mapping = {
            "train": candidate / "train_preprocessed.parquet",
            "val": candidate / "val_preprocessed.parquet",
            "test": candidate / "test_preprocessed.parquet",
        }
        result = {name: str(path) for name, path in mapping.items() if path.exists()}
        if "train" not in result:
            raise FileNotFoundError(f"train_preprocessed.parquet not found under {dataset_path}")
        return result

    @staticmethod
    def _resolve_engineered_paths(dataset_path: str) -> Dict[str, str]:
        candidate = Path(str(dataset_path))
        if candidate.is_file():
            station_name = candidate.stem.replace("engineered_dataset_", "") or "default"
            return {station_name: str(candidate)}

        search_dirs: List[Path] = [candidate]
        iteration_dirs = sorted(
            [path for path in candidate.iterdir() if path.is_dir() and path.name.startswith("iteration_")],
            reverse=True,
        ) if candidate.exists() and candidate.is_dir() else []
        search_dirs.extend(iteration_dirs)

        engineered_paths: Dict[str, str] = {}
        for search_dir in search_dirs:
            for path in sorted(search_dir.glob("feature_engineering_engineered_dataset_*.parquet")):
                station_name = path.stem.replace("feature_engineering_engineered_dataset_", "")
                engineered_paths[station_name] = str(path)
            if engineered_paths:
                return engineered_paths
        raise FileNotFoundError(f"No engineered_dataset parquet found under {dataset_path}")

    @staticmethod
    def _resolve_split_row_ids(dataset_path: str) -> Dict[str, List[int]]:
        candidate = Path(str(dataset_path))
        if candidate.is_file():
            return {}

        split_dir_candidates = [candidate / "data" / "split", candidate / "split", candidate]
        mapping = {
            "train": "train_dataset.parquet",
            "val": "val_dataset.parquet",
            "test": "test_dataset.parquet",
        }
        for split_dir in split_dir_candidates:
            if not split_dir.exists():
                continue
            row_ids: Dict[str, List[int]] = {}
            for split_name, filename in mapping.items():
                path = split_dir / filename
                if not path.exists():
                    continue
                frame = pd.read_parquet(path)
                if "__row_id__" in frame.columns:
                    row_ids[split_name] = frame["__row_id__"].astype(int).tolist()
            if row_ids:
                return row_ids
        return {}

    @staticmethod
    def _infer_target_and_features(df: pd.DataFrame, target_hint: str) -> Tuple[str, List[str], str | None]:
        columns = df.columns.tolist()
        target = ""
        if target_hint:
            if target_hint in columns:
                target = target_hint
            else:
                matches = [column for column in columns if column.startswith(f"{target_hint}_future_step_")]
                if matches:
                    target = matches[0]
        if not target:
            future_candidates = [column for column in columns if str(column).endswith("_future_step_0")]
            target = future_candidates[0] if future_candidates else columns[-1]

        date_col = "timestamp_win" if "timestamp_win" in columns else None
        feature_cols = [
            column
            for column in df.select_dtypes(include=["number"]).columns.tolist()
            if column not in {target, "__row_id__"}
        ]
        return target, feature_cols, date_col

    def _bootstrap_data_formatter_input(self, dataset_path: str) -> None:
        ds_path = self._resolve_ds_path(dataset_path)
        self.state.write(
            "dataset_loading_result",
            {
                "dataset_path": str(Path(dataset_path)),
                "ds_dataset_path": str(ds_path),
                "selected_units": self._parse_name_list(self.state.read("formatter_unit", "")) or self._parse_name_list(self.state.read("unit", "")),
            },
        )

    def _bootstrap_formatted_profile(self, dataset_path: str) -> None:
        formatted_paths = self._resolve_formatted_paths(dataset_path)
        formatter_units = self._parse_name_list(self.state.read("formatter_unit", ""))
        if formatter_units:
            formatted_paths = {station: path for station, path in formatted_paths.items() if station in formatter_units}
            if not formatted_paths:
                raise ValueError(f"No formatted parquet matched formatter_unit={formatter_units}")

        first_path = next(iter(formatted_paths.values()))
        first_df = pd.read_parquet(Path(first_path))
        target, feature_cols, date_col = self._infer_target_and_features(first_df, str(self.state.read("target_col", "")))
        dataset_profile = {
            "dataset_path": str(Path(dataset_path)),
            "formatted_dataset_paths": formatted_paths,
            "formatted_dataset_path": first_path,
            "selected_units": list(formatted_paths.keys()),
            "shape": [sum(len(pd.read_parquet(Path(path))) for path in formatted_paths.values()), int(first_df.shape[1])],
            "columns": first_df.columns.tolist(),
            "date_column": date_col,
            "target_column": target,
            "target_columns": [target],
            "feature_columns": feature_cols,
            "input_feature_columns": feature_cols,
        }
        self.state.write("dataset_profile", dataset_profile)

    def _bootstrap_preprocess_profile(self, dataset_path: str) -> None:
        preprocess_paths = self._resolve_preprocess_paths(dataset_path)
        train_df = pd.read_parquet(Path(preprocess_paths["train"]))
        target, feature_cols, date_col = self._infer_target_and_features(train_df, str(self.state.read("target_col", "")))
        self.state.write(
            "dataset_profile",
            {
                "dataset_path": str(Path(dataset_path)),
                "date_column": date_col,
                "target_column": target,
                "target_columns": [target],
                "feature_columns": feature_cols,
                "input_feature_columns": feature_cols,
                "shape": [int(train_df.shape[0]), int(train_df.shape[1])],
                "columns": train_df.columns.tolist(),
            },
        )
        self.state.write("preprocess_result", {"dataset_paths": preprocess_paths, "target_column": target})

    def _bootstrap_engineered_profile(self, dataset_path: str) -> None:
        engineered_paths = self._resolve_engineered_paths(dataset_path)
        first_path = next(iter(engineered_paths.values()))
        first_df = pd.read_parquet(Path(first_path))
        target, feature_cols, date_col = self._infer_target_and_features(first_df, str(self.state.read("target_col", "")))
        self.state.write(
            "dataset_profile",
            {
                "dataset_path": str(Path(dataset_path)),
                "date_column": date_col,
                "target_column": target,
                "target_columns": [target],
                "feature_columns": feature_cols,
                "input_feature_columns": feature_cols,
                "shape": [int(sum(len(pd.read_parquet(Path(path))) for path in engineered_paths.values())), int(first_df.shape[1])],
                "columns": first_df.columns.tolist(),
            },
        )
        self.state.write(
            "feature_engineering_result",
            {
                "engineered_dataset_paths": engineered_paths,
                "engineered_dataset_path": first_path,
            },
        )
        row_ids = self._resolve_split_row_ids(dataset_path)
        if row_ids:
            self.state.write("split_strategy_result", {"row_ids": row_ids})

    def _bootstrap_from_stage(self, start_stage: str, dataset_path: str) -> None:
        if start_stage == "data_reading":
            return
        if start_stage == "data_formatter":
            self._bootstrap_data_formatter_input(dataset_path)
            return
        if start_stage in {"data_analysis", "feature_engineering", "split_strategy", "datanorm"}:
            self._bootstrap_formatted_profile(dataset_path)
            return
        if start_stage == "preprocess":
            self._bootstrap_engineered_profile(dataset_path)
            return
        if start_stage in {"model_selection", "model_training", "model_integration", "evaluator", "summary"}:
            self._bootstrap_preprocess_profile(dataset_path)
            return
        raise ValueError(f"Unsupported start_stage bootstrap: {start_stage}")

    def _run_model_iterations(self, plan: List[str]) -> Dict:
        iterative_steps = self._iterative_steps(plan, requires_modeling=True)
        outputs: Dict[str, Dict] = {}
        history: List[Dict] = []
        best_iteration_artifacts: List[Dict] = []
        max_iterations_value = self.state.read("max_iterations")
        if max_iterations_value is None:
            raise ValueError("max_iterations is missing in state. Please provide it in config_all.json or runtime args.")
        max_iterations = min(int(max_iterations_value), MAX_ITERATIONS_CAP)
        self.state.write("max_iterations", max_iterations)

        for iteration_index in range(1, max_iterations + 1):
            prepare_iteration_artifacts(self.state, plan, iteration_index)
            self.state.write("iteration_history", history)
            for step in iterative_steps:
                outputs[f"{step}_{iteration_index:03d}"] = run_subagent(
                    step, SUBAGENT_REGISTRY[step], self.state, self.task_manager
                )

            evaluator_result = self.state.read("evaluator_result", {})
            best_iteration_artifact = self.state.read_runtime("iteration_best_artifact")
            if best_iteration_artifact:
                best_iteration_artifacts.append(best_iteration_artifact)
            history.append(
                {
                    "iteration_index": iteration_index,
                    "best_score": evaluator_result.get("best_score"),
                    "selected_strategy": evaluator_result.get("selected_strategy"),
                    "feature_methods": self.state.read("feature_engineering_result", {}).get("selected_methods", []),
                    "report_path": self.state.read("report_path"),
                    "best_result": evaluator_result.get("iteration_best_result"),
                }
            )
            self.state.write("iteration_history", history)
            if not evaluator_result.get("should_continue", False):
                break

        cross_iteration_ensemble = self._build_cross_iteration_ensemble(best_iteration_artifacts)
        final_payload = {
            "iterations_completed": len(history),
            "iteration_history": history,
            "cross_iteration_ensemble": cross_iteration_ensemble,
            "final_report_path": self.state.read("final_report_path"),
        }
        if cross_iteration_ensemble.get("available") and self.state.read("final_report_path"):
            final_report_path = Path(str(self.state.read("final_report_path")))
            if final_report_path.exists():
                final_report_path.write_text(
                    final_report_path.read_text(encoding="utf-8")
                    + "\n\n## Cross Iteration Ensemble\n"
                    + json.dumps(cross_iteration_ensemble, ensure_ascii=False, indent=2)
                    + "\n",
                    encoding="utf-8",
                )
        return final_payload

    def _build_cross_iteration_ensemble(self, best_iteration_artifacts: List[Dict]) -> Dict:
        if not best_iteration_artifacts:
            result: Dict = {"available": False, "reason": "no_iteration_best_models"}
            self.state.write("cross_iteration_ensemble_result", result)
            return result

        test_target = self.state.read_runtime("test_target")
        if test_target is None:
            result = {"available": False, "reason": "missing_test_target"}
            self.state.write("cross_iteration_ensemble_result", result)
            return result

        ensemble_result = mean_ensemble(best_iteration_artifacts, test_target)
        payload = {
            "available": True,
            "strategy": "cross_iteration_best_model_average",
            "iteration_count": len(best_iteration_artifacts),
            "member_iterations": [item.get("iteration_index") for item in best_iteration_artifacts],
            "member_strategies": [item.get("strategy") for item in best_iteration_artifacts],
            "member_names": [item.get("name") for item in best_iteration_artifacts],
            "metrics": ensemble_result.get("metrics", {}),
            "predictions": ensemble_result.get("predictions", []),
            "member_count": ensemble_result.get("member_count", len(best_iteration_artifacts)),
        }
        self.state.write("cross_iteration_ensemble_result", payload)
        return payload

    def _ensure_final_summary(self, outputs: Dict[str, Dict]) -> None:
        final_report_path = self.state.read("final_report_path")
        if final_report_path and Path(str(final_report_path)).exists():
            return
        outputs["summary"] = run_subagent("summary", SUBAGENT_REGISTRY["summary"], self.state, self.task_manager)

    def _persist_task_plan(self, plan: List[str], plan_meta: Dict, dataset_path: str, dataset_name: str) -> None:
        task_dir_value = self.state.read("task_dir")
        if not task_dir_value:
            return
        payload = {
            "plan": plan,
            "plan_meta": plan_meta,
            "dataset_path": dataset_path,
            "dataset_name": dataset_name,
            "user_query": self.state.read("user_query"),
        }
        Path(str(task_dir_value)).joinpath(self.TASK_PLAN_FILE).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def run(self, user_query: str, dataset_path: str, dataset_name: str = "custom") -> Tuple[List[str], Dict]:
        intent = self._analyze_intent(user_query)
        start_stage = str(self.state.read("start_stage") or "").strip()
        end_stage = str(self.state.read("end_stage") or "").strip()
        if not start_stage or not end_stage:
            raise ValueError("start_stage/end_stage is missing in state. Please provide them in config_all.json or runtime args.")
        skip_split_value = self.state.read("skip_split")
        if skip_split_value is None:
            raise ValueError("skip_split is missing in state. Please provide it in config_all.json or runtime args.")
        skip_split = bool(skip_split_value)
        plan, plan_meta = self._canonical_plan(intent, start_stage, end_stage, skip_split)
        self.state.update(
            {
                "user_query": user_query,
                "dataset_name": dataset_name,
                "dataset_path": dataset_path,
                "plan": plan,
                "plan_meta": plan_meta,
                "selected_teams": plan_meta.get("teams", []),
            }
        )
        self._bootstrap_from_stage(start_stage, dataset_path)
        ensure_task_context(self.state)
        self._persist_task_plan(plan, plan_meta, dataset_path, dataset_name)
        PLAN_FILE.write_text(json.dumps({"plan": plan, "plan_meta": plan_meta}, ensure_ascii=False, indent=2), encoding="utf-8")
        self.task_manager.set_plan(plan)
        self.state.write("tasks", self.task_manager.list_tasks())

        prepare_iteration_artifacts(self.state, plan, 1)
        pre_steps = self._pre_steps(plan, requires_modeling=bool(plan_meta.get("requires_modeling")))
        outputs = agent_loop(pre_steps, SUBAGENT_REGISTRY, self.state, self.task_manager)

        if plan_meta.get("requires_modeling") and any(step in plan for step in {"model_selection", "model_training", "model_integration", "evaluator"}):
            outputs["iterations"] = self._run_model_iterations(plan)
        else:
            outputs.update(agent_loop([step for step in plan if step not in pre_steps], SUBAGENT_REGISTRY, self.state, self.task_manager))

        self._ensure_final_summary(outputs)

        return plan, outputs
