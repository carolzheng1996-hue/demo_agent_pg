from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from .agent_loop import agent_loop, run_subagent
    from .config import DEFAULT_MAX_ITERATIONS, MAX_ITERATIONS_CAP, PLAN_FILE
    from .state import DGGlobalState
    from .subagents import SUBAGENT_REGISTRY
    from .task_manager import DGTaskManager
    from .teams import TEAM_REGISTRY
    from .tools import ensure_task_context, mean_ensemble, prepare_iteration_artifacts, write_task_text_artifact
except ImportError:
    from agent_loop import agent_loop, run_subagent
    from config import DEFAULT_MAX_ITERATIONS, MAX_ITERATIONS_CAP, PLAN_FILE
    from state import DGGlobalState
    from subagents import SUBAGENT_REGISTRY
    from task_manager import DGTaskManager
    from teams import TEAM_REGISTRY
    from tools import ensure_task_context, mean_ensemble, prepare_iteration_artifacts, write_task_text_artifact


class DGOrchestrator:
    def __init__(self, state: DGGlobalState, task_manager: DGTaskManager):
        self.state = state
        self.task_manager = task_manager

    @staticmethod
    def _analyze_intent(user_query: str) -> str:
        query = str(user_query or "").strip().lower()
        analysis_kw = ["统计", "分析", "概览", "distribution", "analysis", "summary"]
        forecast_kw = ["预测", "建模", "训练", "forecast", "model", "train"]
        if any(keyword in query for keyword in forecast_kw):
            return "build_forecast_model"
        if any(keyword in query for keyword in analysis_kw):
            return "analysis_only"
        return "build_forecast_model"

    @staticmethod
    def _canonical_plan(intent: str) -> Tuple[List[str], Dict]:
        if intent == "analysis_only":
            plan = ["data_reading", "data_formatter", "data_analysis", "feature_engineering", "summary"]
            task_type = "analysis_only"
            requires_modeling = False
        else:
            plan = [
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
            task_type = "forecast_modeling"
            requires_modeling = True

        teams = [
            team_name
            for team_name, payload in TEAM_REGISTRY.items()
            if any(step in plan for step in payload["subagents"])
        ]
        return plan, {
            "task_type": task_type,
            "requires_modeling": requires_modeling,
            "requires_split": requires_modeling,
            "teams": teams,
            "reason": "deterministic_pipeline",
            "plan_source": "rule_based",
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

    def _run_model_iterations(self, plan: List[str]) -> Dict:
        iterative_steps = self._iterative_steps(plan, requires_modeling=True)
        outputs: Dict[str, Dict] = {}
        history: List[Dict] = []
        best_iteration_artifacts: List[Dict] = []
        max_iterations = min(int(self.state.read("max_iterations", DEFAULT_MAX_ITERATIONS)), MAX_ITERATIONS_CAP)
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
        write_task_text_artifact(
            state=self.state,
            relative_name="iteration_history.json",
            content=json.dumps(final_payload, ensure_ascii=False, indent=2),
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
        write_task_text_artifact(
            state=self.state,
            relative_name="cross_iteration_ensemble.json",
            content=json.dumps(payload, ensure_ascii=False, indent=2),
        )
        return payload

    def run(self, user_query: str, dataset_path: str, dataset_name: str = "custom") -> Tuple[List[str], Dict]:
        intent = self._analyze_intent(user_query)
        plan, plan_meta = self._canonical_plan(intent)
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
        ensure_task_context(self.state)
        PLAN_FILE.write_text(json.dumps({"plan": plan, "plan_meta": plan_meta}, ensure_ascii=False, indent=2), encoding="utf-8")
        self.task_manager.set_plan(plan)
        self.state.write("tasks", self.task_manager.list_tasks())

        prepare_iteration_artifacts(self.state, plan, 1)
        pre_steps = self._pre_steps(plan, requires_modeling=bool(plan_meta.get("requires_modeling")))
        outputs = agent_loop(pre_steps, SUBAGENT_REGISTRY, self.state, self.task_manager)

        if plan_meta.get("requires_modeling"):
            outputs["iterations"] = self._run_model_iterations(plan)
        else:
            outputs.update(agent_loop([step for step in plan if step not in pre_steps], SUBAGENT_REGISTRY, self.state, self.task_manager))

        return plan, outputs
