from __future__ import annotations

import json
from typing import Dict, List, Tuple

try:
    from .agent_loop import agent_loop, run_subagent
    from .config import DEFAULT_MAX_ITERATIONS, MAX_ITERATIONS_CAP, PLAN_FILE
    from .llm_utils import invoke_json
    from .state import DGGlobalState
    from .subagents import SUBAGENT_REGISTRY
    from .task_manager import DGTaskManager
    from .teams import TEAM_REGISTRY
    from .tools import ensure_task_context, prepare_iteration_artifacts, write_task_text_artifact
except ImportError:
    from agent_loop import agent_loop, run_subagent
    from config import DEFAULT_MAX_ITERATIONS, MAX_ITERATIONS_CAP, PLAN_FILE
    from llm_utils import invoke_json
    from state import DGGlobalState
    from subagents import SUBAGENT_REGISTRY
    from task_manager import DGTaskManager
    from teams import TEAM_REGISTRY
    from tools import ensure_task_context, prepare_iteration_artifacts, write_task_text_artifact


class DGOrchestrator:
    def __init__(self, state: DGGlobalState, task_manager: DGTaskManager):
        self.state = state
        self.task_manager = task_manager

    @staticmethod
    def _analyze_intent_fallback(user_query: str) -> str:
        query = user_query.strip()
        q = query.lower()
        analysis_kw = ["统计", "特性分析", "分析", "statistics", "analysis"]
        forecast_kw = ["预测", "建模", "训练", "模型", "forecast", "train", "model"]
        context_kw = ["时序", "时间序列", "time series", "dataset", "数据集", "csv", "etth", "ettm", "数据"]

        has_analysis = any(k in q for k in analysis_kw)
        has_forecast = any(k in q for k in forecast_kw)
        has_context = any(k in q for k in context_kw)

        if has_forecast:
            return "build_forecast_model"
        if has_analysis or has_context:
            return "analysis_only"
        return "general_chat"

    @staticmethod
    def _fallback_plan(intent: str) -> Tuple[List[str], Dict]:
        if intent == "build_forecast_model":
            plan = [
                "data_reading",
                "data_analysis",
                "feature_engineering",
                "split_strategy",
                "preprocess",
                "model_selection",
                "model_training",
                "model_integration",
                "evaluator",
                "summary",
            ]
            meta = {
                "task_type": "forecast_modeling",
                "requires_modeling": True,
                "requires_split": True,
            }
        elif intent == "analysis_only":
            plan = ["data_reading", "data_analysis", "feature_engineering", "summary"]
            meta = {
                "task_type": "analysis_only",
                "requires_modeling": False,
                "requires_split": False,
            }
        else:
            plan = ["summary"]
            meta = {
                "task_type": "general_chat",
                "requires_modeling": False,
                "requires_split": False,
            }
        meta["teams"] = DGOrchestrator._map_teams(plan)
        meta["plan_source"] = "fallback"
        return plan, meta

    @staticmethod
    def _canonical_plan(raw_plan: List[str]) -> List[str]:
        canonical = [
            "data_reading",
            "data_analysis",
            "feature_engineering",
            "split_strategy",
            "preprocess",
            "model_selection",
            "model_training",
            "model_integration",
            "evaluator",
            "summary",
        ]
        allowed = set(canonical)
        clean = []
        for step in raw_plan:
            name = str(step).strip()
            if name in allowed and name not in clean:
                clean.append(name)

        if not clean:
            clean = ["data_reading", "data_analysis", "feature_engineering", "summary"]

        if any(step in clean for step in ["model_selection", "model_training", "evaluator", "model_integration"]):
            for required in ["data_reading", "data_analysis", "model_selection", "model_training", "model_integration"]:
                if required not in clean:
                    clean.append(required)
            for required in ["feature_engineering", "split_strategy", "preprocess", "evaluator"]:
                if required not in clean:
                    clean.append(required)

        ordered = [step for step in canonical if step in clean and step != "summary"]
        return ordered + ["summary"]

    @staticmethod
    def _map_teams(plan: List[str]) -> List[str]:
        selected = []
        for team_name, payload in TEAM_REGISTRY.items():
            subagents = payload["subagents"]
            if any(step in plan for step in subagents):
                selected.append(team_name)
        return selected

    def _generate_plan(self, user_query: str) -> Tuple[List[str], Dict]:
        fallback_intent = self._analyze_intent_fallback(user_query)
        payload = invoke_json(
            system_prompt=(
                "You are the main agent of a data analysis system. Understand the user task semantically and produce an execution plan. "
                "Do not rely on keyword matching. Return JSON with keys: task_type, requires_modeling, requires_split, teams, subagents, reason. "
                "Allowed subagents: data_reading, data_analysis, feature_engineering, split_strategy, preprocess, model_selection, model_training, evaluator, model_integration, summary. "
                "If modeling is required, include split_strategy, preprocess, model_selection, model_training, evaluator, model_integration. summary must be last."
            ),
            user_prompt=json.dumps(
                {
                    "user_query": user_query,
                    "available_teams": TEAM_REGISTRY,
                    "available_subagents": list(SUBAGENT_REGISTRY.keys()),
                },
                ensure_ascii=False,
            ),
            max_tokens=400,
            temperature=0.0,
        )

        if not payload:
            plan, metadata = self._fallback_plan(fallback_intent)
            metadata["reason"] = f"llm_unavailable_{fallback_intent}"
            return plan, metadata

        plan = self._canonical_plan(payload.get("subagents", []))
        requires_modeling = bool(payload.get("requires_modeling", False))
        if not plan:
            plan, metadata = self._fallback_plan(fallback_intent)
            metadata["reason"] = f"llm_empty_plan_{fallback_intent}"
            return plan, metadata
        metadata = {
            "task_type": payload.get("task_type", "unknown"),
            "requires_modeling": requires_modeling,
            "requires_split": bool(payload.get("requires_split", requires_modeling)),
            "teams": payload.get("teams") or self._map_teams(plan),
            "reason": payload.get("reason", "llm_generated"),
            "plan_source": "llm",
        }
        return plan, metadata

    @staticmethod
    def _pre_steps(plan: List[str]) -> List[str]:
        return [step for step in plan if step in ["data_reading", "data_analysis", "feature_engineering", "split_strategy", "preprocess"]]

    @staticmethod
    def _iterative_steps(plan: List[str]) -> List[str]:
        return [step for step in plan if step in ["model_selection", "model_training", "model_integration", "evaluator", "summary"]]

    def _run_model_iterations(self, plan: List[str], plan_meta: Dict) -> Dict:
        iterative_steps = self._iterative_steps(plan)
        outputs: Dict[str, Dict] = {}
        history: List[Dict] = []
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
            history.append(
                {
                    "iteration_index": iteration_index,
                    "best_score": evaluator_result.get("best_score"),
                    "selected_strategy": evaluator_result.get("selected_strategy"),
                    "report_path": self.state.read("report_path"),
                }
            )
            self.state.write("iteration_history", history)
            if not evaluator_result.get("should_continue", False):
                break

        final_payload = {
            "iterations_completed": len(history),
            "iteration_history": history,
            "final_report_path": self.state.read("final_report_path"),
        }
        write_task_text_artifact(state=self.state, relative_name="iteration_history.json", content=json.dumps(final_payload, ensure_ascii=False, indent=2))
        return final_payload

    def run(self, user_query: str, dataset_path: str, dataset_name: str = "custom") -> Tuple[List[str], Dict]:
        plan, plan_meta = self._generate_plan(user_query)
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
        pre_steps = self._pre_steps(plan)
        outputs = agent_loop(pre_steps, SUBAGENT_REGISTRY, self.state, self.task_manager)
        if any(result.get("pause_execution") for result in outputs.values()):
            return plan, outputs

        if plan_meta.get("requires_modeling"):
            outputs["iterations"] = self._run_model_iterations(plan, plan_meta)
        else:
            outputs.update(
                agent_loop(
                    [step for step in plan if step not in pre_steps],
                    SUBAGENT_REGISTRY,
                    self.state,
                    self.task_manager,
                )
            )
        return plan, outputs
