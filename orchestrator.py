from __future__ import annotations

from typing import Dict, List, Tuple

from agent_loop import agent_loop
from global_state import GlobalState
from llm_client import LLMClient
from subagents import SUBAGENT_REGISTRY
from task_manager import TaskManager


class Orchestrator:
    def __init__(self, state: GlobalState, task_manager: TaskManager):
        self.state = state
        self.task_manager = task_manager
        self.llm = LLMClient(self.state.read("api_config", {}))

    @staticmethod
    def _analyze_intent_fallback(user_query: str) -> str:
        q = user_query.lower()
        analysis_kw = ["统计", "特性分析", "分析", "statistics", "analysis"]
        forecast_kw = ["预测", "建模", "训练", "模型", "forecast", "train", "model"]

        has_analysis = any(k in q for k in analysis_kw)
        has_forecast = any(k in q for k in forecast_kw)

        if has_analysis and not has_forecast:
            return "stats_analysis"
        return "build_forecast_model"

    def analyze_intent(self, user_query: str) -> str:
        llm_payload = self.llm.complete_json(
            system_prompt=(
                "You classify user intent for a time-series pipeline. "
                "Return JSON: {\"intent\": \"stats_analysis\"|\"build_forecast_model\"} only."
            ),
            user_prompt=f"user_query: {user_query}",
            max_tokens=64,
            temperature=0.0,
        )
        if llm_payload:
            intent = str(llm_payload.get("intent", "")).strip()
            if intent in {"stats_analysis", "build_forecast_model"}:
                return intent
        return self._analyze_intent_fallback(user_query)

    @staticmethod
    def make_plan(intent: str) -> List[str]:
        if intent == "stats_analysis":
            return ["data_reading", "data_analysis", "summary"]
        return [
            "data_reading",
            "data_analysis",
            "model_selection",
            "model_training",
            "result_integration",
            "summary",
        ]

    def run(self, user_query: str, dataset_name: str = "etth", dataset_path: str | None = None) -> Tuple[List[str], Dict]:
        self.llm = LLMClient(self.state.read("api_config", {}))
        intent = self.analyze_intent(user_query)
        plan = self.make_plan(intent)

        self.state.update(
            {
                "user_query": user_query,
                "dataset_name": dataset_name,
                "dataset_path": dataset_path,
                "intent": intent,
                "plan": plan,
                "llm_enabled": self.llm.enabled,
            }
        )

        self.task_manager.set_plan(plan)
        self.state.write("tasks", self.task_manager.list_tasks())

        outputs = agent_loop(plan, SUBAGENT_REGISTRY, self.state, self.task_manager)
        return plan, outputs
