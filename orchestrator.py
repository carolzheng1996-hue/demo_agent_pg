from __future__ import annotations

import json
from typing import Dict, List, Tuple

from agent_loop import agent_loop
from global_state import GlobalState
from llm_utils import invoke_json
from subagents import SUBAGENT_REGISTRY
from task_manager import TaskManager


class Orchestrator:
    def __init__(self, state: GlobalState, task_manager: TaskManager):
        self.state = state
        self.task_manager = task_manager

    @staticmethod
    def _analyze_intent_fallback(user_query: str) -> str:
        query = user_query.strip()
        q = query.lower()
        analysis_kw = ["统计", "特性分析", "分析", "statistics", "analysis"]
        forecast_kw = ["预测", "建模", "训练", "模型", "forecast", "train", "model"]
        context_kw = [
            "时序",
            "时间序列",
            "time series",
            "dataset",
            "数据集",
            "csv",
            "etth",
            "ettm",
            "trend",
            "seasonality",
            "stationarity",
            "趋势",
            "季节性",
            "平稳性",
            "数据",
        ]

        has_analysis = any(k in q for k in analysis_kw)
        has_forecast = any(k in q for k in forecast_kw)
        has_context = any(k in q for k in context_kw)

        if has_forecast:
            return "build_forecast_model"
        if has_analysis or has_context:
            return "stats_analysis"
        return "general_chat"

    @staticmethod
    def make_plan(intent: str) -> List[str]:
        if intent == "general_chat":
            return ["summary"]
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

    @staticmethod
    def _normalize_plan(candidate_plan: List[str], available_steps: List[str]) -> List[str]:
        available = set(available_steps)
        plan = []
        for step in candidate_plan:
            s = str(step).strip()
            if s in available and s not in plan:
                plan.append(s)

        has_modeling = any(x in plan for x in ["model_selection", "model_training", "result_integration"])
        has_data_step = any(x in plan for x in ["data_reading", "data_analysis"])
        summary_only = bool(plan) and all(x == "summary" for x in plan)
        if has_modeling:
            required_chain = [
                "data_reading",
                "data_analysis",
                "model_selection",
                "model_training",
                "result_integration",
            ]
            for req in required_chain:
                if req not in plan:
                    plan.append(req)
        elif has_data_step:
            if "data_reading" not in plan:
                plan.insert(0, "data_reading")
            if "data_analysis" not in plan:
                plan.append("data_analysis")
        elif not summary_only:
            if "data_reading" in available and "data_reading" not in plan:
                plan.insert(0, "data_reading")
            if "data_analysis" in available and "data_analysis" not in plan:
                plan.append("data_analysis")

        # Put core dependencies in canonical order.
        canonical = [
            "data_reading",
            "data_analysis",
            "model_selection",
            "model_training",
            "result_integration",
        ]
        fixed = [x for x in canonical if x in plan]
        # Preserve any additional steps (if future subagents are added).
        fixed.extend([x for x in plan if x not in fixed and x != "summary"])
        if "summary" in available:
            fixed = [x for x in fixed if x != "summary"] + ["summary"]
        return fixed

    def _generate_plan_with_llm(self, user_query: str) -> Tuple[List[str], str, str]:
        available_steps = list(SUBAGENT_REGISTRY.keys())
        fallback_intent = self._analyze_intent_fallback(user_query)
        fallback_plan = self.make_plan(fallback_intent)
        if fallback_intent == "general_chat":
            return fallback_plan, fallback_intent, "fallback_general_query_guard"

        llm_payload = invoke_json(
            system_prompt=(
                "You are the master orchestrator for a time-series multi-agent system. "
                "Generate an executable plan from allowed steps only.\n"
                "Return JSON: {\"intent\":\"general_chat\"|\"stats_analysis\"|\"build_forecast_model\","
                "\"plan\":[\"step1\",\"step2\",...],\"reason\":\"short text\"}.\n"
                "Rules:\n"
                "1) Allowed steps only.\n"
                "2) data_reading must be before data_analysis.\n"
                "3) If model_training is used, also include model_selection before it and result_integration after it.\n"
                "4) summary must be the last step.\n"
                "5) For greeting/chitchat/unrelated requests, intent must be general_chat and plan must be [\"summary\"] only.\n"
                "6) Use build_forecast_model only when user explicitly asks prediction/model training.\n"
                "7) Prefer short plan for analysis-only queries."
            ),
            user_prompt=json.dumps(
                {
                    "user_query": user_query,
                    "allowed_steps": available_steps,
                },
                ensure_ascii=False,
            ),
            max_tokens=220,
            temperature=0.0,
        )

        if not llm_payload:
            return fallback_plan, fallback_intent, "fallback_no_llm_payload"

        raw_plan = llm_payload.get("plan", [])
        if not isinstance(raw_plan, list):
            return fallback_plan, fallback_intent, "fallback_invalid_llm_plan_type"

        normalized = self._normalize_plan(raw_plan, available_steps)
        if not normalized:
            return fallback_plan, fallback_intent, "fallback_empty_llm_plan"

        llm_intent = str(llm_payload.get("intent", "")).strip()
        if llm_intent not in {"general_chat", "stats_analysis", "build_forecast_model"}:
            if "model_training" in normalized:
                llm_intent = "build_forecast_model"
            elif any(x in normalized for x in ["data_reading", "data_analysis"]):
                llm_intent = "stats_analysis"
            else:
                llm_intent = "general_chat"
        reason = str(llm_payload.get("reason", "llm_generated"))
        return normalized, llm_intent, reason

    def run(self, user_query: str, dataset_name: str = "etth", dataset_path: str | None = None) -> Tuple[List[str], Dict]:
        plan, intent, plan_reason = self._generate_plan_with_llm(user_query)
        plan_source = "llm" if not plan_reason.startswith("fallback_") else "fallback_rules"

        self.state.update(
            {
                "user_query": user_query,
                "dataset_name": dataset_name,
                "dataset_path": dataset_path,
                "intent": intent,
                "plan": plan,
                "plan_source": plan_source,
                "plan_reason": plan_reason,
                "llm_enabled": plan_source == "llm",
            }
        )

        self.task_manager.set_plan(plan)
        self.state.write("tasks", self.task_manager.list_tasks())

        outputs = agent_loop(plan, SUBAGENT_REGISTRY, self.state, self.task_manager)
        return plan, outputs
