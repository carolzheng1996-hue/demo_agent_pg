from __future__ import annotations

import json
from typing import Dict, List, Tuple

from .agent_loop import agent_loop
from .config import PLAN_FILE
from .llm_utils import invoke_json
from .state import DGGlobalState
from .subagents import SUBAGENT_REGISTRY
from .task_manager import DGTaskManager
from .teams import TEAM_REGISTRY


class DGOrchestrator:
    def __init__(self, state: DGGlobalState, task_manager: DGTaskManager):
        self.state = state
        self.task_manager = task_manager

    @staticmethod
    def _canonical_plan(raw_plan: List[str]) -> List[str]:
        canonical = [
            "data_reading",
            "data_analysis",
            "model_selection",
            "model_training",
            "model_integration",
            "summary",
        ]
        allowed = set(canonical)
        clean = []
        for step in raw_plan:
            name = str(step).strip()
            if name in allowed and name not in clean:
                clean.append(name)

        if not clean:
            clean = ["data_reading", "data_analysis", "summary"]

        if any(step in clean for step in ["model_selection", "model_training", "model_integration"]):
            for required in ["data_reading", "data_analysis", "model_selection", "model_training", "model_integration"]:
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
        payload = invoke_json(
            system_prompt=(
                "You are the main agent of a data analysis system. Understand the user task semantically and produce an execution plan. "
                "Do not rely on keyword matching. Return JSON with keys: task_type, requires_modeling, requires_split, teams, subagents, reason. "
                "Allowed subagents: data_reading, data_analysis, model_selection, model_training, model_integration, summary. "
                "If modeling is required, include model_selection, model_training, model_integration. summary must be last."
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
            plan = ["data_reading", "data_analysis", "summary"]
            metadata = {
                "task_type": "analysis_only",
                "requires_modeling": False,
                "requires_split": False,
                "teams": self._map_teams(plan),
                "reason": "llm_unavailable_default_to_analysis_only",
                "plan_source": "fallback",
            }
            return plan, metadata

        plan = self._canonical_plan(payload.get("subagents", []))
        requires_modeling = bool(payload.get("requires_modeling", False))
        metadata = {
            "task_type": payload.get("task_type", "unknown"),
            "requires_modeling": requires_modeling,
            "requires_split": bool(payload.get("requires_split", requires_modeling)),
            "teams": payload.get("teams") or self._map_teams(plan),
            "reason": payload.get("reason", "llm_generated"),
            "plan_source": "llm",
        }
        return plan, metadata

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
        PLAN_FILE.write_text(json.dumps({"plan": plan, "plan_meta": plan_meta}, ensure_ascii=False, indent=2), encoding="utf-8")
        self.task_manager.set_plan(plan)
        self.state.write("tasks", self.task_manager.list_tasks())
        outputs = agent_loop(plan, SUBAGENT_REGISTRY, self.state, self.task_manager)
        return plan, outputs
