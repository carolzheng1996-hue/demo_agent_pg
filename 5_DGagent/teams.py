from __future__ import annotations

from typing import Dict, List

TEAM_REGISTRY: Dict[str, Dict[str, List[str] | str]] = {
    "discovery_team": {
        "description": "负责数据接入、格式标准化和统计特性分析。",
        "subagents": ["data_reading", "data_analysis"],
    },
    "modeling_team": {
        "description": "负责模型候选匹配、训练与结果集成。",
        "subagents": ["model_selection", "model_training", "model_integration"],
    },
    "reporting_team": {
        "description": "负责沉淀过程和结果报告。",
        "subagents": ["summary"],
    },
}
