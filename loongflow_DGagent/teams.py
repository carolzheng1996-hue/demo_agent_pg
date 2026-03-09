from __future__ import annotations

from typing import Dict, List

TEAM_REGISTRY: Dict[str, Dict[str, List[str] | str]] = {
    "discovery_team": {
        "description": "负责数据接入、格式标准化和统计特性分析。",
        "subagents": ["data_reading", "data_analysis", "feature_engineering"],
    },
    "modeling_team": {
        "description": "负责切分策略、预处理、候选匹配、训练、评估与结果集成。",
        "subagents": [
            "split_strategy",
            "preprocess",
            "model_selection",
            "model_training",
            "evaluator",
            "model_integration",
        ],
    },
    "reporting_team": {
        "description": "负责沉淀过程和结果报告。",
        "subagents": ["summary"],
    },
}
