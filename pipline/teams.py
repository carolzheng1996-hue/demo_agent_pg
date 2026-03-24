from __future__ import annotations

from typing import Dict, List

TEAM_REGISTRY: Dict[str, Dict[str, List[str] | str]] = {
    "discovery_team": {
        "description": "负责数据接入、多类型格式整理和统计特性分析。",
        "subagents": ["data_reading", "data_analysis", "feature_engineering"],
    },
    "modeling_team": {
        "description": "负责切分策略、归一化决策、预处理、候选匹配、训练、评估与结果集成。",
        "subagents": [
            "split_strategy",
            "datanorm",
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
