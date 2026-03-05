from __future__ import annotations

from . import data_analysis, data_reading, model_selection, model_training, result_integration, summary

SUBAGENT_REGISTRY = {
    "data_reading": data_reading.run,
    "data_analysis": data_analysis.run,
    "model_selection": model_selection.run,
    "model_training": model_training.run,
    "result_integration": result_integration.run,
    "summary": summary.run,
}

__all__ = ["SUBAGENT_REGISTRY"]
