from . import data_analysis, data_reading, model_integration, model_selection, model_training, summary

SUBAGENT_REGISTRY = {
    "data_reading": data_reading.run,
    "data_analysis": data_analysis.run,
    "model_selection": model_selection.run,
    "model_training": model_training.run,
    "model_integration": model_integration.run,
    "summary": summary.run,
}

__all__ = ["SUBAGENT_REGISTRY"]
