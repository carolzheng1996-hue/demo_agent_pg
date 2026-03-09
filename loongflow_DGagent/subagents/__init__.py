from . import (
    data_analysis,
    data_reading,
    evaluator,
    feature_engineering,
    model_integration,
    model_selection,
    model_training,
    preprocess,
    split_strategy,
    summary,
)

SUBAGENT_REGISTRY = {
    "data_reading": data_reading.run,
    "data_analysis": data_analysis.run,
    "feature_engineering": feature_engineering.run,
    "split_strategy": split_strategy.run,
    "preprocess": preprocess.run,
    "model_selection": model_selection.run,
    "model_training": model_training.run,
    "evaluator": evaluator.run,
    "model_integration": model_integration.run,
    "summary": summary.run,
}

__all__ = ["SUBAGENT_REGISTRY"]
