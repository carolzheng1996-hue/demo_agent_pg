from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def _make_supervised(series: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(window, len(series)):
        xs.append(series[i - window : i])
        ys.append(series[i])
    if not xs:
        raise ValueError("Series is too short for selected window.")
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _linear_fit_predict(train: np.ndarray, horizon: int, window: int) -> np.ndarray:
    x, y = _make_supervised(train, window)
    x_aug = np.concatenate([x, np.ones((len(x), 1), dtype=np.float32)], axis=1)
    w, *_ = np.linalg.lstsq(x_aug, y, rcond=None)

    history = train.astype(np.float32).tolist()
    out = []
    for _ in range(horizon):
        feat = np.asarray(history[-window:], dtype=np.float32)
        feat_aug = np.append(feat, 1.0)
        pred = float(np.dot(feat_aug, w))
        history.append(pred)
        out.append(pred)
    return np.asarray(out, dtype=np.float32)


def _window_statistics(window_values: np.ndarray, feature_methods: List[str]) -> np.ndarray:
    window_values = np.asarray(window_values, dtype=np.float32)
    features: List[float] = window_values.tolist()

    if "rolling_signature" in feature_methods:
        short = window_values[-min(6, len(window_values)) :]
        long = window_values[-min(24, len(window_values)) :]
        features.extend([float(short.mean()), float(long.mean()), float(long.std()), float(long.min()), float(long.max())])
    if "difference_signature" in feature_methods and len(window_values) > 1:
        diffs = np.diff(window_values)
        features.extend([float(diffs[-1]), float(diffs.mean()), float(np.abs(diffs).mean())])
    if "ewm_signature" in feature_methods:
        alpha = 2.0 / (min(12, len(window_values)) + 1.0)
        ewm = window_values[0]
        for value in window_values[1:]:
            ewm = alpha * value + (1.0 - alpha) * ewm
        features.append(float(ewm))
    if "peak_signature" in feature_methods and len(window_values) > 2:
        turning_points = np.sum(np.diff(np.sign(np.diff(window_values))) != 0)
        features.extend([float(np.square(window_values).sum()), float(window_values.max() - window_values.min()), float(turning_points)])

    return np.asarray(features, dtype=np.float32)


def _make_supervised_with_features(series: np.ndarray, window: int, feature_methods: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(window, len(series)):
        xs.append(_window_statistics(series[i - window : i], feature_methods))
        ys.append(series[i])
    if not xs:
        raise ValueError("Series is too short for selected window.")
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _frame_window_features(frame: np.ndarray, start: int, end: int) -> np.ndarray:
    window = np.asarray(frame[start:end], dtype=np.float32)
    return window.reshape(-1)


def _make_supervised_from_frame(frame: np.ndarray, target_idx: int, window: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(window, len(frame)):
        xs.append(_frame_window_features(frame, i - window, i))
        ys.append(frame[i, target_idx])
    if not xs:
        raise ValueError("Frame is too short for selected window.")
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _make_sequence_supervised_from_frame(frame: np.ndarray, target_idx: int, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(seq_len, len(frame)):
        xs.append(np.asarray(frame[i - seq_len : i], dtype=np.float32))
        ys.append(np.float32(frame[i, target_idx]))
    if not xs:
        raise ValueError("Frame is too short for selected sequence length.")
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def train_arima(train: np.ndarray, test: np.ndarray, order: Tuple[int, int, int] = (2, 1, 2)) -> Dict[str, Any]:
    try:
        from statsmodels.tsa.arima.model import ARIMA

        fitted = ARIMA(train, order=order).fit()
        pred = np.asarray(fitted.forecast(steps=len(test)), dtype=np.float32)
        backend = "statsmodels"
    except Exception:
        pred = np.full(shape=(len(test),), fill_value=float(train[-1]), dtype=np.float32)
        backend = "fallback_last_value"

    return {
        "name": "arima",
        "params": {"order": order},
        "backend": backend,
        "predictions": pred.tolist(),
    }


def train_xgboost(
    train: np.ndarray,
    test: np.ndarray,
    window: int = 48,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_seed: int = 42,
    feature_methods: List[str] | None = None,
    train_frame: pd.DataFrame | None = None,
    test_frame: pd.DataFrame | None = None,
    target_col: str | None = None,
) -> Dict[str, Any]:
    feature_methods = list(feature_methods or [])
    try:
        import xgboost as xgb

        if train_frame is not None and test_frame is not None and target_col and target_col in train_frame.columns:
            numeric_columns = train_frame.select_dtypes(include=["number"]).columns.tolist()
            if target_col not in numeric_columns:
                numeric_columns.append(target_col)
            combined_frame = pd.concat(
                [
                    train_frame[numeric_columns],
                    test_frame[numeric_columns],
                ],
                axis=0,
                ignore_index=True,
            ).astype(np.float32)
            target_idx = combined_frame.columns.get_loc(target_col)
            train_array = combined_frame.iloc[: len(train_frame)].to_numpy(dtype=np.float32)
            combined_array = combined_frame.to_numpy(dtype=np.float32)
            x, y = _make_supervised_from_frame(train_array, target_idx=target_idx, window=window)
            prediction_features = []
            start_index = len(train_frame)
            for i in range(start_index, len(combined_array)):
                prediction_features.append(_frame_window_features(combined_array, i - window, i))
            prediction_matrix = np.asarray(prediction_features, dtype=np.float32)
        else:
            x, y = _make_supervised_with_features(train, window, feature_methods)
            prediction_matrix = None

        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="reg:squarederror",
            random_state=random_seed,
        )
        model.fit(x, y)

        if prediction_matrix is not None and len(prediction_matrix):
            pred = np.asarray(model.predict(prediction_matrix), dtype=np.float32)
        else:
            history = train.astype(np.float32).tolist()
            preds = []
            for _ in range(len(test)):
                feat = _window_statistics(np.asarray(history[-window:], dtype=np.float32), feature_methods)[None, :]
                pred_value = float(model.predict(feat)[0])
                history.append(pred_value)
                preds.append(pred_value)
            pred = np.asarray(preds, dtype=np.float32)
        backend = "xgboost"
    except Exception:
        pred = _linear_fit_predict(train, horizon=len(test), window=window)
        backend = "fallback_linear"

    return {
        "name": "xgboost",
        "params": {
            "window": window,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_seed": random_seed,
            "feature_methods": feature_methods,
        },
        "backend": backend,
        "predictions": pred.tolist(),
    }


def train_linear(
    train: np.ndarray,
    test: np.ndarray,
    window: int = 96,
    fit_intercept: bool = True,
    train_frame: pd.DataFrame | None = None,
    test_frame: pd.DataFrame | None = None,
    target_col: str | None = None,
) -> Dict[str, Any]:
    try:
        if train_frame is not None and test_frame is not None and target_col and target_col in train_frame.columns:
            numeric_columns = train_frame.select_dtypes(include=["number"]).columns.tolist()
            if target_col not in numeric_columns:
                numeric_columns.append(target_col)
            combined_frame = pd.concat(
                [train_frame[numeric_columns], test_frame[numeric_columns]],
                axis=0,
                ignore_index=True,
            ).astype(np.float32)
            target_idx = combined_frame.columns.get_loc(target_col)
            train_array = combined_frame.iloc[: len(train_frame)].to_numpy(dtype=np.float32)
            combined_array = combined_frame.to_numpy(dtype=np.float32)
            x, y = _make_supervised_from_frame(train_array, target_idx=target_idx, window=window)
            if fit_intercept:
                x = np.concatenate([x, np.ones((len(x), 1), dtype=np.float32)], axis=1)
            weights, *_ = np.linalg.lstsq(x, y, rcond=None)

            prediction_features = []
            start_index = len(train_frame)
            for i in range(start_index, len(combined_array)):
                features = _frame_window_features(combined_array, i - window, i)
                if fit_intercept:
                    features = np.append(features, 1.0)
                prediction_features.append(features)
            prediction_matrix = np.asarray(prediction_features, dtype=np.float32)
            pred = prediction_matrix @ weights
            training_mode = "multivariate_linear"
        else:
            pred = _linear_fit_predict(train, horizon=len(test), window=window)
            training_mode = "univariate_linear"
        backend = "numpy_linear"
        error_message = None
    except Exception as exc:
        pred = _linear_fit_predict(train, horizon=len(test), window=min(window, max(8, len(train) // 10)))
        backend = "fallback_linear"
        training_mode = "fallback_univariate"
        error_message = str(exc)

    return {
        "name": "linear",
        "params": {
            "window": window,
            "fit_intercept": fit_intercept,
        },
        "backend": backend,
        "training_mode": training_mode,
        "fallback_error": error_message,
        "predictions": np.asarray(pred, dtype=np.float32).tolist(),
    }


def train_lstm(
    train: np.ndarray,
    test: np.ndarray,
    seq_len: int = 96,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    epochs: int = 8,
    lr: float = 1e-3,
    batch_size: int = 64,
    random_seed: int = 42,
    train_frame: pd.DataFrame | None = None,
    test_frame: pd.DataFrame | None = None,
    target_col: str | None = None,
) -> Dict[str, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

        backend_mode = "univariate"
        target_idx = 0
        feature_columns = [target_col] if target_col else ["target"]
        feature_stats: Dict[str, Dict[str, float]] = {}

        if train_frame is not None and test_frame is not None and target_col and target_col in train_frame.columns:
            numeric_columns = train_frame.select_dtypes(include=["number"]).columns.tolist()
            if target_col not in numeric_columns:
                numeric_columns.append(target_col)
            ordered_columns = [target_col] + [column for column in numeric_columns if column != target_col]
            train_matrix_raw = train_frame[ordered_columns].astype(np.float32).to_numpy(dtype=np.float32)
            test_matrix_raw = test_frame[ordered_columns].astype(np.float32).to_numpy(dtype=np.float32)

            feature_columns = ordered_columns
            target_idx = 0
            means = train_matrix_raw.mean(axis=0)
            stds = train_matrix_raw.std(axis=0)
            stds = np.where(stds > 1e-8, stds, 1.0)
            feature_stats = {
                column: {"mean": float(means[idx]), "std": float(stds[idx])}
                for idx, column in enumerate(ordered_columns)
            }
            train_matrix = ((train_matrix_raw - means) / stds).astype(np.float32)
            test_matrix = ((test_matrix_raw - means) / stds).astype(np.float32)
            x_np, y_np = _make_sequence_supervised_from_frame(train_matrix, target_idx=target_idx, seq_len=seq_len)
            y_np = y_np[..., None]
            backend_mode = "multivariate"
        else:
            mean = float(train.mean())
            std = float(train.std()) if float(train.std()) > 1e-8 else 1.0
            train_norm = (train - mean) / std
            feature_stats = {"target": {"mean": mean, "std": std}}
            x_np, y_np = _make_supervised(train_norm, seq_len)
            x_np = x_np[..., None]
            y_np = y_np[..., None]
            train_matrix = train_norm[..., None].astype(np.float32)
            test_matrix = np.asarray(test, dtype=np.float32)[..., None]

        class LSTMModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=x_np.shape[-1],
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    batch_first=True,
                )
                self.head = nn.Linear(hidden_size, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :])

        device = "cuda" if torch.cuda.is_available() else "cpu"
        ds = TensorDataset(torch.from_numpy(x_np.astype(np.float32)), torch.from_numpy(y_np.astype(np.float32)))
        generator = torch.Generator().manual_seed(random_seed)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=generator)

        model = LSTMModel().to(device)
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        model.train()
        for _ in range(epochs):
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)
                optim.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optim.step()

        model.eval()
        preds_norm = []
        with torch.no_grad():
            if backend_mode == "multivariate":
                target_mean = feature_stats[feature_columns[target_idx]]["mean"]
                target_std = feature_stats[feature_columns[target_idx]]["std"]
                future_known = test_matrix.copy()
                future_known[:, target_idx] = 0.0
                combined_matrix = np.concatenate([train_matrix, future_known], axis=0).astype(np.float32)
                start_index = len(train_matrix)
                for i in range(start_index, len(combined_matrix)):
                    feat = np.asarray(combined_matrix[i - seq_len : i], dtype=np.float32)[None, :, :]
                    xt = torch.from_numpy(feat).to(device)
                    next_target_norm = float(model(xt).squeeze().item())
                    combined_matrix[i, target_idx] = next_target_norm
                    preds_norm.append(next_target_norm)
                pred = np.asarray(preds_norm, dtype=np.float32) * target_std + target_mean
            else:
                mean = feature_stats["target"]["mean"]
                std = feature_stats["target"]["std"]
                history = train_matrix[:, 0].tolist()
                for _ in range(len(test)):
                    feat = np.asarray(history[-seq_len:], dtype=np.float32)[None, :, None]
                    xt = torch.from_numpy(feat).to(device)
                    nxt = float(model(xt).squeeze().item())
                    history.append(nxt)
                    preds_norm.append(nxt)
                pred = np.asarray(preds_norm, dtype=np.float32) * std + mean
        backend = "torch"
        error_message = None
    except Exception as exc:
        pred = _linear_fit_predict(train, horizon=len(test), window=min(seq_len, max(8, len(train) // 10)))
        backend = "fallback_linear"
        backend_mode = "fallback_univariate"
        feature_columns = [target_col] if target_col else ["target"]
        error_message = str(exc)

    return {
        "name": "lstm",
        "params": {
            "seq_len": seq_len,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "random_seed": random_seed,
        },
        "backend": backend,
        "training_mode": backend_mode,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "fallback_error": error_message,
        "predictions": pred.tolist(),
    }
