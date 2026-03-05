from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


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
) -> Dict[str, Any]:
    try:
        import xgboost as xgb

        x, y = _make_supervised(train, window)
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
        )
        model.fit(x, y)

        history = train.astype(np.float32).tolist()
        preds = []
        for _ in range(len(test)):
            feat = np.asarray(history[-window:], dtype=np.float32)[None, :]
            pred = float(model.predict(feat)[0])
            history.append(pred)
            preds.append(pred)
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
        },
        "backend": backend,
        "predictions": pred.tolist(),
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
) -> Dict[str, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        mean = float(train.mean())
        std = float(train.std()) if float(train.std()) > 1e-8 else 1.0
        train_norm = (train - mean) / std

        x_np, y_np = _make_supervised(train_norm, seq_len)
        x_np = x_np[..., None]
        y_np = y_np[..., None]

        class LSTMModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=1,
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
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

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
        history = train_norm.tolist()
        preds_norm = []
        with torch.no_grad():
            for _ in range(len(test)):
                feat = np.asarray(history[-seq_len:], dtype=np.float32)[None, :, None]
                xt = torch.from_numpy(feat).to(device)
                nxt = float(model(xt).squeeze().item())
                history.append(nxt)
                preds_norm.append(nxt)

        pred = np.asarray(preds_norm, dtype=np.float32) * std + mean
        backend = "torch"
    except Exception:
        pred = _linear_fit_predict(train, horizon=len(test), window=min(seq_len, max(8, len(train) // 10)))
        backend = "fallback_linear"

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
        },
        "backend": backend,
        "predictions": pred.tolist(),
    }
