from __future__ import annotations

import numpy as np
import pandas as pd


def smape(actual: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.abs(actual) + np.abs(prediction)
    ratio = np.divide(
        2.0 * np.abs(actual - prediction),
        denominator,
        out=np.zeros_like(actual, dtype=float),
        where=denominator != 0,
    )
    return 100.0 * ratio.mean()


def mase(actual: np.ndarray, prediction: np.ndarray, train_values: np.ndarray, season_length: int) -> float:
    if len(train_values) <= season_length:
        scale = np.abs(np.diff(train_values)).mean()
    else:
        scale = np.abs(train_values[season_length:] - train_values[:-season_length]).mean()
    if scale == 0:
        scale = 1.0
    return np.abs(actual - prediction).mean() / scale


def summarize_metrics(forecasts: pd.DataFrame, season_length: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for keys, frame in forecasts.groupby(["model_name", "window", "series_id"]):
        actual = frame["actual"].to_numpy()
        prediction = frame["prediction"].to_numpy()
        train_values = np.asarray(frame["train_history"].iloc[0], dtype=float)

        rows.append(
            {
                "model_name": keys[0],
                "window": keys[1],
                "series_id": keys[2],
                "smape": smape(actual, prediction),
                "mase": mase(actual, prediction, train_values, season_length),
            }
        )

    series_metrics = pd.DataFrame(rows)
    summary = (
        series_metrics.groupby("model_name", as_index=False)[["smape", "mase"]]
        .mean()
        .sort_values("smape")
    )
    horizon_rows = []
    for keys, frame in forecasts.groupby(["model_name", "horizon_step"]):
        horizon_rows.append(
            {
                "model_name": keys[0],
                "horizon_step": keys[1],
                "smape": smape(frame["actual"].to_numpy(), frame["prediction"].to_numpy()),
            }
        )
    horizon_metrics = pd.DataFrame(horizon_rows)
    return summary, series_metrics, horizon_metrics
