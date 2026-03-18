from __future__ import annotations

from itertools import product
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.forecasting.theta import ThetaModel


def naive_forecast(train_values: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(train_values[-1], horizon)


def seasonal_naive_forecast(train_values: np.ndarray, horizon: int, season_length: int) -> np.ndarray:
    if len(train_values) < season_length:
        return naive_forecast(train_values, horizon)

    forecast = []
    tail = train_values[-season_length:]
    for step in range(horizon):
        forecast.append(tail[step % season_length])
    return np.asarray(forecast)


def theta_forecast(train_values: np.ndarray, horizon: int, season_length: int) -> np.ndarray:
    model = ThetaModel(train_values, period=season_length)
    fitted = model.fit()
    return np.asarray(fitted.forecast(horizon))


def ets_forecast(train_values: np.ndarray, horizon: int, season_length: int) -> np.ndarray:
    candidates = []
    allow_multiplicative = np.all(train_values > 0)

    for trend, seasonal, damped in product([None, "add"], [None, "add", "mul"], [False, True]):
        if trend is None and damped:
            continue
        if seasonal == "mul" and not allow_multiplicative:
            continue
        if seasonal is not None and len(train_values) < 2 * season_length:
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                fitted = ExponentialSmoothing(
                    train_values,
                    trend=trend,
                    damped_trend=damped,
                    seasonal=seasonal,
                    seasonal_periods=season_length if seasonal else None,
                    initialization_method="estimated",
                ).fit(optimized=True)
        except ValueError:
            continue

        candidates.append((fitted.aic, fitted))

    if not candidates:
        return naive_forecast(train_values, horizon)

    best_model = min(candidates, key=lambda item: item[0])[1]
    return np.asarray(best_model.forecast(horizon))


def run_baseline_backtest(
    data: pd.DataFrame,
    origins_by_window: list[dict[str, int]],
    horizon: int,
    season_length: int,
) -> pd.DataFrame:
    rows = []

    methods = {
        "naive": lambda values: naive_forecast(values, horizon),
        "seasonal_naive": lambda values: seasonal_naive_forecast(values, horizon, season_length),
        "auto_theta": lambda values: theta_forecast(values, horizon, season_length),
        "auto_ets": lambda values: ets_forecast(values, horizon, season_length),
    }

    for window_index, origin_by_series in enumerate(origins_by_window, start=1):
        for series_id, series_frame in data.groupby("series_id"):
            values = series_frame["y"].to_numpy()
            origin = origin_by_series[series_id]
            train_values = values[: origin + 1]
            actual = values[origin + 1 : origin + horizon + 1]

            for method_name, method in methods.items():
                forecast = method(train_values)
                for horizon_step, (prediction, target) in enumerate(zip(forecast, actual), start=1):
                    rows.append(
                        {
                            "model_name": method_name,
                            "window": window_index,
                            "series_id": series_id,
                            "horizon_step": horizon_step,
                            "actual": target,
                            "prediction": prediction,
                            "train_history": train_values.tolist(),
                        }
                    )

    return pd.DataFrame(rows)
