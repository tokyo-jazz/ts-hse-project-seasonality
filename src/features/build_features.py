from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSet:
    name: str
    use_seasonal_lags: bool = False
    use_month: bool = False
    use_fourier: bool = False


FEATURE_SETS = [
    FeatureSet(name="lags"),
    FeatureSet(name="lags_and_seasonal", use_seasonal_lags=True),
    FeatureSet(name="lags_and_month", use_month=True),
    FeatureSet(name="lags_and_fourier", use_fourier=True),
    FeatureSet(name="lags_seasonal_month", use_seasonal_lags=True, use_month=True),
    FeatureSet(name="lags_seasonal_fourier", use_seasonal_lags=True, use_fourier=True),
]


def make_origin_positions(length: int, horizon: int, n_windows: int) -> list[int]:
    return [length - horizon * step - 1 for step in range(n_windows, 0, -1)]


def make_fourier_features(step_index: int, season_length: int, order: int) -> dict[str, float]:
    features = {}
    for harmonic in range(1, order + 1):
        angle = 2 * pi * harmonic * step_index / season_length
        features[f"fourier_sin_{harmonic}"] = np.sin(angle)
        features[f"fourier_cos_{harmonic}"] = np.cos(angle)
    return features


def build_training_rows(
    data: pd.DataFrame,
    origin_by_series: dict[str, int],
    horizon: int,
    season_length: int,
    base_lags: list[int],
    seasonal_lags: list[int],
    fourier_order: int,
    feature_set: FeatureSet,
) -> pd.DataFrame:
    used_seasonal_lags = seasonal_lags if feature_set.use_seasonal_lags else []
    max_lag = max(base_lags + used_seasonal_lags)
    rows = []

    for series_id, series_frame in data.groupby("series_id"):
        values = series_frame["y"].to_numpy()
        dates = series_frame["ds"].to_numpy()
        origin = origin_by_series[series_id]
        last_cutoff = origin - horizon

        for cutoff in range(max_lag, last_cutoff + 1):
            history = values[: cutoff + 1]
            for horizon_step in range(1, horizon + 1):
                target_index = cutoff + horizon_step
                row = {
                    "series_id": series_id,
                    "cutoff": cutoff,
                    "horizon_step": horizon_step,
                    "target": values[target_index],
                }

                for lag in base_lags:
                    row[f"lag_{lag}"] = history[-lag]

                if feature_set.use_seasonal_lags:
                    for lag in used_seasonal_lags:
                        row[f"lag_{lag}"] = history[-lag]

                forecast_date = pd.Timestamp(dates[target_index])
                if feature_set.use_month:
                    row["month"] = str(forecast_date.month)

                if feature_set.use_fourier:
                    row.update(make_fourier_features(cutoff + horizon_step, season_length, fourier_order))

                rows.append(row)

    return pd.DataFrame(rows)


def build_prediction_rows(
    data: pd.DataFrame,
    origin_by_series: dict[str, int],
    horizon: int,
    season_length: int,
    base_lags: list[int],
    seasonal_lags: list[int],
    fourier_order: int,
    feature_set: FeatureSet,
) -> pd.DataFrame:
    used_seasonal_lags = seasonal_lags if feature_set.use_seasonal_lags else []
    rows = []

    for series_id, series_frame in data.groupby("series_id"):
        values = series_frame["y"].to_numpy()
        dates = series_frame["ds"].to_numpy()
        origin = origin_by_series[series_id]
        history = values[: origin + 1]

        for horizon_step in range(1, horizon + 1):
            target_index = origin + horizon_step
            row = {
                "series_id": series_id,
                "origin": origin,
                "horizon_step": horizon_step,
                "actual": values[target_index],
            }

            for lag in base_lags:
                row[f"lag_{lag}"] = history[-lag]

            if feature_set.use_seasonal_lags:
                for lag in used_seasonal_lags:
                    row[f"lag_{lag}"] = history[-lag]

            forecast_date = pd.Timestamp(dates[target_index])
            row["forecast_date"] = forecast_date

            if feature_set.use_month:
                row["month"] = str(forecast_date.month)

            if feature_set.use_fourier:
                row.update(make_fourier_features(origin + horizon_step, season_length, fourier_order))

            rows.append(row)

    return pd.DataFrame(rows)
