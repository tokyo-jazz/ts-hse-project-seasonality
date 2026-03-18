from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


def estimate_seasonality_strength(values: np.ndarray, season_length: int) -> float:
    stl = STL(values, period=season_length, robust=True)
    result = stl.fit()
    remainder = result.resid
    seasonal_plus_remainder = result.seasonal + result.resid
    strength = 1.0 - np.var(remainder) / np.var(seasonal_plus_remainder)
    return float(np.clip(strength, 0.0, 1.0))


def make_seasonality_groups(data: pd.DataFrame, season_length: int) -> pd.DataFrame:
    rows = []
    for series_id, series_frame in data.groupby("series_id"):
        strength = estimate_seasonality_strength(series_frame["y"].to_numpy(), season_length)
        rows.append({"series_id": series_id, "seasonality_strength": strength})

    strengths = pd.DataFrame(rows)
    ranked_strength = strengths["seasonality_strength"].rank(method="first")
    strengths["seasonality_group"] = pd.qcut(
        ranked_strength,
        q=3,
        labels=["low", "medium", "high"],
    )
    return strengths


def summarize_by_seasonality_group(series_metrics: pd.DataFrame, seasonality_groups: pd.DataFrame) -> pd.DataFrame:
    merged = series_metrics.merge(seasonality_groups, on="series_id", how="left")
    return (
        merged.groupby(["seasonality_group", "model_name"], as_index=False, observed=False)[["smape", "mase"]]
        .mean()
        .sort_values(["seasonality_group", "smape"])
    )
