from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

from src.evaluation.metrics import summarize_metrics
from src.evaluation.seasonality_analysis import make_seasonality_groups, summarize_by_seasonality_group
from src.features.build_features import (
    FEATURE_SETS,
    build_prediction_rows,
    build_training_rows,
    make_origin_positions,
)
from src.models.baselines import run_baseline_backtest
from src.models.catboost_model import fit_catboost, predict_catboost


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_origins_by_window(data: pd.DataFrame, horizon: int, n_windows: int) -> list[dict[str, int]]:
    origins_by_window = []
    series_lengths = data.groupby("series_id").size().to_dict()

    for window_index in range(n_windows):
        window_origins = {}
        for series_id, length in series_lengths.items():
            positions = make_origin_positions(length, horizon, n_windows)
            window_origins[series_id] = positions[window_index]
        origins_by_window.append(window_origins)

    return origins_by_window


def run_catboost_backtest(
    data: pd.DataFrame,
    origins_by_window: list[dict[str, int]],
    horizon: int,
    season_length: int,
    base_lags: list[int],
    seasonal_lags: list[int],
    fourier_order: int,
    model_config: dict,
) -> pd.DataFrame:
    rows = []

    for feature_set in FEATURE_SETS:
        for window_index, origin_by_series in enumerate(origins_by_window, start=1):
            train_frame = build_training_rows(
                data=data,
                origin_by_series=origin_by_series,
                horizon=horizon,
                season_length=season_length,
                base_lags=base_lags,
                seasonal_lags=seasonal_lags,
                fourier_order=fourier_order,
                feature_set=feature_set,
            )
            prediction_frame = build_prediction_rows(
                data=data,
                origin_by_series=origin_by_series,
                horizon=horizon,
                season_length=season_length,
                base_lags=base_lags,
                seasonal_lags=seasonal_lags,
                fourier_order=fourier_order,
                feature_set=feature_set,
            )

            model = fit_catboost(train_frame, **model_config)
            prediction_frame = predict_catboost(model, prediction_frame)
            prediction_frame["model_name"] = feature_set.name
            prediction_frame["window"] = window_index
            prediction_frame["train_history"] = prediction_frame["series_id"].map(
                lambda series_id: data.loc[
                    (data["series_id"] == series_id) & (data["t"] <= origin_by_series[series_id]),
                    "y",
                ].tolist()
            )
            rows.append(prediction_frame)

    return pd.concat(rows, ignore_index=True)


def save_outputs(
    forecasts: pd.DataFrame,
    summary: pd.DataFrame,
    series_metrics: pd.DataFrame,
    horizon_metrics: pd.DataFrame,
    seasonality_group_metrics: pd.DataFrame,
    seasonality_groups: pd.DataFrame,
    output_dir: Path,
) -> None:
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    forecasts.to_csv(tables_dir / "forecast_rows.csv", index=False)
    summary.to_csv(tables_dir / "summary_metrics.csv", index=False)
    series_metrics.to_csv(tables_dir / "series_metrics.csv", index=False)
    horizon_metrics.to_csv(tables_dir / "horizon_smape.csv", index=False)
    seasonality_groups.to_csv(tables_dir / "seasonality_strength.csv", index=False)
    seasonality_group_metrics.to_csv(tables_dir / "seasonality_group_metrics.csv", index=False)

    plt.figure(figsize=(10, 6))
    for model_name, frame in horizon_metrics.groupby("model_name"):
        plt.plot(frame["horizon_step"], frame["smape"], marker="o", label=model_name)
    plt.xlabel("Forecast horizon")
    plt.ylabel("sMAPE")
    plt.title("Quality by forecast step")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "smape_by_horizon.png", dpi=160)
    plt.close()


def run_full_backtest(
    data: pd.DataFrame,
    dataset_config: dict,
    feature_config: dict,
    model_config: dict,
    output_dir: Path,
) -> None:
    origins_by_window = make_origins_by_window(
        data=data,
        horizon=dataset_config["horizon"],
        n_windows=dataset_config["n_windows"],
    )

    baseline_forecasts = run_baseline_backtest(
        data=data,
        origins_by_window=origins_by_window,
        horizon=dataset_config["horizon"],
        season_length=dataset_config["season_length"],
    )
    catboost_forecasts = run_catboost_backtest(
        data=data,
        origins_by_window=origins_by_window,
        horizon=dataset_config["horizon"],
        season_length=dataset_config["season_length"],
        base_lags=feature_config["base_lags"],
        seasonal_lags=feature_config["seasonal_lags"],
        fourier_order=feature_config["fourier_order"],
        model_config=model_config,
    )

    forecasts = pd.concat([baseline_forecasts, catboost_forecasts], ignore_index=True)
    summary, series_metrics, horizon_metrics = summarize_metrics(
        forecasts=forecasts,
        season_length=dataset_config["season_length"],
    )
    seasonality_groups = make_seasonality_groups(data, dataset_config["season_length"])
    seasonality_group_metrics = summarize_by_seasonality_group(series_metrics, seasonality_groups)
    save_outputs(
        forecasts,
        summary,
        series_metrics,
        horizon_metrics,
        seasonality_group_metrics,
        seasonality_groups,
        output_dir,
    )
