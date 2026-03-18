from __future__ import annotations

import pandas as pd
from catboost import CatBoostRegressor


def fit_catboost(
    train_frame: pd.DataFrame,
    iterations: int,
    depth: int,
    learning_rate: float,
    random_seed: int,
) -> CatBoostRegressor:
    feature_columns = [column for column in train_frame.columns if column not in {"cutoff", "target"}]
    cat_features = [column for column in ["series_id", "month"] if column in feature_columns]

    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        random_seed=random_seed,
        verbose=False,
    )
    model.fit(train_frame[feature_columns], train_frame["target"], cat_features=cat_features)
    return model


def predict_catboost(model: CatBoostRegressor, prediction_frame: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        column
        for column in prediction_frame.columns
        if column not in {"origin", "forecast_date", "actual"}
    ]
    result = prediction_frame.copy()
    result["prediction"] = model.predict(prediction_frame[feature_columns])
    return result
