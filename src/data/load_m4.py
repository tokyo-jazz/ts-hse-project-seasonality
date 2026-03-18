from __future__ import annotations

from pathlib import Path
import urllib.request

import pandas as pd


DATASET_ROOT = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset"
MONTHLY_FILES = {
    "train": "Train/Monthly-train.csv",
    "test": "Test/Monthly-test.csv",
    "info": "M4-info.csv",
}


def download_m4_monthly(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)

    for key, remote_path in MONTHLY_FILES.items():
        target_path = raw_dir / Path(remote_path).name
        if target_path.exists():
            continue

        url = f"{DATASET_ROOT}/{remote_path}"
        urllib.request.urlretrieve(url, target_path)


def _row_to_series(series_id: str, row: pd.Series, start_date: pd.Timestamp) -> pd.DataFrame:
    values = row.dropna().to_numpy()
    dates = pd.date_range(start=start_date, periods=len(values), freq="MS")
    return pd.DataFrame(
        {
            "series_id": series_id,
            "ds": dates,
            "y": values,
        }
    )


def load_m4_monthly(raw_dir: Path, sample_size: int, random_state: int) -> pd.DataFrame:
    download_m4_monthly(raw_dir)

    train = pd.read_csv(raw_dir / "Monthly-train.csv")
    test = pd.read_csv(raw_dir / "Monthly-test.csv")
    info = pd.read_csv(raw_dir / "M4-info.csv")
    info["StartingDate"] = pd.to_datetime(info["StartingDate"], format="mixed", dayfirst=True)

    monthly_info = info.loc[info["SP"] == "Monthly", ["M4id", "StartingDate"]].copy()
    monthly_info["row_number"] = range(len(monthly_info))
    monthly_info = monthly_info.sample(n=sample_size, random_state=random_state)
    monthly_info = monthly_info.sort_values("M4id").reset_index(drop=True)

    frames = []
    for row in monthly_info.itertuples(index=False):
        train_values = train.iloc[row.row_number, 1:]
        test_values = test.iloc[row.row_number, 1:]
        full_values = pd.concat([train_values, test_values], ignore_index=True)
        frames.append(_row_to_series(row.M4id, full_values, row.StartingDate))

    data = pd.concat(frames, ignore_index=True)
    data["y"] = data["y"].astype(float)
    data["t"] = data.groupby("series_id").cumcount()
    return data
