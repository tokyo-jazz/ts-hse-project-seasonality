from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.load_m4 import load_m4_monthly
from src.evaluation.backtest import run_full_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    with open(project_root / args.config, "r", encoding="utf-8") as file:
        config = json.load(file)

    data = load_m4_monthly(
        raw_dir=project_root / "data" / "raw",
        sample_size=config["dataset"]["sample_size"],
        random_state=config["dataset"]["random_state"],
    )

    run_full_backtest(
        data=data,
        dataset_config=config["dataset"],
        feature_config=config["features"],
        model_config=config["model"],
        output_dir=project_root / "results",
    )


if __name__ == "__main__":
    main()
