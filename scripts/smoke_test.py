"""Exercise deterministic core preprocessing and SOC metric calculations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import generate_synthetic_battery_data
from src.feature_engineering import engineer_all_features, get_feature_columns
from src.soc_regression import evaluate_regression


def main() -> None:
    data = generate_synthetic_battery_data(n_cycles=3, points_per_cycle=20, seed=7)
    assert len(data) == 60
    assert data["soc"].between(0, 1).all()

    featured = engineer_all_features(data, window_sizes=[5])
    features = get_feature_columns(featured)
    assert features
    assert featured[features].notna().all().all()

    metrics = evaluate_regression(np.array([0.2, 0.8]), np.array([0.3, 0.7]))
    assert metrics["RMSE_pct"] == 10.0
    print("Smoke test passed: synthetic preprocessing and SOC metrics are available.")


if __name__ == "__main__":
    main()
