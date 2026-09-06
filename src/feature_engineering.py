"""
Feature Engineering for Battery SOC Estimation.

Extracts domain-specific features from raw battery voltage, current,
and temperature measurements for ML model input.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Statistical window features
# ---------------------------------------------------------------------------


def compute_rolling_features(
    df: pd.DataFrame,
    window_size: int = 50,
    columns: list | None = None,
) -> pd.DataFrame:
    """
    Compute rolling window statistics for measurement columns.

    Features per column:
        - rolling mean, std, min, max
        - rolling slope (linear trend)
        - rolling rate of change
    """
    df = df.copy()
    columns = columns or ["voltage", "current", "temperature"]
    columns = [c for c in columns if c in df.columns]

    for col in columns:
        prefix = col[:4]  # Short prefix: volt, curr, temp

        # Basic rolling stats
        df[f"{prefix}_mean_{window_size}"] = (
            df[col].rolling(window=window_size, min_periods=1).mean()
        )
        df[f"{prefix}_std_{window_size}"] = (
            df[col].rolling(window=window_size, min_periods=1).std().fillna(0)
        )
        df[f"{prefix}_min_{window_size}"] = (
            df[col].rolling(window=window_size, min_periods=1).min()
        )
        df[f"{prefix}_max_{window_size}"] = (
            df[col].rolling(window=window_size, min_periods=1).max()
        )

        # Rate of change (first difference)
        df[f"{prefix}_diff"] = df[col].diff().fillna(0)

        # Rolling slope via linear regression
        df[f"{prefix}_slope_{window_size}"] = _rolling_slope(df[col], window_size)

    logger.info("Added rolling features (window=%d) for %s", window_size, columns)
    return df


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling linear regression slope."""
    slopes = pd.Series(index=series.index, dtype=float)

    for i in range(len(series)):
        start = max(0, i - window + 1)
        window_data = series.iloc[start : i + 1].values

        if len(window_data) < 3:
            slopes.iloc[i] = 0
            continue

        x = np.arange(len(window_data))
        try:
            slope, _, _, _, _ = stats.linregress(x, window_data)
            slopes.iloc[i] = slope
        except (ValueError, FloatingPointError):
            slopes.iloc[i] = 0

    return slopes


# ---------------------------------------------------------------------------
# Electrochemical features
# ---------------------------------------------------------------------------


def compute_voltage_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract voltage-specific features for SOC estimation.

    - Voltage derivative (dV/dt, dV/dQ)
    - Voltage plateau detection
    - Distance from cutoff voltages
    """
    df = df.copy()

    if "voltage" in df.columns and "time" in df.columns:
        # dV/dt (voltage rate of change)
        dt = df["time"].diff().replace(0, np.nan)
        dv = df["voltage"].diff()
        df["dv_dt"] = (dv / dt).fillna(0)

        # Second derivative (curvature)
        df["d2v_dt2"] = df["dv_dt"].diff().fillna(0)

        # Distance from typical cutoff voltages
        df["dist_from_upper_cutoff"] = 4.2 - df["voltage"]
        df["dist_from_lower_cutoff"] = df["voltage"] - 2.5

        # Voltage plateau indicator (low dV/dt region)
        dv_dt_abs = df["dv_dt"].abs()
        threshold = dv_dt_abs.quantile(0.25) if len(dv_dt_abs) > 0 else 0.001
        df["on_plateau"] = (dv_dt_abs < threshold).astype(int)

    if "voltage" in df.columns and "current" in df.columns:
        # dV/dQ (incremental capacity analysis) - proxy
        dq = df["current"].diff().replace(0, np.nan)
        df["dv_dq"] = (dv / dq).fillna(0)
        # Clip extreme values
        df["dv_dq"] = df["dv_dq"].clip(-10, 10)

    logger.info(
        "Added voltage features: dV/dt, d2V/dt2, cutoff distances, plateau, dV/dQ"
    )
    return df


def compute_thermal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temperature-related features.

    - Temperature rate of change
    - Deviation from ambient
    - Temperature times current interaction (Joule heating proxy)
    """
    df = df.copy()

    if "temperature" in df.columns:
        # dT/dt
        if "time" in df.columns:
            dt = df["time"].diff().replace(0, np.nan)
            df["dt_dt"] = (df["temperature"].diff() / dt).fillna(0)

        # Deviation from assumed ambient (25°C)
        df["temp_deviation"] = df["temperature"] - 25.0

        # Cumulative temperature integral (thermal energy proxy)
        df["temp_integral"] = df["temperature"].expanding().mean()

    # Interaction: I²R heating proxy
    if "current" in df.columns and "temperature" in df.columns:
        df["i2_heating"] = df["current"] ** 2  # Proportional to Joule heating
        df["current_temp_interaction"] = df["current"].abs() * df["temp_deviation"]

    logger.info("Added thermal features")
    return df


# ---------------------------------------------------------------------------
# Energy & power features
# ---------------------------------------------------------------------------


def compute_energy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute energy and power-related features."""
    df = df.copy()

    if all(c in df.columns for c in ["voltage", "current"]):
        # Instantaneous power
        df["power"] = df["voltage"] * df["current"]

        # Cumulative energy (Wh)
        if "time" in df.columns:
            dt = df["time"].diff().fillna(0)
            df["energy_wh"] = (df["power"] * dt / 3600).cumsum()

        # Power moving average
        df["power_ma_20"] = df["power"].rolling(window=20, min_periods=1).mean()

    # Internal resistance estimate: R = dV/dI
    if all(c in df.columns for c in ["voltage", "current"]):
        di = df["current"].diff().replace(0, np.nan)
        dv = df["voltage"].diff()
        df["internal_resistance"] = (dv / di).fillna(0).clip(-1, 1)

    logger.info("Added energy/power features")
    return df


# ---------------------------------------------------------------------------
# Cycle-level features
# ---------------------------------------------------------------------------


def compute_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cycle-level contextual features.

    - Normalized time within cycle
    - Cycle number (aging proxy)
    - Capacity delivered so far in cycle
    """
    df = df.copy()

    if "cycle" in df.columns:
        # Cycle number as aging feature
        df["cycle_normalized"] = df["cycle"] / df["cycle"].max()

        # Normalized time within each cycle
        for cycle_id in df["cycle"].unique():
            mask = df["cycle"] == cycle_id
            cycle_time = df.loc[mask, "time"]
            if len(cycle_time) > 0:
                t_min = cycle_time.min()
                t_max = cycle_time.max()
                if t_max > t_min:
                    df.loc[mask, "time_normalized"] = (cycle_time - t_min) / (
                        t_max - t_min
                    )
                else:
                    df.loc[mask, "time_normalized"] = 0.0

        # Cumulative capacity within cycle (Ah)
        if "current" in df.columns and "time" in df.columns:
            df["cycle_capacity_ah"] = 0.0
            for cycle_id in df["cycle"].unique():
                mask = df["cycle"] == cycle_id
                cycle_data = df.loc[mask]
                if len(cycle_data) < 2:
                    continue
                dt = cycle_data["time"].diff().fillna(0).values
                current = np.abs(cycle_data["current"].values)
                cum_ah = np.cumsum(current * dt) / 3600.0
                df.loc[mask, "cycle_capacity_ah"] = cum_ah

    logger.info("Added cycle-level features")
    return df


# ---------------------------------------------------------------------------
# Full feature pipeline
# ---------------------------------------------------------------------------


def engineer_all_features(
    df: pd.DataFrame,
    window_sizes: list | None = None,
) -> pd.DataFrame:
    """
    Run complete feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned battery data.
    window_sizes : list
        Rolling window sizes (default: [20, 50]).

    Returns
    -------
    pd.DataFrame
        DataFrame with all engineered features.
    """
    window_sizes = window_sizes or [20, 50]

    df = compute_voltage_features(df)
    df = compute_thermal_features(df)
    df = compute_energy_features(df)
    df = compute_cycle_features(df)

    for ws in window_sizes:
        df = compute_rolling_features(df, window_size=ws)

    # Replace any remaining inf
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill NaN from feature engineering
    df = df.ffill().fillna(0)

    n_features = len(
        [c for c in df.columns if c not in ["cell_id", "cycle", "type", "soc"]]
    )
    logger.info("Total features: %d", n_features)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of feature column names (excluding target and identifiers)."""
    exclude = {"cell_id", "cycle", "type", "soc", "time"}
    return [c for c in df.columns if c not in exclude]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Battery Feature Engineering")
    parser.add_argument(
        "--input", type=str, default="data/processed/battery_processed.csv"
    )
    parser.add_argument(
        "--output", type=str, default="data/processed/battery_features.csv"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = engineer_all_features(df)

    features = get_feature_columns(df)
    print(f"\nEngineered {len(features)} features:")
    for f in features:
        print(f"  - {f}")

    df.to_csv(args.output, index=False)
    logger.info("Saved featured data to %s", args.output)
