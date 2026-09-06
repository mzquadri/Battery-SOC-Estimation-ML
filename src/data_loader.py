"""
Battery Data Loader and Preprocessor.

Handles loading NASA battery dataset (MATLAB .mat files or CSV exports),
cleaning, resampling, and preparing data for ML models.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATTERY_CELLS = ["B0005", "B0006", "B0007", "B0018"]
CYCLE_TYPES = ["charge", "discharge", "impedance"]

DISCHARGE_COLUMNS = [
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
    "Current_load",
    "Voltage_load",
    "Time",
]

CHARGE_COLUMNS = [
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
    "Current_charge",
    "Voltage_charge",
    "Time",
]


# ---------------------------------------------------------------------------
# MATLAB data loader
# ---------------------------------------------------------------------------


def load_mat_file(filepath: str | Path) -> dict:
    """
    Load NASA battery .mat file using scipy.

    The NASA dataset stores data in nested MATLAB structs:
        battery.cycle[i].type = 'charge'|'discharge'|'impedance'
        battery.cycle[i].data.Voltage_measured, etc.
    """
    from scipy.io import loadmat

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"MAT file not found: {filepath}")

    logger.info("Loading MAT file: %s", filepath)
    mat = loadmat(str(filepath), simplify_cells=True)
    return mat


def parse_nasa_battery(mat_data: dict, cell_id: str = "B0005") -> pd.DataFrame:
    """
    Parse NASA battery MATLAB structure into a flat DataFrame.

    Returns DataFrame with columns:
        cycle, type, time, voltage, current, temperature, capacity
    """
    cycles = mat_data.get(cell_id, {}).get("cycle", [])
    if not cycles:
        # Try alternative structure
        for key in mat_data:
            if isinstance(mat_data[key], dict) and "cycle" in mat_data[key]:
                cycles = mat_data[key]["cycle"]
                break

    if not cycles:
        logger.warning("No cycle data found for cell %s", cell_id)
        return pd.DataFrame()

    records = []
    for i, cycle in enumerate(cycles):
        cycle_type = cycle.get("type", "unknown")
        data = cycle.get("data", {})

        if cycle_type == "discharge":
            n_points = len(data.get("Voltage_measured", []))
            for j in range(n_points):
                records.append(
                    {
                        "cell_id": cell_id,
                        "cycle": i + 1,
                        "type": cycle_type,
                        "time": _safe_index(data.get("Time", []), j),
                        "voltage": _safe_index(data.get("Voltage_measured", []), j),
                        "current": _safe_index(data.get("Current_measured", []), j),
                        "temperature": _safe_index(
                            data.get("Temperature_measured", []), j
                        ),
                        "current_load": _safe_index(data.get("Current_load", []), j),
                        "voltage_load": _safe_index(data.get("Voltage_load", []), j),
                    }
                )

        elif cycle_type == "charge":
            n_points = len(data.get("Voltage_measured", []))
            for j in range(n_points):
                records.append(
                    {
                        "cell_id": cell_id,
                        "cycle": i + 1,
                        "type": cycle_type,
                        "time": _safe_index(data.get("Time", []), j),
                        "voltage": _safe_index(data.get("Voltage_measured", []), j),
                        "current": _safe_index(data.get("Current_measured", []), j),
                        "temperature": _safe_index(
                            data.get("Temperature_measured", []), j
                        ),
                    }
                )

    df = pd.DataFrame(records)
    logger.info(
        "Parsed %d records for cell %s (%d cycles)",
        len(df),
        cell_id,
        df["cycle"].nunique() if len(df) > 0 else 0,
    )
    return df


def _safe_index(arr, idx):
    """Safely index into array or return NaN."""
    try:
        if hasattr(arr, "__len__") and idx < len(arr):
            return float(arr[idx])
    except (TypeError, IndexError, ValueError):
        pass
    return np.nan


# ---------------------------------------------------------------------------
# CSV data loader (for Kaggle exports)
# ---------------------------------------------------------------------------


def load_csv_battery_data(data_dir: str | Path) -> pd.DataFrame:
    """
    Load battery data from CSV files (Kaggle format).

    Expects files like: B0005.csv, B0006.csv, etc.
    Or a single combined file: battery_data.csv
    """
    data_dir = Path(data_dir)

    # Check for single combined file
    combined_path = data_dir / "battery_data.csv"
    if combined_path.exists():
        logger.info("Loading combined CSV: %s", combined_path)
        return pd.read_csv(combined_path)

    # Load individual cell files
    dfs = []
    for cell_file in sorted(data_dir.glob("B*.csv")):
        logger.info("Loading %s", cell_file.name)
        cell_df = pd.read_csv(cell_file)
        cell_df["cell_id"] = cell_file.stem
        dfs.append(cell_df)

    if not dfs:
        logger.warning("No battery CSV files found in %s", data_dir)
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %d total records from %d cells", len(combined), len(dfs))
    return combined


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------


def clean_battery_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate battery measurement data."""
    df = df.copy()
    n_before = len(df)

    # Remove rows with all NaN measurements
    measurement_cols = [
        c
        for c in df.columns
        if c
        in [
            "voltage",
            "current",
            "temperature",
            "Voltage_measured",
            "Current_measured",
            "Temperature_measured",
        ]
    ]
    if measurement_cols:
        df = df.dropna(subset=measurement_cols, how="all")

    # Standardize column names
    rename_map = {
        "Voltage_measured": "voltage",
        "Current_measured": "current",
        "Temperature_measured": "temperature",
        "Current_load": "current_load",
        "Voltage_load": "voltage_load",
        "Current_charge": "current_charge",
        "Voltage_charge": "voltage_charge",
        "Time": "time",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Physical range validation
    if "voltage" in df.columns:
        # Li-ion voltage range: ~2.5V to ~4.2V (allow some margin)
        valid_v = (df["voltage"] >= 1.0) & (df["voltage"] <= 5.0)
        invalid_count = (~valid_v & df["voltage"].notna()).sum()
        if invalid_count > 0:
            logger.warning("Removing %d rows with out-of-range voltage", invalid_count)
            df = df[valid_v | df["voltage"].isna()]

    if "temperature" in df.columns:
        # Temperature range: -20C to 80C
        valid_t = (df["temperature"] >= -20) & (df["temperature"] <= 80)
        invalid_count = (~valid_t & df["temperature"].notna()).sum()
        if invalid_count > 0:
            logger.warning(
                "Removing %d rows with out-of-range temperature", invalid_count
            )
            df = df[valid_t | df["temperature"].isna()]

    logger.info("Cleaned data: %d -> %d rows", n_before, len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# SOC computation (Coulomb counting)
# ---------------------------------------------------------------------------


def compute_soc_coulomb_counting(
    df: pd.DataFrame,
    nominal_capacity_ah: float = 2.0,
    initial_soc: float = 1.0,
) -> pd.DataFrame:
    """
    Compute SOC via Coulomb counting for each discharge cycle.

    SOC(t) = SOC(0) - (1 / Q_nom) * integral(I dt)

    Parameters
    ----------
    df : pd.DataFrame
        Battery data with 'current', 'time', 'cycle', 'type' columns.
    nominal_capacity_ah : float
        Nominal battery capacity in Ah.
    initial_soc : float
        SOC at start of discharge (1.0 = fully charged).
    """
    df = df.copy()
    df["soc"] = np.nan

    for cycle_id in df["cycle"].unique():
        mask = (df["cycle"] == cycle_id) & (df["type"] == "discharge")
        cycle_data = df.loc[mask].copy()

        if len(cycle_data) < 2:
            continue

        # Time in seconds, current in Amps
        time_s = cycle_data["time"].values
        current_a = cycle_data["current"].values

        # Trapezoidal integration of current
        dt = np.diff(time_s)
        avg_current = (current_a[:-1] + current_a[1:]) / 2

        # Cumulative charge consumed (Ah)
        charge_consumed = np.cumsum(avg_current * dt) / 3600.0
        charge_consumed = np.insert(charge_consumed, 0, 0.0)

        # SOC
        soc = initial_soc - charge_consumed / nominal_capacity_ah
        soc = np.clip(soc, 0.0, 1.0)

        df.loc[mask, "soc"] = soc

    logger.info("Computed SOC for %d cycles", df[df["soc"].notna()]["cycle"].nunique())
    return df


# ---------------------------------------------------------------------------
# Cycle capacity extraction
# ---------------------------------------------------------------------------


def extract_cycle_capacities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract discharge capacity for each cycle (for SOH analysis).

    Capacity = integral(|I| dt) during discharge.
    """
    capacities = []

    for cell_id in df["cell_id"].unique() if "cell_id" in df.columns else ["default"]:
        cell_mask = (
            df["cell_id"] == cell_id
            if "cell_id" in df.columns
            else pd.Series(True, index=df.index)
        )
        cell_data = df[cell_mask]

        for cycle_id in cell_data["cycle"].unique():
            cycle_mask = (cell_data["cycle"] == cycle_id) & (
                cell_data["type"] == "discharge"
            )
            cycle_data = cell_data[cycle_mask]

            if len(cycle_data) < 2:
                continue

            time_s = cycle_data["time"].values
            current_a = np.abs(cycle_data["current"].values)

            dt = np.diff(time_s)
            avg_current = (current_a[:-1] + current_a[1:]) / 2
            capacity_ah = np.sum(avg_current * dt) / 3600.0

            avg_temp = (
                cycle_data["temperature"].mean()
                if "temperature" in cycle_data.columns
                else np.nan
            )

            capacities.append(
                {
                    "cell_id": cell_id,
                    "cycle": cycle_id,
                    "capacity_ah": round(capacity_ah, 4),
                    "avg_temperature": round(avg_temp, 2)
                    if not np.isnan(avg_temp)
                    else np.nan,
                    "min_voltage": cycle_data["voltage"].min(),
                    "max_voltage": cycle_data["voltage"].max(),
                }
            )

    cap_df = pd.DataFrame(capacities)
    logger.info("Extracted capacities for %d cycles", len(cap_df))
    return cap_df


# ---------------------------------------------------------------------------
# Train/test split for time series
# ---------------------------------------------------------------------------


def temporal_train_test_split(
    df: pd.DataFrame,
    test_cycles: int = 50,
) -> tuple:
    """
    Split battery data by cycle number (temporal split).

    Last `test_cycles` cycles become the test set.
    """
    max_cycle = df["cycle"].max()
    split_cycle = max_cycle - test_cycles

    train = df[df["cycle"] <= split_cycle].copy()
    test = df[df["cycle"] > split_cycle].copy()

    logger.info(
        "Train: cycles 1-%d (%d rows), Test: cycles %d-%d (%d rows)",
        split_cycle,
        len(train),
        split_cycle + 1,
        max_cycle,
        len(test),
    )
    return train, test


# ---------------------------------------------------------------------------
# Synthetic data generator (for demonstration)
# ---------------------------------------------------------------------------


def generate_synthetic_battery_data(
    n_cycles: int = 200,
    points_per_cycle: int = 500,
    nominal_capacity: float = 2.0,
    degradation_rate: float = 0.001,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic battery discharge data for demonstration.

    Simulates realistic voltage, current, temperature profiles with
    gradual capacity fade (degradation).
    """
    rng = np.random.default_rng(seed)
    records = []

    for cycle in range(1, n_cycles + 1):
        # Capacity degrades linearly
        current_capacity = nominal_capacity * (1 - degradation_rate * cycle)
        current_capacity = max(current_capacity, 0.5)

        # Discharge at constant current (~2A) with noise
        discharge_current = 2.0 + rng.normal(0, 0.05, points_per_cycle)

        # Time points (seconds)
        total_time = current_capacity / 2.0 * 3600  # approx discharge time
        time_points = np.linspace(0, total_time, points_per_cycle)

        # SOC via Coulomb counting
        dt = np.diff(time_points, prepend=0)
        charge_consumed = np.cumsum(discharge_current * dt) / 3600.0
        soc = 1.0 - charge_consumed / current_capacity
        soc = np.clip(soc, 0, 1)

        # Voltage profile (OCV approximation)
        # V(SOC) = a0 + a1*SOC + a2*SOC^2 + a3*log(SOC) + a4*log(1-SOC)
        safe_soc = np.clip(soc, 0.01, 0.99)
        voltage = (
            3.4
            + 0.5 * safe_soc
            + 0.2 * safe_soc**2
            + 0.05 * np.log(safe_soc)
            - 0.02 * np.log(1 - safe_soc)
            + rng.normal(0, 0.01, points_per_cycle)
            - 0.0001 * cycle  # Internal resistance increase
        )

        # Temperature (slight increase during discharge)
        base_temp = 25 + rng.normal(0, 0.5)
        temp_rise = 5 * (1 - soc)  # Temperature rises as battery discharges
        temperature = base_temp + temp_rise + rng.normal(0, 0.2, points_per_cycle)

        for j in range(points_per_cycle):
            records.append(
                {
                    "cell_id": "SYN001",
                    "cycle": cycle,
                    "type": "discharge",
                    "time": time_points[j],
                    "voltage": round(voltage[j], 4),
                    "current": round(discharge_current[j], 4),
                    "temperature": round(temperature[j], 2),
                    "soc": round(soc[j], 4),
                }
            )

    df = pd.DataFrame(records)
    logger.info("Generated synthetic data: %d cycles, %d records", n_cycles, len(df))
    return df


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    use_synthetic: bool = False,
) -> pd.DataFrame:
    """Run full data loading and preprocessing pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_synthetic:
        logger.info("Generating synthetic battery data...")
        df = generate_synthetic_battery_data()
    else:
        input_path = Path(input_path)
        if input_path.suffix == ".mat":
            mat = load_mat_file(input_path)
            df = parse_nasa_battery(mat)
        elif input_path.is_dir():
            df = load_csv_battery_data(input_path)
        else:
            df = pd.read_csv(input_path)

    # Clean
    df = clean_battery_data(df)

    # Compute SOC if not present
    if "soc" not in df.columns:
        df = compute_soc_coulomb_counting(df)

    # Extract cycle capacities
    capacities = extract_cycle_capacities(df)

    # Save
    df.to_csv(output_dir / "battery_processed.csv", index=False)
    capacities.to_csv(output_dir / "cycle_capacities.csv", index=False)
    logger.info("Saved processed data to %s", output_dir)

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Battery Data Loader")
    parser.add_argument(
        "--input",
        type=str,
        default="data/",
        help="Path to raw data (directory, CSV, or MAT file)",
    )
    parser.add_argument(
        "--output", type=str, default="data/processed", help="Output directory"
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Generate synthetic demo data"
    )
    args = parser.parse_args()

    df = run_pipeline(args.input, args.output, use_synthetic=args.synthetic)
    print(f"\nProcessed {len(df)} records, {df['cycle'].nunique()} cycles")
    print(df.describe())
