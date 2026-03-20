"""
State of Health (SOH) Analysis for Battery Degradation Tracking.

Monitors battery capacity fade and internal resistance growth over
charge/discharge cycles to estimate remaining useful life.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SOH computation
# ---------------------------------------------------------------------------


def compute_soh(
    cycle_capacities: pd.DataFrame,
    nominal_capacity: float = 2.0,
    capacity_col: str = "capacity_ah",
) -> pd.DataFrame:
    """
    Compute SOH as ratio of current capacity to nominal capacity.

    SOH(k) = C(k) / C_nominal * 100%

    End-of-life (EOL) is typically defined as SOH < 80%.
    """
    df = cycle_capacities.copy()
    df["soh"] = (df[capacity_col] / nominal_capacity * 100).clip(0, 100)
    df["soh_normalized"] = df["soh"] / 100.0

    # Mark EOL
    df["below_eol"] = df["soh"] < 80

    logger.info("SOH range: %.1f%% - %.1f%%", df["soh"].min(), df["soh"].max())
    if df["below_eol"].any():
        eol_cycle = df[df["below_eol"]]["cycle"].min()
        logger.info("EOL (80%%) reached at cycle %d", eol_cycle)

    return df


# ---------------------------------------------------------------------------
# Degradation curve fitting
# ---------------------------------------------------------------------------


def fit_linear_degradation(
    cycles: np.ndarray,
    soh: np.ndarray,
) -> dict:
    """
    Fit linear degradation model: SOH = a - b * cycle

    Returns slope, intercept, and predicted EOL cycle.
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(cycles, soh)

    # Predict EOL (SOH = 80%)
    eol_cycle = (intercept - 80) / (-slope) if slope < 0 else np.inf

    predictions = intercept + slope * cycles
    rmse = np.sqrt(mean_squared_error(soh, predictions))

    logger.info(
        "Linear fit: SOH = %.4f - %.6f * cycle (R²=%.4f)", intercept, -slope, r_value**2
    )

    return {
        "model": "linear",
        "slope": round(slope, 6),
        "intercept": round(intercept, 4),
        "r_squared": round(r_value**2, 4),
        "rmse": round(rmse, 4),
        "predicted_eol_cycle": round(eol_cycle, 0) if np.isfinite(eol_cycle) else None,
        "predictions": predictions,
    }


def fit_exponential_degradation(
    cycles: np.ndarray,
    soh: np.ndarray,
) -> dict:
    """
    Fit exponential degradation: SOH = a * exp(-b * cycle) + c

    Common model for Li-ion battery capacity fade.
    """

    def exp_model(x, a, b, c):
        return a * np.exp(-b * x) + c

    try:
        # Initial guesses
        popt, pcov = optimize.curve_fit(
            exp_model,
            cycles,
            soh,
            p0=[20, 0.001, 80],
            maxfev=10000,
            bounds=([0, 0, 50], [50, 0.1, 100]),
        )

        a, b, c = popt
        predictions = exp_model(cycles, *popt)
        rmse = np.sqrt(mean_squared_error(soh, predictions))
        r2 = r2_score(soh, predictions)

        # Predicted EOL
        if c < 80:
            eol_func = lambda x: exp_model(x, *popt) - 80
            try:
                from scipy.optimize import brentq

                eol_cycle = brentq(eol_func, 0, 10000)
            except (ValueError, RuntimeError):
                eol_cycle = None
        else:
            eol_cycle = None

        logger.info(
            "Exponential fit: SOH = %.2f*exp(-%.6f*cycle) + %.2f (R²=%.4f)", a, b, c, r2
        )

        return {
            "model": "exponential",
            "params": {"a": round(a, 4), "b": round(b, 6), "c": round(c, 4)},
            "r_squared": round(r2, 4),
            "rmse": round(rmse, 4),
            "predicted_eol_cycle": round(eol_cycle, 0) if eol_cycle else None,
            "predictions": predictions,
        }

    except (RuntimeError, ValueError) as e:
        logger.warning("Exponential fit failed: %s", e)
        return {"model": "exponential", "error": str(e)}


def fit_power_law_degradation(
    cycles: np.ndarray,
    soh: np.ndarray,
) -> dict:
    """
    Fit power-law degradation: SOH = 100 - a * cycle^b

    Alternative model for capacity fade.
    """

    def power_model(x, a, b):
        return 100 - a * np.power(x + 1, b)  # +1 to avoid 0^b

    try:
        popt, pcov = optimize.curve_fit(
            power_model,
            cycles,
            soh,
            p0=[0.01, 0.5],
            maxfev=10000,
            bounds=([0, 0], [10, 2]),
        )

        a, b = popt
        predictions = power_model(cycles, *popt)
        rmse = np.sqrt(mean_squared_error(soh, predictions))
        r2 = r2_score(soh, predictions)

        # EOL prediction
        try:
            eol_cycle = ((100 - 80) / a) ** (1 / b) - 1
        except (ZeroDivisionError, ValueError):
            eol_cycle = None

        logger.info("Power-law fit: SOH = 100 - %.4f * cycle^%.4f (R²=%.4f)", a, b, r2)

        return {
            "model": "power_law",
            "params": {"a": round(a, 6), "b": round(b, 4)},
            "r_squared": round(r2, 4),
            "rmse": round(rmse, 4),
            "predicted_eol_cycle": round(eol_cycle, 0) if eol_cycle else None,
            "predictions": predictions,
        }

    except (RuntimeError, ValueError) as e:
        logger.warning("Power-law fit failed: %s", e)
        return {"model": "power_law", "error": str(e)}


# ---------------------------------------------------------------------------
# Internal resistance tracking
# ---------------------------------------------------------------------------


def estimate_internal_resistance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate internal resistance trend from voltage-current data per cycle.

    R_int approx = |delta_V / delta_I| at current steps.
    """
    resistance_per_cycle = []

    for cycle_id in df["cycle"].unique():
        cycle_data = df[df["cycle"] == cycle_id]

        if len(cycle_data) < 10:
            continue

        voltage = cycle_data["voltage"].values
        current = cycle_data["current"].values

        # Find significant current changes
        di = np.diff(current)
        dv = np.diff(voltage)

        # Avoid division by small numbers
        significant = np.abs(di) > 0.01
        if significant.sum() == 0:
            continue

        r_estimates = np.abs(dv[significant] / di[significant])
        # Filter outliers
        r_estimates = r_estimates[(r_estimates > 0.001) & (r_estimates < 5.0)]

        if len(r_estimates) > 0:
            resistance_per_cycle.append(
                {
                    "cycle": cycle_id,
                    "r_internal_mean": round(np.mean(r_estimates), 6),
                    "r_internal_std": round(np.std(r_estimates), 6),
                    "r_internal_median": round(np.median(r_estimates), 6),
                }
            )

    r_df = pd.DataFrame(resistance_per_cycle)
    if len(r_df) > 0:
        # Percentage increase from first cycle
        r0 = r_df["r_internal_mean"].iloc[0]
        r_df["r_increase_pct"] = ((r_df["r_internal_mean"] - r0) / r0 * 100).round(2)
        logger.info(
            "Internal resistance: %.4f -> %.4f ohm (%+.1f%%)",
            r0,
            r_df["r_internal_mean"].iloc[-1],
            r_df["r_increase_pct"].iloc[-1],
        )

    return r_df


# ---------------------------------------------------------------------------
# Remaining Useful Life (RUL) estimation
# ---------------------------------------------------------------------------


def estimate_rul(
    soh_df: pd.DataFrame,
    current_cycle: Optional[int] = None,
    eol_threshold: float = 80.0,
) -> dict:
    """
    Estimate Remaining Useful Life using degradation curve extrapolation.

    Returns RUL estimates from linear, exponential, and power-law models.
    """
    if current_cycle is None:
        current_cycle = soh_df["cycle"].max()

    cycles = soh_df["cycle"].values
    soh = soh_df["soh"].values

    rul_estimates = {}

    # Linear
    linear = fit_linear_degradation(cycles, soh)
    if linear.get("predicted_eol_cycle"):
        rul_estimates["linear"] = {
            "eol_cycle": linear["predicted_eol_cycle"],
            "rul": max(0, linear["predicted_eol_cycle"] - current_cycle),
            "r_squared": linear["r_squared"],
        }

    # Exponential
    exp = fit_exponential_degradation(cycles, soh)
    if exp.get("predicted_eol_cycle"):
        rul_estimates["exponential"] = {
            "eol_cycle": exp["predicted_eol_cycle"],
            "rul": max(0, exp["predicted_eol_cycle"] - current_cycle),
            "r_squared": exp.get("r_squared", 0),
        }

    # Power law
    power = fit_power_law_degradation(cycles, soh)
    if power.get("predicted_eol_cycle"):
        rul_estimates["power_law"] = {
            "eol_cycle": power["predicted_eol_cycle"],
            "rul": max(0, power["predicted_eol_cycle"] - current_cycle),
            "r_squared": power.get("r_squared", 0),
        }

    # Weighted average (by R²)
    if rul_estimates:
        total_r2 = sum(v["r_squared"] for v in rul_estimates.values())
        if total_r2 > 0:
            weighted_rul = sum(
                v["rul"] * v["r_squared"] / total_r2 for v in rul_estimates.values()
            )
            rul_estimates["weighted_avg_rul"] = round(weighted_rul, 0)

    logger.info("RUL estimates at cycle %d: %s", current_cycle, rul_estimates)
    return rul_estimates


# ---------------------------------------------------------------------------
# Full SOH analysis pipeline
# ---------------------------------------------------------------------------


def run_soh_analysis(
    cycle_capacities: pd.DataFrame,
    battery_data: Optional[pd.DataFrame] = None,
    nominal_capacity: float = 2.0,
) -> dict:
    """Run complete SOH analysis pipeline."""
    # Compute SOH
    soh_df = compute_soh(cycle_capacities, nominal_capacity)

    cycles = soh_df["cycle"].values
    soh = soh_df["soh"].values

    # Fit degradation models
    linear = fit_linear_degradation(cycles, soh)
    exponential = fit_exponential_degradation(cycles, soh)
    power_law = fit_power_law_degradation(cycles, soh)

    # RUL estimation
    rul = estimate_rul(soh_df)

    # Internal resistance (if battery data provided)
    resistance = None
    if battery_data is not None:
        resistance = estimate_internal_resistance(battery_data)

    return {
        "soh": soh_df,
        "models": {
            "linear": linear,
            "exponential": exponential,
            "power_law": power_law,
        },
        "rul": rul,
        "resistance": resistance,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Battery SOH Analysis")
    parser.add_argument(
        "--capacities", type=str, default="data/processed/cycle_capacities.csv"
    )
    parser.add_argument("--battery-data", type=str, default=None)
    parser.add_argument("--nominal-capacity", type=float, default=2.0)
    parser.add_argument("--output", type=str, default="results")
    args = parser.parse_args()

    cap_df = pd.read_csv(args.capacities)
    bat_df = pd.read_csv(args.battery_data) if args.battery_data else None

    results = run_soh_analysis(cap_df, bat_df, args.nominal_capacity)

    print("\n=== SOH Summary ===")
    print(results["soh"][["cycle", "capacity_ah", "soh"]].describe())

    print("\n=== Degradation Models ===")
    for name, model in results["models"].items():
        if "error" not in model:
            print(
                f"  {name}: R²={model.get('r_squared', 'N/A')}, "
                f"EOL cycle={model.get('predicted_eol_cycle', 'N/A')}"
            )

    print("\n=== RUL Estimates ===")
    for key, val in results["rul"].items():
        print(f"  {key}: {val}")

    from pathlib import Path

    Path(args.output).mkdir(parents=True, exist_ok=True)
    results["soh"].to_csv(f"{args.output}/soh_analysis.csv", index=False)
