"""Evaluate SOC estimation on the generated data, with the split and the feature
set both varied, because either one alone can produce a meaningless score.

    python -m src.benchmark

Two things are measured against each other.

**The split.** Rows inside one discharge cycle are nearly identical to their
neighbours, sampled seconds apart. Splitting rows at random puts a sample's own
neighbours on the other side of the boundary, so the model is scored on points it
has effectively already seen. Splitting whole cycles does not.

**The feature set.** SOC here is defined by Coulomb counting: one minus the
cumulative amp-hours drawn, divided by the cycle's capacity. Several engineered
features are functions of that same cumulative quantity, `cycle_capacity_ah` most
directly and `time_normalized` as its monotone proxy under constant current.
Giving those to a model asks it to invert the target's own definition.

Nothing here is evidence about real battery estimation. The generator builds
voltage and temperature as closed-form functions of the SOC it just computed, so
recovering SOC from them recovers a formula this repository wrote. The point of
the numbers below is the comparison between columns, not their size.

Writes results/benchmark.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from .data_loader import generate_synthetic_battery_data
from .feature_engineering import engineer_all_features, get_feature_columns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "benchmark.json"

SEED = 42
N_CYCLES = 200
#: Samples per discharge cycle. The generator can produce 500, but a battery
#: management system logs at a fixed interval, and 120 points over an hour-long
#: discharge is one reading every thirty seconds. The thinner sampling keeps the
#: cycle structure the leakage argument depends on and makes the run reproducible
#: in about a minute.
POINTS_PER_CYCLE = 120
TEST_CYCLES = 50

#: Features that are functions of the cumulative charge or elapsed time within a
#: cycle. SOC is defined from exactly those quantities, so supplying them asks
#: the model to restate the target rather than to estimate it.
CIRCULAR = {
    "time_normalized",     # (t - t_start) / cycle duration; equals 1 - SOC at constant current
    "cycle_capacity_ah",   # cumulative amp-hours, the numerator of the SOC definition
    "energy_wh",           # cumulative power integral, the same quantity scaled by voltage
    "temp_integral",       # expanding mean, carries elapsed time
}


def soc_mae_points(actual, predicted) -> float:
    """Mean absolute error in SOC percentage points, not fractions."""
    return float(np.abs(np.asarray(predicted) - np.asarray(actual)).mean() * 100.0)


def scores(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    err = predicted - actual
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    return {
        "mae_soc_points": float(np.abs(err).mean() * 100.0),
        "bias_soc_points": float(err.mean() * 100.0),
        "rmse_soc_points": float(np.sqrt((err**2).mean()) * 100.0),
        "r2": 1.0 - float((err**2).sum()) / ss_tot if ss_tot else float("nan"),
        "outside_valid_range_fraction": float(
            ((predicted < 0.0) | (predicted > 1.0)).mean()),
        "n": len(actual),
    }


def split_by_cycle(df: pd.DataFrame, test_cycles: int = TEST_CYCLES):
    """Whole cycles held out. No cycle appears on both sides."""
    boundary = df["cycle"].max() - test_cycles
    return df[df["cycle"] <= boundary].copy(), df[df["cycle"] > boundary].copy()


def split_random_rows(df: pd.DataFrame, test_fraction: float = 0.25, seed: int = SEED):
    """Rows shuffled and cut. Included to show what it does to the score."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    cut = int(len(df) * (1 - test_fraction))
    return df.iloc[order[:cut]].copy(), df.iloc[order[cut:]].copy()


def fit_and_score(train, test, columns, model_name: str):
    """Fit one model on one feature set. Scaler is fitted on the training rows only."""
    x_train = train[columns].to_numpy(dtype=float)
    x_test = test[columns].to_numpy(dtype=float)
    y_train = train["soc"].to_numpy(dtype=float)
    y_test = test["soc"].to_numpy(dtype=float)

    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    if model_name == "ridge":
        model = Ridge(alpha=1.0)
    elif model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=200, max_depth=16, random_state=SEED, n_jobs=-1)
    else:
        raise ValueError(model_name)

    model.fit(x_train, y_train)
    return scores(y_test, model.predict(x_test))


def coulomb_counting(test: pd.DataFrame) -> dict:
    """The classical method, with no machine learning at all.

    Integrate current over time and divide by the cycle's capacity. This is how
    the target was produced, so on generated data it is close to exact. That is
    the point: it shows how little a model has left to contribute here.
    """
    predicted = np.empty(len(test), dtype=float)
    position = 0
    for _, block in test.groupby("cycle", sort=False):
        seconds = block["time"].to_numpy(dtype=float)
        current = block["current"].to_numpy(dtype=float)
        steps = np.diff(seconds, prepend=seconds[0])
        drawn = np.cumsum(current * steps) / 3600.0
        capacity = max(drawn[-1], 1e-9)
        predicted[position:position + len(block)] = np.clip(1.0 - drawn / capacity, 0, 1)
        position += len(block)
    return scores(test["soc"].to_numpy(dtype=float), predicted)


def voltage_only(train, test) -> dict:
    """A single sensor and a straight line, as the floor for any model."""
    model = LinearRegression().fit(train[["voltage"]], train["soc"])
    return scores(test["soc"].to_numpy(dtype=float), model.predict(test[["voltage"]]))


def train_mean(train, test) -> dict:
    value = float(train["soc"].mean())
    return scores(test["soc"].to_numpy(dtype=float), np.full(len(test), value))


def drift_past_training(train, test, columns, block: int = 10) -> dict:
    """Does the error grow the further the test cycle sits past the training data?

    A model that has learned SOC from the sensors should not care how old the
    cell is. A model that has instead memorised the ageing state of the training
    cycles will drift, and the drift will widen with distance. Reported for both
    a random forest and a ridge, because they fail differently.
    """
    out = {}
    boundary = int(train["cycle"].max())
    actual = test["soc"].to_numpy(dtype=float)
    distance = test["cycle"].to_numpy() - boundary

    for name in ("random_forest", "ridge"):
        x_train = train[columns].to_numpy(dtype=float)
        scaler = StandardScaler().fit(x_train)
        model = (RandomForestRegressor(n_estimators=200, max_depth=16,
                                       random_state=SEED, n_jobs=-1)
                 if name == "random_forest" else Ridge(alpha=1.0))
        model.fit(scaler.transform(x_train), train["soc"].to_numpy(dtype=float))
        err = model.predict(
            scaler.transform(test[columns].to_numpy(dtype=float))) - actual

        blocks = []
        for low in range(0, TEST_CYCLES, block):
            mask = (distance > low) & (distance <= low + block)
            if mask.sum() == 0:
                continue
            blocks.append({
                "cycles_past_training_from": low + 1,
                "cycles_past_training_to": low + block,
                "n": int(mask.sum()),
                "bias_soc_points": round(float(err[mask].mean() * 100.0), 4),
                "mae_soc_points": round(float(np.abs(err[mask]).mean() * 100.0), 4),
            })
        out[name] = blocks
    return out


def neighbour_leak_fraction(train, test) -> float:
    """Fraction of test rows whose neighbouring sample in the same cycle is in training.

    Two rows seconds apart in one discharge are nearly the same measurement, so a
    test row whose neighbour was trained on is not really held out. Neighbours
    across a cycle boundary do not count: they sit at opposite ends of the charge
    range and are not near-duplicates.
    """
    cycle_of = dict(zip(train.index, train["cycle"], strict=True))
    leaked = sum(
        1 for position, cycle in zip(test.index, test["cycle"], strict=True)
        if any(cycle_of.get(position + step) == cycle for step in (-1, 1)))
    return leaked / len(test)


def horizon_sensitivity(df, columns, horizons=(6, 25, 50)) -> list:
    """How the two splits respond to holding out more cycles.

    The size of the inflation from a random split is not a fixed number. What the
    leak conceals is ageing drift, and drift needs cycles to accumulate, so the
    cycle-wise error grows with the horizon while the random-row error, which does
    not depend on it at all, stays put.
    """
    random_train, random_test = split_random_rows(df)
    random_mae = fit_and_score(random_train, random_test, columns,
                               "random_forest")["mae_soc_points"]
    rows = []
    for horizon in horizons:
        train, test = split_by_cycle(df, test_cycles=horizon)
        rows.append({
            "held_out_cycles": horizon,
            "by_cycle_mae_soc_points": round(
                fit_and_score(train, test, columns, "random_forest")["mae_soc_points"], 4),
            "random_rows_mae_soc_points": round(random_mae, 4),
        })
    return rows


def extrapolation_probe(train, test, columns) -> dict:
    """Is the drift a missing feature, or an inability to extrapolate?

    If the random forest read low merely because it could not see how old the
    cell was, handing it the cycle number would fix it. Every test row sits beyond
    the training range of that column, so a tree cannot extrapolate along it and
    the error should barely move. Recorded because the alternative explanation is
    the obvious one and deserves to be ruled out rather than assumed away.
    """
    without = fit_and_score(train, test, columns, "random_forest")
    with_cycle = fit_and_score(train, test, [*columns, "cycle"], "random_forest")
    return {
        "mae_without_cycle_number": round(without["mae_soc_points"], 4),
        "mae_with_cycle_number": round(with_cycle["mae_soc_points"], 4),
        "difference": round(
            with_cycle["mae_soc_points"] - without["mae_soc_points"], 4),
        "test_rows_beyond_training_cycle_range_fraction": round(
            float((test["cycle"] > train["cycle"].max()).mean()), 4),
    }


def error_by_soc_band(train, test, columns) -> list:
    """Where in the SOC range the error sits, on the honest configuration."""
    x_train = train[columns].to_numpy(dtype=float)
    scaler = StandardScaler().fit(x_train)
    model = RandomForestRegressor(n_estimators=200, max_depth=16, random_state=SEED,
                                  n_jobs=-1).fit(scaler.transform(x_train),
                                                train["soc"].to_numpy(dtype=float))
    predicted = model.predict(scaler.transform(test[columns].to_numpy(dtype=float)))
    actual = test["soc"].to_numpy(dtype=float)

    bands = []
    edges = np.arange(0.0, 1.01, 0.2)
    for low, high in pairwise(edges):
        mask = (actual >= low) & (actual < high) if high < 1.0 else (actual >= low)
        if mask.sum() == 0:
            continue
        bands.append({
            "soc_from": round(float(low), 2),
            "soc_to": round(float(high), 2),
            "n": int(mask.sum()),
            "mae_soc_points": round(soc_mae_points(actual[mask], predicted[mask]), 4),
        })
    return bands


def main(n_cycles: int = N_CYCLES, out: Path = OUT) -> int:
    print(f"  generating {n_cycles} synthetic discharge cycles at "
          f"{POINTS_PER_CYCLE} samples each, seed {SEED}")
    raw = generate_synthetic_battery_data(n_cycles=n_cycles,
                                          points_per_cycle=POINTS_PER_CYCLE,
                                          seed=SEED)
    df = engineer_all_features(raw).replace([np.inf, -np.inf], np.nan).dropna()

    all_columns = get_feature_columns(df)
    causal_columns = [c for c in all_columns if c not in CIRCULAR]
    print(f"  {len(df):,} rows, {df['cycle'].nunique()} cycles, "
          f"{df['cell_id'].nunique()} cell")
    print(f"  {len(all_columns)} features, {len(causal_columns)} after removing "
          f"the {len(CIRCULAR)} derived from cumulative charge or time\n")

    train_cycle, test_cycle = split_by_cycle(df)
    train_random, test_random = split_random_rows(df)

    results = {}
    header = f"  {'split':<14} {'features':<9} {'model':<15} {'MAE (SOC pts)':>14}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for split_name, (train, test) in (("by_cycle", (train_cycle, test_cycle)),
                                      ("random_rows", (train_random, test_random))):
        for feature_name, columns in (("all", all_columns), ("causal", causal_columns)):
            for model_name in ("ridge", "random_forest"):
                key = f"{split_name}|{feature_name}|{model_name}"
                results[key] = fit_and_score(train, test, columns, model_name)
                results[key].update({"split": split_name, "features": feature_name,
                                     "model": model_name})
                print(f"  {split_name:<14} {feature_name:<9} {model_name:<15} "
                      f"{results[key]['mae_soc_points']:14.4f}")

    baselines = {
        "coulomb_counting": coulomb_counting(test_cycle),
        "voltage_only_linear": voltage_only(train_cycle, test_cycle),
        "train_mean": train_mean(train_cycle, test_cycle),
    }
    print("\n  baselines on the held-out cycles:")
    for name, score in baselines.items():
        print(f"    {name:<22} MAE {score['mae_soc_points']:8.4f} SOC points   "
              f"R2 {score['r2']:8.4f}")

    leak = {
        "by_cycle": round(neighbour_leak_fraction(train_cycle, test_cycle), 4),
        "random_rows": round(neighbour_leak_fraction(train_random, test_random), 4),
    }
    print("\n  test rows whose neighbouring sample in the same cycle is in training:")
    for name, fraction in leak.items():
        print(f"    {name:<14} {fraction * 100:6.2f} percent")

    horizon = horizon_sensitivity(df, causal_columns)
    print("\n  how each split responds to a longer held-out horizon:")
    print(f"    {'held out':>9} {'by cycle':>10} {'random rows':>12}")
    for entry in horizon:
        print(f"    {entry['held_out_cycles']:>9} "
              f"{entry['by_cycle_mae_soc_points']:10.4f} "
              f"{entry['random_rows_mae_soc_points']:12.4f}")

    target_correlation = {
        column: round(float(abs(np.corrcoef(df[column], df["soc"])[0, 1])), 6)
        for column in ("time_normalized", "cycle_capacity_ah", "voltage", "temperature")
        if column in df.columns
    }

    probe = extrapolation_probe(train_cycle, test_cycle, causal_columns)
    print("\n  is the drift a missing feature or an inability to extrapolate?")
    print(f"    without the cycle number  MAE {probe['mae_without_cycle_number']:.4f}")
    print(f"    with the cycle number     MAE {probe['mae_with_cycle_number']:.4f}"
          f"   (difference {probe['difference']:+.4f})")
    print(f"    test rows beyond the training range of that column: "
          f"{probe['test_rows_beyond_training_cycle_range_fraction'] * 100:.0f} percent")

    drift = drift_past_training(train_cycle, test_cycle, causal_columns)
    print("\n  error against distance past the last training cycle, causal features:")
    for name, blocks in drift.items():
        print(f"    {name}:")
        for entry in blocks:
            print(f"      cycles +{entry['cycles_past_training_from']:>2} to "
                  f"+{entry['cycles_past_training_to']:<3}  "
                  f"bias {entry['bias_soc_points']:+7.3f}   "
                  f"MAE {entry['mae_soc_points']:6.3f} SOC points")

    bands = error_by_soc_band(train_cycle, test_cycle, causal_columns)
    print("\n  error by SOC band, random forest on causal features, held-out cycles:")
    for band in bands:
        print(f"    {band['soc_from']:.1f} to {band['soc_to']:.1f}   "
              f"MAE {band['mae_soc_points']:7.3f} SOC points   n={band['n']:,}")

    single = {}
    for column in ("time_normalized", "cycle_capacity_ah", "voltage"):
        if column not in df.columns:
            continue
        model = LinearRegression().fit(train_cycle[[column]], train_cycle["soc"])
        single[column] = round(
            soc_mae_points(test_cycle["soc"], model.predict(test_cycle[[column]])), 4)
    print("\n  a single feature alone, straight line, held-out cycles:")
    for column, mae in single.items():
        print(f"    {column:<20} MAE {mae:8.4f} SOC points")

    honest = results["by_cycle|causal|random_forest"]["mae_soc_points"]
    leaky_split = results["random_rows|causal|random_forest"]["mae_soc_points"]
    leaky_features = results["by_cycle|all|random_forest"]["mae_soc_points"]

    payload = {
        "environment": {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "data": {
            "source": "generated by src.data_loader.generate_synthetic_battery_data",
            "is_synthetic": True,
            "seed": SEED,
            "cycles": int(df["cycle"].nunique()),
            "cells": int(df["cell_id"].nunique()),
            "rows": len(df),
            "target": "SOC by Coulomb counting, 1 - cumulative Ah / cycle capacity",
            "warning": "voltage and temperature are generated as closed-form "
                       "functions of this SOC, so recovering it recovers a formula "
                       "written in this repository",
        },
        "features": {
            "total": len(all_columns),
            "causal": len(causal_columns),
            "removed_as_circular": sorted(CIRCULAR),
        },
        "split": {
            "by_cycle_test_cycles": TEST_CYCLES,
            "train_rows": len(train_cycle),
            "test_rows": len(test_cycle),
            "setting": "held-out cycles from the same cell; not a held-out cell "
                       "and not a held-out temperature",
        },
        "results": results,
        "baselines": baselines,
        "error_by_soc_band": bands,
        "neighbour_leak_fraction": leak,
        "horizon_sensitivity": horizon,
        "extrapolation_probe": probe,
        "target_correlation": target_correlation,
        "drift_past_training": drift,
        "ageing": {
            "note": "the generator fades capacity by 0.1 percent per cycle and "
                    "drops the voltage curve by 0.1 mV per cycle, so the test "
                    "cycles are in a state the training cycles never reach",
            "mean_train_cycle_capacity_ah": round(float(
                train_cycle.groupby("cycle")["cycle_capacity_ah"].max().mean()), 4),
            "mean_test_cycle_capacity_ah": round(float(
                test_cycle.groupby("cycle")["cycle_capacity_ah"].max().mean()), 4),
            "test_rows_beyond_training_cycle_range_fraction": round(float(
                (test_cycle["cycle"] > train_cycle["cycle"].max()).mean()), 4),
        },
        "single_feature_mae_soc_points": single,
        "summary": {
            "honest_mae_soc_points": honest,
            "random_split_mae_soc_points": leaky_split,
            "circular_features_mae_soc_points": leaky_features,
            "random_split_understates_error_by": round(honest / leaky_split, 1)
            if leaky_split else None,
            "circular_features_understate_error_by": round(honest / leaky_features, 1)
            if leaky_features else None,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\n  honest configuration: {honest:.3f} SOC points")
    print(f"  a random row split reports {leaky_split:.3f}, "
          f"{honest / leaky_split:.0f} times smaller")
    print(f"  keeping the circular features reports {leaky_features:.3f}, "
          f"{honest / leaky_features:.0f} times smaller")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=N_CYCLES)
    parser.add_argument("--out", type=Path, default=OUT,
                        help="where to write the results, for a run that "
                             "should not overwrite the recorded one")
    arguments = parser.parse_args()
    raise SystemExit(main(arguments.cycles, arguments.out))
