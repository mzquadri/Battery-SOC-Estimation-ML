"""Fail if the README stops agreeing with the recorded results.

    python scripts/check_repository.py

Two things are checked. The files the README points at exist and compile, and
every number the README quotes still matches `results/benchmark.json`. The second
is the one that matters: a results file is easy to regenerate and a README is easy
to forget, and a repository whose headline numbers no longer describe its own
output is worse than one with no numbers at all.

Each claim is matched against the README with the surrounding words included, so
that a value moving to a different sentence does not accidentally satisfy a check.
"""

from __future__ import annotations

import json
import py_compile
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "benchmark.json"

REQUIRED_FILES = (
    "README.md",
    "requirements.txt",
    "requirements-optional.txt",
    "pyproject.toml",
    "src/data_loader.py",
    "src/feature_engineering.py",
    "src/benchmark.py",
    "src/soc_regression.py",
    "src/clustering_analysis.py",
    "src/genetic_fuzzy.py",
    "src/soh_analysis.py",
    "tests/test_leakage.py",
    "tests/test_soc.py",
    "scripts/figures/generate_figures.py",
    "notebooks/01_EDA_Battery_Data.ipynb",
    "notebooks/02_SOC_Estimation_Models.ipynb",
    "docs/figures/01_split_and_features.png",
    "docs/figures/02_circular_feature.png",
    "docs/figures/03_drift_past_training.png",
)


def claims(data: dict) -> list[tuple[str, str]]:
    """Every quoted number, paired with enough context to anchor it."""
    results = data["results"]
    baselines = data["baselines"]
    single = data["single_feature_mae_soc_points"]
    horizon = {row["held_out_cycles"]: row for row in data["horizon_sensitivity"]}
    bands = {band["soc_from"]: band for band in data["error_by_soc_band"]}
    forest = data["drift_past_training"]["random_forest"]
    honest = results["by_cycle|causal|random_forest"]

    def points(value: float) -> str:
        return f"{value:.3f}"

    return [
        # What the data is.
        (f"produces {data['data']['cycles']} discharge cycles",
         "the number of generated cycles"),
        (f"sampled {data['split']['train_rows'] // 150} times per cycle",
         "the samples per cycle"),
        (f"{data['data']['rows']:,} rows in total", "the row count"),
        (f"{data['split']['test_rows']:,} test rows", "the test row count"),

        # The feature audit.
        (f"produces {data['features']['total']} columns", "the feature count"),
        (f"all {data['features']['causal']} \nremaining features",
         "the causal feature count"),
        (f"target at {data['target_correlation']['time_normalized']:.6f}",
         "the correlation between normalised time and the target"),
        (f"reaches \\*\\*{points(single['time_normalized'])} SOC \npercentage points",
         "the single-feature error"),

        # The split.
        (f"this data {data['neighbour_leak_fraction']['random_rows'] * 100:.1f} percent "
         f"of test rows", "the neighbour leak under a random split"),
        (f"against {data['neighbour_leak_fraction']['by_cycle'] * 100:.0f} percent under "
         f"a cycle-wise split", "the neighbour leak under a cycle split"),
        (f"| 6 | {points(horizon[6]['by_cycle_mae_soc_points'])} | "
         f"{points(horizon[6]['random_rows_mae_soc_points'])} |",
         "the six-cycle horizon row"),
        (f"| 25 | {points(horizon[25]['by_cycle_mae_soc_points'])} | "
         f"{points(horizon[25]['random_rows_mae_soc_points'])} |",
         "the twenty-five-cycle horizon row"),
        (f"| 50 | {points(horizon[50]['by_cycle_mae_soc_points'])} | "
         f"{points(horizon[50]['random_rows_mae_soc_points'])} |",
         "the fifty-cycle horizon row"),
        (f"{results['random_rows|causal|ridge']['mae_soc_points']:.4f} against "
         f"{results['by_cycle|causal|ridge']['mae_soc_points']:.4f}",
         "ridge being unaffected by the split"),

        # What is left.
        (f"reaches {points(honest['mae_soc_points'])} SOC", "the honest error"),
        (f"signed error is {honest['bias_soc_points']:.3f}", "the honest bias"),
        (f"{forest[0]['bias_soc_points']:.3f} over the first ten",
         "the drift over the first ten held-out cycles"),
        (f"{forest[-1]['bias_soc_points']:.3f} over the last ten",
         "the drift over the last ten held-out cycles"),
        (f"from {data['ageing']['mean_train_cycle_capacity_ah']:.3f} Ah in training",
         "the training capacity"),
        (f"to {data['ageing']['mean_test_cycle_capacity_ah']:.3f} Ah in test",
         "the test capacity"),
        (f"error by {abs(data['extrapolation_probe']['difference']):.4f} SOC points",
         "the extrapolation probe"),

        # The verdict table.
        (f"Coulomb counting, no model | {points(baselines['coulomb_counting']['mae_soc_points'])} "
         f"| {baselines['coulomb_counting']['bias_soc_points']:+.3f}",
         "the Coulomb counting row"),
        (f"Random forest, {data['features']['causal']} features | "
         f"{points(honest['mae_soc_points'])} | {honest['bias_soc_points']:+.3f}",
         "the random forest row"),
        (f"Ridge, {data['features']['causal']} features | "
         f"{points(results['by_cycle|causal|ridge']['mae_soc_points'])} | "
         f"{results['by_cycle|causal|ridge']['bias_soc_points']:+.3f}",
         "the ridge row"),
        (f"Voltage alone, straight line | "
         f"{points(baselines['voltage_only_linear']['mae_soc_points'])} | "
         f"{baselines['voltage_only_linear']['bias_soc_points']:+.3f}",
         "the voltage-only row"),
        (f"Training mean | {points(baselines['train_mean']['mae_soc_points'])} | "
         f"{baselines['train_mean']['bias_soc_points']:+.3f}",
         "the training mean row"),
        (f"forest scores {honest['r2']:.5f}", "the honest R squared"),
        (f"restored scores \n{results['by_cycle|all|random_forest']['r2']:.5f}",
         "the circular R squared"),

        # Where the error sits.
        *[(f"| {low:.1f} to {low + 0.2:.1f} | {points(bands[low]['mae_soc_points'])} |",
           f"the {low:.1f} to {low + 0.2:.1f} error band")
          for low in sorted(bands)],
    ]


def ratio_claims(data: dict) -> list[tuple[str, float, float]]:
    """Ratios the README states in words, recomputed rather than trusted."""
    honest = data["results"]["by_cycle|causal|random_forest"]["mae_soc_points"]
    baselines = data["baselines"]
    bands = {band["soc_from"]: band["mae_soc_points"] for band in data["error_by_soc_band"]}
    return [
        ("five times better than a single-sensor linear fit",
         baselines["voltage_only_linear"]["mae_soc_points"] / honest, 5.0),
        ("69 times better than predicting the mean",
         baselines["train_mean"]["mae_soc_points"] / honest, 69.0),
        ("3.6 times worse than integrating the current",
         honest / baselines["coulomb_counting"]["mae_soc_points"], 3.6),
        ("roughly eight times worse on a nearly empty cell",
         bands[0.0] / bands[0.8], 8.0),
        ("the reported error moves by a factor of five",
         honest / data["results"]["random_rows|all|random_forest"]["mae_soc_points"], 5.0),
    ]


def main() -> int:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"  missing required files: {', '.join(missing)}")

    for source in sorted((ROOT / "src").glob("*.py")):
        py_compile.compile(source, doraise=True)
    print(f"  {len(REQUIRED_FILES)} required files present, src/ compiles")

    if not RESULTS.exists():
        raise SystemExit("  results/benchmark.json is missing; run python -m src.benchmark")

    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    flat = re.sub(r"\s+", " ", readme)

    failures = []
    checks = claims(data)
    for expected, description in checks:
        if re.sub(r"\s+", " ", expected.replace("\\*", "*")) not in flat:
            failures.append(f"  README does not state {description}: expected "
                            f"{expected.strip()!r}")
    print(f"  {len(checks) - len(failures)} of {len(checks)} recorded numbers "
          f"found in the README")

    for phrase, actual, stated in ratio_claims(data):
        if abs(actual - stated) / stated > 0.06:
            failures.append(f"  README says {phrase!r} but the recorded ratio is "
                            f"{actual:.2f}, not {stated}")
        if re.sub(r"\s+", " ", phrase) not in flat:
            failures.append(f"  README no longer contains the phrase {phrase!r}")
    print(f"  {len(ratio_claims(data))} stated ratios recomputed from the results")

    if failures:
        print()
        for failure in failures:
            print(failure)
        raise SystemExit(f"\n  {len(failures)} claims in the README no longer match "
                         f"results/benchmark.json")

    print("  README and results/benchmark.json agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
