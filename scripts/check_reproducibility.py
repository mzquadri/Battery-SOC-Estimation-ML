"""Rerun the benchmark and check that it still supports the same conclusions.

    python scripts/check_reproducibility.py

The recorded numbers in `results/benchmark.json` were produced on one machine with
one set of pinned library versions. Requiring a rerun to match them digit for digit
would make this check fail on a different platform for reasons that have nothing to
do with the code being wrong: a different BLAS, a different thread count, a different
summation order.

So two different standards are applied. The findings the README argues from are
qualitative, and those must hold exactly, because if any of them flips the README
is saying something false. The numbers are compared with a tolerance wide enough to
absorb platform arithmetic and narrow enough that a real regression fails it.

Writes nothing. The recorded results file is not overwritten.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDED = ROOT / "results" / "benchmark.json"

#: Relative tolerance on the headline error values. A rerun that moves any of them
#: by more than this is a change in behaviour, not in floating point.
TOLERANCE = 0.05


def findings(data: dict) -> dict:
    """The qualitative claims the README rests on. Each must survive a rerun."""
    results = data["results"]
    honest = results["by_cycle|causal|random_forest"]["mae_soc_points"]
    forest_drift = [block["bias_soc_points"]
                    for block in data["drift_past_training"]["random_forest"]]
    horizon = {row["held_out_cycles"]: row for row in data["horizon_sensitivity"]}

    return {
        "a random row split reports a smaller error than held-out cycles":
            results["random_rows|causal|random_forest"]["mae_soc_points"] < honest,
        "keeping the circular features reports a smaller error still":
            results["by_cycle|all|random_forest"]["mae_soc_points"] < honest,
        "one column and a straight line beats the causal random forest":
            data["single_feature_mae_soc_points"]["time_normalized"] < honest,
        "Coulomb counting beats every model on this data":
            data["baselines"]["coulomb_counting"]["mae_soc_points"] < honest,
        "the model still beats a voltage-only linear fit":
            honest < data["baselines"]["voltage_only_linear"]["mae_soc_points"],
        "the model still beats predicting the training mean":
            honest < data["baselines"]["train_mean"]["mae_soc_points"],
        "the random forest reads low rather than scattering":
            results["by_cycle|causal|random_forest"]["bias_soc_points"] < -0.1,
        "its drift is monotone in distance past the training cycles":
            all(a >= b for a, b in pairwise(forest_drift)),
        "ridge stays closer to unbiased than the random forest":
            abs(results["by_cycle|causal|ridge"]["bias_soc_points"])
            < abs(results["by_cycle|causal|random_forest"]["bias_soc_points"]),
        "ridge is barely affected by the random split":
            abs(results["random_rows|causal|ridge"]["mae_soc_points"]
                - results["by_cycle|causal|ridge"]["mae_soc_points"]) < 0.01,
        "the honest error grows as more cycles are held out":
            horizon[6]["by_cycle_mae_soc_points"]
            < horizon[25]["by_cycle_mae_soc_points"]
            < horizon[50]["by_cycle_mae_soc_points"],
        "the random split error does not respond to the horizon at all":
            len({row["random_rows_mae_soc_points"]
                 for row in data["horizon_sensitivity"]}) == 1,
        "the cycle-wise split leaks no neighbours":
            data["neighbour_leak_fraction"]["by_cycle"] == 0.0,
        "the random split leaks most of them":
            data["neighbour_leak_fraction"]["random_rows"] > 0.85,
        "the cycle number does not rescue the drift":
            abs(data["extrapolation_probe"]["difference"]) < 0.05,
        "the error is worse on a nearly empty cell than a nearly full one":
            data["error_by_soc_band"][0]["mae_soc_points"]
            > data["error_by_soc_band"][-1]["mae_soc_points"],
    }


def values(data: dict) -> dict:
    """The numbers compared with a tolerance."""
    out = {f"results.{key}": entry["mae_soc_points"]
           for key, entry in data["results"].items()}
    out.update({f"baselines.{key}": entry["mae_soc_points"]
                for key, entry in data["baselines"].items()})
    out.update({f"single_feature.{key}": value
                for key, value in data["single_feature_mae_soc_points"].items()})
    return out


def main() -> int:
    if not RECORDED.exists():
        raise SystemExit("  results/benchmark.json is missing; run python -m src.benchmark")
    recorded = json.loads(RECORDED.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory() as directory:
        fresh_path = Path(directory) / "benchmark.json"
        print("  rerunning the benchmark into a temporary file")
        completed = subprocess.run(
            [sys.executable, "-m", "src.benchmark", "--out", str(fresh_path)],
            cwd=ROOT, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            print(completed.stdout[-2000:])
            print(completed.stderr[-2000:])
            raise SystemExit(f"  the benchmark failed with exit code {completed.returncode}")
        fresh = json.loads(fresh_path.read_text(encoding="utf-8"))

    failures = []

    recorded_findings, fresh_findings = findings(recorded), findings(fresh)
    for description, holds in recorded_findings.items():
        if not holds:
            failures.append(f"  the recorded results no longer support: {description}")
        elif not fresh_findings[description]:
            failures.append(f"  a rerun no longer supports: {description}")
    print(f"  {len(recorded_findings)} findings checked, "
          f"{sum(1 for d in recorded_findings if recorded_findings[d] and fresh_findings[d])} hold")

    recorded_values, fresh_values = values(recorded), values(fresh)
    if recorded_values.keys() != fresh_values.keys():
        failures.append("  the rerun produced a different set of measurements")
    else:
        drifted = 0
        for key, was in recorded_values.items():
            now = fresh_values[key]
            if was and abs(now - was) / abs(was) > TOLERANCE:
                drifted += 1
                failures.append(f"  {key} moved from {was:.4f} to {now:.4f}, "
                                f"more than {TOLERANCE:.0%}")
        print(f"  {len(recorded_values)} values compared, {len(recorded_values) - drifted} "
              f"within {TOLERANCE:.0%}")

    if failures:
        print()
        for failure in failures:
            print(failure)
        raise SystemExit(f"\n  {len(failures)} checks failed")

    print("  the rerun supports the same conclusions as the recorded results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
