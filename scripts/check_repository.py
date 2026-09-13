"""Fail if the README stops agreeing with the recorded results.

    python scripts/check_repository.py

Three things are checked. The files the README points at exist and compile, every
number the README quotes still matches `results/benchmark.json`, and the three
committed figures were drawn from that same file. The last two are the ones that
matter: a results file is easy to regenerate and a README is easy to forget, and a
repository whose headline numbers no longer describe its own output is worse than
one with no numbers at all.

Run this before regenerating the figures, not after. It reads what is committed.

Each claim is matched against the README with the surrounding words included, so
that a value moving to a different sentence does not accidentally satisfy a check.
"""

from __future__ import annotations

import json
import py_compile
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "benchmark.json"
FIGURES = ROOT / "docs" / "figures"

#: tEXt key that scripts/figures/portfolio_style.py writes the source values to.
BENCHMARK_KEY = "Benchmark"

PNG_SIGNATURE = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
NUL = bytes([0x00])

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
    probe = data["extrapolation_probe"]
    honest = results["by_cycle|causal|random_forest"]

    def points(value: float) -> str:
        return f"{value:.3f}"

    return [
        # What the data is.
        (f"produces {data['data']['cycles']} discharge cycles",
         "the number of generated cycles"),
        (f"sampled {data['split']['train_rows'] //
                    (data['data']['cycles'] - data['split']['by_cycle_test_cycles'])} "
         f"times per cycle", "the samples per cycle"),
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
        (f"rises to {probe['mae_without_the_cycle_index']:.3f} SOC points",
         "the error with the cycle index removed"),
        # The with-index bias is read from the headline result rather than from
        # the probe's own copy of it. They are the same fit, and the probe rounds
        # to four places, so formatting that to three gave -0.315 where the rest
        # of the README says -0.314.
        (f"bias from {honest['bias_soc_points']:.3f} to "
         f"{probe['bias_without_the_cycle_index']:.3f}",
         "the bias with the cycle index removed"),
        (f"moves the error by {abs(probe['effect_of_adding_a_collinear_copy']):.4f} "
         f"SOC points", "the effect of adding a collinear copy"),

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



def png_text(path: Path) -> dict[str, str]:
    """The tEXt entries of a PNG, read without a third-party imaging library."""
    raw = path.read_bytes()
    if raw[:len(PNG_SIGNATURE)] != PNG_SIGNATURE:
        raise SystemExit(f"  {path.name} is not a PNG")
    entries, offset = {}, len(PNG_SIGNATURE)
    while offset + 8 <= len(raw):
        length = struct.unpack(">I", raw[offset:offset + 4])[0]
        kind = raw[offset + 4:offset + 8]
        if kind == b"tEXt":
            key, _, value = raw[offset + 8:offset + 8 + length].partition(NUL)
            entries[key.decode("latin-1")] = value.decode("latin-1")
        elif kind == b"IEND":
            break
        offset += 12 + length
    return entries


def resolve(data: object, path: str) -> object:
    """Follow a recorded path through the results, indexing lists by number."""
    node = data
    for key in path.split("/"):
        if isinstance(node, dict):
            if key not in node:
                return None
            node = node[key]
        elif isinstance(node, list) and key.isdigit() and int(key) < len(node):
            node = node[int(key)]
        else:
            return None
    return node


def figure_claims(data: dict) -> list[str]:
    """Check the committed figures against the results they were drawn from.

    The figures carry numbers a reader can see, and nothing tied them to
    results/benchmark.json. Regenerating the benchmark and forgetting the figures
    left three images stating superseded values, and continuous integration was
    happy because it rendered them into the working tree and never looked at what
    it had replaced.

    Comparing bytes cannot do this. The figures use whichever of the fonts in
    portfolio_style.FONTS the machine provides, so the same data rendered here
    and on the Linux runner agrees on every number and on none of the pixels.
    What is compared is the values each figure recorded when it was written.
    """
    failures = []
    present = sorted(path.name for path in FIGURES.glob("*.png"))
    expected = sorted(Path(name).name for name in REQUIRED_FILES
                      if name.startswith("docs/figures/"))
    if present != expected:
        failures.append(f"  docs/figures holds {present}, expected {expected}")

    renderers, checked = set(), 0
    for name in expected:
        path = FIGURES / name
        if not path.is_file():
            continue
        text = png_text(path)
        renderers.add(text.get("Software", "unrecorded"))
        if BENCHMARK_KEY not in text:
            failures.append(
                f"  {name} records no source values; rerun "
                f"scripts/figures/generate_figures.py")
            continue
        for recorded_path, drawn in json.loads(text[BENCHMARK_KEY]).items():
            now = resolve(data, recorded_path)
            if now is None:
                failures.append(f"  {name} was drawn from {recorded_path}, "
                                f"which the results no longer contain")
            elif now != drawn:
                failures.append(f"  {name} shows {recorded_path} as {drawn}, "
                                f"the results now say {now}")
            checked += 1

    # A figure regenerated on its own carries a different matplotlib version from
    # the rest as soon as the pinned version moves, which is what a half-finished
    # regeneration looks like.
    if len(renderers) > 1:
        failures.append(f"  the figures were not rendered together: {sorted(renderers)}")

    print(f"  {checked} values behind {len(expected)} figures match the results")
    return failures


def findings_count_claim(data: dict) -> list[str]:
    """The README states how many findings the reproducibility check enforces.

    Counted from that script rather than typed here, because the number moved
    when a finding was added and nothing noticed. Imported rather than
    recalculated, so the two cannot describe different lists.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_reproducibility import findings

    total = len(findings(data))
    flat = re.sub(r"\s+", " ", (ROOT / "README.md").read_text(encoding="utf-8"))
    print(f"  {total} findings enforced by scripts/check_reproducibility.py")
    if f"the {total} findings the README argues from" in flat:
        return []
    return [f"  the README does not say that {total} findings are enforced"]


def test_count_claim() -> list[str]:
    """The README quotes a test count beside the command that produces it.

    Discovered rather than counted from the source, because discovery is what
    the command in the README and the CI job both do, so it is the number a
    reader would see. Nothing else here checked it.
    """
    import unittest

    suite = unittest.defaultTestLoader.discover(
        str(ROOT / "tests"), top_level_dir=str(ROOT))
    total = suite.countTestCases()
    flat = re.sub(r"\s+", " ", (ROOT / "README.md").read_text(encoding="utf-8"))
    print(f"  {total} tests discovered")
    if f"# {total} tests" in flat:
        return []
    stated = re.search(r"unittest discover -s tests\s*# (\d+) tests", flat)
    return [f"  the README says {stated.group(1) if stated else 'no'} tests beside "
            f"the discover command; discovery finds {total}"]


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

    failures += figure_claims(data)
    failures += test_count_claim()
    failures += findings_count_claim(data)

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
        raise SystemExit(f"\n  {len(failures)} claims in the README or the figures "
                         f"no longer match results/benchmark.json")

    print("  README and results/benchmark.json agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
