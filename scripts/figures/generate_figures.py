"""Render the three figures in the README from results/benchmark.json.

    python scripts/figures/generate_figures.py

Each figure answers one question:

  01  what the split and the feature set do to the reported error
  02  why one column and a straight line is enough
  03  whether the error grows as the cell ages past the training cycles
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import portfolio_style as ps

RESULTS = ROOT / "results" / "benchmark.json"
FIGURES = ROOT / "docs" / "figures"


def load() -> dict:
    if not RESULTS.exists():
        raise SystemExit("run `python -m src.benchmark` first")
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def figure_01_split_and_features(data: dict) -> None:
    """What does the reported error depend on, besides the model?"""
    results, baselines = data["results"], data["baselines"]

    rows = [
        ("Random forest, random row split,\nall features", "leaky",
         results["random_rows|all|random_forest"]["mae_soc_points"]),
        ("Random forest, held-out cycles,\nall features", "leaky",
         results["by_cycle|all|random_forest"]["mae_soc_points"]),
        ("Coulomb counting, no model at all", "classical",
         baselines["coulomb_counting"]["mae_soc_points"]),
        ("Random forest, random row split,\ncausal features", "leaky",
         results["random_rows|causal|random_forest"]["mae_soc_points"]),
        ("Random forest, held-out cycles,\ncausal features", "honest",
         results["by_cycle|causal|random_forest"]["mae_soc_points"]),
        ("Ridge, held-out cycles,\ncausal features", "honest",
         results["by_cycle|causal|ridge"]["mae_soc_points"]),
    ]
    rows.sort(key=lambda r: r[2])

    fill = {"leaky": ps.AMBER_SOFT, "honest": ps.BLUE, "classical": ps.GREEN_SOFT}
    edge = {"leaky": ps.AMBER, "honest": ps.BLUE, "classical": ps.GREEN}

    fig, ax = plt.subplots(figsize=(11.6, 6.8))
    fig.subplots_adjust(left=0.335, right=0.955, top=0.775, bottom=0.215)

    y = np.arange(len(rows))
    ax.barh(y, [r[2] for r in rows],
            color=[fill[r[1]] for r in rows],
            edgecolor=[edge[r[1]] for r in rows], linewidth=1.5, height=0.66)
    for index, (_, kind, value) in enumerate(rows):
        ax.text(value + 0.011, index, f"{value:.3f}", va="center", fontsize=11,
                color=edge[kind], fontweight="600")

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10.2, color=ps.INK)
    ax.set_xlabel("Mean absolute error, SOC percentage points  (lower is better)",
                  fontsize=11, color=ps.MUTED)
    ax.set_xlim(0, max(r[2] for r in rows) * 1.13)
    ps.clean(ax, grid_axis="x")

    ps.title_block(
        fig, "The same random forest, four different answers",
        "Estimating state of charge on generated discharge cycles. Nothing about the "
        "model changes across the\nfour random forest rows. Only the split and the "
        "feature set do.")

    handles = [
        plt.Line2D([], [], marker="s", linestyle="", markersize=11,
                   markerfacecolor=ps.AMBER_SOFT, markeredgecolor=ps.AMBER,
                   label="inflated by a leak, in the split or the features"),
        plt.Line2D([], [], marker="s", linestyle="", markersize=11,
                   markerfacecolor=ps.BLUE, markeredgecolor=ps.BLUE,
                   label="held-out cycles, no target-derived feature"),
        plt.Line2D([], [], marker="s", linestyle="", markersize=11,
                   markerfacecolor=ps.GREEN_SOFT, markeredgecolor=ps.GREEN,
                   label="no machine learning"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=10.2,
              labelcolor=ps.MUTED, handletextpad=0.7)

    ps.footnote(fig, [
        "Causal features exclude the four built from cumulative charge or elapsed time. "
        "State of charge is defined here as one",
        "minus cumulative amp-hours over cycle capacity, and Coulomb counting is that "
        "definition applied directly, which is why it wins.",
    ])
    ps.save(fig, FIGURES, "01_split_and_features")


def figure_02_circular_feature(data: dict) -> None:
    """Why is one column and a straight line enough?"""
    from src.benchmark import POINTS_PER_CYCLE, SEED
    from src.data_loader import generate_synthetic_battery_data

    raw = generate_synthetic_battery_data(n_cycles=40, points_per_cycle=POINTS_PER_CYCLE,
                                          seed=SEED)
    elapsed = raw.groupby("cycle")["time"].transform(
        lambda t: (t - t.min()) / max(t.max() - t.min(), 1e-9))

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.0, 6.6))
    fig.subplots_adjust(left=0.075, right=0.965, top=0.70, bottom=0.245, wspace=0.235)

    left.scatter(elapsed, raw["soc"], s=5, color=ps.BLUE, alpha=0.16,
                 edgecolors="none", rasterized=True)
    left.plot([0, 1], [1, 0], color=ps.INK, linewidth=1.6, linestyle=(0, (5, 3)),
              label="SOC = 1 - normalised elapsed time")
    left.set_xlabel("Elapsed time within the cycle, normalised", fontsize=11,
                    color=ps.MUTED)
    left.set_ylabel("State of charge", fontsize=11, color=ps.MUTED)
    left.set_title("Normalised time against the target", fontsize=12.5, color=ps.INK,
                   pad=11, loc="left")
    left.legend(loc="upper right", frameon=False, fontsize=10, labelcolor=ps.MUTED)
    left.set_xlim(-0.02, 1.02)
    left.set_ylim(-0.02, 1.02)
    ps.clean(left, grid_axis="both")

    single = data["single_feature_mae_soc_points"]
    honest = data["results"]["by_cycle|causal|random_forest"]["mae_soc_points"]
    names = ["Normalised time,\none straight line",
             f"Random forest,\n{data['features']['causal']} causal features",
             "Voltage alone,\none straight line"]
    values = [single["time_normalized"], honest,
              data["baselines"]["voltage_only_linear"]["mae_soc_points"]]
    colours = [ps.AMBER, ps.BLUE, ps.GREEN]

    bars = right.bar(np.arange(3), values, color=colours, width=0.56)
    for bar, value in zip(bars, values, strict=True):
        right.text(bar.get_x() + bar.get_width() / 2, value * 1.14, f"{value:.3f}",
                   ha="center", fontsize=11.5, color=ps.INK, fontweight="600")
    right.set_xticks(np.arange(3))
    right.set_xticklabels(names, fontsize=10.2, color=ps.INK)
    right.set_yscale("log")
    right.set_ylabel("Mean absolute error, SOC points (log scale)", fontsize=11,
                     color=ps.MUTED)
    right.set_title("What each one achieves on held-out cycles", fontsize=12.5,
                    color=ps.INK, pad=11, loc="left")
    right.set_ylim(min(values) * 0.42, max(values) * 3.1)
    ps.clean(right, grid_axis="y")

    ps.title_block(
        fig, "One column already contains the answer",
        "The generator discharges at a nearly constant current, so the fraction of the "
        "cycle elapsed is the fraction of\ncharge drawn. Feeding that to a model asks "
        "it to invert the definition of its own target.")

    ps.footnote(fig, [
        "Left: forty generated cycles. Right: a straight line on normalised time alone "
        "beats a random forest on all fifty causal",
        "features. That is the signature of a circular feature, not of a good model.",
    ])
    ps.save(fig, FIGURES, "02_circular_feature")


def figure_03_drift(data: dict) -> None:
    """Does the error grow as the cell ages past the training cycles?"""
    drift = data["drift_past_training"]
    #: The far edge of the test window, in cycles past the last training cycle.
    TEST_MARGIN = drift["random_forest"][-1]["cycles_past_training_to"]

    fig, ax = plt.subplots(figsize=(11.4, 6.6))
    fig.subplots_adjust(left=0.088, right=0.79, top=0.765, bottom=0.225)

    for name, colour, marker in (("random_forest", ps.BLUE, "o"),
                                 ("ridge", ps.GREEN, "s")):
        blocks = drift[name]
        centres = [(b["cycles_past_training_from"] + b["cycles_past_training_to"]) / 2
                   for b in blocks]
        bias = [b["bias_soc_points"] for b in blocks]
        ax.plot(centres, bias, color=colour, linewidth=2.0, marker=marker,
                markersize=7)

    ax.axhline(0, color=ps.INK, linewidth=1.1)
    ax.text(0.7, 0.012, "unbiased", fontsize=10, color=ps.MUTED,
            va="bottom", ha="left")
    ax.set_xlabel("Cycles beyond the last cycle seen in training", fontsize=11,
                  color=ps.MUTED)
    ax.set_ylabel("Mean signed error, SOC percentage points", fontsize=11, color=ps.MUTED)
    ps.clean(ax, grid_axis="y")

    # Label each series at its right end rather than with an arrow, which had to
    # cross the other line to reach its target.
    for name, colour, label in (("random_forest", ps.BLUE, "Random forest"),
                                ("ridge", ps.GREEN, "Ridge")):
        last = drift[name][-1]
        centre = (last["cycles_past_training_from"] + last["cycles_past_training_to"]) / 2
        ax.text(centre + 1.4, last["bias_soc_points"],
                f"{label}\n{last['bias_soc_points']:+.2f} SOC points",
                fontsize=10.4, color=colour, va="center", ha="left")
    ax.set_xlim(0, TEST_MARGIN)

    ps.title_block(
        fig, "The remaining error is drift, not noise",
        "Both models are trained on cycles 1 to 150 and scored on 151 to 200, where the "
        "cell has faded further than\nanything they were shown.")

    ps.footnote(fig, [
        "The random forest reads progressively low as the cell ages, because it cannot "
        "extrapolate past its training range. Ridge",
        "stays near unbiased but fits the nonlinear voltage curve worse, so its total "
        "error is larger. Neither shows in one average.",
    ])
    ps.save(fig, FIGURES, "03_drift_past_training")


def main() -> int:
    ps.apply()
    data = load()
    print(f"  reading {RESULTS.relative_to(ROOT).as_posix()}")
    figure_01_split_and_features(data)
    figure_02_circular_feature(data)
    figure_03_drift(data)
    print(f"  figures in {FIGURES.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
