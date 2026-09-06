"""Tests for the target itself, and for what the generator actually produces.

The state of charge here is not measured. It is computed by Coulomb counting from
a current the generator chose, and voltage and temperature are then written as
functions of it. These tests pin that structure down, so the README's claim that
the numbers are about signal recovery and not about batteries stays checkable.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.benchmark import coulomb_counting, split_by_cycle, train_mean, voltage_only
from src.data_loader import generate_synthetic_battery_data

CYCLES = 24
POINTS = 60


def raw(seed: int = 42):
    return generate_synthetic_battery_data(n_cycles=CYCLES, points_per_cycle=POINTS,
                                           seed=seed)


class TargetBounds(unittest.TestCase):
    def setUp(self):
        self.df = raw()

    def test_state_of_charge_stays_within_zero_and_one(self):
        self.assertGreaterEqual(self.df["soc"].min(), 0.0)
        self.assertLessEqual(self.df["soc"].max(), 1.0)

    def test_it_spans_most_of_the_range(self):
        """A target squeezed into a narrow band would make any error look small."""
        self.assertLess(self.df["soc"].min(), 0.02)
        self.assertGreater(self.df["soc"].max(), 0.98)

    def test_it_never_increases_within_a_discharge_cycle(self):
        for cycle, block in self.df.groupby("cycle"):
            with self.subTest(cycle=int(cycle)):
                self.assertTrue((np.diff(block["soc"].to_numpy()) <= 1e-12).all())

    def test_there_are_no_missing_or_infinite_targets(self):
        self.assertTrue(np.isfinite(self.df["soc"].to_numpy()).all())


class TargetConstruction(unittest.TestCase):
    """The target is one minus cumulative amp-hours over the cycle's capacity."""

    def test_coulomb_counting_reproduces_it_almost_exactly(self):
        _, test = split_by_cycle(raw(), test_cycles=6)
        score = coulomb_counting(test)
        self.assertLess(score["mae_soc_points"], 0.5)
        self.assertGreater(score["r2"], 0.999)

    def test_it_beats_a_regression_on_voltage(self):
        """The classical method is the bar, and this records where that bar sits."""
        train, test = split_by_cycle(raw(), test_cycles=6)
        self.assertLess(coulomb_counting(test)["mae_soc_points"],
                        voltage_only(train, test)["mae_soc_points"])

    def test_predicting_the_training_mean_explains_nothing(self):
        train, test = split_by_cycle(raw(), test_cycles=6)
        score = train_mean(train, test)
        self.assertLess(abs(score["r2"]), 0.05)
        self.assertGreater(score["mae_soc_points"], 20.0)


class GeneratorStructure(unittest.TestCase):
    def test_the_same_seed_gives_the_same_data(self):
        np.testing.assert_array_equal(raw(seed=7)["soc"].to_numpy(),
                                      raw(seed=7)["soc"].to_numpy())

    def test_a_different_seed_gives_different_data(self):
        self.assertFalse(np.array_equal(raw(seed=7)["voltage"].to_numpy(),
                                        raw(seed=8)["voltage"].to_numpy()))

    def test_capacity_fades_across_cycles(self):
        """The ageing the held-out cycles ask a model to extrapolate into."""
        df = raw()
        duration = df.groupby("cycle")["time"].max()
        self.assertLess(duration.iloc[-1], duration.iloc[0])

    def test_voltage_is_a_function_of_the_target_plus_noise(self):
        """Why recovering SOC from voltage recovers a formula written here.

        Within one cycle the generated voltage is monotone in SOC up to the added
        noise, so the correlation is near perfect by construction.
        """
        block = raw()[lambda d: d["cycle"] == 1]
        correlation = np.corrcoef(block["voltage"], block["soc"])[0, 1]
        self.assertGreater(correlation, 0.99)

    def test_temperature_is_also_written_from_the_target(self):
        block = raw()[lambda d: d["cycle"] == 1]
        correlation = np.corrcoef(block["temperature"], block["soc"])[0, 1]
        self.assertLess(correlation, -0.9)

    def test_every_cycle_has_the_requested_number_of_samples(self):
        counts = raw().groupby("cycle").size().unique()
        np.testing.assert_array_equal(counts, [POINTS])


class PredictionPlausibility(unittest.TestCase):
    def test_coulomb_counting_never_leaves_the_valid_range(self):
        _, test = split_by_cycle(raw(), test_cycles=6)
        self.assertEqual(coulomb_counting(test)["outside_valid_range_fraction"], 0.0)

    def test_the_out_of_range_check_can_actually_fire(self):
        """Otherwise the test above would pass on a check that always returns zero."""
        from src.benchmark import scores

        score = scores(np.array([0.5, 0.5]), np.array([-0.2, 1.4]))
        self.assertEqual(score["outside_valid_range_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
