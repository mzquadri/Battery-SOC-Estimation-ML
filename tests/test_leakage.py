"""Tests for the ways this pipeline can score well without having learned anything.

Each check that the honest configuration is clean is paired with a check that the
same probe fires on the configuration that is not. A suite containing only the
first kind would pass on a probe that always returned "no leak".
"""

from __future__ import annotations

import unittest

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.benchmark import CIRCULAR, split_by_cycle, split_random_rows
from src.data_loader import generate_synthetic_battery_data
from src.feature_engineering import engineer_all_features, get_feature_columns

#: Small enough to keep the suite quick, large enough that a split has both sides.
CYCLES = 24
POINTS = 60


def raw():
    return generate_synthetic_battery_data(n_cycles=CYCLES, points_per_cycle=POINTS,
                                           seed=42)


def engineered():
    df = engineer_all_features(raw())
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def neighbour_leak_fraction(train, test) -> float:
    """Fraction of test rows whose immediate neighbour in time is in training.

    This is the mechanism, stated directly. Two rows seconds apart in the same
    discharge are nearly the same measurement, so a test row with its neighbour
    in training is being scored on something it has effectively seen.

    Only neighbours inside the same cycle count. The last row of one cycle and
    the first row of the next are adjacent in the index but sit at opposite ends
    of the charge range, so they are not near-duplicates and treating them as a
    leak would have reported one on the cycle split.
    """
    cycle_of = dict(zip(train.index, train["cycle"], strict=True))
    leaked = 0
    for position, cycle in zip(test.index, test["cycle"], strict=True):
        if any(cycle_of.get(position + step) == cycle for step in (-1, 1)):
            leaked += 1
    return leaked / len(test)


class CycleSplit(unittest.TestCase):
    """Whole cycles held out, which is the configuration the README reports."""

    def setUp(self):
        self.train, self.test = split_by_cycle(raw(), test_cycles=6)

    def test_no_cycle_appears_on_both_sides(self):
        self.assertEqual(set(self.train["cycle"]) & set(self.test["cycle"]), set())

    def test_the_split_is_chronological(self):
        self.assertLess(self.train["cycle"].max(), self.test["cycle"].min())

    def test_both_sides_are_non_empty(self):
        self.assertGreater(len(self.train), 0)
        self.assertGreater(len(self.test), 0)

    def test_no_test_row_has_its_neighbour_in_training(self):
        self.assertEqual(neighbour_leak_fraction(self.train, self.test), 0.0)


class RandomRowSplit(unittest.TestCase):
    """The configuration the README warns about. These tests assert it is bad."""

    def setUp(self):
        self.train, self.test = split_random_rows(raw(), test_fraction=0.25)

    def test_the_same_cycle_lands_on_both_sides(self):
        shared = set(self.train["cycle"]) & set(self.test["cycle"])
        self.assertEqual(len(shared), self.train["cycle"].nunique())

    def test_almost_every_test_row_has_its_neighbour_in_training(self):
        """The leak, measured rather than asserted."""
        self.assertGreater(neighbour_leak_fraction(self.train, self.test), 0.85)

    def test_it_reports_a_smaller_error_than_the_cycle_split(self):
        """The consequence. If this ever stops holding, the README's claim is wrong.

        Run over a longer horizon than the rest of the suite, because the size of
        the inflation is not a fixed property of the random split. What the leak
        hides is ageing drift, and drift needs cycles to accumulate. Measured over
        six held-out cycles the two splits agree; over fifty the honest error is
        several times larger. That dependence is a real property of the setup, not
        a threshold picked to make a test pass.
        """
        from src.benchmark import fit_and_score

        df = engineer_all_features(generate_synthetic_battery_data(
            n_cycles=100, points_per_cycle=POINTS, seed=42))
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        columns = [c for c in get_feature_columns(df) if c not in CIRCULAR]

        honest = fit_and_score(*split_by_cycle(df, test_cycles=50), columns,
                               "random_forest")
        leaky = fit_and_score(*split_random_rows(df, test_fraction=0.25), columns,
                              "random_forest")
        self.assertLess(leaky["mae_soc_points"], honest["mae_soc_points"])


class CircularFeatures(unittest.TestCase):
    """State of charge is defined from cumulative charge over elapsed time."""

    def setUp(self):
        self.df = engineered()
        self.all_columns = get_feature_columns(self.df)
        self.causal = [c for c in self.all_columns if c not in CIRCULAR]

    def test_every_excluded_feature_is_actually_present(self):
        """Otherwise the exclusion list would be silently doing nothing."""
        for name in CIRCULAR:
            with self.subTest(feature=name):
                self.assertIn(name, self.all_columns)

    def test_none_of_them_survives_into_the_causal_set(self):
        self.assertEqual(set(self.causal) & CIRCULAR, set())

    def test_normalised_time_all_but_determines_the_target(self):
        """The evidence that excluding it is warranted, not a matter of taste."""
        correlation = abs(np.corrcoef(self.df["time_normalized"],
                                      self.df["soc"])[0, 1])
        self.assertGreater(correlation, 0.999)

    def test_no_causal_feature_determines_the_target_that_closely(self):
        worst = max(abs(np.corrcoef(self.df[c], self.df["soc"])[0, 1])
                    for c in self.causal)
        self.assertLess(worst, 0.999)

    def test_the_causal_set_is_not_empty_and_is_smaller(self):
        self.assertGreater(len(self.causal), 20)
        self.assertEqual(len(self.causal), len(self.all_columns) - len(CIRCULAR))


class Normalisation(unittest.TestCase):
    def test_the_scaler_is_fitted_on_training_rows_only(self):
        """Fitting on everything shifts the parameters, which is the tell."""
        df = engineered()
        columns = [c for c in get_feature_columns(df) if c not in CIRCULAR]
        train, _ = split_by_cycle(df, test_cycles=6)

        on_train = StandardScaler().fit(train[columns].to_numpy(dtype=float))
        on_everything = StandardScaler().fit(df[columns].to_numpy(dtype=float))
        self.assertFalse(np.allclose(on_train.mean_, on_everything.mean_))

    def test_transforming_the_test_set_does_not_refit(self):
        df = engineered()
        columns = [c for c in get_feature_columns(df) if c not in CIRCULAR]
        train, test = split_by_cycle(df, test_cycles=6)

        scaler = StandardScaler().fit(train[columns].to_numpy(dtype=float))
        before = scaler.mean_.copy()
        scaler.transform(test[columns].to_numpy(dtype=float))
        np.testing.assert_array_equal(scaler.mean_, before)


class CellIdentity(unittest.TestCase):
    def test_the_generator_produces_a_single_cell(self):
        """Recorded as a fact, because it bounds what the results can mean.

        With one cell there is no held-out cell, so nothing here measures whether
        an estimator transfers to a different one.
        """
        self.assertEqual(raw()["cell_id"].nunique(), 1)


if __name__ == "__main__":
    unittest.main()
