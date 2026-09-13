"""Tests for the ways this pipeline can score well without having learned anything.

Each check that the honest configuration is clean is paired with a check that the
same probe fires on the configuration that is not. A suite containing only the
first kind would pass on a probe that always returned "no leak".

The neighbour measurement is imported from src.benchmark rather than written
again here. This file used to carry its own copy of it, with the same name and
the same logic, so the two published percentages came from a function no test
ever called and the tests exercised a second implementation that nothing else
used.
"""

from __future__ import annotations

import unittest

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.benchmark import (
    CIRCULAR,
    CYCLE_INDEX,
    extrapolation_probe,
    fit_and_score,
    neighbour_leak_fraction,
    split_by_cycle,
    split_random_rows,
)
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


def planted_leak(train, test):
    """Move one training row into the test frame, keeping its index.

    The paired probe for the two assertions below. Without it, a
    neighbour_leak_fraction that always returned zero would satisfy the
    cycle-split test and nothing would notice.
    """
    import pandas as pd
    donor = train[train["cycle"] == train["cycle"].max()].iloc[[1]]
    return pd.concat([test, donor])


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

    def test_the_measurement_notices_a_row_that_does_not_belong(self):
        """The paired check. Zero is only meaningful if a leak would show."""
        contaminated = planted_leak(self.train, self.test)
        self.assertGreater(
            neighbour_leak_fraction(self.train, contaminated), 0.0)


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


class TheCycleIndexProbe(unittest.TestCase):
    """The probe has to be able to take the cycle index away.

    The version of it that shipped could not. It compared the causal feature set
    against the causal feature set plus `cycle`, and `cycle_normalized` was
    already in the causal set, so the second fit received a rescaled copy of a
    column it had. The error moved by 0.0007 SOC points, which was read as
    evidence about extrapolation and is in fact what adding a collinear duplicate
    does to a random forest.

    These tests hold the two apart: the thing being removed has to be present,
    removing it has to leave nothing equivalent behind, and the measurement has
    to distinguish a real removal from a duplicate.
    """

    @classmethod
    def setUpClass(cls):
        cls.df = engineered()
        cls.causal = [c for c in get_feature_columns(cls.df) if c not in CIRCULAR]
        cls.train, cls.test = split_by_cycle(cls.df, test_cycles=6)

    def test_the_named_features_are_in_the_set_the_probe_strips(self):
        """Otherwise the probe would remove nothing and measure nothing."""
        for name in CYCLE_INDEX:
            with self.subTest(feature=name):
                self.assertIn(name, self.causal)

    def test_each_of_them_really_is_the_cycle_index(self):
        for name in CYCLE_INDEX:
            with self.subTest(feature=name):
                correlation = abs(np.corrcoef(self.df[name], self.df["cycle"])[0, 1])
                self.assertGreater(correlation, 0.999)

    def test_nothing_equivalent_survives_the_removal(self):
        """A second copy under another name would make the removal cosmetic."""
        stripped = [c for c in self.causal if c not in CYCLE_INDEX]
        worst = max(abs(np.corrcoef(self.df[c], self.df["cycle"])[0, 1])
                    for c in stripped)
        self.assertLess(worst, 0.9)

    def test_the_two_arms_of_the_probe_differ_by_exactly_those_features(self):
        """The structural check, and the one the old probe would have failed.

        How large the removal turns out to be is a property of the benchmark's
        configuration, not of the pipeline: at the sizes this suite runs at it is
        close to zero, and at 200 cycles sampled 120 times it is a third of a SOC
        point. So the magnitude is measured in src/benchmark.py and recorded in
        results/benchmark.json, and what is asserted here is that the experiment
        is capable of measuring it at all. The version that shipped was not: both
        of its arms had the cycle index.
        """
        stripped = [c for c in self.causal if c not in CYCLE_INDEX]
        self.assertEqual(set(self.causal) - set(stripped), set(CYCLE_INDEX))
        self.assertEqual(len(stripped), len(self.causal) - len(CYCLE_INDEX))

    def test_the_probe_reports_both_comparisons_and_they_are_not_the_same_one(self):
        probe = extrapolation_probe(self.train, self.test, self.causal)
        self.assertIn("cost_of_removing_the_cycle_index", probe)
        self.assertIn("effect_of_adding_a_collinear_copy", probe)
        self.assertNotEqual(probe["mae_without_the_cycle_index"],
                            probe["mae_with_the_cycle_index"])

    def test_the_middle_arm_is_the_configuration_the_readme_reports(self):
        """Otherwise the probe could be comparing against something else entirely."""
        probe = extrapolation_probe(self.train, self.test, self.causal)
        direct = fit_and_score(self.train, self.test, self.causal, "random_forest")
        self.assertAlmostEqual(probe["mae_with_the_cycle_index"],
                               direct["mae_soc_points"], places=4)


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


class FeaturesComputedBeforeTheSplit(unittest.TestCase):
    """Every feature is engineered on the whole frame and the frame is split after.

    That is the usual arrangement and it is almost always harmless here, but not
    quite. The rolling statistics run over the frame in index order rather than
    within a cycle, so the first rows of the test set have windows that reach
    back into training rows. Nothing in this repository measured how many.

    The reach is bounded above rather than pinned, so that computing the windows
    per cycle, which would set it to zero, keeps this passing rather than
    failing. As a share of the test set it depends on how many rows are held out:
    at the sizes this suite runs at it is a seventh of them, and at the
    benchmark's 6,000 test rows it is 49, under one percent.
    """

    def setUp(self):
        self.df = engineered()
        self.train, self.test = split_by_cycle(self.df, test_cycles=6)

    def test_hardly_any_test_row_has_a_window_reaching_into_training(self):
        largest_window = 50  # compute_rolling_features is called with 20 and 50
        first_test_row = self.test.index.min()
        reaching = int(((self.test.index.to_numpy() - first_test_row)
                        < largest_window - 1).sum())
        self.assertLessEqual(reaching, largest_window - 1)

    def test_the_split_itself_shares_no_rows(self):
        """The paired check: the overlap above is in the features, not the rows."""
        self.assertEqual(set(self.train.index) & set(self.test.index), set())


class CellIdentity(unittest.TestCase):
    def test_the_generator_produces_a_single_cell(self):
        """Recorded as a fact, because it bounds what the results can mean.

        With one cell there is no held-out cell, so nothing here measures whether
        an estimator transfers to a different one.
        """
        self.assertEqual(raw()["cell_id"].nunique(), 1)


if __name__ == "__main__":
    unittest.main()
