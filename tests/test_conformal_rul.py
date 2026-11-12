"""Tests for production tool-trajectory CV+ RUL intervals."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from train_conformal_rul import (
    _cvplus_rank_pair,
    _life_frac_to_phase,
    build_tool_cvplus_ensemble,
    conformal_predict,
    cvplus_interval,
    prepare_training_data,
    tool_group_holdout_cv,
    tool_trajectory_scores,
)
from milling_dataset import SIGNAL_FEATURE_NAMES


# ── Synthetic data helpers ────────────────────────────────────────────────────

def _make_regression_data(n=500, n_feat=12, noise=10.0, seed=0):
    rng  = np.random.default_rng(seed)
    X    = rng.uniform(0, 1, (n, n_feat))
    true = 100. * (1. - X[:, 0]) + 20. * X[:, 1]
    y    = true + rng.normal(0., noise, n)
    return X, np.maximum(y, 0.)


class TestToolTrajectoryCVPlus:
    def test_one_score_per_phase_uses_trajectory_maximum(self):
        residuals = np.array([1., 4., 2., 7., 3., 5.])
        life = np.array([.9, .8, .5, .4, .2, .1])
        scores = tool_trajectory_scores(residuals, life)
        assert scores["early"] == 4.0
        assert scores["mid"] == 7.0
        assert scores["late"] == 5.0
        assert scores["global"] == 7.0

    def test_cvplus_interval_uses_tool_count_not_row_count(self):
        class ConstantModel:
            def __init__(self, value):
                self.value = value

            def predict(self, X):
                return np.full(len(X), self.value)

        model = {
            "alpha": 0.10,
            "tool_cvplus_ensemble": [
                {
                    "tool_id": tool,
                    "rf": ConstantModel(100.0 + tool),
                    "scores": {
                        "global": 10.0 + tool,
                        "early": 10.0 + tool,
                        "mid": 10.0 + tool,
                        "late": 10.0 + tool,
                    },
                }
                for tool in range(1, 10)
            ],
        }
        lower, upper = cvplus_interval(
            model, np.zeros((500, 12)), damage_frac=0.5
        )
        # Repeating 500 correlated rows cannot create 500 calibration units:
        # only the nine tool candidates determine the bounds.
        assert lower == pytest.approx(90.0)
        assert upper == pytest.approx(128.0)

    def test_ensemble_has_exactly_one_entry_per_tool(self):
        X, y = _make_regression_data(n=120, seed=30)
        groups = np.repeat(np.arange(1, 7), 20)
        machines = np.repeat([1, 1, 2, 2, 3, 3], 20)
        ensemble = build_tool_cvplus_ensemble(
            X, y, groups, machines, seed=3, n_estimators=8
        )
        assert len(ensemble) == 6
        assert {entry["tool_id"] for entry in ensemble} == set(range(1, 7))
        assert all(entry["scores"]["global"] >= 0 for entry in ensemble)

    def test_nested_validation_holds_out_whole_tools(self):
        X, y = _make_regression_data(n=80, seed=31)
        groups = np.repeat(np.arange(1, 5), 20)
        machines = np.repeat([1, 1, 2, 3], 20)
        folds = tool_group_holdout_cv(
            X,
            y,
            groups,
            machines,
            n_folds=4,
            seed=4,
            n_estimators=5,
        )
        assert len(folds) == 4
        assert all(fold["n_cal_tools"] == 3 for fold in folds)
        assert {
            fold["heldout_tools"][0] for fold in folds
        } == set(range(1, 5))

    def test_feature_slope_is_causal(self):
        base = {
            "machine": [1] * 5,
            "tool": [1] * 5,
            "run": [1, 2, 3, 4, 5],
            "cumulated_tool_contact_time": [1., 2., 3., 4., 5.],
            "wear": [1., 2., 4., 7., 11.],
        }
        for index, name in enumerate(SIGNAL_FEATURE_NAMES):
            base[name] = np.linspace(index + 1, index + 2, 5)
        original = prepare_training_data(pd.DataFrame(base))
        changed = dict(base)
        changed["wear"] = [1., 2., 4., 7., 100.]
        modified = prepare_training_data(pd.DataFrame(changed))
        # Future wear at run 5 must not alter the slope feature at run 4.
        assert original["X"][3, 1] == pytest.approx(modified["X"][3, 1])

    def test_first_run_slope_does_not_leak_the_second_run(self):
        # slope_values[1:] = diff(v)/diff(t) can only ever depend on runs
        # >= its own index, so the test above (which perturbs the LAST run
        # and checks a middle one) can never catch a leak at row 0
        # specifically -- that boundary needs its own perturbation-at-run-2
        # test. A tool's first run has no earlier reading to difference
        # against, so its slope feature must be exactly 0.0 regardless of
        # what run 2 records (previously it borrowed run 2's own backward
        # difference, a one-run-ahead leak affecting exactly this row).
        base = {
            "machine": [1] * 4,
            "tool": [1] * 4,
            "run": [1, 2, 3, 4],
            "cumulated_tool_contact_time": [1., 2., 3., 4.],
            "wear": [1., 3., 6., 10.],
        }
        for index, name in enumerate(SIGNAL_FEATURE_NAMES):
            base[name] = np.linspace(index + 1, index + 2, 4)
        original = prepare_training_data(pd.DataFrame(base))
        assert original["X"][0, 1] == pytest.approx(0.0)

        changed = dict(base)
        changed["wear"] = [1., 300., 6., 10.]  # only run 2 changes
        modified = prepare_training_data(pd.DataFrame(changed))
        assert modified["X"][0, 1] == pytest.approx(0.0)
        assert original["X"][0, 1] == pytest.approx(modified["X"][0, 1])

    def test_life_fraction_phase_boundaries(self):
        fracs = np.array([0.05, 0.30, 0.45, 0.60, 0.85, 1.0])
        phases = _life_frac_to_phase(fracs)
        assert phases.tolist() == [
            "late", "late", "mid", "mid", "early", "early"
        ]

    def test_conformal_predict_requires_production_ensemble(self):
        class ConstantModel:
            def predict(self, X):
                return np.full(len(X), 100.0)

        with pytest.raises(ValueError, match="CV\\+ ensemble missing"):
            conformal_predict(
                {"rf": ConstantModel(), "alpha": 0.10},
                np.zeros((2, 12)),
            )


class TestCvPlusRankCollapse:
    """Covers the fix for the review finding that Cov90/Cov95 can be the same
    interval reported twice at small n_cal_tools, not two independent checks.
    """

    def test_ranks_collapse_at_n_cal_tools_8_for_90_vs_95(self):
        # The real pipeline's leave-one-tool-out design: 9 tools total, 8
        # calibration tools per fold -- the exact scenario the review found
        # (conformal_rul_summary.json's mean_cov90/mean_cov95 identical to
        # full float precision).
        assert _cvplus_rank_pair(8, 0.10) == _cvplus_rank_pair(8, 0.05) == (1, 8)

    def test_ranks_do_not_collapse_at_a_larger_n(self):
        # Sanity check the helper actually discriminates when there's enough
        # resolution -- confirms the n=8 collapse is a real small-n effect,
        # not a bug making every alpha pair look identical.
        assert _cvplus_rank_pair(50, 0.10) != _cvplus_rank_pair(50, 0.05)

    def test_tool_group_holdout_cv_flags_the_collapse(self):
        X, y = _make_regression_data(n=80, seed=31)
        groups = np.repeat(np.arange(1, 5), 20)
        machines = np.repeat([1, 1, 2, 3], 20)
        folds = tool_group_holdout_cv(
            X, y, groups, machines, n_folds=4, seed=4, n_estimators=5,
        )
        # 4 tools -> 3 calibration tools/fold, which also collapses
        # (_cvplus_rank_pair(3, 0.10) == _cvplus_rank_pair(3, 0.05)).
        assert all(fold["cov90_cov95_ranks_collapsed"] for fold in folds)
        for fold in folds:
            assert fold["coverage_90"] == fold["coverage_95"]
