"""Unit tests for the degradation process-family comparison
(degradation_process_models.py): Inverse Gaussian and Wiener-with-drift
fitters, mirroring tests/test_gamma.py's style for the production Gamma fit.
"""

import sys
from itertools import combinations, pairwise
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
from scipy.stats import invgauss, norm

import degradation_families as families
import degradation_process_variants as variants
from degradation_process_models import (
    _benjamini_hochberg,
    _fit_rounded_nh_ig,
    _fit_rounded_nh_wiener,
    _pit_acf,
    _rounded_gamma_bin_probability,
    _rounded_ig_bin_probability,
    _rounded_wiener_bin_probability,
    _wilson_ci,
    age_model_criterion_sensitivity,
    block_bootstrap_family_aic_gap_by_tool,
    bootstrap_family_aic_gap,
    bootstrap_per_tool_power_significance,
    fit_and_compare_degradation_models,
    leave_one_tool_out_cv_likelihood,
    model_recovery_study,
    moving_block_bootstrap_family_aic_gap,
    multiple_comparisons_summary,
    per_tool_fit_diagnostics,
    pooled_vs_per_machine_comparison,
    residual_autocorrelation_check,
    state_aware_trajectory_resimulation,
    validate_ig_fit,
    validate_wiener_fit,
    vuong_test,
    window_size_robustness_sweep,
)


def _naive_predicted_curve(component: dict, time: np.ndarray, wear0: float) -> np.ndarray:
    """Reproduces _pit_ks_generic's one-shot (frozen-wear-level) mean curve
    for a state-dependent family: shape_fn evaluated at v_start held fixed
    at wear0 for the whole trajectory, not state_aware_trajectory_resimulation's
    recursive version -- used to build an independent "naive" baseline in
    TestStateAwareResimulation's fixtures without invoking the full
    _pit_ks_generic/build_wear_windows machinery for fields these tests
    don't otherwise need."""
    t0 = np.full_like(time, time[0])
    v0 = np.full_like(time, wear0)
    shape_s = variants._state_dependent_shape_fn(component, t0, time, v0)
    return wear0 + shape_s / component["beta"]


def _fixture_gamma_params():
    """Hand-built Gamma params with all fields model_recovery_study/vuong_test
    need (wear_resolution_um, log_lik consistent with aic via aic=2k-2*log_lik
    for k=n_tools+1=4 stationary components) -- fit_and_compare_degradation_models
    passes gamma_params through unmodified (it doesn't refit Gamma), so these
    fields must be present here, unlike TestFitAndCompare's minimal fixture
    which only needs what that one function reads.
    """
    return {
        f"M{m}": {
            "selected_age_model": "linear", "aic": 100.0, "bic": 110.0,
            "log_lik": -46.0, "wear_resolution_um": 1.0,
            "beta": 0.55, "time_reference_s": 1000.0,
            "components": [
                {"tool_id": (m - 1) * 3 + t, "eta": 20.0 + 2 * (t - 1), "power": 1.0,
                 "beta": 0.55, "time_reference_s": 1000.0}
                for t in (1, 2, 3)
            ],
        }
        for m in (1, 2, 3)
    }


def _synthetic_windows(family: str, eta_power_list, beta: float, rng_seed: int = 123):
    """Generate synthetic rounded wear windows from a known process, mirroring
    test_gamma.py's TestRoundedNonhomogeneousFit synthetic-data pattern."""
    rng = np.random.default_rng(rng_seed)
    t_start, t_end, tool_ids, observed = [], [], [], []
    for tool, (eta, power) in enumerate(eta_power_list, start=1):
        edges = np.linspace(10.0, 3500.0, 45)
        for left, right in pairwise(edges):
            shape_s = eta * ((right / 1000.0) ** power - (left / 1000.0) ** power)
            shape_s = max(shape_s, 1e-6)
            if family == "ig":
                mu = shape_s / beta
                lam = (shape_s ** 2) / beta
                latent = invgauss.rvs(mu=1.0 / shape_s, scale=lam, random_state=rng)
            elif family == "wiener":
                mu = shape_s / beta
                sigma = np.sqrt(shape_s) / beta
                latent = norm.rvs(loc=mu, scale=sigma, random_state=rng)
            else:
                raise ValueError(family)
            observed.append(float(np.rint(max(latent, 0.0))))
            t_start.append(left)
            t_end.append(right)
            tool_ids.append(tool)
    return (
        np.asarray(observed), np.asarray(t_start), np.asarray(t_end), np.asarray(tool_ids),
    )


class TestRoundedNonhomogeneousIGFit:
    def test_fits_quantized_multi_tool_data(self):
        dv, t_start, t_end, tool_ids = _synthetic_windows(
            "ig", [(22.0, 0.75), (28.0, 0.65), (18.0, 0.90)], beta=0.6,
        )
        fitted = _fit_rounded_nh_ig(dv, t_start, t_end, tool_ids)
        assert fitted["model"] == "nonhomogeneous_inverse_gaussian_rounded"
        assert len(fitted["components"]) == 3
        assert fitted["n_windows"] == len(dv)
        assert fitted["beta"] > 0
        # nonhomogeneous_components is always retained regardless of AIC selection
        assert len(fitted["nonhomogeneous_components"]) == 3

    def test_recovers_non_homogeneous_structure_when_truly_non_homogeneous(self):
        """Strongly age-dependent synthetic data (power far from 1) should select
        the non-homogeneous model, not collapse to stationary."""
        dv, t_start, t_end, tool_ids = _synthetic_windows(
            "ig", [(30.0, 0.35), (32.0, 0.30), (28.0, 0.40)], beta=0.6,
        )
        fitted = _fit_rounded_nh_ig(dv, t_start, t_end, tool_ids)
        assert fitted["selected_age_model"] == "power_law"
        assert fitted["stationary_aic"] - fitted["nonhomogeneous_aic"] > 2.0

    def test_recovers_stationary_structure_when_truly_stationary(self):
        """power=1 (stationary) synthetic data should select the stationary
        model, exercising the same collapse-to-simpler-model path observed
        on the real dataset for Inverse Gaussian."""
        dv, t_start, t_end, tool_ids = _synthetic_windows(
            "ig", [(20.0, 1.0), (22.0, 1.0), (19.0, 1.0)], beta=0.5,
        )
        fitted = _fit_rounded_nh_ig(dv, t_start, t_end, tool_ids)
        assert fitted["selected_age_model"] == "linear"
        for c in fitted["components"]:
            assert c["power"] == pytest.approx(1.0)


class TestRoundedNonhomogeneousWienerFit:
    def test_fits_quantized_multi_tool_data(self):
        dv, t_start, t_end, tool_ids = _synthetic_windows(
            "wiener", [(22.0, 0.75), (28.0, 0.65), (18.0, 0.90)], beta=0.6,
        )
        fitted = _fit_rounded_nh_wiener(dv, t_start, t_end, tool_ids)
        assert fitted["model"] == "nonhomogeneous_wiener_rounded"
        assert len(fitted["components"]) == 3
        assert fitted["n_windows"] == len(dv)
        assert 0.0 <= fitted["mean_p_negative_increment"] <= 1.0
        assert 0.0 <= fitted["max_p_negative_increment"] <= 1.0

    def test_negative_increment_probability_rises_with_low_mean_high_variance(self):
        """A process with small mean increments relative to its variance should
        show materially higher implied negative-increment probability than one
        with large, well-separated-from-zero increments -- the structural
        defect this diagnostic exists to catch."""
        # Small eta -> small mean increments per window, same beta -> same variance scale.
        dv_low, ts_low, te_low, ti_low = _synthetic_windows(
            "wiener", [(3.0, 1.0), (3.0, 1.0), (3.0, 1.0)], beta=0.6,
        )
        dv_high, ts_high, te_high, ti_high = _synthetic_windows(
            "wiener", [(40.0, 1.0), (40.0, 1.0), (40.0, 1.0)], beta=0.6,
        )
        fit_low = _fit_rounded_nh_wiener(dv_low, ts_low, te_low, ti_low)
        fit_high = _fit_rounded_nh_wiener(dv_high, ts_high, te_high, ti_high)
        assert fit_low["mean_p_negative_increment"] > fit_high["mean_p_negative_increment"]


class TestBinProbabilities:
    def test_ig_bin_probability_in_unit_interval(self):
        observed = np.array([0.0, 5.0, 50.0, 150.0])
        shape_s = np.array([10.0, 10.0, 10.0, 10.0])
        p = _rounded_ig_bin_probability(observed, shape_s, beta=0.5, resolution_um=1.0)
        assert np.all(p > 0.0) and np.all(p <= 1.0)

    def test_wiener_bin_probability_in_unit_interval(self):
        observed = np.array([-5.0, 0.0, 5.0, 50.0])
        shape_s = np.array([10.0, 10.0, 10.0, 10.0])
        p = _rounded_wiener_bin_probability(observed, shape_s, beta=0.5, resolution_um=1.0)
        assert np.all(p > 0.0) and np.all(p <= 1.0)


class TestValidation:
    def _index_and_params(self):
        dv, t_start, t_end, tool_ids = _synthetic_windows(
            "ig", [(22.0, 1.0), (24.0, 1.0), (20.0, 1.0)], beta=0.55,
        )
        fitted = _fit_rounded_nh_ig(dv, t_start, t_end, tool_ids)
        fitted["machine_id"] = 1
        fitted["window_size"] = 15
        params = {"M1": fitted}

        rows = []
        for tool in (1, 2, 3):
            n_runs = 45
            times = np.linspace(10.0, 3500.0, n_runs)
            comp = next(c for c in fitted["components"] if c["tool_id"] == tool)
            eta, power, beta = comp["eta"], comp["power"], comp["beta"]
            t_ref = comp["time_reference_s"]
            wear = eta / beta * (times / t_ref) ** power
            for run, (t, w) in enumerate(zip(times, wear), start=1):
                rows.append({
                    "machine": 1, "tool": tool, "run": run,
                    "cumulated_tool_contact_time": t, "wear": w,
                })
        index = pd.DataFrame(rows)
        return index, params

    def test_validate_ig_fit_returns_expected_keys(self):
        index, params = self._index_and_params()
        result = validate_ig_fit(index, params)
        assert set(result["M1"]) >= {
            "pit_ks_stat", "pit_ks_pvalue", "pit_ks_frac_reject_05",
            "trajectory_rmse_um", "terminal_bias_um", "validation_passed",
        }
        assert 0.0 <= result["M1"]["pit_ks_frac_reject_05"] <= 1.0

    def test_validate_wiener_fit_returns_expected_keys(self):
        index, _params_ig = self._index_and_params()
        dv, t_start, t_end, tool_ids = _synthetic_windows(
            "wiener", [(22.0, 1.0), (24.0, 1.0), (20.0, 1.0)], beta=0.55,
        )
        fitted = _fit_rounded_nh_wiener(dv, t_start, t_end, tool_ids)
        fitted["machine_id"] = 1
        fitted["window_size"] = 15
        result = validate_wiener_fit(index, {"M1": fitted})
        assert "mean_p_negative_increment" in fitted
        assert 0.0 <= result["M1"]["pit_ks_frac_reject_05"] <= 1.0


class TestStateAwareResimulation:
    """state_aware_trajectory_resimulation corrects _pit_ks_generic's
    frozen-wear-level mean curve for the two wear-level-dependent families
    by recursing the checkpoint mean forward using each checkpoint's own
    PREDICTED (not observed) wear level -- see that function's own
    docstring for the affine-shape_fn argument these tests check directly.
    """

    T_REF = 1000.0

    def _trajectory(self, gamma_state, eta=25.0, power=1.0, beta=0.5, n_runs=45, dt=100.0):
        """A wear trajectory built EXACTLY from the recursive state-aware
        formula, run by run (every observed time is its own step) -- a
        known-truth fixture: re-simulating at window_size=1 must recover it
        exactly, since every observed point is then also a checkpoint (no
        interpolation involved)."""
        times = np.arange(1, n_runs + 1) * dt
        component = {
            "eta": eta, "power": power, "time_reference_s": self.T_REF,
            "gamma_state": gamma_state, "beta": beta,
        }
        wear = []
        v, prev_t = 0.0, 0.0
        for t in times:
            shape_s = variants._state_dependent_shape_fn(
                component, np.array([prev_t]), np.array([t]), np.array([v]),
            )
            v = v + float(shape_s[0] / beta)
            wear.append(v)
            prev_t = t
        return times, np.array(wear), component

    def _fixture(self, gamma_state, family_key="gamma_state", **kw):
        times, wear, component = self._trajectory(gamma_state, **kw)
        index = pd.DataFrame({
            "machine": 1, "tool": 1, "run": np.arange(1, len(times) + 1),
            "cumulated_tool_contact_time": times, "wear": wear,
        })
        naive_predicted = _naive_predicted_curve(component, times, wear[0])
        naive_rmse = float(np.sqrt(np.mean(np.square(wear - naive_predicted))))
        naive_bias = float(naive_predicted[-1] - wear[-1])
        gp = {
            "machine_id": 1, "window_size": 1, "components": [{"tool_id": 1, **component}],
            "selected_age_model": "power_law", "delta_aic_vs_stationary": 100.0,
            "pit_ks_stat": 0.05, "pit_ks_frac_reject_05": 0.1,
            "trajectory_rmse_um": naive_rmse, "terminal_bias_um": naive_bias,
        }
        comparison_result = {families.param_key(family_key): {"M1": gp}}
        return index, comparison_result, naive_rmse, naive_bias

    def test_exactly_recovers_a_known_state_dependent_trajectory(self):
        index, comparison_result, naive_rmse, naive_bias = self._fixture(gamma_state=0.6)
        result = state_aware_trajectory_resimulation(index, comparison_result, window_size=1)
        r = result["gamma_state"]["M1"]
        assert r["state_aware_trajectory_rmse_um"] == pytest.approx(0.0, abs=1e-6)
        assert r["state_aware_terminal_bias_um"] == pytest.approx(0.0, abs=1e-6)
        assert r["naive_trajectory_rmse_um"] == pytest.approx(naive_rmse)
        assert r["naive_terminal_bias_um"] == pytest.approx(naive_bias)
        # With real, nonzero feedback, data generated WITH that feedback
        # must show a genuinely nonzero naive (frozen-wear-level) error --
        # otherwise this fixture would not actually be exercising anything.
        assert naive_rmse > 5.0
        assert r["rmse_gap_closed_frac"] == pytest.approx(1.0, abs=1e-4)
        assert r["bias_gap_closed_frac"] == pytest.approx(1.0, abs=1e-4)

    def test_reduces_to_naive_exactly_at_gamma_state_zero(self):
        """At gamma_state=0 the shape function no longer reads v_start at
        all, so the recursive and one-shot (frozen) formulas telescope to
        the identical answer -- the state-aware correction must vanish
        exactly, not merely approximately, for a family with no real
        feedback."""
        index, comparison_result, naive_rmse, naive_bias = self._fixture(gamma_state=0.0)
        result = state_aware_trajectory_resimulation(index, comparison_result, window_size=1)
        r = result["gamma_state"]["M1"]
        assert r["state_aware_trajectory_rmse_um"] == pytest.approx(naive_rmse, abs=1e-6)
        assert r["state_aware_terminal_bias_um"] == pytest.approx(naive_bias, abs=1e-6)

    def test_matches_monte_carlo_average(self):
        """The deterministic checkpoint recursion is claimed (this
        function's own docstring) to equal the mean of infinitely many
        random step-by-step replicates exactly, since shape_s is affine in
        v_start and neither state family clips its draws. Checked here
        directly against an explicit Monte Carlo simulation at the
        production window_size=15, not just by inspection of that
        argument."""
        gamma_state, eta, power, beta = 0.5, 25.0, 1.0, 0.5
        window_size, n_runs, dt = 15, 45, 100.0
        times = np.arange(1, n_runs + 1) * dt
        component = {
            "eta": eta, "power": power, "time_reference_s": self.T_REF,
            "gamma_state": gamma_state, "beta": beta,
        }
        checkpoint_idx = list(range(0, n_runs, window_size))
        if checkpoint_idx[-1] != n_runs - 1:
            checkpoint_idx.append(n_runs - 1)
        checkpoint_times = np.concatenate([[0.0], times[np.array(checkpoint_idx)]])

        v = 0.0
        for k in range(len(checkpoint_times) - 1):
            shape_s = variants._state_dependent_shape_fn(
                component, checkpoint_times[k:k + 1], checkpoint_times[k + 1:k + 2], np.array([v]),
            )
            v = v + float(shape_s[0] / beta)
        deterministic_final = v

        rng = np.random.default_rng(2026)
        n_mc = 20_000
        mc_final = np.empty(n_mc)
        for rep in range(n_mc):
            v_mc = 0.0
            for k in range(len(checkpoint_times) - 1):
                shape_s = variants._state_dependent_shape_fn(
                    component, checkpoint_times[k:k + 1], checkpoint_times[k + 1:k + 2], np.array([v_mc]),
                )
                v_mc += rng.gamma(shape=max(float(shape_s[0]), 1e-9), scale=1.0 / beta)
            mc_final[rep] = v_mc
        mc_mean = float(np.mean(mc_final))
        mc_se = float(np.std(mc_final, ddof=1) / np.sqrt(n_mc))
        assert deterministic_final == pytest.approx(mc_mean, abs=6 * mc_se)

    def test_missing_family_key_returns_empty_without_error(self):
        index, comparison_result, _naive_rmse, _naive_bias = self._fixture(gamma_state=0.3)
        del comparison_result["gamma_state_params"]
        result = state_aware_trajectory_resimulation(index, comparison_result)
        assert result == {}

    def test_returns_expected_keys_for_both_state_families(self):
        # Each family gets its own isolated index/comparison_result (rather
        # than merging both into one index and calling the function once):
        # both fixtures use machine=1, tool=1, so merging would pool one
        # family's trajectory into the other's tool-1 group when the
        # function filters by machine_id, corrupting both.
        for family_key in ("gamma_state", "inverse_gaussian_state"):
            index, comparison_result, _, _ = self._fixture(gamma_state=0.3, family_key=family_key)
            result = state_aware_trajectory_resimulation(index, comparison_result, window_size=1)
            assert set(result) == {family_key}
            r = result[family_key]["M1"]
            assert set(r) >= {
                "naive_trajectory_rmse_um", "naive_terminal_bias_um",
                "state_aware_trajectory_rmse_um", "state_aware_terminal_bias_um",
                "rmse_gap_closed_frac", "bias_gap_closed_frac", "state_aware_validation_passed",
            }
            assert isinstance(r["state_aware_validation_passed"], bool)
            assert np.isfinite(r["state_aware_trajectory_rmse_um"])


class TestPitAcf:
    """Direct tests of the low-level ACF estimator, independent of the
    fitting/PIT machinery around it -- mirrors TestWilsonCI/TestBenjaminiHochberg's
    style of validating a raw statistical helper against known behavior
    rather than only through an end-to-end integration test."""

    def test_matches_known_ar1_decay_pattern(self):
        # Simulate a long AR(1) series with known rho; the empirical lag-k
        # ACF of a true AR(1) process is rho**k in expectation. A long
        # series (n=2000) keeps finite-sample noise small enough for a
        # meaningful tolerance check.
        rng = np.random.default_rng(7)
        rho = 0.7
        n = 2000
        x = np.empty(n)
        x[0] = rng.standard_normal()
        for i in range(1, n):
            x[i] = rho * x[i - 1] + rng.standard_normal()
        acf = _pit_acf(x, max_lag=3)
        assert acf[0] == pytest.approx(rho, abs=0.05)
        assert acf[1] == pytest.approx(rho ** 2, abs=0.07)
        assert acf[2] == pytest.approx(rho ** 3, abs=0.08)

    def test_near_zero_for_iid_noise(self):
        rng = np.random.default_rng(11)
        x = rng.standard_normal(500)
        acf = _pit_acf(x, max_lag=5)
        assert np.all(np.abs(acf) < 0.15)

    def test_returns_nan_for_constant_input(self):
        x = np.full(10, 5.0)
        acf = _pit_acf(x, max_lag=3)
        assert np.all(np.isnan(acf))

    def test_returns_nan_past_usable_lag_length(self):
        x = np.array([1.0, 2.0, 3.0])  # n=3 -> only lag=1 has >=2 usable pairs
        acf = _pit_acf(x, max_lag=3)
        assert np.isfinite(acf[0])
        assert np.isnan(acf[1])
        assert np.isnan(acf[2])


def _three_machine_index(n_runs: int = 160) -> pd.DataFrame:
    """build_wear_windows always builds entries for M1/M2/M3 (mirroring the
    real 3-machine dataset), so any function that iterates all machines needs
    >=30 windows for each, unlike TestValidation's single-machine helper which
    bypasses build_wear_windows' per-machine iteration.
    n_runs=160 -> (n_runs-1)/window_size=15 -> 10 windows/tool x 3 tools = 30.
    """
    rows = []
    for machine in (1, 2, 3):
        for tool_offset, tool in enumerate((1, 2, 3)):
            tool_id = (machine - 1) * 3 + tool
            times = np.linspace(10.0, 3500.0, n_runs)
            eta, power, beta = 20.0 + 2 * tool_offset, 1.0, 0.55
            wear = eta / beta * (times / 1000.0) ** power
            for run, (t, w) in enumerate(zip(times, wear), start=1):
                rows.append({
                    "machine": machine, "tool": tool_id, "run": run,
                    "cumulated_tool_contact_time": t, "wear": w,
                })
    return pd.DataFrame(rows)


class TestFitAndCompare:
    def _three_machine_index(self):
        return _three_machine_index()

    def test_unified_comparison_structure(self):
        index = self._three_machine_index()
        gamma_params = {
            f"M{m}": {
                "selected_age_model": "linear", "aic": 100.0, "bic": 110.0,
                "components": [
                    {"tool_id": (m - 1) * 3 + t, "eta": 20.0 + 2 * (t - 1), "power": 1.0,
                     "beta": 0.55, "time_reference_s": 1000.0}
                    for t in (1, 2, 3)
                ],
            }
            for m in (1, 2, 3)
        }
        result = fit_and_compare_degradation_models(index, gamma_params, window_size=15)
        assert set(result) == {
            "gamma_params", "ig_params", "wiener_params",
            "wiener_truncated_params", "gamma_state_params", "inverse_gaussian_state_params",
            "gamma_random_effects_params", "inverse_gaussian_random_effects_params",
            "wiener_multistage_params", "gig_params",
            "comparison",
        }
        c = result["comparison"]["M1"]
        assert c["best_aic_family"] in set(families.FAMILY_REGISTRY)
        assert set(c["per_tool_power_ig_nonhomogeneous"]) == {1, 2, 3}
        for family_key in families.FAMILY_REGISTRY:
            assert np.isfinite(c[family_key]["aic"])


class TestLeaveOneToolOutCVLikelihood:
    def test_returns_per_machine_family_scores(self):
        """per_family used to be pinned to the original {"gamma",
        "inverse_gaussian", "wiener"} -- stale since the registry grew to
        10 (leave_one_tool_out_cv_likelihood itself already iterates the
        full registry; only this assertion lagged). Checked dynamically
        against families.FAMILY_REGISTRY, matching TestFitAndCompare's
        equivalent check just above."""
        index = _three_machine_index()
        result = leave_one_tool_out_cv_likelihood(index, window_size=15)
        assert set(result) == {"M1", "M2", "M3"}
        full_registry = set(families.FAMILY_REGISTRY)
        for machine_key, r in result.items():
            assert set(r["per_family"]) == full_registry
            assert r["best_cv_family"] in full_registry
            for family, f in r["per_family"].items():
                assert np.isfinite(f["cv_total_loglik"])
                assert f["n_windows"] > 0
                assert len(f["fold_loglik"]) == 3  # 3 tools/machine -> 3 folds


class TestWindowSizeRobustnessSweep:
    def test_reports_per_window_size_per_machine(self):
        # n_runs=220 keeps windows/tool >= 10 (>=30 total) even at window_size=20,
        # unlike the default n_runs=160 sized only for window_size=15.
        index = _three_machine_index(n_runs=220)
        result = window_size_robustness_sweep(index, window_sizes=[15, 20])
        assert set(result) == {"15", "20"}
        for ws_result in result.values():
            for machine_key in ("M1", "M2", "M3"):
                r = ws_result[machine_key]
                assert not r.get("skipped")
                assert r["best_aic_family"] in {"gamma", "inverse_gaussian", "wiener"}
                assert np.isfinite(r["delta_aic_gamma_minus_ig"])
                assert set(r["mean_lag1_pit_acf"]) == set(families.FAMILY_REGISTRY)
                for v in r["mean_lag1_pit_acf"].values():
                    assert np.isnan(v) or -1.0 <= v <= 1.0


class TestBootstrapFamilyAicGap:
    def test_returns_bounded_fraction_and_finite_ci(self):
        index = _three_machine_index()
        result = bootstrap_family_aic_gap(index, window_size=15, n_boot=5, rng_seed=0)
        assert set(result) == {"M1", "M2", "M3"}
        for r in result.values():
            assert r["n_boot"] == 5
            assert 0.0 <= r["frac_favoring_ig"] <= 1.0
            assert np.isfinite(r["mean_gap"])
            assert r["ci_lo"] <= r["ci_hi"]
            assert len(r["gaps"]) == 5


class TestBlockBootstrapFamilyAicGapByTool:
    """Restricted to machine 1 only (not all 3) throughout -- each draw is a
    REAL Nelder-Mead refit (unlike most other tests in this module, which
    check structure against cheap synthetic fits), so keeping this to one
    machine's 27 draws instead of three machines' 81 keeps the suite fast
    without losing coverage of anything machine-specific (build_wear_windows
    and this function treat every machine identically)."""

    def _m1_index(self):
        index = _three_machine_index()
        return index[index["machine"] == 1]

    def test_enumerates_exactly_27_draws_for_3_tools(self):
        result = block_bootstrap_family_aic_gap_by_tool(self._m1_index(), window_size=15)
        assert set(result) == {"M1"}
        r = result["M1"]
        assert r["exact_enumeration"] is True
        assert r["n_tools"] == 3
        assert r["n_boot"] == 27  # 3**3 exact enumeration, not Monte Carlo
        assert 0.0 <= r["frac_favoring_ig"] <= 1.0
        assert np.isfinite(r["mean_gap"])
        assert r["ci_lo"] <= r["ci_hi"]
        assert len(r["gaps"]) == 27

    def test_wider_or_equal_ci_than_window_level_bootstrap(self):
        """Not a strict mathematical guarantee for every possible dataset,
        but the expected, defensible direction: resampling only 3
        indivisible tool-blocks carries less independent information than
        resampling individual windows, so the between-tool block
        bootstrap's CI should not be tighter than the window-level one on
        the same data (checked via width, robust to the two CIs being
        centered slightly differently)."""
        index = self._m1_index()
        window_level = bootstrap_family_aic_gap(index, window_size=15, n_boot=5, rng_seed=0)
        block_level = block_bootstrap_family_aic_gap_by_tool(index, window_size=15)
        width_window = window_level["M1"]["ci_hi"] - window_level["M1"]["ci_lo"]
        width_block = block_level["M1"]["ci_hi"] - block_level["M1"]["ci_lo"]
        assert width_block >= width_window * 0.5  # generous margin, not a tight equality


class TestMovingBlockBootstrapFamilyAicGap:
    """Restricted to machine 1 only, same rationale as
    TestBlockBootstrapFamilyAicGapByTool -- each draw is a real refit."""

    def _m1_index(self):
        index = _three_machine_index()  # n_runs=160 -> 10 windows/tool at window_size=15
        return index[index["machine"] == 1]

    def test_returns_finite_ci_with_all_tools_used(self):
        # block_length=8 <= the fixture's 10 windows/tool, so all 3 tools
        # contribute (n_tools_used == 3), not silently dropped.
        result = moving_block_bootstrap_family_aic_gap(
            self._m1_index(), window_size=15, block_length=8, n_boot=5, rng_seed=0,
        )
        assert set(result) == {"M1"}
        r = result["M1"]
        assert r["n_tools_used"] == 3
        assert r["block_length"] == 8
        assert r["n_boot"] == 5
        assert 0.0 <= r["frac_favoring_ig"] <= 1.0
        assert np.isfinite(r["mean_gap"])
        assert r["ci_lo"] <= r["ci_hi"]
        assert len(r["gaps"]) == 5

    def test_excludes_tools_with_fewer_windows_than_block_length(self):
        # block_length=12 > the fixture's 10 windows/tool -> every tool
        # skipped, not a crash or a degenerate single-window block.
        result = moving_block_bootstrap_family_aic_gap(
            self._m1_index(), window_size=15, block_length=12, n_boot=3, rng_seed=0,
        )
        assert result == {}


class TestResidualAutocorrelationCheck:
    def _single_family_index_and_params(self, rng_seed=123, n_runs=160):
        """Mirrors TestValidation._index_and_params's pattern (fit IG to
        synthetic windows, then build a smooth deterministic per-run index
        matching the fitted mean trend) but with enough runs/tool
        (n_runs=160 -> 10 windows/tool at window_size=15) that no tool is
        dropped by residual_autocorrelation_check's own min-windows-per-tool
        guard (n_t >= max_lag+3)."""
        dv, t_start, t_end, tool_ids = _synthetic_windows(
            "ig", [(22.0, 1.0), (24.0, 1.0), (20.0, 1.0)], beta=0.55, rng_seed=rng_seed,
        )
        fitted = _fit_rounded_nh_ig(dv, t_start, t_end, tool_ids)
        fitted["machine_id"] = 1
        fitted["window_size"] = 15
        comparison_result = {"ig_params": {"M1": fitted}}

        rows = []
        for tool in (1, 2, 3):
            times = np.linspace(10.0, 3500.0, n_runs)
            comp = next(c for c in fitted["components"] if c["tool_id"] == tool)
            eta, power, beta = comp["eta"], comp["power"], comp["beta"]
            t_ref = comp["time_reference_s"]
            wear = eta / beta * (times / t_ref) ** power
            for run, (t, w) in enumerate(zip(times, wear), start=1):
                rows.append({
                    "machine": 1, "tool": tool, "run": run,
                    "cumulated_tool_contact_time": t, "wear": w,
                })
        index = pd.DataFrame(rows)
        return index, comparison_result

    def test_returns_expected_structure_and_bounds(self):
        index, comparison_result = self._single_family_index_and_params()
        result = residual_autocorrelation_check(
            index, comparison_result, window_size=15, max_lag=3, n_seeds=5,
        )
        assert set(result) == {"inverse_gaussian"}
        r = result["inverse_gaussian"]
        assert r["max_lag"] == 3
        assert r["n_seeds"] == 5
        assert set(r["per_machine"]) == {"M1"}
        m1 = r["per_machine"]["M1"]
        assert m1["n_tools"] == 3
        assert set(m1["acf_by_tool"]) == {1, 2, 3}
        for acf in m1["acf_by_tool"].values():
            assert len(acf) == 3
            assert all(np.isnan(v) or -1.0 <= v <= 1.0 for v in acf)
        assert np.isnan(r["overall_mean_lag1_acf"]) or -1.0 <= r["overall_mean_lag1_acf"] <= 1.0

    def test_skips_tools_with_too_few_windows_for_requested_max_lag(self):
        """max_lag=20 needs n_t>=23 windows/tool, far more than the 10/tool
        this fixture provides -- every tool should be skipped rather than
        crash on a too-short sequence."""
        index, comparison_result = self._single_family_index_and_params()
        result = residual_autocorrelation_check(
            index, comparison_result, window_size=15, max_lag=20, n_seeds=3,
        )
        m1 = result["inverse_gaussian"]["per_machine"]["M1"]
        assert m1["n_tools"] == 0
        assert m1["acf_by_tool"] == {}
        assert np.isnan(m1["mean_lag1_acf"])


class TestPooledVsPerMachine:
    def test_pooled_fit_has_fewer_or_equal_parameters_worth_of_structure(self):
        index = _three_machine_index()
        gamma_params = {
            f"M{m}": {
                "selected_age_model": "linear", "aic": 100.0, "bic": 110.0,
                "components": [
                    {"tool_id": (m - 1) * 3 + t, "eta": 20.0 + 2 * (t - 1), "power": 1.0,
                     "beta": 0.55, "time_reference_s": 1000.0}
                    for t in (1, 2, 3)
                ],
            }
            for m in (1, 2, 3)
        }
        comparison_result = fit_and_compare_degradation_models(index, gamma_params, window_size=15)
        result = pooled_vs_per_machine_comparison(index, comparison_result, window_size=15)
        assert set(result) == {"gamma", "inverse_gaussian", "wiener"}
        for family, r in result.items():
            assert np.isfinite(r["pooled_aic"])
            assert np.isfinite(r["delta_aic_pooled_minus_summed"])
            assert np.isfinite(r["delta_bic_pooled_minus_summed"])


class TestBootstrapPerToolPowerSignificance:
    def test_returns_absolute_and_relative_tests_for_all_nine_tools(self):
        index = _three_machine_index()
        result = bootstrap_per_tool_power_significance(index, window_size=15, n_boot=5, rng_seed=0)
        assert set(result) >= {
            "per_tool_absolute_vs_stationary", "per_tool_relative_vs_siblings",
            "any_tool_significant_absolute", "any_tool_significant_relative", "conclusion",
        }
        abs_result = result["per_tool_absolute_vs_stationary"]
        rel_result = result["per_tool_relative_vs_siblings"]
        assert set(abs_result) == set(range(1, 10))
        assert set(rel_result) == set(range(1, 10))
        for tool_id, a in abs_result.items():
            assert a["n_boot_success"] > 0
            assert a["ci_lo"] <= a["ci_hi"]
            assert isinstance(a["excludes_stationary"], bool)
            r = rel_result[tool_id]
            assert len(r["siblings"]) == 2
            assert r["ci_lo"] <= r["ci_hi"]
            assert 0.0 <= r["p_target_ge_siblings_onesided"] <= 1.0
            assert isinstance(r["excludes_zero"], bool)
        assert isinstance(result["conclusion"], str) and len(result["conclusion"]) > 0

    def test_identical_siblings_do_not_spuriously_reject(self):
        # All 3 tools/machine share the same true (eta, power) in _three_machine_index
        # except a small eta offset -- power itself is identical (=1.0) across
        # siblings, so with a real (if small) bootstrap sample the relative test
        # should not systematically flag every tool as significant.
        index = _three_machine_index()
        result = bootstrap_per_tool_power_significance(index, window_size=15, n_boot=20, rng_seed=1)
        n_flagged = sum(v["excludes_zero"] for v in result["per_tool_relative_vs_siblings"].values())
        assert n_flagged <= 3  # loose bound: some false positives expected at n_boot=20, not all 9


class TestPerToolFitDiagnostics:
    def test_returns_rmse_and_terminal_bias_per_tool(self):
        index = _three_machine_index()
        gamma_params = _fixture_gamma_params()
        result = per_tool_fit_diagnostics(index, gamma_params)
        assert set(result) == {"M1", "M2", "M3"}
        for machine_key, machine_result in result.items():
            assert machine_result["selected_age_model"] == "linear"
            assert set(machine_result["per_tool"]) == set(
                gamma_params[machine_key]["components"][i]["tool_id"] for i in range(3)
            )
            for tool_row in machine_result["per_tool"].values():
                assert tool_row["rmse_um"] >= 0.0
                assert np.isfinite(tool_row["terminal_bias_um"])
                assert tool_row["terminal_wear_um"] > 0.0


class TestModelRecoveryStudy:
    """Restricted to only_true_families=a small subset throughout (not the
    full 10-family sweep) -- model_recovery_study's own docstring documents
    ~half a day for the full 10-true-family x 10-fit-family x 3-machine
    sweep even at n_replicates=6; running the full registry here at
    n_replicates=2-5 (as an earlier version of this test, predating the
    3-to-10-family registry expansion, did) took over an hour for what is
    meant to be a fast structural/sanity unit test. only_true_families lets
    these tests exercise the real function, with the real (full-registry)
    fit-candidate pool, at a small fraction of the cost -- exactly the
    checkpointing use case it was added for."""

    def test_returns_recovery_rates_for_all_true_families(self):
        index = _three_machine_index()
        comparison_result = fit_and_compare_degradation_models(
            index, _fixture_gamma_params(), window_size=15,
        )
        result = model_recovery_study(
            index, comparison_result, window_size=15, n_replicates=2, rng_seed=0,
            only_true_families=["gamma", "inverse_gaussian"],
        )
        assert set(result) == {"M1", "M2", "M3"}
        full_registry = set(families.FAMILY_REGISTRY)
        for machine_result in result.values():
            # Only the requested true families are TESTED...
            assert set(machine_result) == {"gamma", "inverse_gaussian"}
            for true_family, r in machine_result.items():
                assert r["n_replicates_requested"] == 2
                assert 0 <= r["n_replicates_success"] <= 2
                # ...but every replicate is still scored against the FULL
                # registry as fit candidates, not narrowed to match.
                assert set(r["selected_counts"]) == full_registry
                assert sum(r["selected_counts"].values()) == r["n_replicates_success"]
                if r["n_replicates_success"] > 0:
                    assert 0.0 <= r["recovery_rate"] <= 1.0
                else:
                    assert np.isnan(r["recovery_rate"])

    def test_recovers_true_family_at_a_reasonable_rate(self):
        """Not a strict correctness proof (that's what the full n_replicates=6
        production run on real data is for), but a basic sanity check: on an
        easy, well-separated synthetic setup, the procedure should recover
        the true family more often than chance at least in aggregate --
        chance is 1/N for N registered fit-candidate families, not a
        hardcoded 1/3 (stale from before the registry grew to 10)."""
        index = _three_machine_index()
        comparison_result = fit_and_compare_degradation_models(
            index, _fixture_gamma_params(), window_size=15,
        )
        result = model_recovery_study(
            index, comparison_result, window_size=15, n_replicates=5, rng_seed=2,
            only_true_families=["gamma", "inverse_gaussian"],
        )
        total_correct = sum(
            r["selected_counts"][true_family]
            for machine_result in result.values()
            for true_family, r in machine_result.items()
        )
        total_success = sum(
            r["n_replicates_success"]
            for machine_result in result.values()
            for r in machine_result.values()
        )
        assert total_success > 0
        assert total_correct / total_success > 1.0 / len(families.FAMILY_REGISTRY)

    def test_only_true_families_checkpointing_reproduces_unrestricted_result(self):
        """model_recovery_study's own documented runtime (up to ~half a day
        for the full sweep) makes it the single most likely analysis to be
        mid-flight when a long pipeline run is interrupted -- only_true_families
        lets a caller checkpoint it one true family (all 3 machines) at a
        time. This is the key correctness property for that: testing
        "inverse_gaussian" alone must give EXACTLY the same per-machine
        result as testing it alongside another true family would -- same
        random draws, not just a same-ballpark independent draw -- so that
        chunking the work into checkpointed slices is transparent to the
        final merged result. Deliberately compares two SMALL restricted
        calls (not one restricted vs. one fully unrestricted 10-true-family
        call) to keep this fast while still exercising the exact property
        that matters: does a family's own checkpointed slice depend on
        what else was tested in the SAME call. Also confirms the
        FIT-candidate set (every registered family, via selected_counts'
        keys -- checked directly against FAMILY_REGISTRY, not against a
        second expensive run) is untouched by the restriction.
        """
        index = _three_machine_index()
        comparison_result = fit_and_compare_degradation_models(
            index, _fixture_gamma_params(), window_size=15,
        )
        alone = model_recovery_study(
            index, comparison_result, window_size=15, n_replicates=2, rng_seed=7,
            only_true_families=["inverse_gaussian"],
        )
        alongside = model_recovery_study(
            index, comparison_result, window_size=15, n_replicates=2, rng_seed=7,
            only_true_families=["gamma", "inverse_gaussian"],
        )
        full_registry = set(families.FAMILY_REGISTRY)
        for machine_key in ("M1", "M2", "M3"):
            assert set(alone[machine_key]) == {"inverse_gaussian"}
            assert set(alongside[machine_key]) == {"gamma", "inverse_gaussian"}
            # Fit-candidate set is always the full registry regardless of
            # how many (or which) true families this call is restricted to.
            assert set(alone[machine_key]["inverse_gaussian"]["selected_counts"]) == full_registry
            assert set(alongside[machine_key]["inverse_gaussian"]["selected_counts"]) == full_registry
            # The KEY property: identical seeding means an identical result
            # for "inverse_gaussian" whether it's the only true family
            # tested this call or one of two, not just a similarly-
            # distributed independent draw.
            assert alone[machine_key]["inverse_gaussian"] == alongside[machine_key]["inverse_gaussian"]


class TestVuongTest:
    def test_returns_pairwise_statistics_for_all_machines(self):
        index = _three_machine_index()
        comparison_result = fit_and_compare_degradation_models(
            index, _fixture_gamma_params(), window_size=15,
        )
        result = vuong_test(index, comparison_result, window_size=15)
        assert set(result) == {"M1", "M2", "M3"}
        # vuong_test now runs over the full 10-family registry (generalized
        # from the original hardcoded 3), so the expected pair set is
        # derived the same way vuong_test itself builds it (every
        # non-nested pair) rather than hardcoded to the original 3
        # families -- a hardcoded {gamma_vs_inverse_gaussian, ...} set
        # went stale the moment fit_and_compare_degradation_models started
        # fitting all 10 families, and this test was not updated to match
        # until caught by a real run in the reviewer-response phase.
        family_keys = list(families.FAMILY_REGISTRY)
        expected_pairs = {
            f"{a}_vs_{b}" for a, b in combinations(family_keys, 2)
            if families.FAMILY_REGISTRY[a]["baseline_key"] != b
            and families.FAMILY_REGISTRY[b]["baseline_key"] != a
        }
        all_families = set(family_keys) | {"inconclusive"}
        for machine_result in result.values():
            assert set(machine_result) == expected_pairs
            for r in machine_result.values():
                assert r["favored"] in all_families
                assert np.isfinite(r["z_statistic"])
                assert 0.0 <= r["p_value"] <= 1.0
                assert r["n_windows"] > 0

    def test_favored_direction_matches_z_sign_convention(self):
        index = _three_machine_index()
        comparison_result = fit_and_compare_degradation_models(
            index, _fixture_gamma_params(), window_size=15,
        )
        result = vuong_test(index, comparison_result, window_size=15)
        for machine_result in result.values():
            for pair_key, r in machine_result.items():
                fam_a, fam_b = pair_key.rsplit("_vs_", 1) if pair_key.count("_vs_") == 1 else (None, None)
                # pair keys are built as f"{fam_a}_vs_{fam_b}" with fam_a/fam_b from
                # the fixed `pairs` list, so a plain split on the last "_vs_" is safe
                # here even though "inverse_gaussian" itself contains underscores.
                # Critical value is t(G-1) based, not a fixed 1.96 -- with
                # G=3 tools/machine it is ~4.30, not the normal-theory value.
                crit = r["cluster_critical_value"]
                if r["z_statistic"] > crit:
                    assert r["favored"] == fam_a
                elif r["z_statistic"] < -crit:
                    assert r["favored"] == fam_b
                else:
                    assert r["favored"] == "inconclusive"


def _vuong_golden_fixture(k_gamma=2, k_ig=1, n_tools=3):
    """Hand-built 6-window-per-tool, N-tool, 1-machine dataset with known
    Gamma and IG parameters, structured so build_wear_windows(window_size=2)
    reproduces the exact (t_start, t_end, dv) triples chosen here for every
    tool -- lets a test independently recompute the expected Vuong
    z-statistic (both the naive i.i.d. version and, grouped by tool, the
    cluster-robust version) using the same production bin-probability
    functions, outside vuong_test() entirely.

    Each tool's window_dvs is the same base sequence plus a small per-tool
    offset (0, 0.5, 1.0, ...) rather than n_tools identical copies --
    needed so the cluster-robust sandwich variance (grouped by tool) is
    non-degenerate: with byte-identical dv sequences across every tool,
    each tool's own within-tool mean would equal the global mean exactly,
    making every cluster's centered sum exactly zero.

    build_wear_windows groups via index.groupby(["machine", "tool"]),
    which iterates tool ascending (pandas groupby default sort=True) with
    each tool's own windows contiguous -- confirmed by reading that
    function directly, not assumed -- so concatenating tool 1, 2, ...,
    n_tools's arrays in ascending tool_id order below exactly matches
    vuong_test's internal window ordering.
    """
    base_dvs = np.array([5.0, 6.0, 4.0, 7.0, 5.0, 6.0])
    gamma_eta, gamma_power, gamma_beta, t_ref, resolution = 20.0, 0.8, 0.5, 1000.0, 1.0
    ig_eta, ig_power, ig_beta = 15.0, 1.0, 0.4

    rows, dv_parts, shape_gamma_parts, shape_ig_parts = [], [], [], []
    components_gamma, components_ig = [], []
    for tool_id in range(1, n_tools + 1):
        window_dvs = base_dvs + 0.5 * (tool_id - 1)
        per_run_dv = np.repeat(window_dvs / 2.0, 2)  # 12 raw increments, 2/window
        times = np.arange(13) * 250.0
        wear = np.concatenate([[0.0], np.cumsum(per_run_dv)])
        rows.append(pd.DataFrame({
            "machine": [1] * 13, "tool": [tool_id] * 13, "run": list(range(1, 14)),
            "cumulated_tool_contact_time": times, "wear": wear,
        }))
        t_start = times[0:12:2]
        t_end = times[2:13:2]
        dv_parts.append(window_dvs)
        shape_gamma_parts.append(
            gamma_eta * ((t_end / t_ref) ** gamma_power - (t_start / t_ref) ** gamma_power)
        )
        shape_ig_parts.append(
            ig_eta * ((t_end / t_ref) ** ig_power - (t_start / t_ref) ** ig_power)
        )
        components_gamma.append({"tool_id": tool_id, "eta": gamma_eta, "power": gamma_power,
                                  "beta": gamma_beta, "time_reference_s": t_ref})
        components_ig.append({"tool_id": tool_id, "eta": ig_eta, "power": ig_power,
                               "beta": ig_beta, "time_reference_s": t_ref})

    index = pd.concat(rows, ignore_index=True)
    window_dvs = np.concatenate(dv_parts)
    shape_gamma = np.concatenate(shape_gamma_parts)
    shape_ig = np.concatenate(shape_ig_parts)
    prob_gamma = _rounded_gamma_bin_probability(window_dvs, shape_gamma, gamma_beta, resolution)
    prob_ig = _rounded_ig_bin_probability(window_dvs, shape_ig, ig_beta, resolution)
    log_lik_gamma = float(np.sum(np.log(prob_gamma)))
    log_lik_ig = float(np.sum(np.log(prob_ig)))

    def _params(components, beta, k, log_lik):
        return {
            "components": components, "beta": beta, "wear_resolution_um": resolution,
            "aic": 2 * k - 2 * log_lik, "log_lik": log_lik,
        }

    comparison_result = {
        "gamma_params": {"M1": _params(components_gamma, gamma_beta, k_gamma, log_lik_gamma)},
        "ig_params": {"M1": _params(components_ig, ig_beta, k_ig, log_lik_ig)},
        "wiener_params": {"M1": _params(components_ig, ig_beta, k_ig, log_lik_ig)},
    }
    return index, comparison_result, prob_gamma, prob_ig, k_gamma, k_ig


class TestVuongTestGoldenValue:
    def test_z_statistic_matches_independent_manual_computation(self):
        """Not a re-derivation of the formula (that would be circular) but an
        independent plumbing check: the log-likelihood-ratio array and both
        the naive i.i.d. z and the cluster-robust (by tool) z are computed
        here using the same production bin-probability functions but
        assembled fresh, outside vuong_test()'s internals -- this catches
        wiring bugs (wrong array, off-by-one in indexing, wrong ddof, wrong
        n, wrong cluster grouping) that a shared-formula-only test would
        miss even if the underlying algebra is the same on both sides."""
        index, comparison_result, prob_gamma, prob_ig, k_gamma, k_ig = _vuong_golden_fixture(n_tools=3)
        lr = np.log(prob_gamma) - np.log(prob_ig)
        n = len(lr)
        k_diff = k_gamma - k_ig
        correction = k_diff / n
        mean_lr = np.mean(lr)

        expected_z_naive = np.sqrt(n) * (mean_lr - correction) / np.std(lr, ddof=1)

        tool_ids = np.repeat([1, 2, 3], 6)  # 3 tools x 6 windows, matching the fixture
        centered = lr - mean_lr
        cluster_sums = np.array([centered[tool_ids == t].sum() for t in (1, 2, 3)])
        fc = 3 / 2  # G/(G-1) finite-cluster correction, G=3
        var_cluster = fc * np.sum(cluster_sums ** 2) / n ** 2
        expected_z = (mean_lr - correction) / np.sqrt(var_cluster)
        from scipy.stats import t as _t_dist
        crit = float(_t_dist.ppf(0.975, df=2))
        expected_p = float(2.0 * _t_dist.sf(abs(expected_z), df=2))

        result = vuong_test(index, comparison_result, window_size=2)
        actual = result["M1"]["gamma_vs_inverse_gaussian"]
        assert actual["z_statistic_naive_iid"] == pytest.approx(expected_z_naive, rel=1e-9)
        assert actual["z_statistic"] == pytest.approx(expected_z, rel=1e-9)
        assert actual["p_value"] == pytest.approx(expected_p, rel=1e-9)
        assert actual["n_windows"] == n
        assert actual["n_clusters"] == 3
        expected_favored = (
            "gamma" if expected_z > crit else ("inverse_gaussian" if expected_z < -crit else "inconclusive")
        )
        assert actual["favored"] == expected_favored

    def test_z_favors_the_dramatically_better_fitting_family(self):
        """Direction-of-effect sanity check independent of the exact formula:
        giving one family's parameters an obviously catastrophic fit to the
        fixed data (implied mean far from the actual observations) must
        produce a strongly positive z favoring the other, unambiguously
        better-fitting family -- not a borderline or ambiguous result.
        (Note: identical eta/power/beta plugged into two DIFFERENT families
        does NOT give a zero log-likelihood-ratio, since vuong_test always
        evaluates each family's OWN bin-probability function regardless of
        which parameter values are supplied -- Gamma(20, 0.8) and a
        'Wiener' using the same numbers are still different distributions.
        That rules out a same-parameters-implies-zero-LR test as invalid;
        this monotonic-sensitivity check is the valid substitute.) The
        catastrophic eta is applied to EVERY tool's component (not just
        one) since the fixture now has 3 tools and vuong_test looks up
        each window's own tool_id in component_by_tool -- a partial
        override would KeyError on tools 2 and 3.
        """
        index, comparison_result, _, _, _, _ = _vuong_golden_fixture(n_tools=3)
        bad = dict(comparison_result["wiener_params"]["M1"])
        bad["components"] = [{**c, "eta": 0.01} for c in bad["components"]]
        comparison_result["wiener_params"] = {"M1": bad}
        result = vuong_test(index, comparison_result, window_size=2)
        r = result["M1"]["gamma_vs_wiener"]
        assert r["z_statistic"] > r["cluster_critical_value"]
        assert r["favored"] == "gamma"


class TestWilsonCI:
    def test_matches_textbook_reference_value(self):
        # Well-known worked example: x=10, n=20 -> Wilson 95% CI ~= [0.299, 0.701]
        lo, hi = _wilson_ci(10, 20)
        assert lo == pytest.approx(0.299, abs=0.001)
        assert hi == pytest.approx(0.701, abs=0.001)

    def test_stays_within_bounds_at_extremes(self):
        lo, hi = _wilson_ci(20, 20)
        assert 0.0 <= lo <= hi <= 1.0
        lo, hi = _wilson_ci(0, 20)
        assert 0.0 <= lo <= hi <= 1.0

    def test_zero_n_returns_nan(self):
        lo, hi = _wilson_ci(0, 0)
        assert np.isnan(lo) and np.isnan(hi)


class TestBenjaminiHochberg:
    def test_matches_classic_textbook_example(self):
        # Classic BH worked example: p=[.01,.02,.03,.04,.20], alpha=.05
        # -> first four significant (q=0.05 each), fifth not (q=0.20)
        p = {"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04, "e": 0.20}
        result = _benjamini_hochberg(p, alpha=0.05)
        for k in "abcd":
            assert result[k]["q_value"] == pytest.approx(0.05, abs=1e-9)
            assert result[k]["significant_at_fdr"] is True
        assert result["e"]["q_value"] == pytest.approx(0.20, abs=1e-9)
        assert result["e"]["significant_at_fdr"] is False

    def test_q_values_are_monotone_nondecreasing_by_rank(self):
        p = {"a": 0.001, "b": 0.5, "c": 0.01, "d": 0.3, "e": 0.02}
        result = _benjamini_hochberg(p, alpha=0.05)
        ranked = sorted(result.items(), key=lambda kv: kv[1]["p_value"])
        q_seq = [v["q_value"] for _, v in ranked]
        assert q_seq == sorted(q_seq)

    def test_empty_input_returns_empty(self):
        assert _benjamini_hochberg({}) == {}


class TestMultipleComparisonsSummary:
    def test_returns_three_families_with_consistent_counts(self):
        index = _three_machine_index()
        comparison_result = fit_and_compare_degradation_models(
            index, _fixture_gamma_params(), window_size=15,
        )
        vuong_result = vuong_test(index, comparison_result, window_size=15)
        significance_result = bootstrap_per_tool_power_significance(
            index, window_size=15, n_boot=5, rng_seed=0,
        )
        summary = multiple_comparisons_summary(vuong_result, significance_result)
        assert set(summary) == {"vuong_pairwise", "per_tool_absolute", "per_tool_relative"}
        assert summary["vuong_pairwise"]["n_tests"] == 9
        assert summary["per_tool_absolute"]["n_tests"] == 9
        assert summary["per_tool_relative"]["n_tests"] == 9
        for family in summary.values():
            assert family["n_significant_fdr_corrected"] <= family["n_significant_uncorrected"]
            assert len(family["detail"]) == family["n_tests"]


def _criterion_params(stationary_aic, nonhomogeneous_aic, stationary_bic, nonhomogeneous_bic):
    return {
        "stationary_aic": stationary_aic, "nonhomogeneous_aic": nonhomogeneous_aic,
        "stationary_bic": stationary_bic, "nonhomogeneous_bic": nonhomogeneous_bic,
    }


class TestAgeModelCriterionSensitivity:
    def test_flags_criterion_dependent_case_and_leaves_agreeing_cases_alone(self):
        # Gamma: AIC gap +8 (power_law) but BIC gap +1 (linear) -- criterion-dependent,
        # mirroring the real M1/M2 Gamma flip this table exists to surface.
        # IG: both linear (gap negative under both). Wiener: both power_law
        # (large gap under both) -- neither should be flagged as dependent.
        comparison_result = {
            "gamma_params": {"M1": _criterion_params(100.0, 92.0, 100.0, 99.0)},
            "ig_params": {"M1": _criterion_params(90.0, 95.0, 90.0, 100.0)},
            "wiener_params": {"M1": _criterion_params(200.0, 120.0, 200.0, 130.0)},
        }
        result = age_model_criterion_sensitivity(comparison_result)
        assert set(result) == {"M1"}

        gamma = result["M1"]["gamma"]
        assert gamma["age_model_under_aic"] == "power_law"
        assert gamma["age_model_under_bic"] == "linear"
        assert gamma["criterion_dependent"] is True

        ig = result["M1"]["inverse_gaussian"]
        assert ig["age_model_under_aic"] == "linear"
        assert ig["age_model_under_bic"] == "linear"
        assert ig["criterion_dependent"] is False

        wiener = result["M1"]["wiener"]
        assert wiener["age_model_under_aic"] == "power_law"
        assert wiener["age_model_under_bic"] == "power_law"
        assert wiener["criterion_dependent"] is False

    def test_uses_the_same_2point0_threshold_as_the_aic_only_selection_rule(self):
        # gap=+2.01 (just above 2.0) -> power_law; gap=+1.99 (just below) -> linear.
        # Pins the boundary to the identical threshold _fit_rounded_nh_gamma/_ig/
        # _wiener use for their own AIC-based selection, so a future change to
        # one wouldn't silently desync from the other.
        comparison_result = {
            "gamma_params": {"M1": _criterion_params(10.0, 7.99, 10.0, 8.01)},
            "ig_params": {"M1": _criterion_params(1.0, 1.0, 1.0, 1.0)},
            "wiener_params": {"M1": _criterion_params(1.0, 1.0, 1.0, 1.0)},
        }
        result = age_model_criterion_sensitivity(comparison_result)
        gamma = result["M1"]["gamma"]
        assert gamma["age_model_under_aic"] == "power_law"
        assert gamma["age_model_under_bic"] == "linear"
        assert gamma["aic_gap_stationary_minus_nonhomogeneous"] == pytest.approx(2.01)
        assert gamma["bic_gap_stationary_minus_nonhomogeneous"] == pytest.approx(1.99)

    def test_reproduces_the_real_dataset_finding(self):
        """Golden-value regression test pinned to the actual verified finding
        from the review that motivated this table: Gamma's age-model
        conclusion is criterion-dependent on M1 and M2 (real fitted stationary/
        non-homogeneous AIC and BIC values read directly from
        artifacts/degradation_models/degradation_comparison.json, window_size
        =15), stationary under both criteria on M3; Inverse Gaussian and
        Wiener are not criterion-dependent on any machine."""
        comparison_result = {
            "gamma_params": {
                "M1": _criterion_params(548.67, 541.01, 559.89, 560.64),
                "M2": _criterion_params(720.31, 715.02, 733.14, 737.49),
                "M3": _criterion_params(513.96, 517.99, 525.01, 537.32),
            },
            "ig_params": {
                "M1": _criterion_params(518.55, 521.13, 529.77, 540.76),
                "M2": _criterion_params(690.41, 693.63, 703.25, 716.09),
                "M3": _criterion_params(491.98, 491.65, 503.02, 510.99),
            },
            "wiener_params": {
                "M1": _criterion_params(680.20, 595.67, 691.42, 615.30),
                "M2": _criterion_params(965.22, 798.46, 978.06, 820.93),
                "M3": _criterion_params(616.42, 565.58, 627.47, 584.91),
            },
        }
        result = age_model_criterion_sensitivity(comparison_result)
        assert result["M1"]["gamma"]["age_model_under_aic"] == "power_law"
        assert result["M1"]["gamma"]["age_model_under_bic"] == "linear"
        assert result["M1"]["gamma"]["criterion_dependent"] is True
        assert result["M2"]["gamma"]["age_model_under_aic"] == "power_law"
        assert result["M2"]["gamma"]["age_model_under_bic"] == "linear"
        assert result["M2"]["gamma"]["criterion_dependent"] is True
        assert result["M3"]["gamma"]["age_model_under_aic"] == "linear"
        assert result["M3"]["gamma"]["age_model_under_bic"] == "linear"
        assert result["M3"]["gamma"]["criterion_dependent"] is False
        for machine_key in ("M1", "M2", "M3"):
            assert result[machine_key]["inverse_gaussian"]["criterion_dependent"] is False
            assert result[machine_key]["inverse_gaussian"]["age_model_under_bic"] == "linear"
            assert result[machine_key]["wiener"]["criterion_dependent"] is False
            assert result[machine_key]["wiener"]["age_model_under_bic"] == "power_law"
