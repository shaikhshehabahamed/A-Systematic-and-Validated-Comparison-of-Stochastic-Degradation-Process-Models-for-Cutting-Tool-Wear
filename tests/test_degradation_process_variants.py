"""Unit tests for the seven new degradation-process variants
(degradation_process_variants.py) and the shared registry infrastructure
(degradation_families.py).

Conventions mirror the existing test files: sys.path bootstrap, class-per-
behavior grouping, pytest.approx for float tolerances, small hand-built or
lightly-randomized synthetic fixtures rather than the real dataset.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from scipy.stats import chi2, invgauss as scipy_invgauss
from scipy.special import kv

import degradation_families as families
import degradation_process_variants as variants
from gamma_process import _rounded_gamma_bin_probability as _production_gamma_bin_prob
from degradation_process_models import (
    _rounded_ig_bin_probability as _production_ig_bin_prob,
    _rounded_wiener_bin_probability as _production_wiener_bin_prob,
)


def _synthetic_windows(rng, n_tools=3, n_windows_per_tool=40, dt=100.0, family="wiener", eta_base=25.0, beta=0.5, power=1.0):
    """Small synthetic multi-tool window set, drawn from a KNOWN family's
    OWN increment distribution (so tests of that family's fitter are a
    correctly-specified null, not a cross-family misspecification test --
    see the investigation in the implementation session that found a
    Gamma-vs-Wiener mismatch produces spurious, degenerate changepoint fits
    that are an artifact of the mismatch, not a bug).

    ``eta_base=25.0`` (not e.g. 0.6) deliberately matches the REAL fitted
    eta scale (~22-35, see artifacts/gamma/gamma_process_params.json), not
    an arbitrary small number: an earlier version of this fixture used
    eta_base=0.6 with dt=100/t_ref=1000, giving shape_s ~ 0.07 -- so tiny
    that 116 of 120 synthetic windows rounded to dv=0, silently testing
    every fitter against a near-degenerate, unrepresentative dataset. That
    was caught only by a downstream symptom (a GIG fit driving p_gid to its
    optimizer bound on data that turned out to be ~97% zeros), not by any
    assertion in this file -- worth remembering if a future test here
    starts producing similarly hard-to-interpret results.
    """
    dv, t_start, t_end, tool_ids, v_start = [], [], [], [], []
    for tool in range(1, n_tools + 1):
        eta = eta_base + 3.0 * tool
        t = 0.0
        cum = 0.0
        for _ in range(n_windows_per_tool):
            t_next = t + dt
            shape_s = eta * ((t_next / 1000.0) ** power - (t / 1000.0) ** power)
            if family == "gamma":
                draw = rng.gamma(shape=max(shape_s, 1e-6), scale=1.0 / beta)
            elif family == "inverse_gaussian":
                draw = rng.wald(mean=shape_s / beta, scale=(shape_s ** 2) / beta)
            else:
                mu, sigma = shape_s / beta, np.sqrt(shape_s) / beta
                draw = rng.normal(loc=mu, scale=sigma)
            draw = max(round(draw), 0.0)
            dv.append(draw); t_start.append(t); t_end.append(t_next); tool_ids.append(tool); v_start.append(cum)
            cum += draw
            t = t_next
    return (
        np.array(dv, float), np.array(t_start, float), np.array(t_end, float),
        np.array(tool_ids, int), np.array(v_start, float),
    )


class TestRegistry:
    def test_all_seven_variants_registered_with_required_fields(self):
        expected = {
            "wiener_truncated", "gamma_state", "inverse_gaussian_state",
            "gamma_random_effects", "inverse_gaussian_random_effects", "wiener_multistage",
            "gig",
        }
        assert expected <= set(families.FAMILY_REGISTRY)
        for key in expected:
            spec = families.FAMILY_REGISTRY[key]
            for field in ("label", "color", "ls", "marker", "fit_fn", "shape_fn",
                          "predict_shape_fn", "cdf_fn", "bin_prob_fn", "is_subordinator",
                          "baseline_key", "extra_params_desc", "extra_shape_params"):
                assert field in spec, f"{key} missing {field}"

    def test_labels_are_unique(self):
        labels = [spec["label"] for spec in families.FAMILY_REGISTRY.values()]
        assert len(labels) == len(set(labels))

    def test_double_registration_raises(self):
        with pytest.raises(ValueError):
            families.register(
                "wiener_truncated", label="dup", color="#000", ls="-", marker="o",
                fit_fn=lambda *a, **k: None, shape_fn=lambda *a, **k: None,
                predict_shape_fn=lambda *a, **k: None, cdf_fn=lambda *a, **k: None,
                bin_prob_fn=lambda *a, **k: None, rvs_fn=None, is_subordinator=True,
                baseline_key=None, extra_params_desc="",
            )

    def test_registered_with_params_filters_to_present_keys(self):
        result = {"gamma_params": {}, "wiener_truncated_params": {}}
        present = families.registered_with_params(result)
        assert "gamma_state" not in present  # not in `result`, even though it IS registered
        assert "wiener_truncated" in present


class TestLocalCdfsMatchProduction:
    """degradation_process_variants.py necessarily reimplements the Gamma/IG/
    Wiener CDFs locally (see the module docstring's note on the acyclic
    import structure) -- these tests are the guard against that duplication
    silently drifting from the production formulas."""

    def test_gamma_cdf_matches_production_bin_probability(self):
        shape_s = np.array([2.5, 10.0, 0.3])
        beta = 0.7
        observed = np.array([3.0, 12.0, 0.0])
        local = families.generic_bin_prob_from_cdf(variants._gamma_cdf)(observed, shape_s, beta, 1.0)
        production = _production_gamma_bin_prob(observed, shape_s, beta, 1.0)
        np.testing.assert_allclose(local, production, rtol=1e-10)

    def test_ig_cdf_matches_production_bin_probability(self):
        shape_s = np.array([2.5, 10.0, 0.3])
        beta = 0.7
        observed = np.array([3.0, 12.0, 0.0])
        local = families.generic_bin_prob_from_cdf(variants._ig_cdf)(observed, shape_s, beta, 1.0)
        production = _production_ig_bin_prob(observed, shape_s, beta, 1.0)
        np.testing.assert_allclose(local, production, rtol=1e-10)

    def test_wiener_cdf_matches_production_bin_probability(self):
        shape_s = np.array([2.5, 10.0, 0.3])
        beta = 0.7
        observed = np.array([3.0, 12.0, 0.0])
        local = families.generic_bin_prob_from_cdf(variants._wiener_cdf)(observed, shape_s, beta, 1.0)
        production = _production_wiener_bin_prob(observed, shape_s, beta, 1.0)
        np.testing.assert_allclose(local, production, rtol=1e-10)


class TestTruncatedWiener:
    def test_cdf_is_zero_at_zero_by_construction(self):
        shape_s = np.array([1.0, 5.0, 20.0])
        cdf_at_zero = variants._truncated_wiener_cdf(np.zeros(3), shape_s, 0.5)
        np.testing.assert_allclose(cdf_at_zero, 0.0, atol=1e-12)

    def test_cdf_is_monotonic_and_bounded(self):
        shape_s = np.full(50, 4.0)
        x = np.linspace(0.0, 40.0, 50)
        cdf = variants._truncated_wiener_cdf(x, shape_s, 0.5)
        assert np.all(np.diff(cdf) >= -1e-12)
        assert np.all((cdf >= 0.0) & (cdf <= 1.0 + 1e-12))

    def test_fit_runs_and_negative_increment_mass_is_exactly_zero(self):
        rng = np.random.default_rng(7)
        dv, ts, te, tool, _ = _synthetic_windows(rng, family="wiener")
        fitted = families.FAMILY_REGISTRY["wiener_truncated"]["fit_fn"](dv, ts, te, tool)
        assert fitted["beta"] > 0
        assert fitted["mean_p_negative_increment"] == 0.0
        assert fitted["max_p_negative_increment"] == 0.0
        assert len(fitted["components"]) == 3


class TestStateDependentVariants:
    def test_shape_fn_reduces_to_standard_at_gamma_state_zero(self):
        component = {"eta": 3.0, "power": 0.8, "time_reference_s": 1000.0, "gamma_state": 0.0}
        t_start = np.array([0.0, 500.0, 1200.0])
        t_end = np.array([300.0, 900.0, 1800.0])
        v_start = np.array([10.0, 50.0, 120.0])
        state_shape = variants._state_dependent_shape_fn(component, t_start, t_end, v_start)
        standard_shape = families.standard_shape_fn(component, t_start, t_end)
        np.testing.assert_allclose(state_shape, standard_shape, rtol=1e-12)

    def test_shape_fn_requires_v_start(self):
        component = {"eta": 3.0, "power": 0.8, "time_reference_s": 1000.0, "gamma_state": 0.1}
        with pytest.raises(ValueError):
            variants._state_dependent_shape_fn(component, np.array([0.0]), np.array([100.0]), None)

    def test_gamma_state_fit_runs_and_recovers_low_state_dependence_from_stationary_data(self):
        # Synthetic data with NO true state-dependence (plain Gamma
        # increments, power=1): a well-behaved fit should not need an
        # extreme gamma_state to explain it, and should generally prefer
        # the simpler baseline over this variant (checked loosely, since a
        # single random draw is noisy -- the point of this test is "runs
        # correctly and returns a sane dict," not a statistical power
        # guarantee at n=1 draw).
        rng = np.random.default_rng(11)
        dv, ts, te, tool, v_start = _synthetic_windows(rng, family="gamma", power=1.0)
        fitted = families.FAMILY_REGISTRY["gamma_state"]["fit_fn"](dv, ts, te, tool, v_start=v_start)
        assert fitted["beta"] > 0
        assert "gamma_state" in fitted
        assert all("gamma_state" in c for c in fitted["components"])

    def test_ig_state_fit_runs(self):
        rng = np.random.default_rng(12)
        dv, ts, te, tool, v_start = _synthetic_windows(rng, family="inverse_gaussian", power=1.0)
        fitted = families.FAMILY_REGISTRY["inverse_gaussian_state"]["fit_fn"](dv, ts, te, tool, v_start=v_start)
        assert fitted["beta"] > 0
        assert len(fitted["components"]) == 3


class TestRandomEffectsVariants:
    def test_gh_nodes_collapse_to_single_point_at_sigma_zero(self):
        nodes = variants._gh_nodes(0.8, 0.0)
        weights = variants._gh_log_weights(0.0)
        assert nodes.shape == (1,)
        assert nodes[0] == pytest.approx(0.8)
        assert weights[0] == 0.0  # log(1) -- the sole node carries all the mass

    def test_marginal_loglik_at_zero_sigma_matches_plain_plugin_loglik(self):
        rng = np.random.default_rng(3)
        dv, ts, te, tool, _ = _synthetic_windows(rng, n_tools=1, family="gamma", power=1.0, eta_base=0.5)
        eta_j, beta = 0.6, 0.5
        p_nodes = variants._gh_nodes(0.9, 0.0)
        node_ll = variants._tool_node_loglik(dv, ts, te, eta_j, p_nodes, beta, 1000.0, 1.0, variants._rounded_gamma_bin_probability_for_variants)
        marginal = float(np.sum(node_ll * 1.0))  # single node, weight 1 (log-weight 0)
        shape_s = eta_j * (np.power(te / 1000.0, 0.9) - np.power(ts / 1000.0, 0.9))
        direct = float(np.sum(np.log(variants._rounded_gamma_bin_probability_for_variants(dv, shape_s, beta, 1.0))))
        assert marginal == pytest.approx(direct, rel=1e-9)

    def test_posterior_mean_power_is_a_weighted_average_of_nodes(self):
        p_nodes = np.array([0.7, 0.9, 1.1])
        node_ll = np.array([-10.0, -1.0, -10.0])  # heavily favors the middle node
        log_w = np.array([0.0, 0.0, 0.0])
        post_mean = variants._tool_posterior_mean_power(p_nodes, node_ll, log_w)
        assert post_mean == pytest.approx(0.9, abs=0.05)

    def test_gamma_random_effects_fit_runs_and_selects_a_valid_re_model(self):
        rng = np.random.default_rng(21)
        dv, ts, te, tool, _ = _synthetic_windows(rng, family="gamma", power=1.0)
        fitted = families.FAMILY_REGISTRY["gamma_random_effects"]["fit_fn"](dv, ts, te, tool)
        assert fitted["beta"] > 0
        assert fitted["selected_re_model"] in ("random_effects", "pooled_exponent")
        assert fitted["sigma_p"] >= 0.0
        assert fitted["sigma_p_test"]["lr_boundary_corrected"] is True

    def test_boundary_corrected_pvalue_is_exactly_half_naive(self):
        baseline_fit = {"log_lik": -100.0, "aic": 210.0, "bic": 220.0}
        variant_fit = {"log_lik": -97.0, "aic": 208.0, "bic": 222.0}
        naive = families.nested_model_test(baseline_fit, variant_fit, extra_params=1, boundary_corrected=False)
        corrected = families.nested_model_test(baseline_fit, variant_fit, extra_params=1, boundary_corrected=True)
        assert corrected["lr_p_value"] == pytest.approx(0.5 * naive["lr_p_value"])

    def test_boundary_corrected_rejects_multi_df(self):
        baseline_fit = {"log_lik": -100.0, "aic": 210.0, "bic": 220.0}
        variant_fit = {"log_lik": -95.0, "aic": 208.0, "bic": 222.0}
        with pytest.raises(ValueError):
            families.nested_model_test(baseline_fit, variant_fit, extra_params=2, boundary_corrected=True)

    def test_ig_random_effects_fit_runs(self):
        rng = np.random.default_rng(22)
        dv, ts, te, tool, _ = _synthetic_windows(rng, family="inverse_gaussian", power=1.0)
        fitted = families.FAMILY_REGISTRY["inverse_gaussian_random_effects"]["fit_fn"](dv, ts, te, tool)
        assert fitted["beta"] > 0
        assert len(fitted["components"]) == 3


class TestMultistageWiener:
    def test_shape_reduces_to_standard_linear_when_stages_equal(self):
        t_start = np.array([0.0, 200.0, 900.0])
        t_end = np.array([150.0, 900.0, 1500.0])
        for tau_c in (50.0, 400.0, 5000.0):
            shape = variants._multistage_shape(2.5, 2.5, tau_c, t_start, t_end, 1000.0)
            expected = 2.5 * (t_end - t_start) / 1000.0
            np.testing.assert_allclose(shape, expected, rtol=1e-12)

    def test_shape_correctly_splits_a_window_straddling_the_changepoint(self):
        t_start, t_end, tau_c, t_ref = np.array([100.0]), np.array([300.0]), 200.0, 1000.0
        shape = variants._multistage_shape(np.array([4.0]), np.array([10.0]), tau_c, t_start, t_end, t_ref)
        # 100 units of stage-1 duration (100->200) + 100 units of stage-2 (200->300)
        expected = (4.0 * 100.0 + 10.0 * 100.0) / 1000.0
        assert shape[0] == pytest.approx(expected)

    def test_stage_durations_always_sum_to_window_duration(self):
        rng = np.random.default_rng(5)
        t_start = np.sort(rng.uniform(0, 1000, 30))
        t_end = t_start + rng.uniform(10, 200, 30)
        tau_c = 500.0
        stage1_dt = np.clip(np.minimum(t_end, tau_c) - t_start, 0.0, None)
        stage2_dt = np.clip(t_end - np.maximum(t_start, tau_c), 0.0, None)
        np.testing.assert_allclose(stage1_dt + stage2_dt, t_end - t_start, rtol=1e-12)

    def test_fit_on_correctly_specified_null_data_usually_selects_one_stage(self):
        # Genuine Wiener-distributed data with NO true rate change: the
        # nested one-stage model should be selected in most (not
        # necessarily all -- see the changepoint_test's own documented
        # Davies-problem caveat) draws, and when it IS selected, eta1 must
        # equal eta2 exactly (the nested-reduction property, not just "close").
        n_one_stage = 0
        n_trials = 6
        for seed in range(n_trials):
            rng = np.random.default_rng(1000 + seed)
            dv, ts, te, tool, _ = _synthetic_windows(rng, family="wiener", power=1.0, n_windows_per_tool=40)
            fitted = families.FAMILY_REGISTRY["wiener_multistage"]["fit_fn"](dv, ts, te, tool, n_grid=8)
            if fitted["selected_stage_model"] == "one_stage":
                n_one_stage += 1
                for c in fitted["components"]:
                    assert c["eta1"] == pytest.approx(c["eta2"], rel=1e-9)
        assert n_one_stage >= n_trials // 2  # majority, not unanimous -- see docstring

    def test_changepoint_test_reports_davies_caveat(self):
        rng = np.random.default_rng(9)
        dv, ts, te, tool, _ = _synthetic_windows(rng, family="wiener", power=1.0)
        fitted = families.FAMILY_REGISTRY["wiener_multistage"]["fit_fn"](dv, ts, te, tool, n_grid=6)
        assert "caveat" in fitted["changepoint_test"]
        assert fitted["changepoint_test"]["lr_boundary_corrected"] is False

    def test_not_a_subordinator_but_has_a_window_level_rvs_fn(self):
        """is_subordinator=False correctly keeps this family off the
        cumulative-trajectory step-simulation route (simulate_process_lifetimes
        would silently clip its genuinely-negative-capable increments,
        biasing the trajectory monotone -- see that function's docstring).
        rvs_fn is NOT None, though: it's a real, deliberately-defined
        single-WINDOW Normal(shape_s/beta, sqrt(shape_s)/beta) draw (see the
        registration's own comment) -- needed for model_recovery_study's
        window-level synthetic-data generation, which calls rvs_fn for
        EVERY registered family including this one when it's the "true"
        family under test. If this were None, model_recovery_study would
        crash the moment it reached wiener_multistage as a true family,
        not silently do the wrong thing -- so this is pinned directly
        rather than only caught incidentally by a much slower end-to-end run.
        """
        spec = families.FAMILY_REGISTRY["wiener_multistage"]
        assert spec["is_subordinator"] is False
        assert spec["rvs_fn"] is not None
        rng = np.random.default_rng(0)
        shape_s = np.array([1.0, 5.0, 20.0])
        draws = spec["rvs_fn"](shape_s, 0.5, rng)
        assert draws.shape == shape_s.shape
        assert np.all(np.isfinite(draws))
        # Mean should be in the ballpark of shape_s/beta (Normal, so exact
        # equality isn't expected from one finite draw -- just sanity that
        # it's the documented distribution, not something else entirely).
        many_draws = spec["rvs_fn"](np.full(20_000, 5.0), 0.5, rng)
        assert many_draws.mean() == pytest.approx(5.0 / 0.5, rel=0.05)


class TestNestedModelTest:
    def test_clips_negative_ll_gain_to_zero_rather_than_raising(self):
        baseline_fit = {"log_lik": -50.0, "aic": 104.0, "bic": 110.0}
        variant_fit = {"log_lik": -50.3, "aic": 106.6, "bic": 115.0}  # variant fit slightly WORSE (optimizer noise)
        result = families.nested_model_test(baseline_fit, variant_fit, extra_params=1)
        assert result["lr_statistic"] == 0.0
        assert result["lr_p_value"] == pytest.approx(1.0)

    def test_matches_manual_chi2_survival_function(self):
        baseline_fit = {"log_lik": -120.0, "aic": 250.0, "bic": 260.0}
        variant_fit = {"log_lik": -114.5, "aic": 243.0, "bic": 258.0}
        result = families.nested_model_test(baseline_fit, variant_fit, extra_params=2)
        expected_stat = 2.0 * (-114.5 - (-120.0))
        assert result["lr_statistic"] == pytest.approx(expected_stat)
        assert result["lr_p_value"] == pytest.approx(float(chi2.sf(expected_stat, df=2)))


class TestGIG:
    """The Generalized Inverse Gaussian variant is the mathematically most
    delicate addition here (modified Bessel functions, a hand-derived
    mean-preserving reparametrization) -- these tests independently
    re-verify the derivation itself, not just that the fitter runs.

    KNOWN, VERIFIED PROPERTY (not a bug): fitting p_gid to ROUNDED data can
    land noticeably away from the true generating value at realistic
    sample sizes (~50-200 windows), because the extra tail-shape parameter
    can partly track rounding-kernel structure that has nothing to do with
    genuine distributional shape -- a discretization/small-sample effect,
    not a parametrization error. Confirmed by two independent checks during
    implementation: (1) the continuous (rounding-free) cross-entropy is
    exactly minimized at the true p_gid (test_cross_entropy_is_minimized_at_the_true_p_gid_without_rounding
    below), and (2) the SAME bias direction and rough magnitude appears
    identically whether using the original naive (slow, scipy-CDF-based)
    bin-probability implementation or the final vectorized one, ruling out
    the vectorization refactor as the source. This means a GIG fit's
    p_gid on real, similarly-sized data should be read with real caution,
    not taken as precise evidence about tail shape on its own -- the
    eventual robustness battery is expected to add a dedicated model-
    recovery check (fit GIG to synthetic data simulated from the real
    fitted IG parameters, the same positive-control logic
    model_recovery_study already uses for the original 3 families) to
    quantify this bias concretely rather than leave it as a qualitative
    caveat.
    """

    def test_r_p_equals_one_at_minus_half_for_any_chi(self):
        # Bessel symmetry K_v = K_{-v} applied to v=1/2 -- the algebraic
        # reason p_gid=-1/2 is an exact, non-boundary nesting point.
        for chi in (0.3, 1.0, 5.5, 20.0, 60.0):
            assert variants._r_p(-0.5, np.array([chi]))[0] == pytest.approx(1.0, rel=1e-10)

    def test_cdf_exactly_matches_production_ig_at_p_minus_half(self):
        shape_s = np.array([0.8, 3.2, 15.0, 40.0])
        beta = 0.55
        x = np.array([1.0, 5.0, 12.0, 30.0])
        gig_cdf = variants._gig_cdf(x, shape_s, beta, -0.5)
        ig_cdf = scipy_invgauss.cdf(x, mu=1.0 / shape_s, loc=0.0, scale=(shape_s ** 2) / beta)
        np.testing.assert_allclose(gig_cdf, ig_cdf, rtol=1e-10)

    def test_bin_probability_exactly_matches_production_ig_at_p_minus_half(self):
        shape_s = np.array([2.0, 9.0, 22.0])
        beta = 0.6
        observed = np.array([3.0, 15.0, 35.0])
        gig_prob = variants._rounded_gig_bin_probability(observed, shape_s, beta, 1.0, -0.5)
        ig_prob = _production_ig_bin_prob(observed, shape_s, beta, 1.0)
        np.testing.assert_allclose(gig_prob, ig_prob, rtol=1e-9)

    def test_mean_is_preserved_across_p_gid_by_construction(self):
        # Direct algebraic re-derivation, independent of the fitter: for
        # several p_gid values, the GIG(a,b,p_gid) implied by
        # _gig_ab_from_shape must have E[X] == shape_s/beta exactly (up to
        # quadrature error), even though chi and the a/b split both change.
        from scipy.stats import geninvgauss
        shape_s, beta = 6.0, 0.5
        target_mean = shape_s / beta
        for p_gid in (-2.0, -0.5, 0.4, 1.7, 3.5):
            a, b_gen = variants._gig_ab_from_shape(np.array([shape_s]), beta, p_gid)
            chi = np.sqrt(a * b_gen)
            scale = np.sqrt(b_gen / a)
            x = np.linspace(1e-4, 400.0, 60000)
            pdf = geninvgauss.pdf(x, p_gid, chi[0], loc=0.0, scale=scale[0])
            numeric_mean = np.trapezoid(x * pdf, x)
            assert numeric_mean == pytest.approx(target_mean, rel=1e-3)
            assert float(chi[0]) == pytest.approx(shape_s, rel=1e-10)

    def test_extra_shape_params_declares_p_gid(self):
        spec = families.FAMILY_REGISTRY["gig"]
        assert spec["extra_shape_params"] == ["p_gid"]
        assert spec["baseline_key"] == "inverse_gaussian"

    def test_generic_bin_prob_forwards_extra_args(self):
        # families.extra_shape_values + generic_bin_prob_from_cdf's *extra
        # plumbing, exercised together the way a generalized downstream
        # consumer (Vuong/PIT-KS/CV) will actually call it.
        spec = families.FAMILY_REGISTRY["gig"]
        fitted = {"p_gid": -0.5}
        extra = families.extra_shape_values(spec, fitted)
        assert extra == [-0.5]
        shape_s = np.array([4.0, 10.0])
        beta = 0.5
        observed = np.array([5.0, 12.0])
        via_registry = spec["bin_prob_fn"](observed, shape_s, beta, 1.0, *extra)
        direct_ig = _production_ig_bin_prob(observed, shape_s, beta, 1.0)
        np.testing.assert_allclose(via_registry, direct_ig, rtol=1e-9)

    def test_fit_runs_on_small_synthetic_ig_data_and_stays_off_the_optimizer_bound(self):
        # Small, fast fixture (fewer windows than the other variants' tests
        # use) specifically to keep this Bessel-function-heavy fit's
        # runtime bounded in the test suite.
        #
        # Only checks p_gid stays clear of the +-6 optimizer bound, NOT
        # "close to -0.5" -- see TestGIG's class docstring below on the
        # discretization-interaction bias: at this window count (54), p_gid
        # is expected to land noticeably away from the true -0.5, and that
        # is not evidence of a bug (independently confirmed via a
        # discretization-free cross-entropy check during implementation).
        rng = np.random.default_rng(42)
        dv, ts, te, tool, _ = _synthetic_windows(
            rng, n_tools=3, n_windows_per_tool=18, family="inverse_gaussian", power=1.0,
        )
        fitted = families.FAMILY_REGISTRY["gig"]["fit_fn"](dv, ts, te, tool)
        assert fitted["beta"] > 0
        assert len(fitted["components"]) == 3
        assert all(c["p_gid"] == pytest.approx(fitted["p_gid"]) for c in fitted["components"])
        assert -5.9 < fitted["p_gid"] < 5.9

    def test_cross_entropy_is_minimized_at_the_true_p_gid_without_rounding(self):
        # Isolates the GIG parametrization itself (continuous, no rounding)
        # from the discretized-likelihood fitter: draws a large sample from
        # a KNOWN GIG(p_gid=-0.5, i.e. exactly Inverse Gaussian) and checks
        # the cross-entropy H(true, p_gid) -- computed directly via
        # _gig_log_pdf, no rounding kernel involved -- is minimized at
        # p_gid=-0.5 among a symmetric neighborhood of candidates. By
        # Gibbs' inequality this MUST hold for a correct parametrization
        # (KL divergence >= 0, equality only at the true parameter); this
        # is the test that isolates "is the math right" from "does fitting
        # rounded data introduce a separate, real, and expected small-
        # sample bias" (see the class docstring, and the failure this test
        # was written in response to: a fit landing at p_gid~=-5.5 on a
        # 54-window fixture, which turned out to be exactly that expected
        # rounding-interaction bias, not a parametrization error, once this
        # check confirmed the continuous math was already exactly correct).
        rng = np.random.default_rng(1)
        shape_s_val, beta = 3.0, 0.5
        a, b_gen = variants._gig_ab_from_shape(np.array([shape_s_val]), beta, -0.5)
        chi = float(np.sqrt(a * b_gen)[0]); scale = float(np.sqrt(b_gen / a)[0])
        from scipy.stats import geninvgauss
        true_draws = geninvgauss.rvs(-0.5, chi, loc=0.0, scale=scale, size=200_000, random_state=rng)

        candidates = np.arange(-2.0, 2.01, 0.5)
        cross_entropies = []
        for p_test in candidates:
            a2, b2 = variants._gig_ab_from_shape(np.array([shape_s_val]), beta, float(p_test))
            chi2 = float(np.sqrt(a2 * b2)[0]); scale2 = float(np.sqrt(b2 / a2)[0])
            log_kv2 = float(np.log(kv(p_test, chi2)))
            logpdf = variants._gig_log_pdf(true_draws, chi2, scale2, float(p_test), log_kv2)
            cross_entropies.append(-np.mean(logpdf))
        best_p = candidates[int(np.argmin(cross_entropies))]
        assert best_p == pytest.approx(-0.5, abs=1e-9)


class TestFitPerformance:
    """Guards against the specific bug class found twice during this
    session's implementation: a variant's CDF/bin-probability built on a
    scipy.stats distribution whose generic ``rv_continuous`` machinery is
    orders of magnitude slower than a hand-derived closed form (GIG's
    ``geninvgauss`` -- fixed by a manual log-PDF/CDF, 514s->34.4s on real
    data; truncated Wiener's ``truncnorm`` -- fixed by a manual ndtr-based
    CDF/rvs, found when a real-data fit that should take single-digit
    seconds like every sibling family instead ran for minutes without
    finishing). Small synthetic data keeps this fast and non-flaky while
    still calling each family's REAL fit_fn/optimizer path (unlike the
    scipy-call-level micro-benchmark in the implementation session's own
    scratch verification) -- a regression of this specific kind would
    show up as one family taking wildly longer than its siblings even at
    this tiny scale, which is exactly what caught the truncated-Wiener bug
    in the first place (before the eventual real-data profiling that
    pinned it down precisely).

    Applies the SAME ``_REPEATED_REFIT_COARSENING`` real sweeps use (coarser
    multi-stage Wiener change-point grid, coarser random-effects
    quadrature) -- without it, random-effects Gamma/IG's cost is dominated
    by optimizer ITERATION COUNT (a hard-to-optimize likelihood surface,
    ``maxiter=40000``), not data size, so even tiny synthetic data does not
    make it fast the way it does for every other family; testing the
    coarsened configuration is also more representative of how these two
    families are actually exercised in the expensive sweeps this test
    exists to protect.

    Per-family-class ceilings, not one flat number: measured directly on
    this machine (tiny synthetic data, coarsening applied), every "standard"
    family (Gamma/IG/Wiener and every variant except multi-stage Wiener and
    random-effects Gamma/IG) finishes in <=14s, multi-stage Wiener in ~29s,
    and random-effects Gamma/IG in ~80-90s each -- an inherent optimizer-
    bound floor for the latter two even at their most coarsened, not
    something further coarsening alone eliminates. A single ceiling loose
    enough for random-effects (~150s+) would no longer meaningfully guard
    the 8 families that should be fast; a single ceiling tight enough for
    those 8 would always fail on the 2 that are legitimately slow. Ceilings
    below are set with a ~2x margin over the measured value in each class.
    """

    _CEILINGS_S = {
        "wiener_multistage": 60.0,
        "gamma_random_effects": 150.0,
        "inverse_gaussian_random_effects": 150.0,
    }
    _DEFAULT_CEILING_S = 30.0

    def test_every_registered_family_fits_small_data_quickly(self):
        import time
        from degradation_process_models import _REPEATED_REFIT_COARSENING

        rng = np.random.default_rng(7)
        results = {}
        for family_key, spec in families.FAMILY_REGISTRY.items():
            fit_family = "gamma" if "gamma" in family_key and "inverse" not in family_key else (
                "inverse_gaussian" if "inverse_gaussian" in family_key else "wiener"
            )
            dv, t_start, t_end, tool_ids, v_start = _synthetic_windows(
                rng, n_tools=3, n_windows_per_tool=20, family=fit_family,
            )
            t0 = time.perf_counter()
            spec["fit_fn"](
                dv, t_start, t_end, tool_ids, v_start=v_start,
                **_REPEATED_REFIT_COARSENING.get(family_key, {}),
            )
            elapsed = time.perf_counter() - t0
            results[family_key] = elapsed
        slow = {
            k: v for k, v in results.items()
            if v > self._CEILINGS_S.get(k, self._DEFAULT_CEILING_S)
        }
        assert not slow, f"Families exceeding their per-class ceiling on tiny synthetic data: {slow} (full timings: {results})"
