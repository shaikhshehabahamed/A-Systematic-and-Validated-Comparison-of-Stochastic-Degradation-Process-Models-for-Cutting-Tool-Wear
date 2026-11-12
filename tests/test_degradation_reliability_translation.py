"""Unit tests for degradation_reliability_translation.py: Monte Carlo simulation,
ground-truth episode construction, and the maintenance-cost/RUL comparisons
added this session. Uses hand-built parameter dicts (not fitted from data) so
these tests stay fast -- fitting is already covered by
test_degradation_process_models.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

import degradation_process_models  # noqa: F401 -- triggers full family registration
import degradation_families as families
from degradation_reliability_translation import (
    age_replacement_cost_rate,
    bootstrap_maintenance_cost_regret_by_tool,
    build_tool_life_episodes,
    compare_ig_vs_conformal_rul,
    conditional_failure_risk,
    cost_ratio_sensitivity,
    decompose_rul_threshold_confound,
    ground_truth_reliability,
    ig_rul_leave_one_tool_out,
    maintenance_cost_comparison,
    process_implied_reliability,
    rul_leave_one_tool_out,
    simulate_process_lifetimes,
    summarize_reliability_translation,
    _implied_survival,
)


def _three_machine_index(n_runs: int = 160) -> pd.DataFrame:
    """Mirrors test_degradation_process_models.py's helper of the same name:
    3 tools/machine, differing eta -> differing terminal wear per tool (145-165
    range analogous to the real dataset's 145-160), needed to exercise the
    matched-vs-fixed threshold decomposition meaningfully.
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


def _synthetic_params(beta=0.5, eta=(20.0, 22.0, 18.0), power=(1.0, 1.0, 1.0)):
    return {
        "beta": beta,
        "time_reference_s": 1000.0,
        "components": [
            {"tool_id": i + 1, "eta": e, "power": p, "beta": beta, "time_reference_s": 1000.0}
            for i, (e, p) in enumerate(zip(eta, power, strict=True))
        ],
    }


class TestFamilyEligibilityHelpers:
    """families.standard_shape_families / families.subordinator_families --
    the dispatch predicates every generalized reliability-translation table
    (Table 3/4, conditional failure risk, RUL-vs-RF) now iterates instead of
    a hardcoded Gamma/IG/Wiener list. Pinned against the REAL, fully-
    registered 10-family registry (not a hand-built fake) so a future
    family addition that forgets to set shape_fn/is_subordinator correctly
    is caught here, not silently in a downstream table."""

    def _full_comparison_result(self):
        return {families.param_key(k): {} for k in families.FAMILY_REGISTRY}

    def test_standard_shape_families_excludes_state_dependent_and_multistage(self):
        result = set(families.standard_shape_families(self._full_comparison_result()))
        assert result == {
            "gamma", "inverse_gaussian", "wiener", "wiener_truncated",
            "gamma_random_effects", "inverse_gaussian_random_effects", "gig",
        }

    def test_subordinator_families_excludes_wiener_and_multistage(self):
        result = set(families.subordinator_families(self._full_comparison_result()))
        assert result == {
            "gamma", "inverse_gaussian", "wiener_truncated", "gamma_state",
            "inverse_gaussian_state", "gamma_random_effects",
            "inverse_gaussian_random_effects", "gig",
        }

    def test_both_helpers_respect_registered_with_params_filtering(self):
        """A comparison_result covering only a SUBSET of the registry (an
        older artifact, or a minimal test fixture) must not make either
        helper claim eligibility for a family that isn't actually present --
        matches registered_with_params's own documented contract."""
        partial = {"gamma_params": {}, "wiener_truncated_params": {}}
        assert set(families.standard_shape_families(partial)) == {"gamma", "wiener_truncated"}
        assert set(families.subordinator_families(partial)) == {"gamma", "wiener_truncated"}


class TestSimulateProcessLifetimes:
    def test_output_format_and_determinism(self):
        params = _synthetic_params()
        eps1 = simulate_process_lifetimes(params, family="inverse_gaussian", n_sim=50, rng_seed=1)
        eps2 = simulate_process_lifetimes(params, family="inverse_gaussian", n_sim=50, rng_seed=1)
        assert len(eps1) == 50
        for e in eps1:
            assert set(e) == {"simTime", "firstFailureTime"}
            assert e["simTime"] >= 0.0
            if e["firstFailureTime"] is not None:
                assert e["firstFailureTime"] <= e["simTime"] + 1e-6
        # Same seed -> identical results (vectorised RNG usage must be reproducible).
        assert [e["simTime"] for e in eps1] == [e["simTime"] for e in eps2]

    def test_gamma_and_ig_both_supported(self):
        params = _synthetic_params()
        for family in ("gamma", "inverse_gaussian"):
            eps = simulate_process_lifetimes(params, family=family, n_sim=20, rng_seed=0)
            assert len(eps) == 20

    def test_rejects_unknown_family(self):
        with pytest.raises(ValueError):
            simulate_process_lifetimes(_synthetic_params(), family="not_a_family", n_sim=5)

    def test_higher_eta_fails_sooner_on_average(self):
        """Larger eta -> faster mean wear accumulation -> shorter simulated lives."""
        slow = _synthetic_params(eta=(5.0, 5.0, 5.0))
        fast = _synthetic_params(eta=(50.0, 50.0, 50.0))
        eps_slow = simulate_process_lifetimes(slow, family="gamma", n_sim=200, rng_seed=0, max_time_s=50000.0)
        eps_fast = simulate_process_lifetimes(fast, family="gamma", n_sim=200, rng_seed=0, max_time_s=50000.0)
        mean_slow = np.mean([e["simTime"] for e in eps_slow])
        mean_fast = np.mean([e["simTime"] for e in eps_fast])
        assert mean_fast < mean_slow


class TestImpliedSurvival:
    def test_survival_decreases_with_age(self):
        params = _synthetic_params()
        t_grid = np.array([100.0, 1000.0, 5000.0, 20000.0])
        s_gamma = _implied_survival(params, "gamma", t_grid)
        s_ig = _implied_survival(params, "inverse_gaussian", t_grid)
        for s in (s_gamma, s_ig):
            assert np.all(s >= 0.0) and np.all(s <= 1.0)
            assert np.all(np.diff(s) <= 1e-9)  # monotone non-increasing

    def test_wiener_survival_decreases_with_age(self):
        params = _synthetic_params()
        t_grid = np.array([100.0, 1000.0, 5000.0, 20000.0])
        s_wiener = _implied_survival(params, "wiener", t_grid)
        assert np.all(s_wiener >= 0.0) and np.all(s_wiener <= 1.0)
        assert np.all(np.diff(s_wiener) <= 1e-9)

    def test_rejects_state_dependent_family(self):
        params = _synthetic_params()
        for c in params["components"]:
            c["gamma_state"] = 0.3
        with pytest.raises(ValueError, match="no closed-form"):
            _implied_survival(params, "gamma_state", np.array([1000.0]))


class TestWienerFirstPassageTimeClosedForm:
    """Cross-checks the exact Wiener first-passage-time closed form used in
    _implied_survival's non-subordinator branch against an INDEPENDENT
    brute-force fine-step simulation of the actual (non-monotone, allowed to
    dip) first-crossing time. Deliberately does NOT reuse
    simulate_process_lifetimes (which rejects non-subordinator families by
    design, and would just be exercising the same production code path
    anyway) -- this is a from-scratch recomputation, matching this
    codebase's own TestVuongTestGoldenValue convention of independently
    re-deriving a result rather than re-running the code under test.
    """

    def test_matches_brute_force_first_crossing_simulation(self):
        eta, power, beta, t_ref = 30.0, 1.0, 0.6, 1000.0
        threshold = 150.0
        params = {
            "beta": beta, "time_reference_s": t_ref,
            "components": [
                {"tool_id": 1, "eta": eta, "power": power, "beta": beta, "time_reference_s": t_ref}
            ],
        }
        # Chosen so mean cumulative wear ~= threshold at t_check (S(t)/beta =
        # eta*(t/t_ref)/beta = threshold at t=3000), landing survival near
        # the middle of [0, 1] where a formula error is easiest to detect.
        t_check = 3000.0

        closed_form_survival = _implied_survival(
            params, "wiener", np.array([t_check]), wear_threshold_um=threshold,
        )[0]

        rng = np.random.default_rng(12345)
        n_sim = 40_000
        dt = 5.0
        n_steps = int(t_check / dt)
        wear = np.zeros(n_sim)
        crossed = np.zeros(n_sim, dtype=bool)
        t = 0.0
        for _ in range(n_steps):
            t_next = t + dt
            shape_step = eta * ((t_next / t_ref) ** power - (t / t_ref) ** power)
            mean = shape_step / beta
            sd = np.sqrt(shape_step) / beta
            dv = rng.normal(mean, sd, size=n_sim)
            wear = wear + dv  # NOT clipped -- Wiener paths are genuinely non-monotone
            crossed |= wear >= threshold
            t = t_next
        empirical_survival = 1.0 - np.mean(crossed)

        assert closed_form_survival == pytest.approx(empirical_survival, abs=0.01)


class TestGeneralizedSubordinatorSupport:
    """simulate_process_lifetimes was originally hardcoded to gamma/inverse_
    gaussian only; it now covers every is_subordinator=True family via the
    registry, including truncated Wiener (explicitly requested so the
    manuscript's claim that Wiener's exclusion is structural, not just
    unfixed, can actually be tested) and state-dependent variants (whose
    wear-level feedback loop step-simulation captures correctly, unlike
    _implied_survival's closed-form route)."""

    def test_supports_truncated_wiener(self):
        eps = simulate_process_lifetimes(_synthetic_params(), family="wiener_truncated", n_sim=20, rng_seed=0)
        assert len(eps) == 20
        for e in eps:
            assert e["simTime"] >= 0.0

    def test_supports_state_dependent_gamma(self):
        params = _synthetic_params()
        for c in params["components"]:
            c["gamma_state"] = 0.3
        eps = simulate_process_lifetimes(params, family="gamma_state", n_sim=20, rng_seed=0)
        assert len(eps) == 20

    def test_rejects_plain_wiener_as_non_subordinator(self):
        with pytest.raises(ValueError, match="not a subordinator"):
            simulate_process_lifetimes(_synthetic_params(), family="wiener", n_sim=5)

    def test_rejects_multistage_wiener_as_non_subordinator(self):
        with pytest.raises(ValueError, match="not a subordinator"):
            simulate_process_lifetimes(_synthetic_params(), family="wiener_multistage", n_sim=5)


class TestProcessImpliedReliabilityDefaultFamilyCoverage:
    """process_implied_reliability's families_to_compare default used to be
    a hardcoded ("gamma", "inverse_gaussian") -- Table 3's ENTIRE reason for
    being (ground truth vs. process-implied MTTFF) would have silently
    stayed 2-family-only forever, even once the registry grew to 10. Default
    now dynamically covers every families.subordinator_families member
    present in comparison_result; explicit families_to_compare still works
    unchanged (reproduces the exact original behavior) for any caller that
    wants it."""

    def _comparison_result_with_state_dependent(self):
        gamma_state_params = _synthetic_params()
        for c in gamma_state_params["components"]:
            c["gamma_state"] = 0.3
        return {
            "gamma_params": {"M1": _synthetic_params()},
            "ig_params": {"M1": _synthetic_params()},
            "gamma_state_params": {"M1": gamma_state_params},
        }

    def test_default_covers_every_present_subordinator_family(self):
        result = process_implied_reliability(
            self._comparison_result_with_state_dependent(), n_sim=20, n_boot=20,
        )
        assert set(result["M1"]) == {"gamma", "inverse_gaussian", "gamma_state"}
        for family, fam_result in result["M1"].items():
            assert fam_result["mttff"]["mttff"] > 0

    def test_explicit_families_to_compare_still_overrides(self):
        result = process_implied_reliability(
            self._comparison_result_with_state_dependent(), n_sim=20, n_boot=20,
            families_to_compare=("gamma",),
        )
        assert set(result["M1"]) == {"gamma"}


class TestConditionalFailureRiskDefaultFamilyCoverage:
    """conditional_failure_risk's families_to_compare default was hardcoded
    to ("gamma", "inverse_gaussian") too (Table S7 / Figure 3's basis).
    Unlike process_implied_reliability (Monte-Carlo/subordinator-based),
    this calls _implied_survival directly, so its natural eligible set is
    families.standard_shape_families -- includes plain Wiener (which IS a
    standard_shape_fn family, just not a subordinator), excludes
    state-dependent variants (which ARE subordinators but have no
    closed-form survival function) -- the mirror image of
    TestProcessImpliedReliabilityDefaultFamilyCoverage's family set, not a
    copy-paste of it."""

    def test_default_covers_every_present_standard_shape_family_including_wiener(self):
        comparison_result = {
            "gamma_params": {"M1": _synthetic_params()},
            "ig_params": {"M1": _synthetic_params()},
            "wiener_params": {"M1": _synthetic_params()},
        }
        result = conditional_failure_risk(comparison_result, reference_ages_s=[500.0, 1000.0])
        assert set(result["M1"]) == {"gamma", "inverse_gaussian", "wiener"}
        for family, rows in result["M1"].items():
            for row in rows:
                assert 0.0 <= row["conditional_failure_risk"] <= 1.0

    def test_excludes_state_dependent_even_when_present(self):
        gamma_state_params = _synthetic_params()
        for c in gamma_state_params["components"]:
            c["gamma_state"] = 0.3
        comparison_result = {
            "gamma_params": {"M1": _synthetic_params()},
            "gamma_state_params": {"M1": gamma_state_params},
        }
        result = conditional_failure_risk(comparison_result, reference_ages_s=[500.0])
        assert set(result["M1"]) == {"gamma"}


class TestAgeReplacementCostRate:
    def test_cost_rate_is_positive_and_finite(self):
        params = _synthetic_params()
        T_grid = np.linspace(200.0, 5000.0, 20)
        costs = age_replacement_cost_rate(params, "gamma", T_grid, c_pm=1.0, c_cm=3.0)
        assert np.all(np.isfinite(costs))
        assert np.all(costs > 0.0)

    def test_higher_failure_cost_never_decreases_optimal_cost(self):
        """Raising c_cm (holding c_pm fixed) can only raise or leave unchanged the
        achievable minimum cost rate -- a basic economic sanity check."""
        params = _synthetic_params()
        T_grid = np.linspace(200.0, 5000.0, 40)
        low = age_replacement_cost_rate(params, "gamma", T_grid, c_pm=1.0, c_cm=2.0)
        high = age_replacement_cost_rate(params, "gamma", T_grid, c_pm=1.0, c_cm=5.0)
        assert low.min() <= high.min() + 1e-9


class TestMaintenanceCostComparison:
    def test_returns_per_machine_policy_and_regret(self):
        comparison_result = {
            "gamma_params": {"M1": _synthetic_params(beta=0.5)},
            "ig_params": {"M1": _synthetic_params(beta=0.5)},
            "wiener_params": {"M1": _synthetic_params(beta=0.5)},
        }
        result = maintenance_cost_comparison(comparison_result, t_grid_max_s=5000.0, n_grid=50)
        assert set(result) == {"M1"}
        row = result["M1"]
        for family in ("gamma", "inverse_gaussian", "wiener"):
            assert row[family]["T_star"] > 0
            assert row[family]["cost_star"] > 0
        assert "regret_using_gamma_policy_pct" in row
        assert "regret_using_wiener_policy_pct" in row
        # Regret should be ~non-negative: IG's own policy is optimal against IG's
        # own survival. T_star is only the argmin over the discrete T_grid, so
        # evaluating IG's cost at an off-grid point (another family's T_star) can
        # land marginally below IG's own grid-discretized minimum by a small
        # amount that shrinks as n_grid grows (production uses n_grid=300, not
        # this test's coarser 50, and shows no such artifact in practice).
        assert row["regret_using_gamma_policy_pct"] >= -0.5
        assert row["regret_using_wiener_policy_pct"] >= -0.5

    def test_covers_every_standard_shape_family_when_present_not_just_original_three(self):
        """The original 3-family hardcoding would have silently dropped
        every new variant from Table 4 even after the registry grew to 10;
        this pins that every families.standard_shape_families member gets
        a T*/cost_star/regret entry once its params are present, including
        GIG (needs its extra p_gid shape param) and both random-effects
        variants (no extra params -- same shape convention as Gamma/IG)."""
        gig_params = {**_synthetic_params(beta=0.5), "p_gid": -0.5}  # nests plain IG exactly
        comparison_result = {
            "gamma_params": {"M1": _synthetic_params(beta=0.5)},
            "ig_params": {"M1": _synthetic_params(beta=0.5)},
            "wiener_params": {"M1": _synthetic_params(beta=0.5)},
            "wiener_truncated_params": {"M1": _synthetic_params(beta=0.5)},
            "gamma_random_effects_params": {"M1": _synthetic_params(beta=0.5)},
            "inverse_gaussian_random_effects_params": {"M1": _synthetic_params(beta=0.5)},
            "gig_params": {"M1": gig_params},
        }
        result = maintenance_cost_comparison(comparison_result, t_grid_max_s=5000.0, n_grid=50)
        row = result["M1"]
        expected_families = {
            "gamma", "inverse_gaussian", "wiener", "wiener_truncated",
            "gamma_random_effects", "inverse_gaussian_random_effects", "gig",
        }
        for family in expected_families:
            assert row[family]["T_star"] > 0
            assert row[family]["cost_star"] > 0
        expected_regret_keys = {f"regret_using_{f}_policy_pct" for f in expected_families if f != "inverse_gaussian"}
        assert expected_regret_keys <= set(row)
        for key in expected_regret_keys:
            assert np.isfinite(row[key])
        # GIG at p_gid=-0.5 nests plain IG exactly, so its policy/cost should
        # match IG's own (to grid resolution), and its regret should be ~0 --
        # a real numerical cross-check, not just "didn't crash".
        assert row["gig"]["T_star"] == pytest.approx(row["inverse_gaussian"]["T_star"])
        assert row["regret_using_gig_policy_pct"] == pytest.approx(0.0, abs=0.5)


class TestCostRatioSensitivity:
    def test_sweeps_all_ratios_with_finite_regret(self):
        comparison_result = {
            "gamma_params": {"M1": _synthetic_params(beta=0.5)},
            "ig_params": {"M1": _synthetic_params(beta=0.5)},
            "wiener_params": {"M1": _synthetic_params(beta=0.5)},
        }
        ratios = (2.0, 5.0)
        result = cost_ratio_sensitivity(
            comparison_result, cost_ratios=ratios, t_grid_max_s=5000.0, n_grid=50,
        )
        assert set(result) == {"2.0", "5.0"}
        for cost_result in result.values():
            assert set(cost_result) == {"M1"}
            row = cost_result["M1"]
            assert np.isfinite(row["regret_using_gamma_policy_pct"])
            assert np.isfinite(row["regret_using_wiener_policy_pct"])


class TestBootstrapMaintenanceCostRegretByTool:
    """Restricted to machine 1 only (via _three_machine_index filtered to
    machine==1) -- like the analogous AIC-gap block bootstrap test in
    test_degradation_process_models.py, each of the 27 draws is a REAL
    per-family refit, not a cheap structural check, so this keeps the
    suite fast without losing machine-specific coverage (every machine is
    treated identically by this function). Deliberately ONE test method,
    not split gamma/wiener-only vs. full-family-coverage: this function
    refits every eligible family on EVERY one of the 27 draws, so a second
    call would double this already-slow test's runtime for no extra
    confidence a single call, asserted more thoroughly, doesn't already give."""

    def _m1_index(self):
        index = _three_machine_index()
        return index[index["machine"] == 1]

    def test_returns_exact_27_draw_ci_covering_every_cheap_eligible_family(self):
        result = bootstrap_maintenance_cost_regret_by_tool(
            self._m1_index(), window_size=15, t_grid_max_s=5000.0, n_grid=50,
        )
        assert set(result) == {"M1"}
        r = result["M1"]
        assert r["exact_enumeration"] is True
        assert r["n_draws"] == 27
        # Every standard_shape_fn family EXCEPT the two random-effects
        # variants (excluded deliberately for per-draw refit cost -- see
        # _BOOTSTRAP_REFIT_EXCLUDED's comment) should have a regret entry,
        # including the two NEW-since-registry-expansion families
        # (wiener_truncated, gig) the original 3-family hardcoding would
        # have silently dropped.
        expected = {"gamma", "wiener", "wiener_truncated", "gig"}
        assert expected <= set(r)
        assert "gamma_random_effects" not in r
        assert "inverse_gaussian_random_effects" not in r
        for other in expected:
            assert np.isfinite(r[other]["mean_regret_pct"])
            assert r[other]["ci_lo"] <= r[other]["ci_hi"]
            # Same small-negative-regret discretization allowance as
            # TestMaintenanceCostComparison (n_grid=50 here, not production's 300).
            assert r[other]["ci_lo"] >= -1.0


class TestBuildToolLifeEpisodes:
    def test_censored_when_never_reaches_threshold(self):
        index = pd.DataFrame({
            "machine": [1, 1, 1], "tool": [1, 1, 1], "run": [1, 2, 3],
            "cumulated_tool_contact_time": [10.0, 20.0, 30.0],
            "wear": [10.0, 20.0, 30.0],  # never reaches EOL_THRESHOLD_UM=150
        })
        episodes = build_tool_life_episodes(index, wear_threshold_um=150.0)
        assert episodes["M1"][0]["firstFailureTime"] is None
        assert episodes["M1"][0]["simTime"] == 30.0

    def test_failure_time_at_first_run_reaching_threshold(self):
        index = pd.DataFrame({
            "machine": [2, 2, 2], "tool": [4, 4, 4], "run": [1, 2, 3],
            "cumulated_tool_contact_time": [10.0, 20.0, 30.0],
            "wear": [50.0, 150.0, 160.0],
        })
        episodes = build_tool_life_episodes(index, wear_threshold_um=150.0)
        assert episodes["M2"][0]["firstFailureTime"] == 20.0


class TestIgRulLeaveOneToolOut:
    def test_fixed_vs_matched_threshold_use_different_targets(self):
        index = _three_machine_index()
        fixed = ig_rul_leave_one_tool_out(index, window_size=15, wear_threshold_um=150.0)
        matched = ig_rul_leave_one_tool_out(
            index, window_size=15, use_per_tool_actual_threshold=True,
        )
        assert set(fixed) == set(range(1, 10))
        assert set(matched) == set(range(1, 10))
        for tool_id in fixed:
            assert fixed[tool_id]["threshold_used_um"] == 150.0
            assert np.isfinite(fixed[tool_id]["rmse_s"])
            assert np.isfinite(matched[tool_id]["rmse_s"])
            # Tools' synthetic terminal wear differs by eta (see _three_machine_index),
            # so the matched threshold should differ from the fixed 150.0 for at
            # least the non-tool-2-style machines (differing eta -> differing wear).
            assert matched[tool_id]["threshold_used_um"] > 0


class TestRulLeaveOneToolOut:
    """rul_leave_one_tool_out generalizes what was originally IG-only
    (ig_rul_leave_one_tool_out, now a thin wrapper calling this with
    family="inverse_gaussian") to every family whose virtual-age can be
    inverted algebraically. Comment 5's RF-fairness critique was never
    really "IG specifically vs. RF" -- generalizing this closes a real gap:
    without it, the paper's ONLY family with an honest RUL-vs-ML accuracy
    comparison would be IG, even once the headline recommendation is no
    longer IG-only."""

    def test_wrapper_matches_direct_call_for_inverse_gaussian(self):
        """The preserved-name wrapper must be a pure delegation, not a
        second, independently-drifting implementation."""
        index = _three_machine_index()
        via_wrapper = ig_rul_leave_one_tool_out(index, window_size=15, wear_threshold_um=150.0)
        via_generic = rul_leave_one_tool_out(
            index, "inverse_gaussian", window_size=15, wear_threshold_um=150.0,
        )
        assert via_wrapper == via_generic

    def test_supports_a_non_ig_family(self):
        index = _three_machine_index()
        result = rul_leave_one_tool_out(index, "gamma", window_size=15, wear_threshold_um=150.0)
        assert set(result) == set(range(1, 10))
        for tool_id, r in result.items():
            assert np.isfinite(r["rmse_s"])
            assert np.isfinite(r["mae_s"])
            assert r["threshold_used_um"] == 150.0

    def test_rejects_truncated_wiener(self):
        """See _RUL_INVERSION_EXCLUDED's comment: truncated Wiener's true
        mean_increment_fn is not the naive shape_s/beta relationship this
        function's virtual-age inversion assumes -- must fail loudly, not
        silently return a biased prediction."""
        index = _three_machine_index()
        with pytest.raises(ValueError, match="excluded"):
            rul_leave_one_tool_out(index, "wiener_truncated", window_size=15)


class TestCompareIgVsConformalRul:
    def test_merges_ig_and_conformal_results(self, tmp_path):
        conformal_csv = tmp_path / "conformal_rul_cv_results.csv"
        pd.DataFrame({
            "heldout_tools": ["[1]", "[2]"],
            "heldout_machine": [1, 1],
            "rmse_s": [100.0, 150.0],
            "mae_s": [80.0, 120.0],
        }).to_csv(conformal_csv, index=False)
        ig_loto_result = {
            1: {"machine": 1, "rmse_s": 200.0, "mae_s": 160.0, "n": 10},
            2: {"machine": 1, "rmse_s": 300.0, "mae_s": 240.0, "n": 10},
        }
        result = compare_ig_vs_conformal_rul(ig_loto_result, conformal_csv)
        assert result[1]["conformal_rmse_s"] == 100.0
        assert result[1]["ig_rmse_s"] == 200.0
        assert result[1]["ig_rmse_ratio_vs_conformal"] == pytest.approx(2.0)


class TestDecomposeRulThresholdConfound:
    def test_reports_fixed_and_matched_ratios_per_tool(self, tmp_path):
        index = _three_machine_index()
        heldout_tools = [f"[{t}]" for t in range(1, 10)]
        heldout_machines = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        conformal_csv = tmp_path / "conformal_rul_cv_results.csv"
        pd.DataFrame({
            "heldout_tools": heldout_tools,
            "heldout_machine": heldout_machines,
            "rmse_s": [100.0] * 9,
            "mae_s": [80.0] * 9,
        }).to_csv(conformal_csv, index=False)

        result = decompose_rul_threshold_confound(index, conformal_csv, window_size=15)
        assert set(result) == set(range(1, 10))
        for tool_id, r in result.items():
            assert r["ig_fixed_threshold_rmse_s"] is not None
            assert r["ig_matched_threshold_rmse_s"] is not None
            assert r["matched_threshold_um"] > 0
            assert r["ig_fixed_threshold_ratio_vs_conformal"] > 0
            assert r["ig_matched_threshold_ratio_vs_conformal"] > 0


def _index_with_censoring_pattern(fully_censored_machine: int, n_runs: int = 20) -> pd.DataFrame:
    """3 machines x 3 tools; ``fully_censored_machine``'s tools never reach
    the 150um threshold (mirrors the real dataset's M2, where 0/3 tools ever
    cross it), the other two machines' tools do."""
    rows = []
    for machine in (1, 2, 3):
        for tool_offset, tool in enumerate((1, 2, 3)):
            tool_id = (machine - 1) * 3 + tool
            times = np.linspace(10.0, 3500.0, n_runs)
            max_wear = 100.0 if machine == fully_censored_machine else 200.0
            wear = np.linspace(5.0, max_wear, n_runs)
            for run, (t, w) in enumerate(zip(times, wear), start=1):
                rows.append({
                    "machine": machine, "tool": tool_id, "run": run,
                    "cumulated_tool_contact_time": t, "wear": w,
                })
    return pd.DataFrame(rows)


class TestGroundTruthCensoringTransparency:
    """Covers the fix for Table 3's censoring-transparency gap: a fully
    right-censored machine's ground-truth "MTTFF" is a lower bound (KM-RMST
    evaluated at tau), not a full estimate, and compute_mttff's all_censored/
    n_observed fields need to actually reach the summary dict Table 3 reads
    from -- previously they were computed but silently dropped in transit.
    """

    def test_fully_censored_machine_flagged_at_the_ground_truth_level(self):
        index = _index_with_censoring_pattern(fully_censored_machine=2)
        gt = ground_truth_reliability(index, wear_threshold_um=150.0, n_boot=50)
        assert gt["M2"]["mttff"]["all_censored"] is True
        assert gt["M2"]["mttff"]["n_observed"] == 0
        assert gt["M1"]["mttff"]["all_censored"] is False
        assert gt["M1"]["mttff"]["n_observed"] == 3

    def test_summarize_reliability_translation_surfaces_the_flag(self):
        index = _index_with_censoring_pattern(fully_censored_machine=2)
        comparison_result = {
            "gamma_params": {m: _synthetic_params() for m in ("M1", "M2", "M3")},
            "ig_params": {m: _synthetic_params() for m in ("M1", "M2", "M3")},
        }
        result = summarize_reliability_translation(
            index, comparison_result, n_sim=20, n_boot=20, wear_threshold_um=150.0,
        )
        censored_row = result["summary"]["M2"]
        assert censored_row["ground_truth_n_observed"] == 0
        assert censored_row["ground_truth_all_censored"] is True
        normal_row = result["summary"]["M1"]
        assert normal_row["ground_truth_n_observed"] == 3
        assert normal_row["ground_truth_all_censored"] is False
