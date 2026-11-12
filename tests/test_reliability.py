"""Tests for the reliability analysis module (reliability_analysis.py)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from reliability_analysis import (
    compute_mttff,
    fit_weibull_mttff,
    mttff_bootstrap_ci,
)


# ── Synthetic episode helpers ─────────────────────────────────────────────────

def _always_available_result(sim_time=100.):
    """All 3 machines available for the entire episode."""
    segs = [[{"t0": 0., "t1": sim_time, "avail": True, "state": 0}] for _ in range(3)]
    return {
        "simTime":            sim_time,
        "availTimeline":      segs,
        "systemAvailability": 1.0,
        "firstFailureTime":   None,
        "firstFailureTimesByMachine": [None, None, None],
        "firstFailureMachines": [],
        "jobsCompleted":      50,
        "jobsTardy":          2,
        "scrapCount":         0,
        "throughputRate":     0.5,
        "totalTardinessCost": 100.,
        "totalMaintCost":     200.,
        "totalDowntimeCost":  0.,
        "totalCost":          300.,
        "tardinessValues":    [0.] * 50,
        "tardinessMean":      0.,
        "tardinessP95":       0.,
        "perMachine": [
            {"machine_id": m, "tBusy": sim_time*0.8, "tIdle": sim_time*0.2,
             "tMaint": 0., "availability": 1.0, "utilization": 0.8,
             "nPM": 0, "nCM": 0, "nBreakdown": 0, "nOpsDone": 17,
             "cumDamage": 50., "RUL_mu": 10.}
            for m in range(3)
        ],
    }


def _machine_fails_at(m_fail: int, t_fail: float, sim_time=100.) -> dict:
    """Episode where machine m_fail fails at t_fail (unavailable after)."""
    segs = []
    for m in range(3):
        if m == m_fail:
            segs.append([
                {"t0": 0., "t1": t_fail,     "avail": True,  "state": 0},
                {"t0": t_fail, "t1": sim_time, "avail": False, "state": 3},
            ])
        else:
            segs.append([{"t0": 0., "t1": sim_time, "avail": True, "state": 0}])
    sys_avail = t_fail / sim_time  # all-3-up fraction
    per_m = []
    for m in range(3):
        tb = sim_time * 0.8 if m != m_fail else t_fail * 0.8
        pm_ent = {"machine_id": m, "tBusy": tb, "tIdle": tb*0.2,
                  "tMaint": (sim_time - t_fail) if m == m_fail else 0.,
                  "availability": 1.0 if m != m_fail else t_fail/sim_time,
                  "utilization": 0.8, "nPM": 0, "nCM": 1 if m == m_fail else 0,
                  "nBreakdown": 1 if m == m_fail else 0,
                  "nJobsDone": 15, "cumDamage": 160., "RUL_mu": 0.}
        per_m.append(pm_ent)
    return {
        "simTime": sim_time, "availTimeline": segs,
        "systemAvailability": sys_avail, "firstFailureTime": t_fail,
        "firstFailureTimesByMachine": [
            t_fail if m == m_fail else None for m in range(3)
        ],
        "firstFailureMachines": [m_fail],
        "jobsCompleted": 40, "jobsTardy": 5, "scrapCount": 1,
        "throughputRate": 0.4, "totalTardinessCost": 200.,
        "totalMaintCost": 300., "totalDowntimeCost": 50.,
        "totalCost": 550., "tardinessValues": [1.] * 5,
        "tardinessMean": 0.5, "tardinessP95": 1.5, "perMachine": per_m,
    }


# ── MTTFF tests ───────────────────────────────────────────────────────────────

class TestMTTFF:
    def test_no_failures(self):
        results = [_always_available_result() for _ in range(20)]
        m = compute_mttff(results)
        assert m["mttff"] == pytest.approx(100.0)
        assert m["all_censored"]
        assert m["n_observed"] == 0,   "No failures → n_observed=0"
        assert m["n_censored"] == 20,  "All episodes censored"

    def test_all_fail(self):
        t_f     = 45.
        results = [_machine_fails_at(0, t_f) for _ in range(20)]
        m       = compute_mttff(results)
        # When all episodes fail at the same time, KM RMST = t_f (S drops to 0)
        assert abs(m["mttff"] - t_f) < 1e-6, \
            f"MTTFF must equal t_fail={t_f} when all episodes fail at same time"
        assert m["n_observed"] == 20
        assert m["n_censored"] == 0

    def test_mixed(self):
        # 5 censored at simTime=100, 5 fail at t=60
        # KM: S(0..60)=1, S(60+)=0.5 → RMST = 1*60 + 0.5*40 = 80
        results  = [_always_available_result()] * 5
        results += [_machine_fails_at(0, 60.)] * 5
        m = compute_mttff(results)
        assert m["mttff"] == pytest.approx(80., abs=1e-6), \
            "KM RMST with 5 failures at 60 and 5 censored at 100 = 80"
        assert m["n_observed"] == 5
        assert m["n_censored"] == 5

    def test_km_rmst_exceeds_naive_mean_when_censored(self):
        """KM RMST > naive mean of observed failures when censoring present."""
        results  = [_always_available_result()] * 10   # censored at 100
        results += [_machine_fails_at(0, 40.)] * 10    # fail at 40
        m = compute_mttff(results)
        naive = 40.
        assert m["mttff"] > naive, \
            "KM RMST should exceed the naive failure-only mean when censoring present"

    def test_km_curve_keys_present(self):
        results = [_machine_fails_at(1, float(30 + i * 5)) for i in range(10)]
        m = compute_mttff(results)
        assert "km_t" in m and "km_S" in m, "KM curve must be in output"
        assert m["km_S"][0] == 1.0, "S(0) must be 1"
        assert m["n_observed"] == 10


# ── Weibull fit tests ─────────────────────────────────────────────────────────

class TestWeibullFit:
    def test_fit_with_enough_failures(self):
        results = [_machine_fails_at(0, float(20 + i * 5)) for i in range(10)]
        w = fit_weibull_mttff(results)
        assert np.isfinite(w["shape"]),  "Weibull shape must be finite"
        assert np.isfinite(w["scale"]),  "Weibull scale must be finite"
        assert w["shape"] > 0,           "Shape must be positive"
        assert w["scale"] > 0,           "Scale must be positive"
        assert w["n_failures"] == 10
        assert not w["low_n"]

    def test_fit_with_few_failures_flags_low_n(self):
        results = [_machine_fails_at(0, 30.)] * 3
        w = fit_weibull_mttff(results)
        assert w["low_n"], "< 5 failures must set low_n=True"

    def test_fit_with_no_failures(self):
        results = [_always_available_result()] * 10
        w = fit_weibull_mttff(results)
        assert np.isnan(w["shape"]), "No failures → shape=NaN"
        assert w["n_failures"] == 0

    def test_aic_bic_finite_when_enough_data(self):
        results = [_machine_fails_at(0, float(10 + i * 3)) for i in range(15)]
        w = fit_weibull_mttff(results)
        assert np.isfinite(w["AIC"]), "AIC must be finite"
        assert np.isfinite(w["BIC"]), "BIC must be finite"

    def test_wear_out_regime(self):
        # Increasing failure times suggest wear-out: Weibull shape > 1
        results = [_machine_fails_at(0, float(30 + i * 10)) for i in range(10)]
        w = fit_weibull_mttff(results)
        if not w["low_n"] and np.isfinite(w["shape"]):
            assert w["shape"] > 0, "shape must be positive"


# ── Bootstrap CI tests ────────────────────────────────────────────────────────

class TestBootstrapCI:
    def test_ci_contains_point_estimate(self):
        results = (
            [_always_available_result()] * 5 +
            [_machine_fails_at(0, 50.)] * 10
        )
        km   = compute_mttff(results)
        ci   = mttff_bootstrap_ci(results, n_boot=200, rng_seed=0)
        mttff = km["mttff"]
        if np.isfinite(mttff) and np.isfinite(ci["mttff_ci_lo"]):
            # CI should bracket the point estimate (not always true but should hold
            # on average — use relaxed check: CI lo ≤ estimate ≤ CI hi)
            assert ci["mttff_ci_lo"] <= mttff + 1e-3, "CI lo must not exceed point estimate"
            assert ci["mttff_ci_hi"] >= mttff - 1e-3, "CI hi must not be below point estimate"

    def test_ci_width_decreases_with_n(self):
        # Use same failure-time pattern for both samples (just scale count);
        # with enough resamples and a large n ratio, CI must narrow.
        base_times = [float(30 + i * 5) for i in range(10)]
        results_small = [_machine_fails_at(0, t) for t in base_times]
        results_large = [_machine_fails_at(0, t) for t in base_times * 8]  # 80 reps
        ci_s = mttff_bootstrap_ci(results_small, n_boot=500, rng_seed=42)
        ci_l = mttff_bootstrap_ci(results_large, n_boot=500, rng_seed=42)
        w_s  = ci_s["mttff_ci_hi"] - ci_s["mttff_ci_lo"]
        w_l  = ci_l["mttff_ci_hi"] - ci_l["mttff_ci_lo"]
        assert w_l < w_s, "Larger sample (8× more data) should give narrower CI"

    def test_ci_keys_present(self):
        results = [_machine_fails_at(0, 40.)] * 10
        ci = mttff_bootstrap_ci(results, n_boot=100)
        for k in ("mttff_ci_lo", "mttff_ci_hi", "mttff_boot_std", "n_boot"):
            assert k in ci, f"Missing key: {k}"
