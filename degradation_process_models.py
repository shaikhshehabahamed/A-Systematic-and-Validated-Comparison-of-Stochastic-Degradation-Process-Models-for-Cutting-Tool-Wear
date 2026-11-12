"""Degradation process-family comparison for milling tool wear.

Companion to ``gamma_process.py``, which fits the production non-homogeneous
Gamma process originally used by the separate, now-abandoned MARL simulation
pipeline (deleted in an earlier cleanup) and reused read-only by this study.
This module fits the same rounded-observation wear windows with two
alternative Levy-process families — Inverse Gaussian and Wiener-with-drift —
using the identical mean/variance parametrisation so that AIC/BIC comparisons
isolate distributional shape rather than moment-matching artefacts:

    mean increment  mu = S / beta
    variance        Var = S / beta^2

where ``S = eta_j * [(t/t_ref)^p_j - (s/t_ref)^p_j]`` is the same per-tool,
age-transformed quantity used as the Gamma shape parameter in
``gamma_process.py``. Under this convention:

  - Gamma(shape=S, rate=beta)            has mean S/beta, Var S/beta^2
  - InverseGaussian(mean=S/beta, shape=S^2/beta) has mean S/beta, Var S/beta^2
  - Normal(mean=S/beta, sd=sqrt(S)/beta) has mean S/beta, Var S/beta^2

so all three families are fit as literal alternatives for the same rounded
wear-increment likelihood, with the same non-homogeneous-vs-stationary AIC
model-selection structure as ``_fit_rounded_nh_gamma``.

Wear is physically monotone non-decreasing. Gamma and Inverse Gaussian are
non-negative Levy subordinators and respect this by construction; Wiener does
not, so this module also reports the probability mass the fitted Wiener model
places on physically impossible negative increments, as a structural
(not just statistical) adequacy check.

Related work (literature scan via web search, 2026-07; not a substitute for a
full database search -- re-verify closer to submission)
----------------------------------------------------------------------------
Gamma has been the default deterioration process in reliability engineering
since Abdel-Hameed, "A gamma wear process," IEEE Trans. Reliability 24(2),
1975, pp.152-153 (bibliographic details for this and the next two citations
verified directly against Crossref's authoritative metadata), surveyed in
van Noortwijk, "A survey of the application of
gamma processes in maintenance," Reliab. Eng. Syst. Saf. 94(1), 2009,
pp.2-21 (doi:10.1016/j.ress.2007.03.019). Inverse Gaussian was established as
a comparably flexible alternative by Wang & Xu, "An inverse Gaussian process
model for degradation data," Technometrics 52(2), 2010, pp.188-197
(doi:10.1198/TECH.2009.08197). Recent cutting-tool-specific work has
independently gravitated toward IG or Wiener processes for tool wear --
e.g. an IG-process RUL model for cutting tools (Appl. Sci. 11(11), 2021,
doi:10.3390/app11115011), a state-dependent transformed IG process for
cutting-tool reliability under varying operating conditions (Int. J. Mach.
Tools Manuf., 2024, doi:10.1016/j.ijmachtools.2024.104144 -- ScienceDirect
S0266892024001206), an improved-IG hybrid RUL model (2024, ScienceDirect
S1526612524006091), and a multi-stage Wiener-process tool-wear model
(Int. J. Adv. Manuf. Technol., 2023, doi:10.1007/s00170-023-12648-8) -- but
none located perform an explicit head-to-head Gamma-vs-IG-vs-Wiener AIC/BIC
family-selection comparison; each commits to one family a priori. The
dataset used here (Denkena, Klemme & Stiehl, Data in Brief 50:109574, 2023,
doi:10.1016/j.dib.2023.109574) has, in all located subsequent work, been
used for ML/feature-based tool-wear prediction (e.g. arXiv:2412.19950), not
classical stochastic degradation-process/reliability-engineering analysis.

The GENERAL (not tool-wear-specific) question of whether AIC-based
Gamma-vs-IG selection is itself trustworthy has an established prior
literature that this study builds on rather than originates: Esposito,
Mele, Castanier & Giorgio, "Misspecification analysis of gamma- and inverse
Gaussian-based perturbed degradation processes," Appl. Stoch. Models Bus.
Ind. 40(3), 2024, pp.640-667 (doi:10.1002/asmb.2824) -- bibliographic details
and abstract verified directly against Crossref's authoritative metadata,
not a search-engine paraphrase. Per the verified abstract, they run a large
Monte Carlo study specifically on PERTURBED (measurement-error-affected)
gamma and inverse Gaussian processes, under both Gaussian and non-Gaussian
error terms, evaluating misspecification risk as "percentage of times the
Akaike information criterion leads to select the wrong model" and its
severity via impact on estimated mean remaining useful life -- i.e. the same
core AIC-wrong-selection-rate logic ``model_recovery_study()`` (below) uses,
independently confirming that is an appropriate way to interrogate this
question, not a technique invented for this study. (A 2017 book-chapter
precursor of the same lineage and a separate Fisher-information-based formal
test purpose-built for Gamma-vs-IG discrimination are also referenced in
this space, but have NOT been directly verified against their primary
sources at the time of writing -- treat those two specific attributions as
provisional pending confirmation, unlike the Esposito et al. citation above.)
This study's own recovery study extends the Esposito et al. logic in two
ways relevant to this paper's claims: (a) to three families rather than two
(Wiener included), and (b) calibrated to REAL fitted parameters from this
actual industrial multi-machine dataset rather than a generic simulation
design, so the recovery-rate evidence speaks directly to whether *this
study's own* family-selection conclusion is trustworthy, not just the
general procedure in the abstract. The Vuong (1989) non-nested test used
elsewhere in this module is the general off-the-shelf choice covering all
three pairwise family comparisons (including Wiener, which a Gamma/IG-
specific Fisher-information test would not, per the unverified reference
above); this is a deliberate choice of a standard, broadly-applicable test
over a bespoke Gamma/IG one.

A recent broad review, Clark, Min, Li, Warr, DeHart, King, Lu & Hong, "What
Quality Engineers Need to Know about Degradation Models" (arXiv:2507.14666,
submitted 2025, revised 2026; journal status unconfirmed), independently
confirms Wiener (citing Whitmore, 1995), Gamma (citing Lawless & Crowder,
2004), and IG (citing Wang & Xu, 2010) as "the three most commonly used
processes" for degradation modeling -- verified directly against the paper's
full text, not a search-summary paraphrase. Note: an earlier draft of this
docstring additionally attributed to this paper a claim that it explicitly
states Wiener is suited to non-monotonic data while Gamma/IG require
monotonicity; on re-checking the primary source directly, the paper does
NOT make that distinction (it presents all three as modeling monotonically
non-decreasing degradation paths) -- that claim has been removed rather than
left uncorrected. This study's own finding that Wiener is the wrong model
class for monotone wear here rests entirely on its own PIT-KS calibration
failures and the direct negative-increment-mass calculation (both computed
from this dataset), not on external corroboration.

To our knowledge this is the first study to (a) perform a formal
Gamma-vs-IG-vs-Wiener family-selection comparison via AIC/BIC/PIT-KS on real
multi-machine tool-wear trajectories, documenting a concrete failure mode
(spurious wear-acceleration conclusions from defaulting to Gamma without
testing alternatives), extending established (2-family, simulation-only)
misspecification-analysis practice to a 3-family real-data application with
a formal non-nested hypothesis test and a dataset-calibrated model-recovery
check; and (b) apply this specific dataset to classical
reliability-engineering methodology (MTTFF, age-replacement maintenance)
rather than ML-based prediction.

Framing note for whoever drafts the manuscript: Tables 3-4 show the
practical MTTFF/maintenance-cost difference between the two *valid*
families (Gamma, IG) is small -- the large practical cost is specifically
in excluding the *invalid* one (Wiener), not in choosing between Gamma and
IG. The contribution reads more defensibly led with as a methodological one
(a replicable AIC/BIC/PIT-KS/Vuong/model-recovery protocol for interrogating
degradation-family choice, validated here with a positive control) than
with a practical-stakes hook the study's own results partly undercut.

Numerical verification (optimizer convergence, 2026-07-04): every reported
fit uses Nelder-Mead with 3 informed starting points. Checked whether this
finds the global optimum by refitting the real M1/M2/M3 Gamma and Inverse
Gaussian models (the two families driving the core finding) with 30
additional, widely-scattered random restarts per machine x family, using
the SAME parameterization actually selected for each cell (fixed power=1
where "linear" was selected, free power where "power_law" was selected --
comparing against the wrong parameterization would spuriously show
"improvement" for every stationary-selected cell, since a stationary fit is
a strict, nested sub-case of the free-power one with fewer degrees of
freedom; this mismatch was caught and corrected mid-verification, not
missed). Result: 0 of 6 (machine, family) cells found a lower NLL than the
already-reported optimum, across 180 additional restarts total -- no
evidence any reported AIC/BIC value here is optimizer noise rather than a
genuine likelihood difference.

Formal-test caveat for manuscript drafting: M1 is the weakest-evidence
machine for the Gamma-vs-IG contrast specifically, on two independent,
out-of-sample-flavored checks, despite a raw Delta-AIC (22.46) comparable to
M2's (24.61) and M3's (21.98, Table 1). Vuong's test is directionally
suggestive but not formally significant there (z=-1.82, p=0.069, Table S10),
and its leave-one-tool-out CV-likelihood margin is the smallest of the three
machines (+4.1 vs. Gamma, against +11.3/+11.7 on M2/M3, Table S1). When
prose eventually compresses Table 1 to something like "IG beats Gamma by
Delta-AIC ~22-25 on every machine, strong evidence," that sentence should
not imply uniform strength of evidence across machines -- M1 specifically
needs this caveat attached.

Analysis timeline / methodological transparency: this study's robustness
battery was built incrementally, each addition made in direct response to a
specific concern raised about an earlier round rather than planned
end-to-end in advance. Recorded here plainly, since a Q1 reviewer is more
likely to react well to an explicit account of this than to notice the same
shape independently: (1) original design -- AIC/BIC/PIT-KS comparison at
window_size=15, the production Gamma pipeline's existing default; (2) after
feedback that an initial pass was too quick given the available time budget
-- leave-one-tool-out CV likelihood, window-size sweep, EOL-threshold
sensitivity; (3) after being asked what else could strengthen the study --
bootstrap CI on the AIC gap, MC seed-sensitivity check, zero-increment-
retention ablation, pooled-vs-per-machine baseline, parametric-vs-ML RUL
comparison, maintenance-cost translation; (4) after a first structured
review pass -- the interpolation-window confound (resolved via a large-
window sweep at/beyond the measurement interval), cost-ratio sensitivity,
the per-tool bootstrap significance test (formalizing point estimates that
had previously been reported without a significance test), a broader
literature search, and the RUL threshold-mismatch decomposition; (5) after
a second review pass -- the model-recovery positive control (built
specifically to test whether the procedure could be structurally biased
toward IG), Vuong's non-nested test, and per-tool fit diagnostics (the M3
calibration investigation); (6) after a third review pass -- Wilson CIs on
recovery rates, the Wiener self-recovery truncation-confound quantification,
Benjamini-Hochberg FDR correction, golden-value tests, and citation
re-verification; (7) after a fourth review pass -- this AIC-vs-BIC
age-model table (Table S13; a check that previously existed only as an
informal note and, on being tabulated properly, turned out to change a
conclusion), the ground-truth censoring-transparency fix in Table 3, the
pooled-baseline BIC-direction correction in Table S5, and the optimizer-
convergence verification above. No formal conclusion in the study has
changed sign across these rounds; several have gained caveats or lost an
overstatement. Benjamini-Hochberg correction (Table S12) confirms none of
the formal hypothesis-test conclusions specifically depend on uncorrected
multiplicity from this many tests.

Output: artifacts/degradation_models/degradation_comparison.json
"""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import gamma as gamma_dist
from scipy.stats import invgauss, kstest, norm
from scipy.stats import t as t_dist

from gamma_process import (
    ROUNDING_OFFSETS_UM,
    ROUNDING_WEIGHTS,
    TIME_REFERENCE_S,
    WEAR_RESOLUTION_UM,
    _aggregated_bin_probability,
    _fit_rounded_nh_gamma,
    _rounded_gamma_bin_probability,
    build_wear_windows,
)
import degradation_families as families

MODEL_SCHEMA_VERSION = 2


# ── Inverse Gaussian ──────────────────────────────────────────────────────────

def _rounded_ig_bin_probability(
    observed: np.ndarray,
    shape_s: np.ndarray,
    beta: float,
    resolution_um: float,
) -> np.ndarray:
    """Probability of each aggregated, two-endpoint-rounded observation under
    IG(S/beta, S^2/beta) (see ``gamma_process._aggregated_bin_probability``
    for the rounding-kernel derivation)."""
    # scipy.stats.invgauss(mu, scale) represents general IG(mu*scale, scale):
    # setting scale = lambda = S^2/beta and mu = 1/S gives mean S/beta.
    mu_scipy = 1.0 / shape_s
    scale = (shape_s ** 2) / beta

    def _cdf(x):
        return invgauss.cdf(np.maximum(1e-9, x), mu=mu_scipy, loc=0.0, scale=scale)

    prob = _aggregated_bin_probability(observed, _cdf, resolution_um)
    return np.maximum(prob, np.finfo(float).tiny)


def _fit_rounded_nh_ig(
    dv: np.ndarray,
    t_start: np.ndarray,
    t_end: np.ndarray,
    tool_ids: np.ndarray,
    *,
    time_reference_s: float = TIME_REFERENCE_S,
    resolution_um: float = WEAR_RESOLUTION_UM,
) -> dict:
    """Fit a tool-heterogeneous non-homogeneous Inverse Gaussian process.

    Mirrors ``gamma_process._fit_rounded_nh_gamma`` exactly (same per-tool
    eta/power, shared beta, log-space Nelder-Mead with 3 restarts,
    non-homogeneous-vs-stationary AIC selection); only the bin-probability
    function differs.
    """
    dv = np.asarray(dv, float)
    t_start = np.asarray(t_start, float)
    t_end = np.asarray(t_end, float)
    tool_ids = np.asarray(tool_ids, int)
    if not (
        dv.ndim == t_start.ndim == t_end.ndim == tool_ids.ndim == 1
        and len(dv) == len(t_start) == len(t_end) == len(tool_ids)
        and len(dv) > 0
    ):
        raise ValueError("Wear-window arrays must be equal-length non-empty vectors.")
    if np.any(dv < 0) or np.any(t_start < 0) or np.any(t_end <= t_start):
        raise ValueError("Invalid wear windows.")

    tools = np.unique(tool_ids)
    tool_col = {int(tool): i for i, tool in enumerate(tools)}
    component_idx = np.array([tool_col[int(tool)] for tool in tool_ids], int)
    n_tools = len(tools)
    t_ref = float(time_reference_s)

    beta0 = 0.5
    power0 = np.full(n_tools, 0.8)
    eta0 = np.empty(n_tools, float)
    for i, tool in enumerate(tools):
        mask = tool_ids == tool
        total_wear = max(float(np.sum(dv[mask])), 1.0)
        terminal_scaled = max(float(np.max(t_end[mask])) / t_ref, 1e-6)
        eta0[i] = max(beta0 * total_wear / terminal_scaled ** power0[i], 1e-4)

    def nll(log_params: np.ndarray, fixed_power: bool = False) -> float:
        eta = np.exp(log_params[:n_tools])
        if fixed_power:
            power = np.ones(n_tools, float)
            beta = float(np.exp(log_params[n_tools]))
        else:
            power = np.exp(log_params[n_tools:2 * n_tools])
            beta = float(np.exp(log_params[-1]))
        if np.any(power < 0.05) or np.any(power > 3.0) or beta < 1e-4 or beta > 100.0:
            return 1e100
        shape_s = eta[component_idx] * (
            np.power(t_end / t_ref, power[component_idx])
            - np.power(t_start / t_ref, power[component_idx])
        )
        if np.any(shape_s <= 0) or not np.all(np.isfinite(shape_s)):
            return 1e100
        prob = _rounded_ig_bin_probability(dv, shape_s, beta, resolution_um)
        return -float(np.sum(np.log(prob)))

    starts = [
        np.log(np.r_[eta0, power0, beta0]),
        np.log(np.r_[eta0 * 0.7, np.full(n_tools, 0.6), 0.4]),
        np.log(np.r_[eta0 * 1.3, np.full(n_tools, 1.0), 0.7]),
    ]
    best = None
    for x0 in starts:
        res = minimize(
            nll, x0, method="Nelder-Mead",
            options={"maxiter": 100000, "xatol": 1e-9, "fatol": 1e-9},
        )
        if best is None or res.fun < best.fun:
            best = res
    if best is None or not np.isfinite(best.fun):
        raise RuntimeError("Non-homogeneous rounded Inverse Gaussian MLE failed.")

    eta = np.exp(best.x[:n_tools])
    power = np.exp(best.x[n_tools:2 * n_tools])
    beta = float(np.exp(best.x[-1]))

    stationary_start = np.log(np.r_[eta, beta])
    stationary = minimize(
        lambda z: nll(z, fixed_power=True), stationary_start, method="Nelder-Mead",
        options={"maxiter": 50000, "xatol": 1e-9, "fatol": 1e-9},
    )

    n = len(dv)
    k_nh = 2 * n_tools + 1
    k_stationary = n_tools + 1
    log_lik = -float(best.fun)
    stationary_log_lik = -float(stationary.fun)
    nh_components = [
        {"tool_id": int(tool), "eta": float(eta[i]), "power": float(power[i]),
         "beta": beta, "time_reference_s": t_ref}
        for i, tool in enumerate(tools)
    ]
    stationary_eta = np.exp(stationary.x[:n_tools])
    stationary_beta = float(np.exp(stationary.x[n_tools]))
    stationary_components = [
        {"tool_id": int(tool), "eta": float(stationary_eta[i]), "power": 1.0,
         "beta": stationary_beta, "time_reference_s": t_ref}
        for i, tool in enumerate(tools)
    ]
    nh_aic = float(2 * k_nh - 2 * log_lik)
    stationary_aic = float(2 * k_stationary - 2 * stationary_log_lik)
    use_nonhomogeneous = bool(stationary_aic - nh_aic > 2.0)
    selected_components = nh_components if use_nonhomogeneous else stationary_components
    selected_beta = beta if use_nonhomogeneous else stationary_beta
    selected_log_lik = log_lik if use_nonhomogeneous else stationary_log_lik
    selected_k = k_nh if use_nonhomogeneous else k_stationary
    return {
        "model": "nonhomogeneous_inverse_gaussian_rounded",
        "schema_version": MODEL_SCHEMA_VERSION,
        "selected_age_model": "power_law" if use_nonhomogeneous else "linear",
        "beta": selected_beta,
        "eta": float(np.median([c["eta"] for c in selected_components])),
        "power": float(np.median([c["power"] for c in selected_components])),
        "time_reference_s": t_ref,
        "wear_resolution_um": float(resolution_um),
        "components": selected_components,
        "n_windows": int(n),
        "n_zero_windows": int(np.sum(dv == 0)),
        "log_lik": selected_log_lik,
        "aic": float(2 * selected_k - 2 * selected_log_lik),
        "bic": float(selected_k * np.log(n) - 2 * selected_log_lik),
        "nonhomogeneous_aic": nh_aic,
        "nonhomogeneous_bic": float(k_nh * np.log(n) - 2 * log_lik),
        "stationary_aic": stationary_aic,
        "stationary_bic": float(k_stationary * np.log(n) - 2 * stationary_log_lik),
        # Retained even when the machine-level AIC prefers the stationary
        # model, so a per-tool diagnostic can surface individual tools that
        # deviate from the fleet even when the joint 3-tool decision doesn't.
        "nonhomogeneous_components": nh_components,
    }


# ── Wiener-with-drift ──────────────────────────────────────────────────────────

def _rounded_wiener_bin_probability(
    observed: np.ndarray,
    shape_s: np.ndarray,
    beta: float,
    resolution_um: float,
) -> np.ndarray:
    """Probability of each aggregated, two-endpoint-rounded observation under
    N(S/beta, S/beta^2) (see ``gamma_process._aggregated_bin_probability`` for
    the rounding-kernel derivation).

    Fit naively (no non-negativity truncation), matching how a Wiener
    degradation model is typically applied in practice — the resulting
    negative-increment probability mass is reported separately as a
    structural-adequacy diagnostic rather than folded into the likelihood.
    """
    mu = shape_s / beta
    sigma = np.sqrt(shape_s) / beta

    def _cdf(x):
        return norm.cdf(x, loc=mu, scale=sigma)

    prob = _aggregated_bin_probability(observed, _cdf, resolution_um)
    return np.maximum(prob, np.finfo(float).tiny)


def _fit_rounded_nh_wiener(
    dv: np.ndarray,
    t_start: np.ndarray,
    t_end: np.ndarray,
    tool_ids: np.ndarray,
    *,
    time_reference_s: float = TIME_REFERENCE_S,
    resolution_um: float = WEAR_RESOLUTION_UM,
) -> dict:
    """Fit a tool-heterogeneous non-homogeneous Wiener-with-drift process.

    Same per-tool eta/power, shared beta, log-space Nelder-Mead, 3 restarts,
    non-homogeneous-vs-stationary AIC selection as the Gamma/IG fitters.
    """
    dv = np.asarray(dv, float)
    t_start = np.asarray(t_start, float)
    t_end = np.asarray(t_end, float)
    tool_ids = np.asarray(tool_ids, int)
    if not (
        dv.ndim == t_start.ndim == t_end.ndim == tool_ids.ndim == 1
        and len(dv) == len(t_start) == len(t_end) == len(tool_ids)
        and len(dv) > 0
    ):
        raise ValueError("Wear-window arrays must be equal-length non-empty vectors.")
    if np.any(dv < 0) or np.any(t_start < 0) or np.any(t_end <= t_start):
        raise ValueError("Invalid wear windows.")

    tools = np.unique(tool_ids)
    tool_col = {int(tool): i for i, tool in enumerate(tools)}
    component_idx = np.array([tool_col[int(tool)] for tool in tool_ids], int)
    n_tools = len(tools)
    t_ref = float(time_reference_s)

    beta0 = 0.5
    power0 = np.full(n_tools, 0.8)
    eta0 = np.empty(n_tools, float)
    for i, tool in enumerate(tools):
        mask = tool_ids == tool
        total_wear = max(float(np.sum(dv[mask])), 1.0)
        terminal_scaled = max(float(np.max(t_end[mask])) / t_ref, 1e-6)
        eta0[i] = max(beta0 * total_wear / terminal_scaled ** power0[i], 1e-4)

    def nll(log_params: np.ndarray, fixed_power: bool = False) -> float:
        eta = np.exp(log_params[:n_tools])
        if fixed_power:
            power = np.ones(n_tools, float)
            beta = float(np.exp(log_params[n_tools]))
        else:
            power = np.exp(log_params[n_tools:2 * n_tools])
            beta = float(np.exp(log_params[-1]))
        if np.any(power < 0.05) or np.any(power > 3.0) or beta < 1e-4 or beta > 100.0:
            return 1e100
        shape_s = eta[component_idx] * (
            np.power(t_end / t_ref, power[component_idx])
            - np.power(t_start / t_ref, power[component_idx])
        )
        if np.any(shape_s <= 0) or not np.all(np.isfinite(shape_s)):
            return 1e100
        prob = _rounded_wiener_bin_probability(dv, shape_s, beta, resolution_um)
        return -float(np.sum(np.log(prob)))

    starts = [
        np.log(np.r_[eta0, power0, beta0]),
        np.log(np.r_[eta0 * 0.7, np.full(n_tools, 0.6), 0.4]),
        np.log(np.r_[eta0 * 1.3, np.full(n_tools, 1.0), 0.7]),
    ]
    best = None
    for x0 in starts:
        res = minimize(
            nll, x0, method="Nelder-Mead",
            options={"maxiter": 100000, "xatol": 1e-9, "fatol": 1e-9},
        )
        if best is None or res.fun < best.fun:
            best = res
    if best is None or not np.isfinite(best.fun):
        raise RuntimeError("Non-homogeneous rounded Wiener MLE failed.")

    eta = np.exp(best.x[:n_tools])
    power = np.exp(best.x[n_tools:2 * n_tools])
    beta = float(np.exp(best.x[-1]))

    stationary_start = np.log(np.r_[eta, beta])
    stationary = minimize(
        lambda z: nll(z, fixed_power=True), stationary_start, method="Nelder-Mead",
        options={"maxiter": 50000, "xatol": 1e-9, "fatol": 1e-9},
    )

    n = len(dv)
    k_nh = 2 * n_tools + 1
    k_stationary = n_tools + 1
    log_lik = -float(best.fun)
    stationary_log_lik = -float(stationary.fun)
    nh_components = [
        {"tool_id": int(tool), "eta": float(eta[i]), "power": float(power[i]),
         "beta": beta, "time_reference_s": t_ref}
        for i, tool in enumerate(tools)
    ]
    stationary_eta = np.exp(stationary.x[:n_tools])
    stationary_beta = float(np.exp(stationary.x[n_tools]))
    stationary_components = [
        {"tool_id": int(tool), "eta": float(stationary_eta[i]), "power": 1.0,
         "beta": stationary_beta, "time_reference_s": t_ref}
        for i, tool in enumerate(tools)
    ]
    nh_aic = float(2 * k_nh - 2 * log_lik)
    stationary_aic = float(2 * k_stationary - 2 * stationary_log_lik)
    use_nonhomogeneous = bool(stationary_aic - nh_aic > 2.0)
    selected_components = nh_components if use_nonhomogeneous else stationary_components
    selected_beta = beta if use_nonhomogeneous else stationary_beta
    selected_log_lik = log_lik if use_nonhomogeneous else stationary_log_lik
    selected_k = k_nh if use_nonhomogeneous else k_stationary

    # Structural-inadequacy diagnostic: P(negative increment) implied by the
    # fitted model, evaluated at each observed window's own (mu, sigma).
    sel_eta = np.array([c["eta"] for c in selected_components])
    sel_power = np.array([c["power"] for c in selected_components])
    shape_s_all = sel_eta[component_idx] * (
        np.power(t_end / t_ref, sel_power[component_idx])
        - np.power(t_start / t_ref, sel_power[component_idx])
    )
    mu_all = shape_s_all / selected_beta
    sigma_all = np.sqrt(shape_s_all) / selected_beta
    p_negative = norm.cdf(0.0, loc=mu_all, scale=sigma_all)

    return {
        "model": "nonhomogeneous_wiener_rounded",
        "schema_version": MODEL_SCHEMA_VERSION,
        "selected_age_model": "power_law" if use_nonhomogeneous else "linear",
        "beta": selected_beta,
        "eta": float(np.median(sel_eta)),
        "power": float(np.median(sel_power)),
        "time_reference_s": t_ref,
        "wear_resolution_um": float(resolution_um),
        "components": selected_components,
        "n_windows": int(n),
        "n_zero_windows": int(np.sum(dv == 0)),
        "log_lik": selected_log_lik,
        "aic": float(2 * selected_k - 2 * selected_log_lik),
        "bic": float(selected_k * np.log(n) - 2 * selected_log_lik),
        "nonhomogeneous_aic": nh_aic,
        "nonhomogeneous_bic": float(k_nh * np.log(n) - 2 * log_lik),
        "stationary_aic": stationary_aic,
        "stationary_bic": float(k_stationary * np.log(n) - 2 * stationary_log_lik),
        "mean_p_negative_increment": float(np.mean(p_negative)),
        "max_p_negative_increment": float(np.max(p_negative)),
        "nonhomogeneous_components": nh_components,
    }


# ── Fit-all-machines wrappers ──────────────────────────────────────────────────

def fit_all_groups_family(index: pd.DataFrame, family_key: str, window_size: int = 15, **fit_kwargs) -> dict[str, dict]:
    """Fit validated non-homogeneous models for every machine, for any
    registered family -- every family (all 10) is fit by calling this
    directly with its key. Extra ``**fit_kwargs`` (e.g. multi-stage
    Wiener's ``n_grid``) are forwarded to the family's fit_fn.
    """
    spec = families.FAMILY_REGISTRY[family_key]
    print(f"Fitting rounded non-homogeneous {spec['label']} process per machine ...")
    windows = build_wear_windows(index, window_size=window_size)
    results = {}
    for machine_key, data in sorted(windows.items()):
        n_raw = len(data["dv"])
        if n_raw < 30:
            raise ValueError(
                f"Insufficient wear windows for {machine_key} ({n_raw} < 30)."
            )
        params = spec["fit_fn"](
            data["dv"], data["t_start"], data["t_end"], data["tool"],
            v_start=data["v_start"], **fit_kwargs,
        )
        params["machine_id"] = int(machine_key[1:])
        params["window_size"] = int(window_size)
        results[machine_key] = params
        neg_p = params.get("mean_p_negative_increment")
        neg_p_note = f"  P(negative increment)={neg_p:.3f}" if neg_p is not None else ""
        print(
            f"  {machine_key}: beta={params['beta']:.6g}  "
            f"age model={params.get('selected_age_model')}  "
            f"median power={params['power']:.3f}  "
            f"AIC={params['aic']:.2f}{neg_p_note}"
        )
    return results


# ── Validation (mirrors gamma_process.validate_gamma_fit) ─────────────────────

def _rounding_kernel_cdf_bounds(data: dict, gp: dict, spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-window (cdf_lo, bin_prob) under the two-endpoint aggregated
    rounding kernel (see ``gamma_process._aggregated_bin_probability``):
    the observed integer window value is round(X + u), u~Uniform(-.5,.5),
    marginalized over ``ROUNDING_OFFSETS_UM``/``ROUNDING_WEIGHTS`` -- a PIT
    residual must be de-quantised against these SAME marginalised bin
    edges, not a flat +/-0.5um bin, or a downstream diagnostic would be
    validating the fit against a rounding model it was not actually fit
    under. Factored out of ``_pit_ks_generic`` so any OTHER PIT-residual-
    based diagnostic (e.g. ``residual_autocorrelation_check``, which needs
    the identical per-window PIT machinery but organized per-tool rather
    than pooled per-machine) reuses this exactly rather than duplicating
    it -- two independently-maintained copies of the rounding-kernel
    marginalization is exactly the kind of thing that silently drifts out
    of sync.
    """
    cdf_fn = spec["cdf_fn"]
    positive_support = bool(spec["is_subordinator"])
    component_by_tool = {int(c["tool_id"]): c for c in gp["components"]}
    extra = families.extra_shape_values(spec, gp)
    shapes = np.empty(len(data["dv"]), float)
    for i, tool in enumerate(data["tool"]):
        c = component_by_tool[int(tool)]
        shapes[i:i + 1] = spec["shape_fn"](
            c, data["t_start"][i:i + 1], data["t_end"][i:i + 1], data["v_start"][i:i + 1],
        )
    resolution = float(gp["wear_resolution_um"])
    half = 0.5 * resolution
    beta = float(gp["beta"])
    dv = data["dv"]
    n_nodes = len(ROUNDING_OFFSETS_UM)
    cdf_lo_nodes = np.empty((n_nodes, len(dv)), float)
    cdf_hi_nodes = np.empty((n_nodes, len(dv)), float)
    for j, u in enumerate(ROUNDING_OFFSETS_UM):
        shift = u * resolution
        lo_edge = dv - half - shift
        hi_edge = dv + half - shift
        if positive_support:
            lo_edge = np.maximum(1e-9, lo_edge)
            hi_edge = np.maximum(1e-9, hi_edge)
        cdf_lo_nodes[j] = cdf_fn(lo_edge, shapes, beta, *extra)
        cdf_hi_nodes[j] = cdf_fn(hi_edge, shapes, beta, *extra)
    weights = ROUNDING_WEIGHTS[:, None]
    cdf_lo = np.sum(weights * cdf_lo_nodes, axis=0)
    bin_prob = np.sum(weights * (cdf_hi_nodes - cdf_lo_nodes), axis=0)
    return cdf_lo, bin_prob


def _pit_ks_generic(
    index, params, windows_by_machine, spec, n_seeds: int = 200,
) -> dict[str, dict]:
    """Shared PIT-KS + trajectory RMSE + terminal bias validation.

    ``spec`` is a ``degradation_families.FAMILY_REGISTRY`` entry: its
    ``cdf_fn`` gives the family's CDF at x given the per-window shape
    quantity S and shared beta (plus any ``extra_shape_values``, e.g. GIG's
    p_gid); ``is_subordinator`` clamps the rounding-bin lower edge away
    from zero for families whose CDF is undefined for x <= 0 (every
    subordinator family) -- non-subordinator Wiener-family CDFs are defined
    everywhere, so no clamp is needed there.

    Originally hardcoded to an inline age-only shape formula (both for the
    PIT residual's shape quantity and the trajectory-RMSE mean curve),
    which would KeyError on any family whose components don't have plain
    "eta"/"power" keys (multi-stage Wiener has "eta1"/"eta2"/"tau_c"
    instead) and silently ignore wear-level feedback for state-dependent
    variants. Both now go through ``spec["shape_fn"]``. The trajectory-RMSE
    mean curve specifically holds wear level fixed at each tool's own
    FIRST recorded reading for its entire duration (rather than tracking
    it as it accumulates) for every family, including state-dependent ones
    -- a deliberate, documented approximation matching the level of
    rigor this diagnostic already had before this generalization (a single
    smooth mean curve from the trajectory's start, not a full re-simulation
    at every point), not a claim that state-dependent feedback is being
    fully accounted for in this specific diagnostic.

    The mean curve's increment formula goes through
    ``spec["mean_increment_fn"]`` (default ``shape_s/beta`` for every
    family except truncated Wiener -- see
    ``degradation_families.standard_mean_increment``), not a further
    hardcoded ``shape_s/beta``: a real-data run caught this the hard way
    when truncated Wiener showed a -105 to -145um terminal bias with every
    machine failing validation, which looked like (and was initially
    suspected to be) a bad fit -- it was this diagnostic silently using
    the untruncated normal's mean instead of the truncated one's genuinely
    higher mean (confirmed ~1.8x too low on a real fitted component), not
    a problem with the MLE fit itself (which already used the correct
    truncated CDF throughout, via ``_truncated_wiener_cdf``).

    The rounded-observation likelihood is only identified up to a randomized
    de-quantization draw within each rounding bin, so the PIT-KS statistic
    from any single draw is noisy at this sample size (n windows ~ 120-180):
    a single unlucky/lucky seed can swing a borderline case across the p=0.05
    boundary. Averaging the KS statistic and p-value over many independent
    draws gives a stable, defensible calibration verdict instead of relying
    on one arbitrary realization.
    """
    validation = {}
    for machine_key, gp in params.items():
        data = windows_by_machine[machine_key]
        cdf_lo, bin_prob = _rounding_kernel_cdf_bounds(data, gp, spec)
        # _rounding_kernel_cdf_bounds builds its own component_by_tool
        # internally (needed there for the PIT shape computation); the
        # trajectory-RMSE loop below needs the identical per-tool lookup
        # for its own, separate purpose (the mean-curve shape_fn calls), so
        # it is rebuilt here rather than threaded out of the helper --
        # cheap (a few-entry dict) and keeps the helper's return signature
        # focused on what it's actually for (PIT bounds).
        component_by_tool = {int(c["tool_id"]): c for c in gp["components"]}

        ks_stats = np.empty(n_seeds, float)
        ks_pvalues = np.empty(n_seeds, float)
        base_seed = 2026 + int(gp["machine_id"]) * 10_000
        for s in range(n_seeds):
            rng = np.random.default_rng(base_seed + s)
            pit = cdf_lo + rng.random(len(cdf_lo)) * bin_prob
            pit = np.clip(pit, 1e-12, 1.0 - 1e-12)
            ks_stats[s], ks_pvalues[s] = kstest(pit, "uniform")
        ks_stat = float(np.mean(ks_stats))
        ks_pvalue = float(np.mean(ks_pvalues))
        frac_reject_05 = float(np.mean(ks_pvalues < 0.05))

        machine_id = int(gp["machine_id"])
        sub = index[index["machine"] == machine_id]
        squared_errors = []
        terminal_biases = []
        # For a family whose mean is LINEAR in shape_s (shape_s/beta --
        # every family except truncated Wiener), shape_s(s,t) telescopes
        # exactly regardless of step granularity, so a single
        # shape_s(t[0], t[i]) call per observed point gives the identical,
        # exact cumulative mean as accumulating many smaller steps would --
        # this is the ORIGINAL, unchanged computation, preserving every
        # already-validated family's numbers exactly.
        #
        # Truncated Wiener's mean is NOT linear in shape_s (the truncation
        # correction term), so this equivalence breaks: summing many small
        # per-step truncated means != the truncated mean of one big
        # aggregate shape_s, and -- confirmed empirically on real data --
        # the discrepancy is highly sensitive to the step size used (one
        # big step under-predicted cumulative wear by >100um; per-OBSERVED-
        # RUN steps, much finer than the window_size the process was
        # actually fit at, over-predicted by >300um in the other
        # direction). The only granularity consistent with what the fitted
        # (eta, beta) actually represent is the SAME window_size the
        # likelihood was fit at, so this family's predicted curve is built
        # by accumulating truncated-mean steps at window-size checkpoints
        # (plus a final checkpoint for any partial trailing window) and
        # linearly interpolating between them for the intermediate runs
        # RMSE compares against -- exact at every checkpoint (a telescoping
        # sum of window-granularity steps) and terminal_bias always lands
        # exactly on the last checkpoint regardless of interpolation.
        is_linear_mean = spec["mean_increment_fn"] is families.standard_mean_increment
        window_size = int(gp.get("window_size", 15))
        for tool, traj in sub.groupby("tool"):
            traj = traj.sort_values("run")
            c = component_by_tool[int(tool)]
            time = traj["cumulated_tool_contact_time"].to_numpy(float)
            wear = traj["wear"].to_numpy(float)
            beta_c = float(c["beta"])
            if is_linear_mean:
                t_start_fixed = np.full_like(time, time[0])
                v_start_fixed = np.full_like(time, wear[0])
                mean_shape = spec["shape_fn"](c, t_start_fixed, time, v_start_fixed)
                mean_increment = spec["mean_increment_fn"](mean_shape, beta_c)
                predicted = wear[0] + mean_increment
            else:
                checkpoint_idx = list(range(0, len(time), window_size))
                if checkpoint_idx[-1] != len(time) - 1:
                    checkpoint_idx.append(len(time) - 1)
                checkpoint_idx = np.array(checkpoint_idx)
                checkpoint_times = time[checkpoint_idx]
                v_start_fixed = np.full(len(checkpoint_times) - 1, wear[0])
                step_shape = spec["shape_fn"](c, checkpoint_times[:-1], checkpoint_times[1:], v_start_fixed)
                step_mean_increment = spec["mean_increment_fn"](step_shape, beta_c)
                checkpoint_predicted = wear[0] + np.concatenate([[0.0], np.cumsum(step_mean_increment)])
                predicted = np.interp(time, checkpoint_times, checkpoint_predicted)
            squared_errors.extend(np.square(wear - predicted))
            terminal_biases.append(float(predicted[-1] - wear[-1]))

        rmse = float(np.sqrt(np.mean(squared_errors)))
        terminal_bias = float(np.mean(terminal_biases))
        delta_aic = float(gp["stationary_aic"] - gp["nonhomogeneous_aic"])
        age_model_supported = bool(gp.get("selected_age_model") == "linear" or delta_aic > 2.0)
        # A single realization can cross the p=0.05 boundary by chance at this
        # sample size; require the *majority* of independent PIT draws to
        # fail before declaring miscalibration, in addition to the mean
        # KS-statistic threshold used by the existing Gamma validation.
        passed = bool(
            np.isfinite(rmse) and rmse <= 20.0 and abs(terminal_bias) <= 15.0
            and ks_stat <= 0.15 and frac_reject_05 < 0.5 and age_model_supported
        )
        result = {
            "pit_ks_stat": ks_stat, "pit_ks_pvalue": ks_pvalue,
            "pit_ks_frac_reject_05": frac_reject_05, "pit_ks_n_seeds": int(n_seeds),
            "trajectory_rmse_um": rmse, "terminal_bias_um": terminal_bias,
            "delta_aic_vs_stationary": delta_aic, "n_windows": len(data["dv"]),
            "validation_passed": passed,
        }
        validation[machine_key] = result
        gp.update(result)
    return validation


def _ig_cdf(x, shape_s, beta):
    x = np.maximum(x, 1e-9)
    return invgauss.cdf(x, mu=1.0 / shape_s, loc=0.0, scale=(shape_s ** 2) / beta)


def _wiener_cdf(x, shape_s, beta):
    mu = shape_s / beta
    sigma = np.sqrt(shape_s) / beta
    return norm.cdf(x, loc=mu, scale=sigma)


def validate_family_fit(index: pd.DataFrame, params: dict[str, dict], family_key: str) -> dict[str, dict]:
    """PIT-KS + trajectory RMSE + terminal bias validation for any registered
    family. ``validate_ig_fit``/``validate_wiener_fit`` below are thin,
    signature-preserving wrappers around this (kept because existing tests
    call them directly by name with exactly ``(index, params)``); every
    other family's fit is validated by calling this directly with its key.
    """
    spec = families.FAMILY_REGISTRY[family_key]
    windows = build_wear_windows(
        index, window_size=next(iter(params.values())).get("window_size", 15)
    )
    result = _pit_ks_generic(index, params, windows, spec)
    for machine_key, r in result.items():
        extra_note = ""
        if "mean_p_negative_increment" in params[machine_key]:
            extra_note = f"  P(negative increment)={params[machine_key]['mean_p_negative_increment']:.3f}"
        print(
            f"  Validation {machine_key} ({spec['label']}): mean PIT-KS={r['pit_ks_stat']:.3f} "
            f"(mean p={r['pit_ks_pvalue']:.3f}, reject@.05={r['pit_ks_frac_reject_05']:.0%} "
            f"of {r['pit_ks_n_seeds']} seeds), trajectory RMSE={r['trajectory_rmse_um']:.2f} µm, "
            f"terminal bias={r['terminal_bias_um']:+.2f} µm, passed={r['validation_passed']}{extra_note}"
        )
    return result


def validate_ig_fit(index: pd.DataFrame, params: dict[str, dict]) -> dict[str, dict]:
    """PIT-KS + trajectory RMSE + terminal bias validation for the IG fit."""
    return validate_family_fit(index, params, "inverse_gaussian")


def validate_wiener_fit(index: pd.DataFrame, params: dict[str, dict]) -> dict[str, dict]:
    """PIT-KS + trajectory RMSE + terminal bias validation for the Wiener fit."""
    return validate_family_fit(index, params, "wiener")


# The two families whose shape_fn genuinely reads v_start as feedback rather
# than as a fixed age anchor -- state_aware_trajectory_resimulation below
# exists specifically to correct _pit_ks_generic's trajectory-RMSE/terminal-
# bias mean curve for exactly these two, so this list, not a general
# registry-introspection rule, is the right scope: it names the two families
# the correction is FOR, rather than trying to infer "state-dependence" from
# shape_fn identity (multi-stage Wiener's shape_fn also ignores v_start, but
# for an unrelated reason -- it has no wear-level feedback term at all -- so
# it is correctly excluded here without needing a separate check for it).
STATE_DEPENDENT_FAMILY_KEYS = ("gamma_state", "inverse_gaussian_state")


def state_aware_trajectory_resimulation(
    index: pd.DataFrame, comparison_result: dict, window_size: int = 15,
) -> dict[str, dict]:
    """Corrected trajectory-RMSE/terminal-bias re-simulation for the two
    wear-level-dependent families, tracking each tool's own accumulating
    SIMULATED wear level window by window instead of freezing it at the
    tool's first recorded reading for the whole trajectory.

    ``_pit_ks_generic``'s mean-curve diagnostic (used for every family,
    including these two) evaluates ``spec["shape_fn"](c, t[0], t[i], v_start=
    wear[0])`` for every later time t[i] -- correct for every family whose
    shape_fn ignores v_start, but for these two, whose whole premise is that
    the wear-level-dependent multiplicative term ``1 + gamma_state *
    v_start/EOL_THRESHOLD_UM`` feeds back into the shape function, this holds
    the feedback term fixed at the tool's STARTING wear for its entire life
    instead of letting it evolve, exactly the approximation Section 4.5 of
    the paper documents. This function recomputes the same two diagnostics
    the correct way: recursively, at the same window_size granularity the
    model was actually fit at (mirroring _pit_ks_generic's own truncated-
    Wiener checkpoint/interpolation handling for a DIFFERENT nonlinearity),
    but feeding each checkpoint's own PREDICTED wear level forward into the
    next checkpoint's shape_fn call, rather than the tool's fixed starting
    reading.

    This checkpoint recursion is the tool's EXACT conditional-mean
    trajectory under the fitted model, not a Monte Carlo approximation of
    it: both families here are non-negative subordinators (Gamma, Inverse
    Gaussian) fit with mean_increment_fn = shape_s/beta (no clipping is ever
    applied to their draws, unlike Wiener's), and shape_s itself is affine
    in v_start (``eta*[...]*(1+gamma_state*v_start/EOL)``). For an affine
    function of a random variable, E[f(V)] = f(E[V]) exactly, so recursing
    the checkpoint mean forward is mathematically identical to the mean of
    infinitely many random step-by-step replicates seeded from the same
    starting wear -- the same recursive construction
    ``_simulate_synthetic_dv`` already uses for model recovery (feeding
    accumulating simulated wear back into shape_fn one window at a time),
    evaluated at its exact expectation rather than a single random draw per
    step, which is a strictly more precise (zero Monte-Carlo-noise) way to
    get the same quantity a large-N Monte Carlo average would converge to.
    ``TestStateAwareResimulation.test_matches_monte_carlo_average`` in
    tests/test_degradation_process_models.py checks this equivalence
    directly against an explicit random-draw simulation, not just by
    inspection of this argument.

    Returns, per state-dependent family key and machine key: the naive
    (frozen-wear-level) and state-aware trajectory_rmse_um/terminal_bias_um,
    the fraction of each naive statistic's own magnitude the state-aware
    version closes, and whether the state-aware RMSE/bias alone (holding
    every other Section 4.5 calibration condition -- PIT-KS, age-model --
    at its already-computed value, since those do not depend on this
    specific approximation; see this function's own module-level docstring
    discussion) would flip that family's calibration verdict on that
    machine.
    """
    result: dict[str, dict] = {}
    for family_key in STATE_DEPENDENT_FAMILY_KEYS:
        params_key = families.param_key(family_key)
        if params_key not in comparison_result:
            continue
        spec = families.FAMILY_REGISTRY[family_key]
        per_machine: dict[str, dict] = {}
        for machine_key, gp in comparison_result[params_key].items():
            if not isinstance(gp, dict) or "components" not in gp:
                continue
            component_by_tool = {int(c["tool_id"]): c for c in gp["components"]}
            machine_id = int(gp["machine_id"])
            sub = index[index["machine"] == machine_id]
            w_size = int(gp.get("window_size", window_size))
            squared_errors = []
            terminal_biases = []
            for tool, traj in sub.groupby("tool"):
                traj = traj.sort_values("run")
                c = component_by_tool[int(tool)]
                time = traj["cumulated_tool_contact_time"].to_numpy(float)
                wear = traj["wear"].to_numpy(float)
                beta_c = float(c["beta"])
                checkpoint_idx = list(range(0, len(time), w_size))
                if checkpoint_idx[-1] != len(time) - 1:
                    checkpoint_idx.append(len(time) - 1)
                checkpoint_times = time[np.array(checkpoint_idx)]
                predicted_checkpoints = [wear[0]]
                v_running = wear[0]
                for k in range(len(checkpoint_times) - 1):
                    s_k = checkpoint_times[k:k + 1]
                    t_k = checkpoint_times[k + 1:k + 2]
                    shape_s = spec["shape_fn"](c, s_k, t_k, np.array([v_running]))
                    increment = float(spec["mean_increment_fn"](shape_s, beta_c)[0])
                    v_running = v_running + increment
                    predicted_checkpoints.append(v_running)
                predicted = np.interp(time, checkpoint_times, np.array(predicted_checkpoints))
                squared_errors.extend(np.square(wear - predicted))
                terminal_biases.append(float(predicted[-1] - wear[-1]))
            rmse = float(np.sqrt(np.mean(squared_errors)))
            terminal_bias = float(np.mean(terminal_biases))
            naive_rmse = float(gp["trajectory_rmse_um"])
            naive_bias = float(gp["terminal_bias_um"])
            rmse_gap_closed_frac = (1.0 - rmse / naive_rmse) if naive_rmse > 0 else float("nan")
            bias_gap_closed_frac = (
                1.0 - abs(terminal_bias) / abs(naive_bias)
            ) if naive_bias != 0 else float("nan")
            state_aware_rmse_bias_pass = bool(rmse <= 20.0 and abs(terminal_bias) <= 15.0)
            other_conditions_pass = bool(
                gp["pit_ks_stat"] <= 0.15 and gp["pit_ks_frac_reject_05"] < 0.5
                and bool(gp.get("selected_age_model") == "linear" or gp["delta_aic_vs_stationary"] > 2.0)
            )
            per_machine[machine_key] = {
                "naive_trajectory_rmse_um": naive_rmse, "naive_terminal_bias_um": naive_bias,
                "state_aware_trajectory_rmse_um": rmse, "state_aware_terminal_bias_um": terminal_bias,
                "rmse_gap_closed_frac": rmse_gap_closed_frac, "bias_gap_closed_frac": bias_gap_closed_frac,
                "state_aware_validation_passed": bool(state_aware_rmse_bias_pass and other_conditions_pass),
            }
            print(
                f"  State-aware re-simulation {machine_key} ({spec['label']}): "
                f"RMSE {naive_rmse:.2f}->{rmse:.2f} µm ({rmse_gap_closed_frac:+.0%} of gap), "
                f"terminal bias {naive_bias:+.2f}->{terminal_bias:+.2f} µm "
                f"({bias_gap_closed_frac:+.0%} of gap), "
                f"state-aware pass={per_machine[machine_key]['state_aware_validation_passed']}"
            )
        result[family_key] = per_machine
    return result


# ── Validation: PIT-residual autocorrelation (independence-assumption check) ──

def _pit_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation of a 1-D sequence at lags 1..max_lag (plain
    mean-centered, variance-normalized estimator -- adequate here since
    ``x`` is always one tool's modestly-sized window sequence, not a long
    series where an FFT-based estimator's efficiency would matter). Returns
    NaN at any lag with fewer than 2 usable pairs, or throughout if ``x``
    is (numerically) constant.
    """
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    out = np.full(max_lag, np.nan)
    denom = np.sum(x ** 2)
    if denom < 1e-12:
        return out
    for lag in range(1, max_lag + 1):
        if n - lag < 2:
            break
        out[lag - 1] = np.sum(x[:-lag] * x[lag:]) / denom
    return out


def residual_autocorrelation_check(
    index: pd.DataFrame,
    comparison_result: dict,
    window_size: int = 15,
    max_lag: int = 5,
    n_seeds: int = 50,
) -> dict:
    """Lag-1..max_lag autocorrelation of PIT residuals, per tool, for every
    registered family with fitted params in ``comparison_result`` -- the
    direct test of the likelihood's core independence assumption (each
    window is treated, conditional on its own fitted shape quantity S, as
    independent of every other window). If PIT residuals remain
    autocorrelated even after conditioning on the fitted age-(and, for
    state-dependent variants, wear-level-)dependent mean/variance, that
    assumption is violated in a way plain AIC/BIC comparison, PIT-KS
    calibration, and simulated-lifetime/maintenance-cost inference all
    implicitly rely on -- raised in peer review and confirmed here on the
    real fitted models, not merely a theoretical worry: at the production
    window_size=15, PIT-residual lag-1 autocorrelation is approximately
    0.5-0.85 across machines and families, decaying roughly geometrically
    at longer lags (consistent with an AR(1)-like structure, not white
    noise).

    A second, equally important finding this check documents: the
    autocorrelation is strongly WINDOW-SIZE dependent. Refitting Inverse
    Gaussian at window_size in {3,5,8,10,15,20,30} (all else equal) on the
    real dataset shows mean lag-1 PIT-residual autocorrelation growing
    from ~0.10 at window_size=3 (the finest granularity the data supports)
    to ~0.70-0.73 at window_size=15-20 -- near zero at fine temporal
    granularity, growing systematically as windows coarsen. This is the
    signature of a TEMPORAL-AGGREGATION artifact (interacting with an
    imperfectly-specified smooth power-law age trend) rather than solely a
    deeply memoried physical process: summing many fine increments into
    one coarse window blurs any local mismatch between the fitted smooth
    trend and the true, more locally variable trend into a slowly-varying
    residual pattern, which surfaces as positive autocorrelation between
    adjacent coarse windows even when the finer-grained process is close
    to memoryless. This does NOT make the autocorrelation at window_size=15
    (the production setting) any less real, and it is not a license to
    ignore it -- it is corrected for via cluster-robust inference (see
    ``vuong_test``) -- but it means the finding should not be read as
    proof that milling-tool wear itself is a long-memory process; see
    ``window_size_robustness_sweep`` (Table S2) for whether this study's
    qualitative conclusions survive at less-aggregated window sizes where
    this specific artifact is much smaller.

    Computed on the SAME rounded-observation, rounding-kernel-marginalized
    PIT residuals as ``_pit_ks_generic``'s calibration check (via
    ``_rounding_kernel_cdf_bounds``, reusing that exact, already-validated
    machinery rather than a second, independently-drifting copy), split
    into each tool's own chronological window sequence (``build_wear_windows``
    orders each tool's own windows by run/age) -- autocorrelation is only
    meaningful WITHIN a tool's own sequence, never computed across a tool
    boundary, since tools are the genuinely independent unit here.
    Averaged over ``n_seeds`` independent de-quantization draws for the
    same reason ``_pit_ks_generic`` averages its KS statistic over many
    seeds (the rounded-observation likelihood is only identified up to a
    randomized within-bin draw), though in practice the ACF estimate is
    far less seed-sensitive than the KS statistic since the correlation
    signal here is much larger than the rounding-offset noise.
    """
    windows = build_wear_windows(index, window_size=window_size)
    family_keys = families.registered_with_params(comparison_result)
    result = {}
    for family_key in family_keys:
        spec = families.FAMILY_REGISTRY[family_key]
        params = comparison_result[families.param_key(family_key)]
        per_machine = {}
        pooled_lag1 = []
        for machine_key, gp in sorted(params.items()):
            data = windows.get(machine_key)
            if data is None or len(data["dv"]) == 0:
                continue
            cdf_lo, bin_prob = _rounding_kernel_cdf_bounds(data, gp, spec)
            tool_ids = data["tool"]
            acf_by_tool = {}
            for tool_val in np.unique(tool_ids):
                mask = tool_ids == tool_val
                cdf_lo_t, bin_prob_t = cdf_lo[mask], bin_prob[mask]
                n_t = int(np.sum(mask))
                if n_t < max_lag + 3:
                    continue  # too few windows on this tool for a stable ACF at this many lags
                lag_draws = np.empty((n_seeds, max_lag), float)
                for s in range(n_seeds):
                    rng = np.random.default_rng(5_000 + s)
                    pit = np.clip(cdf_lo_t + rng.random(n_t) * bin_prob_t, 1e-12, 1.0 - 1e-12)
                    lag_draws[s] = _pit_acf(pit, max_lag)
                acf_mean = np.nanmean(lag_draws, axis=0)
                acf_by_tool[int(tool_val)] = acf_mean.tolist()
                pooled_lag1.append(float(acf_mean[0]))
            machine_lag1 = [v[0] for v in acf_by_tool.values()]
            per_machine[machine_key] = {
                "acf_by_tool": acf_by_tool,
                "mean_lag1_acf": float(np.mean(machine_lag1)) if machine_lag1 else float("nan"),
                "n_tools": len(acf_by_tool),
            }
        overall_range = (
            [float(np.min(pooled_lag1)), float(np.max(pooled_lag1))]
            if pooled_lag1 else [float("nan"), float("nan")]
        )
        result[family_key] = {
            "per_machine": per_machine,
            "overall_mean_lag1_acf": float(np.mean(pooled_lag1)) if pooled_lag1 else float("nan"),
            "overall_lag1_acf_range": overall_range,
            "max_lag": max_lag,
            "n_seeds": n_seeds,
        }
        print(
            f"  {spec['label']}: mean lag-1 PIT-residual ACF = "
            f"{result[family_key]['overall_mean_lag1_acf']:.3f} "
            f"(range [{overall_range[0]:.3f}, {overall_range[1]:.3f}] across "
            f"{len(pooled_lag1)} tools)"
        )
    return result


# ── Unified comparison ─────────────────────────────────────────────────────────

# Extra family-specific diagnostic fields surfaced in each comparison entry
# when present (checked via plain ``in``, never assumed) -- generalizes what
# used to be a Wiener-only hardcoded pair of keys to all 7 variants.
_EXTRA_DIAGNOSTIC_KEYS = (
    "mean_p_negative_increment", "max_p_negative_increment",  # Wiener family
    "gamma_state",                                             # state-dependent Gamma/IG
    "mu_p", "sigma_p", "sigma_p_test",                         # random-effects Gamma/IG
    "tau_c", "changepoint_test",                                # multi-stage Wiener
    "p_gid",                                                    # GIG
)


def fit_and_compare_degradation_models(
    index: pd.DataFrame,
    gamma_params: dict[str, dict],
    window_size: int = 15,
) -> dict:
    """Fit and validate every non-Gamma registered family (Inverse Gaussian,
    Wiener, and all 7 variants), and return a unified comparison across the
    full ``degradation_families.FAMILY_REGISTRY``.

    ``gamma_params`` is the already-fitted, already-validated production Gamma
    result (loaded from artifacts/gamma/gamma_process_params.json) — it is not
    refit here.
    """
    all_params = {"gamma": gamma_params}
    for family_key in families.FAMILY_REGISTRY:
        if family_key == "gamma":
            continue
        params = fit_all_groups_family(index, family_key, window_size=window_size)
        validate_family_fit(index, params, family_key)
        all_params[family_key] = params

    comparison = {}
    for machine_key in sorted(gamma_params):
        family_results = {f: p[machine_key] for f, p in all_params.items()}
        machine_comparison = {}
        for family, r in family_results.items():
            entry = {
                "age_model": r.get("selected_age_model"), "aic": r.get("aic"), "bic": r.get("bic"),
                "pit_ks_stat": r.get("pit_ks_stat"), "pit_ks_pvalue": r.get("pit_ks_pvalue"),
                "pit_ks_frac_reject_05": r.get("pit_ks_frac_reject_05"),
                "pit_ks_n_seeds": r.get("pit_ks_n_seeds"),
                "validation_passed": r.get("validation_passed"),
            }
            entry.update({k: r[k] for k in _EXTRA_DIAGNOSTIC_KEYS if k in r})
            machine_comparison[family] = entry

        g, ig = family_results["gamma"], family_results["inverse_gaussian"]
        machine_comparison["best_aic_family"] = min(
            family_results.items(), key=lambda kv: kv[1]["aic"],
        )[0]
        machine_comparison["delta_aic_gamma_minus_ig"] = float(g["aic"] - ig["aic"])
        # Per-tool free (non-homogeneous) power, retained even when the
        # machine-level AIC prefers the stationary model: the joint
        # 3-tool decision can mask individual tools whose point estimate
        # deviates from an otherwise-stationary fleet (e.g. Tool 2 on
        # M1). Point estimates alone cannot say whether such deviations
        # are genuine or sampling noise from only 3 tools/machine -- see
        # bootstrap_per_tool_power_significance() for the formal test.
        # Scoped to the original Gamma/IG pair (Table 5's own scope), not
        # generalized to the 7 new variants.
        machine_comparison["per_tool_power_gamma"] = {
            int(c["tool_id"]): float(c["power"])
            for c in g.get("components", [])
        }
        machine_comparison["per_tool_power_ig_nonhomogeneous"] = {
            int(c["tool_id"]): float(c["power"])
            for c in ig.get("nonhomogeneous_components", [])
        }
        comparison[machine_key] = machine_comparison

    result = {families.param_key(f): p for f, p in all_params.items()}
    result["comparison"] = comparison
    return result


# ── Out-of-sample validation: leave-one-tool-out predictive likelihood ────────
#
# _FIT_FNS / _BIN_PROB_FNS are defined at the BOTTOM of this file (after
# every family, including the six variants imported from
# degradation_process_variants.py, has registered) as thin views over
# degradation_families.FAMILY_REGISTRY -- Python resolves module-level
# globals at CALL time, not at the function-definition time of the
# functions below that reference them, so this forward reference is safe
# (nothing in this module calls these functions at import time). Kept as
# plain module-level dicts, not registry lookups inline at every call site,
# so the existing "for family, fit_fn in _FIT_FNS.items()" call sites below
# did not all need editing individually.
#
# IMPORTANT for any NEW call site added here: _BIN_PROB_FNS[family] is NOT
# always callable as bin_prob_fn(observed, shape_s, beta, resolution_um)
# alone -- GIG additionally needs its tail-shape parameter p_gid appended
# as a trailing positional arg. Use
# families.extra_shape_values(families.FAMILY_REGISTRY[family], fitted)
# to get that (an empty list for every family except GIG) and pass it as
# *extra; see leave_one_tool_out_cv_likelihood and vuong_test below for
# the pattern.


def leave_one_tool_out_cv_likelihood(index: pd.DataFrame, window_size: int = 15) -> dict:
    """Out-of-sample predictive log-likelihood comparison across families.

    Complements the in-sample AIC comparison in ``fit_and_compare_degradation_models``
    (which can only show a family fits its OWN training data well) with genuine
    held-out evidence: for each machine, each of its 3 tools is held out in
    turn, the family is refit on the *other* 2 tools only, and the held-out
    tool's windows are scored under that fit. Since a genuinely new tool has
    no fitted per-tool component of its own, the median (eta, power) across
    the *trained* tools is used as the prediction for it -- this mirrors how
    the model would actually be applied to an unseen tool in practice, and
    is not favorable to any one family a priori.

    Higher (less negative) cv_total_loglik is better.

    Generalized to the full family registry (originally hardcoded to
    Gamma/IG/Wiener): the per-family shape-quantity prediction for the
    unseen tool is delegated to each family's own registered
    ``predict_shape_fn`` (median eta/power -- and, for state-dependent
    variants, median gamma_state -- across the *trained* tools, matching
    how each family's own fitter already predicts for an unobserved
    component) rather than a single hardcoded age-only formula, which
    would silently mispredict for any family whose shape quantity depends
    on more than (eta, power, age) -- state-dependent variants also need
    wear level, multi-stage Wiener needs its own two-rate/change-point
    form. ``families.extra_shape_values`` supplies GIG's p_gid (empty for
    every other family) to ``bin_prob_fn``.

    Every family is refit up to 3 times per machine (once per held-out
    tool) -- ``_REPEATED_REFIT_COARSENING`` applies the same precision/cost
    tradeoff used in ``model_recovery_study`` (coarser multi-stage Wiener
    change-point grid, coarser random-effects quadrature) here too, since
    random-effects Gamma/IG's fit cost (~250-330s per machine at full
    precision on real data) multiplied by up to 3 folds x 3 machines is the
    difference between a runnable check and one that does not finish in a
    practical amount of time.
    """
    windows = build_wear_windows(index, window_size=window_size)
    result = {}
    for machine_key, data in sorted(windows.items()):
        tools = np.unique(data["tool"])
        if len(tools) < 2:
            continue
        per_family = {family: {"fold_loglik": [], "fold_n": []} for family in _FIT_FNS}
        for held_tool in tools:
            train_mask = data["tool"] != held_tool
            held_mask = data["tool"] == held_tool
            for family, fit_fn in _FIT_FNS.items():
                spec = families.FAMILY_REGISTRY[family]
                fitted = fit_fn(
                    data["dv"][train_mask], data["t_start"][train_mask],
                    data["t_end"][train_mask], data["tool"][train_mask],
                    v_start=data["v_start"][train_mask],
                    **_REPEATED_REFIT_COARSENING.get(family, {}),
                )
                beta = float(fitted["beta"])
                t_start_h = data["t_start"][held_mask]
                t_end_h = data["t_end"][held_mask]
                v_start_h = data["v_start"][held_mask]
                dv_h = data["dv"][held_mask]
                shape_s = spec["predict_shape_fn"](fitted, t_start_h, t_end_h, v_start_h)
                shape_s = np.maximum(shape_s, 1e-12)
                extra = families.extra_shape_values(spec, fitted)
                prob = _BIN_PROB_FNS[family](
                    dv_h, shape_s, beta, float(fitted["wear_resolution_um"]), *extra
                )
                per_family[family]["fold_loglik"].append(float(np.sum(np.log(prob))))
                per_family[family]["fold_n"].append(int(held_mask.sum()))

        summary = {}
        for family, folds in per_family.items():
            total_loglik = float(np.sum(folds["fold_loglik"]))
            total_n = int(np.sum(folds["fold_n"]))
            summary[family] = {
                "cv_total_loglik": total_loglik,
                "cv_mean_loglik_per_window": total_loglik / max(total_n, 1),
                "fold_loglik": folds["fold_loglik"],
                "n_windows": total_n,
            }
        best_family = max(summary, key=lambda f: summary[f]["cv_total_loglik"])
        result[machine_key] = {"per_family": summary, "best_cv_family": best_family}
        print(
            f"  {machine_key} CV (leave-one-tool-out): "
            + "  ".join(
                f"{f}={summary[f]['cv_total_loglik']:.2f}"
                for f in summary
            )
            + f"  -> best={best_family}"
        )
    return result


# ── Model recovery (positive control) ──────────────────────────────────────────

def _simulate_synthetic_dv(
    spec: dict,
    params: dict,
    t_start: np.ndarray,
    t_end: np.ndarray,
    tool_ids: np.ndarray,
    resolution_um: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate rounded wear increments at a given window structure under a
    KNOWN family (``spec``, a ``degradation_families.FAMILY_REGISTRY`` entry)
    and its fitted ``params``. Draws are rounded but NOT clipped at zero
    here -- the caller must drop (not clip) any negative draws to faithfully
    reproduce ``build_wear_windows``'s own ``dv < 0: continue`` filter,
    which is what a hypothetical Wiener-shaped real process would actually
    look like after passing through the real data pipeline.

    Matches the two-endpoint aggregated rounding model now used by the
    likelihood (``gamma_process._aggregated_bin_probability``): the observed
    integer window value is round(X + u), not round(X), where u is an
    additional Uniform(-0.5, +0.5) um offset representing the window-start
    reading's own independent rounding remainder.

    Generalized to the full family registry (originally hardcoded to
    Gamma/IG/Wiener via a plain if/elif on distribution family): shape is
    computed via ``spec["shape_fn"]`` and drawn via ``spec["rvs_fn"]``
    rather than an inline age-only formula and a hardcoded dispatch, since
    neither is valid for every family (multi-stage Wiener's components
    don't even have "eta"/"power" keys -- only "eta1"/"eta2"/"tau_c" --
    so the old inline formula would KeyError on it; state-dependent
    variants need current wear level, not just age).

    State-dependent variants' shape_fn needs the wear level AT EACH
    WINDOW'S START, but that level is itself the running sum of this
    function's OWN simulated draws (a truly random, not-yet-known
    trajectory for a hypothetical synthetic tool) -- inherently sequential,
    since window i's input depends on window i-1's simulated output, not
    vectorizable across a tool's windows the way the other 9 families are.
    ``t_start``/``t_end``/``tool_ids`` are assumed grouped by tool with each
    tool's own windows in chronological order, exactly how
    ``gamma_process.build_wear_windows`` constructs them (grouped by
    (machine, tool), each tool's windows appended in increasing time order)
    -- required for the cumulative wear-level tracking below to be correct.
    """
    component_by_tool = {int(c["tool_id"]): c for c in params["components"]}
    extra = families.extra_shape_values(spec, params)
    beta = float(params["beta"])
    latent = np.empty(len(t_start), float)
    cum_wear: dict[int, float] = {}
    for i, tool in enumerate(tool_ids):
        tool = int(tool)
        component = component_by_tool[tool]
        v_start_i = np.array([cum_wear.get(tool, 0.0)])
        shape_s_i = spec["shape_fn"](
            component, t_start[i:i + 1], t_end[i:i + 1], v_start_i,
        )
        draw = spec["rvs_fn"](shape_s_i, beta, rng, *extra)
        # np.atleast_1d, not draw[0] directly: most registered rvs_fn
        # implementations delegate to scipy.stats.X.rvs(a=array, ...)
        # without an explicit size= -- for a genuinely 1-element input
        # array (exactly what every call here is), scipy's own broadcasting
        # collapses the result to a bare scalar for at least gamma,
        # invgauss, norm, and geninvgauss (confirmed directly: 9 of this
        # registry's 10 rvs_fn return shape=() here, not shape=(1,); only
        # truncated Wiener's custom implementation, which explicitly sizes
        # its output via rng's own array-shaped draw rather than relying on
        # scipy's broadcasting, was already immune). Every OTHER caller of
        # rvs_fn in this codebase passes a much larger batch array (e.g.
        # simulate_process_lifetimes's n_sim=500), where this ambiguity
        # does not arise -- this sequential, one-window-at-a-time loop is
        # the only place it does, so the fix belongs here, not in 9
        # otherwise-correct rvs_fn implementations.
        latent[i] = float(np.atleast_1d(draw)[0])
        cum_wear[tool] = cum_wear.get(tool, 0.0) + max(latent[i], 0.0)
    offset = rng.uniform(-0.5, 0.5, size=latent.shape) * resolution_um
    return np.rint((latent + offset) / resolution_um) * resolution_um


def _compute_v_start_from_synthetic_dv(dv: np.ndarray, tool_ids: np.ndarray) -> np.ndarray:
    """Cumulative synthetic wear level at each window's start, matching
    ``gamma_process.build_wear_windows``'s definition of v_start (the
    running sum of a tool's PRIOR windows' increments, this window's own
    increment excluded) -- needed to refit state-dependent variants
    against model-recovery's simulated data, since their shape_fn requires
    it. Assumes dv/tool_ids are grouped by tool with each tool's windows
    in chronological order (true for build_wear_windows' own output and
    preserved by any subsequent boolean masking, which is how this is
    always used)."""
    v_start = np.zeros(len(dv), float)
    cum: dict[int, float] = {}
    for i, tool in enumerate(tool_ids):
        tool = int(tool)
        v_start[i] = cum.get(tool, 0.0)
        cum[tool] = cum.get(tool, 0.0) + max(float(dv[i]), 0.0)
    return v_start


def _wilson_ci(x: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion x/n -- more reliable
    than the normal approximation near 0 or 1 (several recovery-rate cells
    here are exactly 100%, where a symmetric normal-approximation interval
    would be nonsensical, extending above 1.0)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p_hat = x / n
    denom = 1.0 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denom
    half_width = (z * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2))) / denom
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def model_recovery_study(
    index: pd.DataFrame,
    comparison_result: dict,
    window_size: int = 15,
    n_replicates: int = 6,
    rng_seed: int = 0,
    only_true_families: list[str] | None = None,
) -> dict:
    """Positive-control / model-recovery check: does the AIC family-selection
    procedure correctly recover the TRUE generating family when applied to
    synthetic data simulated from each machine's own REAL fitted parameters?

    This addresses a specific, legitimate concern about the headline finding
    (Inverse Gaussian has the lowest AIC on every real machine): could the
    windowed-increment construction or the shared mean/variance
    parametrisation structurally favor Inverse Gaussian regardless of the
    true data-generating process, rather than IG genuinely being the better
    description of this fleet's wear? For each machine and each candidate
    true family, synthetic ``dv`` observations are simulated at the SAME
    window timing and sample size as the real data (using that family's own
    real fitted parameters as ground truth, so the test is representative of
    the actual fitting regime, not an arbitrary toy setup), every registered
    family is refit to the synthetic data exactly as it was to the real
    data, and the fraction of replicates where AIC selects the TRUE family
    is reported, with a Wilson score 95% CI (explicit rather than a bare
    point estimate, since ``n_replicates`` is intentionally small here --
    see below -- and gives correspondingly wide intervals).

    An unbiased procedure should recover all three families at a high rate,
    not just Inverse Gaussian; a low recovery rate for a family (especially
    Gamma or Wiener) would indicate the procedure itself is biased rather
    than genuinely discriminating between families.

    Non-monotone-family-generated replicates (Wiener, multi-stage Wiener)
    lose some windows to the same ``dv < 0`` filter real data would apply
    (see ``_simulate_synthetic_dv``); replicates left with fewer than 10
    windows after filtering are skipped and do not count toward the
    recovery rate denominator. The mean fraction of windows dropped this
    way is reported per cell (``mean_frac_windows_dropped``): this
    filtering also means such a family's OWN fit, when it truly generated
    the data, is scored against a left-truncated sample using its
    untruncated likelihood (matching how it is fit everywhere else in this
    codebase including on real data) -- a real, quantifiable confound on
    that family's self-recovery rate specifically, not purely a
    discriminability-power story. Reporting the drop fraction lets a
    reader judge how much of the gap plausibly owes to this rather than
    assert it is negligible.

    Generalized to the full family registry (originally hardcoded to
    Gamma/IG/Wiener): every family present in ``comparison_result`` is
    tested both as a candidate TRUE generating family and as a fit
    candidate. Two families are deliberately coarsened for every refit
    inside this sweep, each via the documented precision/cost tradeoff its
    own module docstring anticipates for exactly this kind of expensive
    repeated-refit sweep, never applied to the primary real-data fit
    itself: multi-stage Wiener's change-point grid (``n_grid=8`` instead of
    the full-resolution default of 25), and random-effects Gamma/IG's
    Gauss-Hermite quadrature (``n_quad_nodes=7`` instead of the default 21
    -- every quadrature node multiplies the cost of every likelihood
    evaluation the optimizer performs, making it that family's single
    biggest cost driver). This function alone requires up to (n_replicates
    x n_true_families x n_machines) refits of every registered family, so
    both coarsenings matter for keeping total runtime bounded.

    ``n_replicates`` defaults to 6 rather than the original 3-family
    study's 20: even with the coarsening above, one full-registry refit
    cycle (all 10 families, one machine) measures at roughly 300s on this
    project's real data, and the 10-true-family x 3-machine expansion
    means the ORIGINAL n_replicates=20 would put this single check's total
    runtime at a measured-and-extrapolated ~30-50 hours -- impractical for
    a check that is one of thirteen in this study's robustness battery, not
    the whole study. n_replicates=6 keeps the full 10x10 family coverage
    (every family is still tested as both a true-generating and a
    fit-candidate family) while bringing total runtime to roughly half a
    day, at the honestly-reported cost of wider Wilson CIs than the
    original study's -- a deliberate precision/practicality tradeoff made
    explicitly (not silently), consistent with how this function already
    reports its CI width rather than a bare point estimate for exactly
    this reason.

    ``only_true_families``, if given, restricts which families are tested
    as the TRUE generating family (the outer loop) -- e.g. a single-family
    list to run just that family's 3-machine slice. The set of FIT
    candidates each replicate is scored against is untouched by this (still
    every family in ``comparison_result``, via the unrestricted
    ``families.registered_with_params`` call below) -- restricting only
    which true families get tested still gives a correctly-computed
    recovery rate for each one tested, just fewer of the
    ``n_true_families x n_machines`` cells in one call. This exists purely
    so an external caller can checkpoint this function's own documented
    ~half-day runtime one true-family (i.e. ~3-machine, ~1-1.5 hour) slice
    at a time and resume after an interruption without repeating already-
    completed slices, rather than needing this single call to run to
    completion uninterrupted -- default (None) is the original, full,
    single-call behavior, unchanged.
    """
    windows = build_wear_windows(index, window_size=window_size)
    all_registered_families = families.registered_with_params(comparison_result)
    true_family_keys = (
        all_registered_families if only_true_families is None
        else [f for f in all_registered_families if f in only_true_families]
    )
    result = {}
    for machine_key, data in sorted(windows.items()):
        if len(data["t_start"]) == 0 or any(
            machine_key not in comparison_result[families.param_key(f)] for f in all_registered_families
        ):
            continue  # no windows or no fitted params for this machine -- skip rather than crash
        t_start, t_end, tool_ids = data["t_start"], data["t_end"], data["tool"]
        n_total_windows = len(t_start)
        machine_result = {}
        for true_family in true_family_keys:
            true_spec = families.FAMILY_REGISTRY[true_family]
            params = comparison_result[families.param_key(true_family)][machine_key]
            resolution_um = float(params["wear_resolution_um"])

            # Seeded by true_family's index in the FULL registry, not its
            # position within this call's (possibly only_true_families-
            # restricted) true_family_keys -- so a checkpointed run that
            # tests one true family at a time reproduces the exact same
            # per-cell result a single unrestricted call would have, rather
            # than a different draw depending on how the work was chunked.
            fi = all_registered_families.index(true_family)
            rng = np.random.default_rng(rng_seed + 1000 * fi + 37 * int(machine_key[1:]))
            # Fit-candidate set is always the FULL registry, regardless of
            # which true families only_true_families restricts THIS call to
            # testing -- every replicate must still be scored against every
            # possible family for AIC-selection to be meaningful.
            selected_counts = {f: 0 for f in all_registered_families}
            n_success = 0
            frac_dropped_per_rep = []
            for _rep in range(n_replicates):
                dv_sim = _simulate_synthetic_dv(
                    true_spec, params, t_start, t_end, tool_ids, resolution_um, rng,
                )
                keep = dv_sim >= 0
                frac_dropped_per_rep.append(1.0 - float(np.sum(keep)) / n_total_windows)
                if int(np.sum(keep)) < 10:
                    continue
                dv_f, t_start_f, t_end_f, tool_f = (
                    dv_sim[keep], t_start[keep], t_end[keep], tool_ids[keep]
                )
                v_start_f = _compute_v_start_from_synthetic_dv(dv_f, tool_f)
                aics = {}
                try:
                    for family in all_registered_families:
                        fit_fn = _FIT_FNS[family]
                        extra_kwargs = _REPEATED_REFIT_COARSENING.get(family, {})
                        fitted = fit_fn(
                            dv_f, t_start_f, t_end_f, tool_f,
                            v_start=v_start_f, **extra_kwargs,
                        )
                        aics[family] = fitted["aic"]
                except Exception:
                    continue
                if len(aics) < len(all_registered_families):
                    continue
                best = min(aics, key=aics.get)
                selected_counts[best] += 1
                n_success += 1

            recovery_rate = selected_counts[true_family] / n_success if n_success else float("nan")
            ci_lo, ci_hi = _wilson_ci(selected_counts[true_family], n_success)
            machine_result[true_family] = {
                "n_replicates_requested": n_replicates, "n_replicates_success": n_success,
                "recovery_rate": recovery_rate, "recovery_rate_ci_lo": ci_lo, "recovery_rate_ci_hi": ci_hi,
                "selected_counts": selected_counts,
                "mean_frac_windows_dropped": float(np.mean(frac_dropped_per_rep)),
            }
            print(
                f"  {machine_key} true={true_family}: recovery_rate={recovery_rate:.0%} "
                f"[{ci_lo:.0%},{ci_hi:.0%}] ({n_success}/{n_replicates} usable replicates) "
                f"mean_frac_dropped={np.mean(frac_dropped_per_rep):.1%} selected={selected_counts}"
            )
        result[machine_key] = machine_result
    return result


# ── Per-tool fit diagnostics (M3 calibration-failure investigation) ───────────

def per_tool_fit_diagnostics(index: pd.DataFrame, params_by_machine: dict) -> dict:
    """Per-tool trajectory RMSE and terminal bias under a fitted family,
    breaking the machine-level validation summary down by individual tool.

    Written to investigate why M3's Gamma fit fails the 200-seed PIT-KS
    calibration check while M1/M2 pass: this shows the same-magnitude,
    opposite-signed per-tool terminal bias pattern (one tool over-predicts,
    another under-predicts, by a similar ~10-25um amount) is present on ALL
    THREE machines, not unique to M3. What differs is that M1/M2's Gamma fit
    selected the non-homogeneous (power-law) age model -- 2 free parameters
    per tool -- while M3's selected the simpler stationary model (1 free
    parameter per tool, power fixed at 1.0), leaving less flexibility to
    absorb the same real between-tool heterogeneity every machine shows.
    M3's calibration failure is best explained by this loss of flexibility
    from Gamma's own age-model selection landing on the simpler choice for
    that machine, not by an M3-specific physical anomaly (e.g. its
    documented different machine controller) that this per-tool evidence
    does not support as the primary driver.
    """
    result = {}
    for machine_key, params in sorted(params_by_machine.items()):
        component_by_tool = {int(c["tool_id"]): c for c in params["components"]}
        beta = float(params["beta"])
        machine_id = int(machine_key[1:])
        sub = index[index["machine"] == machine_id]
        tool_rows = {}
        for tool, traj in sub.groupby("tool"):
            traj = traj.sort_values("run")
            time = traj["cumulated_tool_contact_time"].to_numpy(float)
            wear = traj["wear"].to_numpy(float)
            c = component_by_tool[int(tool)]
            eta, power, t_ref = float(c["eta"]), float(c["power"]), float(c["time_reference_s"])
            mean_increment = eta / beta * (
                np.power(time / t_ref, power) - (time[0] / t_ref) ** power
            )
            predicted = wear[0] + mean_increment
            resid = wear - predicted
            tool_rows[int(tool)] = {
                "eta": eta, "power": power,
                "rmse_um": float(np.sqrt(np.mean(resid ** 2))),
                "terminal_bias_um": float(predicted[-1] - wear[-1]),
                "terminal_wear_um": float(wear[-1]),
            }
        result[machine_key] = {
            "selected_age_model": params.get("selected_age_model"),
            "per_tool": tool_rows,
        }
    return result


# ── Formal non-nested hypothesis test (Vuong) ──────────────────────────────────

def vuong_test(index: pd.DataFrame, comparison_result: dict, window_size: int = 15) -> dict:
    """Vuong's (1989) closeness test for non-nested model comparison,
    AIC-corrected for unequal parameter counts, as a formal statistical
    complement to the informal Delta-AIC evidence-strength heuristics
    (Burnham & Anderson's >2 substantial / >10 strong conventions) used
    elsewhere in this study.

    For each machine and each pair of families, computes the per-window
    log-likelihood ratio LR_i = log f(y_i) - log g(y_i) under each family's
    own AIC-selected fitted parameters (the SAME fits already reported in
    Table 1, not a refit), then the Vuong z-statistic

        z = sqrt(n) * (mean(LR_i) - (k_f - k_g)/n) / std(LR_i)

    which is asymptotically standard normal under the null that both models
    are equally close (in Kullback-Leibler divergence) to the true
    data-generating process; the (k_f - k_g)/n term is the Schwarz/AIC
    correction for models with a different number of free parameters (this
    matters here since, e.g., Wiener selects the power-law age model on
    every machine while Gamma/IG sometimes select the simpler stationary
    one for the same machine).

    Generalized to the full family registry (originally hardcoded to
    Gamma/IG/Wiener and their 3 pairwise comparisons): every family
    present in ``comparison_result`` is included, compared pairwise against
    every other (n families -> n*(n-1)/2 raw combinations before the
    nested-pair exclusion below, e.g. 45 for the full 10-family registry
    rather than 3) via ``itertools.combinations`` rather
    than a hand-written pair list. Each family's own per-window shape
    quantity is computed via its registered ``shape_fn`` (passing wear
    level at each window's start, needed by state-dependent variants;
    ignored by every family that doesn't use it) rather than a single
    hardcoded age-only formula, and ``families.extra_shape_values`` supplies
    GIG's tail-shape parameter to its bin-probability function.

    Cluster-robust by tool, not the naive i.i.d. std(LR_i)/sqrt(n) SE.
    ``residual_autocorrelation_check`` confirms (on the real fitted models)
    that PIT residuals within a tool's own window sequence carry lag-1
    autocorrelation in roughly the 0.5-0.85 range, decaying geometrically
    at longer lags -- consistent with an AR(1)-like structure, not white
    noise. The per-window log-likelihood ratio LR_i inherits this same
    within-tool dependence (it is built from the same PIT-adjacent
    quantities), so treating each of the ~120-180 windows on a machine as
    an independent draw for variance-estimation purposes overstates the
    precision of mean(LR_i) severely. The 3 tools on a machine, by
    contrast, ARE independent physical units (separate inserts, no shared
    history) -- the correct unit of independence here is the tool, not the
    window.

    The fix is the standard cluster-robust ("sandwich") variance for a
    simple mean (Liang & Zeger, 1986; Cameron, Gelbach & Miller, 2011):
    sum each tool's own TOTAL centered log-lik-ratio, square it, sum across
    tools, normalize by n^2. This places no assumption on the WITHIN-tool
    correlation structure (does not need to be exactly AR(1); works
    whatever the true structure is) -- it only assumes independence ACROSS
    tools. It reduces exactly to the naive i.i.d. variance only in the
    degenerate case of one window per cluster.

    With only G=3 clusters (tools) per machine, the z-statistic's usual
    asymptotic normality (which relies on G -> infinity) is itself not
    trustworthy -- standard small-number-of-clusters practice (Cameron &
    Miller, 2015, "A Practitioner's Guide to Cluster-Robust Inference") is
    to reference a t distribution with (G-1) degrees of freedom instead of
    the standard normal, and to apply a G/(G-1) finite-cluster inflation to
    the sandwich variance itself. Both are applied here. With G=3, df=2 and
    the two-sided 5% critical value is ~4.30 rather than 1.96 -- a much
    higher bar, appropriately reflecting that there are genuinely only 3
    independent replicates per machine feeding this test, not 120-180. Each
    pair's naive (uncorrected) z and p-value are ALSO reported alongside
    the cluster-robust ones (not silently replaced) specifically so the
    gap between them is visible -- direct, quantified evidence of how much
    the within-tool dependence actually matters for inference, not just
    for calibration.
    """
    windows = build_wear_windows(index, window_size=window_size)
    family_keys = families.registered_with_params(comparison_result)
    # Vuong's test assumes strictly non-nested models -- its z-statistic's
    # asymptotic normality derivation does not hold for a variant compared
    # against its own baseline family (nested exactly at the variant's new
    # parameter(s)' null value). Those pairs are excluded here rather than
    # silently computing a z-statistic under the wrong null distribution;
    # families.nested_model_test (a proper likelihood-ratio test) is the
    # statistically correct tool for a variant-vs-baseline comparison and is
    # used for that instead (see model_recovery/the per-variant nested test
    # already run when each variant is fit).
    pairs = [
        (a, b) for a, b in itertools.combinations(family_keys, 2)
        if families.FAMILY_REGISTRY[a]["baseline_key"] != b
        and families.FAMILY_REGISTRY[b]["baseline_key"] != a
    ]
    result = {}
    for machine_key, data in sorted(windows.items()):
        if len(data["dv"]) == 0 or any(
            machine_key not in comparison_result[families.param_key(f)] for f in family_keys
        ):
            continue  # no windows or no fitted params for this machine -- skip rather than crash
        dv, t_start, t_end, tool = data["dv"], data["t_start"], data["t_end"], data["tool"]
        v_start = data["v_start"]
        loglik_i = {}
        k_of = {}
        for family in family_keys:
            spec = families.FAMILY_REGISTRY[family]
            params = comparison_result[families.param_key(family)][machine_key]
            component_by_tool = {int(c["tool_id"]): c for c in params["components"]}
            beta = float(params["beta"])
            shape_s = np.empty(len(dv), float)
            for i, t in enumerate(tool):
                c = component_by_tool[int(t)]
                shape_s[i:i + 1] = spec["shape_fn"](c, t_start[i:i + 1], t_end[i:i + 1], v_start[i:i + 1])
            shape_s = np.maximum(shape_s, 1e-12)
            extra = families.extra_shape_values(spec, params)
            prob = _BIN_PROB_FNS[family](dv, shape_s, beta, float(params["wear_resolution_um"]), *extra)
            loglik_i[family] = np.log(prob)
            aic = float(params["aic"])
            log_lik = float(params["log_lik"])
            k_of[family] = (aic + 2.0 * log_lik) / 2.0

        n = len(dv)
        unique_tools = np.unique(tool)
        tool_masks = [tool == tv for tv in unique_tools]
        n_clusters = len(unique_tools)
        df_cluster = n_clusters - 1
        crit = float(t_dist.ppf(0.975, df=df_cluster)) if df_cluster > 0 else float("nan")
        fc = n_clusters / (n_clusters - 1) if n_clusters > 1 else float("nan")
        machine_result = {}
        for fam_a, fam_b in pairs:
            lr = loglik_i[fam_a] - loglik_i[fam_b]
            k_diff = k_of[fam_a] - k_of[fam_b]
            mean_lr = float(np.mean(lr))
            correction = k_diff / n

            # Naive i.i.d. Vuong z (the original computation) -- retained
            # for direct before/after comparison against the cluster-robust
            # version below, not used for the "favored" decision.
            std_lr = float(np.std(lr, ddof=1))
            z_naive = float(np.sqrt(n) * (mean_lr - correction) / std_lr) if std_lr > 0 else float("nan")
            p_naive = float(2.0 * (1.0 - norm.cdf(abs(z_naive)))) if np.isfinite(z_naive) else float("nan")

            # Cluster-robust (by tool) sandwich variance of mean(LR_i); see
            # this function's docstring for the derivation and the G/(G-1),
            # t(G-1) small-cluster corrections.
            centered = lr - mean_lr
            cluster_sums = np.array([float(np.sum(centered[m])) for m in tool_masks])
            var_cluster = fc * float(np.sum(cluster_sums ** 2)) / (n ** 2)
            se_cluster = float(np.sqrt(var_cluster)) if var_cluster > 0 else float("nan")
            z = float((mean_lr - correction) / se_cluster) if se_cluster > 0 else float("nan")
            p_value = (
                float(2.0 * t_dist.sf(abs(z), df=df_cluster)) if np.isfinite(z) else float("nan")
            )
            if np.isfinite(z) and z > crit:
                favored = fam_a
            elif np.isfinite(z) and z < -crit:
                favored = fam_b
            else:
                favored = "inconclusive"
            machine_result[f"{fam_a}_vs_{fam_b}"] = {
                "z_statistic": z, "p_value": p_value, "favored": favored,
                "mean_loglik_ratio": mean_lr, "k_diff": k_diff, "n_windows": n,
                "n_clusters": n_clusters, "cluster_critical_value": crit,
                "z_statistic_naive_iid": z_naive, "p_value_naive_iid": p_naive,
            }
        result[machine_key] = machine_result
        for pair_key, r in machine_result.items():
            print(
                f"  {machine_key} {pair_key}: z={r['z_statistic']:+.2f} (naive z={r['z_statistic_naive_iid']:+.2f})  "
                f"p={r['p_value']:.4f}  favored={r['favored']}"
            )
    return result


def variant_vs_baseline_comparison(comparison_result: dict) -> dict:
    """Nested-model test (delta-AIC/BIC, likelihood-ratio p-value) for 6 of
    the 7 variants against the baseline (or, for the two random-effects
    families, pooled-exponent-restricted) model each reduces to at its new
    parameter(s)' null value, per machine -- the direct answer to "is this
    variant justified" for each of the 6 it applies to, and the
    statistically correct tool for the comparison (a variant's own
    baseline is nested inside it, which is exactly what ``vuong_test``'s
    non-nested asymptotics do not cover -- see that function's docstring).
    Truncated Wiener is the one variant excluded here -- see this
    docstring's own last paragraph for why.

    Random-effects Gamma/IG have ``baseline_key=None`` -- deliberately, NOT
    an oversight: at sigma_p=0 they collapse to every tool sharing one
    POOLED aging exponent, which is a different parameter structure than
    plain Gamma/IG's own two branches (all-tools-power=1, or each tool with
    an INDEPENDENT free power -- see ``per_tool_power_gamma``), so they are
    genuinely NOT nested inside the plain family the way state-dependent/
    multi-stage/GIG are (confirmed: excluding them from ``vuong_test``'s
    nested-pair filter would be wrong, since gamma vs. gamma_random_effects
    is a legitimate non-nested comparison there). They DO, however, have
    their own genuinely nested, SELF-referential test already computed
    while fitting (``sigma_p_test``: the full random-effects model vs. its
    own pooled-exponent restriction at sigma_p=0, boundary-corrected for
    testing a variance component at its zero boundary) -- included here
    under that honest description, not mislabeled as "vs. Gamma". This
    check must come BEFORE the ``baseline_key is None`` skip below, not
    after -- an earlier version had the ordering backwards, which silently
    dropped both random-effects variants from this comparison entirely
    (caught via a real Table S14 run showing only 4 of 7 variants).

    Multi-stage Wiener's own nested test (``changepoint_test``) is reused
    the same way, subject to the Davies (1987) caveat about tau_c being
    unidentified under the null -- see that field's own note in
    ``degradation_process_variants.py``. State-dependent Gamma/IG and GIG
    have no pre-computed nested test (their new parameter, gamma_state or
    p_gid, is an ordinary interior parameter, not a variance component at a
    boundary) and are computed fresh here via
    ``families.nested_model_test(..., extra_params=1, boundary_corrected=False)``.
    Truncated Wiener has neither a pre-computed test nor a ``baseline_key``
    (it is a structural change -- left-truncating the increment
    distribution -- not a parameter reducing to plain Wiener at some null
    value, so no nested test applies at all) and is not included here; it
    is compared against Gamma/IG/Wiener directly via ``vuong_test`` instead.
    """
    result = {}
    for machine_key in sorted(comparison_result["gamma_params"]):
        machine_result = {}
        for variant_key, spec in families.FAMILY_REGISTRY.items():
            baseline_key = spec["baseline_key"]
            variant_params = comparison_result[families.param_key(variant_key)][machine_key]
            if "sigma_p_test" in variant_params:
                test = variant_params["sigma_p_test"]
                baseline_label = "pooled-exponent (own sigma_p=0)"
            elif "changepoint_test" in variant_params:
                test = variant_params["changepoint_test"]
                baseline_label = baseline_key
            elif baseline_key is not None:
                baseline_params = comparison_result[families.param_key(baseline_key)][machine_key]
                test = families.nested_model_test(baseline_params, variant_params, extra_params=1)
                baseline_label = baseline_key
            else:
                continue
            machine_result[variant_key] = {"baseline": baseline_label, **test}
        result[machine_key] = machine_result
        for variant_key, r in machine_result.items():
            print(
                f"  {machine_key} {variant_key} vs {r['baseline']}: "
                f"deltaAIC={r['delta_aic_baseline_minus_variant']:+.2f}  "
                f"deltaBIC={r['delta_bic_baseline_minus_variant']:+.2f}  "
                f"LR p={r['lr_p_value']:.4f}  supported(AIC)={r['variant_supported_aic']}"
            )
    return result


# ── Multiple-comparisons summary (FDR correction) ──────────────────────────────

def _benjamini_hochberg(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Benjamini-Hochberg step-up FDR correction. Returns, per key, the raw
    p-value, the BH-adjusted q-value (monotone, i.e. non-decreasing as the
    sorted rank increases), and whether it remains significant at ``alpha``
    after correction."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    if m == 0:
        return {}
    raw_q = [min(1.0, p * m / (i + 1)) for i, (_, p) in enumerate(items)]
    q_values = [0.0] * m
    running_min = 1.0
    for i in range(m - 1, -1, -1):
        running_min = min(running_min, raw_q[i])
        q_values[i] = running_min
    return {
        key: {
            "p_value": p, "q_value": q_values[i],
            "significant_at_fdr": bool(q_values[i] <= alpha),
        }
        for i, (key, p) in enumerate(items)
    }


def multiple_comparisons_summary(
    vuong_result: dict, significance_result: dict, alpha: float = 0.05,
) -> dict:
    """Benjamini-Hochberg FDR correction applied WITHIN each natural family of
    simultaneous formal hypothesis tests conducted across this study, and a
    check of whether any conclusion changes after correction.

    Three families are corrected separately (not pooled into one FDR pass
    across all of them, which would mix conceptually different questions and
    is itself questionable practice): (1) the Vuong pairwise tests (3
    machines x every pairwise family comparison Vuong's own test actually
    ran -- 9 with the original 3-family registry, up to 123 [3 x 41] with
    the full 10-family one, since vuong_test itself excludes only the 4
    genuinely nested variant-vs-baseline pairs (gamma_state, ig_state,
    wiener_multistage, and gig each reduce exactly to their own baseline
    family at a null parameter value; C(10,2)=45 total pairs minus those 4
    = 41) that Vuong's non-nested asymptotics don't cover -- truncated
    Wiener and both random-effects families have NO baseline_key (see
    variant_vs_baseline_comparison's docstring for why) and so are correctly
    NOT excluded, still appearing among the 41; not hardcoded to a specific
    count, so this scales automatically with however many families
    vuong_test was run against), (2) the per-tool
    "power vs. stationary" bootstrap tests, (3)
    the per-tool "power vs. siblings" bootstrap tests (9 each, one per
    tool, unaffected by the family-registry expansion since these two are
    specifically about Inverse Gaussian's own per-tool aging exponent, not
    a per-family test). PIT-KS calibration decisions are not included here:
    each already reflects 200-seed averaging specifically to avoid the
    single-draw instability a per-test correction is meant to guard
    against, not a single p-value conducted once, so pooling it into the
    same FDR family would conflate two different remedies for two
    different problems.
    """
    vuong_p = {
        f"{machine_key}:{pair_key}": r["p_value"]
        for machine_key, machine_result in vuong_result.items()
        for pair_key, r in machine_result.items()
    }
    abs_p = {
        f"tool{tool_id}": r["p_value_two_sided"]
        for tool_id, r in significance_result["per_tool_absolute_vs_stationary"].items()
    }
    rel_p = {
        f"tool{tool_id}": r["p_value_two_sided"]
        for tool_id, r in significance_result["per_tool_relative_vs_siblings"].items()
    }

    test_families = {
        "vuong_pairwise": _benjamini_hochberg(vuong_p, alpha),
        "per_tool_absolute": _benjamini_hochberg(abs_p, alpha),
        "per_tool_relative": _benjamini_hochberg(rel_p, alpha),
    }

    summary = {}
    for family_name, corrected in test_families.items():
        n_sig_before = sum(1 for v in corrected.values() if v["p_value"] <= alpha)
        n_sig_after = sum(1 for v in corrected.values() if v["significant_at_fdr"])
        summary[family_name] = {
            "n_tests": len(corrected), "n_significant_uncorrected": n_sig_before,
            "n_significant_fdr_corrected": n_sig_after,
            "conclusion_changed": bool(n_sig_before != n_sig_after),
            "detail": corrected,
        }
        print(
            f"  {family_name}: {n_sig_before}/{len(corrected)} significant uncorrected, "
            f"{n_sig_after}/{len(corrected)} after BH-FDR (alpha={alpha}) -- "
            f"conclusion changed: {summary[family_name]['conclusion_changed']}"
        )
    return summary


# ── Robustness: window-size sweep ──────────────────────────────────────────────

def window_size_robustness_sweep(index: pd.DataFrame, window_sizes: list[int]) -> dict:
    """Refit every registered family at each candidate window_size and report
    whether the best-AIC-family ranking is stable, so the headline finding
    isn't an artifact of the (somewhat arbitrary) window_size=15 aggregation
    choice. Originally hardcoded to 3 families; now iterates the full
    registry via ``_FIT_FNS`` and passes each window's wear level at start
    (needed by state-dependent variants, ignored by every other family).

    This is the most refit-heavy of the three repeated-refit sweeps
    (window_sizes x 3 machines x every family, vs. up to 3 folds or
    n_replicates x n_true_families elsewhere) -- ``_REPEATED_REFIT_COARSENING``
    applies here too, for the same reason: at full precision, random-effects
    Gamma/IG alone (~250-330s/machine on real data) would put the default
    8-window-size sweep's total runtime in the multi-hour range.

    Also records each family's mean lag-1 PIT-residual autocorrelation at
    every window_size (a single de-quantization draw, not
    ``residual_autocorrelation_check``'s 50-seed average -- adequate here
    since this sweep already reports approximate, coarsened numbers by
    design, and the ACF estimate is much less seed-sensitive than a KS
    statistic would be). This is the evidence that the independence-
    assumption violation ``residual_autocorrelation_check`` documents at
    the production window_size=15 is substantially a temporal-aggregation
    artifact: refitting Inverse Gaussian on the real dataset shows mean
    lag-1 PIT-ACF growing from ~0.10 at window_size=3 (the finest
    granularity available) to ~0.70-0.73 at window_size=15-20, i.e. near
    zero at fine granularity and growing systematically as windows
    coarsen -- reusing this sweep's already-fitted ``fitted`` dict at each
    window size rather than a second, separate round of refits.
    """
    result = {}
    for ws in window_sizes:
        windows = build_wear_windows(index, window_size=ws)
        ws_result = {}
        for machine_key, data in sorted(windows.items()):
            n_windows = len(data["dv"])
            if n_windows < 30:
                ws_result[machine_key] = {"skipped": True, "reason": f"only {n_windows} windows"}
                continue
            aics = {}
            acf_by_family = {}
            for family, fit_fn in _FIT_FNS.items():
                fitted = fit_fn(
                    data["dv"], data["t_start"], data["t_end"], data["tool"],
                    v_start=data["v_start"],
                    **_REPEATED_REFIT_COARSENING.get(family, {}),
                )
                aics[family] = float(fitted["aic"])
                spec = families.FAMILY_REGISTRY[family]
                cdf_lo, bin_prob = _rounding_kernel_cdf_bounds(data, fitted, spec)
                tool_ids = data["tool"]
                rng = np.random.default_rng(9_000)
                lag1_by_tool = []
                for tool_val in np.unique(tool_ids):
                    mask = tool_ids == tool_val
                    n_t = int(np.sum(mask))
                    if n_t < 4:
                        continue  # too few windows on this tool for a single lag-1 estimate
                    pit = np.clip(
                        cdf_lo[mask] + rng.random(n_t) * bin_prob[mask], 1e-12, 1.0 - 1e-12,
                    )
                    lag1_by_tool.append(_pit_acf(pit, 1)[0])
                acf_by_family[family] = float(np.nanmean(lag1_by_tool)) if lag1_by_tool else float("nan")
            best = min(aics, key=aics.get)
            ws_result[machine_key] = {
                "aic": aics, "best_aic_family": best,
                "delta_aic_gamma_minus_ig": aics["gamma"] - aics["inverse_gaussian"],
                "n_windows_per_machine": n_windows,
                "mean_lag1_pit_acf": acf_by_family,
            }
        result[str(ws)] = ws_result
        acf_values = [
            v for m in ws_result.values() for v in m.get("mean_lag1_pit_acf", {}).values()
            if np.isfinite(v)
        ]
        mean_acf_str = f"{np.mean(acf_values):.2f}" if acf_values else "n/a"
        print(
            f"  window_size={ws}: "
            + "  ".join(
                f"{m}={ws_result[m].get('best_aic_family', 'SKIPPED')}"
                for m in sorted(ws_result)
            )
            + f"   mean lag-1 PIT-ACF (all families/machines) = {mean_acf_str}"
        )
    return result


# ── Robustness: bootstrap CI on the family AIC gap ─────────────────────────────

def bootstrap_family_aic_gap(
    index: pd.DataFrame,
    window_size: int = 15,
    n_boot: int = 50,
    rng_seed: int = 0,
) -> dict:
    """Bootstrap resample wear windows (with replacement, ignoring tool grouping)
    and refit Gamma and Inverse Gaussian on each draw, to attach a confidence
    distribution to the AIC gap between the two families rather than reporting
    a single point estimate.

    This captures within-tool-composition sampling uncertainty given the
    observed wear process; it does NOT capture between-tool generalization
    uncertainty (n=3 tools/machine is too small for that) -- that question is
    separately addressed by leave_one_tool_out_cv_likelihood().

    ``build_wear_windows`` always creates an entry for every machine in
    ``MACHINE_GROUPS`` regardless of whether ``index`` actually has rows for
    it (e.g. a caller filtering to a subset of machines), so a machine with
    zero windows is skipped here rather than handed to the fit functions,
    which raise on an empty array.
    """
    windows = build_wear_windows(index, window_size=window_size)
    rng = np.random.default_rng(rng_seed)
    result = {}
    for machine_key, data in sorted(windows.items()):
        n = len(data["dv"])
        if n == 0:
            continue
        gaps = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            gamma_fit = _FIT_FNS["gamma"](
                data["dv"][idx], data["t_start"][idx], data["t_end"][idx], data["tool"][idx],
            )
            ig_fit = _FIT_FNS["inverse_gaussian"](
                data["dv"][idx], data["t_start"][idx], data["t_end"][idx], data["tool"][idx],
            )
            gaps.append(gamma_fit["aic"] - ig_fit["aic"])
        gaps = np.asarray(gaps, float)
        result[machine_key] = {
            "n_boot": n_boot,
            "mean_gap": float(np.mean(gaps)),
            "std_gap": float(np.std(gaps)),
            "ci_lo": float(np.percentile(gaps, 2.5)),
            "ci_hi": float(np.percentile(gaps, 97.5)),
            "frac_favoring_ig": float(np.mean(gaps > 0)),
            "gaps": gaps.tolist(),
        }
        print(
            f"  {machine_key}: mean_gap(Gamma-IG)={result[machine_key]['mean_gap']:.2f} "
            f"+/- {result[machine_key]['std_gap']:.2f}  "
            f"95% CI=[{result[machine_key]['ci_lo']:.2f}, {result[machine_key]['ci_hi']:.2f}]  "
            f"frac favoring IG={result[machine_key]['frac_favoring_ig']:.0%}"
        )
    return result


def block_bootstrap_family_aic_gap_by_tool(
    index: pd.DataFrame,
    window_size: int = 15,
    n_boot: int = 2000,
    rng_seed: int = 0,
) -> dict:
    """Block bootstrap resampling WHOLE TOOLS with replacement (never
    individual windows) and refitting Gamma and Inverse Gaussian on each
    draw -- the between-tool-generalization confidence distribution that
    ``bootstrap_family_aic_gap``'s own docstring says it does NOT capture
    and defers to ``leave_one_tool_out_cv_likelihood`` for. Each tool's
    windows are resampled together as one indivisible block, respecting
    this study's own diagnostics: windows within a tool carry substantial
    PIT-residual autocorrelation (``residual_autocorrelation_check``, 0.5-
    0.85 at lag 1), but the tools themselves are independent physical
    units (separate inserts, no shared history) -- the standard
    prescription for bootstrapping dependent data (Kunsch, 1989; Politis &
    Romano's block/moving-block bootstrap literature) is to resample
    intact dependence-preserving blocks, never individual correlated
    observations.

    With exactly n_tools=3 tools on every machine in this dataset,
    resampling 3-with-replacement from {tool1, tool2, tool3} has exactly
    3**3=27 equally likely ordered outcomes -- small enough to enumerate
    EXACTLY rather than approximate by Monte Carlo, giving a
    deterministic, exactly reproducible distribution with zero residual
    sampling noise (``n_boot``/``rng_seed`` are then unused; kept for
    signature compatibility and as a Monte Carlo fallback if ``n_tools``
    is ever large enough that exact enumeration, n_tools**n_tools, stops
    being practical).

    This is deliberately reported as a HONEST, likely-wide interval, not a
    reassuring one: with genuinely only 3 independent tools per machine, a
    between-tool bootstrap has very little information to work with. It
    answers "is the AIC-gap conclusion fragile to which specific 3 tools
    happened to be observed" -- a materially different, complementary
    question to the larger within-tool-composition sample
    ``bootstrap_family_aic_gap`` already reports, not a replacement for it.
    """
    windows = build_wear_windows(index, window_size=window_size)
    result = {}
    for machine_key, data in sorted(windows.items()):
        tool_ids = data["tool"]
        unique_tools = np.unique(tool_ids)
        n_tools = len(unique_tools)
        if n_tools < 2:
            continue  # cannot block-bootstrap between-tool variability with < 2 blocks
        tool_index_map = {int(t): np.where(tool_ids == t)[0] for t in unique_tools}

        exact = n_tools ** n_tools <= 5000
        if exact:
            draws = list(itertools.product(unique_tools, repeat=n_tools))
        else:
            rng = np.random.default_rng(rng_seed)
            draws = [rng.choice(unique_tools, size=n_tools, replace=True) for _ in range(n_boot)]

        gaps = []
        for draw in draws:
            idx = np.concatenate([tool_index_map[int(t)] for t in draw])
            gamma_fit = _FIT_FNS["gamma"](
                data["dv"][idx], data["t_start"][idx], data["t_end"][idx], data["tool"][idx],
            )
            ig_fit = _FIT_FNS["inverse_gaussian"](
                data["dv"][idx], data["t_start"][idx], data["t_end"][idx], data["tool"][idx],
            )
            gaps.append(gamma_fit["aic"] - ig_fit["aic"])
        gaps = np.asarray(gaps, float)
        result[machine_key] = {
            "n_boot": len(draws), "exact_enumeration": exact, "n_tools": n_tools,
            "mean_gap": float(np.mean(gaps)),
            "std_gap": float(np.std(gaps)),
            "ci_lo": float(np.percentile(gaps, 2.5)),
            "ci_hi": float(np.percentile(gaps, 97.5)),
            "frac_favoring_ig": float(np.mean(gaps > 0)),
            "gaps": gaps.tolist(),
        }
        print(
            f"  {machine_key}: block-bootstrap-by-tool "
            f"({'exact enum.' if exact else 'Monte Carlo'}, {len(draws)} draws) "
            f"mean_gap(Gamma-IG)={result[machine_key]['mean_gap']:.2f} "
            f"+/- {result[machine_key]['std_gap']:.2f}  "
            f"95% CI=[{result[machine_key]['ci_lo']:.2f}, {result[machine_key]['ci_hi']:.2f}]  "
            f"frac favoring IG={result[machine_key]['frac_favoring_ig']:.0%}"
        )
    return result


def moving_block_bootstrap_family_aic_gap(
    index: pd.DataFrame,
    window_size: int = 15,
    block_length: int = 8,
    n_boot: int = 25,
    rng_seed: int = 0,
) -> dict:
    """Moving-block bootstrap (Kunsch, 1989) resampling OVERLAPPING blocks of
    ``block_length`` consecutive windows, independently per tool, and
    concatenating them back to that tool's own original window count --
    preserving SHORT-RANGE serial dependence within a tool's own window
    sequence. A third, genuinely complementary angle on the same AIC-gap
    point estimate alongside ``bootstrap_family_aic_gap`` (plain i.i.d.
    window resampling, which destroys within-tool serial structure
    entirely) and ``block_bootstrap_family_aic_gap_by_tool`` (whole-tool
    blocks, which preserve ALL of a tool's internal structure but only
    vary BETWEEN-tool composition, not within-tool structure, across
    replicates) -- not a replacement for either.

    For a tool with n windows, valid block start positions are
    0..n-block_length (n-block_length+1 of them); ceil(n/block_length)
    such blocks are drawn with replacement and concatenated, then
    truncated back to exactly n windows (the standard moving-block
    bootstrap reconstruction). The 3 tools' own independently-resampled
    sequences are then pooled into one joint per-machine dataset for a
    single refit, matching how the real per-machine fit already combines
    all 3 tools (per-tool eta/power, shared per-machine beta). Tools with
    fewer than ``block_length`` windows are excluded (too few windows for
    even one full block); ``n_tools_used`` reports how many tools actually
    contributed, rather than silently producing a degenerate result on a
    coarse window-size sweep or a small test fixture.
    """
    windows = build_wear_windows(index, window_size=window_size)
    rng = np.random.default_rng(rng_seed)
    result = {}
    for machine_key, data in sorted(windows.items()):
        tool_ids = data["tool"]
        tool_index_map = {}
        for t in np.unique(tool_ids):
            idx_t = np.where(tool_ids == t)[0]
            if len(idx_t) >= block_length:
                tool_index_map[int(t)] = idx_t
        if not tool_index_map:
            continue

        gaps = []
        for _ in range(n_boot):
            resampled_idx_parts = []
            for idx_t in tool_index_map.values():
                n_t = len(idx_t)
                n_blocks = -(-n_t // block_length)  # ceil(n_t / block_length)
                starts = rng.integers(0, n_t - block_length + 1, size=n_blocks)
                concatenated = np.concatenate([idx_t[s:s + block_length] for s in starts])[:n_t]
                resampled_idx_parts.append(concatenated)
            idx = np.concatenate(resampled_idx_parts)
            gamma_fit = _FIT_FNS["gamma"](
                data["dv"][idx], data["t_start"][idx], data["t_end"][idx], data["tool"][idx],
            )
            ig_fit = _FIT_FNS["inverse_gaussian"](
                data["dv"][idx], data["t_start"][idx], data["t_end"][idx], data["tool"][idx],
            )
            gaps.append(gamma_fit["aic"] - ig_fit["aic"])
        gaps = np.asarray(gaps, float)
        result[machine_key] = {
            "n_boot": n_boot, "block_length": block_length, "n_tools_used": len(tool_index_map),
            "mean_gap": float(np.mean(gaps)),
            "std_gap": float(np.std(gaps)),
            "ci_lo": float(np.percentile(gaps, 2.5)),
            "ci_hi": float(np.percentile(gaps, 97.5)),
            "frac_favoring_ig": float(np.mean(gaps > 0)),
            "gaps": gaps.tolist(),
        }
        print(
            f"  {machine_key}: moving-block bootstrap (L={block_length}, {n_boot} draws) "
            f"mean_gap(Gamma-IG)={result[machine_key]['mean_gap']:.2f} "
            f"+/- {result[machine_key]['std_gap']:.2f}  "
            f"95% CI=[{result[machine_key]['ci_lo']:.2f}, {result[machine_key]['ci_hi']:.2f}]  "
            f"frac favoring IG={result[machine_key]['frac_favoring_ig']:.0%}"
        )
    return result


# ── Baseline: pooled-across-machines model ──────────────────────────────────────

def pooled_vs_per_machine_comparison(index: pd.DataFrame, comparison_result: dict, window_size: int = 15) -> dict:
    """Fit one global model per family (single shared beta across all 9 tools,
    ignoring machine identity) and compare its total AIC/BIC to the sum of the
    3 separate per-machine fits already in ``comparison_result``.

    Tests whether machine-level modeling (a separate dispersion parameter beta
    per machine) is actually justified, versus a single fleet-wide beta with
    only per-tool eta/power heterogeneity (already present in both designs).

    Each family is fit exactly ONCE here (not a repeated-refit sweep), at
    full precision -- ``_REPEATED_REFIT_COARSENING`` is deliberately NOT
    applied, since this specific comparison's validity depends on it. Still
    worth flagging: random-effects Gamma/IG's cost scales with tool count
    (9 here vs. 3 per machine), so this single fit is likely to take
    noticeably longer than the per-machine primary fit for those two
    families specifically (measured per-machine at ~250-335s on real data;
    see ``model_recovery_study``'s docstring for the full per-family cost
    table).
    """
    windows = build_wear_windows(index, window_size=window_size)
    dv = np.concatenate([windows[m]["dv"] for m in ("M1", "M2", "M3")])
    t_start = np.concatenate([windows[m]["t_start"] for m in ("M1", "M2", "M3")])
    t_end = np.concatenate([windows[m]["t_end"] for m in ("M1", "M2", "M3")])
    tool = np.concatenate([windows[m]["tool"] for m in ("M1", "M2", "M3")])
    v_start = np.concatenate([windows[m]["v_start"] for m in ("M1", "M2", "M3")])

    result = {}
    for family, fit_fn in _FIT_FNS.items():
        pooled = fit_fn(dv, t_start, t_end, tool, v_start=v_start)
        sum_aic = sum(comparison_result["comparison"][m][family]["aic"] for m in ("M1", "M2", "M3"))
        sum_bic = sum(comparison_result["comparison"][m][family]["bic"] for m in ("M1", "M2", "M3"))
        result[family] = {
            "pooled_aic": pooled["aic"], "pooled_bic": pooled["bic"],
            "pooled_beta": pooled["beta"], "pooled_age_model": pooled["selected_age_model"],
            "sum_per_machine_aic": sum_aic, "sum_per_machine_bic": sum_bic,
            "delta_aic_pooled_minus_summed": pooled["aic"] - sum_aic,
            "delta_bic_pooled_minus_summed": pooled["bic"] - sum_bic,
        }
        print(
            f"  {family}: pooled AIC={pooled['aic']:.2f} (sum-per-machine={sum_aic:.2f}, "
            f"delta={pooled['aic']-sum_aic:+.2f})  pooled BIC={pooled['bic']:.2f} "
            f"(sum-per-machine={sum_bic:.2f}, delta={pooled['bic']-sum_bic:+.2f})"
        )
    return result


# ── AIC-vs-BIC age-model criterion sensitivity ─────────────────────────────────

def age_model_criterion_sensitivity(comparison_result: dict) -> dict:
    """For each machine x family, report which age model AIC selects vs. which
    BIC would select, using the already-fitted stationary/non-homogeneous AIC
    and BIC values already present in ``comparison_result`` -- no refit needed.

    ``fit_and_compare_degradation_models`` (and the underlying per-family
    fitters) always select the age model via the AIC rule
    (``stationary_aic - nonhomogeneous_aic > 2.0``). This answers a narrower
    but important question that AIC-only selection can hide: would the
    stricter BIC criterion reach the same age-model conclusion for every
    family on every machine, or is the conclusion criterion-dependent?

    Reuses the identical > 2.0 threshold for the BIC comparison, which is not
    an arbitrary borrowing from the AIC rule: Kass & Raftery, "Bayes
    Factors," Journal of the American Statistical Association 90(430), 1995,
    pp.773-795, doi:10.1080/01621459.1995.10476572, independently establish
    a > 2 cutoff between "not worth more than a bare mention" and "positive"
    evidence specifically for differences on the BIC/2-log-Bayes-factor
    scale (their full scale: 0-2 negligible, 2-6 positive, 6-10 strong, >10
    very strong). Bibliographic details verified via Crossref and an
    independent secondary source quoting the primary table directly; the
    primary PDF itself did not render locally (poppler-utils unavailable) so
    this citation, unlike the fully primary-source-verified ones in this
    module's "Related work" docstring, rests one step removed from the
    original text -- flagged here rather than presented with equal
    confidence to those.

    On the real dataset this reveals that Gamma's non-homogeneous ("aging")
    conclusion on M1 and M2 -- the two machines behind the "Gamma finds
    spurious aging that Inverse Gaussian doesn't" contrast reported in
    Table 1 -- is AIC-specific: BIC's stricter penalty selects the stationary
    model for Gamma on both. Inverse Gaussian's and Wiener's age-model
    conclusions are unchanged under BIC on every machine. This does not
    affect the family-ranking conclusion (Inverse Gaussian remains the
    lowest-AIC and lowest-BIC family on every machine regardless, Table 1);
    only the age-model contrast specifically is criterion-dependent.

    Generalized to the full family registry (originally hardcoded to
    Gamma/IG/Wiener): every registered family present in
    ``comparison_result`` is included. The mechanical AIC-vs-BIC comparison
    (built from each family's own already-fitted stationary/non-homogeneous
    AIC/BIC pair, present in every family's fit dict regardless of what
    "stationary"/"non-homogeneous" concretely means for it) is meaningful
    as-is for every family that shares the original age-only power-law-vs-
    linear axis (Gamma, IG, Wiener, the three state-dependent/truncated
    variants, GIG); for random-effects Gamma/IG and multi-stage Wiener,
    which repurpose the identical field names for a DIFFERENT internal axis
    (random-effects-vs-pooled-exponent; two-stage-vs-one-stage), the
    mechanics still run correctly but "age_model_under_aic/bic" should be
    read as that family's own selected-vs-nested-null label, not literally
    "power law vs linear" -- each variant's own module docstring in
    degradation_process_variants.py already flags this "interface parity"
    relabeling explicitly.
    """
    family_keys = families.registered_with_params(comparison_result)
    result = {}
    for machine_key in sorted(comparison_result[families.param_key("gamma")]):
        machine_result = {}
        for family in family_keys:
            p = comparison_result[families.param_key(family)][machine_key]
            aic_gap = float(p["stationary_aic"]) - float(p["nonhomogeneous_aic"])
            bic_gap = float(p["stationary_bic"]) - float(p["nonhomogeneous_bic"])
            age_model_aic = "power_law" if aic_gap > 2.0 else "linear"
            age_model_bic = "power_law" if bic_gap > 2.0 else "linear"
            machine_result[family] = {
                "aic_gap_stationary_minus_nonhomogeneous": aic_gap,
                "bic_gap_stationary_minus_nonhomogeneous": bic_gap,
                "age_model_under_aic": age_model_aic,
                "age_model_under_bic": age_model_bic,
                "criterion_dependent": bool(age_model_aic != age_model_bic),
            }
        result[machine_key] = machine_result
        for family, r in machine_result.items():
            print(
                f"  {machine_key} {family}: AIC->{r['age_model_under_aic']} "
                f"(gap={r['aic_gap_stationary_minus_nonhomogeneous']:+.2f})  "
                f"BIC->{r['age_model_under_bic']} "
                f"(gap={r['bic_gap_stationary_minus_nonhomogeneous']:+.2f})  "
                f"criterion-dependent={r['criterion_dependent']}"
            )
    return result


# ── Per-tool outlier significance (individual vs. stationary, and vs. siblings) ─

def bootstrap_per_tool_power_significance(
    index: pd.DataFrame,
    *,
    window_size: int = 15,
    n_boot: int = 300,
    rng_seed: int = 0,
) -> dict:
    """Formal significance test for per-tool free (non-homogeneous) IG power.

    Table D4 flags each machine's most fleet-deviant tool by point estimate
    alone (e.g. Tool 2 on M1). With only 3 tools per machine, a maximum
    deviation tool always exists by construction -- point estimates alone
    cannot say whether that tool is a genuine outlier or just the largest of
    three noisy draws. This function answers that with two bootstrap tests,
    fit on each tool IN ISOLATION (not pooled with its machine-siblings):

      1. absolute: resample the tool's own wear windows `n_boot` times,
         refit a single-tool free non-homogeneous IG model each draw, and
         check whether the resulting 95% CI on that tool's own power
         excludes 1.0 (stationary).
      2. relative: using the same bootstrap draws, form the sampling
         distribution of (target tool's power - mean of its two
         machine-siblings' power) by pairing independently-drawn bootstrap
         samples (valid Monte-Carlo convolution of two independent
         streams), and check whether its 95% CI excludes 0. This is the
         test that actually answers "outlier or noise" -- test 1 answers a
         related but distinct question about whether any individual tool
         alone carries a significant aging signal.

    Both use `nonhomogeneous_components` (the always-free MLE power, not the
    AIC-gated selected one) so the bootstrap distribution isn't contaminated
    by the discrete NH-vs-stationary selection step.
    """
    windows = build_wear_windows(index, window_size=window_size)
    tool_to_machine = index.groupby("tool")["machine"].first().to_dict()

    machine_tools: dict[int, list[int]] = {}
    for tool_id, machine_id in tool_to_machine.items():
        machine_tools.setdefault(int(machine_id), []).append(int(tool_id))
    for tools in machine_tools.values():
        tools.sort()

    boot_powers: dict[int, np.ndarray] = {}
    per_tool: dict[int, dict] = {}
    for machine_id, tools in sorted(machine_tools.items()):
        data = windows[f"M{machine_id}"]
        for tool_id in tools:
            mask = data["tool"] == tool_id
            dv, t_start, t_end, tool = (
                data["dv"][mask], data["t_start"][mask], data["t_end"][mask], data["tool"][mask]
            )
            n = len(dv)
            rng = np.random.default_rng(rng_seed + 1000 * tool_id)
            powers = []
            for _ in range(n_boot):
                idx = rng.integers(0, n, size=n)
                try:
                    fitted = _fit_rounded_nh_ig(dv[idx], t_start[idx], t_end[idx], tool[idx])
                    powers.append(fitted["nonhomogeneous_components"][0]["power"])
                except Exception:
                    continue
            powers_arr = np.array(powers)
            boot_powers[tool_id] = powers_arr
            if len(powers_arr):
                ci_lo, ci_hi = (float(x) for x in np.percentile(powers_arr, [2.5, 97.5]))
                # Bootstrap two-sided p-value: double the smaller one-sided tail
                # fraction (standard construction), for use in the formal
                # multiple-comparisons summary alongside the relative test's
                # own p-value -- the CI-exclusion boolean above is equivalent
                # at the 5% level but a p-value is needed for FDR correction.
                p_two_sided = float(
                    2.0 * min(np.mean(powers_arr >= 1.0), np.mean(powers_arr <= 1.0))
                )
                p_two_sided = min(p_two_sided, 1.0)
            else:
                ci_lo, ci_hi, p_two_sided = float("nan"), float("nan"), float("nan")
            per_tool[tool_id] = {
                "machine": f"M{machine_id}", "n_windows": int(n), "n_boot_success": int(len(powers_arr)),
                "median_power": float(np.median(powers_arr)) if len(powers_arr) else float("nan"),
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "excludes_stationary": bool(len(powers_arr) and (ci_lo > 1.0 or ci_hi < 1.0)),
                "p_value_two_sided": p_two_sided,
            }

    relative: dict[int, dict] = {}
    rel_rng = np.random.default_rng(rng_seed + 999)
    for machine_id, tools in sorted(machine_tools.items()):
        for target in tools:
            siblings = [t for t in tools if t != target]
            lengths = [len(boot_powers[target])] + [len(boot_powers[s]) for s in siblings]
            n = min(lengths)
            if n == 0:
                continue
            t_draw = rel_rng.permutation(boot_powers[target])[:n]
            sib_draw = np.mean([rel_rng.permutation(boot_powers[s])[:n] for s in siblings], axis=0)
            diff = t_draw - sib_draw
            ci_lo, ci_hi = (float(x) for x in np.percentile(diff, [2.5, 97.5]))
            p_onesided = float(np.mean(diff >= 0))
            relative[target] = {
                "machine": f"M{machine_id}", "siblings": siblings,
                "median_diff": float(np.median(diff)), "ci_lo": ci_lo, "ci_hi": ci_hi,
                "excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
                "p_target_ge_siblings_onesided": p_onesided,
                "p_value_two_sided": min(2.0 * min(p_onesided, 1.0 - p_onesided), 1.0),
            }

    any_absolute = any(v["excludes_stationary"] for v in per_tool.values())
    any_relative = any(v["excludes_zero"] for v in relative.values())
    conclusion = (
        "At least one tool's individual power estimate is statistically distinguishable "
        "from stationarity or from its machine-siblings at the 95% level."
        if (any_absolute or any_relative) else
        "No individual tool's power estimate is statistically distinguishable from "
        "stationarity (1.0) or from its machine-siblings at the 95% level; the largest "
        "point-estimate deviations (Table D4) are consistent with sampling noise given "
        "only 3 tools per machine and should not be read as evidence of a genuine "
        "per-tool outlier."
    )

    return {
        "window_size": window_size, "n_boot": n_boot,
        "per_tool_absolute_vs_stationary": per_tool,
        "per_tool_relative_vs_siblings": relative,
        "any_tool_significant_absolute": any_absolute,
        "any_tool_significant_relative": any_relative,
        "conclusion": conclusion,
    }


def save_degradation_comparison(result: dict, output_dir: str | Path) -> Path:
    """Save the unified comparison (every registered family) to JSON.

    Written atomically (serialize to a same-directory temp file, then
    ``os.replace`` into place): Phase 1's fit alone takes on the order of
    tens of minutes for the full 10-family registry on real data, so this
    is the single highest-cost write in the whole pipeline to lose to an
    interrupted process (killed, crashed, power loss) leaving a truncated,
    unparseable JSON file in place of the last known-good artifact -- a
    plain ``open(path, "w")`` write is vulnerable to exactly that for the
    whole duration of the write, however brief; ``os.replace`` is a single
    atomic filesystem operation on both Windows and POSIX, so a reader
    always sees either the complete old file or the complete new one.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "degradation_comparison.json"

    def _default(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (bool, np.bool_)):
            return bool(o)
        return str(o)

    tmp_path = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=_default)
    os.replace(tmp_path, path)
    print(f"Saved degradation process comparison -> {path}")
    return path


# ── Family registry: the original three ─────────────────────────────────────
#
# Placed at the very end of the file, after every function it references
# (_ig_cdf, _wiener_cdf, _rounded_ig_bin_probability, _rounded_wiener_bin_probability,
# _fit_rounded_nh_ig, _fit_rounded_nh_wiener) is already defined -- register()
# calls run at import time (module load), top to bottom, so referencing a
# not-yet-defined name here would raise NameError immediately on import.
# _fit_rounded_nh_gamma, _rounded_gamma_bin_probability come from
# gamma_process (imported at the top of this file) and _gamma_cdf is defined
# locally just below, purely so it exists as a standalone callable (Gamma's
# own fitter/validator build their CDF as an inline closure instead, which
# works fine for their own internal use but isn't something the registry can
# call directly by name).
#
# None of the three original fitters are modified by this registration --
# each is wrapped in a thin adapter with the uniform
# (dv, t_start, t_end, tool_ids, v_start=None) -> dict signature every
# family in the registry shares (v_start ignored, none of these three
# depend on wear level); their own golden-value-tested behavior is
# untouched.

def _gamma_cdf_for_registry(x: np.ndarray, shape_s: np.ndarray, beta) -> np.ndarray:
    return gamma_dist.cdf(np.maximum(0.0, x), a=shape_s, scale=1.0 / beta)


def _fit_gamma_adapter(dv, t_start, t_end, tool_ids, v_start=None, **kw):
    return _fit_rounded_nh_gamma(dv, t_start, t_end, tool_ids, **kw)


def _fit_ig_adapter(dv, t_start, t_end, tool_ids, v_start=None, **kw):
    return _fit_rounded_nh_ig(dv, t_start, t_end, tool_ids, **kw)


def _fit_wiener_adapter(dv, t_start, t_end, tool_ids, v_start=None, **kw):
    return _fit_rounded_nh_wiener(dv, t_start, t_end, tool_ids, **kw)


families.register(
    "gamma",
    label="Gamma",
    color="#0072B2", ls="-", marker="o",
    fit_fn=_fit_gamma_adapter,
    shape_fn=families.standard_shape_fn,
    predict_shape_fn=families.standard_predict_shape_fn,
    cdf_fn=_gamma_cdf_for_registry,
    bin_prob_fn=_rounded_gamma_bin_probability,
    rvs_fn=lambda shape_s, beta, rng: gamma_dist.rvs(a=shape_s, scale=1.0 / beta, random_state=rng),
    is_subordinator=True,
    baseline_key=None,
    extra_params_desc="original family",
)

families.register(
    "inverse_gaussian",
    label="Inverse Gaussian",
    color="#D55E00", ls="--", marker="s",
    fit_fn=_fit_ig_adapter,
    shape_fn=families.standard_shape_fn,
    predict_shape_fn=families.standard_predict_shape_fn,
    cdf_fn=_ig_cdf,
    bin_prob_fn=_rounded_ig_bin_probability,
    rvs_fn=lambda shape_s, beta, rng: invgauss.rvs(mu=1.0 / shape_s, scale=(shape_s ** 2) / beta, random_state=rng),
    is_subordinator=True,
    baseline_key=None,
    extra_params_desc="original family",
)

families.register(
    "wiener",
    label="Wiener",
    color="#009E73", ls=":", marker="^",
    fit_fn=_fit_wiener_adapter,
    shape_fn=families.standard_shape_fn,
    predict_shape_fn=families.standard_predict_shape_fn,
    cdf_fn=_wiener_cdf,
    bin_prob_fn=_rounded_wiener_bin_probability,
    # rvs_fn draws a single WINDOW-level increment from the marginal
    # Normal(shape_s/beta, sqrt(shape_s)/beta) -- always well-defined
    # regardless of monotonicity, and what model_recovery_study's
    # _simulate_synthetic_dv needs (one synthetic window at a time, no
    # accumulation). This is a DIFFERENT question from whether repeated,
    # ACCUMULATING draws are valid for cumulative-trajectory MTTFF
    # simulation, which is what is_subordinator=False actually gates
    # (Wiener's non-monotone paths make step-accumulation with the
    # existing zero-clipping logic in degradation_reliability_translation.py
    # inconsistent with how this family is fit everywhere else -- see that
    # module's docstring) -- MTTFF uses the exact closed-form first-passage-
    # time route instead, never this rvs_fn.
    rvs_fn=lambda shape_s, beta, rng: norm.rvs(loc=shape_s / beta, scale=np.sqrt(shape_s) / beta, random_state=rng),
    is_subordinator=False,
    baseline_key=None,
    extra_params_desc="original family",
)

# Imported here, deliberately AFTER the three original families register
# above -- not for correctness (registration order doesn't affect any
# lookup), but so degradation_families.FAMILY_REGISTRY's insertion order
# (used as the canonical display/iteration order throughout, per that
# module's docstring) puts Gamma/Inverse-Gaussian/Wiener first and each
# variant after, matching how the manuscript and every table/figure
# presents them, rather than variants-before-originals. Otherwise purely a
# module-level side effect (registers the seven variants); not referenced
# by name below.
import degradation_process_variants as _degradation_process_variants  # noqa: F401,E402


# ── Registry-derived dispatch dicts (all 10 families, once populated) ──────
#
# Now that every family has registered (the three originals above, the
# seven variants via the import just above), build the dicts the mid-file
# functions above already iterate generically. See this file's
# leave-one-tool-out comment block for why these are safe to define here,
# after those functions, and why _BIN_PROB_FNS specifically cannot be
# called with just (observed, shape_s, beta, resolution_um) for every
# family (GIG needs its p_gid appended -- families.extra_shape_values
# supplies it, empty for every other family).
_FIT_FNS = {key: spec["fit_fn"] for key, spec in families.FAMILY_REGISTRY.items()}
_BIN_PROB_FNS = {key: spec["bin_prob_fn"] for key, spec in families.FAMILY_REGISTRY.items()}

# Precision/cost tradeoff kwargs for families whose fit cost is dominated by
# one configurable knob (multi-stage Wiener's change-point grid resolution;
# random-effects Gamma/IG's Gauss-Hermite quadrature node count), applied
# ONLY inside expensive repeated-refit sweeps (model recovery, leave-one-
# tool-out CV, window-size robustness) that refit every family many times
# over -- never for the primary real-data fit in
# fit_and_compare_degradation_models, which uses each family's full-
# precision default. Centralized here (rather than duplicated as a
# per-function if/elif) so every sweep that needs this applies the same
# coarsening the same way; a sweep missing it entirely is exactly the kind
# of thing that turned into an hours-long stuck real-data run before this
# fix (random-effects' quadrature-multiplied cost, ~250s for a single
# real-data machine-level fit at full precision, refit 9-24+ times over
# inside these sweeps without coarsening).
_REPEATED_REFIT_COARSENING: dict[str, dict] = {
    "wiener_multistage": {"n_grid": 8},
    "gamma_random_effects": {"n_quad_nodes": 7},
    "inverse_gaussian_random_effects": {"n_quad_nodes": 7},
}
