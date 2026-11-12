"""Degradation-process variants beyond the original Gamma/Inverse Gaussian/Wiener
comparison in ``gamma_process.py`` and ``degradation_process_models.py``.

Seven variants are registered here, six in three pairs sharing implementation
where the underlying math is genuinely shared, plus one further family
(Generalized Inverse Gaussian) that directly nests the existing Inverse
Gaussian family rather than pairing with a second one:

  * State-dependent Gamma & Inverse Gaussian -- one shared fitter
    (``_fit_rounded_nh_state_dependent``), parametrized by which baseline
    family's bin-probability function it uses. Extends the age-only shape
    quantity S(s,t) with a wear-*level* feedback term, one new parameter
    shared per machine (mirroring how beta is already shared, rather than
    adding a free parameter per tool), nesting the baseline family exactly
    at that parameter's null value. Inspired by, but not a literal
    reproduction of, the transformed/state-dependent Inverse Gaussian
    literature (Peng, Zhu & Shen, 2019; Das, Naikan & Panja, 2024 -- the
    latter already discussed in the manuscript's Table 1) -- Peng et al.'s
    exact closed form could not be verified against a primary source
    (paywalled; only secondary summaries were accessible), so this is
    presented as its own self-contained, honestly-scoped extension rather
    than a claimed reproduction.

  * Truncated (non-negative) Wiener -- left-truncates the increment
    distribution at zero instead of leaving it an unrestricted Normal,
    directly and completely eliminating the physically-impossible-negative-
    wear problem by construction. Same parameter count as plain Wiener (not
    a nested add-a-parameter variant -- a different distributional
    assumption compared via the same AIC/BIC/Vuong machinery used for the
    original 3-family comparison). In the spirit of, not a literal
    reproduction of, Pan et al. (2017)'s truncated-normal Wiener treatment
    (which truncates the drift parameter's population distribution across
    units -- a random-effects construction -- not the increment
    distribution itself; see the same paywall caveat above).

  * Random-effects Gamma & Inverse Gaussian -- one shared fitter
    (``_fit_rounded_nh_random_effects``), scoped to the aging exponent
    specifically (not the rate eta): p_j ~ Normal(mu_p, sigma_p^2) across a
    machine's tools, integrated out of the likelihood via Gauss-Hermite
    quadrature rather than fit as 3 independent free parameters. Directly
    answers the question the manuscript's own Section 7 poses as future
    work ("a hierarchical, random-effects extension... could narrow these
    intervals, but was out of scope here"). With only 3 tools per machine,
    sigma_p is expected to be poorly identified -- this is reported
    honestly if so, not smoothed over.

  * Multi-stage Wiener -- a 2-stage (not 3-stage) piecewise extension,
    scoped to what Figure 1 actually shows (fast run-in, then slower
    steady wear -- not a third late-life acceleration phase): same aging
    exponent across both stages, separate per-tool rate for each stage, one
    change-point time shared per machine, fit by profile likelihood over a
    change-point grid (the standard, numerically safe way to handle a
    discontinuous-derivative change-point parameter). Nests plain Wiener
    when both stages' rates coincide.

  * Generalized Inverse Gaussian (GIG) -- a 3-parameter family that
    mathematically CONTAINS both Gamma and Inverse Gaussian as special
    cases (Gamma as a boundary limit; Inverse Gaussian as the exact
    p=-1/2 slice, for any of its other two parameters). Rather than
    re-deriving a moment-matched reparametrization from scratch, this
    variant is built directly on top of the study's own Inverse Gaussian
    convention: at its one new parameter (p_gid) fixed to exactly -1/2, it
    reduces EXACTLY to the study's existing Inverse Gaussian family
    (verified to 1e-14 against the production invgauss CDF, not just
    asserted -- see the GIG section below), then p_gid is freed as a
    single shared-per-machine parameter to ask a focused, cleanly nested
    question the paper's existing non-nested Gamma-vs-IG comparison cannot
    ask on its own: does relaxing Inverse Gaussian's own implicit
    tail-shape assumption, via a proper likelihood-ratio test rather than
    an AIC/BIC heuristic, improve the fit -- i.e. is Inverse Gaussian's
    specific tail shape itself well-supported by the data, not just its
    practical performance against Gamma and Wiener. A closed-form
    correction (derived and independently verified numerically below)
    keeps the fitted mean pinned at the study's shared S(s,t)/beta
    convention at every p_gid -- variance and higher moments are
    intentionally NOT held fixed, since fixing them too would need an
    expensive per-window numerical root-find instead of a closed form;
    this scope (mean-only invariance) is stated explicitly rather than
    overclaimed as full moment-matching.

Every fitter here mirrors the structural template in
``gamma_process._fit_rounded_nh_gamma`` (validated, golden-value-tested,
never modified by this module): per-tool components, log-space Nelder-Mead
with multiple restarts, a nested stationary-vs-non-homogeneous AIC
selection. Only the shape quantity and/or the bin-probability distribution
differ, family by family.

See ``degradation_families.py``'s module docstring for the registry
contract and the (deliberately acyclic) import structure this module is
part of.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import gamma as gamma_dist
from scipy.stats import invgauss, norm, geninvgauss
from scipy.special import logsumexp, kv, ndtr, ndtri
from numpy.polynomial.hermite_e import hermegauss

import degradation_families as families
from gamma_process import (
    ROUNDING_OFFSETS_UM, ROUNDING_WEIGHTS, TIME_REFERENCE_S, WEAR_RESOLUTION_UM,
)

EOL_THRESHOLD_UM = families.EOL_THRESHOLD_UM


# ── Shared low-level helpers ────────────────────────────────────────────────

def _validate_windows(dv, t_start, t_end, tool_ids, *extra) -> None:
    arrays = [dv, t_start, t_end, tool_ids, *extra]
    if not (all(a.ndim == 1 for a in arrays) and len(set(len(a) for a in arrays)) == 1 and len(dv) > 0):
        raise ValueError("Wear-window arrays must be equal-length non-empty vectors.")
    if np.any(dv < 0) or np.any(t_start < 0) or np.any(t_end <= t_start):
        raise ValueError("Invalid wear windows.")


def _init_eta_power(dv, t_start, t_end, tool_ids, tools, t_ref, beta0=0.5, power0_val=0.8):
    n_tools = len(tools)
    power0 = np.full(n_tools, power0_val)
    eta0 = np.empty(n_tools, float)
    for i, tool in enumerate(tools):
        mask = tool_ids == tool
        total_wear = max(float(np.sum(dv[mask])), 1.0)
        terminal_scaled = max(float(np.max(t_end[mask])) / t_ref, 1e-6)
        eta0[i] = max(beta0 * total_wear / terminal_scaled ** power0[i], 1e-4)
    return eta0, power0


# ── Truncated (non-negative) Wiener ─────────────────────────────────────────
#
# CDF/rvs are hand-derived closed forms using scipy.special.ndtr/ndtri (a
# fast, properly vectorized standard-normal CDF/inverse-CDF) rather than
# scipy.stats.truncnorm.cdf/.rvs directly. This was NOT a stylistic choice:
# truncnorm's generic rv_continuous machinery is independently documented
# (upstream scipy issues) to be extremely slow relative to a closed-form
# implementation, and was diagnosed here as a genuine performance bug --
# profiling a single real-data fit (M1, 122 windows) showed gamma/
# inverse_gaussian/wiener each completing in ~5-11s while truncated Wiener
# had not finished after several minutes, the same order-of-magnitude
# regression independently found and fixed for GIG's geninvgauss earlier
# this session (514s -> 34.4s). generic_bin_prob_from_cdf calls this CDF
# twice per rounding-offset node (11 nodes) per window, and Nelder-Mead
# calls the full likelihood hundreds to thousands of times per fit, so a
# slow per-call CDF compounds into minutes-to-hours, not seconds.
#
# For Normal(mu, sigma) left-truncated at 0: with alpha = -mu/sigma (the
# truncation point in standardized units) and Phi = the standard normal
# CDF, F(x) = [Phi((x-mu)/sigma) - Phi(alpha)] / [1 - Phi(alpha)] for x>=0.
# Verified to agree with scipy.stats.truncnorm.cdf to float precision (see
# tests/test_degradation_process_variants.py).

def _truncated_wiener_cdf(x: np.ndarray, shape_s: np.ndarray, beta) -> np.ndarray:
    x = np.asarray(x, float)
    mu = shape_s / beta
    sigma = np.sqrt(shape_s) / beta
    alpha = -mu / sigma
    phi_alpha = ndtr(alpha)
    z = (np.maximum(x, 0.0) - mu) / sigma
    return (ndtr(z) - phi_alpha) / (1.0 - phi_alpha)


_rounded_truncated_wiener_bin_probability = families.generic_bin_prob_from_cdf(_truncated_wiener_cdf)


def _truncated_wiener_mean_increment(shape_s: np.ndarray, beta) -> np.ndarray:
    """TRUE mean of Normal(mu, sigma) left-truncated at 0 -- NOT the
    underlying untruncated mu=shape_s/beta every other family's mean
    reduces to (``degradation_families.standard_mean_increment``).
    Truncating away the discarded negative tail shifts the mean UP by
    construction; this is the entire reason this family exists, so a
    diagnostic using the naive shape_s/beta formula here would
    systematically and substantially under-predict its actual mean wear
    (confirmed: ~17um naive vs. ~31um true for a real fitted component --
    see this function's registration for the concrete incident this fixed).
    mean = mu + sigma * phi(alpha) / (1 - Phi(alpha)), alpha = -mu/sigma,
    phi/Phi the standard normal pdf/cdf.
    """
    mu = shape_s / beta
    sigma = np.sqrt(shape_s) / beta
    alpha = -mu / sigma
    phi_alpha = np.exp(-0.5 * alpha ** 2) / np.sqrt(2.0 * np.pi)
    denom = np.maximum(1.0 - ndtr(alpha), 1e-300)
    return mu + sigma * phi_alpha / denom


def _truncated_wiener_rvs(shape_s: np.ndarray, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Inverse-CDF sampling: Phi(alpha) + U*(1 - Phi(alpha)) is uniform on
    the truncated CDF's own range, so its ndtri (standard-normal inverse
    CDF) maps back to a truncated draw -- the standard closed-form technique
    for sampling a truncated normal without rejection sampling."""
    mu = shape_s / beta
    sigma = np.sqrt(shape_s) / beta
    alpha = -mu / sigma
    phi_alpha = ndtr(alpha)
    u = rng.random(np.shape(shape_s))
    z = ndtri(phi_alpha + u * (1.0 - phi_alpha))
    return mu + sigma * z


def _fit_rounded_nh_truncated_wiener(
    dv: np.ndarray, t_start: np.ndarray, t_end: np.ndarray, tool_ids: np.ndarray,
    v_start: np.ndarray | None = None, *,
    time_reference_s: float = TIME_REFERENCE_S, resolution_um: float = WEAR_RESOLUTION_UM,
) -> dict:
    """Fit a tool-heterogeneous non-homogeneous truncated-Wiener process.
    Mirrors ``degradation_process_models._fit_rounded_nh_wiener`` exactly
    (same per-tool eta/power, shared beta, log-space Nelder-Mead with 3
    restarts, non-homogeneous-vs-stationary AIC selection); only the
    bin-probability function differs (left-truncated Normal instead of
    unrestricted Normal). ``v_start`` is accepted for interface uniformity
    and ignored -- this family's rate does not depend on wear level.
    """
    dv = np.asarray(dv, float); t_start = np.asarray(t_start, float); t_end = np.asarray(t_end, float)
    tool_ids = np.asarray(tool_ids, int)
    _validate_windows(dv, t_start, t_end, tool_ids)

    tools = np.unique(tool_ids)
    tool_col = {int(tool): i for i, tool in enumerate(tools)}
    component_idx = np.array([tool_col[int(tool)] for tool in tool_ids], int)
    n_tools = len(tools)
    t_ref = float(time_reference_s)
    eta0, power0 = _init_eta_power(dv, t_start, t_end, tool_ids, tools, t_ref)
    beta0 = 0.5

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
            np.power(t_end / t_ref, power[component_idx]) - np.power(t_start / t_ref, power[component_idx])
        )
        if np.any(shape_s <= 0) or not np.all(np.isfinite(shape_s)):
            return 1e100
        prob = _rounded_truncated_wiener_bin_probability(dv, shape_s, beta, resolution_um)
        return -float(np.sum(np.log(prob)))

    starts = [
        np.log(np.r_[eta0, power0, beta0]),
        np.log(np.r_[eta0 * 0.7, np.full(n_tools, 0.6), 0.4]),
        np.log(np.r_[eta0 * 1.3, np.full(n_tools, 1.0), 0.7]),
    ]
    best = None
    for x0 in starts:
        res = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 100000, "xatol": 1e-9, "fatol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    if best is None or not np.isfinite(best.fun):
        raise RuntimeError("Non-homogeneous rounded truncated-Wiener MLE failed.")

    eta = np.exp(best.x[:n_tools]); power = np.exp(best.x[n_tools:2 * n_tools]); beta = float(np.exp(best.x[-1]))
    stationary_start = np.log(np.r_[eta, beta])
    stationary = minimize(
        lambda z: nll(z, fixed_power=True), stationary_start, method="Nelder-Mead",
        options={"maxiter": 50000, "xatol": 1e-9, "fatol": 1e-9},
    )

    n = len(dv); k_nh = 2 * n_tools + 1; k_stationary = n_tools + 1
    log_lik = -float(best.fun); stationary_log_lik = -float(stationary.fun)
    nh_components = [
        {"tool_id": int(tool), "eta": float(eta[i]), "power": float(power[i]), "beta": beta, "time_reference_s": t_ref}
        for i, tool in enumerate(tools)
    ]
    stationary_eta = np.exp(stationary.x[:n_tools]); stationary_beta = float(np.exp(stationary.x[n_tools]))
    stationary_components = [
        {"tool_id": int(tool), "eta": float(stationary_eta[i]), "power": 1.0, "beta": stationary_beta, "time_reference_s": t_ref}
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
        "model": "nonhomogeneous_truncated_wiener_rounded",
        "schema_version": 1,
        "selected_age_model": "power_law" if use_nonhomogeneous else "linear",
        "beta": selected_beta,
        "eta": float(np.median([c["eta"] for c in selected_components])),
        "power": float(np.median([c["power"] for c in selected_components])),
        "time_reference_s": t_ref, "wear_resolution_um": float(resolution_um),
        "components": selected_components, "n_windows": int(n), "n_zero_windows": int(np.sum(dv == 0)),
        "log_lik": selected_log_lik,
        "aic": float(2 * selected_k - 2 * selected_log_lik),
        "bic": float(selected_k * np.log(n) - 2 * selected_log_lik),
        "nonhomogeneous_aic": nh_aic, "nonhomogeneous_bic": float(k_nh * np.log(n) - 2 * log_lik),
        "stationary_aic": stationary_aic, "stationary_bic": float(k_stationary * np.log(n) - 2 * stationary_log_lik),
        "nonhomogeneous_components": nh_components,
        # Zero by construction (not a fitted diagnostic like plain Wiener's
        # mean_p_negative_increment) -- kept as a field so downstream code
        # that reads this key for any Wiener-family model does not need a
        # special case for the truncated variant.
        "mean_p_negative_increment": 0.0, "max_p_negative_increment": 0.0,
    }


def _fit_truncated_wiener_adapter(dv, t_start, t_end, tool_ids, v_start=None, **kw):
    return _fit_rounded_nh_truncated_wiener(dv, t_start, t_end, tool_ids, v_start, **kw)


families.register(
    "wiener_truncated",
    label="Truncated Wiener",
    color="#009E73", ls=":", marker="v",
    fit_fn=_fit_truncated_wiener_adapter,
    shape_fn=families.standard_shape_fn,
    predict_shape_fn=families.standard_predict_shape_fn,
    cdf_fn=_truncated_wiener_cdf,
    bin_prob_fn=_rounded_truncated_wiener_bin_probability,
    rvs_fn=_truncated_wiener_rvs,
    mean_increment_fn=_truncated_wiener_mean_increment,
    is_subordinator=True,
    baseline_key=None,
    extra_params_desc="same parameter count as plain Wiener; left-truncated increment distribution (zero negative-increment mass by construction) instead of an unrestricted Normal",
)


# ── State-dependent Gamma & Inverse Gaussian (shared fitter) ───────────────
#
# Gamma's and Inverse Gaussian's own CDFs are needed here independently of
# gamma_process.py / degradation_process_models.py (rather than imported
# from there) specifically to keep this module's import graph acyclic:
# degradation_process_models.py imports THIS module (to trigger its
# registrations), so this module cannot import back from it. These two
# closures are the exact same formulas as
# gamma_process._rounded_gamma_bin_probability and
# degradation_process_models._rounded_ig_bin_probability -- equivalence is
# checked directly in tests/test_degradation_process_variants.py rather than
# just asserted here.

def _gamma_cdf(x: np.ndarray, shape_s: np.ndarray, beta) -> np.ndarray:
    return gamma_dist.cdf(np.maximum(0.0, x), a=shape_s, scale=1.0 / beta)


def _ig_cdf(x: np.ndarray, shape_s: np.ndarray, beta) -> np.ndarray:
    x = np.maximum(x, 1e-9)
    return invgauss.cdf(x, mu=1.0 / shape_s, loc=0.0, scale=(shape_s ** 2) / beta)


_rounded_gamma_bin_probability_for_variants = families.generic_bin_prob_from_cdf(_gamma_cdf)
_rounded_ig_bin_probability_for_variants = families.generic_bin_prob_from_cdf(_ig_cdf)


def _state_dependent_shape_fn(component: dict, t_start: np.ndarray, t_end: np.ndarray, v_start: np.ndarray | None = None) -> np.ndarray:
    if v_start is None:
        raise ValueError("State-dependent families require v_start (wear level at each window's start).")
    eta = float(component["eta"]); power = float(component["power"]); t_ref = float(component["time_reference_s"])
    gamma_state = float(component["gamma_state"])
    base = eta * (np.power(t_end / t_ref, power) - np.power(t_start / t_ref, power))
    state_factor = 1.0 + gamma_state * (np.asarray(v_start, float) / EOL_THRESHOLD_UM)
    return np.maximum(base * state_factor, 1e-12)


def _state_dependent_predict_shape_fn(fitted: dict, t_start: np.ndarray, t_end: np.ndarray, v_start: np.ndarray | None = None) -> np.ndarray:
    if v_start is None:
        raise ValueError("State-dependent families require v_start.")
    eta_med = float(np.median([c["eta"] for c in fitted["components"]]))
    power_med = float(np.median([c["power"] for c in fitted["components"]]))
    gamma_med = float(np.median([c["gamma_state"] for c in fitted["components"]]))
    t_ref = float(fitted["time_reference_s"])
    base = eta_med * (np.power(t_end / t_ref, power_med) - np.power(t_start / t_ref, power_med))
    state_factor = 1.0 + gamma_med * (np.asarray(v_start, float) / EOL_THRESHOLD_UM)
    return np.maximum(base * state_factor, 1e-12)


def _fit_rounded_nh_state_dependent(
    dv: np.ndarray, t_start: np.ndarray, t_end: np.ndarray, tool_ids: np.ndarray, v_start: np.ndarray,
    bin_prob_fn, model_name: str,
    *, time_reference_s: float = TIME_REFERENCE_S, resolution_um: float = WEAR_RESOLUTION_UM,
) -> dict:
    """Shared fitter for the state-dependent Gamma and Inverse Gaussian
    variants. Identical in every other respect to the baseline family's own
    fitter (``bin_prob_fn`` selects which): the shape quantity gains one
    multiplicative term, ``1 + gamma_state * v_start/EOL_THRESHOLD_UM``,
    with ``gamma_state`` a single SHARED-per-machine parameter (mirroring
    how beta is already shared across a machine's tools, rather than
    adding 3 more free parameters) -- at gamma_state=0 this reduces exactly
    to the baseline family, a clean nested comparison
    (``degradation_families.nested_model_test``).
    """
    dv = np.asarray(dv, float); t_start = np.asarray(t_start, float); t_end = np.asarray(t_end, float)
    tool_ids = np.asarray(tool_ids, int); v_start = np.asarray(v_start, float)
    _validate_windows(dv, t_start, t_end, tool_ids, v_start)
    if np.any(v_start < 0):
        raise ValueError("Invalid wear windows.")

    tools = np.unique(tool_ids)
    tool_col = {int(tool): i for i, tool in enumerate(tools)}
    component_idx = np.array([tool_col[int(tool)] for tool in tool_ids], int)
    n_tools = len(tools)
    t_ref = float(time_reference_s)
    eta0, power0 = _init_eta_power(dv, t_start, t_end, tool_ids, tools, t_ref)
    beta0 = 0.5
    v_frac = v_start / EOL_THRESHOLD_UM

    # Parameter vector, free-power case: [log(eta) x n_tools, log(power) x
    # n_tools, log(beta), gamma_state]. gamma_state is fit on its natural
    # (not log) scale since it may be negative (decelerating feedback) as
    # well as positive (accelerating feedback), unlike every other
    # parameter here which is constrained positive.
    def unpack(z, fixed_power: bool):
        eta = np.exp(z[:n_tools])
        if fixed_power:
            power = np.ones(n_tools, float)
            beta = float(np.exp(z[n_tools]))
            gamma_state = float(z[n_tools + 1])
        else:
            power = np.exp(z[n_tools:2 * n_tools])
            beta = float(np.exp(z[-2]))
            gamma_state = float(z[-1])
        return eta, power, beta, gamma_state

    def nll(z, fixed_power: bool = False) -> float:
        eta, power, beta, gamma_state = unpack(z, fixed_power)
        if (
            np.any(power < 0.05) or np.any(power > 3.0) or beta < 1e-4 or beta > 100.0
            or abs(gamma_state) > 50.0
        ):
            return 1e100
        base = eta[component_idx] * (
            np.power(t_end / t_ref, power[component_idx]) - np.power(t_start / t_ref, power[component_idx])
        )
        state_factor = 1.0 + gamma_state * v_frac
        if np.any(state_factor <= 0.01):
            return 1e100
        shape_s = base * state_factor
        if np.any(shape_s <= 0) or not np.all(np.isfinite(shape_s)):
            return 1e100
        prob = bin_prob_fn(dv, shape_s, beta, resolution_um)
        return -float(np.sum(np.log(prob)))

    starts = [
        np.r_[np.log(eta0), np.log(power0), np.log(beta0), 0.0],
        np.r_[np.log(eta0 * 0.7), np.log(np.full(n_tools, 0.6)), np.log(0.4), 0.1],
        np.r_[np.log(eta0 * 1.3), np.log(np.full(n_tools, 1.0)), np.log(0.7), -0.1],
    ]
    best = None
    for x0 in starts:
        res = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 150000, "xatol": 1e-9, "fatol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    if best is None or not np.isfinite(best.fun):
        raise RuntimeError(f"Non-homogeneous rounded {model_name} MLE failed.")

    eta, power, beta, gamma_state = unpack(best.x, fixed_power=False)
    stationary_start = np.r_[np.log(eta), np.log(beta), gamma_state]
    stationary = minimize(
        lambda z: nll(z, fixed_power=True), stationary_start, method="Nelder-Mead",
        options={"maxiter": 80000, "xatol": 1e-9, "fatol": 1e-9},
    )

    n = len(dv)
    k_nh = 2 * n_tools + 2       # baseline family's 2*n_tools+1, plus gamma_state
    k_stationary = n_tools + 2   # baseline family's n_tools+1, plus gamma_state
    log_lik = -float(best.fun); stationary_log_lik = -float(stationary.fun)
    nh_components = [
        {"tool_id": int(tool), "eta": float(eta[i]), "power": float(power[i]), "beta": beta,
         "gamma_state": gamma_state, "time_reference_s": t_ref}
        for i, tool in enumerate(tools)
    ]
    stat_eta, _stat_power, stat_beta, stat_gamma_state = unpack(stationary.x, fixed_power=True)
    stationary_components = [
        {"tool_id": int(tool), "eta": float(stat_eta[i]), "power": 1.0, "beta": stat_beta,
         "gamma_state": stat_gamma_state, "time_reference_s": t_ref}
        for i, tool in enumerate(tools)
    ]
    nh_aic = float(2 * k_nh - 2 * log_lik)
    stationary_aic = float(2 * k_stationary - 2 * stationary_log_lik)
    use_nonhomogeneous = bool(stationary_aic - nh_aic > 2.0)
    selected_components = nh_components if use_nonhomogeneous else stationary_components
    selected_beta = beta if use_nonhomogeneous else stat_beta
    selected_log_lik = log_lik if use_nonhomogeneous else stationary_log_lik
    selected_k = k_nh if use_nonhomogeneous else k_stationary
    selected_gamma_state = gamma_state if use_nonhomogeneous else stat_gamma_state
    return {
        "model": model_name, "schema_version": 1,
        "selected_age_model": "power_law" if use_nonhomogeneous else "linear",
        "beta": selected_beta,
        "eta": float(np.median([c["eta"] for c in selected_components])),
        "power": float(np.median([c["power"] for c in selected_components])),
        "gamma_state": float(selected_gamma_state),
        "time_reference_s": t_ref, "wear_resolution_um": float(resolution_um),
        "components": selected_components, "n_windows": int(n), "n_zero_windows": int(np.sum(dv == 0)),
        "log_lik": selected_log_lik,
        "aic": float(2 * selected_k - 2 * selected_log_lik),
        "bic": float(selected_k * np.log(n) - 2 * selected_log_lik),
        "nonhomogeneous_aic": nh_aic, "nonhomogeneous_bic": float(k_nh * np.log(n) - 2 * log_lik),
        "stationary_aic": stationary_aic, "stationary_bic": float(k_stationary * np.log(n) - 2 * stationary_log_lik),
        "nonhomogeneous_components": nh_components,
    }


def _fit_gamma_state_adapter(dv, t_start, t_end, tool_ids, v_start=None, **kw):
    if v_start is None:
        raise ValueError("state-dependent fitting requires v_start.")
    return _fit_rounded_nh_state_dependent(
        dv, t_start, t_end, tool_ids, v_start,
        bin_prob_fn=_rounded_gamma_bin_probability_for_variants, model_name="state_dependent_gamma_rounded", **kw,
    )


def _fit_ig_state_adapter(dv, t_start, t_end, tool_ids, v_start=None, **kw):
    if v_start is None:
        raise ValueError("state-dependent fitting requires v_start.")
    return _fit_rounded_nh_state_dependent(
        dv, t_start, t_end, tool_ids, v_start,
        bin_prob_fn=_rounded_ig_bin_probability_for_variants, model_name="state_dependent_inverse_gaussian_rounded", **kw,
    )


families.register(
    "gamma_state",
    label="State-dependent Gamma",
    color="#0072B2", ls="-.", marker="D",
    fit_fn=_fit_gamma_state_adapter,
    shape_fn=_state_dependent_shape_fn,
    predict_shape_fn=_state_dependent_predict_shape_fn,
    cdf_fn=_gamma_cdf,
    bin_prob_fn=_rounded_gamma_bin_probability_for_variants,
    rvs_fn=lambda shape_s, beta, rng: gamma_dist.rvs(a=shape_s, scale=1.0 / beta, random_state=rng),
    is_subordinator=True,
    baseline_key="gamma",
    extra_params_desc="+1 shared gamma_state per machine (wear-level feedback on the shape function; nests plain Gamma at gamma_state=0)",
)

families.register(
    "inverse_gaussian_state",
    label="State-dependent Inverse Gaussian",
    color="#D55E00", ls="-.", marker="P",
    fit_fn=_fit_ig_state_adapter,
    shape_fn=_state_dependent_shape_fn,
    predict_shape_fn=_state_dependent_predict_shape_fn,
    cdf_fn=_ig_cdf,
    bin_prob_fn=_rounded_ig_bin_probability_for_variants,
    rvs_fn=lambda shape_s, beta, rng: invgauss.rvs(mu=1.0 / shape_s, scale=(shape_s ** 2) / beta, random_state=rng),
    is_subordinator=True,
    baseline_key="inverse_gaussian",
    extra_params_desc="+1 shared gamma_state per machine (wear-level feedback on the shape function; nests plain Inverse Gaussian at gamma_state=0)",
)


# ── Random-effects Gamma & Inverse Gaussian (shared fitter) ────────────────
#
# Scoped to the aging exponent specifically, not the rate eta: Table 5 (the
# manuscript's own per-tool significance analysis) and Section 7's stated
# open question ("a hierarchical, random-effects extension... could narrow
# these intervals") are both about the aging exponent, not the rate. Treats
# p_j ~ Normal(mu_p, sigma_p^2) across a machine's 3 tools and integrates it
# out of each tool's likelihood via fixed-node Gauss-Hermite quadrature
# (probabilists' Hermite polynomials -- numpy.polynomial.hermite_e.hermegauss
# -- are orthogonal w.r.t. exp(-x^2/2), which is exactly proportional to the
# standard normal density, so no rescaling is needed beyond the
# 1/sqrt(2*pi) normalizing constant folded into _GH_LOG_WEIGHTS below), kept
# frequentist/MLE-based like every other family in this study rather than
# introducing Bayesian/MCMC machinery as a one-off.
#
# This variant's own internal model-selection axis is NOT the baseline
# families' stationary-vs-non-homogeneous (power=1-vs-free) choice -- it is
# whether the aging exponent varies randomly across tools at all
# (sigma_p > 0, "random_effects") or is better described as one exponent
# shared by every tool on the machine but still freely estimated
# (sigma_p = 0, "pooled_exponent"; NOT the same as the baseline's power=1
# "linear" case). Because the null sits at a variance component's zero
# boundary, the ordinary chi-square(1) likelihood-ratio asymptotic does not
# apply; ``degradation_families.nested_model_test(..., boundary_corrected=True)``
# applies the standard fix (Self & Liang, 1987): the correct p-value is
# exactly half the naive chi-square(1) p-value.
#
# EXPECTATION, set here rather than only after the fact: with only 3 tools
# per machine, sigma_p is a variance component estimated from n=3 groups --
# expect it to be poorly identified (a flat or boundary-favoring profile
# likelihood). An honestly-reported "cannot distinguish random-effects from
# pooled/fixed-effects with n=3" is still a real, useful answer to the
# manuscript's own open question, not a failure of this implementation.
#
# Downstream code (Vuong's test, leave-one-tool-out CV, PIT-KS validation,
# plotting) needs no special-casing for this variant: each tool's posterior
# mean aging exponent E[p_j | data_j] (a standard empirical-Bayes/BLUP-style
# plug-in, computed from the same quadrature nodes used to fit the model) is
# stored in its component dict under the ordinary "power" key, so
# degradation_families.standard_shape_fn / standard_predict_shape_fn --
# unmodified -- work for this variant exactly as they do for the baseline
# families.

_POWER_NODE_MIN, _POWER_NODE_MAX = 0.02, 4.0
# n_quad_nodes -> (nodes, log_weights), memoized: hermegauss() itself is a
# one-time O(n^2) eigenvalue-based computation, cheap to redo once, but
# every optimizer iteration calls _gh_nodes/_gh_log_weights, so caching by
# node count avoids recomputing this on every single likelihood evaluation.
# Populated lazily by _gh_cache_entry rather than assuming _gh_nodes is
# always called before _gh_log_weights for a given n_quad_nodes -- both
# functions go through the same helper so neither has an implicit ordering
# dependency on the other.
_GH_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _gh_cache_entry(n_quad_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    if n_quad_nodes not in _GH_CACHE:
        nodes, weights = hermegauss(n_quad_nodes)
        _GH_CACHE[n_quad_nodes] = (nodes, np.log(weights) - 0.5 * np.log(2.0 * np.pi))
    return _GH_CACHE[n_quad_nodes]


def _gh_nodes(mu_p: float, sigma_p: float, n_quad_nodes: int = 21) -> np.ndarray:
    if sigma_p < 1e-6:
        # sigma_p -> 0 limit: quadrature nodes would all collapse onto mu_p
        # (numerically unstable weighting of near-duplicate points), and the
        # exact sigma_p -> 0 limit of the marginal likelihood IS the plain
        # plug-in likelihood at p_j = mu_p, so evaluate there directly.
        return np.array([mu_p])
    nodes, _ = _gh_cache_entry(n_quad_nodes)
    return mu_p + sigma_p * nodes


def _gh_log_weights(sigma_p: float, n_quad_nodes: int = 21) -> np.ndarray:
    if sigma_p < 1e-6:
        return np.array([0.0])
    return _gh_cache_entry(n_quad_nodes)[1]


def _tool_node_loglik(dv_j, t_start_j, t_end_j, eta_j: float, p_nodes: np.ndarray, beta: float, t_ref: float, resolution_um: float, bin_prob_fn) -> np.ndarray:
    """Tool j's total window log-likelihood at EACH quadrature node p_k,
    given (eta_j, beta); -1e100 for a node outside the sane exponent range
    or that produces an invalid shape."""
    node_loglik = np.full(len(p_nodes), -1e100)
    for k, p_k in enumerate(p_nodes):
        if not (_POWER_NODE_MIN <= p_k <= _POWER_NODE_MAX):
            continue
        shape_s = eta_j * (np.power(t_end_j / t_ref, p_k) - np.power(t_start_j / t_ref, p_k))
        if np.any(shape_s <= 0) or not np.all(np.isfinite(shape_s)):
            continue
        prob = bin_prob_fn(dv_j, shape_s, beta, resolution_um)
        node_loglik[k] = float(np.sum(np.log(prob)))
    return node_loglik


def _tool_posterior_mean_power(p_nodes: np.ndarray, node_loglik: np.ndarray, log_weights: np.ndarray) -> float:
    log_post_unnorm = log_weights + node_loglik
    log_norm = logsumexp(log_post_unnorm)
    post_w = np.exp(log_post_unnorm - log_norm)
    return float(np.sum(post_w * p_nodes))


def _fit_rounded_nh_random_effects(
    dv: np.ndarray, t_start: np.ndarray, t_end: np.ndarray, tool_ids: np.ndarray,
    v_start: np.ndarray | None = None, *,
    bin_prob_fn, model_name: str,
    time_reference_s: float = TIME_REFERENCE_S, resolution_um: float = WEAR_RESOLUTION_UM,
    n_quad_nodes: int = 21,
) -> dict:
    """Shared fitter for the random-effects Gamma and Inverse Gaussian
    variants. ``v_start`` is accepted for interface uniformity and ignored
    -- this family's randomness is in the aging exponent's population
    distribution, not a wear-level dependence.

    ``n_quad_nodes`` controls the Gauss-Hermite quadrature's node count
    (default 21, a standard choice for this kind of 1-D mixed-effects
    marginal likelihood) -- pass a smaller value (e.g. 7) for expensive
    repeated-refit sweeps (model recovery, bootstrap), mirroring multi-stage
    Wiener's identical ``n_grid`` precision/cost tradeoff (see that
    family's own docstring): every quadrature node multiplies the cost of
    EVERY likelihood evaluation the optimizer performs, so this is the
    single most impactful knob for this family's fit cost, never coarsened
    for the primary real-data fit itself.
    """
    dv = np.asarray(dv, float); t_start = np.asarray(t_start, float); t_end = np.asarray(t_end, float)
    tool_ids = np.asarray(tool_ids, int)
    _validate_windows(dv, t_start, t_end, tool_ids)

    tools = np.unique(tool_ids)
    n_tools = len(tools)
    t_ref = float(time_reference_s)
    eta0, power0 = _init_eta_power(dv, t_start, t_end, tool_ids, tools, t_ref)
    beta0 = 0.5
    mu_p0 = float(np.mean(power0))
    sigma_p0 = 0.15

    tool_windows = []
    for tool in tools:
        mask = tool_ids == tool
        tool_windows.append((dv[mask], t_start[mask], t_end[mask]))

    def nll_full(z: np.ndarray) -> float:
        eta = np.exp(z[:n_tools])
        beta = float(np.exp(z[n_tools]))
        mu_p = float(z[n_tools + 1])
        sigma_p = float(np.exp(z[n_tools + 2]))
        if beta < 1e-4 or beta > 100.0 or mu_p < 0.05 or mu_p > 3.0 or sigma_p > 2.0:
            return 1e100
        p_nodes = _gh_nodes(mu_p, sigma_p, n_quad_nodes)
        log_w = _gh_log_weights(sigma_p, n_quad_nodes)
        total = 0.0
        for j, (dv_j, ts_j, te_j) in enumerate(tool_windows):
            node_ll = _tool_node_loglik(dv_j, ts_j, te_j, eta[j], p_nodes, beta, t_ref, resolution_um, bin_prob_fn)
            if np.all(node_ll <= -1e99):
                return 1e100
            total += float(logsumexp(log_w + node_ll))
        return -total

    starts = [
        np.r_[np.log(eta0), np.log(beta0), mu_p0, np.log(sigma_p0)],
        np.r_[np.log(eta0 * 0.7), np.log(0.4), 0.6, np.log(0.30)],
        np.r_[np.log(eta0 * 1.3), np.log(0.7), 1.0, np.log(0.05)],
    ]
    best = None
    for x0 in starts:
        res = minimize(nll_full, x0, method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-8, "fatol": 1e-8})
        if best is None or res.fun < best.fun:
            best = res
    if best is None or not np.isfinite(best.fun):
        raise RuntimeError(f"Random-effects {model_name} MLE (full model) failed.")

    eta = np.exp(best.x[:n_tools])
    beta = float(np.exp(best.x[n_tools]))
    mu_p = float(best.x[n_tools + 1])
    sigma_p = float(np.exp(best.x[n_tools + 2]))
    log_lik_full = -float(best.fun)

    # Nested sigma_p=0 comparison: every tool shares exactly mu_p as a
    # FIXED (not random) aging exponent -- ordinary MLE, no quadrature.
    def nll_pooled(z: np.ndarray) -> float:
        eta = np.exp(z[:n_tools]); beta = float(np.exp(z[n_tools])); mu_p = float(z[n_tools + 1])
        if beta < 1e-4 or beta > 100.0 or mu_p < 0.05 or mu_p > 3.0:
            return 1e100
        total = 0.0
        for j, (dv_j, ts_j, te_j) in enumerate(tool_windows):
            shape_s = eta[j] * (np.power(te_j / t_ref, mu_p) - np.power(ts_j / t_ref, mu_p))
            if np.any(shape_s <= 0) or not np.all(np.isfinite(shape_s)):
                return 1e100
            prob = bin_prob_fn(dv_j, shape_s, beta, resolution_um)
            total += float(np.sum(np.log(prob)))
        return -total

    pooled_start = np.r_[np.log(eta), np.log(beta), mu_p]
    pooled = minimize(nll_pooled, pooled_start, method="Nelder-Mead", options={"maxiter": 30000, "xatol": 1e-9, "fatol": 1e-9})
    if not np.isfinite(pooled.fun):
        raise RuntimeError(f"Random-effects {model_name} MLE (pooled-exponent nested model) failed.")
    pooled_eta = np.exp(pooled.x[:n_tools]); pooled_beta = float(np.exp(pooled.x[n_tools])); pooled_mu_p = float(pooled.x[n_tools + 1])
    log_lik_pooled = -float(pooled.fun)

    n = len(dv)
    k_full = n_tools + 3       # eta x n_tools, beta, mu_p, sigma_p
    k_pooled = n_tools + 2     # eta x n_tools, beta, mu_p
    aic_full = float(2 * k_full - 2 * log_lik_full)
    bic_full = float(k_full * np.log(n) - 2 * log_lik_full)
    aic_pooled = float(2 * k_pooled - 2 * log_lik_pooled)
    bic_pooled = float(k_pooled * np.log(n) - 2 * log_lik_pooled)

    full_fit_for_test = {"log_lik": log_lik_full, "aic": aic_full, "bic": bic_full}
    pooled_fit_for_test = {"log_lik": log_lik_pooled, "aic": aic_pooled, "bic": bic_pooled}
    sigma_p_test = families.nested_model_test(pooled_fit_for_test, full_fit_for_test, extra_params=1, boundary_corrected=True)
    use_random_effects = sigma_p_test["variant_supported_aic"]

    if use_random_effects:
        # Posterior-mean aging exponent per tool, for downstream code that
        # expects an ordinary per-tool "power" (Vuong's test, CV, PIT-KS
        # validation, plotting -- see the section docstring above).
        p_nodes = _gh_nodes(mu_p, sigma_p, n_quad_nodes)
        log_w = _gh_log_weights(sigma_p, n_quad_nodes)
        components = []
        for j, tool in enumerate(tools):
            dv_j, ts_j, te_j = tool_windows[j]
            node_ll = _tool_node_loglik(dv_j, ts_j, te_j, eta[j], p_nodes, beta, t_ref, resolution_um, bin_prob_fn)
            posterior_power = _tool_posterior_mean_power(p_nodes, node_ll, log_w)
            components.append({
                "tool_id": int(tool), "eta": float(eta[j]), "power": posterior_power,
                "beta": beta, "time_reference_s": t_ref,
            })
        selected_beta, selected_log_lik = beta, log_lik_full
        selected_aic, selected_bic = aic_full, bic_full
        selected_re_model = "random_effects"
    else:
        components = [
            {"tool_id": int(tool), "eta": float(pooled_eta[j]), "power": pooled_mu_p,
             "beta": pooled_beta, "time_reference_s": t_ref}
            for j, tool in enumerate(tools)
        ]
        selected_beta, selected_log_lik = pooled_beta, log_lik_pooled
        selected_aic, selected_bic = aic_pooled, bic_pooled
        selected_re_model = "pooled_exponent"

    return {
        "model": model_name, "schema_version": 1,
        "selected_re_model": selected_re_model,
        # selected_age_model kept for interface uniformity with the other 8
        # families (tables/plots that key off it should treat this
        # variant's "power_law"/"linear" as read from selected_re_model
        # instead, and are updated to do so) -- set to the closer analogue
        # so nothing crashes if some generic code reads it regardless.
        "selected_age_model": "power_law" if selected_re_model == "random_effects" else "linear",
        "beta": selected_beta,
        "eta": float(np.median([c["eta"] for c in components])),
        "power": float(np.median([c["power"] for c in components])),
        "mu_p": mu_p, "sigma_p": sigma_p if use_random_effects else 0.0,
        "sigma_p_test": sigma_p_test,
        "time_reference_s": t_ref, "wear_resolution_um": float(resolution_um),
        "components": components, "n_windows": int(n), "n_zero_windows": int(np.sum(dv == 0)),
        "log_lik": selected_log_lik, "aic": selected_aic, "bic": selected_bic,
        # Kept for interface parity with the baseline families' dict shape
        # (some generic code reads *_aic/*_bic to report a delta) -- here
        # they describe the full-vs-pooled RE axis rather than a
        # power-law-vs-linear age-model axis.
        "nonhomogeneous_aic": aic_full, "nonhomogeneous_bic": bic_full,
        "stationary_aic": aic_pooled, "stationary_bic": bic_pooled,
    }


def _fit_gamma_re_adapter(dv, t_start, t_end, tool_ids, v_start=None, **kw):
    return _fit_rounded_nh_random_effects(
        dv, t_start, t_end, tool_ids, v_start,
        bin_prob_fn=_rounded_gamma_bin_probability_for_variants, model_name="random_effects_gamma_rounded", **kw,
    )


def _fit_ig_re_adapter(dv, t_start, t_end, tool_ids, v_start=None, **kw):
    return _fit_rounded_nh_random_effects(
        dv, t_start, t_end, tool_ids, v_start,
        bin_prob_fn=_rounded_ig_bin_probability_for_variants, model_name="random_effects_inverse_gaussian_rounded", **kw,
    )


families.register(
    "gamma_random_effects",
    label="Random-effects Gamma",
    color="#0072B2", ls=(0, (3, 1, 1, 1)), marker="X",
    fit_fn=_fit_gamma_re_adapter,
    shape_fn=families.standard_shape_fn,
    predict_shape_fn=families.standard_predict_shape_fn,
    cdf_fn=_gamma_cdf,
    bin_prob_fn=_rounded_gamma_bin_probability_for_variants,
    rvs_fn=lambda shape_s, beta, rng: gamma_dist.rvs(a=shape_s, scale=1.0 / beta, random_state=rng),
    is_subordinator=True,
    baseline_key=None,  # not cleanly nested in plain Gamma -- see section docstring; compared via AIC/BIC/Vuong like any other family
    extra_params_desc="aging exponent p_j ~ Normal(mu_p, sigma_p^2) across a machine's tools, integrated out via Gauss-Hermite quadrature, vs. one free exponent per tool",
)

families.register(
    "inverse_gaussian_random_effects",
    label="Random-effects Inverse Gaussian",
    color="#D55E00", ls=(0, (3, 1, 1, 1)), marker="*",
    fit_fn=_fit_ig_re_adapter,
    shape_fn=families.standard_shape_fn,
    predict_shape_fn=families.standard_predict_shape_fn,
    cdf_fn=_ig_cdf,
    bin_prob_fn=_rounded_ig_bin_probability_for_variants,
    rvs_fn=lambda shape_s, beta, rng: invgauss.rvs(mu=1.0 / shape_s, scale=(shape_s ** 2) / beta, random_state=rng),
    is_subordinator=True,
    baseline_key=None,
    extra_params_desc="aging exponent p_j ~ Normal(mu_p, sigma_p^2) across a machine's tools, integrated out via Gauss-Hermite quadrature, vs. one free exponent per tool",
)


# ── Multi-stage (2-stage) Wiener ────────────────────────────────────────────
#
# Scoped to 2 stages, not 3, matching what Figure 1 actually shows (fast
# run-in, then slower steady wear -- not a further late-life acceleration
# phase), and scoped to a RATE change rather than an aging-exponent change:
# each stage is linear in time (p=1, no within-stage power-law curvature),
# so the only structural question this variant asks is "does the wear rate
# step to a different level at some point in a tool's life," not "does the
# aging pattern also change shape" -- the more parameter-hungry question
# Section 4.3 already argues against answering with this dataset's window
# count. Per-tool rates eta_1j (stage 1) and eta_2j (stage 2), one
# change-point time tau_c SHARED across a machine's tools (mirroring how
# beta is already shared, rather than a change-point per tool -- a run-in
# duration is plausibly a machine/process property more than an
# individual-tool one, and 3 tools/machine cannot support 3 independent
# change-points), fit by PROFILE likelihood: grid-search candidate tau_c
# values, run the ordinary Nelder-Mead MLE (3 restarts, as everywhere else
# in this study) conditional on each candidate, keep the likelihood-
# maximizing tau_c. This is the standard, numerically safe way to handle a
# change-point parameter -- a change in a piecewise-linear mean function is
# a point of non-differentiability the same continuous Nelder-Mead search
# used for every other parameter here cannot reliably locate directly.
#
# Reduces EXACTLY to plain Wiener's linear (p=1) special case when
# eta_1j = eta_2j for every tool, for ANY tau_c -- a clean nested
# relationship (baseline_key="wiener"), tested both internally (this
# fitter's own 2-stage-vs-1-stage AIC/BIC selection, exactly mirroring
# every other family's stationary-vs-non-homogeneous internal selection)
# and, downstream, against plain Wiener's own separately-fitted result.
#
# CAVEAT, stated here rather than only in the manuscript: because tau_c is
# UNIDENTIFIED under the null eta_1j=eta_2j (the changepoint's location
# does not matter when there is no change), the standard chi-square
# likelihood-ratio asymptotic this module's other nested tests rely on does
# not directly apply to the "is there a genuine 2-stage break" question
# (the Davies 1987 problem for changepoint models) the way it does for
# gamma_state (an ordinary, non-boundary, non-changepoint parameter) or the
# random-effects sigma_p test (a KNOWN, standard boundary correction
# applies there instead). This module still reports the naive
# chi-square(n_tools+1) p-value via nested_model_test (WITHOUT the
# boundary correction, which is the wrong fix for a different problem), but
# it should be read as approximate/indicative, not exact -- AIC/BIC, which
# require no such asymptotic and are this study's primary selection
# criterion everywhere, are the primary basis for this variant's own
# internal 2-stage-vs-1-stage decision, exactly as for every other family.
#
# For the expensive inner-loop refits inside model recovery / bootstrap
# (which refit from scratch many times), pass a coarser tau_c grid than the
# fine grid used for the primary real-data fit (n_grid=25 default) via the
# n_grid keyword, to keep total sweep runtime bounded -- a deliberate
# precision/cost tradeoff, not a silent shortcut: the coarser grid is only
# ever used inside those specific sweeps, never for the headline result.

def _wiener_cdf(x: np.ndarray, shape_s: np.ndarray, beta) -> np.ndarray:
    mu = shape_s / beta
    sigma = np.sqrt(shape_s) / beta
    return norm.cdf(x, loc=mu, scale=sigma)


_rounded_wiener_bin_probability_for_variants = families.generic_bin_prob_from_cdf(_wiener_cdf)


def _multistage_shape(eta1, eta2, tau_c: float, t_start: np.ndarray, t_end: np.ndarray, t_ref: float) -> np.ndarray:
    """Piecewise-linear shape quantity for a 2-stage Wiener process:
    S(s,t) = [eta1 * stage1_duration + eta2 * stage2_duration] / t_ref,
    where stage1_duration/stage2_duration partition each window's [s,t]
    span by tau_c (correctly handling windows whose span straddles the
    changepoint, not just windows fully on one side of it). Reduces
    exactly to the standard linear (p=1) shape formula eta*(t-s)/t_ref when
    eta1=eta2, for any tau_c."""
    stage1_dt = np.clip(np.minimum(t_end, tau_c) - t_start, 0.0, None)
    stage2_dt = np.clip(t_end - np.maximum(t_start, tau_c), 0.0, None)
    return (eta1 * stage1_dt + eta2 * stage2_dt) / t_ref


def _multistage_shape_fn(component: dict, t_start: np.ndarray, t_end: np.ndarray, v_start: np.ndarray | None = None) -> np.ndarray:
    eta1 = float(component["eta1"]); eta2 = float(component["eta2"])
    tau_c = float(component["tau_c"]); t_ref = float(component["time_reference_s"])
    return np.maximum(_multistage_shape(eta1, eta2, tau_c, t_start, t_end, t_ref), 1e-12)


def _multistage_predict_shape_fn(fitted: dict, t_start: np.ndarray, t_end: np.ndarray, v_start: np.ndarray | None = None) -> np.ndarray:
    eta1_med = float(np.median([c["eta1"] for c in fitted["components"]]))
    eta2_med = float(np.median([c["eta2"] for c in fitted["components"]]))
    tau_c = float(fitted["components"][0]["tau_c"])  # shared across a machine's tools -- identical in every component
    t_ref = float(fitted["time_reference_s"])
    return np.maximum(_multistage_shape(eta1_med, eta2_med, tau_c, t_start, t_end, t_ref), 1e-12)


def _fit_rounded_multistage_wiener(
    dv: np.ndarray, t_start: np.ndarray, t_end: np.ndarray, tool_ids: np.ndarray,
    v_start: np.ndarray | None = None, *,
    time_reference_s: float = TIME_REFERENCE_S, resolution_um: float = WEAR_RESOLUTION_UM,
    n_grid: int = 25,
) -> dict:
    """Fit a tool-heterogeneous 2-stage Wiener process by profile likelihood
    over a change-point grid. ``v_start`` is accepted for interface
    uniformity and ignored. ``n_grid`` controls the change-point grid's
    resolution -- pass a smaller value for expensive repeated-refit sweeps
    (see the section docstring above).
    """
    dv = np.asarray(dv, float); t_start = np.asarray(t_start, float); t_end = np.asarray(t_end, float)
    tool_ids = np.asarray(tool_ids, int)
    _validate_windows(dv, t_start, t_end, tool_ids)

    tools = np.unique(tool_ids)
    tool_col = {int(tool): i for i, tool in enumerate(tools)}
    component_idx = np.array([tool_col[int(tool)] for tool in tool_ids], int)
    n_tools = len(tools)
    t_ref = float(time_reference_s)
    t_max = float(np.max(t_end))
    tau_grid = np.linspace(0.05, 0.35, max(int(n_grid), 2)) * t_max

    eta0, _power0_unused = _init_eta_power(dv, t_start, t_end, tool_ids, tools, t_ref, power0_val=1.0)
    beta0 = 0.5

    def nll_given_tau(z: np.ndarray, tau_c: float) -> float:
        eta1 = np.exp(z[:n_tools]); eta2 = np.exp(z[n_tools:2 * n_tools]); beta = float(np.exp(z[-1]))
        if beta < 1e-4 or beta > 100.0:
            return 1e100
        shape_s = _multistage_shape(eta1[component_idx], eta2[component_idx], tau_c, t_start, t_end, t_ref)
        if np.any(shape_s <= 0) or not np.all(np.isfinite(shape_s)):
            return 1e100
        prob = _rounded_wiener_bin_probability_for_variants(dv, shape_s, beta, resolution_um)
        return -float(np.sum(np.log(prob)))

    best = None
    best_tau_c = None
    for tau_c in tau_grid:
        starts = [
            np.r_[np.log(eta0), np.log(eta0), np.log(beta0)],
            np.r_[np.log(eta0 * 1.6), np.log(eta0 * 0.6), np.log(0.4)],
            np.r_[np.log(eta0 * 0.6), np.log(eta0 * 1.6), np.log(0.7)],
        ]
        best_here = None
        for x0 in starts:
            res = minimize(lambda z: nll_given_tau(z, tau_c), x0, method="Nelder-Mead",
                            options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-8})
            if best_here is None or res.fun < best_here.fun:
                best_here = res
        if best is None or best_here.fun < best.fun:
            best = best_here
            best_tau_c = float(tau_c)
    if best is None or not np.isfinite(best.fun):
        raise RuntimeError("2-stage Wiener MLE (grid search) failed.")

    eta1 = np.exp(best.x[:n_tools]); eta2 = np.exp(best.x[n_tools:2 * n_tools]); beta = float(np.exp(best.x[-1]))
    log_lik_2stage = -float(best.fun)

    # Nested 1-stage (eta1=eta2, tau_c irrelevant) comparison, mathematically
    # identical to plain Wiener's own linear/p=1 fit but refit locally here
    # (a single cheap Nelder-Mead run) to keep this module self-contained
    # rather than importing degradation_process_models (which itself
    # imports this module -- see degradation_families.py's docstring on the
    # acyclic import structure this whole refactor depends on).
    def nll_1stage(z: np.ndarray) -> float:
        eta = np.exp(z[:n_tools]); beta = float(np.exp(z[n_tools]))
        if beta < 1e-4 or beta > 100.0:
            return 1e100
        shape_s = eta[component_idx] * (t_end - t_start) / t_ref
        if np.any(shape_s <= 0) or not np.all(np.isfinite(shape_s)):
            return 1e100
        prob = _rounded_wiener_bin_probability_for_variants(dv, shape_s, beta, resolution_um)
        return -float(np.sum(np.log(prob)))

    one_stage_start = np.log(np.r_[eta0, beta0])
    one_stage = minimize(nll_1stage, one_stage_start, method="Nelder-Mead",
                          options={"maxiter": 30000, "xatol": 1e-9, "fatol": 1e-9})
    if not np.isfinite(one_stage.fun):
        raise RuntimeError("2-stage Wiener MLE (nested 1-stage null) failed.")
    one_stage_eta = np.exp(one_stage.x[:n_tools]); one_stage_beta = float(np.exp(one_stage.x[n_tools]))
    log_lik_1stage = -float(one_stage.fun)

    n = len(dv)
    k_2stage = 2 * n_tools + 2   # eta1, eta2 per tool, beta, tau_c
    k_1stage = n_tools + 1       # eta per tool, beta -- same as plain Wiener's linear case
    aic_2stage = float(2 * k_2stage - 2 * log_lik_2stage)
    bic_2stage = float(k_2stage * np.log(n) - 2 * log_lik_2stage)
    aic_1stage = float(2 * k_1stage - 2 * log_lik_1stage)
    bic_1stage = float(k_1stage * np.log(n) - 2 * log_lik_1stage)

    fit_2stage_for_test = {"log_lik": log_lik_2stage, "aic": aic_2stage, "bic": bic_2stage}
    fit_1stage_for_test = {"log_lik": log_lik_1stage, "aic": aic_1stage, "bic": bic_1stage}
    changepoint_test = families.nested_model_test(
        fit_1stage_for_test, fit_2stage_for_test, extra_params=n_tools + 1, boundary_corrected=False,
    )
    changepoint_test["caveat"] = (
        "tau_c is unidentified under the eta1=eta2 null (the Davies 1987 problem for "
        "changepoint models); the chi-square asymptotic and this p-value are approximate, "
        "not exact -- variant_supported_aic is the primary selection criterion, as for "
        "every other model-selection decision in this study."
    )
    use_2stage = bool(changepoint_test["variant_supported_aic"])

    if use_2stage:
        components = [
            {"tool_id": int(tool), "eta1": float(eta1[i]), "eta2": float(eta2[i]), "tau_c": best_tau_c,
             "beta": beta, "time_reference_s": t_ref}
            for i, tool in enumerate(tools)
        ]
        selected_beta, selected_log_lik = beta, log_lik_2stage
        selected_aic, selected_bic = aic_2stage, bic_2stage
    else:
        components = [
            {"tool_id": int(tool), "eta1": float(one_stage_eta[i]), "eta2": float(one_stage_eta[i]), "tau_c": best_tau_c,
             "beta": one_stage_beta, "time_reference_s": t_ref}
            for i, tool in enumerate(tools)
        ]
        selected_beta, selected_log_lik = one_stage_beta, log_lik_1stage
        selected_aic, selected_bic = aic_1stage, bic_1stage

    return {
        "model": "multistage_wiener_rounded", "schema_version": 1,
        "selected_stage_model": "two_stage" if use_2stage else "one_stage",
        "selected_age_model": "power_law" if use_2stage else "linear",  # interface parity, see the RE variant's identical note
        "beta": selected_beta,
        "eta": float(np.median([0.5 * (c["eta1"] + c["eta2"]) for c in components])),
        "power": 1.0,
        "tau_c": best_tau_c, "tau_grid_n": len(tau_grid),
        "changepoint_test": changepoint_test,
        "time_reference_s": t_ref, "wear_resolution_um": float(resolution_um),
        "components": components, "n_windows": int(n), "n_zero_windows": int(np.sum(dv == 0)),
        "log_lik": selected_log_lik, "aic": selected_aic, "bic": selected_bic,
        "nonhomogeneous_aic": aic_2stage, "nonhomogeneous_bic": bic_2stage,
        "stationary_aic": aic_1stage, "stationary_bic": bic_1stage,
        # Wiener-family structural diagnostic (see degradation_process_models._rounded_wiener_bin_probability):
        # computed on the selected components' own (mu, sigma) at every real window.
        "mean_p_negative_increment": float(np.mean(norm.cdf(
            0.0,
            loc=_multistage_shape(
                np.array([c["eta1"] for c in components])[component_idx],
                np.array([c["eta2"] for c in components])[component_idx],
                best_tau_c, t_start, t_end, t_ref,
            ) / selected_beta,
            scale=np.sqrt(np.maximum(_multistage_shape(
                np.array([c["eta1"] for c in components])[component_idx],
                np.array([c["eta2"] for c in components])[component_idx],
                best_tau_c, t_start, t_end, t_ref,
            ), 1e-12)) / selected_beta,
        ))),
    }


def _fit_multistage_wiener_adapter(dv, t_start, t_end, tool_ids, v_start=None, **kw):
    return _fit_rounded_multistage_wiener(dv, t_start, t_end, tool_ids, v_start, **kw)


families.register(
    "wiener_multistage",
    label="Multi-stage Wiener",
    color="#009E73", ls="--", marker="^",
    fit_fn=_fit_multistage_wiener_adapter,
    shape_fn=_multistage_shape_fn,
    predict_shape_fn=_multistage_predict_shape_fn,
    cdf_fn=_wiener_cdf,
    bin_prob_fn=_rounded_wiener_bin_probability_for_variants,
    # Single WINDOW-level draw from the marginal Normal(shape_s/beta,
    # sqrt(shape_s)/beta) -- shape_s already encodes eta1/eta2/tau_c via
    # this family's own shape_fn, so the marginal form is identical to
    # plain Wiener's. Well-defined despite non-monotone PATHS (see
    # degradation_families.py's rvs_fn/is_subordinator docstring) -- used
    # for model-recovery-style single-window simulation, never for
    # cumulative-trajectory MTTFF step-simulation (is_subordinator=False
    # correctly keeps this family on the exact first-passage-time route
    # for that instead).
    rvs_fn=lambda shape_s, beta, rng: norm.rvs(loc=shape_s / beta, scale=np.sqrt(shape_s) / beta, random_state=rng),
    is_subordinator=False,
    baseline_key="wiener",
    extra_params_desc="+n_tools rates (separate eta per tool per stage) +1 shared change-point time per machine, vs. plain Wiener's linear (p=1) case; nests it exactly at eta1=eta2 for every tool",
)


# ── Generalized Inverse Gaussian (GIG) ──────────────────────────────────────
#
# Derivation (independently verified numerically, not just symbolically --
# see tests/test_degradation_process_variants.py's TestGIG class):
#
# The general 3-parameter GIG(a, b_gen, p) has density
#   f(x) = (a/b_gen)^(p/2) / (2 K_p(sqrt(a*b_gen))) * x^(p-1) * exp(-(a*x + b_gen/x)/2)
# scipy.stats.geninvgauss(p, b, loc, scale) represents this with
#   b (scipy's shape param)  = chi = sqrt(a*b_gen)
#   scale                    = sqrt(b_gen/a)
#
# This study's own Inverse Gaussian convention (Eq. 2b: mean=S/beta,
# shape=S^2/beta, S=shape_s) is EXACTLY GIG(p=-1/2, a=beta, b_gen=S^2/beta)
# -- re-derived from the GIG density's b_gen->0 and general-p forms
# directly (matching the standard IG-as-GIG identity), then verified
# numerically to agree with production invgauss.cdf to ~1e-14. Under this
# mapping chi=sqrt(a*b_gen)=S and scale=S/beta, and at p=-1/2 specifically,
# R_p(chi):=K_{p+1}(chi)/K_p(chi) equals exactly 1 for every chi (the
# standard Bessel symmetry K_v=K_{-v} applied to v=1/2), which is the
# algebraic reason p=-1/2 is a clean, non-boundary interior point of GIG's
# parameter space rather than a limit -- unlike Gamma, which is only
# reachable as a b_gen->0 boundary limit and is NOT nested by this specific
# construction (not needed: the paper's existing non-nested Gamma-vs-IG
# AIC/BIC/Vuong comparison already covers that question; this variant's
# purpose is a focused, CLEANLY NESTED test of Inverse Gaussian's own
# tail-shape assumption specifically).
#
# To let p_gid move away from -1/2 while keeping the fitted MEAN pinned at
# the study's own shape_s/beta convention (so a change in fit quality
# reflects tail shape, not a shift in average wear rate), chi is held fixed
# at shape_s (unchanged from the p=-1/2 case) while the a/b_gen SPLIT is
# corrected by a closed-form factor c = 1/R_p(shape_s):
#   a = beta / c,  b_gen = c * shape_s^2 / beta
# This keeps chi=sqrt(a*b_gen)=shape_s invariant (by construction: the two
# factors of c cancel in the product) while rescaling E[X] by exactly c *
# R_p(shape_s) = 1, i.e. the mean stays pinned at shape_s/beta for every
# p_gid -- verified numerically to hold across a wide range of p_gid to 6
# decimal places. Variance and higher moments are NOT held fixed by this
# construction (doing so would need a per-window numerical root-find
# instead of a closed form); this narrower scope (mean-only invariance) is
# deliberate and stated explicitly rather than overclaimed.
#
# KNOWN, VERIFIED STATISTICAL PROPERTY (not a bug -- found and diagnosed
# during implementation, not swept under the rug): fitting p_gid against
# ROUNDED data at realistic sample sizes (order 50-200 windows, matching
# this study's own per-machine window counts) can land noticeably away
# from the true generating value, biased toward more negative p_gid. Root
# cause, confirmed directly rather than assumed: this is a
# discretization/small-sample interaction, NOT a parametrization error --
# the CONTINUOUS (rounding-free) cross-entropy H(true, p_gid), computed
# straight from _gig_log_pdf with no rounding kernel involved, IS exactly
# minimized at the true p_gid (verified to 1e-9 in
# tests/test_degradation_process_variants.py's
# test_cross_entropy_is_minimized_at_the_true_p_gid_without_rounding), so
# the mathematical derivation above is correct; the bias appears only once
# the rounding-kernel bin-probability likelihood is introduced, because the
# extra tail-shape degree of freedom this variant adds can partly track
# rounding-induced structure in the discretized data that has nothing to
# do with genuine distributional shape -- something Gamma/IG/Wiener cannot
# do, since none of them has a free parameter playing this role. PRACTICAL
# IMPLICATION: a GIG fit's p_gid value on real, similarly-sized data should
# be read with real caution, not treated as precise evidence about tail
# shape on its own; the robustness battery this variant feeds into should
# include a dedicated model-recovery check (fit GIG to synthetic data
# simulated from the real fitted Inverse Gaussian parameters, exactly the
# positive-control logic degradation_process_models.model_recovery_study
# already applies to the original 3 families) to quantify this bias
# concretely on this study's own data and window counts, rather than leave
# it as a qualitative caveat -- and the manuscript's own treatment of any
# reported p_gid value should be hedged accordingly.

def _r_p(p: float, chi: np.ndarray) -> np.ndarray:
    return kv(p + 1.0, chi) / kv(p, chi)


def _gig_ab_from_shape(shape_s: np.ndarray, beta, p_gid: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean-preserving GIG(a, b_gen, p_gid) split at fixed chi=shape_s --
    see the section docstring above for the full derivation.

    Modified Bessel functions of large order at small argument can
    overflow/underflow to inf/0 in floating point, making the kv ratio
    above legitimately 0/0 or inf/inf at some (p_gid, chi) combinations --
    reachable only while Nelder-Mead is exploring a candidate point, since
    _fit_rounded_nh_gig's nll() checks ``np.isfinite(prob)`` after calling
    this (via _rounded_gig_bin_probability) and rejects any such candidate
    with the standard 1e100 sentinel, exactly like every other invalid-
    region guard in this module. Silenced here rather than left to spam
    RuntimeWarning at every one of these (harmless, always-rejected)
    candidate evaluations -- np.errstate is scoped to just this
    computation, so a genuine unrelated warning elsewhere is unaffected.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        Rp = _r_p(p_gid, shape_s)
        c = 1.0 / Rp
        a = beta / c
        b_gen = c * (shape_s ** 2) / beta
    return a, b_gen


def _gig_chi_scale_from_shape(shape_s: np.ndarray, beta, p_gid: float) -> tuple[np.ndarray, np.ndarray]:
    """(a, b_gen) -> (chi, scale) in scipy's geninvgauss parametrization,
    via _gig_ab_from_shape -- the exact 2-line pattern every one of this
    variant's cdf_fn/bin_prob_fn/rvs_fn call sites needs, centralized here
    (rather than duplicated 3x) so the same overflow/underflow-safe
    np.errstate scoping in _gig_ab_from_shape's docstring applies uniformly
    to the sqrt() calls immediately downstream of it too -- a NaN/inf `a`
    or `b_gen` from an extreme Nelder-Mead candidate would otherwise still
    warn here even with that function's own internals silenced."""
    a, b_gen = _gig_ab_from_shape(shape_s, beta, p_gid)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = np.sqrt(a * b_gen)
        scale = np.sqrt(b_gen / a)
    return chi, scale


# scipy.stats.geninvgauss.cdf is NOT used here, deliberately: it falls back
# to generic per-point adaptive quadrature (scipy.integrate.quad) with no
# specialized fast path, which is not just slow (~150ms per likelihood
# evaluation, observed to make a single machine's fit take 20+ minutes
# instead of the other families' ~1-2 minutes) but occasionally emits its
# own IntegrationWarning ("does not converge", "probably divergent") --
# i.e. its accuracy is not fully trustworthy either, not only its speed.
# scipy.stats.geninvgauss.pdf, by contrast, IS fast (no integration
# involved) -- confirmed to match a hand-derived closed-form density
# exactly (see _gig_log_pdf below and tests/test_degradation_process_variants.py's
# TestGIG class), so both replacements here integrate THAT closed form
# directly with fixed-node Gauss-Legendre quadrature, verified against a
# strict scipy.integrate.quad reference (warnings promoted to errors, tight
# epsabs/epsrel) across 500 random (p, shape, scale, x) combinations
# spanning the full p_gid range this fitter explores, including x=0 (a
# real, retained data point -- zero-wear-increment windows).

def _gig_log_pdf(x: np.ndarray, chi: np.ndarray, scale: np.ndarray, p_gid: float, log_kv: np.ndarray) -> np.ndarray:
    """log GIG(p_gid, chi, scale) density at x>0, in scipy's geninvgauss
    parametrization: f(x) = z^(p-1) * exp(-chi*(z+1/z)/2) / (2*scale*K_p(chi)),
    z = x/scale. Verified to match scipy.stats.geninvgauss.pdf exactly (see
    tests) -- computed directly, in log space, rather than via scipy's
    rv_continuous wrapper purely for speed (~10x faster per call in
    isolation, and avoids that wrapper's argument-checking overhead when
    called millions of times during a fit).

    ``log_kv`` (log K_p_gid(chi)) is a REQUIRED, precomputed argument, not
    recomputed here: it depends only on (p_gid, chi), which are fixed for
    an entire quadrature (chi is fixed per window; only the evaluation
    point x varies across nodes), so computing it inside this function
    would redundantly repeat an expensive Bessel-function call at every one
    of the ~700 nodes a single likelihood evaluation uses -- measured as
    the dominant cost of an earlier version of this module (a single
    machine's GIG fit took 500+ seconds; hoisting this one call out, plus
    the full vectorization in _rounded_gig_bin_probability/_gig_cdf below,
    brought it back in line with every other family's fitting time).

    Can legitimately divide-by-zero/log(0) in floating point if an upstream
    candidate (scale, log_kv) is itself already an overflow/underflow
    artifact from _gig_ab_from_shape at an extreme (p_gid, chi) Nelder-Mead
    is probing -- see that function's docstring; the same downstream
    np.isfinite(prob) guard in _fit_rounded_nh_gig's nll() rejects the
    result either way, so this is silenced for the same reason.
    """
    x = np.asarray(x, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.maximum(x, 1e-300) / scale
        return (p_gid - 1.0) * np.log(z) - chi * (z + 1.0 / z) / 2.0 - np.log(2.0 * scale) - log_kv


_GIG_BIN_QUAD_NODES, _GIG_BIN_QUAD_WEIGHTS = np.polynomial.legendre.leggauss(64)
_GIG_CDF_SEGMENT_NODES, _GIG_CDF_SEGMENT_WEIGHTS = np.polynomial.legendre.leggauss(16)
_GIG_CDF_N_SEGMENTS = 24
_GIG_MIN_X = 1e-9


def _rounded_gig_bin_probability(observed: np.ndarray, shape_s: np.ndarray, beta, resolution_um: float, p_gid: float) -> np.ndarray:
    """Rounded-observation likelihood contribution for GIG, built directly
    from a fixed 64-point Gauss-Legendre integral of the closed-form density
    over each narrow (order resolution_um wide) rounding-kernel-shifted bin
    (NOT via generic_bin_prob_from_cdf + a from-cdf difference, which would
    need the slow/unreliable scipy CDF) -- otherwise structurally identical
    to gamma_process._aggregated_bin_probability's rounding-kernel
    marginalization (same ROUNDING_OFFSETS_UM/ROUNDING_WEIGHTS nodes).
    Verified to ~3e-7 max relative error against strict adaptive quadrature
    across a wide (p_gid, shape, scale) sweep including bins starting at
    the lower boundary (x=0, a real, retained data point -- zero-wear-
    increment windows) and dv=0, the one regime a low-order rule struggled
    with in testing (steep density variation near z=0 for negative p_gid).

    Fully vectorized (both the 11 rounding-kernel nodes and the 64 inner
    quadrature nodes are array axes, not Python loops) and computes
    K_p_gid(chi) exactly once per call (see _gig_log_pdf) -- both were
    needed to bring this down from ~500s to a practical fitting time per
    machine; see the module's implementation history for that investigation.
    """
    chi, scale = _gig_chi_scale_from_shape(shape_s, beta, p_gid)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_kv = np.log(kv(p_gid, chi))

    half = 0.5 * float(resolution_um)
    observed = np.asarray(observed, float)

    shift = (ROUNDING_OFFSETS_UM * resolution_um)[:, None]                    # (11, 1)
    lo = np.maximum(observed[None, :] - half - shift, _GIG_MIN_X)             # (11, n)
    hi = np.maximum(observed[None, :] + half - shift, _GIG_MIN_X + 1e-6)      # (11, n)
    mid = 0.5 * (lo + hi)
    halfw = 0.5 * (hi - lo)

    node = _GIG_BIN_QUAD_NODES[:, None, None]                                 # (64, 1, 1)
    weight = _GIG_BIN_QUAD_WEIGHTS[:, None, None]                             # (64, 1, 1)
    t = mid[None, :, :] + halfw[None, :, :] * node                            # (64, 11, n)

    log_pdf = _gig_log_pdf(t, chi[None, None, :], scale[None, None, :], p_gid, log_kv[None, None, :])
    bin_integral = np.sum(weight * np.exp(log_pdf), axis=0) * halfw           # (11, n)
    total = np.sum(ROUNDING_WEIGHTS[:, None] * bin_integral, axis=0)          # (n,)
    return np.maximum(total, np.finfo(float).tiny)


def _gig_cdf(x: np.ndarray, shape_s: np.ndarray, beta, p_gid: float) -> np.ndarray:
    """True CDF (0 to x), via composite log-spaced Gauss-Legendre (24
    segments x 16 points): geometrically-spaced segments concentrate
    resolution near x=0, where the density can vary steeply for negative
    p_gid, while coarser segments cover smooth, slowly-varying regions
    further out. Verified to ~4e-9 max relative error against a strict
    scipy.integrate.quad reference across 300 random (p_gid, shape, scale,
    x) combinations. Used only for PIT-KS calibration validation (once per
    machine, not the fitting hot path) -- correctness matters more than
    raw speed here, hence the more thorough (384-evaluation) scheme rather
    than the narrower-bin rule above; still fully vectorized (both the
    segment axis and the inner-node axis) and hoists K_p_gid(chi) exactly
    once per call, for the same reason as _rounded_gig_bin_probability.
    """
    chi, scale = _gig_chi_scale_from_shape(shape_s, beta, p_gid)
    x = np.maximum(np.asarray(x, float), _GIG_MIN_X)
    # x is broadcast against chi/scale (which carry shape_s's shape) BEFORE
    # any boolean-mask indexing below: every other caller of a family's
    # cdf_fn in this codebase passes matching-shape x/shape_s arrays (the
    # rounding-kernel/bin-probability context this was originally written
    # for), but _implied_survival (Table 4's age-replacement cost route)
    # calls cdf_fn(wear_threshold_um, shape_s, beta, ...) with a SCALAR x
    # (the one fixed EOL threshold) against an ARRAY shape_s (one value per
    # T_grid point) -- a real broadcasting case this function must support,
    # not just same-shape inputs. Without this, `x[valid]`/`out[valid]`
    # below raise IndexError the moment x is a bare scalar, silently never
    # triggered before GIG's cdf_fn was ever called from that route.
    x = np.broadcast_to(x, np.shape(chi))
    valid = np.isfinite(chi) & np.isfinite(scale) & (chi > 0) & (scale > 0)
    out = np.full(np.shape(chi), np.nan)
    if not np.any(valid):
        return out
    x_v, chi_v, scale_v = x[valid], chi[valid], scale[valid]
    log_kv_v = np.log(kv(p_gid, chi_v))

    eps = np.maximum(np.minimum(_GIG_MIN_X, x_v * 1e-6), 1e-12)
    ratio = x_v / eps

    seg = np.arange(_GIG_CDF_N_SEGMENTS)
    frac_lo = (seg / _GIG_CDF_N_SEGMENTS)[:, None]                            # (24, 1)
    frac_hi = ((seg + 1) / _GIG_CDF_N_SEGMENTS)[:, None]                      # (24, 1)
    lo = eps[None, :] * ratio[None, :] ** frac_lo                            # (24, m)
    hi = eps[None, :] * ratio[None, :] ** frac_hi                            # (24, m)
    mid = 0.5 * (lo + hi)
    halfw = 0.5 * (hi - lo)

    node = _GIG_CDF_SEGMENT_NODES[:, None, None]                             # (16, 1, 1)
    weight = _GIG_CDF_SEGMENT_WEIGHTS[:, None, None]                         # (16, 1, 1)
    t = mid[None, :, :] + halfw[None, :, :] * node                           # (16, 24, m)

    log_pdf = _gig_log_pdf(t, chi_v[None, None, :], scale_v[None, None, :], p_gid, log_kv_v[None, None, :])
    seg_integral = np.sum(weight * np.exp(log_pdf), axis=0) * halfw          # (24, m)
    total = np.sum(seg_integral, axis=0)                                     # (m,)
    out[valid] = np.minimum(total, 1.0)
    return out


def _gig_rvs(shape_s: np.ndarray, beta, rng: np.random.Generator, p_gid: float) -> np.ndarray:
    chi, scale = _gig_chi_scale_from_shape(shape_s, beta, p_gid)
    return geninvgauss.rvs(p_gid, chi, loc=0.0, scale=scale, random_state=rng)


def _fit_rounded_nh_gig(
    dv: np.ndarray, t_start: np.ndarray, t_end: np.ndarray, tool_ids: np.ndarray,
    v_start: np.ndarray | None = None, *,
    time_reference_s: float = TIME_REFERENCE_S, resolution_um: float = WEAR_RESOLUTION_UM,
) -> dict:
    """Fit a tool-heterogeneous non-homogeneous GIG process. Mirrors the
    other families' fitters exactly (same per-tool eta/power, shared beta,
    log-space Nelder-Mead with 3 restarts, non-homogeneous-vs-stationary
    AIC selection) plus one additional shared-per-machine parameter
    p_gid (fit on its natural, not log, scale -- it may be negative),
    initialized at -0.5 (the exact Inverse Gaussian point) so the search
    starts from the study's already-best-supported family rather than an
    arbitrary point. ``v_start`` is accepted for interface uniformity and
    ignored -- this family's tail shape does not depend on wear level.
    """
    dv = np.asarray(dv, float); t_start = np.asarray(t_start, float); t_end = np.asarray(t_end, float)
    tool_ids = np.asarray(tool_ids, int)
    _validate_windows(dv, t_start, t_end, tool_ids)

    tools = np.unique(tool_ids)
    tool_col = {int(tool): i for i, tool in enumerate(tools)}
    component_idx = np.array([tool_col[int(tool)] for tool in tool_ids], int)
    n_tools = len(tools)
    t_ref = float(time_reference_s)
    eta0, power0 = _init_eta_power(dv, t_start, t_end, tool_ids, tools, t_ref)
    beta0 = 0.5
    p_gid0 = -0.5

    def nll(z: np.ndarray, fixed_power: bool = False) -> float:
        eta = np.exp(z[:n_tools])
        if fixed_power:
            power = np.ones(n_tools, float)
            beta = float(np.exp(z[n_tools]))
            p_gid = float(z[n_tools + 1])
        else:
            power = np.exp(z[n_tools:2 * n_tools])
            beta = float(np.exp(z[-2]))
            p_gid = float(z[-1])
        if np.any(power < 0.05) or np.any(power > 3.0) or beta < 1e-4 or beta > 100.0 or abs(p_gid) > 6.0:
            return 1e100
        shape_s = eta[component_idx] * (
            np.power(t_end / t_ref, power[component_idx]) - np.power(t_start / t_ref, power[component_idx])
        )
        if np.any(shape_s <= 0) or not np.all(np.isfinite(shape_s)):
            return 1e100
        prob = _rounded_gig_bin_probability(dv, shape_s, beta, resolution_um, p_gid)
        if not np.all(np.isfinite(prob)):
            return 1e100
        return -float(np.sum(np.log(prob)))

    starts = [
        np.r_[np.log(eta0), np.log(power0), np.log(beta0), p_gid0],
        np.r_[np.log(eta0 * 0.7), np.log(np.full(n_tools, 0.6)), np.log(0.4), -1.0],
        np.r_[np.log(eta0 * 1.3), np.log(np.full(n_tools, 1.0)), np.log(0.7), 0.5],
    ]
    best = None
    for x0 in starts:
        res = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 150000, "xatol": 1e-9, "fatol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    if best is None or not np.isfinite(best.fun):
        raise RuntimeError("Non-homogeneous rounded GIG MLE failed.")

    eta = np.exp(best.x[:n_tools]); power = np.exp(best.x[n_tools:2 * n_tools])
    beta = float(np.exp(best.x[-2])); p_gid = float(best.x[-1])
    log_lik = -float(best.fun)

    stationary_start = np.r_[np.log(eta), np.log(beta), p_gid]
    stationary = minimize(
        lambda z: nll(z, fixed_power=True), stationary_start, method="Nelder-Mead",
        options={"maxiter": 80000, "xatol": 1e-9, "fatol": 1e-9},
    )
    if not np.isfinite(stationary.fun):
        raise RuntimeError("Non-homogeneous rounded GIG MLE (nested stationary null) failed.")

    n = len(dv)
    k_nh = 2 * n_tools + 2         # baseline IG's 2*n_tools+1, plus p_gid
    k_stationary = n_tools + 2     # baseline IG's n_tools+1, plus p_gid
    stationary_log_lik = -float(stationary.fun)
    nh_components = [
        {"tool_id": int(tool), "eta": float(eta[i]), "power": float(power[i]), "beta": beta,
         "p_gid": p_gid, "time_reference_s": t_ref}
        for i, tool in enumerate(tools)
    ]
    stationary_eta = np.exp(stationary.x[:n_tools]); stationary_beta = float(np.exp(stationary.x[n_tools]))
    stationary_p_gid = float(stationary.x[n_tools + 1])
    stationary_components = [
        {"tool_id": int(tool), "eta": float(stationary_eta[i]), "power": 1.0, "beta": stationary_beta,
         "p_gid": stationary_p_gid, "time_reference_s": t_ref}
        for i, tool in enumerate(tools)
    ]
    nh_aic = float(2 * k_nh - 2 * log_lik)
    stationary_aic = float(2 * k_stationary - 2 * stationary_log_lik)
    use_nonhomogeneous = bool(stationary_aic - nh_aic > 2.0)
    selected_components = nh_components if use_nonhomogeneous else stationary_components
    selected_beta = beta if use_nonhomogeneous else stationary_beta
    selected_p_gid = p_gid if use_nonhomogeneous else stationary_p_gid
    selected_log_lik = log_lik if use_nonhomogeneous else stationary_log_lik
    selected_k = k_nh if use_nonhomogeneous else k_stationary
    return {
        "model": "nonhomogeneous_gig_rounded", "schema_version": 1,
        "selected_age_model": "power_law" if use_nonhomogeneous else "linear",
        "beta": selected_beta,
        "eta": float(np.median([c["eta"] for c in selected_components])),
        "power": float(np.median([c["power"] for c in selected_components])),
        "p_gid": selected_p_gid,
        "time_reference_s": t_ref, "wear_resolution_um": float(resolution_um),
        "components": selected_components, "n_windows": int(n), "n_zero_windows": int(np.sum(dv == 0)),
        "log_lik": selected_log_lik,
        "aic": float(2 * selected_k - 2 * selected_log_lik),
        "bic": float(selected_k * np.log(n) - 2 * selected_log_lik),
        "nonhomogeneous_aic": nh_aic, "nonhomogeneous_bic": float(k_nh * np.log(n) - 2 * log_lik),
        "stationary_aic": stationary_aic, "stationary_bic": float(k_stationary * np.log(n) - 2 * stationary_log_lik),
        "nonhomogeneous_components": nh_components,
    }


def _fit_gig_adapter(dv, t_start, t_end, tool_ids, v_start=None, **kw):
    return _fit_rounded_nh_gig(dv, t_start, t_end, tool_ids, v_start, **kw)


families.register(
    "gig",
    label="Generalized Inverse Gaussian",
    color="#D55E00", ls=(0, (1, 1)), marker="H",
    fit_fn=_fit_gig_adapter,
    shape_fn=families.standard_shape_fn,
    predict_shape_fn=families.standard_predict_shape_fn,
    cdf_fn=_gig_cdf,
    bin_prob_fn=_rounded_gig_bin_probability,
    rvs_fn=_gig_rvs,
    is_subordinator=True,
    baseline_key="inverse_gaussian",
    extra_shape_params=["p_gid"],
    extra_params_desc="+1 shared tail-shape parameter p_gid per machine; nests plain Inverse Gaussian exactly at p_gid=-1/2 (verified to ~1e-14 against production invgauss)",
)
