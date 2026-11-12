"""Central family registry for the degradation process-family comparison.

Every degradation-process family or variant used in this study -- the
original Gamma, Inverse Gaussian, and Wiener processes (fit in
``gamma_process.py`` / ``degradation_process_models.py``) plus the seven
variants added in ``degradation_process_variants.py`` (state-dependent Gamma
and Inverse Gaussian, truncated Wiener, random-effects Gamma and Inverse
Gaussian, multi-stage Wiener, and Generalized Inverse Gaussian) -- registers
itself here exactly once, at import time. Every dispatch site that used to
hardcode a family tuple/dict
(``_FIT_FNS``, several separate ``{"gamma": "gamma_params", ...}`` literals,
Vuong's pair list, the per-family loops inside the table and figure
functions, ...) now derives that information from ``FAMILY_REGISTRY``
instead, so adding a family means adding one registration call rather than
hand-editing a dozen call sites -- the thing that made the original 3-family
dispatch pattern unsafe to extend by hand to 9 families.

Import structure (deliberately acyclic): this module imports only from
``gamma_process`` (the rounding-kernel primitives every family's likelihood
reuses) and the standard library / numpy / scipy. It does not import from
``degradation_process_models`` or ``degradation_process_variants`` --
instead, each of those two modules imports *this* module and calls
``register()`` for its own families at the bottom of the file, after its
fitters are defined. ``degradation_process_models.py`` additionally imports
``degradation_process_variants`` at its own top purely for that import's
side effect (triggering the six variants' registration), since it is the
module every downstream consumer (tables, plots, reliability translation,
``run_degradation_study.py``) already imports. This keeps the dependency
graph one-directional: gamma_process -> degradation_families ->
{degradation_process_variants} -> degradation_process_models -> everything
else. The three original fitters (``_fit_rounded_nh_gamma`` in
gamma_process.py, ``_fit_rounded_nh_ig``/``_fit_rounded_nh_wiener`` in
degradation_process_models.py) are never modified by this refactor, only
wrapped -- their golden-value-tested behavior is unchanged.

Registry contract
------------------
Every entry is a dict with these keys:

  key                str, matches the registry dict key (redundant but
                      convenient when a FamilySpec is passed around alone)
  label               display name (tables/figures)
  color, ls, marker   plot style (Okabe-Ito-derived; new variants reuse
                      their baseline family's color with a distinct
                      linestyle/marker so a reader can still group by family)
  fit_fn(dv, t_start, t_end, tool_ids, v_start=None) -> dict
                      uniform signature; ``v_start`` (wear level at each
                      window's start) is accepted by every family and
                      ignored by the ones that don't need it, so callers
                      never need to branch on family identity to decide
                      whether to pass it
  shape_fn(component, t_start, t_end, v_start=None) -> ndarray
                      per-tool (``component`` is that one tool's fitted
                      parameter dict), vectorized over windows; the
                      family's per-window shape quantity S, i.e. the
                      Gamma-shape / IG-shape / Wiener-mean-driving quantity
                      every family's increment distribution is built from
  predict_shape_fn(fitted, t_start, t_end, v_start=None) -> ndarray
                      cross-tool-median-based prediction for a tool held
                      out of fitting entirely (leave-one-tool-out CV);
                      ``fitted`` is the family's full fit dict (so it can
                      see every trained tool's component)
  cdf_fn(x, shape_s, beta) -> ndarray
                      the family's CDF, elementwise
  bin_prob_fn(observed, shape_s, beta, resolution_um) -> ndarray
                      rounded-observation likelihood contribution; for
                      every family here this is generated generically from
                      cdf_fn via ``generic_bin_prob_from_cdf`` except where
                      a family reuses an existing, already-tested
                      implementation directly (Gamma/IG/Wiener reuse the
                      production functions in gamma_process.py /
                      degradation_process_models.py verbatim)
  rvs_fn(shape_s, beta, rng) -> ndarray
                      draws ONE synthetic WINDOW-level increment per call
                      from the family's own fitted distribution -- always
                      well-defined for every family, including plain and
                      multi-stage Wiener (a single draw from the marginal
                      Normal has no monotonicity requirement), so every
                      family provides this. Used by model_recovery_study's
                      _simulate_synthetic_dv (one independent window at a
                      time, never accumulated). Deliberately NOT used for
                      cumulative-trajectory MTTFF step-simulation -- that is
                      a different question, gated by is_subordinator below,
                      not by whether rvs_fn exists.
  is_subordinator     bool. True for every family whose sample PATHS
                      (cumulative wear as a function of age, i.e. many
                      accumulated increments) are non-negative and non-
                      decreasing (Gamma, Inverse Gaussian, and every
                      state-dependent/random-effects/GIG variant of either,
                      plus truncated Wiener): these are genuine
                      subordinators, so (a) the marginal-distribution-
                      equals-survival-function identity used in
                      ``degradation_reliability_translation._implied_survival``
                      is exact for them, and (b) repeatedly calling rvs_fn
                      and ACCUMULATING the draws (cumulative-trajectory
                      MTTFF step-simulation) is valid, since there is no
                      clipping inconsistency to worry about. False for
                      plain and multi-stage Wiener, whose sample PATHS are
                      not monotone even though a single rvs_fn draw is
                      still perfectly well-defined -- these two use the
                      exact first-passage-time construction for MTTFF
                      instead (see the module docstring of
                      ``degradation_reliability_translation.py``), never
                      the marginal-as-survival shortcut and never
                      cumulative step-simulation
  baseline_key        str | None. For a variant, the family it nests inside
                      at its new parameter(s)' null value (e.g.
                      state-dependent Gamma nests plain Gamma at
                      gamma_state=0) -- used for the nested AIC/BIC/
                      likelihood-ratio comparison every variant is checked
                      against. None for the three original families.
  extra_params_desc   short human-readable description of what the variant
                      adds relative to its baseline, for manuscript/table
                      bookkeeping (e.g. "+1 shared gamma_state per machine")

``FAMILY_REGISTRY`` is an ordinary dict; insertion order is preserved
(Python 3.7+) and is used as the canonical display/iteration order
throughout: the three original families first (in their existing order),
then each variant immediately after its baseline family.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import chi2

from gamma_process import _aggregated_bin_probability

EOL_THRESHOLD_UM = 150.0

FAMILY_REGISTRY: dict[str, dict] = {}


def register(key: str, **fields) -> None:
    """Register one family/variant. Raises on a duplicate key (a duplicate
    registration is always a bug -- either an accidental re-import side
    effect or a copy-pasted key -- never a legitimate re-registration).

    ``extra_shape_params`` (optional, default ``[]``) names any additional
    per-machine fitted scalars (beyond eta/power/beta) a family's
    ``cdf_fn``/``bin_prob_fn`` need on top of ``(shape_s, beta)`` -- e.g.
    the Generalized Inverse Gaussian variant's tail-shape parameter
    ``p_gid``. For such a family, ``cdf_fn(x, shape_s, beta, *extra)`` and
    ``bin_prob_fn(observed, shape_s, beta, resolution_um, *extra)`` accept
    these as ADDITIONAL positional arguments, supplied in the listed order
    by whatever fitted-parameter dict the caller has on hand (looked up by
    name, e.g. ``fitted["p_gid"]``). Every one of the other 9 families
    leaves this empty, so their ``cdf_fn``/``bin_prob_fn`` keep the exact
    3-arg / 4-arg signature they already had -- this is a purely additive
    extension for families that need it, not a change to the existing
    interface."""
    required = {
        "label", "color", "ls", "marker", "fit_fn", "shape_fn",
        "predict_shape_fn", "cdf_fn", "bin_prob_fn", "rvs_fn",
        "is_subordinator", "baseline_key", "extra_params_desc",
    }
    missing = required - fields.keys()
    if missing:
        raise ValueError(f"register({key!r}): missing required field(s) {sorted(missing)}")
    if key in FAMILY_REGISTRY:
        raise ValueError(f"register({key!r}): family already registered")
    fields.setdefault("extra_shape_params", [])
    fields.setdefault("mean_increment_fn", standard_mean_increment)
    FAMILY_REGISTRY[key] = {"key": key, **fields}


def extra_shape_values(spec: dict, fitted: dict) -> list:
    """Look up a family's ``extra_shape_params`` (see ``register``) by name
    in a fitted-parameter dict, in registration order -- the values every
    generalized downstream consumer (Vuong's test, PIT-KS validation,
    leave-one-tool-out CV) passes as the trailing positional args to that
    family's ``cdf_fn``/``bin_prob_fn``. Returns ``[]`` for every family
    that doesn't declare any (i.e. every family except GIG today), so
    callers can use this unconditionally without a per-family branch."""
    return [fitted[name] for name in spec.get("extra_shape_params", [])]


def registered_with_params(comparison_result: dict) -> list[str]:
    """Family keys that are both registered AND present in a given
    ``comparison_result``-shaped dict (i.e. it has a ``f"{key}_params"``
    entry), in registry order.

    Every generalized dispatch function iterates this instead of
    ``FAMILY_REGISTRY`` directly, so a ``comparison_result`` that only
    covers a subset of the full registry (an older artifact reloaded from
    disk before every variant was added, or a minimal hand-built test
    fixture) is handled gracefully rather than raising a KeyError -- this
    is not just test-friendliness, it is the genuinely correct behavior
    for a codebase where the registry is expected to keep growing.
    """
    return [k for k in FAMILY_REGISTRY if param_key(k) in comparison_result]


# Pre-registry code used the abbreviation "ig_params" (not the regular
# "inverse_gaussian_params") for the Inverse Gaussian family's key in
# comparison_result. That spelling is baked into ~16 call sites across
# degradation_process_models.py, degradation_plots.py,
# degradation_reliability_translation.py, and two test modules -- rewriting
# all of them (including tests whose job is to pin exact dict shapes) would
# trade one localized irregularity for a wide, error-prone edit. param_key()
# special-cases this one family so every NEW/generic call site stays
# consistent with the pre-existing on-disk convention instead of silently
# missing it.
_IRREGULAR_PARAM_KEYS = {"inverse_gaussian": "ig_params"}


def param_key(family: str) -> str:
    """The ``comparison_result`` dict key holding a family's fitted params,
    e.g. ``"gamma"`` -> ``"gamma_params"``. Centralizes a naming convention
    that used to be reimplemented as ~5 separate identical dict literals.
    ``"inverse_gaussian"`` is the one irregular case -- see
    ``_IRREGULAR_PARAM_KEYS``."""
    return _IRREGULAR_PARAM_KEYS.get(family, f"{family}_params")


# ── Generic, reusable building blocks ──────────────────────────────────────

def generic_bin_prob_from_cdf(cdf_fn: Callable) -> Callable:
    """Build a ``bin_prob_fn`` from a family's ``cdf_fn`` via the shared
    two-endpoint rounding-kernel correction (``gamma_process._aggregated_bin_probability``),
    clamped away from exactly zero. Every family registered from
    ``degradation_process_variants.py`` uses this rather than reimplementing
    the rounding-kernel loop -- the three original families keep their own
    existing, already-tested ``_rounded_*_bin_probability`` implementations
    instead (behaviorally identical to what this would generate, but left
    untouched deliberately).

    ``*extra`` (empty for every family except GIG) is forwarded verbatim to
    ``cdf_fn`` after ``(x, shape_s, beta)`` -- see ``register``'s docstring
    on ``extra_shape_params``."""
    def _bin_prob(observed: np.ndarray, shape_s: np.ndarray, beta: float, resolution_um: float, *extra) -> np.ndarray:
        def _cdf(x):
            return cdf_fn(x, shape_s, beta, *extra)
        prob = _aggregated_bin_probability(observed, _cdf, resolution_um)
        return np.maximum(prob, np.finfo(float).tiny)
    return _bin_prob


def standard_shape_fn(component: dict, t_start: np.ndarray, t_end: np.ndarray, v_start: np.ndarray | None = None) -> np.ndarray:
    """The age-only power-law shape quantity S(s,t) = eta*[(t/t_ref)^p -
    (s/t_ref)^p] shared by Gamma, Inverse Gaussian, Wiener, truncated
    Wiener, and (for downstream/predictive use -- see the module
    docstring's note on random effects) random-effects Gamma and Inverse
    Gaussian alike. ``v_start`` is accepted for interface uniformity with
    ``degradation_families``'s other ``shape_fn`` implementations and
    ignored -- this family's rate does not depend on current wear level."""
    eta = float(component["eta"])
    power = float(component["power"])
    t_ref = float(component["time_reference_s"])
    shape_s = eta * (np.power(t_end / t_ref, power) - np.power(t_start / t_ref, power))
    return np.maximum(shape_s, 1e-12)


def standard_predict_shape_fn(fitted: dict, t_start: np.ndarray, t_end: np.ndarray, v_start: np.ndarray | None = None) -> np.ndarray:
    """Cross-tool median-eta/median-power prediction for a tool held out of
    fitting entirely, matching the leave-one-tool-out CV convention already
    used for the three original families (median across the *trained*
    tools' fitted components stands in for the unseen tool's own, since it
    has none)."""
    eta_med = float(np.median([c["eta"] for c in fitted["components"]]))
    power_med = float(np.median([c["power"] for c in fitted["components"]]))
    t_ref = float(fitted["time_reference_s"])
    shape_s = eta_med * (np.power(t_end / t_ref, power_med) - np.power(t_start / t_ref, power_med))
    return np.maximum(shape_s, 1e-12)


def standard_mean_increment(shape_s: np.ndarray, beta) -> np.ndarray:
    """E[wear increment | shape_s, beta] = shape_s/beta -- the shared
    mean/variance convention every family in this study is built around
    (Eq. 2b), true for Gamma, Inverse Gaussian, plain Wiener, GIG (by
    explicit mean-preserving construction, see its own module docstring),
    and, conditional on the aging exponent, the two random-effects
    variants. The default for ``mean_increment_fn``, used by every family
    that does not override it (today, every one except truncated Wiener --
    see ``degradation_process_variants._truncated_wiener_mean_increment``).

    Used wherever a diagnostic needs "the predicted mean wear a fitted
    process implies" without simulating a full trajectory (currently
    ``degradation_process_models._pit_ks_generic``'s trajectory-RMSE/
    terminal-bias check) -- getting this formula right per family matters:
    an early version of this generalization used this SAME formula
    unconditionally for every family, which is exactly correct for 9 of
    10 but silently wrong for truncated Wiener specifically (whose actual
    mean, after conditioning on non-negativity, is systematically HIGHER
    than shape_s/beta -- truncating away the discarded negative tail shifts
    the mean up, which is the whole reason that family exists). Left
    unfixed, that mismatch manufactured a large, purely diagnostic-side
    negative terminal bias having nothing to do with the fit's actual
    quality (confirmed on real data: naive formula gave ~17um where the
    true truncated mean is ~31um for one real fitted component, an
    error that compounds additively across an entire trajectory)."""
    return shape_s / beta


def standard_shape_families(comparison_result: dict) -> list[str]:
    """Registered-and-fitted family keys (see ``registered_with_params``)
    whose ``shape_fn`` is the shared age-only ``standard_shape_fn`` -- i.e.
    families whose cumulative-wear shape quantity depends only on age, not
    on the process's own realized path. This is exactly the set that
    ``degradation_reliability_translation._implied_survival`` can evaluate
    in closed form (direct marginal for subordinators, first-passage-time
    for plain Wiener) and that a virtual-age-inversion RUL calculation can
    invert algebraically -- so every reliability-translation table (implied
    MTTFF, conditional failure risk, age-replacement cost, RUL-vs-ML
    comparison) that isn't doing full trajectory simulation should iterate
    THIS set, not a hardcoded family list, to stay in sync with however
    many families the registry grows to.

    Excludes the two state-dependent families (their shape depends on the
    realized wear level, not age alone -- ``simulate_process_lifetimes``'s
    step simulation handles those instead) AND ``wiener_multistage``: its
    shape function is genuinely piecewise in age via a fitted change-point,
    which disqualifies it from BOTH routes -- not ``standard_shape_fn`` (so
    no closed form here) and not ``is_subordinator`` (so
    ``simulate_process_lifetimes`` also rejects it, see its own
    docstring/tests). No tractable first-passage or unbiased-simulation
    treatment is implemented for the piecewise process yet; this is a
    known, explicitly-documented limitation of the reliability-translation
    tables specifically (Table 1's AIC/BIC/Vuong/validation/model-recovery
    comparison, which needs no survival function, already fully covers
    wiener_multistage).
    """
    return [
        k for k in registered_with_params(comparison_result)
        if FAMILY_REGISTRY[k]["shape_fn"] is standard_shape_fn
    ]


def subordinator_families(comparison_result: dict) -> list[str]:
    """Registered-and-fitted family keys (see ``registered_with_params``)
    with ``is_subordinator=True`` -- genuinely non-negative, non-decreasing
    sample paths, so step-by-step Monte Carlo accumulation in
    ``degradation_reliability_translation.simulate_process_lifetimes``
    faithfully reproduces the fitted process (no clipping bias). Every
    table built from that function's output (implied MTTFF, conditional
    failure risk) should iterate THIS set.

    Excludes plain and multi-stage Wiener (Normal increments, can go
    negative -- clipping them to simulate would silently bias the
    trajectory monotone, inconsistent with how they were actually fit; see
    ``simulate_process_lifetimes``'s own docstring). Plain Wiener's implied
    MTTFF/survival is still fully computable, just via the closed-form
    first-passage route in ``_implied_survival`` instead (used by
    ``age_replacement_cost_rate`` / Table 4) rather than this
    Monte-Carlo-based route -- consistent with the ORIGINAL 3-family
    paper's own design (Table 3 was Gamma-vs-IG only from the start, never
    Wiener, for exactly this reason, well before the registry grew to 10).
    Multi-stage Wiener has no route here for the same reason it has none
    in ``standard_shape_families``: see that function's docstring.
    """
    return [
        k for k in registered_with_params(comparison_result)
        if FAMILY_REGISTRY[k]["is_subordinator"]
    ]


def nested_model_test(
    baseline_fit: dict, variant_fit: dict, extra_params: int, *, boundary_corrected: bool = False,
) -> dict:
    """Formal nested-model comparison between a variant and the baseline
    family it reduces to at its new parameter(s)' null value: delta-AIC,
    delta-BIC (both baseline-minus-variant, so positive favors the
    variant, matching the sign convention already used throughout this
    study's stationary-vs-non-homogeneous AIC comparisons), and a proper
    likelihood-ratio chi-square test (exact for genuinely nested models,
    unlike the AIC/BIC heuristic thresholds used for the non-nested
    Gamma/IG/Wiener family comparison elsewhere in this study) -- twice the
    log-likelihood improvement is chi-square distributed with
    ``extra_params`` degrees of freedom under the null that the variant's
    extra parameter(s) are truly zero.

    ``boundary_corrected=True`` applies the standard fix (Self & Liang,
    1987) for testing a VARIANCE COMPONENT at its zero boundary --
    specifically the random-effects variants' own sigma_p=0-vs-free
    selection: the null hypothesis sits at the edge of the parameter space
    (a variance cannot be negative), which violates the regularity
    conditions the plain chi-square asymptotic result assumes. The
    corrected null distribution of the LR statistic is a 50:50 mixture of a
    point mass at 0 and chi-square(``extra_params``), which for a single
    variance component (``extra_params=1``) means the correct p-value is
    exactly HALF the naive chi-square(1) p-value -- using the uncorrected
    value here would systematically overstate significance. Do not set this
    for the state-dependent or multi-stage variants' own nested tests
    (gamma_state and the multi-stage rate-equality test are ordinary,
    non-boundary parameters -- gamma_state can be negative, and the
    multi-stage test is of an equality constraint, not a
    variance-at-zero constraint -- so the plain chi-square applies there).

    Requires ``variant_fit["log_lik"] >= baseline_fit["log_lik"]`` up to a
    small numerical tolerance (a strictly nested model cannot fit worse
    than its own special case at the true MLE; a violation beyond
    optimizer noise indicates the variant's optimizer failed to find at
    least as good a point as the baseline, not a real result) -- clipped to
    zero rather than raising, since occasional small negative slack from
    independent Nelder-Mead searches is optimizer noise, not a modeling
    error, and should not crash a long-running sweep.
    """
    ll_gain = max(0.0, float(variant_fit["log_lik"]) - float(baseline_fit["log_lik"]))
    lr_stat = 2.0 * ll_gain
    p_value = float(chi2.sf(lr_stat, df=extra_params))
    if boundary_corrected:
        if extra_params != 1:
            raise ValueError("boundary_corrected assumes a single variance component (extra_params=1).")
        p_value = 0.5 * p_value
    return {
        "baseline_log_lik": float(baseline_fit["log_lik"]),
        "variant_log_lik": float(variant_fit["log_lik"]),
        "delta_aic_baseline_minus_variant": float(baseline_fit["aic"] - variant_fit["aic"]),
        "delta_bic_baseline_minus_variant": float(baseline_fit["bic"] - variant_fit["bic"]),
        "lr_statistic": lr_stat,
        "lr_df": int(extra_params),
        "lr_p_value": p_value,
        "lr_boundary_corrected": bool(boundary_corrected),
        "variant_supported_aic": bool(baseline_fit["aic"] - variant_fit["aic"] > 2.0),
        "variant_supported_lr_05": bool(p_value < 0.05),
    }


__all__ = [
    "EOL_THRESHOLD_UM", "FAMILY_REGISTRY", "register", "extra_shape_values",
    "registered_with_params", "param_key", "generic_bin_prob_from_cdf",
    "standard_shape_fn", "standard_predict_shape_fn", "standard_mean_increment",
    "standard_shape_families", "subordinator_families", "nested_model_test",
]
