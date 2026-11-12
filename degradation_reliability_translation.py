"""Reliability translation for the degradation process-family comparison.

Converts real tool-life trajectories into the list-of-dicts format expected
by ``reliability_analysis.py`` (``simTime`` / ``firstFailureTime``, exactly
as MARL episode results use it) and compares:

  1. Ground-truth per-machine MTTFF/Weibull fit from the 9 real tool lives.
  2. Process-implied MTTFF/Weibull fit obtained by Monte Carlo simulating
     synthetic tool lives under the fitted Gamma and Inverse Gaussian
     degradation processes (Section: does the process-family choice change
     the practical reliability conclusion?).

Only 9 independent tool trajectories exist (3 per machine); ground-truth
bootstrap CIs are correspondingly wide, especially for per-machine subsets
(n=3). This is a real constraint of the dataset, reported honestly via
``mttff_bootstrap_ci`` throughout rather than suppressed.
"""

from __future__ import annotations

import ast
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import invgauss

import degradation_families as families
from degradation_families import EOL_THRESHOLD_UM
from gamma_process import build_wear_windows
from degradation_process_models import _FIT_FNS
from reliability_analysis import compute_mttff, fit_weibull_mttff, mttff_bootstrap_ci


# ── Ground truth from real trajectories ────────────────────────────────────────

def build_tool_life_episodes(
    index: pd.DataFrame,
    wear_threshold_um: float = EOL_THRESHOLD_UM,
) -> dict[str, list[dict]]:
    """One "episode" dict per physical tool: {simTime, firstFailureTime, tool_id}.

    ``firstFailureTime`` is the earliest observed run at which wear reaches
    ``wear_threshold_um``. Tools whose terminal recorded wear never reaches
    the threshold (this dataset's terminal wear ranges 145-160um across
    tools) are right-censored at their last observed contact time, exactly
    as ``reliability_analysis.py`` expects for any other censored unit.
    """
    episodes_by_machine: dict[str, list[dict]] = {"M1": [], "M2": [], "M3": []}
    for (machine, tool), traj in index.groupby(["machine", "tool"]):
        traj = traj.sort_values("run")
        times = traj["cumulated_tool_contact_time"].to_numpy(float)
        wear = traj["wear"].to_numpy(float)
        reached = np.nonzero(wear >= wear_threshold_um)[0]
        sim_time = float(times[-1])
        first_failure_time = float(times[reached[0]]) if len(reached) else None
        machine_key = f"M{int(machine)}"
        episodes_by_machine[machine_key].append({
            "simTime": sim_time,
            "firstFailureTime": first_failure_time,
            "tool_id": int(tool),
        })
    return episodes_by_machine


def ground_truth_reliability(
    index: pd.DataFrame,
    wear_threshold_um: float = EOL_THRESHOLD_UM,
    n_boot: int = 1000,
    rng_seed: int = 0,
) -> dict[str, dict]:
    """Ground-truth per-machine MTTFF/Weibull fit with tool-level bootstrap CIs."""
    episodes_by_machine = build_tool_life_episodes(index, wear_threshold_um)
    out = {}
    for machine_key, episodes in episodes_by_machine.items():
        blocks = [[episode] for episode in episodes]  # 1 tool = 1 independent block
        out[machine_key] = {
            "n_tools": len(episodes),
            "mttff": compute_mttff(episodes),
            "weibull": fit_weibull_mttff(episodes),
            "mttff_bootstrap_ci": mttff_bootstrap_ci(
                episodes, blocks=blocks, n_boot=n_boot, rng_seed=rng_seed,
            ),
            "episodes": episodes,
        }
    return out


# ── Process-implied synthetic tool lives (Monte Carlo) ─────────────────────────

def simulate_process_lifetimes(
    params_for_machine: dict,
    family: str,
    n_sim: int = 500,
    dt_s: float = 100.0,
    max_time_s: float = 12000.0,
    wear_threshold_um: float = EOL_THRESHOLD_UM,
    rng_seed: int = 0,
) -> list[dict]:
    """Monte Carlo simulate synthetic tool lives under a fitted degradation process.

    One fitted tool component is drawn per simulated lifetime (a simulated
    tool life follows one fixed component's wear-rate trajectory throughout,
    just as a real physical tool does). Wear increments over each ``dt_s``
    step are drawn from the family's own registered ``rvs_fn``, using the
    same shared mean/variance convention as the fits themselves (mean=S/beta,
    Var=S/beta^2 for Gamma/IG-family members). Valid for every
    ``is_subordinator=True`` family (Gamma, Inverse Gaussian, truncated
    Wiener, and every state-dependent/random-effects/GIG variant of either)
    -- their sample paths are genuinely non-negative and non-decreasing, so
    accumulating draws step by step faithfully simulates the fitted process
    rather than approximating it. Plain and multi-stage Wiener are NOT
    subordinators (non-monotone paths); their MTTFF uses the exact
    first-passage-time closed form in ``_implied_survival`` instead (see
    this module's docstring) -- calling this function for either raises.

    State-dependent families (gamma_state, inverse_gaussian_state) feed the
    running simulated wear level back into the next step's shape quantity,
    exactly reproducing the fitted feedback loop -- this is the one case
    where Monte Carlo captures something the closed-form marginal route in
    ``_implied_survival`` fundamentally cannot (a path-dependent process's
    marginal at age t is not a function of t alone). Returns episodes in the
    same {simTime, firstFailureTime} format as the ground truth so the same
    reliability_analysis.py functions apply unchanged.

    For truncated Wiener specifically, ``dt_s`` is not a free precision
    knob the way it is for the other subordinators here: unlike Gamma/IG
    (genuine Levy subordinators, infinitely divisible -- summing many
    small independent increments has EXACTLY the same law as one large
    increment over the combined interval), a left-truncated normal is NOT
    closed under convolution, so this family's simulated trajectories are
    only faithful to how it was actually FIT if ``dt_s`` produces steps
    comparable to the fitted ``window_size`` (in real elapsed time, not run
    count). A real-data investigation found this the hard way in a
    different diagnostic (``degradation_process_models._pit_ks_generic``):
    accumulating this family's mean at native per-observation granularity
    over-predicted cumulative wear by >300um, while one big aggregate-shape
    step under-predicted by >100um, for the SAME fitted parameters -- the
    only granularity that reproduces the fit is the one it was fit at.
    Before using this function for truncated Wiener's MTTFF specifically,
    pick ``dt_s`` accordingly (or note the mismatch explicitly) rather than
    trusting the default. The other 9 families (all genuine subordinators)
    have no such sensitivity -- any dt_s gives the same expected trajectory.
    """
    if family not in families.FAMILY_REGISTRY:
        raise ValueError(f"Unknown family: {family!r}")
    spec = families.FAMILY_REGISTRY[family]
    if not spec["is_subordinator"]:
        raise ValueError(
            f"{family!r} is not a subordinator (non-monotone sample paths); "
            "use the exact first-passage-time route in _implied_survival instead."
        )
    components = params_for_machine["components"]
    beta = float(params_for_machine["beta"])
    t_ref = float(params_for_machine.get("time_reference_s", 1000.0))
    extra = families.extra_shape_values(spec, params_for_machine)
    # Only state-dependent variants deviate from the shared age-only shape
    # formula among is_subordinator=True families (multi-stage Wiener's
    # custom shape_fn is excluded above via the is_subordinator gate).
    state_dependent = spec["shape_fn"] is not families.standard_shape_fn
    rng = np.random.default_rng(rng_seed)

    # Vectorised over the n_sim trajectories: one call per time step
    # (not one per trajectory per step) -- otherwise a full study can require
    # >1e6 individual scipy.stats.rvs() calls and take tens of minutes.
    comp_idx = rng.integers(0, len(components), size=n_sim)
    eta = np.array([float(components[i]["eta"]) for i in comp_idx])
    power = np.array([float(components[i]["power"]) for i in comp_idx])
    if state_dependent:
        gamma_state = np.array([float(components[i]["gamma_state"]) for i in comp_idx])

    wear = np.zeros(n_sim)
    failure_time = np.full(n_sim, np.nan)
    alive = np.ones(n_sim, dtype=bool)
    t = 0.0
    n_steps = int(np.ceil(max_time_s / dt_s))
    for _step in range(n_steps):
        if not np.any(alive):
            break
        t_next = t + dt_s
        shape_s = eta * ((t_next / t_ref) ** power - (t / t_ref) ** power)
        if state_dependent:
            # Matches degradation_process_variants._state_dependent_shape_fn
            # exactly, including normalizing by the family's FITTED
            # convention constant (families.EOL_THRESHOLD_UM) rather than
            # this function's own (possibly swept) wear_threshold_um
            # failure-crossing parameter -- gamma_state was estimated
            # against the fixed constant, not whatever threshold a caller
            # here happens to pass.
            shape_s = shape_s * (1.0 + gamma_state * (wear / EOL_THRESHOLD_UM))
        shape_s = np.maximum(shape_s, 1e-12)
        dv = spec["rvs_fn"](shape_s, beta, rng, *extra)
        dv = np.maximum(dv, 0.0)
        new_wear = wear + dv
        crossing = alive & (new_wear >= wear_threshold_um)
        if np.any(crossing):
            frac = (wear_threshold_um - wear[crossing]) / np.maximum(
                new_wear[crossing] - wear[crossing], 1e-9
            )
            failure_time[crossing] = t + frac * dt_s
            alive[crossing] = False
        wear = np.where(alive, new_wear, wear)
        t = t_next

    sim_time = np.where(np.isnan(failure_time), t, failure_time)
    episodes = [
        {
            "simTime": float(sim_time[i]),
            "firstFailureTime": None if np.isnan(failure_time[i]) else float(failure_time[i]),
        }
        for i in range(n_sim)
    ]
    return episodes


def process_implied_reliability(
    comparison_result: dict,
    n_sim: int = 500,
    dt_s: float = 100.0,
    max_time_s: float = 12000.0,
    n_boot: int = 1000,
    rng_seed: int = 0,
    wear_threshold_um: float = EOL_THRESHOLD_UM,
    families_to_compare: tuple[str, ...] | None = None,
) -> dict[str, dict]:
    """Process-implied MTTFF/Weibull fit under each of ``families_to_compare``
    (default: ``None``, meaning every ``families.subordinator_families``
    member -- every registered family Monte Carlo simulation can faithfully
    handle, not just Gamma/IG, the paper's ORIGINAL 3-family-comparison
    headline pair; see that function's docstring for exactly which 2 of the
    10 registered families are excluded and why).

    ``comparison_result`` is ``degradation_process_models.fit_and_compare_degradation_models()``'s
    return value. Every member of ``families_to_compare``, if passed
    explicitly, must be ``is_subordinator=True`` (see
    ``simulate_process_lifetimes``) -- pass a narrower tuple explicitly to
    reproduce the original Gamma/IG-only behavior.

    Reuses each family's already-fitted params from ``comparison_result``
    (no new fitting), so covering every eligible family instead of 2 adds
    only more (cheap, vectorized) forward simulation, not more optimization.
    """
    if families_to_compare is None:
        families_to_compare = families.subordinator_families(comparison_result)
    out = {}
    for machine_key in sorted(comparison_result["gamma_params"]):
        machine_out = {}
        for family in families_to_compare:
            params_for_machine = comparison_result[families.param_key(family)][machine_key]
            episodes = simulate_process_lifetimes(
                params_for_machine, family=family, n_sim=n_sim, dt_s=dt_s,
                max_time_s=max_time_s, rng_seed=rng_seed,
                wear_threshold_um=wear_threshold_um,
            )
            machine_out[family] = {
                "n_sim": n_sim,
                "mttff": compute_mttff(episodes),
                "weibull": fit_weibull_mttff(episodes),
                "mttff_bootstrap_ci": mttff_bootstrap_ci(
                    episodes, n_boot=n_boot, rng_seed=rng_seed,
                ),
            }
        out[machine_key] = machine_out
    return out


def _implied_survival(
    params_for_machine: dict,
    family: str,
    t_grid: np.ndarray,
    wear_threshold_um: float = EOL_THRESHOLD_UM,
) -> np.ndarray:
    """Closed-form process-implied survival S(t) = P(cumulative wear at age t < threshold).

    Two distinct closed forms are used, dispatched by the family's registry
    entry:

    - Subordinator families using the shared age-only shape formula (Gamma,
      Inverse Gaussian, truncated Wiener, random-effects Gamma/IG, GIG) are
      Levy-type processes started at 0, so cumulative wear at age t is
      exactly distributed as the family's law with shape S(0, t) -- no
      simulation needed.
    - Plain and multi-stage Wiener are NOT subordinators (non-monotone
      paths), so the marginal-distribution-at-age-t shortcut above does not
      give a survival function for them. Instead this uses the EXACT
      first-passage-time distribution: time-changing the process to a
      standard Brownian motion with drift 1/beta shows the first-passage
      time to the fixed barrier ``wear_threshold_um`` is Inverse-Gaussian in
      S-units with mean=threshold*beta, shape=(threshold*beta)^2 (derived
      two independent ways -- direct FPT-to-barrier formula and the
      time-change argument -- and cross-checked against brute-force
      fine-step Monte Carlo of the actual first-crossing time; see
      tests/test_degradation_reliability_translation.py). This replaces a
      previous approximation that used the marginal Normal CDF directly,
      which is only exact for monotone processes.

    State-dependent families (gamma_state, inverse_gaussian_state) have
    neither closed form available -- their shape at age t depends on the
    process's own realized past, not t alone -- and raise; use
    ``simulate_process_lifetimes`` for those instead.

    Each tool's component is weighted equally and averaged to obtain the
    machine-level ensemble survival curve.
    """
    if family not in families.FAMILY_REGISTRY:
        raise ValueError(f"Unknown family: {family!r}")
    spec = families.FAMILY_REGISTRY[family]
    if spec["shape_fn"] is not families.standard_shape_fn:
        raise ValueError(
            f"{family!r} has no closed-form survival function (its shape "
            "depends on the process's own realized path, not age alone); "
            "use simulate_process_lifetimes for a Monte Carlo estimate instead."
        )
    components = params_for_machine["components"]
    beta = float(params_for_machine["beta"])
    t_ref = float(params_for_machine.get("time_reference_s", 1000.0))
    extra = families.extra_shape_values(spec, params_for_machine)
    survivals = np.zeros((len(components), len(t_grid)))
    for ci, comp in enumerate(components):
        eta = float(comp["eta"])
        power = float(comp["power"])
        shape_s = eta * np.power(np.maximum(t_grid, 1e-9) / t_ref, power)
        shape_s = np.maximum(shape_s, 1e-12)
        if spec["is_subordinator"]:
            # P(cumulative wear <= threshold) = P(tool still alive) = survival.
            survival_i = spec["cdf_fn"](wear_threshold_um, shape_s, beta, *extra)
        else:
            survival_i = 1.0 - invgauss.cdf(
                shape_s, mu=1.0 / (wear_threshold_um * beta), loc=0.0,
                scale=(wear_threshold_um * beta) ** 2,
            )
        survivals[ci] = survival_i
    return np.mean(survivals, axis=0)


def conditional_failure_risk(
    comparison_result: dict,
    reference_ages_s: list[float],
    lookahead_s: float = 500.0,
    wear_threshold_um: float = EOL_THRESHOLD_UM,
    families_to_compare: tuple[str, ...] | None = None,
) -> dict[str, dict]:
    """P(fail within [t_ref, t_ref+lookahead] | survived to t_ref), across
    ``families_to_compare`` (default: ``None``, meaning every
    ``families.standard_shape_families`` member -- every registered family
    with a closed-form survival function, not just the original Gamma/IG
    pair; ``_implied_survival`` is the closed-form call this function
    makes, so eligibility here matches Table 4's exactly, not Table 3's --
    plain Wiener IS included).

    This isolates the aging-vs-stationary distinction more directly than the
    MTTFF point estimate: two processes with near-identical mean life can
    still imply very different near-term failure risk profiles depending on
    whether the fitted age exponent supports acceleration.
    """
    if families_to_compare is None:
        families_to_compare = families.standard_shape_families(comparison_result)
    out = {}
    for machine_key in sorted(comparison_result["gamma_params"]):
        machine_out = {}
        for family in families_to_compare:
            if machine_key not in comparison_result[families.param_key(family)]:
                continue
            params_for_machine = comparison_result[families.param_key(family)][machine_key]
            rows = []
            for t_ref in reference_ages_s:
                grid = np.array([t_ref, t_ref + lookahead_s])
                s = _implied_survival(params_for_machine, family, grid, wear_threshold_um)
                s_t, s_t_dt = float(s[0]), float(s[1])
                risk = 0.0 if s_t <= 1e-12 else float(1.0 - s_t_dt / s_t)
                rows.append({
                    "t_ref_s": float(t_ref), "lookahead_s": float(lookahead_s),
                    "survival_at_t_ref": s_t, "conditional_failure_risk": risk,
                })
            machine_out[family] = rows
        out[machine_key] = machine_out
    return out


def summarize_reliability_translation(
    index: pd.DataFrame,
    comparison_result: dict,
    n_sim: int = 500,
    n_boot: int = 1000,
    rng_seed: int = 0,
    wear_threshold_um: float = EOL_THRESHOLD_UM,
) -> dict:
    """Full translation: ground truth vs. every process-implied MTTFF (every
    ``families.subordinator_families`` member -- see ``process_implied_reliability``,
    not just the original Gamma/IG headline pair), plus a conditional-failure-risk
    comparison at 20/50/80% of each machine's median observed tool life (the
    near-term-risk metric that should actually separate an aging from a
    stationary process, unlike the MTTFF point estimate which Gamma and IG,
    the two families this comparison originally shipped with, turn out to
    agree on closely)."""
    ground_truth = ground_truth_reliability(
        index, wear_threshold_um=wear_threshold_um, n_boot=n_boot, rng_seed=rng_seed,
    )
    implied = process_implied_reliability(
        comparison_result, n_sim=n_sim, n_boot=n_boot, rng_seed=rng_seed,
        wear_threshold_um=wear_threshold_um,
    )
    summary = {}
    hazard = {}
    for machine_key in sorted(ground_truth):
        gt = ground_truth[machine_key]
        gt_ci = gt["mttff_bootstrap_ci"]
        row = {
            "ground_truth_mttff_s": gt["mttff"]["mttff"],
            "ground_truth_mttff_ci_lo": gt_ci.get("mttff_ci_lo"),
            "ground_truth_mttff_ci_hi": gt_ci.get("mttff_ci_hi"),
            "ground_truth_n_tools": gt["n_tools"],
            # When all_censored, "mttff" is the KM-RMST evaluated at tau (the
            # largest observed, still-right-censored contact time) -- a lower
            # bound on true MTTFF, not a full estimate the way a machine with
            # >=1 observed failure gets. Surfaced explicitly (not just buried
            # in the underlying compute_mttff() call) so a reader of Table 3
            # doesn't read a fully-censored row as commensurable with the
            # others without noticing.
            "ground_truth_n_observed": gt["mttff"]["n_observed"],
            "ground_truth_all_censored": gt["mttff"]["all_censored"],
        }
        for family in implied[machine_key]:
            fam = implied[machine_key][family]
            fam_ci = fam["mttff_bootstrap_ci"]
            row[f"{family}_implied_mttff_s"] = fam["mttff"]["mttff"]
            row[f"{family}_implied_mttff_ci_lo"] = fam_ci.get("mttff_ci_lo")
            row[f"{family}_implied_mttff_ci_hi"] = fam_ci.get("mttff_ci_hi")
        summary[machine_key] = row

        median_life = float(np.median([e["simTime"] for e in gt["episodes"]]))
        reference_ages = [0.2 * median_life, 0.5 * median_life, 0.8 * median_life]
        risk = conditional_failure_risk(
            comparison_result, reference_ages_s=reference_ages,
            lookahead_s=0.1 * median_life, wear_threshold_um=wear_threshold_um,
        )
        hazard[machine_key] = risk[machine_key]

    return {
        "ground_truth": ground_truth, "process_implied": implied,
        "summary": summary, "conditional_failure_risk": hazard,
    }


# ── Parametric vs. data-driven (conformal RF) RUL comparison ──────────────────

# standard_shape_families() minus truncated Wiener specifically: the virtual-
# age inversion below assumes E[wear | shape_s] = shape_s/beta (solving that
# relationship for t is the whole trick), which is exactly right for every
# standard_shape_fn family EXCEPT truncated Wiener, whose registered
# mean_increment_fn is deliberately overridden (see
# families.standard_mean_increment's docstring) because left-truncation
# shifts its true conditional mean systematically ABOVE shape_s/beta -- reusing
# the naive relationship for it would reintroduce, in a new diagnostic, the
# same bias already identified and fixed for the trajectory-RMSE check
# (~17um naive vs ~31um true mean for one real fitted component). Properly
# supporting it here would need numerically inverting its true mean_increment_fn
# per data point rather than the closed-form power-law inversion below; not
# done yet -- a known, narrower limitation than wiener_multistage's complete
# absence from every reliability-translation table.
_RUL_INVERSION_EXCLUDED = {"wiener_truncated"}


def rul_leave_one_tool_out(
    index: pd.DataFrame,
    family: str,
    window_size: int = 15,
    wear_threshold_um: float = EOL_THRESHOLD_UM,
    *,
    use_per_tool_actual_threshold: bool = False,
) -> dict:
    """Point RUL prediction from ``family``'s virtual-age inversion,
    leave-one-tool-out, evaluated against each tool's actual observed
    remaining life -- same protocol as the existing conformal RF's
    ``tool_group_holdout_cv()``, so the two RMSE/MAE numbers are directly
    comparable. Generalizes what was originally an IG-only comparison (see
    ``ig_rul_leave_one_tool_out``, now a thin wrapper around this) to every
    ``families.standard_shape_families`` member except truncated Wiener
    (see ``_RUL_INVERSION_EXCLUDED``) -- the paper's headline "IG vs.
    richer-feature RF" comparison was never meant to imply IG is the only
    stochastic model worth checking this way, just the one this comparison
    originally shipped with.

    This is expected to lose to the ML model on raw accuracy (it uses only
    the wear level, not the auxiliary signal features the RF has access to);
    the point of the comparison is to honestly quantify that gap, not close it.

    With ``use_per_tool_actual_threshold=False`` (default), every tool's RUL
    target is "time to reach the fixed ``wear_threshold_um``" -- but the
    dataset's actual observed terminal wear ranges 145-160um across the 9
    tools (not exactly the fixed value), so part of any RMSE gap against the
    conformal RF (whose target is each tool's own actual last-recorded time,
    with no fixed-threshold assumption) is a threshold-mismatch artifact, not
    a genuine feature-richness effect. With ``use_per_tool_actual_threshold=
    True``, each held-out tool's own actual final recorded wear reading is
    used as its threshold instead, removing that confound so the remaining
    gap isolates the feature-richness effect alone.
    """
    if family in _RUL_INVERSION_EXCLUDED:
        raise ValueError(
            f"{family!r} is excluded from rul_leave_one_tool_out -- see "
            "_RUL_INVERSION_EXCLUDED's comment for why."
        )
    windows = build_wear_windows(index, window_size=window_size)
    tool_to_machine = index.groupby("tool")["machine"].first().to_dict()
    fit_fn = _FIT_FNS[family]
    results = {}
    for machine_key, data in sorted(windows.items()):
        for held_tool in np.unique(data["tool"]):
            train_mask = data["tool"] != held_tool
            fitted = fit_fn(
                data["dv"][train_mask], data["t_start"][train_mask],
                data["t_end"][train_mask], data["tool"][train_mask],
            )
            eta_med = float(np.median([c["eta"] for c in fitted["components"]]))
            power_med = float(np.median([c["power"] for c in fitted["components"]]))
            beta = float(fitted["beta"])
            t_ref = float(fitted["time_reference_s"])

            traj = index[index["tool"] == held_tool].sort_values("run")
            times = traj["cumulated_tool_contact_time"].to_numpy(float)
            wear = traj["wear"].to_numpy(float)
            rul_actual = np.maximum(0.0, float(times[-1]) - times)

            tool_threshold = float(wear[-1]) if use_per_tool_actual_threshold else wear_threshold_um

            wear_clipped = np.clip(wear, 1e-6, None)
            t_virtual = t_ref * (wear_clipped * beta / eta_med) ** (1.0 / power_med)
            t_fail_pred = t_ref * (tool_threshold * beta / eta_med) ** (1.0 / power_med)
            rul_pred = np.maximum(0.0, t_fail_pred - t_virtual)

            resid = rul_pred - rul_actual
            results[int(held_tool)] = {
                "machine": int(tool_to_machine[held_tool]),
                "rmse_s": float(np.sqrt(np.mean(resid ** 2))),
                "mae_s": float(np.mean(np.abs(resid))),
                "n": int(len(times)),
                "threshold_used_um": tool_threshold,
            }
    return results


def ig_rul_leave_one_tool_out(
    index: pd.DataFrame,
    window_size: int = 15,
    wear_threshold_um: float = EOL_THRESHOLD_UM,
    *,
    use_per_tool_actual_threshold: bool = False,
) -> dict:
    """Inverse-Gaussian-specific case of ``rul_leave_one_tool_out`` --
    preserved under its original name/signature (this study's original
    headline RUL-vs-RF comparison) for backward compatibility; new code
    comparing other families should call ``rul_leave_one_tool_out`` directly.
    """
    return rul_leave_one_tool_out(
        index, "inverse_gaussian", window_size=window_size,
        wear_threshold_um=wear_threshold_um,
        use_per_tool_actual_threshold=use_per_tool_actual_threshold,
    )


def compare_ig_vs_conformal_rul(
    ig_loto_result: dict,
    conformal_cv_csv_path: str | Path,
) -> dict:
    """Merge the IG leave-one-tool-out RMSE/MAE with the already-trained
    production conformal RF's tool_group_holdout_cv() results (read from its
    saved CSV) into one side-by-side comparison table.
    """
    conformal_df = pd.read_csv(conformal_cv_csv_path)
    comparison = {}
    for _, row in conformal_df.iterrows():
        heldout = row["heldout_tools"]
        tool_id = int(ast.literal_eval(heldout)[0]) if isinstance(heldout, str) else int(heldout)
        ig_row = ig_loto_result.get(tool_id, {})
        comparison[tool_id] = {
            "machine": int(row["heldout_machine"]),
            "conformal_rmse_s": float(row["rmse_s"]),
            "conformal_mae_s": float(row["mae_s"]),
            "ig_rmse_s": ig_row.get("rmse_s"),
            "ig_mae_s": ig_row.get("mae_s"),
            "ig_rmse_ratio_vs_conformal": (
                ig_row["rmse_s"] / float(row["rmse_s"]) if ig_row.get("rmse_s") else None
            ),
        }
    return comparison


def decompose_rul_threshold_confound(
    index: pd.DataFrame,
    conformal_cv_csv_path: str | Path,
    window_size: int = 15,
) -> dict:
    """Decompose the IG-vs-conformal RUL RMSE gap (``compare_ig_vs_conformal_rul``)
    into a threshold-mismatch component and a feature-richness component.

    The original comparison scores every tool's IG-implied RUL against a
    fixed EOL threshold (150um) even though actual observed terminal wear
    ranges 145-160um across the 9 tools -- part of the reported 1.45-4.84x
    RMSE gap could be this mismatch rather than the conformal RF's richer
    feature set. Runs ``ig_rul_leave_one_tool_out`` twice (fixed threshold vs.
    each tool's own actual observed terminal wear) and reports both ratios
    side by side so the mismatch contribution is visible, not asserted.
    """
    fixed = ig_rul_leave_one_tool_out(index, window_size=window_size, use_per_tool_actual_threshold=False)
    matched = ig_rul_leave_one_tool_out(index, window_size=window_size, use_per_tool_actual_threshold=True)

    conformal_df = pd.read_csv(conformal_cv_csv_path)

    result = {}
    for _, row in conformal_df.iterrows():
        heldout = row["heldout_tools"]
        tool_id = int(ast.literal_eval(heldout)[0]) if isinstance(heldout, str) else int(heldout)
        conformal_rmse = float(row["rmse_s"])
        f = fixed.get(tool_id, {})
        m = matched.get(tool_id, {})
        result[tool_id] = {
            "machine": int(row["heldout_machine"]),
            "conformal_rmse_s": conformal_rmse,
            "ig_fixed_threshold_rmse_s": f.get("rmse_s"),
            "ig_matched_threshold_rmse_s": m.get("rmse_s"),
            "matched_threshold_um": m.get("threshold_used_um"),
            "ig_fixed_threshold_ratio_vs_conformal": (
                f["rmse_s"] / conformal_rmse if f.get("rmse_s") else None
            ),
            "ig_matched_threshold_ratio_vs_conformal": (
                m["rmse_s"] / conformal_rmse if m.get("rmse_s") else None
            ),
        }
    return result


# ── Maintenance-cost / decision translation (age-replacement policy) ──────────

def age_replacement_cost_rate(
    params_for_machine: dict,
    family: str,
    T_grid: np.ndarray,
    c_pm: float,
    c_cm: float,
    wear_threshold_um: float = EOL_THRESHOLD_UM,
) -> np.ndarray:
    """Classic Barlow-Proschan age-replacement long-run cost rate:

        C(T) = [c_pm * R(T) + c_cm * (1 - R(T))] / M(T)

    where R(T) is the process-implied survival function and M(T) is the mean
    replacement-cycle length under policy T (integral of R up to T). c_pm/c_cm
    are illustrative (not derived from the dataset, which has no cost data) --
    report cost RATIOS and relative regret, not absolute currency figures.
    """
    t_fine = np.linspace(0.0, float(T_grid.max()), 2000)
    S_fine = _implied_survival(params_for_machine, family, t_fine, wear_threshold_um)
    S_at_T = _implied_survival(params_for_machine, family, T_grid, wear_threshold_um)
    costs = np.zeros(len(T_grid))
    for i, T in enumerate(T_grid):
        mask = t_fine <= T
        mean_cycle_length = np.trapezoid(S_fine[mask], t_fine[mask])
        survival_at_T = S_at_T[i]
        costs[i] = (
            c_pm * survival_at_T + c_cm * (1.0 - survival_at_T)
        ) / max(mean_cycle_length, 1e-9)
    return costs


def maintenance_cost_comparison(
    comparison_result: dict,
    c_pm: float = 1.0,
    c_cm: float = 3.0,
    t_grid_max_s: float = 6000.0,
    n_grid: int = 300,
) -> dict:
    """Optimal preventive-replacement age T* and cost rate under each family's
    implied hazard, plus the "regret" (% extra cost) of using Gamma's or
    Wiener's optimal policy if Inverse Gaussian (the best-supported family) is
    actually true. Ground truth for "which policy is really best" is IG since
    it wins both in-sample AIC and out-of-sample CV likelihood everywhere.

    IG's cost at another family's T* is read directly off IG's own
    already-computed ``costs`` array (indexed at that family's own
    ``best_idx`` -- valid since every family's cost curve is evaluated over
    the SAME shared ``T_grid``) rather than re-evaluating
    ``age_replacement_cost_rate`` at a single-point array. The two are not
    numerically identical: that function's internal integration grid
    (``t_fine = linspace(0, T_grid.max(), 2000)``) depends on the ``T_grid``
    it's called with, so a single-point call integrates over a different
    (finer, differently-bounded) grid than the original per-machine sweep
    used to obtain ``cost_star`` -- a real, if tiny (parts-per-million-scale
    trapezoidal quadrature discrepancy at 2000 points), inconsistency
    between the two costs being differenced for "regret". Reusing the
    already-computed array eliminates it exactly, at no extra cost (avoids
    a redundant survival-function evaluation too).

    Covers every family in ``families.standard_shape_families`` (every
    registered family with a closed-form survival function -- see that
    function's docstring for exactly which, and why 3 of the 10 registered
    families are not eligible), not just the original Gamma/IG/Wiener trio
    -- reuses each family's ALREADY-fitted params from ``comparison_result``
    (no new fitting), so this stays cheap regardless of family count.
    Regret is still reported relative to IG specifically (this study's
    best-supported family under the original 3-way comparison) for EVERY
    other eligible family, not just Gamma/Wiener, generalizing the original
    two-column regret report to however many families are eligible.
    """
    T_grid = np.linspace(200.0, t_grid_max_s, n_grid)
    result = {}
    for machine_key in sorted(comparison_result["gamma_params"]):
        eligible_families = [
            f for f in families.standard_shape_families(comparison_result)
            if machine_key in comparison_result[families.param_key(f)]
        ]
        row = {}
        costs_by_family = {}
        best_idx_by_family = {}
        for family in eligible_families:
            costs = age_replacement_cost_rate(
                comparison_result[families.param_key(family)][machine_key], family, T_grid, c_pm, c_cm,
            )
            costs_by_family[family] = costs
            best_idx = int(np.argmin(costs))
            best_idx_by_family[family] = best_idx
            row[family] = {"T_star": float(T_grid[best_idx]), "cost_star": float(costs[best_idx])}

        ig_star_cost = row["inverse_gaussian"]["cost_star"]
        ig_costs = costs_by_family["inverse_gaussian"]
        for other_family in eligible_families:
            if other_family == "inverse_gaussian":
                continue
            cost_ig_at_other_T = float(ig_costs[best_idx_by_family[other_family]])
            row[f"regret_using_{other_family}_policy_pct"] = float(
                (cost_ig_at_other_T - ig_star_cost) / ig_star_cost * 100.0
            )
        result[machine_key] = row
        t_star_str = "  ".join(
            f"{family}={row[family]['T_star']:.0f}s" for family in eligible_families
        )
        regret_str = "  ".join(
            f"regret({family})={row[f'regret_using_{family}_policy_pct']:+.1f}%"
            for family in eligible_families if family != "inverse_gaussian"
        )
        print(f"  {machine_key}: T* {t_star_str}  |  {regret_str}")
    return result


def cost_ratio_sensitivity(
    comparison_result: dict,
    cost_ratios: tuple[float, ...] = (2.0, 3.0, 5.0, 7.0, 10.0),
    t_grid_max_s: float = 6000.0,
    n_grid: int = 300,
) -> dict:
    """Sweep c_cm/c_pm across a realistic range (c_pm fixed at 1.0) and check
    whether the "Wiener regret > Gamma/IG regret" ordering from
    ``maintenance_cost_comparison`` (reported at a single illustrative ratio
    of 3) is a robust conclusion or an artifact of that one arbitrary choice.

    Reports mean regret at each ratio for every family
    ``maintenance_cost_comparison`` covers (see that function's docstring --
    every ``families.standard_shape_families`` member, not just the
    original Gamma/Wiener pair), plus the original, narrower "does Wiener's
    regret stay >= Gamma's on every machine" check specifically (both
    families present in every registered variant of this study to date, so
    this specific ordering check remains meaningful unchanged).
    """
    result = {}
    for ratio in cost_ratios:
        cost_result = maintenance_cost_comparison(
            comparison_result, c_pm=1.0, c_cm=float(ratio),
            t_grid_max_s=t_grid_max_s, n_grid=n_grid,
        )
        result[str(ratio)] = cost_result
        eligible_families = [
            f for f in families.standard_shape_families(comparison_result) if f != "inverse_gaussian"
        ]
        mean_regrets = {
            family: float(np.mean([cost_result[m][f"regret_using_{family}_policy_pct"] for m in cost_result]))
            for family in eligible_families
        }
        ordering_holds = all(
            cost_result[m]["regret_using_wiener_policy_pct"]
            >= cost_result[m]["regret_using_gamma_policy_pct"]
            for m in cost_result
        )
        regret_str = "  ".join(f"mean regret({f})={v:+.2f}%" for f, v in mean_regrets.items())
        print(f"  ratio={ratio:.0f}: {regret_str}  Wiener>=Gamma on all machines: {ordering_holds}")
    return result


def bootstrap_maintenance_cost_regret_by_tool(
    index: pd.DataFrame,
    window_size: int = 15,
    c_pm: float = 1.0,
    c_cm: float = 3.0,
    t_grid_max_s: float = 6000.0,
    n_grid: int = 300,
) -> dict:
    """Block bootstrap (whole tools, exact enumeration where feasible --
    mirrors ``degradation_process_models.block_bootstrap_family_aic_gap_by_tool``)
    propagating parameter ESTIMATION uncertainty into
    ``maintenance_cost_comparison``'s T*/regret numbers, which otherwise
    plug in a single fixed point-estimate fit per family with no attached
    uncertainty -- directly addressing the "reported maintenance-cost
    increases... depend on... fixed fitted parameters" critique.
    ``cost_ratio_sensitivity`` already answers the separate "illustrative
    cost ratios" half of that critique (sweeping c_cm/c_pm); this answers
    the other half (sweeping the fitted parameters themselves, via
    resampling).

    For each of the 3**3=27 exact tool-resamples on a machine, refits
    Gamma/Inverse Gaussian/Wiener and recomputes each family's regret
    exactly as ``maintenance_cost_comparison`` does (same c_pm/c_cm/T-grid),
    then reports the resulting distribution's mean and 95% CI. Same scope
    and honesty caveat as the AIC-gap block bootstrap it mirrors: this is
    the between-tool-generalization uncertainty specifically, genuinely
    informative with only 3 tools but honestly wide -- not a claim of
    higher precision than the data actually supports.

    ``_implied_survival`` (called inside ``age_replacement_cost_rate``)
    averages EQUALLY across whatever is in a fit's ``components`` list --
    correct for the true 3-distinct-tool population, but WRONG for a
    resampled draw where a tool repeats (e.g. draw=(1,1,2)): the fit
    itself correctly double-weights tool 1's likelihood contribution
    during MLE, but its ``components`` list still has only 2 entries (one
    per unique tool_id seen), so an unweighted average over that list
    would give tool 1 and tool 2 equal 50/50 ensemble weight instead of
    the draw's actual 2/3-1/3 split. Fixed by rebuilding the components
    list handed to ``age_replacement_cost_rate`` to literally repeat each
    tool's component once per its multiplicity IN THIS DRAW, so the
    existing (unmodified) equal-average-over-the-list logic naturally
    reproduces the correct resample weighting -- no change needed to
    ``_implied_survival`` itself.

    Covers every ``families.standard_shape_families``-eligible family
    EXCEPT the two random-effects variants (deliberately, not an
    oversight): unlike ``maintenance_cost_comparison`` (which reuses
    already-fitted params, so is cheap regardless of family count), THIS
    function refits from scratch on every one of the (up to 27) resampled
    draws -- the random-effects families' Gauss-Hermite-quadrature fit
    already costs ~250-335s for a single real 3-tool fit (see
    ``model_recovery_study``'s docstring), so refitting either 27 times per
    machine x 3 machines would add on the order of another day of runtime
    for a between-tool-uncertainty check whose main scientific point
    (propagating parameter uncertainty into the cost-regret ordering) is
    already served by the 5 cheaper eligible families. Excluded here,
    still fully covered by the cheaper, non-bootstrapped
    ``maintenance_cost_comparison``.
    """
    windows = build_wear_windows(index, window_size=window_size)
    T_grid = np.linspace(200.0, t_grid_max_s, n_grid)
    bootstrap_families = [
        k for k in families.FAMILY_REGISTRY
        if families.FAMILY_REGISTRY[k]["shape_fn"] is families.standard_shape_fn
        and k not in ("gamma_random_effects", "inverse_gaussian_random_effects")
    ]
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
            rng = np.random.default_rng(0)
            draws = [rng.choice(unique_tools, size=n_tools, replace=True) for _ in range(2000)]

        other_families = [f for f in bootstrap_families if f != "inverse_gaussian"]
        regrets = {family: [] for family in other_families}
        for draw in draws:
            idx = np.concatenate([tool_index_map[int(t)] for t in draw])
            fits = {}
            for family in bootstrap_families:
                fitted = _FIT_FNS[family](
                    data["dv"][idx], data["t_start"][idx], data["t_end"][idx], data["tool"][idx],
                )
                # Re-weight the ensemble to match this draw's actual tool
                # multiplicities (see docstring) -- component_by_tool has
                # one entry per UNIQUE tool_id fitted; repeating each per
                # its count in `draw` makes _implied_survival's plain
                # per-entry average correct for a resample with repeats.
                component_by_tool = {int(c["tool_id"]): c for c in fitted["components"]}
                fits[family] = {**fitted, "components": [component_by_tool[int(t)] for t in draw]}
            # Cost at each family's own T* is read off that family's own
            # already-computed costs array (all evaluated over the SAME
            # shared T_grid), never re-evaluated at a single-point array --
            # age_replacement_cost_rate's internal integration grid depends
            # on T_grid.max(), so a single-point re-evaluation would
            # integrate over a different grid than the sweep that found T*
            # in the first place (see maintenance_cost_comparison, which
            # has the identical fix for the same reason).
            costs_by_family = {}
            best_idx_by_family = {}
            for family, fitted in fits.items():
                costs = age_replacement_cost_rate(fitted, family, T_grid, c_pm, c_cm)
                costs_by_family[family] = costs
                best_idx_by_family[family] = int(np.argmin(costs))
            ig_costs = costs_by_family["inverse_gaussian"]
            ig_costs_at_star = float(ig_costs[best_idx_by_family["inverse_gaussian"]])
            for other in other_families:
                ig_cost_at_other_T = float(ig_costs[best_idx_by_family[other]])
                regrets[other].append(
                    float((ig_cost_at_other_T - ig_costs_at_star) / ig_costs_at_star * 100.0)
                )

        result[machine_key] = {"n_draws": len(draws), "exact_enumeration": exact}
        for other in other_families:
            r = np.asarray(regrets[other], float)
            result[machine_key][other] = {
                "mean_regret_pct": float(np.mean(r)),
                "std_regret_pct": float(np.std(r)),
                "ci_lo": float(np.percentile(r, 2.5)),
                "ci_hi": float(np.percentile(r, 97.5)),
            }
        regret_str = "  ".join(
            f"regret({family})={result[machine_key][family]['mean_regret_pct']:+.2f}% "
            f"[{result[machine_key][family]['ci_lo']:.2f}, {result[machine_key][family]['ci_hi']:.2f}]"
            for family in other_families
        )
        print(f"  {machine_key} ({'exact enum.' if exact else 'MC'}, {len(draws)} draws): {regret_str}")
    return result
