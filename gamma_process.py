"""Gamma-process degradation model for milling tool wear.

The production model is a rounded-observation, non-homogeneous Gamma process:

    VB(t) - VB(s) ~ Gamma(η_j[(t/t_ref)^p_j - (s/t_ref)^p_j], β)

where ``j`` identifies a physical tool trajectory.  A common rate ``β`` is
estimated per machine, while ``η_j`` and ``p_j`` capture between-tool wear-rate
and ageing-pattern heterogeneity.

The dataset records integer-micrometre wear labels.  The likelihood therefore
integrates the continuous Gamma density over each observed rounding interval
instead of treating integer labels as exact continuous measurements.  Zero
increments are retained; dropping them would condition the sample on positive
wear and severely bias the inferred degradation rate.

This is the production Gamma fit reused (read-only) by the active degradation
process-family comparison study (``degradation_process_models.py``, which
fits Inverse Gaussian and Wiener alternatives against the same wear windows
via ``build_wear_windows()``).

Output
------
A dict of {group_name: GammaParams} saved as JSON under artifacts/gamma/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import gamma as gamma_dist, kstest

from milling_dataset import MACHINE_GROUPS


WEAR_SCALE_UM = 160.0   # EOL scale used throughout the project
TIME_REFERENCE_S = 1000.0
WEAR_RESOLUTION_UM = 1.0
MODEL_SCHEMA_VERSION = 3

# ── Aggregated-rounding-error kernel (peer-review Major Comment 1) ────────────
#
# A window's observed wear increment ``dv`` is built by build_wear_windows()
# as the sum of the 15 per-run rounded first-differences inside it, which
# telescopes exactly to (rounded reading at the window's last run) minus
# (rounded reading at its first run) -- a two-endpoint difference of two
# INDEPENDENTLY rounded integer-micrometre readings, regardless of the
# window size chosen. Writing the true (unrounded) window increment as X and
# the two endpoints' rounding errors as e_end, e_start ~ iid Uniform(-0.5,
# +0.5) um, the observed integer dv equals round(X + u) where u = e_end -
# e_start's role is played by a single Uniform(-0.5, 0.5) offset (the
# window-start reading's own fractional rounding offset; see derivation in
# the module docstring reference below) -- NOT round(X) alone. Treating the
# rounding correction as a flat +/-0.5um bin around the aggregated dv (the
# original Eq. 3 as submitted) is therefore only correct for a *single*
# rounded reading; for a two-endpoint difference it understates the rounding
# uncertainty, which is instead recovered by marginalising the flat +/-0.5um
# bin over this additional Uniform(-0.5, 0.5) offset, i.e.
#
#   Pr(observed = d) = integral_{u=-0.5}^{+0.5} [F(d+0.5-u) - F(d-0.5-u)] du
#
# evaluated by fixed 11-point Gauss-Legendre quadrature (exact for smooth
# CDFs to well beyond the precision this likelihood needs; verified against
# a 41-point rule to agree to >10 significant figures on this dataset's
# fitted parameter ranges). This is used for every family (Gamma, Inverse
# Gaussian, Wiener) so the correction is applied identically across the
# three-family comparison; see also the matching fix in
# ``degradation_process_models._simulate_synthetic_dv``, which must draw the
# same extra offset when generating synthetic "observed" windows for the
# model-recovery positive control, or that check would validate the
# procedure against a rounding model the real-data fits no longer use.
_ROUNDING_QUAD_NODES, _ROUNDING_QUAD_WEIGHTS = np.polynomial.legendre.leggauss(11)
ROUNDING_OFFSETS_UM = _ROUNDING_QUAD_NODES * 0.5   # nodes rescaled from [-1,1] to [-0.5,0.5]
ROUNDING_WEIGHTS = _ROUNDING_QUAD_WEIGHTS * 0.5    # weights rescaled to sum to 1.0


def _aggregated_bin_probability(observed: np.ndarray, cdf_fn, resolution_um: float) -> np.ndarray:
    """Two-endpoint-difference rounding-error correction, shared by all three
    degradation-process families.

    ``cdf_fn(x)`` must return the family's CDF evaluated elementwise at ``x``
    (any per-window shape/rate parameters already baked in via closure);
    clamping to a family's support (e.g. x >= 0 for Gamma/IG) is the caller's
    responsibility inside ``cdf_fn``, exactly as it was for the un-aggregated
    bin before this fix.
    """
    half = 0.5 * float(resolution_um)
    observed = np.asarray(observed, float)
    total = np.zeros_like(observed)
    for u, w in zip(ROUNDING_OFFSETS_UM, ROUNDING_WEIGHTS):
        shift = u * resolution_um
        total += w * (cdf_fn(observed + half - shift) - cdf_fn(observed - half - shift))
    return total


def build_wear_windows(
    index: pd.DataFrame,
    *,
    window_size: int = 15,
) -> dict[str, dict[str, np.ndarray]]:
    """Build non-overlapping rounded wear windows for each physical machine.

    Zero wear increments are retained because they carry information about the
    lower tail of a quantized degradation process.
    """
    if window_size < 2:
        raise ValueError("window_size must be at least 2.")

    out: dict[str, dict[str, list]] = {
        f"M{machine_id}": {
            "dv": [], "t_start": [], "t_end": [], "tool": [], "v_start": [],
        }
        for machine_id in sorted(MACHINE_GROUPS)
    }
    for (machine, tool), traj in index.groupby(["machine", "tool"]):
        machine_id = int(machine)
        key = f"M{machine_id}"
        if key not in out:
            continue
        traj = traj.sort_values("run")
        times = traj["cumulated_tool_contact_time"].to_numpy(float)
        wear = traj["wear"].to_numpy(float)
        dv_step = np.diff(wear)
        dt_step = np.diff(times)
        for start in range(0, len(dv_step) - window_size + 1, window_size):
            end = start + window_size
            dv_w = float(np.sum(dv_step[start:end]))
            dt_w = float(np.sum(dt_step[start:end]))
            if dv_w < 0 or dt_w <= 0:
                continue
            out[key]["dv"].append(dv_w)
            out[key]["t_start"].append(float(times[start]))
            out[key]["t_end"].append(float(times[end]))
            out[key]["tool"].append(int(tool))
            # Wear level AT the window's start reading -- not part of the
            # original 3-family likelihood (which only ever used age, s and
            # t), added for the state-dependent Gamma/IG variants
            # (degradation_process_variants.py), which need "how worn is
            # this tool already" as a second predictor alongside age.
            # Every existing consumer of build_wear_windows() ignores the
            # extra dict key, so this is additive and backward-compatible.
            out[key]["v_start"].append(float(wear[start]))

    return {
        key: {name: np.asarray(values) for name, values in data.items()}
        for key, data in out.items()
    }


def _rounded_gamma_bin_probability(
    observed: np.ndarray,
    shape: np.ndarray,
    beta: float,
    resolution_um: float,
) -> np.ndarray:
    """Probability of each aggregated, two-endpoint-rounded observation under
    a Gamma law (see ``_aggregated_bin_probability`` for the rounding-kernel
    derivation)."""
    def _cdf(x):
        return gamma_dist.cdf(np.maximum(0.0, x), a=shape, scale=1.0 / beta)
    prob = _aggregated_bin_probability(observed, _cdf, resolution_um)
    return np.maximum(prob, np.finfo(float).tiny)


def _fit_rounded_nh_gamma(
    dv: np.ndarray,
    t_start: np.ndarray,
    t_end: np.ndarray,
    tool_ids: np.ndarray,
    *,
    time_reference_s: float = TIME_REFERENCE_S,
    resolution_um: float = WEAR_RESOLUTION_UM,
) -> dict:
    """Fit a tool-heterogeneous non-homogeneous Gamma process.

    Each tool receives its own ``eta`` and ageing exponent ``power`` while the
    Gamma rate ``beta`` is shared across tools from the same machine.
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

    # Initialise each component from its total observed wear and terminal age.
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
        if (
            np.any(power < 0.05) or np.any(power > 3.0)
            or beta < 1e-4 or beta > 100.0
        ):
            return 1e100
        shape = eta[component_idx] * (
            np.power(t_end / t_ref, power[component_idx])
            - np.power(t_start / t_ref, power[component_idx])
        )
        if np.any(shape <= 0) or not np.all(np.isfinite(shape)):
            return 1e100
        prob = _rounded_gamma_bin_probability(dv, shape, beta, resolution_um)
        return -float(np.sum(np.log(prob)))

    starts = [
        np.log(np.r_[eta0, power0, beta0]),
        np.log(np.r_[eta0 * 0.7, np.full(n_tools, 0.6), 0.4]),
        np.log(np.r_[eta0 * 1.3, np.full(n_tools, 1.0), 0.7]),
    ]
    best = None
    for x0 in starts:
        res = minimize(
            nll,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 100000, "xatol": 1e-9, "fatol": 1e-9},
        )
        if best is None or res.fun < best.fun:
            best = res
    if best is None or not np.isfinite(best.fun):
        raise RuntimeError("Non-homogeneous rounded Gamma MLE failed.")

    eta = np.exp(best.x[:n_tools])
    power = np.exp(best.x[n_tools:2 * n_tools])
    beta = float(np.exp(best.x[-1]))

    # Stationary rounded model with the same per-tool heterogeneity is the
    # nested benchmark used to verify that non-stationarity is warranted.
    stationary_start = np.log(np.r_[eta, beta])
    stationary = minimize(
        lambda z: nll(z, fixed_power=True),
        stationary_start,
        method="Nelder-Mead",
        options={"maxiter": 50000, "xatol": 1e-9, "fatol": 1e-9},
    )

    n = len(dv)
    k_nh = 2 * n_tools + 1
    k_stationary = n_tools + 1
    log_lik = -float(best.fun)
    stationary_log_lik = -float(stationary.fun)
    nh_components = [
        {
            "tool_id": int(tool),
            "eta": float(eta[i]),
            "power": float(power[i]),
            "beta": beta,
            "time_reference_s": t_ref,
        }
        for i, tool in enumerate(tools)
    ]
    stationary_eta = np.exp(stationary.x[:n_tools])
    stationary_beta = float(np.exp(stationary.x[n_tools]))
    stationary_components = [
        {
            "tool_id": int(tool),
            "eta": float(stationary_eta[i]),
            "power": 1.0,
            "beta": stationary_beta,
            "time_reference_s": t_ref,
        }
        for i, tool in enumerate(tools)
    ]
    nh_aic = float(2 * k_nh - 2 * log_lik)
    stationary_aic = float(2 * k_stationary - 2 * stationary_log_lik)
    use_nonhomogeneous = bool(stationary_aic - nh_aic > 2.0)
    selected_components = (
        nh_components if use_nonhomogeneous else stationary_components
    )
    selected_beta = beta if use_nonhomogeneous else stationary_beta
    selected_log_lik = log_lik if use_nonhomogeneous else stationary_log_lik
    selected_k = k_nh if use_nonhomogeneous else k_stationary
    return {
        "model": "nonhomogeneous_gamma_rounded",
        "schema_version": MODEL_SCHEMA_VERSION,
        "selected_age_model": (
            "power_law" if use_nonhomogeneous else "linear"
        ),
        "beta": selected_beta,
        "eta": float(np.median([
            component["eta"] for component in selected_components
        ])),
        "power": float(np.median([
            component["power"] for component in selected_components
        ])),
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
        "stationary_bic": float(
            k_stationary * np.log(n) - 2 * stationary_log_lik
        ),
    }


def fit_all_groups(index: pd.DataFrame, window_size: int = 15) -> dict[str, dict]:
    """Fit validated non-homogeneous Gamma models for every machine.

    Parameters
    ----------
    index : DataFrame from load_file_index()
    window_size : aggregation window for rounded wear observations

    Returns
    -------
    dict keyed by group name → GammaParams dict
    """
    print("Fitting rounded non-homogeneous Gamma process per machine ...")
    windows = build_wear_windows(index, window_size=window_size)
    results = {}
    for machine_key, data in sorted(windows.items()):
        n_raw = len(data["dv"])
        print(
            f"  {machine_key}: {n_raw} windows "
            f"({int(np.sum(data['dv'] == 0))} rounded-zero; "
            f"window_size={window_size})"
        )
        if n_raw < 30:
            raise ValueError(
                f"Insufficient wear windows for {machine_key} "
                f"({n_raw} < 30). Reduce window_size."
            )
        params = _fit_rounded_nh_gamma(
            data["dv"],
            data["t_start"],
            data["t_end"],
            data["tool"],
        )
        params["machine_id"] = int(machine_key[1:])
        params["window_size"] = int(window_size)
        params["wear_scale_um"] = WEAR_SCALE_UM
        results[machine_key] = params
        print(
            f"    beta={params['beta']:.6g}  "
            f"age model={params['selected_age_model']}  "
            f"median power={params['power']:.3f}  "
            f"delta_AIC(stationary-NH)="
            f"{params['stationary_aic'] - params['nonhomogeneous_aic']:.2f}"
        )
    return results


def save_gamma_params(params: dict[str, dict], output_dir: str | Path) -> Path:
    """Save fitted Gamma parameters to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "gamma_process_params.json"
    serialisable = {
        g: {k: float(v) if isinstance(v, (float, int, np.floating, np.integer)) else v
            for k, v in p.items()}
        for g, p in params.items()
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2)
    print(f"Saved Gamma process parameters -> {path}")
    return path


def validate_gamma_fit(
    index: pd.DataFrame,
    params: dict[str, dict],
) -> dict[str, dict]:
    """Validate rounded non-homogeneous fits against the observed trajectories.

    Validation combines randomized quantile residuals (appropriate for rounded
    observations), trajectory RMSE, terminal bias, and AIC improvement relative
    to a nested stationary Gamma model.

    The rounded-observation likelihood is only identified up to a randomized
    de-quantization draw within each rounding bin, so the PIT-KS statistic
    from any single draw is noisy at this sample size: a single unlucky/lucky
    seed can swing a borderline case across the p=0.05 boundary. Averaging the
    KS statistic and p-value over 200 independent draws (mirroring
    ``degradation_process_models._pit_ks_generic``, used for the Inverse
    Gaussian and Wiener alternatives fit against the same wear windows) gives
    a stable, defensible calibration verdict instead of relying on one
    arbitrary realization -- keeping all three compared families on the same
    validation protocol.
    """
    validation = {}
    windows = build_wear_windows(
        index,
        window_size=next(iter(params.values())).get("window_size", 15),
    )
    for machine_key, gp in params.items():
        if gp.get("model") != "nonhomogeneous_gamma_rounded":
            raise ValueError(
                "Legacy stationary Gamma artifact detected; refit Phase 1."
            )
        data = windows[machine_key]
        component_by_tool = {
            int(component["tool_id"]): component
            for component in gp["components"]
        }
        shapes = np.empty(len(data["dv"]), float)
        for i, tool in enumerate(data["tool"]):
            component = component_by_tool[int(tool)]
            eta = float(component["eta"])
            power = float(component["power"])
            t_ref = float(component["time_reference_s"])
            shapes[i] = eta * (
                (float(data["t_end"][i]) / t_ref) ** power
                - (float(data["t_start"][i]) / t_ref) ** power
            )

        resolution = float(gp["wear_resolution_um"])
        half = 0.5 * resolution
        beta = float(gp["beta"])
        dv = data["dv"]
        # Two-endpoint aggregated rounding kernel (see
        # _aggregated_bin_probability's module-level derivation): de-quantise
        # against the same marginalised-offset bin edges the likelihood uses.
        n_nodes = len(ROUNDING_OFFSETS_UM)
        cdf_lo_nodes = np.empty((n_nodes, len(dv)), float)
        cdf_hi_nodes = np.empty((n_nodes, len(dv)), float)
        for j, u in enumerate(ROUNDING_OFFSETS_UM):
            shift = u * resolution
            lo_edge = np.maximum(0.0, dv - half - shift)
            hi_edge = np.maximum(0.0, dv + half - shift)
            cdf_lo_nodes[j] = gamma_dist.cdf(lo_edge, a=shapes, scale=1.0 / beta)
            cdf_hi_nodes[j] = gamma_dist.cdf(hi_edge, a=shapes, scale=1.0 / beta)
        weights = ROUNDING_WEIGHTS[:, None]
        cdf_lo = np.sum(weights * cdf_lo_nodes, axis=0)
        bin_prob = np.sum(weights * (cdf_hi_nodes - cdf_lo_nodes), axis=0)
        # Randomized PIT resolves the discreteness induced by integer labels;
        # average over 200 independent draws rather than one (see docstring).
        n_seeds = 200
        ks_stats = np.empty(n_seeds, float)
        ks_pvalues = np.empty(n_seeds, float)
        base_seed = 2026 + int(gp["machine_id"]) * 10_000
        for s in range(n_seeds):
            rng = np.random.default_rng(base_seed + s)
            pit = cdf_lo + rng.random(len(cdf_lo)) * bin_prob
            ks_stats[s], ks_pvalues[s] = kstest(pit, "uniform")
        ks_stat = float(np.mean(ks_stats))
        ks_pvalue = float(np.mean(ks_pvalues))
        frac_reject_05 = float(np.mean(ks_pvalues < 0.05))

        machine_id = int(gp["machine_id"])
        sub = index[index["machine"] == machine_id]
        squared_errors = []
        terminal_biases = []
        for tool, traj in sub.groupby("tool"):
            traj = traj.sort_values("run")
            component = component_by_tool[int(tool)]
            time = traj["cumulated_tool_contact_time"].to_numpy(float)
            wear = traj["wear"].to_numpy(float)
            eta = float(component["eta"])
            power = float(component["power"])
            beta = float(component["beta"])
            t_ref = float(component["time_reference_s"])
            mean_increment = eta / beta * (
                np.power(time / t_ref, power)
                - (time[0] / t_ref) ** power
            )
            predicted = wear[0] + mean_increment
            squared_errors.extend(np.square(wear - predicted))
            terminal_biases.append(float(predicted[-1] - wear[-1]))

        rmse = float(np.sqrt(np.mean(squared_errors)))
        terminal_bias = float(np.mean(terminal_biases))
        delta_aic = float(
            gp["stationary_aic"] - gp["nonhomogeneous_aic"]
        )
        age_model_supported = bool(
            gp.get("selected_age_model") == "linear"
            or delta_aic > 2.0
        )
        passed = bool(
            np.isfinite(rmse)
            and rmse <= 20.0
            and abs(terminal_bias) <= 15.0
            and ks_stat <= 0.15
            and frac_reject_05 < 0.5
            and age_model_supported
        )
        result = {
            "pit_ks_stat": ks_stat,
            "pit_ks_pvalue": ks_pvalue,
            "pit_ks_frac_reject_05": frac_reject_05,
            "pit_ks_n_seeds": int(n_seeds),
            "trajectory_rmse_um": rmse,
            "terminal_bias_um": terminal_bias,
            "delta_aic_vs_stationary": delta_aic,
            "n_windows": len(data["dv"]),
            "validation_passed": passed,
        }
        validation[machine_key] = result
        gp.update(result)
        print(
            f"  Validation {machine_key}: mean PIT-KS={ks_stat:.3f} "
            f"(mean p={ks_pvalue:.3f}, reject@.05={frac_reject_05:.0%} of {n_seeds} seeds), "
            f"trajectory RMSE={rmse:.2f} µm, "
            f"terminal bias={terminal_bias:+.2f} µm, "
            f"delta_AIC={delta_aic:.2f}, passed={passed}"
        )
    return validation
