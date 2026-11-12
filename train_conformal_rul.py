"""Train a Random Forest RUL model with tool-trajectory CV+ intervals.

The nine physical tool trajectories—not their 6,418 temporally correlated
rows—are the uncertainty-calibration units. Each leave-one-tool-out model
produces one maximum residual score per held-out trajectory and wear phase.
Runtime intervals combine these trajectory scores and model predictions using
finite-sample CV+ order statistics.

This is intentionally described as cluster-aware empirical coverage rather
than an exact 90% row-level guarantee: with only nine independent tools, strong
conditional or exact high-confidence claims would be statistically unjustified.

Output: artifacts/rul/conformal_rul_model.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from milling_dataset import (
    DATASET_DOI,
    SIGNAL_FEATURE_NAMES,
    dataset_summary,
    find_dataset_dir,
    load_file_index,
    load_or_build_feature_table,
    validate_hdf5_labels,
)


WEAR_SCALE_UM = 160.0
MODEL_SCHEMA_VERSION = 6         # v6 uses trajectory-level CV+ calibration

# Coverage level for conformal intervals (1 - ALPHA = 90% coverage)
CONFORMAL_ALPHA = 0.10


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _smooth(values: np.ndarray, width: int = 7) -> np.ndarray:
    width = max(1, int(width))
    if width == 1:
        return values.copy()
    left = width // 2
    right = width - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def _trajectory_map(age_fraction: np.ndarray, values: np.ndarray,
                     n_grid: int = 60, bandwidth: float = 0.03) -> tuple:
    age_grid = np.linspace(0.0, 1.0, int(n_grid))
    value_grid = np.full((len(age_grid), values.shape[1]), np.nan)
    for i, age in enumerate(age_grid):
        mask = np.abs(age_fraction - age) <= bandwidth
        if np.any(mask):
            value_grid[i] = np.median(values[mask], axis=0)
    for col in range(value_grid.shape[1]):
        valid = np.isfinite(value_grid[:, col])
        if not np.any(valid):
            raise ValueError("Cannot construct trajectory map: all values are NaN.")
        value_grid[:, col] = np.interp(age_grid, age_grid[valid], value_grid[valid, col])
        value_grid[:, col] = _smooth(value_grid[:, col])
    return age_grid, value_grid


def prepare_training_data(table: pd.DataFrame) -> dict:
    """Construct feature matrix, RUL targets, and runtime profiles from the
    full feature table (6418 rows).

    Feature vector per run (10 features):
        [VB_norm, causal_VB_slope, *5_raw_signal_features (SIGNAL_FEATURE_NAMES),
         3-column machine_one_hot]

    RUL target: remaining cumulative contact seconds to the recorded trajectory
    endpoint (same definition as the original MoE-GP).
    """
    required = ["machine", "tool", "run", "cumulated_tool_contact_time",
                 "wear", *SIGNAL_FEATURE_NAMES]
    missing = [c for c in required if c not in table.columns]
    if missing:
        raise ValueError(f"Training table missing columns: {missing}")

    table = table.sort_values(["machine", "tool", "run"]).reset_index(drop=True).copy()
    machine_ids = sorted(int(v) for v in table["machine"].unique())
    tool_ids    = table["tool"].to_numpy(int)
    times       = table["cumulated_tool_contact_time"].to_numpy(float)
    wear        = table["wear"].to_numpy(float)
    machines    = table["machine"].to_numpy(int)

    vb_norm       = np.clip(wear / WEAR_SCALE_UM, 0.0, 1.0)
    rul_seconds   = np.zeros(len(table), float)
    slope         = np.zeros(len(table), float)
    age_fraction  = np.zeros(len(table), float)

    for _, grp in table.groupby("tool", sort=False):
        idx   = grp.index.to_numpy()
        t     = times[idx]
        v     = vb_norm[idx]
        t_end = float(t[-1])
        rul_seconds[idx]  = np.maximum(0.0, t_end - t)
        age_fraction[idx] = t / max(t_end, 1e-12)
        # Strictly causal slope: the feature at run k uses only runs ≤ k.
        # The first run of a trajectory has no earlier reading to difference
        # against, so its slope is 0 (no rate information is available yet)
        # rather than borrowing the backward difference from run 1 -- using
        # slope_values[1] here would pull in run 1's own reading, a
        # one-run-ahead leak at exactly this first row of every trajectory.
        slope_values = np.zeros(len(idx), float)
        if len(idx) > 1:
            slope_values[1:] = np.diff(v) / np.maximum(np.diff(t), 1e-12)
        slope[idx] = np.clip(slope_values, 0.0, None)

    slope_scale = float(np.percentile(slope, 99.0))
    if not np.isfinite(slope_scale) or slope_scale <= 0:
        raise ValueError("Wear-slope 99th percentile is non-positive.")

    raw_signals   = table[SIGNAL_FEATURE_NAMES].to_numpy(float)
    signal_scales = {}
    for m in machine_ids:
        mask = machines == m
        raw  = raw_signals[mask]
        lo   = np.percentile(raw, 1.0, axis=0)
        hi   = np.percentile(raw, 99.0, axis=0)
        signal_scales[m] = {"lowerP01": lo, "upperP99": hi}

    machine_oh = np.zeros((len(table), len(machine_ids)), float)
    for col, m in enumerate(machine_ids):
        machine_oh[:, col] = machines == m

    # RFs do not require feature scaling. Keeping raw signal statistics avoids
    # estimating preprocessing parameters from held-out tool trajectories.
    X = np.column_stack([vb_norm, slope, raw_signals, machine_oh])
    feature_names = (
        ["VB_norm", "VB_slope_per_second"]
        + list(SIGNAL_FEATURE_NAMES)
        + [f"machine_M{m}" for m in machine_ids]
    )

    # Trajectory profiles for simulation condition mapping
    vb_maps       = {}
    signal_profiles = {}
    for m in machine_ids:
        mask = machines == m
        ag, vb_grid_2d = _trajectory_map(age_fraction[mask], vb_norm[mask, None])
        _, sig_grid    = _trajectory_map(age_fraction[mask], raw_signals[mask])
        vb_maps[m]       = {"ageGrid": ag, "vbGrid": np.clip(vb_grid_2d[:, 0], 0.0, 1.0)}
        signal_profiles[m] = {"ageGrid": ag, "featureGrid": sig_grid,
                               "featureNames": list(SIGNAL_FEATURE_NAMES)}

    valid = (
        np.all(np.isfinite(X), axis=1)
        & np.isfinite(rul_seconds)
        & (rul_seconds >= 0)
    )
    n_valid = int(valid.sum())
    print(f"  Samples: {n_valid}/{len(table)}  features={X.shape[1]}  "
          f"tools={len(np.unique(tool_ids[valid]))}  machines={machine_ids}  "
          f"slope_scale_p99={slope_scale:.6g}")

    return {
        "X": X[valid],
        "Y": rul_seconds[valid],
        "groups": tool_ids[valid],
        "machines": machines[valid],
        "machine_ids": machine_ids,
        "feature_names": feature_names,
        "slope_scale": slope_scale,
        "vb_maps": vb_maps,
        "signal_profiles": signal_profiles,
        "signal_scales": signal_scales,
        "wear_scale_um": WEAR_SCALE_UM,
    }


# ---------------------------------------------------------------------------
# Random Forest training
# ---------------------------------------------------------------------------

def train_random_forest(
    X: np.ndarray,
    Y: np.ndarray,
    seed: int = 42,
    n_estimators: int = 500,
    max_features: str = "sqrt",
    min_samples_leaf: int = 5,
) -> RandomForestRegressor:
    """Fit a Random Forest on the FULL dataset (no sample cap)."""
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(X, Y)
    return rf


# ---------------------------------------------------------------------------
# Split conformal calibration
# ---------------------------------------------------------------------------

def _life_frac_to_phase(life_frac: np.ndarray) -> np.ndarray:
    """Map remaining-life fraction [0, 1] → "early" / "mid" / "late".

    Thresholds chosen so roughly equal-population strata form across typical
    Gamma-process wear trajectories (early: >60% life, late: <30%).
    """
    phase = np.full(len(life_frac), "late", dtype=object)
    phase[life_frac > 0.30] = "mid"
    phase[life_frac > 0.60] = "early"
    return phase


def _remaining_life_fraction(
    Y: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """Normalize remaining life within each physical tool trajectory."""
    Y = np.asarray(Y, float)
    groups = np.asarray(groups)
    fractions = np.zeros(len(Y), float)
    for tool in np.unique(groups):
        mask = groups == tool
        fractions[mask] = Y[mask] / max(float(np.max(Y[mask])), 1.0)
    return np.clip(fractions, 0.0, 1.0)


def tool_trajectory_scores(
    residuals: np.ndarray,
    life_fraction: np.ndarray,
) -> dict[str, float]:
    """Return one simultaneous residual score per trajectory and wear phase."""
    residuals = np.asarray(residuals, float).reshape(-1)
    life_fraction = np.asarray(life_fraction, float).reshape(-1)
    if len(residuals) == 0 or len(residuals) != len(life_fraction):
        raise ValueError("Trajectory residuals and life fractions must align.")
    if not np.all(np.isfinite(residuals)) or np.any(residuals < 0):
        raise ValueError("Trajectory residuals must be finite and non-negative.")

    phases = _life_frac_to_phase(life_fraction)
    global_score = float(np.max(residuals))
    scores = {"global": global_score}
    for phase in ("early", "mid", "late"):
        mask = phases == phase
        scores[phase] = (
            float(np.max(residuals[mask])) if np.any(mask) else global_score
        )
    return scores


def _cvplus_rank_pair(n_tools: int, alpha: float) -> tuple[int, int]:
    """Order-statistic rank pair (1-indexed) used by _cvplus_bounds.

    Exposed standalone (not just inlined in _cvplus_bounds) so callers can
    check whether two different alpha values collapse to the SAME rank pair
    at a given sample size. This happens whenever n_tools is small enough
    that floor/ceil round two distinct alpha targets to the same integer --
    e.g. n_tools=8 (this project's leave-one-tool-out calibration size, 9
    tools minus 1 held out) makes alpha=0.10 and alpha=0.05 both resolve to
    (lower_rank=1, upper_rank=8): floor(9*.10)=0->1 and floor(9*.05)=0->1;
    ceil(9*.90)=9->8 and ceil(9*.95)=9->8. When this collapses, a reported
    "Cov90" and "Cov95" are the same interval's empirical coverage computed
    twice, not two independent checks at two different nominal levels --
    worth knowing explicitly rather than discovering by comparing floats.
    """
    lower_rank = max(1, int(np.floor((n_tools + 1) * float(alpha))))
    upper_rank = min(n_tools, int(np.ceil((n_tools + 1) * (1.0 - float(alpha)))))
    return lower_rank, upper_rank


def _cvplus_bounds(
    lower_candidates: np.ndarray,
    upper_candidates: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Finite-sample order-statistic bounds across tool-level candidates."""
    lower_candidates = np.asarray(lower_candidates, float)
    upper_candidates = np.asarray(upper_candidates, float)
    if lower_candidates.shape != upper_candidates.shape:
        raise ValueError("Lower and upper CV+ candidates must align.")
    if lower_candidates.ndim == 1:
        lower_candidates = lower_candidates[:, None]
        upper_candidates = upper_candidates[:, None]
    n_tools = lower_candidates.shape[0]
    if n_tools < 2:
        raise ValueError("At least two tool-level CV+ candidates are required.")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1.")

    lower_rank, upper_rank = _cvplus_rank_pair(n_tools, alpha)
    lower = np.sort(lower_candidates, axis=0)[lower_rank - 1]
    upper = np.sort(upper_candidates, axis=0)[upper_rank - 1]
    return lower, upper


def _phase_for_damage(damage_frac: float | None) -> str:
    if damage_frac is None:
        return "global"
    life_fraction = float(np.clip(1.0 - damage_frac, 0.0, 1.0))
    return str(_life_frac_to_phase(np.array([life_fraction]))[0])


def build_tool_cvplus_ensemble(
    X: np.ndarray,
    Y: np.ndarray,
    groups: np.ndarray,
    machines: np.ndarray,
    *,
    seed: int = 42,
    n_estimators: int = 500,
) -> list[dict]:
    """Fit one model per held-out tool and store one score per trajectory."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    groups = np.asarray(groups)
    machines = np.asarray(machines)
    life_fraction = _remaining_life_fraction(Y, groups)
    ensemble = []
    for fold_index, tool in enumerate(np.unique(groups)):
        held = groups == tool
        fit = ~held
        rf = train_random_forest(
            X[fit],
            Y[fit],
            seed=seed + 37 * fold_index,
            n_estimators=n_estimators,
        )
        residuals = np.abs(Y[held] - rf.predict(X[held]))
        scores = tool_trajectory_scores(
            residuals, life_fraction[held]
        )
        machine_values = np.unique(machines[held])
        ensemble.append({
            "tool_id": int(tool),
            "machine_id": int(machine_values[0]),
            "rf": rf,
            "scores": scores,
            "n_rows": int(np.sum(held)),
        })
    return ensemble


def cvplus_interval(
    model: dict,
    X: np.ndarray,
    *,
    damage_frac: float | None = None,
    alpha: float | None = None,
) -> tuple[float, float]:
    """Return a trajectory-calibrated CV+ interval for one runtime state."""
    ensemble = model.get("tool_cvplus_ensemble", [])
    if len(ensemble) < 2:
        raise ValueError(
            "Tool-level CV+ ensemble missing; retrain the schema-v6 RUL model."
        )
    X = np.atleast_2d(np.asarray(X, float))
    phase = _phase_for_damage(damage_frac)
    lower_candidates = []
    upper_candidates = []
    for entry in ensemble:
        prediction = float(np.mean(entry["rf"].predict(X)))
        scores = entry["scores"]
        score = float(scores.get(phase, scores["global"]))
        lower_candidates.append(prediction - score)
        upper_candidates.append(prediction + score)
    lower, upper = _cvplus_bounds(
        np.asarray(lower_candidates),
        np.asarray(upper_candidates),
        float(model.get("alpha", CONFORMAL_ALPHA) if alpha is None else alpha),
    )
    return max(0.0, float(lower[0])), max(0.0, float(upper[0]))


# ---------------------------------------------------------------------------
# Tool-group holdout: leave-tool-out cross-validation for coverage reporting
# ---------------------------------------------------------------------------

def tool_group_holdout_cv(
    X: np.ndarray,
    Y: np.ndarray,
    groups: np.ndarray,
    machines: np.ndarray,
    n_folds: int = 9,
    seed: int = 42,
    alpha: float = CONFORMAL_ALPHA,
    n_estimators: int = 200,
) -> list[dict]:
    """Nested leave-tool-out validation of trajectory-level CV+ intervals.

    For each test tool, every remaining tool acts once as the calibration
    trajectory while the RF is fitted on the other tools. Thus no row from the
    test trajectory or calibration trajectory enters that candidate model.
    """
    unique_tools = np.unique(groups)
    if int(n_folds) != len(unique_tools):
        raise ValueError(
            "Trajectory-level validation requires one fold per physical tool "
            f"({len(unique_tools)} folds for this dataset)."
        )
    life_fraction = _remaining_life_fraction(Y, groups)
    fold_results = []
    for fold_index, test_tool in enumerate(unique_tools):
        test_mask = groups == test_tool
        calibration_tools = [
            tool for tool in unique_tools if tool != test_tool
        ]
        lower_candidates = []
        upper_candidates = []
        point_candidates = []
        fit_sizes = []
        test_phases = _life_frac_to_phase(life_fraction[test_mask])

        for calibration_index, calibration_tool in enumerate(
            calibration_tools
        ):
            calibration_mask = groups == calibration_tool
            fit_mask = ~(test_mask | calibration_mask)
            fit_sizes.append(int(fit_mask.sum()))
            rf = train_random_forest(
                X[fit_mask],
                Y[fit_mask],
                seed=seed + fold_index * 101 + calibration_index,
                n_estimators=n_estimators,
            )
            calibration_residuals = np.abs(
                Y[calibration_mask] - rf.predict(X[calibration_mask])
            )
            scores = tool_trajectory_scores(
                calibration_residuals,
                life_fraction[calibration_mask],
            )
            prediction = rf.predict(X[test_mask])
            score_vector = np.array([
                scores.get(str(phase), scores["global"])
                for phase in test_phases
            ])
            lower_candidates.append(prediction - score_vector)
            upper_candidates.append(prediction + score_vector)
            point_candidates.append(prediction)

        lower90, upper90 = _cvplus_bounds(
            np.asarray(lower_candidates),
            np.asarray(upper_candidates),
            alpha,
        )
        lower95, upper95 = _cvplus_bounds(
            np.asarray(lower_candidates),
            np.asarray(upper_candidates),
            0.05,
        )
        # At small n_cal_tools, the alpha=0.10 and alpha=0.05 rank formulas
        # can round to the SAME integer ranks (e.g. n_cal_tools=8 -> both
        # give rank (1, 8)), making "Cov90" and "Cov95" the same interval's
        # coverage reported twice rather than two independent checks. Detect
        # this explicitly per fold rather than leaving it to be noticed by
        # comparing floats (see _cvplus_rank_pair's docstring).
        ranks_collapsed = _cvplus_rank_pair(len(calibration_tools), alpha) == \
            _cvplus_rank_pair(len(calibration_tools), 0.05)
        point_prediction = np.mean(np.asarray(point_candidates), axis=0)
        residuals = np.abs(Y[test_mask] - point_prediction)
        covered90 = (Y[test_mask] >= lower90) & (Y[test_mask] <= upper90)
        covered95 = (Y[test_mask] >= lower95) & (Y[test_mask] <= upper95)
        fold_res = {
            "fold": fold_index,
            "heldout_tools": [int(test_tool)],
            "heldout_machine": int(np.unique(machines[test_mask])[0]),
            "n_train_fit": int(round(float(np.mean(fit_sizes)))),
            "n_cal": len(calibration_tools),
            "n_cal_tools": len(calibration_tools),
            "n_test": int(test_mask.sum()),
            "rmse_s": float(np.sqrt(np.mean(
                np.square(Y[test_mask] - point_prediction)
            ))),
            "mae_s": float(np.mean(residuals)),
            "coverage_90": float(np.mean(covered90)),
            "coverage_95": float(np.mean(covered95)),
            "trajectory_covered_90": bool(np.all(covered90)),
            "trajectory_covered_95": bool(np.all(covered95)),
            "mean_interval_width_90_s": float(np.mean(upper90 - lower90)),
            "mean_interval_width_95_s": float(np.mean(upper95 - lower95)),
            "cov90_cov95_ranks_collapsed": bool(ranks_collapsed),
        }
        fold_results.append(fold_res)
        collapse_note = "  [Cov90==Cov95: ranks collapsed at this n_cal_tools]" if ranks_collapsed else ""
        print(
            f"  Tool {int(test_tool)}: "
            f"RMSE={fold_res['rmse_s']:.1f}s  "
            f"MAE={fold_res['mae_s']:.1f}s  "
            f"Cov90={fold_res['coverage_90']:.3f}  Cov95={fold_res['coverage_95']:.3f}"
            f"{collapse_note}"
        )

    return fold_results


# ---------------------------------------------------------------------------
# Final model assembly
# ---------------------------------------------------------------------------

def train_final_model(
    prepared: dict,
    seed: int = 42,
    alpha: float = CONFORMAL_ALPHA,
    n_estimators: int = 500,
) -> dict:
    """Train the point RF and leave-one-tool-out CV+ ensemble."""
    X = prepared["X"]
    Y = prepared["Y"]
    groups = prepared["groups"]
    machines = prepared["machines"]

    print(
        f"  Final RF: point fit={len(X)} samples; "
        f"CV+ calibration units={len(np.unique(groups))} tools"
    )
    rf = train_random_forest(
        X, Y,
        seed=seed,
        n_estimators=n_estimators,
    )
    ensemble = build_tool_cvplus_ensemble(
        X,
        Y,
        groups,
        machines,
        seed=seed,
        n_estimators=n_estimators,
    )

    Y_train_pred = rf.predict(X)
    rmse_train = float(np.sqrt(np.mean((Y - Y_train_pred) ** 2)))
    print(f"  Train RMSE: {rmse_train:.1f} s  ({rmse_train/3600:.3f} h)")

    return {
        "rf": rf,
        "tool_cvplus_ensemble": ensemble,
        "alpha": alpha,
        "n_fit": len(X),
        "n_cal": len(ensemble),
        "n_cal_tools": len(ensemble),
        "rmse_train_s": rmse_train,
    }


# ---------------------------------------------------------------------------
# Prediction helper (public runtime-inference entry point for a saved bundle)
# ---------------------------------------------------------------------------

def conformal_predict(
    model: dict,
    X: np.ndarray,
    damage_frac: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Return point predictions, mean prediction, and CV+ half-width.

    The half-width conservatively encloses the possibly asymmetric
    trajectory-level CV+ interval around the full-data RF point prediction.
    """
    rf = model["rf"]
    y_pred = rf.predict(np.atleast_2d(X))
    point = float(np.mean(y_pred))
    if model.get("tool_cvplus_ensemble"):
        lower, upper = cvplus_interval(
            model, X, damage_frac=damage_frac
        )
        half_width = max(point - lower, upper - point, 0.0)
        return y_pred, point, float(half_width)

    raise ValueError(
        "Tool-level CV+ ensemble missing; retrain the schema-v6 RUL model."
    )


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", default=None)
    p.add_argument(
        "--feature-cache",
        default=os.path.join("artifacts", "cache", "milling_timeseries_features.csv"),
    )
    p.add_argument(
        "--output-dir",
        default=os.path.join("artifacts", "rul"),
    )
    p.add_argument("--rebuild-features", action="store_true")
    p.add_argument("--validate-h5", choices=("none", "sample", "all"), default="sample")
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-cv-folds", type=int, default=9)
    p.add_argument("--alpha", type=float, default=CONFORMAL_ALPHA)
    return p.parse_args(argv)


def main(
    argv: list[str] | None = None,
    project_dir: str | Path | None = None,
):
    args = _parse_args(argv)
    project_dir = (
        Path(__file__).resolve().parent
        if project_dir is None
        else Path(project_dir).resolve()
    )

    dataset_dir = find_dataset_dir(project_dir, args.dataset_dir)
    cache_path  = Path(args.feature_cache)
    if not cache_path.is_absolute():
        cache_path = project_dir / cache_path
    output_dir  = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Conformal Random Forest RUL training")
    print(f"Dataset: {dataset_dir}")
    print(f"Conformal alpha: {args.alpha}  (target coverage: {(1-args.alpha)*100:.0f}%)")
    print("=" * 60)

    index   = load_file_index(dataset_dir)
    summary = dataset_summary(index)
    print(json.dumps(summary, indent=2))

    if args.validate_h5 != "none":
        audit = validate_hdf5_labels(dataset_dir, index, mode=args.validate_h5)
        print(f"HDF5 audit: {audit}")

    table = load_or_build_feature_table(
        dataset_dir, index, cache_path, rebuild=args.rebuild_features
    )
    prepared = prepare_training_data(table)

    print("\n--- Leave-tool-group-out cross-validation ---")
    fold_results = tool_group_holdout_cv(
        prepared["X"],
        prepared["Y"],
        prepared["groups"],
        prepared["machines"],
        n_folds=args.n_cv_folds,
        seed=args.seed,
        alpha=args.alpha,
        n_estimators=min(args.n_estimators, 200),
    )

    print("\n--- Training final model on full dataset ---")
    final = train_final_model(
        prepared, seed=args.seed, alpha=args.alpha, n_estimators=args.n_estimators
    )

    # Assemble bundle (schema v6: tool-trajectory CV+ calibration)
    cov_ranks_collapsed = all(f["cov90_cov95_ranks_collapsed"] for f in fold_results)
    coverage_claim = (
        "cluster-aware empirical CV+ coverage over physical tool "
        "trajectories; no exact row-level guarantee claimed"
    )
    if cov_ranks_collapsed:
        coverage_claim += (
            f". With n_cal_tools={fold_results[0]['n_cal_tools']} calibration units per "
            "leave-one-tool-out fold, the order-statistic rank formula makes the nominal "
            "90% and 95% targets collapse to the identical rank pair (see "
            "cvSummary.cov90_cov95_ranks_collapsed) -- the reported Cov90/Cov95 are the "
            "same interval's empirical coverage at two nominal labels, not two "
            "independent checks"
        )
    bundle = {
        "schemaVersion": MODEL_SCHEMA_VERSION,
        "modelType": "ConformalRandomForest",
        "dataset": "multivariate_milling_hdf5",
        "datasetDOI": DATASET_DOI,
        "datasetMeta": summary,
        "conformalAlpha": args.alpha,
        "conformalCoverageTarget": 1.0 - args.alpha,
        "coverageClaim": coverage_claim,
        "nEstimators": args.n_estimators,
        "nFeaturesPerSample": int(prepared["X"].shape[1]),
        "featureNames": prepared["feature_names"],
        "wearScaleUm": WEAR_SCALE_UM,
        # Model artefacts
        "mdlMill": {
            "rf": final["rf"],
            "tool_cvplus_ensemble": final["tool_cvplus_ensemble"],
            "alpha": final["alpha"],
            "n_fit": final["n_fit"],
            "n_cal": final["n_cal"],
            "n_cal_tools": final["n_cal_tools"],
            "rmse_train_s": final["rmse_train_s"],
            "nFeatures": int(prepared["X"].shape[1]),
            "modelType": "ConformalRF",
            "featureNames": prepared["feature_names"],
            "meta": {
                "machineIds": prepared["machine_ids"],
                "wearScaleUm": WEAR_SCALE_UM,
            },
        },
        # Runtime simulation profiles (unchanged from original design)
        "vbMapMillByMachine":            prepared["vb_maps"],
        "signalFeatureProfilesByMachine": prepared["signal_profiles"],
        "signalFeatureScaleByMachine":   prepared["signal_scales"],
        # Cross-validation results for reporting
        "cvFoldResults": fold_results,
        "cvSummary": {
            "mean_rmse_s":   float(np.mean([f["rmse_s"]      for f in fold_results])),
            "mean_mae_s":    float(np.mean([f["mae_s"]       for f in fold_results])),
            "mean_cov90":    float(np.mean([f["coverage_90"] for f in fold_results])),
            "mean_cov95":    float(np.mean([f["coverage_95"] for f in fold_results])),
            "std_rmse_s":    float(np.std ([f["rmse_s"]      for f in fold_results], ddof=1)),
            "trajectory_cov90": float(np.mean([
                f["trajectory_covered_90"] for f in fold_results
            ])),
            "trajectory_cov95": float(np.mean([
                f["trajectory_covered_95"] for f in fold_results
            ])),
            "n_independent_tools": len(np.unique(prepared["groups"])),
            "cov90_cov95_ranks_collapsed": bool(cov_ranks_collapsed),
        },
    }

    model_path = output_dir / "conformal_rul_model.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(bundle, fh)

    cv_path = output_dir / "conformal_rul_cv_results.csv"
    pd.DataFrame(fold_results).to_csv(cv_path, index=False)

    summary_path = output_dir / "conformal_rul_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump({
            "schemaVersion": MODEL_SCHEMA_VERSION,
            "modelType": "ConformalRandomForest",
            "conformalAlpha": args.alpha,
            "nSamples": len(prepared["X"]),
            "nFit": final["n_fit"],
            "nCal": final["n_cal"],
            "nCalTools": final["n_cal_tools"],
            "rmseTrainS": final["rmse_train_s"],
            "cvSummary": bundle["cvSummary"],
            "coverageClaim": bundle["coverageClaim"],
            "featureNames": prepared["feature_names"],
            "nCVFolds": args.n_cv_folds,
        }, fh, indent=2)

    print("\n" + "=" * 60)
    print(f"Saved model      -> {model_path}")
    print(f"Saved CV results -> {cv_path}")
    print(f"Saved summary    -> {summary_path}")
    print("\nCross-validation summary:")
    print(f"  RMSE: {bundle['cvSummary']['mean_rmse_s']:.1f} ± "
          f"{bundle['cvSummary']['std_rmse_s']:.1f} s")
    print(f"  Cov90: {bundle['cvSummary']['mean_cov90']:.3f}  "
          f"Cov95: {bundle['cvSummary']['mean_cov95']:.3f}"
          + ("  [ranks collapsed at this n_cal_tools -- these are the same "
             "interval reported twice, not independent checks]"
             if bundle["cvSummary"]["cov90_cov95_ranks_collapsed"] else ""))


def train_and_save_conformal_rul(
    project_dir,
    out_path=None,
    conformal_alpha: float = CONFORMAL_ALPHA,
    n_trees: int = 500,
    rng_seed: int = 42,
):
    """Programmatic entry point for Phase 2 (called from main.py).

    Trains the conformal RF RUL model and saves the bundle to out_path.
    All arguments mirror their CLI equivalents.
    """
    project_dir = Path(project_dir).resolve()
    if out_path is None:
        out_path = project_dir / "artifacts" / "rul" / "conformal_rul_model.pkl"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    main(
        [
            "--output-dir", str(out_path.parent),
            "--alpha", str(conformal_alpha),
            "--n-estimators", str(n_trees),
            "--seed", str(rng_seed),
        ],
        project_dir=project_dir,
    )
    generated = out_path.parent / "conformal_rul_model.pkl"
    if generated.resolve() != out_path.resolve():
        generated.replace(out_path)


if __name__ == "__main__":
    main()
