"""Wear-only feature ablation for the Random Forest RUL baseline.

Directly answers the peer-review critique that the Gamma/Inverse-Gaussian-vs-
Random-Forest RUL comparison (``degradation_reliability_translation.
compare_ig_vs_conformal_rul``, Table S8) "is not entirely fair": the
production Random Forest (``train_conformal_rul.py``) is fed a 10-column
feature vector -- wear level, its causal slope, 5 raw cutting-force/
vibration-derived signal statistics (``SIGNAL_FEATURE_NAMES``), and a
3-column machine one-hot -- while the stochastic degradation-process models
see only the wear trajectory itself (age and, for state-dependent variants,
prior wear level). That asymmetry is real and does not disappear by
reframing it; what this script does is QUANTIFY how much of the RF's
accuracy advantage over IG actually comes from those extra sensor features,
versus from the RF's general nonparametric flexibility, by retraining a
second RF restricted to the SAME wear-trajectory-only information the
stochastic models use (VB_norm, its causal slope, and machine identity --
dropping the 5 raw signal columns) and re-running the identical, already-
validated leave-one-tool-out CV+ procedure (``tool_group_holdout_cv``) on it.

This module does NOT modify train_conformal_rul.py -- it imports its
already-audited data-prep and CV machinery read-only and reuses them
verbatim, so the wear-only run is apples-to-apples with the production
full-feature run on every axis except the feature columns themselves (same
RF hyperparameters, same leave-one-tool-out fold structure, same CV+
conformal scoring). The full-feature numbers are read from the already-
produced artifacts/rul/conformal_rul_cv_results.csv rather than re-run, to
avoid a redundant second training pass of a model that hasn't changed.

Usage:
    ./.venv/Scripts/python.exe rf_feature_ablation.py
Output:
    artifacts/rul/rf_feature_ablation.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from milling_dataset import find_dataset_dir, load_file_index, load_or_build_feature_table
from train_conformal_rul import prepare_training_data, tool_group_holdout_cv

PROJECT_DIR = Path(__file__).resolve().parent
RUL_DIR = PROJECT_DIR / "artifacts" / "rul"


def _wear_only_columns(feature_names: list[str]) -> list[int]:
    """Column indices for VB_norm, VB_slope_per_second, and every
    machine_M* one-hot column -- i.e. every column EXCEPT the 5 raw signal
    features -- located by name rather than a hardcoded index range, so
    this stays correct if the feature vector's column order or count ever
    changes upstream."""
    keep = [
        i for i, name in enumerate(feature_names)
        if name in ("VB_norm", "VB_slope_per_second") or name.startswith("machine_M")
    ]
    if len(keep) != len(feature_names) - 5:
        raise ValueError(
            f"Expected to drop exactly 5 raw-signal columns; kept {len(keep)} of "
            f"{len(feature_names)} ({feature_names}). Upstream feature layout may have changed."
        )
    return keep


def run_wear_only_ablation(
    n_folds: int = 9, seed: int = 42, alpha: float = 0.10, n_estimators: int = 200,
) -> dict:
    """Retrains the RF on wear-trajectory-only features (matching what the
    stochastic degradation-process models see) via the identical
    leave-one-tool-out CV+ procedure the production full-feature RF uses,
    and compares both against IG's own already-computed RMSE (Table S8)."""
    dataset_dir = find_dataset_dir(PROJECT_DIR)
    index = load_file_index(dataset_dir)
    cache_path = RUL_DIR.parent / "cache" / "milling_timeseries_features.csv"
    table = load_or_build_feature_table(dataset_dir, index, cache_path, rebuild=False)
    prepared = prepare_training_data(table)

    keep_cols = _wear_only_columns(prepared["feature_names"])
    X_wear_only = prepared["X"][:, keep_cols]
    wear_only_feature_names = [prepared["feature_names"][i] for i in keep_cols]
    print(f"Wear-only feature set ({len(keep_cols)} cols): {wear_only_feature_names}")

    print("\n-- Leave-tool-group-out CV, wear-trajectory-only features --")
    fold_results = tool_group_holdout_cv(
        X_wear_only, prepared["Y"], prepared["groups"], prepared["machines"],
        n_folds=n_folds, seed=seed, alpha=alpha, n_estimators=n_estimators,
    )
    wear_only_by_tool = {int(f["heldout_tools"][0]): f for f in fold_results}

    full_csv = RUL_DIR / "conformal_rul_cv_results.csv"
    if not full_csv.is_file():
        raise FileNotFoundError(
            f"{full_csv} not found -- run train_conformal_rul.py first to produce the "
            "full-feature baseline this ablation compares against."
        )
    full_df = pd.read_csv(full_csv)
    full_by_tool = {int(row["heldout_tools"].strip("[]")): row for _, row in full_df.iterrows()}

    ig_json = RUL_DIR.parent / "degradation_models" / "extended_rigor.json"
    ig_by_tool = {}
    if ig_json.is_file():
        with open(ig_json, "r", encoding="utf-8") as fh:
            extended = json.load(fh)
        ig_by_tool = {int(k): v for k, v in extended.get("ig_vs_conformal_rul", {}).items()}

    comparison = {}
    for tool_id in sorted(wear_only_by_tool):
        wo = wear_only_by_tool[tool_id]
        full = full_by_tool.get(tool_id)
        ig = ig_by_tool.get(tool_id)
        row = {
            "wear_only_rmse_s": wo["rmse_s"],
            "full_feature_rmse_s": float(full["rmse_s"]) if full is not None else None,
            "ig_rmse_s": ig["ig_rmse_s"] if ig else None,
        }
        if full is not None:
            row["wear_only_rmse_ratio_vs_full"] = float(wo["rmse_s"] / float(full["rmse_s"]))
        if ig is not None:
            row["ig_rmse_ratio_vs_wear_only_rf"] = float(ig["ig_rmse_s"] / wo["rmse_s"])
        comparison[tool_id] = row
        full_rmse_str = "n/a" if full is None else f"{float(full['rmse_s']):.1f}"
        ig_rmse_str = "n/a" if ig is None else f"{ig['ig_rmse_s']:.1f}"
        print(
            f"  Tool {tool_id}: wear-only RF RMSE={wo['rmse_s']:.1f}s  "
            f"full-feature RF RMSE={full_rmse_str}s  IG RMSE={ig_rmse_str}s"
        )

    result = {
        "wear_only_feature_names": wear_only_feature_names,
        "n_wear_only_features": len(wear_only_feature_names),
        "n_full_features": len(prepared["feature_names"]),
        "fold_results": fold_results,
        "comparison_by_tool": comparison,
    }
    return result


if __name__ == "__main__":
    result = run_wear_only_ablation()
    out_path = RUL_DIR / "rf_feature_ablation.json"
    # Atomic write (temp file + os.replace) -- see run_degradation_study.py's
    # _atomic_json_dump for why: an interrupted process should never leave
    # a truncated JSON file in place of the last known-good artifact.
    tmp_path = out_path.with_suffix(out_path.suffix + f".tmp{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    os.replace(tmp_path, out_path)
    print(f"\nSaved -> {out_path}")
