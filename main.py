"""Upstream artifact regeneration for the MillingRESS real-data pipeline.

Produces the two artifacts the active degradation-process-comparison study
(``run_degradation_study.py``) reads as inputs:

  1  Gamma process degradation model fit to real VB data
     -> artifacts/gamma/gamma_process_params.json
  2  Conformal RF-based RUL predictor
     -> artifacts/rul/conformal_rul_model.pkl, conformal_rul_cv_results.csv

Both artifacts already exist on disk from this project's prior runs; this
script exists so they can be regenerated from the raw dataset if ever lost,
updated, or needed on a fresh clone. Neither phase is otherwise on the
critical path of the active study.

Run all phases:
    uv run python main.py

Run specific phases:
    uv run python main.py --phases 1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACT    = PROJECT_DIR / "artifacts"


def _ensure_dirs():
    for sub in ("gamma", "rul"):
        (ARTIFACT / sub).mkdir(parents=True, exist_ok=True)


# ── Phase 1: Gamma process ────────────────────────────────────────────────────

def phase1_gamma(args):
    print("\n" + "=" * 60)
    print("PHASE 1 — Gamma Process Degradation Fitting")
    print("=" * 60)

    from milling_dataset import find_dataset_dir, load_file_index
    from gamma_process import fit_all_groups, save_gamma_params, validate_gamma_fit

    ds_dir = find_dataset_dir(PROJECT_DIR)
    index  = load_file_index(ds_dir, require_h5_files=False)
    params = fit_all_groups(index, window_size=args.gamma_window)
    val = validate_gamma_fit(index, params)
    failed = [name for name, result in val.items()
              if not result.get("validation_passed", False)]
    if failed:
        # Do not hard-block: the active degradation-process-family comparison
        # study needs the Gamma fit regardless of whether it individually
        # calibrates, exactly as it already uses the Wiener fit despite Wiener
        # failing PIT-KS everywhere -- a family failing to calibrate for a
        # given machine is a reportable comparison result (see Table 1), not
        # a broken artifact. Warn loudly instead of silently proceeding.
        print(
            f"  WARNING: Gamma calibration failed for {failed} under the "
            "200-seed PIT-KS protocol -- saving anyway; this is reported "
            "honestly in the degradation-comparison table, not corrected."
        )
    save_gamma_params(params, ARTIFACT / "gamma")
    with open(ARTIFACT / "gamma" / "validation.json", "w") as f:
        json.dump(val, f, indent=2)

    print("Phase 1 complete.")


# ── Phase 2: Conformal RUL ────────────────────────────────────────────────────

def phase2_rul(args):
    print("\n" + "=" * 60)
    print("PHASE 2 — Conformal RUL Model Training")
    print("=" * 60)

    from train_conformal_rul import train_and_save_conformal_rul

    train_and_save_conformal_rul(
        project_dir=PROJECT_DIR,
        out_path=ARTIFACT / "rul" / "conformal_rul_model.pkl",
        conformal_alpha=args.conformal_alpha,
        n_trees=args.rf_trees,
        rng_seed=args.seed,
    )
    print("Phase 2 complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(description="MillingRESS upstream artifact regeneration")
    p.add_argument("--phases", nargs="+", type=int, default=[1, 2],
                   help="Phases to run (default: 1 2)")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--gamma-window", type=int, default=15,  dest="gamma_window")
    p.add_argument("--conformal-alpha", type=float, default=0.10, dest="conformal_alpha")
    p.add_argument("--rf-trees",   type=int,   default=500, dest="rf_trees")
    return p.parse_args()


def main():
    args = _parse()
    _ensure_dirs()

    phase_map = {
        1: phase1_gamma,
        2: phase2_rul,
    }

    t0 = time.time()
    for ph in sorted(set(args.phases)):
        fn = phase_map.get(ph)
        if fn is None:
            print(f"Unknown phase {ph}, skipping.")
            continue
        fn(args)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"All requested phases complete.  Total wall time: {elapsed/3600:.2f} h")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
