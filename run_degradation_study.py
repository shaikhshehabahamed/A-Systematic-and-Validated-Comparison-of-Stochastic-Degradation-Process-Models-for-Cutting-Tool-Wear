"""Orchestration script for the degradation process-family comparison study.

Standalone paper: Gamma vs. Inverse Gaussian vs. Wiener-with-drift degradation
processes fit to the real 3-machine, 9-tool milling flank-wear dataset
(Denkena, Klemme & Stiehl, Data in Brief 50:109574, 2023), with reliability
translation. Independent of, and does not modify, the separate (abandoned)
MillingRESS MARL scheduling pipeline in this same directory (main.py,
gamma_process.py's production Gamma fit, train_conformal_rul.py) -- those
are reused read-only where noted.

Phases
------
  1  Fit and compare all ten degradation-process families per machine
     (Gamma, Inverse Gaussian, Wiener, plus the 7 variants: state-dependent
     and random-effects Gamma/IG, truncated and multi-stage Wiener,
     Generalized Inverse Gaussian); also derives the AIC-vs-BIC age-model
     criterion sensitivity (feeds Table S13) and each variant's nested
     likelihood-ratio test against its baseline family (feeds Table S14) --
     both free, reusing fields already computed above, no refit
  2  Reliability translation: ground truth vs. process-implied MTTFF/hazard
  3  Camera-ready tables -> paper_assets/tables/{main,supplementary}/
     (Tables 1-4 main text, S1-S19 supplementary, as each upstream phase has
     been run; see degradation_tables.py's module docstring for the mapping)
  4  Camera-ready figures -> paper_assets/figures/{main,supplementary}/
     (Figures 1-3 main text, S1 supplementary)
  5  Robustness: leave-one-tool-out CV likelihood, window-size sweep,
     EOL-threshold sensitivity, and the formal per-tool outlier significance
     bootstrap (feeds Tables S1-S3 and Table 2's significance columns).
  6  Extended rigor: bootstrap CI on the AIC gap (window-level, and,
     addressing the independence-assumption critique below, TWO
     dependence-aware alternatives -- a block bootstrap resampling whole
     tools and a moving-block bootstrap resampling within-tool blocks),
     pooled-across-machines baseline,
     IG-vs-conformal-RF RUL comparison and its threshold-mismatch
     decomposition, maintenance-cost/age-replacement translation, its
     cost-ratio sensitivity sweep and a complementary block-bootstrap CI
     propagating fitted-parameter uncertainty, a model-recovery positive
     control, Vuong's non-nested hypothesis test (cluster-robust by tool,
     not the naive i.i.d. variance -- see vuong_test's docstring),
     PIT-residual autocorrelation by tool (the direct check of the
     likelihood's independence assumption), and per-tool fit diagnostics
     (feeds Table 4 and Tables S4-S6, S8-S12, S15-S19); requires
     artifacts/rul/conformal_rul_cv_results.csv from the separate
     (already-run) conformal RUL pipeline.

Runtime (10-family registry): every per-family runtime estimate elsewhere
in this file predates the 7-variant expansion and is now WRONG by roughly
an order of magnitude for two families specifically -- measured directly
on this machine against real M1 data (122 windows), a single fit at each
family's full-precision default takes: ~10-21s for Gamma/IG/Wiener/
truncated-Wiener/state-dependent-Gamma/state-dependent-IG/GIG, ~97s for
multi-stage Wiener, and ~250-335s for random-effects Gamma/IG (their
Gauss-Hermite quadrature multiplies the cost of every likelihood
evaluation the optimizer performs). Phase 1 alone (one fit per family per
machine, no repetition) is therefore roughly 15-20 min just for the two
random-effects families, tens of minutes total across all 10 -- a real but
bounded ONE-TIME cost. Phases 5/6 refit every family repeatedly (up to 3
folds, or across `--window-sizes-sweep`'s 8 default sizes, or
`--recovery-n-replicates` x true-families x machines) -- without
mitigation this would put those phases' random-effects/multi-stage
contribution in the multi-hour-to-day range, which is why
``degradation_process_models._REPEATED_REFIT_COARSENING`` coarsens both
families' most expensive knob (multi-stage Wiener's change-point grid,
random-effects' quadrature node count) specifically inside those repeated-
refit sweeps, never for the one-time primary fit above. Even coarsened,
expect Phases 5 and 6 to take substantially longer than the numbers
quoted in this file's other docstrings/help text (written for the
original 3-family registry) -- run Phase 1 alone first and use its
per-family timing to calibrate expectations for your own machine before
committing to a full Phase 5/6 run.

``artifacts/degradation_models/`` holds raw pipeline JSON (reproducibility
state, not paper-facing); ``paper_assets/`` holds only the curated,
camera-ready tables and figures meant for direct inclusion in the manuscript.

Phases 5 and 6 are each expensive enough to risk exceeding a single
foreground call's practical limit -- run them as separate calls, e.g.:
    uv run python run_degradation_study.py --phases 1 2
    uv run python run_degradation_study.py --phases 5
    uv run python run_degradation_study.py --phases 6
    uv run python run_degradation_study.py --phases 3 4

Run specific phases:
    uv run python run_degradation_study.py --phases 1
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACT = PROJECT_DIR / "artifacts" / "degradation_models"
PAPER_ASSETS = PROJECT_DIR / "paper_assets"


def _ensure_dirs():
    (ARTIFACT).mkdir(parents=True, exist_ok=True)
    for sub in ("tables/main", "tables/supplementary", "figures/main", "figures/supplementary"):
        (PAPER_ASSETS / sub).mkdir(parents=True, exist_ok=True)


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    return str(o)


def _atomic_json_dump(obj, path: Path) -> None:
    """Write ``obj`` as JSON to ``path`` atomically: serialize to a temp
    file in the SAME directory (so the final rename is on one filesystem,
    not a cross-device copy) and ``os.replace`` it into place. Phases 5/6
    each run for a long time (tens of minutes to hours) computing this
    file's own contents -- a plain ``open(path, "w")`` write is only
    reached at the very end, but if the process is interrupted mid-write
    (killed, crashed, power loss -- exactly what happened to the background
    runs that motivated this fix) a straight write leaves a truncated,
    unparseable JSON file in place of the last known-good artifact.
    ``os.replace`` is a single atomic filesystem operation on both Windows
    and POSIX: readers always see either the complete old file or the
    complete new one, never a partial write.
    """
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_default)
    os.replace(tmp_path, path)


def _checkpoint_dir() -> Path:
    d = ARTIFACT / "phase6_checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _checkpointed(name: str, compute_fn):
    """Load ``name``'s result from ``phase6_checkpoints/{name}.json`` if a
    checkpoint from an earlier, interrupted run already exists; otherwise
    call ``compute_fn()``, save its result as a checkpoint (atomically,
    same guarantee as ``_atomic_json_dump``), and return it either way.

    Phase 6's individual analyses each take anywhere from minutes to (for
    model_recovery_study alone) most of a day, and the whole phase's
    result is otherwise only written to disk in ONE piece at the very
    end -- an interruption (killed process, crash, power loss; this
    project has now lost two multi-hour Phase 6 runs to exactly that)
    partway through analysis 8 of 13 previously meant losing the FIRST 7
    analyses' results too, even though they were already correctly
    computed and just sitting in memory. Checkpointing each analysis the
    moment it completes means a resumed run only re-does whatever was
    genuinely in flight when the interruption happened, not everything
    before it.
    """
    path = _checkpoint_dir() / f"{name}.json"
    if path.is_file():
        print(f"  [checkpoint] {name}: found existing result, skipping recomputation")
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    result = compute_fn()
    _atomic_json_dump(result, path)
    return result


# ── Phase 1: fit & compare ──────────────────────────────────────────────────────

def phase1_fit_and_compare(args):
    print("\n" + "=" * 60)
    print("PHASE 1 - Degradation process-family comparison")
    print("=" * 60)

    import degradation_families as families
    from milling_dataset import find_dataset_dir, load_file_index
    from degradation_process_models import (
        fit_and_compare_degradation_models, save_degradation_comparison,
        age_model_criterion_sensitivity, variant_vs_baseline_comparison,
        state_aware_trajectory_resimulation,
    )

    ds_dir = find_dataset_dir(PROJECT_DIR)
    index = load_file_index(ds_dir, require_h5_files=False)

    gamma_path = PROJECT_DIR / "artifacts" / "gamma" / "gamma_process_params.json"
    if not gamma_path.is_file():
        raise FileNotFoundError(
            f"{gamma_path} not found. This study reuses the existing production "
            "Gamma fit rather than refitting it; run "
            "'python main.py --phases 1' once first (fast, <1 min) to produce it."
        )
    with open(gamma_path, "r", encoding="utf-8") as fh:
        gamma_params = json.load(fh)

    result = fit_and_compare_degradation_models(index, gamma_params, window_size=args.window_size)

    print("\nAge-model selection: AIC vs. BIC (free -- no refit, reuses fields above)")
    result["age_model_criterion_sensitivity"] = age_model_criterion_sensitivity(result)

    print("\nVariant vs. baseline nested-model tests (free -- no refit, reuses/derives from fields above)")
    result["variant_vs_baseline"] = variant_vs_baseline_comparison(result)

    print("\nState-aware trajectory re-simulation for the two wear-level-dependent "
          "families (free -- no refit, corrects the frozen-wear-level mean-curve "
          "approximation in the RMSE/terminal-bias check above)")
    result["state_aware_resimulation"] = state_aware_trajectory_resimulation(index, result)

    save_degradation_comparison(result, ARTIFACT)

    print("\nSummary (lower AIC is better):")
    for machine_key, c in sorted(result["comparison"].items()):
        family_aics = ", ".join(
            f"{families.FAMILY_REGISTRY[f]['label']}={c[f]['aic']:.1f}"
            for f in families.FAMILY_REGISTRY if f in c
        )
        print(
            f"  {machine_key}: best={c['best_aic_family']:16s}  {family_aics}  "
            f"delta(Gamma-IG)={c['delta_aic_gamma_minus_ig']:.2f}"
        )
    print("\nPhase 1 complete.")
    return result


# ── Phase 2: reliability translation ────────────────────────────────────────────

def phase2_reliability(args, comparison_result: dict):
    print("\n" + "=" * 60)
    print("PHASE 2 - Reliability translation")
    print("=" * 60)

    from milling_dataset import find_dataset_dir, load_file_index
    from degradation_reliability_translation import summarize_reliability_translation

    ds_dir = find_dataset_dir(PROJECT_DIR)
    index = load_file_index(ds_dir, require_h5_files=False)

    result = summarize_reliability_translation(
        index, comparison_result, n_sim=args.n_sim, n_boot=args.n_boot,
    )
    _atomic_json_dump(result, ARTIFACT / "reliability_translation.json")

    print("\nMTTFF summary (ground truth n=3 tools/machine; wide CIs expected):")
    for machine_key, row in sorted(result["summary"].items()):
        print(
            f"  {machine_key}: ground truth={row['ground_truth_mttff_s']:.0f}s "
            f"[{row['ground_truth_mttff_ci_lo']:.0f}, {row['ground_truth_mttff_ci_hi']:.0f}]  "
            f"Gamma-implied={row['gamma_implied_mttff_s']:.0f}s  "
            f"IG-implied={row['inverse_gaussian_implied_mttff_s']:.0f}s"
        )
    print("Phase 2 complete.")
    return result


# ── Phase 3: tables ──────────────────────────────────────────────────────────────

def phase3_tables(
    args, comparison_result: dict, reliability_result: dict,
    robustness_result: dict | None, extended_result: dict | None,
):
    print("\n" + "=" * 60)
    print("PHASE 3 - Tables")
    print("=" * 60)

    from degradation_tables import (
        save_degradation_comparison_table, save_reliability_translation_table,
        save_per_tool_heterogeneity_table, save_conditional_risk_table,
        save_cv_likelihood_table, save_window_size_sweep_table,
        save_eol_threshold_sensitivity_table, save_bootstrap_aic_gap_table,
        save_block_bootstrap_aic_gap_table,
        save_pooled_baseline_table, save_rul_comparison_table,
        save_maintenance_cost_table, save_cost_ratio_sensitivity_table,
        save_model_recovery_table, save_vuong_test_table, save_per_tool_diagnostics_table,
        save_multiple_comparisons_table, save_age_model_criterion_sensitivity_table,
        save_variant_vs_baseline_table, save_residual_autocorrelation_table,
        save_maintenance_cost_regret_bootstrap_table, save_moving_block_bootstrap_aic_gap_table,
        save_rf_feature_ablation_table, save_state_aware_resimulation_table,
    )

    main_dir = PAPER_ASSETS / "tables" / "main"
    supp_dir = PAPER_ASSETS / "tables" / "supplementary"

    significance_result = None
    if robustness_result is not None and "per_tool_significance" in robustness_result:
        significance_result = robustness_result["per_tool_significance"]
    else:
        print("  (per-tool significance not available -- Table 2 will omit those columns; rerun --phases 5)")

    save_degradation_comparison_table(comparison_result["comparison"], main_dir)
    if "age_model_criterion_sensitivity" in comparison_result:
        save_age_model_criterion_sensitivity_table(
            comparison_result["age_model_criterion_sensitivity"], supp_dir
        )
    else:
        print("  (age-model criterion sensitivity not available in this "
              "degradation_comparison.json -- rerun --phases 1 for Table S13)")
    if "variant_vs_baseline" in comparison_result:
        save_variant_vs_baseline_table(comparison_result["variant_vs_baseline"], supp_dir)
    else:
        print("  (variant-vs-baseline comparison not available in this "
              "degradation_comparison.json -- rerun --phases 1 for Table S14)")
    if "state_aware_resimulation" in comparison_result:
        save_state_aware_resimulation_table(comparison_result["state_aware_resimulation"], supp_dir)
    else:
        print("  (state-aware resimulation not available in this "
              "degradation_comparison.json -- rerun --phases 1 for Table S20)")
    save_per_tool_heterogeneity_table(comparison_result["comparison"], significance_result, main_dir)
    save_reliability_translation_table(reliability_result["summary"], main_dir)
    save_conditional_risk_table(reliability_result["conditional_failure_risk"], supp_dir)

    if robustness_result is not None:
        save_cv_likelihood_table(robustness_result["cv_likelihood"], supp_dir)
        save_window_size_sweep_table(robustness_result["window_size_sweep"], supp_dir)
        save_eol_threshold_sensitivity_table(robustness_result["eol_threshold_sensitivity"], supp_dir)
    else:
        print("  (robustness result not available -- run --phases 5 for Tables S1-S3)")
    if extended_result is not None:
        save_bootstrap_aic_gap_table(extended_result["bootstrap_aic_gap"], supp_dir)
        if "block_bootstrap_aic_gap_by_tool" in extended_result:
            save_block_bootstrap_aic_gap_table(extended_result["block_bootstrap_aic_gap_by_tool"], supp_dir)
        else:
            print("  (block bootstrap not available in this extended_rigor.json -- rerun --phases 6 for Table S16)")
        if "moving_block_bootstrap_aic_gap" in extended_result:
            save_moving_block_bootstrap_aic_gap_table(extended_result["moving_block_bootstrap_aic_gap"], supp_dir)
        else:
            print("  (moving-block bootstrap not available in this extended_rigor.json -- rerun --phases 6 for Table S18)")
        save_pooled_baseline_table(extended_result["pooled_baseline"], supp_dir)
        save_maintenance_cost_table(extended_result["maintenance_cost"], main_dir)
        if "cost_ratio_sensitivity" in extended_result:
            save_cost_ratio_sensitivity_table(extended_result["cost_ratio_sensitivity"], supp_dir)
        else:
            print("  (cost-ratio sensitivity not available in this extended_rigor.json -- rerun --phases 6)")
        if "maintenance_cost_regret_bootstrap" in extended_result:
            save_maintenance_cost_regret_bootstrap_table(
                extended_result["maintenance_cost_regret_bootstrap"], supp_dir,
            )
        else:
            print("  (maintenance-cost regret bootstrap not available in this "
                  "extended_rigor.json -- rerun --phases 6 for Table S17)")
        if "rul_threshold_decomposition" in extended_result:
            save_rul_comparison_table(extended_result["rul_threshold_decomposition"], supp_dir)
        else:
            print("  (RUL comparison not available in this extended_rigor.json -- rerun --phases 6)")
        ablation_path = PROJECT_DIR / "artifacts" / "rul" / "rf_feature_ablation.json"
        if ablation_path.is_file():
            with open(ablation_path, "r", encoding="utf-8") as fh:
                ablation_result = json.load(fh)
            save_rf_feature_ablation_table(ablation_result, supp_dir)
        else:
            print("  (RF feature ablation not available -- run 'python rf_feature_ablation.py' "
                  "first for Table S19)")
        if "model_recovery" in extended_result:
            save_model_recovery_table(extended_result["model_recovery"], supp_dir)
        else:
            print("  (model recovery not available in this extended_rigor.json -- rerun --phases 6)")
        if "vuong_test" in extended_result:
            save_vuong_test_table(extended_result["vuong_test"], supp_dir)
        else:
            print("  (Vuong test not available in this extended_rigor.json -- rerun --phases 6)")
        if "residual_autocorrelation" in extended_result:
            save_residual_autocorrelation_table(extended_result["residual_autocorrelation"], supp_dir)
        else:
            print("  (residual autocorrelation not available in this extended_rigor.json -- rerun --phases 6 for Table S15)")
        if "per_tool_diagnostics" in extended_result:
            save_per_tool_diagnostics_table(extended_result["per_tool_diagnostics"], supp_dir)
        else:
            print("  (per-tool diagnostics not available in this extended_rigor.json -- rerun --phases 6)")
        if "multiple_comparisons" in extended_result:
            save_multiple_comparisons_table(extended_result["multiple_comparisons"], supp_dir)
        else:
            print("  (multiple-comparisons summary not available in this extended_rigor.json -- rerun --phases 6)")
    else:
        print("  (extended-rigor result not available -- run --phases 6 for Table 4, S4-S6, S8-S12)")
    print(f"\nMain tables -> {main_dir}\nSupplementary tables -> {supp_dir}")
    print("Phase 3 complete.")


# ── Phase 4: figures ─────────────────────────────────────────────────────────────

def phase4_figures(args, comparison_result: dict, reliability_result: dict):
    print("\n" + "=" * 60)
    print("PHASE 4 - Figures")
    print("=" * 60)

    from milling_dataset import find_dataset_dir, load_file_index
    from degradation_plots import (
        plot_degradation_process_fits, plot_aic_gap, plot_conditional_failure_risk,
        plot_pit_diagnostics, plot_methodology_overview,
    )

    ds_dir = find_dataset_dir(PROJECT_DIR)
    index = load_file_index(ds_dir, require_h5_files=False)

    main_dir = PAPER_ASSETS / "figures" / "main"
    supp_dir = PAPER_ASSETS / "figures" / "supplementary"
    plot_methodology_overview(main_dir)
    plot_degradation_process_fits(index, comparison_result, main_dir)
    plot_aic_gap(comparison_result["comparison"], main_dir)
    plot_conditional_failure_risk(comparison_result, reliability_result, main_dir)
    plot_pit_diagnostics(index, comparison_result, supp_dir)
    print(f"\nMain figures -> {main_dir}\nSupplementary figures -> {supp_dir}")
    print("Phase 4 complete.")


# ── Phase 6: extended rigor ──────────────────────────────────────────────────────

def phase6_extended_rigor(args, comparison_result: dict):
    print("\n" + "=" * 60)
    print("PHASE 6 - Extended rigor (bootstrap, pooled baseline, RUL, cost)")
    print("=" * 60)

    import degradation_families as families
    from milling_dataset import find_dataset_dir, load_file_index
    from degradation_process_models import (
        bootstrap_family_aic_gap, block_bootstrap_family_aic_gap_by_tool,
        moving_block_bootstrap_family_aic_gap,
        pooled_vs_per_machine_comparison,
        model_recovery_study, vuong_test, per_tool_fit_diagnostics,
        multiple_comparisons_summary, residual_autocorrelation_check,
    )
    from degradation_reliability_translation import (
        ig_rul_leave_one_tool_out, compare_ig_vs_conformal_rul,
        maintenance_cost_comparison, cost_ratio_sensitivity,
        decompose_rul_threshold_confound, bootstrap_maintenance_cost_regret_by_tool,
    )

    ds_dir = find_dataset_dir(PROJECT_DIR)
    index = load_file_index(ds_dir, require_h5_files=False)

    print("\n-- Bootstrap CI on the Gamma-IG AIC gap (window-level resampling) --")
    bootstrap_result = _checkpointed("01_bootstrap_aic_gap", lambda: bootstrap_family_aic_gap(
        index, window_size=args.window_size, n_boot=args.bootstrap_n_boot,
    ))

    print("\n-- Block bootstrap CI on the Gamma-IG AIC gap (whole-tool resampling) --")
    block_bootstrap_result = _checkpointed("02_block_bootstrap_aic_gap", lambda: block_bootstrap_family_aic_gap_by_tool(
        index, window_size=args.window_size,
    ))

    print("\n-- Moving-block bootstrap CI on the Gamma-IG AIC gap (within-tool resampling) --")
    moving_block_bootstrap_result = _checkpointed(
        "03_moving_block_bootstrap_aic_gap",
        lambda: moving_block_bootstrap_family_aic_gap(index, window_size=args.window_size),
    )

    print("\n-- Pooled-across-machines baseline --")
    pooled_result = _checkpointed(
        "04_pooled_baseline",
        lambda: pooled_vs_per_machine_comparison(index, comparison_result, window_size=args.window_size),
    )

    print("\n-- IG-implied vs. conformal RF RUL comparison --")
    conformal_csv = PROJECT_DIR / "artifacts" / "rul" / "conformal_rul_cv_results.csv"
    if not conformal_csv.is_file():
        raise FileNotFoundError(
            f"{conformal_csv} not found. This comparison reuses the already-trained "
            "production conformal RUL model's CV results rather than retraining it -- "
            "run 'python main.py --phases 2' once first if this artifact is missing."
        )
    rul_comparison = _checkpointed("05_ig_vs_conformal_rul", lambda: compare_ig_vs_conformal_rul(
        ig_rul_leave_one_tool_out(index, window_size=args.window_size), conformal_csv,
    ))
    for tool_id, r in sorted(rul_comparison.items()):
        # ig_rmse_s/ig_rmse_ratio_vs_conformal can be None if conformal_csv was
        # produced against a different tool universe than `index` currently
        # has (e.g. a stale artifact after a dataset/config change) -- guard
        # the format strings rather than crash on a boundary mismatch.
        ig_rmse = r["ig_rmse_s"]
        ig_ratio = r["ig_rmse_ratio_vs_conformal"]
        print(
            f"  Tool {tool_id} (M{r['machine']}): conformal RMSE={r['conformal_rmse_s']:.1f}s  "
            f"IG-implied RMSE={'n/a' if ig_rmse is None else f'{ig_rmse:.1f}'}s  "
            f"ratio={'n/a' if ig_ratio is None else f'{ig_ratio:.2f}'}x"
        )

    print("\n-- RUL comparison threshold-mismatch decomposition --")
    rul_decomposition = _checkpointed(
        "06_rul_threshold_decomposition",
        lambda: decompose_rul_threshold_confound(index, conformal_csv, window_size=args.window_size),
    )
    for tool_id, r in sorted(rul_decomposition.items()):
        print(
            f"  Tool {tool_id} (M{r['machine']}): fixed-threshold ratio="
            f"{r['ig_fixed_threshold_ratio_vs_conformal']:.2f}x  "
            f"matched-threshold ({r['matched_threshold_um']:.0f}um) ratio="
            f"{r['ig_matched_threshold_ratio_vs_conformal']:.2f}x"
        )

    print("\n-- Maintenance-cost / age-replacement translation --")
    cost_result = _checkpointed("07_maintenance_cost", lambda: maintenance_cost_comparison(comparison_result))

    print("\n-- Cost-ratio sensitivity of the maintenance-cost regret ordering --")
    cost_ratio_result = _checkpointed(
        "08_cost_ratio_sensitivity", lambda: cost_ratio_sensitivity(comparison_result),
    )

    print("\n-- Block bootstrap CI on the maintenance-cost regret (whole-tool resampling) --")
    cost_regret_bootstrap_result = _checkpointed(
        "09_maintenance_cost_regret_bootstrap",
        lambda: bootstrap_maintenance_cost_regret_by_tool(index, window_size=args.window_size),
    )

    print("\n-- Model-recovery (positive-control) study --")
    # By FAR the most expensive analysis in Phase 6 (its own docstring:
    # roughly half a day for the full 10-true-family x 3-machine sweep at
    # n_replicates=6) and the one most likely to still be mid-flight if a
    # run gets interrupted -- checkpointed per TRUE family (10 slices, each
    # covering all 3 machines, ~1-1.5 hours) via model_recovery_study's own
    # only_true_families parameter, rather than as one all-or-nothing call,
    # so an interruption only re-does the one true-family slice that was
    # actually in progress. Each slice merged into the same
    # {machine: {true_family: {...}}} shape a single unrestricted call
    # would have produced.
    recovery_result = {}
    for true_family in families.registered_with_params(comparison_result):
        print(f"  -- model recovery: true={true_family} (all machines) --")
        slice_result = _checkpointed(
            f"10_model_recovery_{true_family}",
            lambda tf=true_family: model_recovery_study(
                index, comparison_result, window_size=args.window_size,
                n_replicates=args.recovery_n_replicates, only_true_families=[tf],
            ),
        )
        for machine_key, machine_slice in slice_result.items():
            recovery_result.setdefault(machine_key, {}).update(machine_slice)

    print("\n-- Vuong's non-nested hypothesis test (cluster-robust by tool) --")
    vuong_result = _checkpointed(
        "11_vuong_test", lambda: vuong_test(index, comparison_result, window_size=args.window_size),
    )

    print("\n-- PIT-residual autocorrelation by tool (independence-assumption check) --")
    autocorrelation_result = _checkpointed(
        "12_residual_autocorrelation",
        lambda: residual_autocorrelation_check(index, comparison_result, window_size=args.window_size),
    )

    print("\n-- Per-tool Gamma fit residual diagnostics --")
    def _per_tool_diagnostics():
        with open(PROJECT_DIR / "artifacts" / "gamma" / "gamma_process_params.json", "r", encoding="utf-8") as fh:
            gamma_params_raw = json.load(fh)
        return per_tool_fit_diagnostics(index, gamma_params_raw)
    diagnostics_result = _checkpointed("13_per_tool_diagnostics", _per_tool_diagnostics)

    print("\n-- Multiple-comparisons (BH-FDR) summary --")
    robustness_path = ARTIFACT / "robustness.json"
    if not robustness_path.is_file():
        raise FileNotFoundError(
            f"{robustness_path} not found. The multiple-comparisons summary needs "
            "the per-tool significance test's p-values -- run --phases 5 first."
        )
    with open(robustness_path, "r", encoding="utf-8") as fh:
        robustness_for_mc = json.load(fh)
    mc_summary = _checkpointed(
        "14_multiple_comparisons",
        lambda: multiple_comparisons_summary(vuong_result, robustness_for_mc["per_tool_significance"]),
    )

    extended = {
        "bootstrap_aic_gap": bootstrap_result,
        "block_bootstrap_aic_gap_by_tool": block_bootstrap_result,
        "moving_block_bootstrap_aic_gap": moving_block_bootstrap_result,
        "pooled_baseline": pooled_result,
        "ig_vs_conformal_rul": rul_comparison,
        "rul_threshold_decomposition": rul_decomposition,
        "maintenance_cost": cost_result,
        "cost_ratio_sensitivity": cost_ratio_result,
        "maintenance_cost_regret_bootstrap": cost_regret_bootstrap_result,
        "model_recovery": recovery_result,
        "vuong_test": vuong_result,
        "residual_autocorrelation": autocorrelation_result,
        "per_tool_diagnostics": diagnostics_result,
        "multiple_comparisons": mc_summary,
    }
    _atomic_json_dump(extended, ARTIFACT / "extended_rigor.json")
    print("\nPhase 6 complete.")
    return extended


# ── Phase 5: robustness checks ──────────────────────────────────────────────────

def phase5_robustness(args, comparison_result: dict):
    print("\n" + "=" * 60)
    print("PHASE 5 - Robustness checks")
    print("=" * 60)

    from milling_dataset import find_dataset_dir, load_file_index
    from degradation_process_models import (
        leave_one_tool_out_cv_likelihood, window_size_robustness_sweep,
        bootstrap_per_tool_power_significance,
    )
    from degradation_reliability_translation import summarize_reliability_translation

    ds_dir = find_dataset_dir(PROJECT_DIR)
    index = load_file_index(ds_dir, require_h5_files=False)

    print("\n-- Leave-one-tool-out out-of-sample predictive likelihood --")
    cv_result = leave_one_tool_out_cv_likelihood(index, window_size=args.window_size)

    print("\n-- Window-size robustness sweep --")
    sweep_result = window_size_robustness_sweep(index, args.window_sizes_sweep)

    print("\n-- EOL-threshold sensitivity --")
    sensitivity_result = {}
    for threshold in args.eol_thresholds:
        r = summarize_reliability_translation(
            index, comparison_result, n_sim=args.n_sim, n_boot=args.n_boot,
            wear_threshold_um=threshold,
        )
        sensitivity_result[str(threshold)] = r["summary"]
        for machine_key, row in sorted(r["summary"].items()):
            print(
                f"  threshold={threshold}  {machine_key}: "
                f"GT={row['ground_truth_mttff_s']:.0f}s  "
                f"Gamma={row['gamma_implied_mttff_s']:.0f}s  "
                f"IG={row['inverse_gaussian_implied_mttff_s']:.0f}s"
            )

    print("\n-- Per-tool outlier significance bootstrap (~10 min) --")
    significance_result = bootstrap_per_tool_power_significance(
        index, window_size=args.window_size, n_boot=args.tool_significance_n_boot,
    )
    for tool_id, a in sorted(significance_result["per_tool_absolute_vs_stationary"].items(), key=lambda kv: int(kv[0])):
        r = significance_result["per_tool_relative_vs_siblings"][tool_id]
        print(
            f"  Tool {tool_id} ({a['machine']}): power={a['median_power']:.3f} "
            f"[{a['ci_lo']:.3f},{a['ci_hi']:.3f}] vs.1.0 excl={a['excludes_stationary']}  "
            f"vs.siblings diff={r['median_diff']:+.3f} [{r['ci_lo']:.3f},{r['ci_hi']:.3f}] "
            f"excl0={r['excludes_zero']}"
        )
    print(f"  {significance_result['conclusion']}")

    robustness = {
        "cv_likelihood": cv_result,
        "window_size_sweep": sweep_result,
        "eol_threshold_sensitivity": sensitivity_result,
        "per_tool_significance": significance_result,
    }
    _atomic_json_dump(robustness, ARTIFACT / "robustness.json")
    print("\nPhase 5 complete.")
    return robustness


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(description="Degradation process-family comparison study")
    p.add_argument("--phases", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                    help="Phases to run (default: 1 2 3 4 5). Phase 5 (robustness "
                         "checks: CV likelihood, window-size sweep, EOL-threshold "
                         "sensitivity, per-tool significance) refits every registered "
                         "family repeatedly -- see this file's module docstring for "
                         "measured per-family costs and why Phase 5/6 now take "
                         "substantially longer than their pre-10-family-registry "
                         "estimates; omit it with --phases 1 2 3 4 for a much faster "
                         "check. Phase 6 (extended rigor) is opt-in, excluded from "
                         "the default.")
    p.add_argument("--window-size", type=int, default=15, dest="window_size")
    p.add_argument("--window-sizes-sweep", nargs="+", type=int,
                    default=[5, 8, 10, 12, 15, 20, 25, 30], dest="window_sizes_sweep",
                    help="Window sizes tested in Phase 5's robustness sweep")
    p.add_argument("--eol-thresholds", nargs="+", type=float,
                    default=[145.0, 150.0, 155.0, 160.0], dest="eol_thresholds",
                    help="EOL wear thresholds (um) tested in Phase 5's sensitivity sweep "
                         "-- default spans the dataset's documented observed terminal "
                         "wear range")
    p.add_argument("--n-sim", type=int, default=500, dest="n_sim",
                    help="Monte Carlo lifetimes per family per machine (Phase 2). "
                         "reliability_analysis.py's KM-curve bootstrap is O(n_sim^2) "
                         "per bootstrap draw (fine at its original design scale of "
                         "~20-30 episodes for the now-abandoned, deleted MARL "
                         "pipeline this function predates; left unmodified rather "
                         "than refactored, since it's reused read-only here too) -- "
                         "keep n_sim/n_boot modest rather than raising them without "
                         "checking runtime.")
    p.add_argument("--n-boot", type=int, default=1000, dest="n_boot")
    p.add_argument("--bootstrap-n-boot", type=int, default=50, dest="bootstrap_n_boot",
                    help="Bootstrap draws per machine for Phase 6's AIC-gap CI "
                         "(each draw refits Gamma+IG only, unaffected by the "
                         "10-family registry expansion -- measured ~10-13s/family, "
                         "~23s/draw combined on this machine's real data; 50 takes "
                         "roughly 20 min for all 3 machines combined, machine-speed "
                         "dependent)")
    p.add_argument("--tool-significance-n-boot", type=int, default=300,
                    dest="tool_significance_n_boot",
                    help="Bootstrap draws per tool for Phase 5's per-tool outlier "
                         "significance test (each draw refits a single-tool IG "
                         "model only, unaffected by the 10-family registry "
                         "expansion or its coarsening)")
    p.add_argument("--recovery-n-replicates", type=int, default=6, dest="recovery_n_replicates",
                    help="Synthetic replicates per (machine, true family) for Phase 6's "
                         "model-recovery positive control -- now refits all 10 "
                         "registered families per replicate (was 3), with "
                         "_REPEATED_REFIT_COARSENING applied to the two most "
                         "expensive. Default reduced from the original 3-family "
                         "study's 20 to 6 (see model_recovery_study's own "
                         "docstring): even coarsened, n_replicates=20 across the "
                         "10-family registry measures at an estimated 30-50 hours "
                         "total; 6 keeps full 10x10 family coverage at roughly half "
                         "a day, trading CI width for practicality. Raise this only "
                         "after checking the measured per-family costs in this "
                         "file's module docstring.")
    return p.parse_args()


def main():
    args = _parse()
    _ensure_dirs()
    phases = sorted(set(args.phases))

    t0 = time.time()
    comparison_result = None
    reliability_result = None
    robustness_result = None
    extended_result = None

    if 1 in phases:
        comparison_result = phase1_fit_and_compare(args)
    if any(p in phases for p in (2, 3, 4, 5, 6)) and comparison_result is None:
        # Later phases depend on phase 1's output; reload if run standalone.
        comparison_path = ARTIFACT / "degradation_comparison.json"
        if not comparison_path.is_file():
            raise FileNotFoundError(
                f"{comparison_path} not found; run --phases 1 first."
            )
        with open(comparison_path, "r", encoding="utf-8") as fh:
            comparison_result = json.load(fh)

    if 2 in phases:
        reliability_result = phase2_reliability(args, comparison_result)
    if any(p in phases for p in (3, 4)) and reliability_result is None:
        reliability_path = ARTIFACT / "reliability_translation.json"
        if not reliability_path.is_file():
            raise FileNotFoundError(
                f"{reliability_path} not found; run --phases 2 first."
            )
        with open(reliability_path, "r", encoding="utf-8") as fh:
            reliability_result = json.load(fh)

    if 5 in phases:
        robustness_result = phase5_robustness(args, comparison_result)
    if 3 in phases and robustness_result is None:
        robustness_path = ARTIFACT / "robustness.json"
        if robustness_path.is_file():
            with open(robustness_path, "r", encoding="utf-8") as fh:
                robustness_result = json.load(fh)
    if 6 in phases:
        extended_result = phase6_extended_rigor(args, comparison_result)
    if 3 in phases and extended_result is None:
        extended_path = ARTIFACT / "extended_rigor.json"
        if extended_path.is_file():
            with open(extended_path, "r", encoding="utf-8") as fh:
                extended_result = json.load(fh)
    if 3 in phases:
        phase3_tables(args, comparison_result, reliability_result, robustness_result, extended_result)
    if 4 in phases:
        phase4_figures(args, comparison_result, reliability_result)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"All requested phases complete. Total wall time: {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
