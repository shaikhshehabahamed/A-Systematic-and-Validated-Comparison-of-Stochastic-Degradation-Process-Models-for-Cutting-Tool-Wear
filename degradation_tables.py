"""Camera-ready table generation for the degradation process-family comparison.

Dedicated to this study alone -- the unrelated, now-abandoned MillingRESS MARL
scheduling submission had its own ``tables.py``, deleted in an earlier cleanup
along with the rest of the MARL-only pipeline. Tables are written as CSV and
LaTeX (booktabs), split into a lean main-text set and a supplementary set
carrying the extensive robustness analysis. Formatting conventions are held
constant across every table in
this module: "Yes"/"No" for booleans (not Python's True/False), "[lo, hi]"
for 95% CIs, 2 decimals for AIC/BIC, 3 for power/probability, 0 for seconds.

Table inventory
----------------
Main text:
  Table 1  Degradation process-family comparison (AIC/BIC/PIT-KS/age model)
  Table 2  Per-tool heterogeneity and formal outlier significance
  Table 3  Reliability translation (ground truth vs. process-implied MTTFF)
  Table 4  Maintenance-cost / age-replacement policy comparison
Supplementary:
  Table S1  Leave-one-tool-out out-of-sample predictive log-likelihood
  Table S2  Window-size robustness sweep
  Table S3  EOL-threshold sensitivity
  Table S4  Bootstrap CI on the Gamma-IG AIC gap
  Table S5  Pooled-across-machines baseline
  Table S6  Cost-ratio sensitivity of the maintenance-cost regret ordering
  Table S7  Conditional failure risk by family and reference age (numeric)
  Table S8  IG-implied vs. conformal RF RUL accuracy, with threshold decomposition
  Table S9  Model-recovery (positive-control) study
  Table S10 Vuong's non-nested hypothesis test
  Table S11 Per-tool Gamma fit residual diagnostics (M3 calibration investigation)
  Table S12 Multiple-comparisons (Benjamini-Hochberg FDR) summary
  Table S13 Age-model selection under AIC vs. BIC, per machine x family
  Table S14 6 of the 7 variants vs. their nested baseline (LR test; truncated Wiener excluded)
  Table S15 PIT-residual autocorrelation by tool and lag, per family
  Table S16 Block bootstrap (whole-tool resampling) CI on the Gamma-IG AIC gap
  Table S17 Block bootstrap CI on the maintenance-cost regret (parameter uncertainty)
  Table S18 Moving-block bootstrap (within-tool resampling) CI on the Gamma-IG AIC gap
  Table S19 Wear-only Random Forest feature ablation (see rf_feature_ablation.py)
  Table S20 State-aware trajectory re-simulation, wear-level-dependent Gamma/IG (see degradation_process_models.state_aware_trajectory_resimulation)
"""

from __future__ import annotations

import csv
from pathlib import Path

import degradation_families as families
# Imported for its side effect only (registers all 10 families, including
# the 3 originals registered inside degradation_process_models.py itself --
# degradation_families.py alone never populates FAMILY_REGISTRY). Needed
# because FAMILY_LABELS below is built once at import time, unlike this
# module's other FAMILY_REGISTRY reads (all inside function bodies, so they
# only need the registry populated by the time they're actually CALLED, not
# by the time this file itself is imported).
import degradation_process_models  # noqa: F401

FAMILY_LABELS = {key: spec["label"] for key, spec in families.FAMILY_REGISTRY.items()}


def _yn(value) -> str:
    return "Yes" if value else "No"


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


# Plain-pdflatex-safe substitutes for unicode used in CSV headers/row data
# (CSV keeps the readable unicode form; only .tex output needs this, since a
# standard Elsevier elsarticle pdflatex build has no guaranteed unicode-math
# or newunicodechar support). Applied only to raw data (headers/rows), never
# to `caption`/`label`, which are hand-authored LaTeX source already using
# the correct macros (\%, $c_{cm}$, \textmu, etc.) -- re-escaping those would
# double-escape and break compilation.
_LATEX_SPECIAL_CHARS = {
    "%": r"\%", "_": r"\_", "&": r"\&", "#": r"\#",
}
_LATEX_UNICODE_SAFE = {
    "µ": r"\textmu{}", "Δ": r"$\Delta$", "≥": r"$\geq$",
    "≤": r"$\leq$", "×": r"$\times$", "α": r"$\alpha$",
    "β": r"$\beta$", "η": r"$\eta$",
}


def _latex_safe_data(text: str) -> str:
    """Escape a raw (untrusted, non-LaTeX-aware) data string for safe use in
    a LaTeX table cell: special characters first, then unicode symbols."""
    for char, escaped in _LATEX_SPECIAL_CHARS.items():
        text = text.replace(char, escaped)
    for char, replacement in _LATEX_UNICODE_SAFE.items():
        text = text.replace(char, replacement)
    return text


def _write_latex(path: Path, header: list[str], rows: list[list], caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_cols = len(header)
    header = [_latex_safe_data(str(h)) for h in header]
    lines = [
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{'l' + 'r' * (n_cols - 1)}}}", r"\toprule",
        " & ".join(header) + r" \\", r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_safe_data(str(v)) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ── Main text ───────────────────────────────────────────────────────────────

def save_degradation_comparison_table(comparison: dict, out_dir: str | Path) -> None:
    """Table 1: AIC / BIC / PIT-KS / age-model selected, per family x machine.

    Covers every registered family present in ``comparison`` (originally
    hardcoded to Gamma/IG/Wiener) -- the headline "which family fits best"
    comparison the rest of the study is organized around, so the full
    10-family result belongs in the main-text table itself rather than only
    in supplementary material, per this study's exhaustive-analysis/
    compact-prose split (the analysis stays complete; only the accompanying
    prose is kept compact for the 7 new variants specifically).
    """
    out_dir = Path(out_dir)
    header = ["Machine", "Family", "Age model", "AIC", "BIC", "Mean PIT-KS stat",
              "Mean PIT-KS p", "Reject rate @ .05 (200 seeds)", "Calibrated"]
    rows = []
    for machine_key in sorted(comparison):
        c = comparison[machine_key]
        for family in families.FAMILY_REGISTRY:
            if family not in c:
                continue
            f = c[family]
            reject = f.get("pit_ks_frac_reject_05")
            rows.append([
                machine_key, FAMILY_LABELS[family], f["age_model"],
                f"{f['aic']:.2f}", f"{f['bic']:.2f}",
                f"{f['pit_ks_stat']:.3f}" if f.get("pit_ks_stat") is not None else "-",
                f"{f['pit_ks_pvalue']:.3f}" if f.get("pit_ks_pvalue") is not None else "-",
                f"{reject:.0%}" if reject is not None else "-",
                _yn(f.get("validation_passed")),
            ])
    _write_csv(out_dir / "table1_degradation_comparison.csv", header, rows)
    _write_latex(
        out_dir / "table1_degradation_comparison.tex", header, rows,
        caption="Degradation process-family comparison per machine (lower AIC/BIC is better). "
                "PIT-KS statistic and p-value are averaged over 200 independent randomized-"
                "quantile draws, with the fraction of those 200 draws individually rejecting "
                "calibration at $\\alpha=.05$ reported alongside; a single-draw p-value is "
                "unstable at this sample size, so the reject rate is the more defensible "
                "verdict. All families are validated under the identical protocol. "
                "Wiener fails calibration on every machine; Gamma also fails on M3 (57\\% "
                "reject rate) under this stricter multi-draw protocol, though it passes on "
                "M1/M2 -- Inverse Gaussian is the only family that calibrates robustly "
                "everywhere, in addition to having the lowest AIC/BIC on every machine. "
                "The `Age model' column is selected via AIC throughout this table; Gamma's "
                "non-homogeneous conclusion on M1/M2 specifically is a consequence of that "
                "choice and reverses to stationary under BIC (Table~S13) -- the family-"
                "ranking conclusion above (Inverse Gaussian lowest-AIC and lowest-BIC on "
                "every machine) is unaffected either way.",
        label="tab:degradation-comparison",
    )
    print(f"Saved Table 1 (degradation comparison) -> {out_dir}")


def save_per_tool_heterogeneity_table(comparison: dict, significance_result: dict | None, out_dir: str | Path) -> None:
    """Table 2: per-tool free (non-homogeneous) age exponent (Gamma vs. IG) merged
    with the formal bootstrap significance test of whether any tool's deviation
    from stationarity or from its machine-siblings is genuine rather than the
    expected maximum of three noisy draws.
    """
    out_dir = Path(out_dir)
    header = ["Machine", "Tool", "Gamma power (AIC-selected)", "IG power (free, pre-AIC)"]
    if significance_result is not None:
        header += ["IG power 95% CI (own bootstrap)", "Excludes 1.0",
                    "IG power vs. siblings (Δ)", "95% CI vs. siblings", "Excludes 0"]
    # Keys are plain ints when comparison_result/significance_result are passed
    # in-process (e.g. `--phases 1 2 3 4 5` in one call), but JSON stringifies
    # dict keys, so keys become numeric strings whenever either result was
    # reloaded from disk (e.g. `--phases 3` alone, or `--phases 5 3` without 1).
    # Normalizing both sides to int here keeps cross-dict lookups correct
    # regardless of which combination of results was freshly computed vs.
    # reloaded in a given invocation.
    abs_result = {
        int(k): v for k, v in
        (significance_result or {}).get("per_tool_absolute_vs_stationary", {}).items()
    }
    rel_result = {
        int(k): v for k, v in
        (significance_result or {}).get("per_tool_relative_vs_siblings", {}).items()
    }
    rows = []
    for machine_key in sorted(comparison):
        c = comparison[machine_key]
        gamma_power = {int(k): v for k, v in c["per_tool_power_gamma"].items()}
        ig_power = {int(k): v for k, v in c["per_tool_power_ig_nonhomogeneous"].items()}
        for tool_id in sorted(ig_power):
            row = [
                machine_key, tool_id,
                f"{gamma_power.get(tool_id, float('nan')):.3f}",
                f"{ig_power[tool_id]:.3f}",
            ]
            if significance_result is not None:
                a = abs_result.get(tool_id)
                r = rel_result.get(tool_id)
                row += [
                    f"[{a['ci_lo']:.3f}, {a['ci_hi']:.3f}]" if a else "n/a",
                    _yn(a["excludes_stationary"]) if a else "n/a",
                    f"{r['median_diff']:+.3f}" if r else "n/a",
                    f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]" if r else "n/a",
                    _yn(r["excludes_zero"]) if r else "n/a",
                ]
            rows.append(row)
    _write_csv(out_dir / "table2_per_tool_heterogeneity.csv", header, rows)
    _write_latex(
        out_dir / "table2_per_tool_heterogeneity.tex", header, rows,
        caption="Per-tool age exponent (power=1 is stationary) and formal outlier "
                "significance. Gamma's value reflects whichever model AIC selected at the "
                "machine level (uninformative, i.e. exactly 1.0, when the stationary model "
                "was selected machine-wide); IG's column is the free non-homogeneous "
                "candidate before machine-level AIC selection, surfacing per-tool deviation "
                "the aggregate decision can mask. The right-hand columns test this formally: "
                "each tool is bootstrap-refit in isolation (300 resamples of its own wear "
                "windows only), testing its power against stationarity (`Excludes 1.0') and "
                "against the mean of its two machine-siblings (`Excludes 0', the more direct "
                "outlier test). No tool excludes either at the 95\\% level -- point-estimate "
                "deviations (e.g. Tool 2 on M1, Tool 5 on M2) are consistent with sampling "
                "noise from only three tools per machine, not a demonstrated per-tool effect.",
        label="tab:per-tool-heterogeneity",
    )
    print(f"Saved Table 2 (per-tool heterogeneity + significance) -> {out_dir}")


def save_reliability_translation_table(summary: dict, out_dir: str | Path) -> None:
    """Table 3: ground-truth vs. process-implied MTTFF (with bootstrap CIs), per machine.

    Includes an explicit "observed failures" column: when a machine's ground
    truth is fully right-censored (0 observed failures among its 3 tools),
    the reported "MTTFF" is the restricted-mean-survival-time value evaluated
    at tau (the largest censored observation), which is a lower bound on the
    true mean rather than a full estimate -- a materially different quantity
    from a row where at least one tool's failure was actually observed, and
    one a reader should be able to see without cross-referencing the raw
    episode data themselves.

    Gamma/IG/GIG specifically (not all 8 families process_implied_reliability
    now covers): GIG is this study's single most robust upgrade over IG (see
    Table 1/S14), so belongs alongside the original pair here; the other 5
    (both state-dependent and random-effects variants) are covered in the
    underlying reliability_translation.json for anyone who wants them, but
    adding all of them as columns here would make the main table unreadable
    (see save_model_recovery_table's identical reasoning for why Table S9
    uses one summary column instead of one per family) -- Wiener is excluded
    from this table specifically because it has no Monte-Carlo-simulated
    MTTFF route at all (see process_implied_reliability's docstring), not a
    length-driven choice.
    """
    out_dir = Path(out_dir)
    header = ["Machine", "Ground truth: observed failures", "Ground truth MTTFF (s)", "95% CI",
              "Gamma-implied MTTFF (s)", "95% CI",
              "IG-implied MTTFF (s)", "95% CI",
              "GIG-implied MTTFF (s)", "95% CI"]
    rows = []
    for machine_key in sorted(summary):
        r = summary[machine_key]
        observed = f"{r['ground_truth_n_observed']}/{r['ground_truth_n_tools']}"
        if r.get("ground_truth_all_censored"):
            observed += " (fully censored)"
        row = [
            machine_key,
            observed,
            f"{r['ground_truth_mttff_s']:.0f}",
            f"[{r['ground_truth_mttff_ci_lo']:.0f}, {r['ground_truth_mttff_ci_hi']:.0f}]",
            f"{r['gamma_implied_mttff_s']:.0f}",
            f"[{r['gamma_implied_mttff_ci_lo']:.0f}, {r['gamma_implied_mttff_ci_hi']:.0f}]",
            f"{r['inverse_gaussian_implied_mttff_s']:.0f}",
            f"[{r['inverse_gaussian_implied_mttff_ci_lo']:.0f}, {r['inverse_gaussian_implied_mttff_ci_hi']:.0f}]",
        ]
        if "gig_implied_mttff_s" in r:
            row += [
                f"{r['gig_implied_mttff_s']:.0f}",
                f"[{r['gig_implied_mttff_ci_lo']:.0f}, {r['gig_implied_mttff_ci_hi']:.0f}]",
            ]
        else:
            row += ["-", "-"]
        rows.append(row)
    _write_csv(out_dir / "table3_reliability_translation.csv", header, rows)
    _write_latex(
        out_dir / "table3_reliability_translation.tex", header, rows,
        caption="Ground-truth (n=3 tools/machine; wide CIs reflect this) vs. process-implied "
                "restricted-mean time to first failure, for Gamma, Inverse Gaussian, and the "
                "Generalized Inverse Gaussian upgrade (Table 1/S14). A machine with 0 observed "
                "failures (fully right-censored) has its ground-truth MTTFF equal to the largest "
                "censored observation time -- a lower bound on the true mean, not a full "
                "estimate, unlike a row with at least one observed failure; the "
                "`observed failures' column flags this rather than leaving every row looking "
                "like the same kind of estimate. Process-implied MTTFF exceeds ground truth on "
                "every machine, consistent with this: the ground-truth RMST is truncated at "
                "each machine's observation horizon while the Monte Carlo simulation runs "
                "every trajectory to actual threshold-crossing, so some of the gap (all of it, "
                "for a fully-censored machine) is expected from the estimands' definitions "
                "rather than a modeling discrepancy. MTTFF point estimates otherwise agree "
                "closely across all three families despite their differing AIC/BIC ranking "
                "(Table~1); near-term conditional failure risk (Figure~3, Table~S7) agrees "
                "closely too. The practical stakes of family choice are concentrated in "
                "maintenance-policy timing (Table~4), not this point estimate.",
        label="tab:reliability-translation",
    )
    print(f"Saved Table 3 (reliability translation) -> {out_dir}")


def save_maintenance_cost_table(cost_result: dict, out_dir: str | Path) -> None:
    """Table 4: age-replacement optimal interval and regret, by family.

    Gamma/IG/Wiener/GIG specifically (maintenance_cost_comparison's
    underlying data now covers every families.standard_shape_families
    member, 7 of 10 -- see that function's docstring): GIG added as the
    single most robust upgrade over IG this study identifies (Table 1/S14);
    the two random-effects variants are covered in the underlying
    extended_rigor.json but omitted here as columns for the same
    readability reason Table 3/S9 give (sigma_p=0 exactly on every
    machine anyway, so they add no policy-relevant information beyond
    what Table S14 already reports).
    """
    out_dir = Path(out_dir)
    header = ["Machine", "Gamma T* (s)", "IG T* (s)", "Wiener T* (s)", "GIG T* (s)",
              "Regret using Gamma policy", "Regret using Wiener policy", "Regret using GIG policy"]
    rows = []
    for m in sorted(cost_result):
        r = cost_result[m]
        row = [
            m, f"{r['gamma']['T_star']:.0f}", f"{r['inverse_gaussian']['T_star']:.0f}",
            f"{r['wiener']['T_star']:.0f}",
        ]
        row.append(f"{r['gig']['T_star']:.0f}" if "gig" in r else "-")
        row.append(f"{r['regret_using_gamma_policy_pct']:+.1f}%")
        row.append(f"{r['regret_using_wiener_policy_pct']:+.1f}%")
        row.append(f"{r['regret_using_gig_policy_pct']:+.1f}%" if "regret_using_gig_policy_pct" in r else "-")
        rows.append(row)
    _write_csv(out_dir / "table4_maintenance_cost.csv", header, rows)
    _write_latex(
        out_dir / "table4_maintenance_cost.tex", header, rows,
        caption="Barlow-Proschan age-replacement optimal preventive-replacement age T* under "
                "each family's implied hazard, and the \\% cost-rate regret of using Gamma's, "
                "Wiener's, or GIG's optimal policy if Inverse Gaussian (this table's reference "
                "family, per the original 3-family study's convention) is true. Illustrative "
                "cost ratio $c_{cm}/c_{pm}=3$ (Table~S6 sweeps 2-10); report ratios/regret, not "
                "absolute currency (the dataset has no cost data). Regret from choosing between "
                "the statistically valid families (Gamma/IG/GIG) is consistently small; regret "
                "from the statistically invalid family (Wiener) is consistently larger.",
        label="tab:maintenance-cost",
    )
    print(f"Saved Table 4 (maintenance cost) -> {out_dir}")


# ── Supplementary ───────────────────────────────────────────────────────────

def save_cv_likelihood_table(cv_result: dict, out_dir: str | Path) -> None:
    """Table S1: leave-one-tool-out out-of-sample predictive log-likelihood, by family."""
    out_dir = Path(out_dir)
    header = ["Machine", "Family", "CV total log-lik", "CV mean log-lik/window", "Best"]
    rows = []
    for machine_key in sorted(cv_result):
        r = cv_result[machine_key]
        best = r["best_cv_family"]
        for family in families.FAMILY_REGISTRY:
            if family not in r["per_family"]:
                continue
            f = r["per_family"][family]
            rows.append([
                machine_key, FAMILY_LABELS[family],
                f"{f['cv_total_loglik']:.2f}", f"{f['cv_mean_loglik_per_window']:.4f}",
                _yn(family == best),
            ])
    _write_csv(out_dir / "tableS1_cv_likelihood.csv", header, rows)
    _write_latex(
        out_dir / "tableS1_cv_likelihood.tex", header, rows,
        caption="Leave-one-tool-out out-of-sample predictive log-likelihood (higher is better). "
                "Complements the in-sample AIC comparison (Table~1) with genuinely held-out "
                "evidence: each tool is scored under a model fit only on the other two tools "
                "of its machine, using the median (eta, power) across the trained tools as the "
                "prediction for the unseen tool. With only 3 tools per machine, each fold trains "
                "on just 2 -- removing one tool barely changes a 2-tool fit, so this evidence, "
                "while genuinely out-of-sample, is not strongly decorrelated from the in-sample "
                "comparison and should be read as a modest additional check, not independent "
                "confirmation at the strength Table~S9's model-recovery study provides.",
        label="tab:cv-likelihood",
    )
    print(f"Saved Table S1 (CV likelihood) -> {out_dir}")


def save_window_size_sweep_table(sweep_result: dict, out_dir: str | Path) -> None:
    """Table S2: window-size robustness sweep -- delta-AIC(Gamma-IG) by window_size,
    plus mean lag-1 PIT-residual autocorrelation (averaged across all fitted
    families and machines at that window_size) -- the direct evidence that
    the independence-assumption violation documented in Table S15 grows
    with aggregation and is small at fine window sizes, rather than an
    unavoidable property of the underlying process."""
    out_dir = Path(out_dir)
    header = [
        "Window size", "M1: best / Δ(Gamma-IG)", "M2: best / Δ(Gamma-IG)", "M3: best / Δ(Gamma-IG)",
        "Mean lag-1 PIT-ACF",
    ]
    rows = []
    for ws in sorted(sweep_result, key=int):
        ws_result = sweep_result[ws]
        row = [ws]
        acf_values = []
        for m in ("M1", "M2", "M3"):
            v = ws_result.get(m, {})
            if v.get("skipped"):
                row.append("skipped")
            else:
                row.append(f"{FAMILY_LABELS[v['best_aic_family']]} / {v['delta_aic_gamma_minus_ig']:+.1f}")
                acf_values.extend(x for x in v.get("mean_lag1_pit_acf", {}).values() if x == x)  # NaN check
        row.append(f"{sum(acf_values) / len(acf_values):.2f}" if acf_values else "n/a")
        rows.append(row)
    _write_csv(out_dir / "tableS2_window_size_sweep.csv", header, rows)
    _write_latex(
        out_dir / "tableS2_window_size_sweep.tex", header, rows,
        caption="Window-size robustness sweep: best-AIC family and $\\Delta$AIC(Gamma-IG) "
                "(positive favors IG) across candidate wear-window aggregation sizes, with "
                "mean lag-1 PIT-residual autocorrelation (averaged across all fitted families "
                "and machines) at each size. The production default (window\\_size=15) is not "
                "a cherry-picked value, and its non-trivial autocorrelation (Table S15) is "
                "predominantly a coarse-aggregation artifact: the same diagnostic is near zero "
                "at the finest window sizes here and grows systematically as windows coarsen.",
        label="tab:window-size-sweep",
    )
    print(f"Saved Table S2 (window-size sweep) -> {out_dir}")


def save_eol_threshold_sensitivity_table(sensitivity_result: dict, out_dir: str | Path) -> None:
    """Table S3: EOL-threshold sensitivity -- MTTFF (ground truth / Gamma / IG) by threshold."""
    out_dir = Path(out_dir)
    header = ["EOL threshold (µm)", "Machine", "Ground truth MTTFF (s)",
              "Gamma-implied MTTFF (s)", "IG-implied MTTFF (s)"]
    rows = []
    for threshold in sorted(sensitivity_result, key=float):
        for machine_key, row in sorted(sensitivity_result[threshold].items()):
            rows.append([
                threshold, machine_key,
                f"{row['ground_truth_mttff_s']:.0f}",
                f"{row['gamma_implied_mttff_s']:.0f}",
                f"{row['inverse_gaussian_implied_mttff_s']:.0f}",
            ])
    _write_csv(out_dir / "tableS3_eol_threshold_sensitivity.csv", header, rows)
    _write_latex(
        out_dir / "tableS3_eol_threshold_sensitivity.tex", header, rows,
        caption="EOL-threshold sensitivity, spanning the dataset's documented observed terminal "
                "wear range (145-160\\textmu m). Gamma- and IG-implied MTTFF stay close to each "
                "other at every threshold tested, consistent with the main finding that these "
                "families disagree on the age-model conclusion (Table~1) without disagreeing "
                "much on mean life.",
        label="tab:eol-threshold-sensitivity",
    )
    print(f"Saved Table S3 (EOL threshold sensitivity) -> {out_dir}")


def save_bootstrap_aic_gap_table(bootstrap_result: dict, out_dir: str | Path) -> None:
    """Table S4: bootstrap CI on the Gamma-IG AIC gap (window-level resampling)."""
    out_dir = Path(out_dir)
    header = ["Machine", "n boot", "Mean gap (Gamma-IG)", "Std", "95% CI", "Draws favoring IG"]
    rows = []
    for m in sorted(bootstrap_result):
        r = bootstrap_result[m]
        rows.append([
            m, r["n_boot"], f"{r['mean_gap']:.2f}", f"{r['std_gap']:.2f}",
            f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]", f"{r['frac_favoring_ig']:.0%}",
        ])
    _write_csv(out_dir / "tableS4_bootstrap_aic_gap.csv", header, rows)
    _write_latex(
        out_dir / "tableS4_bootstrap_aic_gap.tex", header, rows,
        caption="Bootstrap distribution of the AIC gap between Gamma and Inverse Gaussian "
                "(window-level resampling, 95\\% CI). Captures within-tool-composition sampling "
                "uncertainty; does not capture between-tool generalization uncertainty (see "
                "Table~S1 for leave-one-tool-out CV, and Table~S16 for a complementary "
                "between-tool block bootstrap).",
        label="tab:bootstrap-aic-gap",
    )
    print(f"Saved Table S4 (bootstrap AIC gap) -> {out_dir}")


def save_block_bootstrap_aic_gap_table(block_bootstrap_result: dict, out_dir: str | Path) -> None:
    """Table S16: block bootstrap CI on the Gamma-IG AIC gap, resampling
    WHOLE TOOLS (not individual windows) with replacement -- the between-
    tool-generalization complement to Table S4's within-tool-composition
    bootstrap (see degradation_process_models.block_bootstrap_family_aic_gap_by_tool).
    With exactly 3 tools/machine the resample is an exact 3**3=27-draw
    enumeration, not an approximate Monte Carlo sample."""
    out_dir = Path(out_dir)
    header = ["Machine", "n draws", "Exact?", "Mean gap (Gamma-IG)", "Std", "95% CI", "Draws favoring IG"]
    rows = []
    for m in sorted(block_bootstrap_result):
        r = block_bootstrap_result[m]
        rows.append([
            m, r["n_boot"], _yn(r["exact_enumeration"]), f"{r['mean_gap']:.2f}", f"{r['std_gap']:.2f}",
            f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]", f"{r['frac_favoring_ig']:.0%}",
        ])
    _write_csv(out_dir / "tableS16_block_bootstrap_aic_gap.csv", header, rows)
    _write_latex(
        out_dir / "tableS16_block_bootstrap_aic_gap.tex", header, rows,
        caption="Block bootstrap distribution of the AIC gap between Gamma and Inverse "
                "Gaussian, resampling whole tools (never individual windows) with "
                "replacement -- an exact enumeration of all 3\\textsuperscript{3}=27 possible "
                "draws given 3 tools/machine, not an approximate Monte Carlo sample. Wider "
                "than Table~S4 by construction (only 3 independent blocks, vs. up to "
                "$\\sim$150 individual windows), and reported honestly as such: the "
                "qualitative conclusion is checked for fragility to which specific tools "
                "were observed, not re-estimated with the same apparent precision as Table~S4.",
        label="tab:block-bootstrap-aic-gap",
    )
    print(f"Saved Table S16 (block bootstrap AIC gap by tool) -> {out_dir}")


def save_moving_block_bootstrap_aic_gap_table(moving_block_result: dict, out_dir: str | Path) -> None:
    """Table S18: moving-block bootstrap CI on the Gamma-IG AIC gap,
    resampling overlapping blocks of consecutive windows WITHIN each tool
    (not whole tools, and not individual windows) -- the third,
    complementary angle alongside Tables S4 and S16 (see
    degradation_process_models.moving_block_bootstrap_family_aic_gap)."""
    out_dir = Path(out_dir)
    header = ["Machine", "Block length", "n draws", "N tools used",
              "Mean gap (Gamma-IG)", "Std", "95% CI", "Draws favoring IG"]
    rows = []
    for m in sorted(moving_block_result):
        r = moving_block_result[m]
        rows.append([
            m, r["block_length"], r["n_boot"], r["n_tools_used"],
            f"{r['mean_gap']:.2f}", f"{r['std_gap']:.2f}",
            f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]", f"{r['frac_favoring_ig']:.0%}",
        ])
    _write_csv(out_dir / "tableS18_moving_block_bootstrap_aic_gap.csv", header, rows)
    _write_latex(
        out_dir / "tableS18_moving_block_bootstrap_aic_gap.tex", header, rows,
        caption="Moving-block bootstrap (Kunsch, 1989) distribution of the AIC gap between "
                "Gamma and Inverse Gaussian, resampling overlapping 8-window blocks "
                "independently within each tool's own sequence and pooling the three tools' "
                "resampled sequences for a joint refit -- preserving short-range within-tool "
                "serial structure, unlike Table~S4's plain i.i.d. window resampling, while "
                "still varying within-tool composition across replicates, unlike Table~S16's "
                "whole-tool blocks. A third, complementary check on the same point estimate.",
        label="tab:moving-block-bootstrap-aic-gap",
    )
    print(f"Saved Table S18 (moving-block bootstrap AIC gap) -> {out_dir}")


def save_pooled_baseline_table(pooled_result: dict, out_dir: str | Path) -> None:
    """Table S5: pooled-across-machines baseline vs. sum of per-machine fits.

    "Favors pooling" columns are computed directly from the sign of each
    delta (negative = pooling favored) rather than left for a reader to infer
    from the raw numbers -- a caption asserting the wrong direction here once
    went undetected because nothing in the table made the direction explicit.
    """
    out_dir = Path(out_dir)
    header = ["Family", "Pooled AIC", "Sum per-machine AIC", "Δ AIC", "AIC favors pooling",
              "Pooled BIC", "Sum per-machine BIC", "Δ BIC", "BIC favors pooling"]
    rows = []
    for family in families.FAMILY_REGISTRY:
        if family not in pooled_result:
            continue
        r = pooled_result[family]
        rows.append([
            FAMILY_LABELS[family], f"{r['pooled_aic']:.2f}", f"{r['sum_per_machine_aic']:.2f}",
            f"{r['delta_aic_pooled_minus_summed']:+.2f}",
            _yn(r['delta_aic_pooled_minus_summed'] < 0),
            f"{r['pooled_bic']:.2f}", f"{r['sum_per_machine_bic']:.2f}",
            f"{r['delta_bic_pooled_minus_summed']:+.2f}",
            _yn(r['delta_bic_pooled_minus_summed'] < 0),
        ])
    _write_csv(out_dir / "tableS5_pooled_baseline.csv", header, rows)
    _write_latex(
        out_dir / "tableS5_pooled_baseline.tex", header, rows,
        caption="Pooled-across-machines baseline: one shared dispersion parameter beta across "
                "all 9 tools (per-tool eta/power still free) vs. three separate per-machine "
                "betas. Negative delta favors pooling. Under AIC the comparison is mixed "
                "(Gamma: essentially a wash at +0.66; Inverse Gaussian/Wiener: mild preference "
                "for pooling). Under BIC's stricter complexity penalty, per-machine dispersion "
                "is preferred for every family instead ($\\Delta$BIC +3.4 to +24.4, strongest for "
                "Gamma) -- machine-level heterogeneity in this fleet is therefore not confined to "
                "per-tool eta/power alone: the shared dispersion parameter beta itself varies "
                "enough across machines that the stricter criterion prefers keeping it separate.",
        label="tab:pooled-baseline",
    )
    print(f"Saved Table S5 (pooled baseline) -> {out_dir}")


def save_cost_ratio_sensitivity_table(sensitivity_result: dict, out_dir: str | Path) -> None:
    """Table S6: cost-ratio sensitivity of the maintenance-cost regret ordering (Table 4)."""
    out_dir = Path(out_dir)
    header = ["c_cm/c_pm ratio", "Machine", "Regret using Gamma policy",
              "Regret using Wiener policy", "Wiener regret ≥ Gamma regret"]
    rows = []
    for ratio in sorted(sensitivity_result, key=float):
        for machine_key, r in sorted(sensitivity_result[ratio].items()):
            holds = r["regret_using_wiener_policy_pct"] >= r["regret_using_gamma_policy_pct"]
            rows.append([
                ratio, machine_key,
                f"{r['regret_using_gamma_policy_pct']:+.2f}%",
                f"{r['regret_using_wiener_policy_pct']:+.2f}%",
                _yn(holds),
            ])
    _write_csv(out_dir / "tableS6_cost_ratio_sensitivity.csv", header, rows)
    _write_latex(
        out_dir / "tableS6_cost_ratio_sensitivity.tex", header, rows,
        caption="Sensitivity of Table~4's regret ordering to the corrective/preventive cost "
                "ratio $c_{cm}/c_{pm}$, swept over a realistic range (2-10) rather than the "
                "single illustrative ratio of 3 used elsewhere. The Wiener-regret-exceeds-"
                "Gamma-regret ordering holds at every tested ratio, confirming it is not an "
                "artifact of the arbitrary ratio choice.",
        label="tab:cost-ratio-sensitivity",
    )
    print(f"Saved Table S6 (cost-ratio sensitivity) -> {out_dir}")


def save_maintenance_cost_regret_bootstrap_table(bootstrap_result: dict, out_dir: str | Path) -> None:
    """Table S17: block bootstrap (whole-tool resampling) 95% CI on the
    maintenance-cost regret from Table 4, propagating parameter estimation
    uncertainty (see
    degradation_reliability_translation.bootstrap_maintenance_cost_regret_by_tool)
    -- Table S6's complement, sweeping the fitted parameters via
    resampling rather than the illustrative cost ratio."""
    out_dir = Path(out_dir)
    header = [
        "Machine", "n draws", "Exact?",
        "Regret using Gamma policy (mean, 95% CI)",
        "Regret using Wiener policy (mean, 95% CI)",
    ]
    rows = []
    for m in sorted(bootstrap_result):
        r = bootstrap_result[m]
        g, w = r["gamma"], r["wiener"]
        rows.append([
            m, r["n_draws"], _yn(r["exact_enumeration"]),
            f"{g['mean_regret_pct']:+.2f}% [{g['ci_lo']:.2f}, {g['ci_hi']:.2f}]",
            f"{w['mean_regret_pct']:+.2f}% [{w['ci_lo']:.2f}, {w['ci_hi']:.2f}]",
        ])
    _write_csv(out_dir / "tableS17_maintenance_cost_regret_bootstrap.csv", header, rows)
    _write_latex(
        out_dir / "tableS17_maintenance_cost_regret_bootstrap.tex", header, rows,
        caption="Block bootstrap (whole-tool resampling, exact 3\\textsuperscript{3}=27-draw "
                "enumeration) 95\\% CI on Table~4's maintenance-cost regret, propagating "
                "parameter estimation uncertainty rather than reporting a single fixed-"
                "parameter point estimate. Complements Table~S6, which instead sweeps the "
                "illustrative cost ratio; together they address both sources of the "
                "\"illustrative cost ratios and fixed fitted parameters\" concern about the "
                "maintenance-cost translation.",
        label="tab:maintenance-cost-regret-bootstrap",
    )
    print(f"Saved Table S17 (maintenance-cost regret bootstrap) -> {out_dir}")


def save_conditional_risk_table(hazard: dict, out_dir: str | Path) -> None:
    """Table S7: conditional failure risk in a fixed lookahead window, by family and reference age."""
    out_dir = Path(out_dir)
    header = ["Machine", "Reference age (s)", "Lookahead (s)",
              "Gamma: survival", "Gamma: risk", "IG: survival", "IG: risk"]
    rows = []
    for machine_key in sorted(hazard):
        fams = hazard[machine_key]
        g_rows = fams["gamma"]
        ig_rows = fams["inverse_gaussian"]
        for g, ig in zip(g_rows, ig_rows, strict=True):
            rows.append([
                machine_key, f"{g['t_ref_s']:.0f}", f"{g['lookahead_s']:.0f}",
                f"{g['survival_at_t_ref']:.4f}", f"{g['conditional_failure_risk']:.4f}",
                f"{ig['survival_at_t_ref']:.4f}", f"{ig['conditional_failure_risk']:.4f}",
            ])
    _write_csv(out_dir / "tableS7_conditional_failure_risk.csv", header, rows)
    _write_latex(
        out_dir / "tableS7_conditional_failure_risk.tex", header, rows,
        caption="Conditional failure risk P(fail within lookahead | survived to reference age) "
                "implied by the Gamma vs. Inverse Gaussian fit, at 20/50/80\\% of median observed "
                "tool life. Figure~3 shows the smooth version of this same computation.",
        label="tab:conditional-risk",
    )
    print(f"Saved Table S7 (conditional failure risk) -> {out_dir}")


def save_rul_comparison_table(decomposition_result: dict, out_dir: str | Path) -> None:
    """Table S8: IG-implied (leave-one-tool-out) vs. conformal RF RUL prediction
    accuracy, together with the threshold-mismatch decomposition -- rescoring
    against each held-out tool's own actual observed terminal wear instead of
    the fixed 150um EOL threshold used elsewhere, to isolate the fixed-
    threshold-mismatch component from the feature-richness component.
    """
    out_dir = Path(out_dir)
    header = ["Tool", "Machine", "Conformal RF RMSE (s)",
              "IG RMSE, fixed 150µm (s)", "Ratio, fixed",
              "Matched threshold (µm)", "IG RMSE, matched threshold (s)", "Ratio, matched"]
    rows = []
    for tool_id in sorted(decomposition_result, key=int):
        r = decomposition_result[tool_id]
        r_fixed = r.get("ig_fixed_threshold_ratio_vs_conformal")
        r_matched = r.get("ig_matched_threshold_ratio_vs_conformal")
        rows.append([
            tool_id, r["machine"], f"{r['conformal_rmse_s']:.1f}",
            f"{r['ig_fixed_threshold_rmse_s']:.1f}" if r.get("ig_fixed_threshold_rmse_s") is not None else "-",
            f"{r_fixed:.2f}x" if r_fixed is not None else "-",
            f"{r['matched_threshold_um']:.0f}" if r.get("matched_threshold_um") is not None else "-",
            f"{r['ig_matched_threshold_rmse_s']:.1f}" if r.get("ig_matched_threshold_rmse_s") is not None else "-",
            f"{r_matched:.2f}x" if r_matched is not None else "-",
        ])
    _write_csv(out_dir / "tableS8_rul_comparison.csv", header, rows)
    _write_latex(
        out_dir / "tableS8_rul_comparison.tex", header, rows,
        caption="Point RUL prediction accuracy: IG-implied (virtual-age inversion, wear only, "
                "leave-one-tool-out) vs. the production conformal Random Forest (wear + signal "
                "features, same protocol). `Fixed' rescores IG against the dataset-wide "
                "150\\textmu m EOL threshold; `matched' rescores against each held-out tool's own "
                "actual observed terminal wear (145-160\\textmu m across tools), removing the "
                "threshold-mismatch confound. The two ratio columns are close for 8 of 9 tools "
                "($\\leq$7\\% relative change), confirming the RF's accuracy advantage there is "
                "genuinely feature-richness, not a threshold-scoring artifact. Tool 9 is the "
                "exception (3.23x fixed vs. 2.54x matched, $-$21.4\\%): it has the highest "
                "terminal wear of all 9 tools (160\\textmu m, the dataset maximum), i.e. it is "
                "furthest from the fixed 150\\textmu m assumption, so it shows the largest "
                "correction from removing that confound -- consistent with, not contradicting, "
                "the overall conclusion. Note the RF's feature set includes signal channels "
                "affected by a documented sensor aliasing defect (disabled anti-aliasing filter) "
                "on M2 tools 4-6 specifically -- this does not touch the wear/time-only "
                "degradation-process analysis (Tables 1-4) at all, but is a relevant caveat here "
                "since it could inflate or deflate the RF's accuracy on those three tools "
                "specifically relative to a defect-free sensor.",
        label="tab:rul-comparison",
    )
    print(f"Saved Table S8 (RUL comparison) -> {out_dir}")


def save_rf_feature_ablation_table(ablation_result: dict, out_dir: str | Path) -> None:
    """Table S19: wear-only Random Forest feature ablation (see
    rf_feature_ablation.py) -- directly tests how much of the RF's RUL
    accuracy advantage over IG (Table S8) comes from its richer sensor
    features versus its general nonparametric flexibility, by retraining
    the RF on ONLY the wear-trajectory information (VB_norm, its causal
    slope, machine identity) the stochastic models themselves see, via the
    identical leave-one-tool-out CV+ procedure."""
    out_dir = Path(out_dir)
    header = ["Tool", "Wear-only RF RMSE (s)", "Full-feature RF RMSE (s)",
              "Ratio, wear-only/full", "IG RMSE (s)", "Ratio, IG/wear-only RF"]
    rows = []
    for tool_id in sorted(ablation_result["comparison_by_tool"], key=int):
        r = ablation_result["comparison_by_tool"][tool_id]
        ratio_full = r.get("wear_only_rmse_ratio_vs_full")
        ratio_ig = r.get("ig_rmse_ratio_vs_wear_only_rf")
        rows.append([
            tool_id, f"{r['wear_only_rmse_s']:.1f}",
            f"{r['full_feature_rmse_s']:.1f}" if r.get("full_feature_rmse_s") is not None else "-",
            f"{ratio_full:.2f}x" if ratio_full is not None else "-",
            f"{r['ig_rmse_s']:.1f}" if r.get("ig_rmse_s") is not None else "-",
            f"{ratio_ig:.2f}x" if ratio_ig is not None else "-",
        ])
    _write_csv(out_dir / "tableS19_rf_feature_ablation.csv", header, rows)
    _write_latex(
        out_dir / "tableS19_rf_feature_ablation.tex", header, rows,
        caption=f"Wear-only Random Forest feature ablation: retraining on only "
                f"{ablation_result['n_wear_only_features']} of the production model's "
                f"{ablation_result['n_full_features']} features (dropping the 5 raw "
                "signal-derived statistics, keeping wear level, its causal slope, and machine "
                "identity -- the same information the stochastic degradation-process models "
                "use) via the identical leave-one-tool-out CV+ procedure as Table~S8. "
                "Wear-only RF accuracy is essentially unchanged from the full-feature RF (and "
                "slightly better on several tools), while both remain substantially more "
                "accurate than IG on every tool: the RF's advantage is therefore attributable "
                "mainly to its nonparametric flexibility, not to its richer sensor inputs.",
        label="tab:rf-feature-ablation",
    )
    print(f"Saved Table S19 (RF feature ablation) -> {out_dir}")


def save_model_recovery_table(recovery_result: dict, out_dir: str | Path) -> None:
    """Table S9: model-recovery (positive-control) study. For each machine and
    candidate true family, the fraction of synthetic replicates (simulated
    from that family's own real fitted parameters) where AIC correctly
    recovers the true generating family -- directly tests whether the
    family-selection procedure is unbiased, rather than assuming it.
    """
    out_dir = Path(out_dir)
    header = ["Machine", "True family", "Recovery rate", "95% CI", "Usable replicates",
              "Mean windows dropped", "Most confused with"]
    rows = []
    for machine_key in sorted(recovery_result):
        for true_family in families.FAMILY_REGISTRY:
            if true_family not in recovery_result[machine_key]:
                continue
            r = recovery_result[machine_key][true_family]
            counts = r["selected_counts"]
            rate = r["recovery_rate"]
            ci_lo, ci_hi = r.get("recovery_rate_ci_lo"), r.get("recovery_rate_ci_hi")
            # A single "confused with" column scales to any registry size,
            # unlike one "selected as X" column per family (10 families ->
            # 10 columns, unreadable); the full selected_counts breakdown is
            # still in the CSV's underlying JSON artifact for anyone who
            # wants it.
            confusions = {f: c for f, c in counts.items() if f != true_family and c > 0}
            confused_with = "-" if not confusions else FAMILY_LABELS[max(confusions, key=confusions.get)]
            rows.append([
                machine_key, FAMILY_LABELS[true_family],
                f"{rate:.0%}" if rate == rate else "-",  # NaN check
                f"[{ci_lo:.0%}, {ci_hi:.0%}]" if ci_lo is not None and ci_lo == ci_lo else "-",
                f"{r['n_replicates_success']}/{r['n_replicates_requested']}",
                f"{r.get('mean_frac_windows_dropped', 0.0):.1%}",
                confused_with,
            ])
    _write_csv(out_dir / "tableS9_model_recovery.csv", header, rows)
    _write_latex(
        out_dir / "tableS9_model_recovery.tex", header, rows,
        caption="Model-recovery (positive-control) study: synthetic wear windows are simulated "
                "from each machine's own real fitted parameters under a KNOWN true family, every "
                "registered family is refit exactly as to the real data, and the fraction of 20 "
                "replicates where AIC correctly selects the true family is reported with a "
                "Wilson score 95\\% CI (n=20 gives wide intervals; treat point estimates "
                "accordingly). Addresses whether the windowed-increment construction "
                "structurally favors any one family regardless of the true generating process. "
                "`Mean windows dropped' reports the fraction of simulated windows discarded to "
                "the same negative-increment filter real data would apply (non-subordinator "
                "families only; every subordinator family, including every state-dependent/"
                "random-effects/GIG variant, is non-negative by construction) -- this filtering "
                "means a non-subordinator family's own fit is scored on a left-truncated sample "
                "against an untruncated likelihood (matching how it is fit everywhere in this "
                "study, real data included), a real, quantifiable contributor to its recovery "
                "rate alongside genuine discriminability from other families, not purely the "
                "latter.",
        label="tab:model-recovery",
    )
    print(f"Saved Table S9 (model recovery) -> {out_dir}")


def save_vuong_test_table(vuong_result: dict, out_dir: str | Path) -> None:
    """Table S10: Vuong's (1989) AIC-corrected non-nested hypothesis test,
    a formal statistical complement to the informal Delta-AIC evidence-
    strength heuristics used elsewhere (Table 1, Figure 2).

    Scoped to the original 3-family core comparison specifically (of the up
    to 41 non-nested pairs ``vuong_test`` now computes across the full
    10-family registry -- see that function's docstring for why the 4
    genuinely nested variant-vs-baseline pairs, out of C(10,2)=45 total,
    are excluded from it entirely). Of the 7 variants, only 4
    (state-dependent Gamma/IG, multi-stage Wiener, GIG) reduce exactly to
    a baseline family at a null parameter value and belong in Table S14
    (``save_variant_vs_baseline_table``) instead, a proper likelihood-ratio
    test rather than Vuong's non-nested one; truncated Wiener and both
    random-effects families have no such baseline (see
    ``variant_vs_baseline_comparison``'s docstring) and so remain among
    Vuong's own non-nested pairs, not shown in this 3-family-scoped table.

    Reports BOTH the cluster-robust (by tool) z/p-value -- the primary,
    correct result, used for the "Favored" column -- and the naive i.i.d.
    z/p-value the original 3-family study reported, side by side, rather
    than silently replacing one with the other: the gap between them is
    itself the direct, quantified answer to whether the within-tool PIT-
    residual autocorrelation documented in Table S15 actually matters for
    inference (it does -- the cluster-robust critical value is ~4.30, not
    1.96, since there are only 3 independent tools/machine feeding this
    test; see vuong_test's docstring for the full derivation).
    """
    out_dir = Path(out_dir)
    header = [
        "Machine", "Comparison", "z (cluster-robust)", "p (cluster-robust)",
        "z (naive i.i.d.)", "p (naive i.i.d.)", "Favored (cluster-robust)",
    ]
    label_pairs = {
        "gamma_vs_inverse_gaussian": "Gamma vs. IG",
        "inverse_gaussian_vs_wiener": "IG vs. Wiener",
        "gamma_vs_wiener": "Gamma vs. Wiener",
    }
    rows = []
    for machine_key in sorted(vuong_result):
        for pair_key, label in label_pairs.items():
            r = vuong_result[machine_key][pair_key]
            favored = r["favored"]
            favored_label = FAMILY_LABELS.get(favored, favored)
            rows.append([
                machine_key, label, f"{r['z_statistic']:+.2f}", f"{r['p_value']:.4f}",
                f"{r['z_statistic_naive_iid']:+.2f}", f"{r['p_value_naive_iid']:.4f}",
                favored_label,
            ])
    _write_csv(out_dir / "tableS10_vuong_test.csv", header, rows)
    _write_latex(
        out_dir / "tableS10_vuong_test.tex", header, rows,
        caption="Vuong's (1989) AIC-corrected closeness test for non-nested model comparison, "
                "with variance clustered by tool (G=3 tools/machine; critical value "
                "$t_{0.975,2} \\approx 4.30$, not the naive-i.i.d. 1.96) to account for the "
                "within-tool PIT-residual autocorrelation documented in Table~S15 -- the naive "
                "i.i.d. z/p-value is also shown for direct comparison. The cluster-robust test "
                "is markedly more conservative, as expected with only 3 independent replicates "
                "per machine; conclusions that do not survive this correction are reported as "
                "inconclusive rather than retained at the naive test's nominal significance.",
        label="tab:vuong-test",
    )
    print(f"Saved Table S10 (Vuong test) -> {out_dir}")


def save_per_tool_diagnostics_table(diagnostics_result: dict, out_dir: str | Path) -> None:
    """Table S11: per-tool Gamma fit residual diagnostics, investigating why
    M3 fails calibration (Table 1) while M1/M2 pass.
    """
    out_dir = Path(out_dir)
    header = ["Machine", "Age model", "Tool", "eta", "RMSE (µm)",
              "Terminal bias (µm)", "Terminal wear (µm)"]
    rows = []
    for machine_key in sorted(diagnostics_result):
        m = diagnostics_result[machine_key]
        for tool_id in sorted(m["per_tool"]):
            t = m["per_tool"][tool_id]
            rows.append([
                machine_key, m["selected_age_model"], tool_id, f"{t['eta']:.2f}",
                f"{t['rmse_um']:.2f}", f"{t['terminal_bias_um']:+.2f}", f"{t['terminal_wear_um']:.0f}",
            ])
    _write_csv(out_dir / "tableS11_per_tool_diagnostics.csv", header, rows)
    _write_latex(
        out_dir / "tableS11_per_tool_diagnostics.tex", header, rows,
        caption="Per-tool Gamma-fit residual diagnostics, investigating M3's calibration "
                "failure (Table 1). Substantial per-tool terminal-bias heterogeneity is not "
                "unique to M3: M2 and M3 each have two tools with opposite-signed biases of "
                "comparable magnitude (roughly 10-15\\textmu m); M1 instead has one tool with "
                "a comparably large bias ($-$14.4\\textmu m) offset by two much smaller, "
                "same-signed ones rather than a matched opposite-signed pair -- a different "
                "shape of heterogeneity, but heterogeneity nonetheless. What "
                "differs is that M1/M2's Gamma fit selected the non-homogeneous (power-law) "
                "age model, giving 2 free parameters per tool to absorb it, while M3 selected "
                "the simpler stationary model (1 free parameter per tool). M3's calibration "
                "failure is best explained by this loss of flexibility from Gamma's own "
                "age-model selection, not by an M3-specific physical anomaly.",
        label="tab:per-tool-diagnostics",
    )
    print(f"Saved Table S11 (per-tool diagnostics) -> {out_dir}")


def save_multiple_comparisons_table(mc_summary: dict, out_dir: str | Path) -> None:
    """Table S12: Benjamini-Hochberg FDR correction applied within each
    natural family of simultaneous formal hypothesis tests conducted across
    this study (Vuong pairwise tests, per-tool absolute/relative
    significance tests), and whether any conclusion changes after
    correction.
    """
    out_dir = Path(out_dir)
    family_labels = {
        "vuong_pairwise": "Vuong pairwise family comparisons (9 tests: 3 machines x 3 pairs)",
        "per_tool_absolute": "Per-tool power vs. stationary (9 tests: one per tool)",
        "per_tool_relative": "Per-tool power vs. siblings (9 tests: one per tool)",
    }
    header = ["Test family", "n tests", "Significant (uncorrected, α=.05)",
              "Significant (BH-FDR corrected)", "Conclusion changed"]
    rows = []
    for family_key, label in family_labels.items():
        f = mc_summary[family_key]
        rows.append([
            label, f["n_tests"], f["n_significant_uncorrected"],
            f["n_significant_fdr_corrected"], _yn(f["conclusion_changed"]),
        ])
    _write_csv(out_dir / "tableS12_multiple_comparisons.csv", header, rows)
    _write_latex(
        out_dir / "tableS12_multiple_comparisons.tex", header, rows,
        caption="Benjamini-Hochberg false-discovery-rate correction applied within each "
                "natural family of simultaneous formal hypothesis tests conducted across this "
                "study (correcting within, not pooled across, families, since pooling "
                "conceptually different test types into one FDR pass is itself questionable "
                "practice). PIT-KS calibration decisions (Table~1) are not included here: each "
                "already reflects averaging over 200 independent draws specifically to avoid "
                "single-draw instability, a different remedy for a different problem than "
                "multiplicity across simultaneously-conducted tests. No conclusion in this "
                "study changes under correction: the Vuong family's already-inconclusive M1 "
                "Gamma-vs-IG result stays inconclusive and its significant pairs stay "
                "significant; the per-tool families already found nothing significant before "
                "correction, so FDR correction (strictly more conservative) cannot create a "
                "significant finding where none existed.",
        label="tab:multiple-comparisons",
    )
    print(f"Saved Table S12 (multiple comparisons) -> {out_dir}")


def save_age_model_criterion_sensitivity_table(sensitivity_result: dict, out_dir: str | Path) -> None:
    """Table S13: age-model selection under AIC vs. BIC, per machine x family.

    Makes explicit which age-model conclusions in Table 1 are robust to the
    AIC-vs-BIC choice and which are not, rather than leaving that question
    answerable only by manually re-deriving it from the raw stationary/
    non-homogeneous AIC and BIC values.
    """
    out_dir = Path(out_dir)
    header = ["Machine", "Family", "Age model (AIC)", "Age model (BIC)",
              "ΔAIC(stat-NH)", "ΔBIC(stat-NH)", "Criterion-dependent"]
    rows = []
    for machine_key in sorted(sensitivity_result):
        for family in families.FAMILY_REGISTRY:
            if family not in sensitivity_result[machine_key]:
                continue
            r = sensitivity_result[machine_key][family]
            rows.append([
                machine_key, FAMILY_LABELS[family],
                r["age_model_under_aic"], r["age_model_under_bic"],
                f"{r['aic_gap_stationary_minus_nonhomogeneous']:+.2f}",
                f"{r['bic_gap_stationary_minus_nonhomogeneous']:+.2f}",
                _yn(r["criterion_dependent"]),
            ])
    _write_csv(out_dir / "tableS13_age_model_criterion_sensitivity.csv", header, rows)
    _write_latex(
        out_dir / "tableS13_age_model_criterion_sensitivity.tex", header, rows,
        caption="Age-model selection (stationary/`linear' vs. non-homogeneous/`power\\_law') "
                "under AIC vs. BIC, using the identical $>2$ evidence threshold for both "
                "criteria (Table~1 always uses the AIC rule) -- not an arbitrary borrowing "
                "from the AIC side: Kass \\& Raftery's (1995) BIC evidence-strength "
                "convention independently uses the same $>2$ cutoff between negligible and "
                "positive evidence. Gamma's non-homogeneous "
                "(`aging') conclusion on M1 and M2 -- the two machines behind the "
                "`Gamma finds spurious aging that Inverse Gaussian doesn't' contrast in "
                "Table~1 -- is specific to AIC: BIC's stricter penalty selects the stationary "
                "model for Gamma on both. Inverse Gaussian's and Wiener's age-model "
                "conclusions are unchanged under BIC on every machine. This does not affect "
                "the family-ranking conclusion (Inverse Gaussian remains the lowest-AIC and "
                "lowest-BIC family on every machine regardless, Table~1) -- only the age-model "
                "contrast specifically should be read as AIC-dependent for Gamma on 2 of 3 "
                "machines.",
        label="tab:age-model-criterion-sensitivity",
    )
    print(f"Saved Table S13 (age-model criterion sensitivity) -> {out_dir}")


def save_variant_vs_baseline_table(variant_result: dict, out_dir: str | Path) -> None:
    """Table S14: 6 of the 7 variants vs. the baseline (or, for the two
    random-effects families, pooled-exponent-restricted) model each nests
    inside at a null parameter value, via a proper likelihood-ratio test
    (see ``degradation_process_models.variant_vs_baseline_comparison`` and
    ``degradation_families.nested_model_test``) rather than Vuong's
    non-nested test (Table S10, which deliberately excludes these pairs).
    Truncated Wiener is the one variant with no nested nullpoint (left-
    truncation is a structural change, not a restriction of an existing
    parameter) and so has no row here -- see ``variant_vs_baseline_comparison``'s
    docstring, and Table S10 (where it IS compared, non-nested, against
    Gamma/IG/Wiener directly). The single compact table directly answering
    "is each new variant justified over its parent family" for the 6 it
    applies to.
    """
    out_dir = Path(out_dir)
    header = ["Machine", "Variant", "Baseline", "ΔAIC (baseline-variant)",
              "ΔBIC (baseline-variant)", "LR p-value", "Supported (AIC>2)", "Supported (LR, α=.05)"]
    rows = []
    for machine_key in sorted(variant_result):
        for variant_key, r in variant_result[machine_key].items():
            # Random-effects Gamma/IG's "baseline" is a descriptive string
            # (their own pooled-exponent restriction, not a literal
            # registered family -- see variant_vs_baseline_comparison's
            # docstring), not a FAMILY_LABELS key; fall back to the raw
            # string for those instead of KeyError-ing.
            baseline_label = FAMILY_LABELS.get(r["baseline"], r["baseline"])
            rows.append([
                machine_key, FAMILY_LABELS[variant_key], baseline_label,
                f"{r['delta_aic_baseline_minus_variant']:+.2f}",
                f"{r['delta_bic_baseline_minus_variant']:+.2f}",
                f"{r['lr_p_value']:.4f}" + ("*" if r.get("lr_boundary_corrected") else ""),
                _yn(r["variant_supported_aic"]), _yn(r["variant_supported_lr_05"]),
            ])
    _write_csv(out_dir / "tableS14_variant_vs_baseline.csv", header, rows)
    _write_latex(
        out_dir / "tableS14_variant_vs_baseline.tex", header, rows,
        caption="Six of the seven degradation-process variants tested against the baseline "
                "(or, for the two random-effects families, pooled-exponent-restricted) model "
                "each reduces to at its new parameter(s)' null value, via a likelihood-"
                "ratio test (positive $\\Delta$AIC/$\\Delta$BIC and a small LR p-value both favor "
                "the variant). State-dependent Gamma/IG test gamma\\_state=0; random-effects "
                "Gamma/IG test the aging-exponent variance component sigma\\_p=0 at its zero "
                "boundary (p-values marked * use the Self \\& Liang (1987) boundary correction, "
                "half the naive chi-square(1) p-value); multi-stage Wiener tests eta1=eta2 "
                "(subject to the Davies (1987) caveat that the change-point time is unidentified "
                "under this null, so the reported p-value is indicative rather than exact); GIG "
                "tests p\\_gid=-1/2, the point at which it reduces exactly to Inverse Gaussian. "
                "Truncated Wiener has no nested baseline (left-truncation is a structural change, "
                "not a parameter restriction) and is compared against Gamma/IG/Wiener directly "
                "via Vuong's test (Table S10) instead of appearing here.",
        label="tab:variant-vs-baseline",
    )
    print(f"Saved Table S14 (variant vs. baseline) -> {out_dir}")


def save_state_aware_resimulation_table(resim_result: dict, out_dir: str | Path) -> None:
    """Table S20: the corrected, state-aware trajectory-RMSE/terminal-bias
    re-simulation for the two wear-level-dependent variants (see
    ``degradation_process_models.state_aware_trajectory_resimulation``),
    against the naive (frozen-wear-level) values Table 4 itself reports.
    "State-aware calibrated" recomputes Section 4.5's pass/fail rule with
    the state-aware RMSE/bias in place of the naive ones and every other
    condition (PIT-KS, age-model) held at its already-computed value, since
    those do not depend on this specific approximation.
    """
    out_dir = Path(out_dir)
    header = [
        "Machine", "Family", "Naive RMSE (µm)", "State-aware RMSE (µm)", "RMSE gap closed",
        "Naive terminal bias (µm)", "State-aware terminal bias (µm)", "Bias gap closed",
        "State-aware calibrated",
    ]
    rows = []
    for family_key in sorted(resim_result):
        for machine_key, r in sorted(resim_result[family_key].items()):
            rows.append([
                machine_key, FAMILY_LABELS[family_key],
                f"{r['naive_trajectory_rmse_um']:.2f}", f"{r['state_aware_trajectory_rmse_um']:.2f}",
                f"{r['rmse_gap_closed_frac']:+.1%}",
                f"{r['naive_terminal_bias_um']:+.2f}", f"{r['state_aware_terminal_bias_um']:+.2f}",
                f"{r['bias_gap_closed_frac']:+.1%}",
                _yn(r["state_aware_validation_passed"]),
            ])
    _write_csv(out_dir / "tableS20_state_aware_resimulation.csv", header, rows)
    _write_latex(
        out_dir / "tableS20_state_aware_resimulation.tex", header, rows,
        caption="Corrected, state-aware trajectory re-simulation for the two wear-level-"
                "dependent variants, against the naive (frozen-wear-level) values Table 4 "
                "itself reports. The naive mean curve (used for every family in Table 4) "
                "holds each tool's wear level fixed at its first recorded reading for the "
                "whole trajectory; the state-aware version instead recurses the mean curve "
                "forward at the same window granularity the model was fit at, feeding each "
                "window's own predicted wear level into the next window's wear-level-"
                "dependent shape quantity, exactly as the fitted feedback term intends. This "
                "recursion is the exact conditional-mean trajectory under the fitted model, "
                "not a Monte Carlo approximation of it, since both families' shape quantity "
                "is affine in wear level and neither family's draws are clipped. \"State-"
                "aware calibrated\" reapplies Section 4.5's five-condition rule with this "
                "corrected RMSE/bias in place of the naive values and every other condition "
                "unchanged.",
        label="tab:state-aware-resimulation",
    )
    print(f"Saved Table S20 (state-aware resimulation) -> {out_dir}")


def _mean_ignoring_nan(values: list[float]) -> float:
    finite = [v for v in values if v == v]  # NaN check without a numpy/math import
    return sum(finite) / len(finite) if finite else float("nan")


def save_residual_autocorrelation_table(autocorrelation_result: dict, out_dir: str | Path) -> None:
    """Table S15: PIT-residual autocorrelation at lags 1..max_lag, averaged
    across each machine's tools, for every fitted family -- the direct
    evidence on the rounded-observation likelihood's within-tool
    independence assumption (see
    degradation_process_models.residual_autocorrelation_check). Row order
    follows FAMILY_REGISTRY's own insertion order (Gamma/IG/Wiener first,
    then the 7 variants), matching every other family-indexed table in
    this module."""
    out_dir = Path(out_dir)
    max_lag = next(iter(autocorrelation_result.values()))["max_lag"]
    header = ["Family", "Machine", "N tools"] + [f"Lag-{k}" for k in range(1, max_lag + 1)]
    rows = []
    for family_key in families.FAMILY_REGISTRY:
        if family_key not in autocorrelation_result:
            continue
        r = autocorrelation_result[family_key]
        label = FAMILY_LABELS.get(family_key, family_key)
        for machine_key in sorted(r["per_machine"]):
            mr = r["per_machine"][machine_key]
            acf_lists = list(mr["acf_by_tool"].values())
            if not acf_lists:
                rows.append([label, machine_key, 0] + ["n/a"] * max_lag)
                continue
            mean_by_lag = [_mean_ignoring_nan([acf[k] for acf in acf_lists]) for k in range(max_lag)]
            rows.append([label, machine_key, mr["n_tools"]] + [f"{v:.3f}" for v in mean_by_lag])
    _write_csv(out_dir / "tableS15_residual_autocorrelation.csv", header, rows)
    _write_latex(
        out_dir / "tableS15_residual_autocorrelation.tex", header, rows,
        caption="Mean PIT-residual autocorrelation by lag, averaged across each machine's "
                "tools, for every fitted family (50 independent de-quantization draws per "
                "tool). Lag-1 values of roughly 0.5-0.85 at the production window size "
                "indicate the rounded-observation likelihood's within-tool independence "
                "assumption is violated there; Table~S2 shows this is substantially a "
                "temporal-aggregation artifact (near zero at finer window sizes), and Table~"
                "S10's Vuong test and Table~S16's block bootstrap correct for it via "
                "cluster-robust and block-resampling inference respectively rather than "
                "assuming it away.",
        label="tab:residual-autocorrelation",
    )
    print(f"Saved Table S15 (residual autocorrelation) -> {out_dir}")
