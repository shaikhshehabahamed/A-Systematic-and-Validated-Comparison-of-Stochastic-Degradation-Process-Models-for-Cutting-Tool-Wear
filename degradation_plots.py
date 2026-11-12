"""Camera-ready figure generation for the degradation process-family comparison.

Dedicated to this study alone -- the unrelated, now-abandoned MillingRESS MARL
scheduling submission had its own ``plots.py``, deleted in an earlier cleanup
along with the rest of the MARL-only pipeline. Figures saved at 300 DPI as PDF
(primary, vector) + PNG (preview). All three compared families share one
consistent, colorblind-safe encoding (Okabe-Ito palette, redundant with
linestyle/marker so figures survive grayscale printing) across every figure
in this module.

Figure inventory
-----------------
Main text:
  Figure 1  Observed wear trajectories vs. Gamma/IG/Wiener-implied mean curves
            (kept to the original 3 families deliberately -- overlaying all
            10 fitted mean curves on one set of axes would stop being
            legible well before reaching that count)
  Figure 2  AIC gap relative to the best-fitting family, per machine (all
            registered families present in the comparison result)
  Figure 3  Conditional failure risk (Gamma vs. IG), smooth curve by reference age
Supplementary:
  Figure S1  PIT histogram + QQ diagnostic plots, family x machine grid (all
             registered families present in the comparison result)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import degradation_families as families
# Imported for its side effect only (registers all 10 families -- see the
# identical note in degradation_tables.py, which hits the same eager
# module-level-dict-at-import-time issue since FAMILY_STYLE below is built
# once at import, not lazily).
import degradation_process_models  # noqa: F401
from gamma_process import build_wear_windows

DPI = 300

# Okabe-Ito colorblind-safe palette (Gamma/IG/Wiener) extended by each
# variant's own registry entry, which deliberately reuses its baseline
# family's color with a distinct linestyle/marker (see
# degradation_families.py's register() docstring) so a reader can still
# group by family at a glance even across all 10.
FAMILY_STYLE = {
    key: {"color": spec["color"], "ls": spec["ls"], "marker": spec["marker"], "label": spec["label"]}
    for key, spec in families.FAMILY_REGISTRY.items()
}

# Bar-chart-specific: color alone is NOT enough to distinguish families once
# every variant reuses its baseline's color (linestyle, which disambiguates
# them in every LINE-based figure here, has no bar-chart equivalent) -- a
# 10-family bar chart colored this way renders 3-4 bars per color group as
# visually identical, an unusable-in-practice bug caught only by actually
# looking at a real render, not by the code running without error. Hatch
# pattern is the bar-chart analogue of linestyle: derived systematically
# (position within its own color group, in registry order) rather than
# hand-assigned, so it stays correct if the registry grows.
_BAR_HATCH_PATTERNS = ("", "///", "xxx", "...", "\\\\")
FAMILY_BAR_HATCH: dict[str, str] = {}
_color_seen: dict[str, int] = {}
for _key, _spec in families.FAMILY_REGISTRY.items():
    _idx = _color_seen.get(_spec["color"], 0)
    FAMILY_BAR_HATCH[_key] = _BAR_HATCH_PATTERNS[_idx % len(_BAR_HATCH_PATTERNS)]
    _color_seen[_spec["color"]] = _idx + 1

MACHINES = ["M1", "M2", "M3"]
PANEL_TAGS = ["(a)", "(b)", "(c)"]


def _setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "lines.linewidth": 1.8,
        "axes.linewidth": 0.8,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
    })


_setup_style()


def _panel_tag(ax: plt.Axes, tag: str) -> None:
    ax.text(-0.06, 1.08, tag, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")


def _save(fig: plt.Figure, path: Path, stem: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(path / f"{stem}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _mean_wear_curve(params_for_machine: dict, t_grid: np.ndarray) -> np.ndarray:
    """Ensemble-averaged mean wear function implied by a fitted process,
    using each family's AIC-selected age model (matching Table 1).

    Hardcodes ``shape_s / beta`` -- correct ONLY for families whose mean is
    linear in shape_s (every family except truncated Wiener; see
    ``degradation_families.standard_mean_increment`` and
    ``degradation_process_variants._truncated_wiener_mean_increment``).
    This is safe today because ``plot_degradation_process_fits`` (Figure 1)
    is deliberately scoped to gamma/inverse_gaussian/wiener/gig only (GIG's
    mean is pinned at shape_s/beta by construction for every tail-shape
    value, so it is exactly as safe here as the three original families) --
    do NOT extend this function to truncated Wiener without also switching
    it to
    ``spec["mean_increment_fn"]`` (and, since that family's mean is
    NONLINEAR in shape_s, accumulating at the fitted window_size's
    granularity rather than evaluating shape_s(0, t) in one shot -- see
    ``degradation_process_models._pit_ks_generic`` for why a naive
    application of this formula to that family manufactured a >100um
    spurious bias on real data before this was understood)."""
    components = params_for_machine["components"]
    beta = float(params_for_machine["beta"])
    t_ref = float(params_for_machine.get("time_reference_s", 1000.0))
    curves = np.zeros((len(components), len(t_grid)))
    for ci, comp in enumerate(components):
        eta = float(comp["eta"]); power = float(comp["power"])
        shape_s = eta * np.power(np.maximum(t_grid, 1e-9) / t_ref, power)
        curves[ci] = shape_s / beta
    return np.mean(curves, axis=0)


def plot_degradation_process_fits(index: pd.DataFrame, comparison_result: dict, out_dir: str | Path) -> None:
    """Figure 1: observed wear trajectories with Gamma/IG/Wiener/GIG-implied mean curves.

    On M1/M2, the straight Inverse Gaussian line vs. the curved Gamma line is
    not a rendering artifact -- it is the paper's central finding rendered
    visually: IG selects a stationary (linear-in-time) age model where Gamma
    selects a non-homogeneous power-law one for the same data. This does
    NOT hold on M3: Gamma's own selected age model there is ALSO linear
    (power=1.0 for all 3 tools, verified against the real fitted
    parameters), so its curve sits within <1% of IG's at the EOL threshold
    -- visually indistinguishable, not a third curved line. Only Wiener is
    visibly curved on M3. A caption or legend claim of "IG straight vs.
    Gamma/Wiener curved" must be scoped to the M1/M2 panels specifically; a
    reader looking at panel (c) alone and given the unscoped version of that
    claim would be misled about what they are looking at.

    Caption note for the manuscript: this is the single most visually
    prominent artifact asserting that specific curved-vs-straight contrast,
    but on M1/M2 the contrast is itself AIC-specific -- Gamma's curve
    reverses to a straight (stationary) one under BIC (Table~S13). The
    figure necessarily renders the AIC-selected fit (the production
    convention used throughout Table~1), so a reader looking only at this
    figure should not come away thinking the curved-vs-straight distinction
    is criterion-independent; the manuscript's actual figure caption needs
    to carry this caveat, not just this docstring.
    """
    out_dir = Path(out_dir)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3), sharey=True)
    handles, labels = None, None
    for pi, (ax, machine_key) in enumerate(zip(axes, MACHINES, strict=True)):
        machine_id = int(machine_key[1:])
        sub = index[index["machine"] == machine_id]
        t_max = 0.0
        for tool, traj in sub.groupby("tool"):
            traj = traj.sort_values("run")
            t = traj["cumulated_tool_contact_time"].to_numpy(float)
            w = traj["wear"].to_numpy(float)
            t_max = max(t_max, float(t[-1]))
            ax.plot(t, w, color="0.65", lw=0.9, alpha=0.8, zorder=1)
        t_grid = np.linspace(1.0, t_max * 1.05, 200)
        for family, key in (
            ("gamma", "gamma_params"), ("inverse_gaussian", "ig_params"), ("wiener", "wiener_params"),
            ("gig", "gig_params"),
        ):
            if key not in comparison_result:
                continue  # older, pre-GIG comparison_result reloaded from disk
            st = FAMILY_STYLE[family]
            curve = _mean_wear_curve(comparison_result[key][machine_key], t_grid)
            ax.plot(t_grid, curve, color=st["color"], ls=st["ls"], lw=2.2, label=st["label"], zorder=2)
        ax.axhline(150.0, color="0.3", ls=(0, (1, 1.5)), lw=1.0, zorder=1)
        ax.text(0.02, 150.0 + 0.025 * ax.get_ylim()[1], "EOL threshold", color="0.3",
                 fontsize=8.5, transform=ax.get_yaxis_transform(), ha="left", va="bottom")
        ax.set_title(machine_key)
        ax.set_xlabel("Cumulated contact time (s)")
        _panel_tag(ax, PANEL_TAGS[pi])
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    axes[0].set_ylabel("Flank wear VB (µm)")
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.10), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out_dir, "figure1_degradation_process_fits")
    print(f"Saved Figure 1 (degradation process fits) -> {out_dir}")


def plot_aic_gap(comparison: dict, out_dir: str | Path) -> None:
    """Figure 2: AIC gap relative to each machine's best-fitting family.

    Raw AIC magnitude is not itself interpretable (it scales with sample
    size and carries no absolute meaning); what matters is the *gap* between
    families, which is what the paper's evidence actually rests on. Plotting
    the gap directly -- rather than raw AIC on a from-zero axis, which visually
    buries a decisive 22-25 point gap against magnitudes near 500-800 -- also
    lets the conventional Burnham & Anderson (2004) evidence-strength
    thresholds (Delta AIC > 2 substantial, > 10 strong) be shown directly.

    The best-fitting family for a given machine has, by construction, a gap
    of exactly 0 relative to itself -- an invisible, zero-height bar. On the
    real data this is Inverse Gaussian on every machine, but the annotation
    below is data-driven (keys off ``v == 0`` exactly, not a hardcoded family
    name) so it stays correct if that ever isn't true for all three
    machines. Without it, the legend would list a family with no visible
    bar, which reads as missing data rather than the intended zero-
    reference convention.

    Generalized to the full family registry (originally hardcoded to
    Gamma/IG/Wiener, 3 fixed-width bars per machine): bar width and
    per-cluster centering both scale to however many families are actually
    present in ``comparison`` (defensively intersected with the registry,
    so an older 3-family-only comparison_result reloaded from disk still
    renders correctly rather than KeyError-ing).
    """
    out_dir = Path(out_dir)
    machines = sorted(comparison)
    family_keys = [f for f in families.FAMILY_REGISTRY if f in comparison[machines[0]]]
    n_fam = len(family_keys)
    x = np.arange(len(machines))
    width = 0.8 / n_fam
    fig, ax = plt.subplots(figsize=(max(7.5, 0.9 * n_fam), 5.2))
    for i, family in enumerate(family_keys):
        st = FAMILY_STYLE[family]
        vals = []
        for m in machines:
            c = comparison[m]
            best_aic = min(c[f]["aic"] for f in family_keys)
            vals.append(c[family]["aic"] - best_aic)
        bars = ax.bar(
            x + (i - (n_fam - 1) / 2) * width, vals, width, label=st["label"], color=st["color"],
            hatch=FAMILY_BAR_HATCH[family], edgecolor="black", linewidth=0.5,
        )
        for rect, v in zip(bars, vals):
            if v > 0.5:
                ax.text(rect.get_x() + rect.get_width() / 2, v + 0.6, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=7, rotation=90 if n_fam > 3 else 0)
            elif np.isclose(v, 0.0, atol=1e-6):
                ax.text(rect.get_x() + rect.get_width() / 2, 0.4, "(ref.)",
                        ha="center", va="bottom", fontsize=7, color="0.4", rotation=90)
    ax.axhline(2.0, color="0.4", ls="--", lw=0.9, zorder=0)
    ax.text(x[-1] + width * n_fam * 0.55 + 0.05, 2.0, "substantial\n(ΔAIC > 2)", fontsize=8, color="0.4", va="center")
    ax.axhline(10.0, color="0.2", ls="--", lw=0.9, zorder=0)
    ax.text(x[-1] + width * n_fam * 0.55 + 0.05, 10.0, "strong\n(ΔAIC > 10)", fontsize=8, color="0.2", va="center")
    ax.set_xticks(x); ax.set_xticklabels(machines)
    ax.set_ylabel(r"$\Delta$AIC relative to best-fitting family")
    ax.set_xlim(-0.6, len(machines) - 1 + 0.6 + width * n_fam * 0.6)
    ax.legend(loc="upper center", ncol=min(n_fam, 5), bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, "figure2_aic_gap")
    print(f"Saved Figure 2 (AIC gap) -> {out_dir}")


def plot_conditional_failure_risk(comparison_result: dict, reliability_result: dict, out_dir: str | Path) -> None:
    """Figure 3: conditional failure risk (Gamma vs. IG), smooth curve by reference age.

    Uses the closed-form implied-survival function on a fine (60-point) age
    grid per machine (5-95% of that machine's ground-truth median life),
    rather than the 3 illustrative ages (20/50/80%) reported in Table S7 --
    those remain the numeric backing; this figure is the smooth visual.
    """
    from degradation_reliability_translation import conditional_failure_risk

    out_dir = Path(out_dir)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3), sharey=False)
    handles, labels = None, None
    for pi, (ax, machine_key) in enumerate(zip(axes, MACHINES, strict=True)):
        episodes = reliability_result["ground_truth"][machine_key]["episodes"]
        median_life = float(np.median([e["simTime"] for e in episodes]))
        reference_ages = np.linspace(0.05, 0.95, 60) * median_life
        risk = conditional_failure_risk(
            comparison_result, reference_ages_s=list(reference_ages),
            lookahead_s=0.1 * median_life,
        )[machine_key]
        for family in ("gamma", "inverse_gaussian"):
            st = FAMILY_STYLE[family]
            rows = risk[family]
            t_ref = [r["t_ref_s"] for r in rows]
            p = [r["conditional_failure_risk"] for r in rows]
            ax.plot(t_ref, p, color=st["color"], ls=st["ls"], lw=2.0, label=st["label"])
        ax.set_title(machine_key)
        ax.set_xlabel("Reference age (s)")
        _panel_tag(ax, PANEL_TAGS[pi])
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    axes[0].set_ylabel("P(fail within 10% life | survived)")
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out_dir, "figure3_conditional_failure_risk")
    print(f"Saved Figure 3 (conditional failure risk) -> {out_dir}")


def _pit_values(data: dict, params_for_machine: dict, family: str, rng_seed: int = 0) -> np.ndarray:
    """Single-draw randomized PIT residuals for a fitted family (visualization only;
    the numeric calibration verdict elsewhere averages over 200 seeds -- one
    representative draw is sufficient for a diagnostic plot).

    Generalized to the full family registry (originally a hardcoded if/elif
    over Gamma/IG/Wiener's inline CDF formulas): shape via ``spec["shape_fn"]``
    (passing wear level at each window's start, needed by state-dependent
    variants and ignored otherwise) and the CDF via ``spec["cdf_fn"]`` (plus
    ``extra_shape_values`` for GIG's tail-shape parameter), matching
    ``degradation_process_models._pit_ks_generic``'s dispatch but keeping
    this function's own simpler flat +/-half-resolution bin (no rounding-
    offset marginalization) -- a deliberate, lighter-weight approximation
    appropriate for a single-draw visual diagnostic, not the numeric
    calibration verdict itself.
    """
    spec = families.FAMILY_REGISTRY[family]
    component_by_tool = {int(c["tool_id"]): c for c in params_for_machine["components"]}
    beta = float(params_for_machine["beta"])
    extra = families.extra_shape_values(spec, params_for_machine)
    shapes = np.empty(len(data["dv"]), float)
    for i, tool in enumerate(data["tool"]):
        c = component_by_tool[int(tool)]
        shapes[i:i + 1] = spec["shape_fn"](
            c, data["t_start"][i:i + 1], data["t_end"][i:i + 1], data["v_start"][i:i + 1],
        )
    half = 0.5 * float(params_for_machine["wear_resolution_um"])
    floor = 1e-9 if spec["is_subordinator"] else -np.inf
    lower = np.maximum(floor, data["dv"] - half)
    upper = data["dv"] + half
    cdf_lo = spec["cdf_fn"](lower, shapes, beta, *extra)
    cdf_hi = spec["cdf_fn"](upper, shapes, beta, *extra)
    rng = np.random.default_rng(rng_seed)
    pit = cdf_lo + rng.random(len(cdf_lo)) * (cdf_hi - cdf_lo)
    return np.clip(pit, 1e-12, 1.0 - 1e-12)


def plot_pit_diagnostics(index: pd.DataFrame, comparison_result: dict, out_dir: str | Path) -> None:
    """Figure S1: PIT histogram (should look ~uniform if well-calibrated) + QQ
    plot (sorted PIT vs. uniform quantiles) per family x machine.

    Generalized to the full family registry (originally hardcoded to
    Gamma/IG/Wiener). The QQ plot is restructured from 3 panels (one per
    machine, with all families overlaid as separate lines) to a family x
    machine GRID matching the histogram's own layout: overlaying up to 10
    families' QQ lines on one set of axes, as the original 3-family design
    did, stops being legible well before N=10 -- a grid of small multiples
    scales to any registry size the same way the histogram already does.
    """
    out_dir = Path(out_dir)
    # Derive the actual fitted window_size (mirrors gamma_process.validate_gamma_fit
    # and degradation_process_models._pit_ks_generic) rather than assuming the
    # production default, so this diagnostic stays correct if the pipeline is
    # ever run with `--window-size` != 15. int(...) guards against
    # gamma_process.save_gamma_params coercing window_size to float when
    # comparison_result was reloaded from degradation_comparison.json (as
    # happens whenever Phase 4 runs without Phase 1 in the same invocation);
    # build_wear_windows' range() call requires a real int.
    window_size = int(next(iter(comparison_result["gamma_params"].values())).get("window_size", 15))
    windows = build_wear_windows(index, window_size=window_size)
    family_keys = [f for f in families.FAMILY_REGISTRY if families.param_key(f) in comparison_result]
    n_fam = len(family_keys)

    fig_qq, axes_qq = plt.subplots(n_fam, 3, figsize=(9.5, 2.55 * n_fam), sharex=True, sharey=True)
    for fi, family in enumerate(family_keys):
        st = FAMILY_STYLE[family]
        for mi, machine_key in enumerate(MACHINES):
            ax_qq = axes_qq[fi, mi]
            pit = _pit_values(windows[machine_key], comparison_result[families.param_key(family)][machine_key], family)
            sorted_pit = np.sort(pit)
            uniform_q = (np.arange(1, len(sorted_pit) + 1) - 0.5) / len(sorted_pit)
            ax_qq.plot(uniform_q, sorted_pit, color=st["color"], ls=st["ls"], lw=1.6)
            ax_qq.plot([0, 1], [0, 1], color="0.4", ls="-", lw=0.8, zorder=0)
            if fi == 0:
                ax_qq.set_title(machine_key)
            if mi == 0:
                ax_qq.set_ylabel(st["label"], fontsize=8.5)
            if fi == n_fam - 1:
                ax_qq.set_xlabel("Uniform quantile")
    fig_qq.tight_layout()
    _save(fig_qq, out_dir, "figureS1b_pit_qq")

    fig, axes = plt.subplots(n_fam, 3, figsize=(10.5, 2.55 * n_fam), sharex=True, sharey=True)
    for fi, family in enumerate(family_keys):
        st = FAMILY_STYLE[family]
        for mi, machine_key in enumerate(MACHINES):
            ax = axes[fi, mi]
            pit = _pit_values(windows[machine_key], comparison_result[families.param_key(family)][machine_key], family)
            ax.hist(pit, bins=10, range=(0, 1), color=st["color"], edgecolor="white", linewidth=0.6)
            ax.axhline(len(pit) / 10, color="0.2", ls="--", lw=0.8)
            if fi == 0:
                ax.set_title(machine_key)
            if mi == 0:
                ax.set_ylabel(st["label"], fontsize=8.5)
            if fi == n_fam - 1:
                ax.set_xlabel("PIT residual")
    fig.tight_layout()
    _save(fig, out_dir, "figureS1a_pit_histogram")
    print(f"Saved Figure S1 (PIT diagnostics) -> {out_dir}")


def _mo_rounded_box(ax, cx, cy, w, h, *, fill, edge, lw=1.1, ls="solid", z=2, radius=0.045):
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill, edgecolor=edge, linewidth=lw, linestyle=ls, zorder=z, mutation_aspect=1,
    )
    ax.add_patch(box)
    return box


def _mo_parallelogram(ax, cx, cy, w, h, *, fill, edge, lw=1.1, z=2, skew=0.22):
    from matplotlib.patches import Polygon
    pts = [
        (cx - w / 2 + skew, cy + h / 2), (cx + w / 2 + skew, cy + h / 2),
        (cx + w / 2 - skew, cy - h / 2), (cx - w / 2 - skew, cy - h / 2),
    ]
    ax.add_patch(Polygon(pts, closed=True, facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z))


def _mo_diamond(ax, cx, cy, w, h, *, fill, edge, lw=1.1, z=2):
    from matplotlib.patches import Polygon
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z))


def _mo_badge(ax, cx, cy, text, *, navy, r=0.15):
    from matplotlib.patches import Circle
    ax.add_patch(Circle((cx, cy), r, facecolor=navy, edgecolor="white", linewidth=1.2, zorder=5))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=7.6, fontweight="bold", color="white", zorder=6)


def _mo_arrow(ax, xy_from, xy_to, *, color, lw=1.3, style="solid", z=1):
    ax.annotate(
        "", xy=xy_to, xytext=xy_from,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=style, shrinkA=0, shrinkB=0),
        zorder=z,
    )


def _mo_text_block(ax, cx, top_y, title, detail, *, title_size, detail_size, title_color="#0D1526"):
    """Top-anchored title + detail text so blocks never collide regardless
    of box height or line count -- each box's own height is sized to fit
    what's placed inside it, rather than text position being guessed from
    the box height (the bug in the first version of this figure)."""
    ax.text(cx, top_y, title, ha="center", va="top", fontsize=title_size, fontweight="bold",
            color=title_color, zorder=4)
    if detail:
        title_h = title_size / 72.0 * 1.25
        detail_top = top_y - title_h - 0.10
        ax.text(cx, detail_top, detail, ha="center", va="top", fontsize=detail_size,
                color="#222222", linespacing=1.55, zorder=4)


def plot_methodology_overview(out_dir: str | Path) -> None:
    """Figure 2: portrait overview of the modeling and validation pipeline
    (Section 4.1) -- a static, data-independent diagram (unlike every other
    figure in this module, it takes no fitted-result argument) since it
    depicts the PROCEDURE, not a result. Regenerate manually if Section 4's
    stage boundaries or family/check counts ever change; nothing in the
    pipeline calls this automatically on every run the way the data-driven
    figures are, since its content would not change from one real-data rerun
    to the next.

    Redesigned (was a bare stack of 5 uniform gray boxes) to match the shape
    vocabulary of the original 3-family manuscript's own flowchart: input/
    output as parallelograms, process steps as rounded rectangles, the age-
    dependence rule as a decision diamond, section numbers as small badges,
    dashed arrows for supporting/validation branches vs. solid for the
    primary flow, and a legend -- extended here to the ten-family registry
    and laid out in portrait (the original was landscape) so it reads as one
    tall column. The main spine stays at one fixed x-center throughout
    (narrower boxes make room for the two side annexes rather than the
    spine itself jogging sideways) so every primary-flow arrow is a clean
    vertical, not a cramped diagonal.
    """
    out_dir = Path(out_dir)
    NAVY = "#1B2A4A"
    STEP_FILL = "#E9EDF4"
    IO_FILL = "#D7DEEA"
    CAND_FILL = "#FFFFFF"
    SIDE_FILL = "#F1F0EA"
    GRAY = "0.35"

    W, H = 8.6, 13.2
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.set_aspect("equal")

    SPINE_X = 3.75
    SPINE_W = 5.6
    GAP = 0.20

    def step(y_top, title, badge, detail, *, shape="rect", fill=STEP_FILL, w=SPINE_W, cx=SPINE_X,
              title_size=10.3, detail_size=7.9, pad_top=0.18, pad_bot=0.20):
        n_lines = detail.count("\n") + 1 if detail else 0
        title_h = title_size / 72.0 * 1.25
        detail_h = n_lines * (detail_size / 72.0 * 1.55) if detail else 0
        h = pad_top + title_h + (0.10 + detail_h if detail else 0) + pad_bot
        cy = y_top - h / 2
        if shape == "parallelogram":
            _mo_parallelogram(ax, cx, cy, w, h, fill=fill, edge=NAVY)
        else:
            _mo_rounded_box(ax, cx, cy, w, h, fill=fill, edge=NAVY)
        if badge:
            _mo_badge(ax, cx - w / 2 + 0.30, y_top - 0.30, badge, navy=NAVY)
        _mo_text_block(ax, cx, y_top - pad_top, title, detail, title_size=title_size, detail_size=detail_size)
        return y_top - h, y_top, h

    y = H - 0.15

    # ---- 1. Dataset (I/O parallelogram) -----------------------------------
    bot, top, h = step(
        y, "Milling Flank-Wear Dataset", "§3",
        "3 machines × 3 tools (n = 9)  ·  560–928 runs/tool\nVB & cumulative contact time t per run — Denkena, Klemme & Stiehl (2023)",
        shape="parallelogram", fill=IO_FILL, title_size=13, detail_size=8.6, pad_top=0.30, pad_bot=0.28,
    )
    y = bot - GAP
    _mo_arrow(ax, (SPINE_X, bot), (SPINE_X, y), color=NAVY, lw=1.3)

    # ---- 2. Wear aggregation ------------------------------------------------
    bot, top, h = step(y, "Wear Aggregation into Windows", "4.2",
                        "Non-overlapping windows of 15 consecutive runs;\nsum ΔVB and Δt per window → (s, t, ΔVB) per window · tool")
    y = bot - GAP
    _mo_arrow(ax, (SPINE_X, bot), (SPINE_X, y), color=NAVY, lw=1.3)

    # ---- 3+4. Shared scale (compact header, no box) + three base families ---
    # Folded into a text header rather than its own boxed step -- the
    # equation is short enough to read as a subtitle above the three
    # candidate boxes it defines, saving a full row's height without
    # dropping any information (portrait layout has to earn its vertical
    # budget back somewhere; this is the safest row to fold, since Equation
    # 1 belongs conceptually to the three boxes right below it anyway).
    hdr_top = y
    _mo_badge(ax, SPINE_X - SPINE_W / 2 + 0.30, hdr_top - 0.02, "4.3", navy=NAVY)
    ax.text(SPINE_X, hdr_top - 0.05, "Degradation-Process Models — Shared Scale",
            ha="center", va="top", fontsize=10.0, fontweight="bold", color="#0D1526", zorder=4)
    ax.text(SPINE_X, hdr_top - 0.30, "S(s,t) = η · [ (t/t_ref)^p − (s/t_ref)^p ]   (Eq. 1)",
            ha="center", va="top", fontsize=8.6, zorder=4)
    fan_y = hdr_top - 0.56

    # ---- Three base families (parallel candidates) --------------------------
    cand_h = 1.12
    cand_w = 1.74
    gap_c = 0.16
    xs = [SPINE_X - cand_w - gap_c, SPINE_X, SPINE_X + cand_w + gap_c]
    cand_top = fan_y - 0.20
    cy_cand = cand_top - cand_h / 2
    labels = [
        ("Gamma Process", "ΔVB ~ Gamma\n(shape=S, rate=β)", "Abdel-Hameed (1975)"),
        ("Inverse Gaussian", "ΔVB ~ IG(mean=S/β,\nshape=S²/β)", "Wang & Xu (2010)"),
        ("Wiener (Drift)", "ΔVB ~ N(mean=S/β,\nvar=S/β²)", "Whitmore (1995)"),
    ]
    for cx, (name, eq, cite) in zip(xs, labels):
        _mo_rounded_box(ax, cx, cy_cand, cand_w, cand_h, fill=CAND_FILL, edge=NAVY, lw=1.0)
        ax.text(cx, cand_top - 0.22, name, ha="center", va="top", fontsize=8.6, fontweight="bold", zorder=4)
        ax.text(cx, cand_top - 0.44, eq, ha="center", va="top", fontsize=7.2, zorder=4, linespacing=1.6)
        ax.text(cx, cand_top - 0.95, cite, ha="center", va="top", fontsize=6.8,
                style="italic", color=GRAY, zorder=4)
        _mo_arrow(ax, (SPINE_X, fan_y), (cx, cand_top + 0.03), color=NAVY, lw=1.0)
    cand_bot = cy_cand - cand_h / 2
    y = cand_bot - 0.24

    # ---- 5. Seven structural variants (single wide box) ---------------------
    var_top = y
    var_bot, _, var_h = step(var_top, "Seven Structural Variants", None,
                              "Wear-level-dependent (Gamma, IG)  ·  Left-truncated (Wiener)\n"
                              "Random-effects (Gamma, IG)  ·  Two-stage (Wiener)  ·  Generalized (IG)",
                              fill="#F3F1E7", title_size=10.0, detail_size=7.8, pad_top=0.24, pad_bot=0.22)
    for cx in xs:
        _mo_arrow(ax, (cx, cand_bot), (SPINE_X + (cx - SPINE_X) * 0.30, var_top + 0.03), color=GRAY, lw=0.9)
    y = var_bot - GAP
    _mo_arrow(ax, (SPINE_X, var_bot), (SPINE_X, y), color=NAVY, lw=1.3)

    # ---- 6+7. Estimation & age-dependence selection (header) + diamond ------
    # Folded the same way as 4.3 above: the standalone 4.4 box said little
    # that isn't already implied by the decision rule it feeds, so its own
    # header sits directly above the diamond that IS that rule, rather than
    # taking a full extra row.
    hdr2_top = y
    _mo_badge(ax, SPINE_X - 2.05, hdr2_top - 0.02, "4.4", navy=NAVY)
    ax.text(SPINE_X, hdr2_top - 0.05, "Estimation & Age-Dependence Selection",
            ha="center", va="top", fontsize=9.6, fontweight="bold", color="#0D1526", zorder=4)
    ax.text(SPINE_X, hdr2_top - 0.28, "Maximum likelihood (rounding-interval correction, Eq. 6) · Nelder–Mead, 3 starts",
            ha="center", va="top", fontsize=7.3, zorder=4)
    y = hdr2_top - 0.52
    _mo_arrow(ax, (SPINE_X, y), (SPINE_X, y - 0.20), color=NAVY, lw=1.3)
    y = y - 0.20

    dia_w, dia_h = 3.7, 1.02
    dia_top = y
    dia_cy = dia_top - dia_h / 2
    _mo_diamond(ax, SPINE_X, dia_cy, dia_w, dia_h, fill="#EDEDED", edge=NAVY)
    ax.text(SPINE_X, dia_cy + 0.16, "ΔAIC > 2 ?", ha="center", va="center", fontsize=9.2,
            fontweight="bold", zorder=4)
    ax.text(SPINE_X, dia_cy - 0.18, "(non-homogeneous vs. stationary)", ha="center", va="center",
            fontsize=7.0, style="italic", color=GRAY, zorder=4)
    side_w, side_h = 1.65, 0.72
    yes_x, no_x = SPINE_X - dia_w / 2 - 0.15 - side_w / 2, SPINE_X + dia_w / 2 + 0.15 + side_w / 2
    for lx, top_lbl, sub_lbl in [
        (yes_x, "YES: non-homog.", "p free, per tool"),
        (no_x, "NO: stationary", "p = 1, shared"),
    ]:
        _mo_rounded_box(ax, lx, dia_cy, side_w, side_h, fill="#EDEDED", edge=NAVY, lw=0.9)
        ax.text(lx, dia_cy + 0.14, top_lbl, ha="center", va="center", fontsize=7.6, fontweight="bold", zorder=4)
        ax.text(lx, dia_cy - 0.15, sub_lbl, ha="center", va="center", fontsize=6.8, color=GRAY, zorder=4)
    _mo_arrow(ax, (SPINE_X - dia_w / 2, dia_cy), (yes_x + side_w / 2, dia_cy), color=GRAY, lw=0.9)
    _mo_arrow(ax, (SPINE_X + dia_w / 2, dia_cy), (no_x - side_w / 2, dia_cy), color=GRAY, lw=0.9)
    dia_bot = dia_cy - dia_h / 2
    y = dia_bot - GAP
    _mo_arrow(ax, (SPINE_X, dia_bot), (SPINE_X, y), color=NAVY, lw=1.3)

    # ---- 8. Calibration & comparison (+ 4.6 side annex) ---------------------
    calib_w = 4.15
    bot, top, h = step(y, "Model Calibration & Family Comparison", "4.5",
                        "Rank by AIC / BIC; PIT–Kolmogorov–Smirnov\ncalibration (200 seeds); RMSE & terminal-bias checks",
                        w=calib_w, title_size=9.6, detail_size=7.6)
    calib_right = SPINE_X + calib_w / 2
    side_w2, side_h2 = 1.72, 1.46
    side_cx = calib_right + 0.18 + side_w2 / 2
    side_cy = top - side_h2 / 2
    _mo_rounded_box(ax, side_cx, side_cy, side_w2, side_h2, fill=SIDE_FILL, edge=NAVY, lw=1.0, ls=(0, (4, 2)))
    _mo_badge(ax, side_cx - side_w2 / 2 + 0.26, top - 0.28, "4.6", navy=NAVY)
    ax.text(side_cx, top - 0.24, "Robustness\nChecks (13)", ha="center", va="top",
            fontsize=7.8, fontweight="bold", zorder=4, linespacing=1.3)
    ax.text(side_cx, top - 0.62,
            "CV likelihood · bootstraps\ncluster-robust Vuong test\nmodel recovery · AIC-vs-BIC\nsensitivity sweeps\nautocorrelation check",
            ha="center", va="top", fontsize=6.5, color="#222222", linespacing=1.6, zorder=4)
    _mo_arrow(ax, (side_cx - side_w2 / 2, side_cy), (calib_right + 0.03, side_cy),
              color=GRAY, lw=1.0, style=(0, (3, 2)))
    # Gap sized to clear the taller 4.6 annex box below the calibration box's
    # own (shorter) bottom edge -- a plain GAP here would let the annex
    # overlap the reliability/cost row that follows.
    y = min(bot - GAP, top - side_h2 - 0.14)
    _mo_arrow(ax, (SPINE_X, bot), (SPINE_X, y), color=NAVY, lw=1.3)

    # ---- 9. Reliability + maintenance-cost translation -----------------------
    pair_w = 2.72
    pair_top = y
    pair_h = 1.15
    pair_cy = pair_top - pair_h / 2
    lx, rx = SPINE_X - pair_w / 2 - 0.10, SPINE_X + pair_w / 2 + 0.10
    _mo_arrow(ax, (SPINE_X, pair_top + GAP), (SPINE_X, pair_top + 0.03), color=NAVY, lw=1.3)
    for cx, badge, title, detail in [
        (lx, "4.7", "Reliability\nTranslation", "Kaplan–Meier restricted-\nmean MTTFF (Eq. 12);\n500 simulated lives"),
        (rx, "4.8", "Maintenance-Cost\nTranslation", "Age-replacement policy\n(Barlow & Proschan 1965);\ncost rate C(T), Eq. 14"),
    ]:
        _mo_rounded_box(ax, cx, pair_cy, pair_w, pair_h, fill=STEP_FILL, edge=NAVY, lw=1.0)
        _mo_badge(ax, cx - pair_w / 2 + 0.26, pair_top - 0.26, badge, navy=NAVY)
        ax.text(cx, pair_top - 0.22, title, ha="center", va="top", fontsize=8.4,
                fontweight="bold", zorder=4, linespacing=1.25)
        ax.text(cx, pair_top - 0.60, detail, ha="center", va="top", fontsize=6.9, color="#222222",
                linespacing=1.5, zorder=4)
    _mo_arrow(ax, (lx + pair_w / 2, pair_cy), (rx - pair_w / 2, pair_cy), color=GRAY, lw=0.9)
    pair_bot = pair_cy - pair_h / 2

    # ---- 10. Benchmark RUL model (4.9 side annex) ---------------------------
    bench_w, bench_h = 1.62, 1.28
    bench_cx = rx + pair_w / 2 + 0.18 + bench_w / 2
    if bench_cx + bench_w / 2 > W - 0.15:
        bench_cx = W - 0.15 - bench_w / 2
    bench_top = pair_top
    bench_cy = bench_top - bench_h / 2
    _mo_rounded_box(ax, bench_cx, bench_cy, bench_w, bench_h, fill=SIDE_FILL, edge=NAVY, lw=1.0, ls=(0, (4, 2)))
    _mo_badge(ax, bench_cx - bench_w / 2 + 0.24, bench_top - 0.24, "4.9", navy=NAVY)
    ax.text(bench_cx, bench_top - 0.22, "Benchmark\nRUL Model", ha="center", va="top",
            fontsize=7.6, fontweight="bold", zorder=4, linespacing=1.25)
    ax.text(bench_cx, bench_top - 0.58,
            "A: IG virtual age\nB: random forest\n(full & wear-only)\ntool-level conformal",
            ha="center", va="top", fontsize=6.3, color="#222222", linespacing=1.5, zorder=4)
    ax.text(bench_cx, bench_cy - bench_h / 2 + 0.14, "§5.5", ha="center", va="center",
            fontsize=6.3, style="italic", color=GRAY, zorder=4)

    # ---- 11. Results (I/O parallelogram) -------------------------------------
    y = pair_bot - GAP
    _mo_arrow(ax, (SPINE_X, pair_bot), (SPINE_X, y), color=NAVY, lw=1.3)
    res_top = y
    bot, top, h = step(res_top, "Results", "§5",
                        "Family ranking & calibration · MTTFF & conditional risk\nMaintenance-cost regret · RUL benchmark comparison",
                        shape="parallelogram", fill=IO_FILL, title_size=13, detail_size=8.4, pad_top=0.30, pad_bot=0.28)
    _mo_arrow(ax, (bench_cx, bench_cy - bench_h / 2), (SPINE_X + 0.9, res_top - 0.05), color=NAVY, lw=1.1)
    y_bottom = bot - 0.18

    # ---- Legend --------------------------------------------------------------
    leg_h = 0.85
    leg_top = y_bottom
    ax.add_patch(plt.Rectangle((0.15, leg_top - leg_h), W - 0.3, leg_h, facecolor="#FAFAFA",
                                edgecolor="0.6", linewidth=0.8, linestyle=(0, (2, 2)), zorder=2))
    ax.text(0.38, leg_top - 0.24, "Legend", fontsize=9.2, fontweight="bold", color="#0D1526", zorder=4)
    leg_items_y = leg_top - 0.55
    _mo_parallelogram(ax, 0.68, leg_items_y, 0.52, 0.24, fill=IO_FILL, edge=NAVY, lw=0.9)
    ax.text(1.08, leg_items_y, "Data input / output", fontsize=7.2, va="center", zorder=4)
    _mo_rounded_box(ax, 3.28, leg_items_y, 0.52, 0.24, fill=STEP_FILL, edge=NAVY, lw=0.9, radius=0.06)
    ax.text(3.68, leg_items_y, "Analysis step", fontsize=7.2, va="center", zorder=4)
    _mo_diamond(ax, 5.55, leg_items_y, 0.5, 0.32, fill="#EDEDED", edge=NAVY, lw=0.9)
    ax.text(5.94, leg_items_y, "Decision point", fontsize=7.2, va="center", zorder=4)
    _mo_rounded_box(ax, 7.55, leg_items_y, 0.5, 0.24, fill=SIDE_FILL, edge=NAVY, lw=0.9, ls=(0, (4, 2)), radius=0.06)
    ax.text(7.94, leg_items_y, "Robustness /\nbenchmark branch", fontsize=6.6, va="center", zorder=4, linespacing=1.2)
    leg_items_y2 = leg_top - 0.92
    _mo_arrow(ax, (0.45, leg_items_y2), (0.98, leg_items_y2), color=NAVY, lw=1.4, z=4)
    ax.text(1.08, leg_items_y2, "Primary sequential flow", fontsize=7.2, va="center", zorder=4)
    _mo_arrow(ax, (3.75, leg_items_y2), (4.28, leg_items_y2), color=GRAY, lw=1.0, style=(0, (3, 2)), z=4)
    ax.text(4.38, leg_items_y2, "Supporting / validation branch", fontsize=7.2, va="center", zorder=4)

    ax.set_ylim(leg_top - leg_h - 0.15, H)
    fig.tight_layout(pad=0.15)
    _save(fig, out_dir, "figure_methodology_overview")
    print(f"Saved Figure 2 (methodology overview) -> {out_dir}")
