"""
step5.3_package_figures.py — Collect & organize paper figures
=============================================================
Collects 34 figures from results/figures/ + output/figures/ into paper_figures/,
renaming by priority order, resolving path/name mismatches.
"""
import os, shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.join(BASE_DIR, "paper_figures")
SRC1 = os.path.join(BASE_DIR, "results", "figures")
SRC2 = os.path.join(BASE_DIR, "output", "figures")
os.makedirs(PAPER_DIR, exist_ok=True)

# Priority figure mapping: (src_dir, src_name, dst_name, label)
PRIORITY_MAP = [
    # Priority 1: CSTR formula + tier schematic (composite from existing)
    (SRC2, "q1_cstr_pred_vs_actual.png", "fig01_cstr_tier_overview.png", "CSTR model + 3-tier"),
    # Priority 2: CSTR ablation bar chart
    (SRC2, "q1_cstr_ablation.png", "fig02_cstr_ablation.png", "CSTR component ablation"),
    # Priority 3: T3 feature importance
    (SRC2, "q1_t3_feature_importance.png", "fig03_t3_feature_importance.png", "T3 feature importance (eta_coag #1)"),
    # Priority 4: Q2 log-AR pred vs actual
    (SRC1, "q2_pred_vs_actual.png", "fig04_q2_pred_vs_actual.png", "Q2 log-AR prediction"),
    # Priority 5: Q2 log-AR ablation
    (SRC2, "q2_logar_ablation_bar.png", "fig05_q2_logar_ablation.png", "Q2 log-AR ablation"),
    # Priority 6: Q4 risk heatmap
    (SRC1, "q4_zone_grade_heatmap.png", "fig06_q4_risk_heatmap.png", "Q4 risk heatmap"),
    # Priority 7: Q1 pred vs actual per tier
    (SRC1, "q1_pred_vs_actual.png", "fig07_q1_pred_vs_actual.png", "Q1 pred vs actual per tier"),
]

# Supplemental figures
EXTRA_FIGS = [
    # Q1
    "q1_tier_distribution.png", "q1_summary.png", "q1_tier_comparison.png",
    "q1_cstr_parameters.png", "q1_time_series.png", "cstr_refinement.png",
    "cstr_final.png", "q1_t1_empirical_dist.png",
    # Q2
    "q2_residual_diagnostics.png", "q2_feature_importance.png", "q2_tier_comparison.png",
    "q2_dual_mode_partition.png", "q2_operator_policy.png",
    "q2_lag_weights.png", "theta_diagnostic.png",
    # Q3
    "q3_prediction_curves.png", "q1_vs_q3_comparison.png",
    "q3_error_accumulation.png", "q3_daily_average.png",
    "q3_all_days_facet.png", "q3_sensitivity_oat.png",
    "q3_sobol_s1.png", "q3_sobol_st.png", "q3_sobol_interaction.png",
    # Q4
    "q4_transition_matrix.png", "q4_risk_timeseries.png",
    "q4_ntu_vs_risk.png", "q4_grade_pie.png",
    "q4_event_confusion.png", "q4_dimension_contribution.png",
]


def find_source(filename):
    """Search both source directories for a figure."""
    for src in [SRC1, SRC2]:
        fpath = os.path.join(src, filename)
        if os.path.exists(fpath):
            return fpath
    return None


def main():
    print("=" * 60)
    print("  Paper Figure Packaging")
    print("=" * 60)

    copied, missing = 0, []

    # Priority figures
    print("\n  --- Priority figures (fig01-fig07) ---")
    for src_dir, src_name, dst_name, label in PRIORITY_MAP:
        src = find_source(src_name) if src_dir is None else os.path.join(src_dir, src_name)
        dst = os.path.join(PAPER_DIR, dst_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
            print(f"  [{label}] {src_name} -> {dst_name}")
        else:
            missing.append(src_name)
            print(f"  [MISS] {src_name} (searched both dirs)")

    # Supplemental figures
    print(f"\n  --- Supplemental figures ---")
    for name in EXTRA_FIGS:
        src = find_source(name)
        if src:
            dst = os.path.join(PAPER_DIR, name)
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing.append(name)

    # Summary
    total = len(PRIORITY_MAP) + len(EXTRA_FIGS)
    print(f"\n{'='*60}")
    print(f"  Copied: {copied}/{total}")
    if missing:
        print(f"  Missing ({len(missing)}):")
        for m in missing:
            print(f"    - {m}")
    print(f"  Output: {PAPER_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
