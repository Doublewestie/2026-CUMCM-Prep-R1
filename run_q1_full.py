"""
run_q1_full.py — Final Summary Table for Q1 three-tier scheme
"""
import numpy as np, os, sys, json
from step0_config import *
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

# Load all results
tier_params = json.load(open(OUT_TIER_PARAMS, encoding="utf-8"))
tier1_report = json.load(open(OUT_TIER1_REPORT, encoding="utf-8"))
tier2_report = json.load(open(OUT_TIER2_REPORT, encoding="utf-8"))
tier3_best = json.load(open(OUT_TIER3_BEST, encoding="utf-8"))
fi_lines = open(OUT_TIER_FACTOR, encoding="utf-8").readlines()

# Tier distribution
print(f"\n{'='*80}")
print(f"  Q1 THREE-TIER GREYBOX SCHEME — FINAL SUMMARY")
print(f"{'='*80}")

print(f"\n  [1] Tier Distribution")
print(f"  {'Tier':<8} {'Threshold':<16} {'n':<8} {'%':<6} {'Strategy':<30}")
print(f"  {'-'*68}")
for t in [1,2,3]:
    k = f"T{t}"
    cnt = tier_params["tier_counts"][k]
    pct = tier_params["tier_pcts"][k]
    thr_l = [0, *TIER_THRESHOLDS][t-1]
    thr_h = [*TIER_THRESHOLDS, float("inf")][t-1]
    thr_s = f"[{thr_l}, {thr_h})" if t < 3 else f">={thr_l}"
    strs = {1: "Empirical frequency sampling",
            2: f"Dual-path (best: {tier2_report['recommendation']})",
            3: "CSTR + linear feedback + sigmoid gamma(t)"}
    print(f"  T{t:<7} {thr_s:<16} {cnt:<8} {pct:<6}% {strs[t]:<30}")

# Classifier
print(f"\n  [2] Tier Classifier")
print(f"  C1 (T1 vs rest):  {tier_params['c1_mean_acc']:.4f}")
print(f"  C2 (T2 vs T3):    {tier_params.get('c2_mean_acc',0):.4f}")
print(f"  Overall 3-class:  {tier_params['overall_acc']:.4f}")

# T1
m_f = tier1_report["validation_filt_metrics"]
m_n = tier1_report["validation_ntu_metrics"]
print(f"\n  [3] T1 (<=0.05, 49.0%) — Empirical Sampling")
print(f"  FILT RMSE={m_f['rmse']}  R2={m_f['r2']}  MAE={m_f['mae']}  "
      f"JS={tier1_report['validation_js_divergence']}")
print(f"  NTU  RMSE={m_n['rmse']}  R2={m_n['r2']}  MAE={m_n['mae']}")

# T2
pa, pb = tier2_report["path_a"], tier2_report.get("path_b_log", {})
ps = tier2_report.get("path_b_sigmoid", {})
rec = tier2_report["recommendation"]
print(f"\n  [4] T2 (0.05~0.15, 30.0%) — Dual Path")
print(f"  Path A (Empirical):     FILT RMSE={pa['filt_rmse']} R2={pa['filt_r2']}")
print(f"  Path B (Log-compress):  FILT RMSE={pb.get('filt_rmse','?')} R2={pb.get('filt_r2','?')}  "
      f"k={pb.get('params',{}).get('k','?')} alpha={pb.get('params',{}).get('alpha','?')}")
print(f"  Path B (Sigmoid var):   FILT RMSE={ps.get('filt_rmse','?')} R2={ps.get('filt_r2','?')}")
print(f"  Best: {rec}")

# T3
print(f"\n  [5] T3 (>0.15, 21.0%) — CSTR+Feedback")
print(f"  Best: fb={tier3_best['fb_type']} gamma={tier3_best['gamma_type']} lam3={tier3_best['lambda3']}")
print(f"  NTU R2={tier3_best['overall_r2']:.4f}  RMSE={tier3_best['cv_rmse_mean']:.4f}")
print(f"  tau1 peak: {tier3_best.get('tau_peak_lag',0)*2}h")
print(f"  tau1 weights: {tier3_best.get('tau_weights',[])}")

# T3 Feature Importance
print(f"\n  [6] T3 Feature Importance (Top 10)")
print(f"  {'Rank':<6} {'Feature':<20} {'SHAP':<8} {'Perm':<8} {'Robust':<8}")
print(f"  {'-'*50}")
for i, l in enumerate(fi_lines[1:11]):
    parts = l.strip().split(",")
    if len(parts) >= 6:
        print(f"  {parts[0]:<6} {parts[1]:<20} {parts[3]:<8} {parts[4]:<8} {parts[5]:<8}")

# Comparison with original
print(f"\n  [7] Comparison with Original XGBoost")
print(f"  {'Method':<55} {'NTU R2':<10} {'NTU RMSE':<10}")
print(f"  {'-'*75}")
print(f"  Original XGBoost (101-dim, all data):                 {'0.34':<10} {'0.437':<10}")
print(f"  Three-Tier Greybox (combined):")
print(f"    T1 (<=0.05, 49%): Empirical sampling               {m_f['r2']!s:<10} {m_f['rmse']!s:<10}")
print(f"    T2 (0.05~0.15, 30%): {rec:<20}             {pb.get('filt_r2',pa.get('filt_r2','?'))!s:<10} {pb.get('filt_rmse',pa.get('filt_rmse','?'))!s:<10}")
print(f"    T3 (>0.15, 21%): CSTR+{tier3_best['fb_type']} feedback         {tier3_best['overall_r2']!s:<10} {tier3_best['cv_rmse_mean']!s:<10}")

# Key innovation summary
print(f"\n\n  {'='*80}")
print(f"  KEY INNOVATIONS")
print(f"  {'='*80}")
innovations = [
    "1. Three-Tier Partition: FILT_NTU split at 0.05 and 0.15 based on empirical distributions",
    "2. T1: Noise-dominated zone treated with empirical sampling (JS divergence=0.05 vs Gaussian 0.64)",
    "3. T2: Log-compressed greybox maps T3's physical structure into transitional zone via learned k,alpha",
    "4. T3: CSTR model found to apply to NTU (清水池混合), not FILT — fundamental structural insight",
    "5. tau_1 learned via softmax (peak at 4h): skip traditional statistical delay estimation",
    "6. Linear operator feedback + sigmoid gamma(t) modeling human control actions in loss function",
    "7. eta_coag discovered as #1 T3 factor (Robust=0.335) — removal efficiency dominates in stress zone",
]
for i in innovations:
    print(f"  {i}")
print(f"{'='*80}")
