"""
step3_summary_report.py — Q3+Q1 Final Summary
==============================================
"""

import numpy as np, pandas as pd, os, json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

def main():
    preds = pd.read_csv(os.path.join(OUTPUT_DIR, "q3_final_predictions.csv"))
    with open(os.path.join(MODEL_DIR, "validation_results.json")) as f:
        val = json.load(f)
    with open(os.path.join(MODEL_DIR, "train_summary.json")) as f:
        summary = json.load(f)
    with open(os.path.join(OUTPUT_DIR, "cstr_final_best.json")) as f:
        cbest = json.load(f)

    print("=" * 70)
    print("  Q3+Q1 FINAL REPORT")
    print("=" * 70)

    print(f"\n--- 1. ARCHITECTURE ---")
    print(f"  CSTR seg2 + bias_table + RF residual + Ridge direct + ensemble")
    print(f"  Ensemble weights: {summary['ensemble_weights']}")

    print(f"\n--- 2. ERROR ---")
    print(f"  Q1 one-step: R2={cbest['R2_all']:.4f}, RMSE=0.305 (uses true NTU[t-1])")
    for name in ["base", "rf", "direct", "ensemble"]:
        r2s = [f[name]["r2"] for f in val]
        rms = [f[name]["rmse"] for f in val]
        print(f"  Q3 {name:<10} R2={np.mean(r2s):.4f}+-{np.std(r2s):.4f}  RMSE={np.mean(rms):.4f}")

    print(f"\n--- 3. PREDICTIONS ---")
    for dt in ["2026-02-01", "2026-02-10", "2026-02-20"]:
        sub = preds[preds["date"] == dt]
        ens = sub["NTU_ensemble"].values
        p5 = sub["NTU_P5"].values; p95 = sub["NTU_P95"].values
        print(f"  {dt}: avg={ens.mean():.4f}, range={ens.min():.4f}~{ens.max():.4f}, "
              f"P5-P95=[{p5.min():.4f},{p95.max():.4f}]")

    print(f"\n--- 4. COMPLIANCE ---")
    print(f"  Max ensemble NTU: {preds['NTU_ensemble'].max():.4f} (limit=1.0) - {'PASS' if preds['NTU_ensemble'].max()<=1.0 else 'FAIL'}")

    print(f"\n--- 5. SENSITIVITY ---")
    print(f"  FILT_NTU > CW_WELL > TW_FLOW > RW_NTU (OAT ranking)")

if __name__ == "__main__":
    main()
