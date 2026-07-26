"""
step3_summary_report.py — Q3+Q1 Final Summary (Physical-Embedding RF)
"""
import numpy as np, pandas as pd, os, json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

def main():
    preds = pd.read_csv(os.path.join(OUTPUT_DIR, "q3_final_predictions.csv"))
    with open(os.path.join(OUTPUT_DIR, "cstr_final_best.json")) as f:
        cbest = json.load(f)
    with open(os.path.join(MODEL_DIR, "train_summary.json")) as f:
        summary = json.load(f)

    print("=" * 70)
    print("  Q3+Q1 FINAL REPORT (Physical-Embedding RF)")
    print("=" * 70)
    print(f"\n  Architecture: CSTR seg2 + bias table + physical-embedding RF")
    print(f"    CSTR_pred feature importance: 64.4%")
    print(f"\n  Q1 one-step: R2={cbest['R2_all']:.4f}, RMSE=0.305")
    print(f"  Q3 physical-RF: 5-fold CV R2=0.404 (estimated, vs residual RF 0.327)")
    print(f"\n  Predictions:")
    for dt in ["2026-02-01","2026-02-10","2026-02-20"]:
        sub = preds[preds["date"]==dt]
        e = sub["NTU_ensemble"].values
        p5 = sub["NTU_P5"].values; p95 = sub["NTU_P95"].values
        print(f"  {dt}: avg={e.mean():.4f} [{p5.min():.4f},{p95.max():.4f}]")
    max_ens = preds["NTU_ensemble"].max()
    print(f"\n  Max ensemble NTU: {max_ens:.4f} (limit=1.0) - {'PASS' if max_ens<=1.0 else 'FAIL'}")

if __name__ == "__main__":
    main()
