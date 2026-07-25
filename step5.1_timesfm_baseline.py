"""
step5.1_timesfm_baseline.py — TimesFM zero-shot baseline for FILT_NTU
=====================================================================
Uses locally cached TimesFM 2.5 weights (E:\AI models\timesfm_weights).
Predicts Feb 2026 FILT_NTU at 2h resolution (3 days x 12 points each = 36 steps).
"""
import os, sys, json
import numpy as np, pandas as pd
import torch, timesfm
import warnings; warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision("high")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "tables")
HF_CACHE = r"E:\AI models\timesfm_weights"
HF_MIRROR = "https://hf-mirror.com"

os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_ENDPOINT"] = HF_MIRROR
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_2025_filt():
    """Load full 2025 FILT_NTU sequence."""
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "clean_data.csv"))
    return df["FILT_NTU"].values.astype(np.float64)


def main():
    print("=" * 60)
    print("  TimesFM 2.5 Zero-Shot FILT_NTU Forecast")
    print("=" * 60)

    print("\n[1/4] Loading model...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    model.compile(timesfm.ForecastConfig(
        max_context=4608, max_horizon=128,
        normalize_inputs=True, use_continuous_quantile_head=True,
        infer_is_positive=True, fix_quantile_crossing=True,
    ))
    print("  Model loaded OK.")

    print("\n[2/4] Loading 2025 FILT_NTU...")
    filt_2025 = load_2025_filt()
    print(f"  Samples: {len(filt_2025)}, mean={filt_2025.mean():.4f}, max={filt_2025.max():.4f}")

    print("\n[3/4] Forecasting Feb 2026 (3 days x 12 steps = 36 points)...")
    point, quantiles = model.forecast(horizon=36, inputs=[filt_2025])
    pred = point[0]
    q = quantiles[0]

    # Organize by day: 12 points per day at 2h resolution
    times_2h = [f"{h:02d}:00" for h in range(7, 31, 2)]  # 7:00-5:00 (wraps)
    days = ["2026-02-01", "2026-02-10", "2026-02-20"]

    rows = []
    for d in range(3):
        for t in range(12):
            idx = d * 12 + t
            rows.append({
                "date": days[d], "time": times_2h[t],
                "FILT_NTU_pred": round(float(pred[idx]), 4),
                "P10": round(float(q[idx, 1]), 4),
                "P50": round(float(q[idx, 5]), 4),
                "P90": round(float(q[idx, 9]), 4),
            })

    df_out = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "timesfm_baseline.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  [DONE] {out_path}")

    print("\n[4/4] Summary:")
    for day in days:
        sub = df_out[df_out["date"] == day]
        print(f"  {day}: min={sub['FILT_NTU_pred'].min():.4f}, "
              f"mean={sub['FILT_NTU_pred'].mean():.4f}, "
              f"max={sub['FILT_NTU_pred'].max():.4f}, "
              f"P90_range=[{sub['P10'].min():.4f},{sub['P90'].max():.4f}]")

    # Save summary metrics
    summary = {
        "model": "TimesFM 2.5 (200M params, zero-shot)",
        "context_length": len(filt_2025),
        "forecast_horizon": 36,
        "feb1_mean": round(float(df_out[df_out["date"]=="2026-02-01"]["FILT_NTU_pred"].mean()), 4),
        "feb10_mean": round(float(df_out[df_out["date"]=="2026-02-10"]["FILT_NTU_pred"].mean()), 4),
        "feb20_mean": round(float(df_out[df_out["date"]=="2026-02-20"]["FILT_NTU_pred"].mean()), 4),
    }
    with open(os.path.join(OUTPUT_DIR, "timesfm_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [DONE] timesfm_summary.json")

    print(f"\n{'='*60}")
    print("  TimesFM baseline complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
