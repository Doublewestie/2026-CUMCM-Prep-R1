"""
step2.3_comfort_report.py — Comfort Zone Statistical Report
=============================================================
FILT < 0.15: report statistics, prove inputs are decoupled.
"""

import os, json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from step0_config import *

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
THETA = 0.15


def main():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "clean_data.csv"))
    required = ["FILT_NTU", "RW_FLOW", "RW_NTU", "ALUM", "NTU"]
    df = df.dropna(subset=required)

    filt = df["FILT_NTU"].values
    comfort = filt < THETA
    stress  = filt >= THETA

    # Comfort zone stats
    c_filt = filt[comfort]
    print(f"Comfort zone (FILT < {THETA}): n={len(c_filt)} ({100*len(c_filt)/len(filt):.1f}%)")
    print(f"  FILT mean={c_filt.mean():.4f} std={c_filt.std():.4f} "
          f"median={np.median(c_filt):.4f} P95={np.percentile(c_filt,95):.4f}")

    # Correlations in comfort zone
    pairs = [("RW_FLOW", "FILT_NTU"), ("ALUM", "FILT_NTU"), ("RW_NTU", "FILT_NTU"),
             ("RW_NTU", "NTU"), ("FILT_NTU", "NTU")]
    print(f"\n  Variable correlations (comfort vs stress):")
    print(f"  {'Variable pair':<24s} {'Comfort r':>10s} {'Stress r':>10s} {'Delta':>8s}")
    print(f"  {'-'*55}")
    for v1, v2 in pairs:
        r_c = np.corrcoef(df.loc[comfort, v1], df.loc[comfort, v2])[0, 1]
        r_s = np.corrcoef(df.loc[stress, v1], df.loc[stress, v2])[0, 1]
        print(f"  {v1:>10s}->{v2:<12s} {r_c:>10.4f} {r_s:>10.4f} {r_s-r_c:>+8.4f}")

    # Save
    with open(os.path.join(OUTPUT_DIR, "comfort_zone_report.json"), "w", encoding="utf-8") as f:
        json.dump({
            "theta": THETA, "n_comfort": int(comfort.sum()),
            "n_stress": int(stress.sum()),
            "comfort_pct": round(100*comfort.sum()/len(filt), 1),
        }, f, indent=2)
    print(f"\n[DONE] comfort_zone_report.json")


if __name__ == "__main__":
    main()
