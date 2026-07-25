"""
step1.8_model_compare.py — Q1 model comparison coordinator
===========================================================
Compares two CSTR model pipelines:
  Pipeline A: step1.7_final_cstr (grid search + Balance Detector)
  Pipeline B: step1.9_physical_reconstruct (Langmuir + L-BFGS-B)

Outputs: results/tables/q1_model_comparison.csv
"""
import os, json, sys, time, io
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "tables")
os.makedirs(RESULTS_DIR, exist_ok=True)
EPS = 1e-6

# ================================================================
# Load common data
# ================================================================
CLEAN_CSV = os.path.join(OUTPUT_DIR, "clean_data.csv")
df = pd.read_csv(CLEAN_CSV)
filt = df["FILT_NTU"].values.astype(float)
ntu = df["NTU"].values.astype(float)
cw = df["CW_WELL_LEVEL"].values.astype(float)
tw = df["TW_FLOW"].values.astype(float)
n = len(ntu)


def compute_metrics(y_true, y_pred):
    ssr = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ssr / (sst + EPS)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return r2, rmse


# ================================================================
# Pipeline A: step1.7 CSTR + Balance Detector
# ================================================================
def eval_pipeline_a():
    cstr_path = os.path.join(OUTPUT_DIR, "cstr_final_best.json")
    if not os.path.exists(cstr_path):
        return {"name": "A: CSTR+Balance (step1.7)", "R2": None, "RMSE": None}

    with open(cstr_path) as f:
        cstr = json.load(f)

    A_T1 = cstr["A_T1"]; A_T2 = cstr["A_T2"]; A_T3 = cstr["A_T3"]
    rule = cstr.get("A_T3_rule", {})
    A_same = rule.get("A_same", 100); A_diff = rule.get("A_diff", 20)
    RL_med = cstr.get("RL_med", 6.09); Q_med = cstr.get("Q_med", 44.0)
    rl = df["RIVER_LEVEL"].values.astype(float)

    pred_a = np.zeros(n); pred_a[0] = ntu[0]
    for t in range(1, n):
        H = max(cw[t-1], 0.1); Qv = max(tw[t-1], 1.0)
        ft = filt[t]
        if ft <= 0.05:
            A0 = A_T1
        elif ft <= 0.15:
            A0 = A_T2
        else:
            rv = rl[t]
            A0 = A_same if (not np.isnan(rv) and (rv - RL_med) * (tw[t] - Q_med) > 0) else A_diff
        theta = A0 * H / Qv
        beta = np.clip(np.exp(-2.0 / max(theta, 0.02)), 0.001, 0.999)
        pred_a[t] = beta * ntu[t-1] + (1 - beta) * ft
    pred_a = np.clip(pred_a, 0, np.inf)

    r2, rmse = compute_metrics(ntu, pred_a)
    t3_mask = filt > 0.15
    r2_t3, _ = compute_metrics(ntu[t3_mask], pred_a[t3_mask]) if t3_mask.sum() >= 10 else (None, None)

    return {
        "name": "A: CSTR+Balance (step1.7)",
        "R2": round(r2, 4), "RMSE": round(rmse, 4), "R2_T3": round(r2_t3, 4) if r2_t3 else None,
        "params": f"A_T1={A_T1}, A_T2={A_T2}, A_T3={A_T3}, A_same={A_same}, A_diff={A_diff}",
        "n_params": 5,
        "method": "Grid search + Balance Detector",
    }


# ================================================================
# Pipeline B: step1.9_physical_reconstruct
# ================================================================
def eval_pipeline_b():
    phys_path = os.path.join(OUTPUT_DIR, "step1_physical_results.json")
    if not os.path.exists(phys_path):
        return {"name": "B: Langmuir+CSTR (physical)", "R2": None, "RMSE": None,
                "R2_T3": None, "n_params": 0, "params": "",
                "method": "Langmuir + L-BFGS-B",
                "note": "Run step1.9_physical_reconstruct.py first"}

    with open(phys_path) as f:
        phys = json.load(f)

    r2 = phys.get("full_r2")
    rmse = phys.get("full_rmse")
    tier_r2 = phys.get("tier_r2", {})
    r2_t3 = tier_r2.get("T3") if isinstance(tier_r2, dict) else None
    params = phys.get("params", {})
    param_str = ", ".join(f"{k}={v:.4g}" for k, v in params.items() if k != "A_cstr")

    return {
        "name": "B: Langmuir+CSTR (physical)",
        "R2": round(r2, 4) if r2 else None, "RMSE": round(rmse, 4) if rmse else None,
        "R2_T3": round(r2_t3, 4) if r2_t3 else None,
        "params": f"A_cstr={params.get('A_cstr','?')}, {param_str}",
        "n_params": len(params),
        "method": "Langmuir + L-BFGS-B",
    }


# ================================================================
# Baseline: mean prediction
# ================================================================
def eval_baseline():
    pred = np.full(n, ntu.mean())
    r2, rmse = compute_metrics(ntu, pred)
    return {"name": "Baseline: Mean", "R2": round(r2, 4), "RMSE": round(rmse, 4),
            "R2_T3": None, "params": "NTU_mean", "n_params": 1, "method": "Constant"}


# ================================================================
# Main
# ================================================================
def main():
    print("=" * 65)
    print("  Q1 Model Comparison: CSTR+Balance vs Physical Reconstruct")
    print("=" * 65)

    results = [eval_baseline(), eval_pipeline_a(), eval_pipeline_b()]

    print(f"\n  {'Model':<35s} {'R2':>8s} {'RMSE':>8s} {'T3_R2':>8s} {'n_params':>8s}")
    print(f"  {'-'*67}")
    for r in results:
        r2_s = f"{r['R2']:.4f}" if r['R2'] is not None else "N/A"
        rmse_s = f"{r['RMSE']:.4f}" if r['RMSE'] is not None else "N/A"
        t3_s = f"{r['R2_T3']:.4f}" if r['R2_T3'] is not None else "N/A"
        print(f"  {r['name']:<35s} {r2_s:>8s} {rmse_s:>8s} {t3_s:>8s} {r['n_params']:>8d}")
        if r.get('note'):
            print(f"    [note] {r['note']}")

    df_out = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, "q1_model_comparison.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  [DONE] {out_path}")

    return df_out


if __name__ == "__main__":
    main()
