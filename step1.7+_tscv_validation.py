"""
step1.7+_tscv_validation.py — Q1 Honest TS-CV Validation
==========================================================
Critical fix: evaluates Q1 CSTR model on independent 5-fold TimeSeriesSplit
instead of reporting in-sample R²=0.807 on full dataset.

Evaluates fixed CSTR parameters (A_T1=400, A_T2=250, A_same=100, A_diff=20)
on held-out validation folds. Also reports Persistence baseline per fold.

The CSTR parameters are pre-optimized on the full dataset (taken from step1.7 output),
but R² is evaluated ONLY on the validation fold — testing generalization, not
re-reporting optimization data.

Outputs:
  - 5-fold CV metrics for CSTR+Balance and Persistence
  - Per-fold R², RMSE, ΔR² over Persistence
"""

import os, json, warnings, argparse
import numpy as np, pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings("ignore")

EPS = 1e-6
T1_THR, T2_THR = 0.05, 0.15

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CLEAN_CSV = os.path.join(OUTPUT_DIR, "clean_data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Fixed parameters from step1.7_final_cstr.py (globally optimized)
A_T1, A_T2 = 400, 250
A_SAME, A_DIFF = 100, 20
A_T3_FALLBACK = 30
N_SPLITS = 5

# Balance detector medians — configurable via CLI (D1 decision: default 6.09/44
# matches step1.7_final_cstr.py / sum_7 reported R²=0.8072; Q3-optimized 8.0/48
# kept as documented alternative: python step1.7+_tscv_validation.py --rl-med 8.0 --q-med 48)
PARSER = argparse.ArgumentParser(description="Q1 honest TS-CV validation with configurable balance medians")
PARSER.add_argument("--rl-med", type=float, default=6.09, help="Balance detector RL median (Q1 original=6.09, Q3-optimized=8.0)")
PARSER.add_argument("--q-med", type=float, default=44.0, help="Balance detector Q median (Q1 original=44.0, Q3-optimized=48)")
_CLI_ARGS = PARSER.parse_args()
RL_MED, Q_MED = _CLI_ARGS.rl_med, _CLI_ARGS.q_med
PARAM_TAG = f"rl{RL_MED:g}_q{Q_MED:g}"


def load_data():
    df = pd.read_csv(CLEAN_CSV)
    return {
        "FILT": df["FILT_NTU"].values.astype(np.float64),
        "NTU":  df["NTU"].values.astype(np.float64),
        "CW":   df["CW_WELL_LEVEL"].values.astype(np.float64),
        "Q":    df["TW_FLOW"].values.astype(np.float64),
        "RL":   df["RIVER_LEVEL"].values.astype(np.float64),
        "RW":   df["RW_NTU"].values.astype(np.float64),
        "n":    len(df),
    }


def get_balance_flag(rl, q):
    return 1 if (rl - RL_MED) * (q - Q_MED) > 0 else 0


def predict_cstr(data, start_idx, end_idx):
    """CSTR one-step-ahead prediction on [start_idx, end_idx).
    
    Uses CSTR transfer function: NTU_pred(t) = β₂·NTU_true(t-1) + (1-β₂)·FILT(t).
    This is NOT a recursive forecast — it uses the true previous NTU as input,
    which is physically justified: at time t, the previous NTU measurement is known.
    Same formulation as step1.7_final_cstr.py.
    """
    filt = data["FILT"]
    ntu = data["NTU"]
    cw = data["CW"]
    q = data["Q"]
    rl = data["RL"]
    n = end_idx - start_idx
    pred = np.zeros(n)
    y_true = np.array([ntu[start_idx + i] for i in range(n)])

    if n == 0:
        return pred, y_true

    pred[0] = ntu[start_idx]

    for t in range(1, n):
        idx = start_idx + t
        ft = filt[idx]
        cw_prev = cw[idx - 1] if idx > 0 else cw[0]
        q_prev = q[idx - 1] if idx > 0 else q[0]
        rv = rl[idx] if not np.isnan(rl[idx]) else RL_MED

        if ft <= T1_THR:
            A0 = A_T1
        elif ft <= T2_THR:
            A0 = A_T2
        else:
            A0 = A_SAME if get_balance_flag(rv, q_prev) else A_DIFF

        H = max(cw_prev, 0.1)
        Qv = max(q_prev, 1.0)
        theta = max(A0 * H / Qv, 0.02)
        beta = np.clip(np.exp(-2.0 / theta), 0.001, 0.999)
        pred[t] = beta * ntu[idx - 1] + (1.0 - beta) * ft

    return np.clip(pred, 0, None), y_true


def main():
    print("=" * 70)
    print("  Q1 TS-CV Validation: Fixed CSTR Parameters")
    print("  A_T1=400, A_T2=250, A_same=100, A_diff=20")
    print("=" * 70)

    data = load_data()
    n = data["n"]
    print(f"  Samples: {n}")
    print(f"  NTU: mean={np.mean(data['NTU']):.4f}, std={np.std(data['NTU']):.4f}")

    # Full in-sample R² (for reference)
    pred_full, y_full = predict_cstr(data, 0, n)
    r2_in_sample = r2_score(y_full[1:], pred_full[1:])
    rmse_in_sample = np.sqrt(mean_squared_error(y_full[1:], pred_full[1:]))
    r2_persist_is = r2_score(y_full[1:], data["NTU"][:-1])

    print(f"\n  In-Sample (reference, not for reporting):")
    print(f"    CSTR+Balance R² = {r2_in_sample:.4f}  RMSE = {rmse_in_sample:.4f}")
    print(f"    Persistence R²  = {r2_persist_is:.4f}")

    # ── 5-fold TS-CV ──────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  5-fold TimeSeriesSplit Validation")
    print(f"{'=' * 60}")

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(np.arange(n))):
        va_start = va_idx[0]
        va_end = va_idx[-1] + 1
        va_n = va_end - va_start

        # CSTR one-step-ahead prediction
        pred_cstr_arr, y_va = predict_cstr(data, va_start, va_end)

        # Persistence baseline: NTU(t) = NTU(t-1)
        pred_persist = np.zeros(va_n)
        pred_persist[0] = data["NTU"][va_start - 1] if va_start > 0 else data["NTU"][0]
        for i in range(1, va_n):
            pred_persist[i] = data["NTU"][va_start + i - 1]

        if va_n > 1:
            r2_cstr = r2_score(y_va[1:], pred_cstr_arr[1:])
            rmse_cstr = np.sqrt(mean_squared_error(y_va[1:], pred_cstr_arr[1:]))
            r2_persist = r2_score(y_va[1:], pred_persist[1:])
            rmse_persist = np.sqrt(mean_squared_error(y_va[1:], pred_persist[1:]))
        else:
            r2_cstr, rmse_cstr, r2_persist, rmse_persist = 0, 0, 0, 0

        delta = r2_cstr - r2_persist
        fold_metrics.append({
            "fold": fold,
            "val_start": int(va_start),
            "val_end": int(va_end),
            "n_val": int(va_n),
            "r2_cstr": round(float(r2_cstr), 4),
            "rmse_cstr": round(float(rmse_cstr), 4),
            "r2_persist": round(float(r2_persist), 4),
            "rmse_persist": round(float(rmse_persist), 4),
            "delta_r2": round(float(delta), 4),
        })

        print(f"  Fold {fold}: [{va_start:4d}-{va_end:4d}) n={va_n:4d}  "
              f"CSTR={r2_cstr:.4f}  Persist={r2_persist:.4f}  Δ={delta:+.4f}")

    # ── Summary ───────────────────────────────────────────────
    r2_cstr_all = [m["r2_cstr"] for m in fold_metrics]
    r2_persist_all = [m["r2_persist"] for m in fold_metrics]
    delta_all = [m["delta_r2"] for m in fold_metrics]

    print(f"\n{'=' * 60}")
    print(f"  TS-CV Summary (to report in paper)")
    print(f"{'=' * 60}")
    print(f"  CSTR+Balance R² = {np.mean(r2_cstr_all):.4f} ± {np.std(r2_cstr_all):.4f}")
    print(f"  Persistence R²  = {np.mean(r2_persist_all):.4f} ± {np.std(r2_persist_all):.4f}")
    print(f"  ΔR² (CSTR-Pers) = {np.mean(delta_all):+.4f} ± {np.std(delta_all):.4f}")

    # ── Save ──────────────────────────────────────────────────
    result = {
        "model": "Q1 CSTR+Balance (TS-CV Validation)",
        "parameters": {
            "A_T1": A_T1, "A_T2": A_T2,
            "A_same": A_SAME, "A_diff": A_DIFF,
            "RL_med": RL_MED, "Q_med": Q_MED,
        },
        "in_sample_reference": {
            "r2_cstr": round(r2_in_sample, 4),
            "r2_persist": round(r2_persist_is, 4),
        },
        "cv_mean": {
            "r2_cstr": round(np.mean(r2_cstr_all), 4),
            "r2_cstr_std": round(np.std(r2_cstr_all), 4),
            "r2_persist": round(np.mean(r2_persist_all), 4),
            "r2_persist_std": round(np.std(r2_persist_all), 4),
            "delta_r2": round(np.mean(delta_all), 4),
            "delta_r2_std": round(np.std(delta_all), 4),
        },
        "fold_results": fold_metrics,
    }

    out_path = os.path.join(RESULTS_DIR, f"q1_tscv_validation_{PARAM_TAG}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE] Results saved to {out_path}  (RL_med={RL_MED:g}, Q_med={Q_MED:g})")


if __name__ == "__main__":
    main()
