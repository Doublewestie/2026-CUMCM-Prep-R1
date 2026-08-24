"""
step3.8+_forecast_cstr.py — Q3 Honest Forecast: AR(6) FILT → CSTR Chain
==========================================================================
Critical fix: replaces true FILT(t) in CSTR chain with AR(6)-forecast FILT(t).
At prediction time (5:00), FILT for 7:00-19:00 is unknown and must be forecast.

Three evaluation modes per fold:
  1. Oracle:  True FILT + persisted CW/Q  (upper bound, honest)
  2. Forecast: AR(6) FILT + persisted CW/Q (deployable model)
  3. Persistence: NTU(t) = NTU(5:00)       (baseline, diagnostic only —
     NOT comparable to Q1 one-step persistence; may be extreme negative)

Four-layer conditional routing (A/B/C_strong/C_weak) applied to all modes.

Outputs:
  - 5-fold CV metrics for all three modes
  - Decomposition table: R² drop from Oracle → Forecast
  - NOTE: 2026 Feb 1/10/20 predictions are NOT implemented here; use
    step3.8_final_stratified.py Part C (2025 same-date proxy, CUMCM
    convention — 2026 monthly files have only 1 day with missing NTU).
"""

import os, json, warnings
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings("ignore")

# ─── Constants ────────────────────────────────────────────────
EPS = 1e-3
T1_THR, T2_THR = 0.05, 0.15
A_T1, A_T2 = 400, 250
A_SAME, A_DIFF = 100, 20
RL_MED, Q_MED = 8.0, 48
C_TH = 1.0
ALPHA_B = 0.34
GAMMA_W = 0.25
N_SPLITS = 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
CLEAN_CSV = os.path.join(OUTPUT_DIR, "clean_data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── CSTR Physics (identical to step3.8) ──────────────────────
def get_tier(ft):
    return 1 if ft <= T1_THR else (2 if ft <= T2_THR else 3)

def get_balance_flag(rl, q):
    return 1 if (rl - RL_MED) * (q - Q_MED) > 0 else 0

def cstr_step(prev_ntu, ft, cw_prev, q_prev, rl_curr=None):
    tier = get_tier(ft)
    if tier == 1:
        A0 = A_T1
    elif tier == 2:
        A0 = A_T2
    else:
        rl_v = rl_curr if rl_curr is not None else RL_MED
        A0 = A_SAME if get_balance_flag(rl_v, q_prev) else A_DIFF
    theta = max(A0 * max(cw_prev, 0.1) / max(q_prev, 1.0), 0.02)
    beta = np.clip(np.exp(-2.0 / theta), 0.001, 0.999)
    return np.clip(beta * prev_ntu + (1.0 - beta) * ft, 0, None)


# ─── Data Loading ─────────────────────────────────────────────
def load_2025():
    df = pd.read_csv(CLEAN_CSV)
    n = len(df)
    doy = pd.to_datetime(df["DATE"]).dt.dayofyear.values
    return {
        "FILT": df["FILT_NTU"].values.astype(float),
        "NTU":  df["NTU"].values.astype(float),
        "CW":   df["CW_WELL_LEVEL"].values.astype(float),
        "Q":    df["TW_FLOW"].values.astype(float),
        "RL":   df["RIVER_LEVEL"].values.astype(float),
        "RW":   df["RW_NTU"].values.astype(float),
        "doy":  doy,
        "n":    n,
    }


# ─── AR(6) Training (per-fold) ───────────────────────────────
def train_ar6(log_filt, train_mask):
    """Train RidgeCV AR(6) on log_filt[train_mask]."""
    n = len(log_filt)
    X = np.zeros((n, 6))
    for lag in range(1, 7):
        X[lag:, lag - 1] = log_filt[:-lag]
        X[:lag, lag - 1] = log_filt[0]
    start = 6
    X_tr = X[start:][train_mask[start:]]
    y_tr = log_filt[start:][train_mask[start:]]
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    m = RidgeCV(alphas=alphas).fit(X_tr, y_tr)
    return m.coef_, m.intercept_


def forecast_filt(ar_coefs, intercept, history_log, n_steps):
    """Rolling AR(6) forecast from history_log (last 6 values)."""
    series = list(history_log[-6:])
    for _ in range(n_steps):
        y = intercept + sum(ar_coefs[i] * series[-6 + i] for i in range(6))
        series.append(y)
    return np.clip(np.exp(np.array(series[-n_steps:])) - EPS, 0, None)


# ─── CSTR Chain with Forecast FILT ────────────────────────────
def build_oracle_chain(data, day_start):
    """Original step3.8 CSTR chain: uses TRUE FILT(t) at each step.
    This is the transfer function formulation — equivalent to original code."""
    cs = np.zeros(7)
    cs[0] = data["NTU"][day_start]
    for i in range(1, 7):
        ft = data["FILT"][day_start + 3 + i]
        cw_prev = data["CW"][day_start + 2 + i]
        q_prev = data["Q"][day_start + 3 + i]
        rl_curr = data["RL"][day_start + 3 + i]
        cs[i] = cstr_step(cs[i - 1], ft, cw_prev, q_prev, rl_curr)
    return cs


def build_forecast_chain(data, day_start, ar_coefs, ar_intercept):
    """CSTR chain with AR(6)-forecast FILT(t) + persisted CW/Q/RL from 5:00.
    
    At prediction time (5:00), FILT for 7:00-19:00 is forecast via AR(6).
    CW, Q, RL are persisted from their last known values at 5:00/3:00.
    """
    # AR(6) forecast: use last 6 known FILT values (ds-3 through ds+2 = 19:00~5:00)
    hist_start = max(0, day_start - 3)
    filt_history = data["FILT"][hist_start:day_start + 3]
    filt_history_log = [np.log(max(f, 0) + EPS) for f in filt_history[-6:]]
    while len(filt_history_log) < 6:
        filt_history_log = [filt_history_log[0]] + filt_history_log
    filt_pred = forecast_filt(ar_coefs, ar_intercept, filt_history_log, 6)

    # Persist CW/Q/RL from last known values at prediction time
    cw_persist = data["CW"][day_start + 2]
    q_persist = data["Q"][day_start + 2]
    rl_persist = data["RL"][day_start + 2]

    cs = np.zeros(7)
    cs[0] = data["NTU"][day_start]
    for i in range(1, 7):
        ft = filt_pred[i - 1]
        cs[i] = cstr_step(cs[i - 1], ft, cw_persist, q_persist, rl_persist)
    return cs


# ─── Day Classification (identical to step3.8) ─────────────────
def classify_day(data, day_start):
    f05 = data["FILT"][day_start + 2]
    n01 = data["NTU"][day_start]
    n05 = data["NTU"][day_start + 2]
    if f05 < 0.05 and abs(n05 - n01) < 0.02:
        return "A"
    elif f05 >= 0.15 or abs(n05 - n01) >= 0.05:
        return "C"
    return "B"

def c_strong(filt_5am):
    return filt_5am >= C_TH


# ─── Four-Layer Routing Prediction ────────────────────────────
def route_predictions(dy, f05, n5, cs_chain):
    """Apply four-layer routing to CSTR chain, return 7 predictions."""
    preds = np.zeros(7)
    for i in range(7):
        if dy == "A":
            preds[i] = n5
        elif dy == "C":
            if c_strong(f05):
                preds[i] = np.clip(cs_chain[i], 0, None)
            else:
                preds[i] = np.clip(n5 + GAMMA_W * (cs_chain[i] - n5), 0, None)
        else:  # B
            preds[i] = np.clip(ALPHA_B * cs_chain[i] + (1 - ALPHA_B) * n5, 0, None)
    return preds


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  Q3 Honest Forecast: AR(6) FILT → CSTR Chain (CV)")
    print("=" * 70)

    data = load_2025()
    n = data["n"]
    log_filt = np.log(data["FILT"] + EPS)
    all_days = list(range(6, n - 30, 12))

    # Bias table from Q1
    bias_path = os.path.join(MODEL_DIR, "bias_table.json")
    if os.path.exists(bias_path):
        with open(bias_path) as f:
            bias_table = json.load(f)
        bias_table = {int(k): {int(kk): vv for kk, vv in v.items()}
                      for k, v in bias_table.items()}
    else:
        bias_table = {}

    # ── Part A: 5-fold TS-CV ──────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Part A: 5-fold TS-CV (honest FILT forecast)")
    print(f"{'=' * 60}")

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_results = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(np.arange(n))):
        # Train AR(6) on train fold
        ar_coefs, ar_intercept = train_ar6(log_filt, tr_idx)

        # Collect validation days
        p_oracle, p_forecast, p_persist, p_truefilt_chain = [], [], [], []
        y_all = []
        type_counts = {"A": 0, "B": 0, "C_strong": 0, "C_weak": 0}

        for ds in all_days:
            if ds + 12 > n:
                continue
            if (ds + 6) not in va_idx:
                continue

            # Known at 5:00 (for routing)
            f05 = data["FILT"][ds + 2]
            ntu_5am = data["NTU"][ds + 2]

            # Day classification + type tracking
            dy = classify_day(data, ds)

            type_key = dy
            if dy == "C":
                type_key = "C_strong" if c_strong(f05) else "C_weak"
            type_counts[type_key] += 1

            # True NTU targets
            targets = np.array([data["NTU"][ds + 3 + i] for i in range(7)])

            # --- Mode 1: Oracle (true FILT, original chain) ---
            oracle_chain = build_oracle_chain(data, ds)
            oracle_preds = route_predictions(dy, f05, ntu_5am, oracle_chain)

            # --- Mode 2: Forecast (AR(6) FILT + persisted CW/Q) ---
            forecast_chain = build_forecast_chain(data, ds, ar_coefs, ar_intercept)
            forecast_preds = route_predictions(dy, f05, ntu_5am, forecast_chain)

            # --- Mode 3: Persistence ---
            persist_preds = np.full(7, ntu_5am)

            for i in range(7):
                p_oracle.append(oracle_preds[i])
                p_forecast.append(forecast_preds[i])
                p_persist.append(persist_preds[i])
                y_all.append(targets[i])

        p_oracle = np.array(p_oracle)
        p_forecast = np.array(p_forecast)
        p_persist = np.array(p_persist)
        y_all = np.array(y_all)

        r2_oracle = r2_score(y_all, p_oracle)
        r2_forecast = r2_score(y_all, p_forecast)
        r2_persist = r2_score(y_all, p_persist)

        fold_results.append({
            "fold": fold,
            "r2_oracle": round(r2_oracle, 4),
            "r2_forecast": round(r2_forecast, 4),
            "r2_persist": round(r2_persist, 4),
            "rmse_forecast": round(float(np.sqrt(mean_squared_error(y_all, p_forecast))), 4),
            "n_samples": len(y_all),
            "type_counts": type_counts,
        })

        print(f"  Fold {fold}: Oracle={r2_oracle:.4f}  Forecast={r2_forecast:.4f}  "
              f"Persist={r2_persist:.4f}  (n={len(y_all)})")

    # ── Summary ───────────────────────────────────────────────
    r2_oracle_all = [f["r2_oracle"] for f in fold_results]
    r2_forecast_all = [f["r2_forecast"] for f in fold_results]
    r2_persist_all = [f["r2_persist"] for f in fold_results]

    print(f"\n{'=' * 60}")
    print(f"  CV Summary")
    print(f"{'=' * 60}")
    print(f"  Oracle (true FILT + pers CW/Q): R² = {np.mean(r2_oracle_all):.4f} "
          f"± {np.std(r2_oracle_all):.4f}")
    print(f"  Forecast (AR6 FILT + pers CW/Q): R² = {np.mean(r2_forecast_all):.4f} "
          f"± {np.std(r2_forecast_all):.4f}")
    print(f"  Persistence:                   R² = {np.mean(r2_persist_all):.4f} "
          f"± {np.std(r2_persist_all):.4f}")

    # ── Decomposition ─────────────────────────────────────────
    delta_filt = np.mean(r2_oracle_all) - np.mean(r2_forecast_all)
    delta_oracle_base = np.mean(r2_oracle_all) - np.mean(r2_persist_all)
    delta_forecast_base = np.mean(r2_forecast_all) - np.mean(r2_persist_all)

    print(f"\n  ── R² Decomposition ──")
    print(f"  Oracle gain over persistence:    +{delta_oracle_base:.4f}")
    print(f"  Forecast gain over persistence:   +{delta_forecast_base:.4f}")
    print(f"  FILT-forecast penalty:            -{delta_filt:.4f} "
          f"({delta_filt/delta_oracle_base*100:.1f}% of oracle advantage lost)")

    # ── Save ──────────────────────────────────────────────────
    summary = {
        "model": "Q3 AR(6) FILT → CSTR Chain (Honest Forecast)",
        "cv_mean": {
            "r2_oracle": round(np.mean(r2_oracle_all), 4),
            "r2_oracle_std": round(np.std(r2_oracle_all), 4),
            "r2_forecast": round(np.mean(r2_forecast_all), 4),
            "r2_forecast_std": round(np.std(r2_forecast_all), 4),
            "r2_persist": round(np.mean(r2_persist_all), 4),
            "r2_persist_std": round(np.std(r2_persist_all), 4),
            "delta_filt_forecast_penalty": round(delta_filt, 4),
        },
        "fold_results": fold_results,
    }

    out_path = os.path.join(RESULTS_DIR, "q3_forecast_cv_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE] Results saved to {out_path}")


if __name__ == "__main__":
    main()
