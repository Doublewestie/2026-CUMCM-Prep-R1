"""
step3_residual_train.py — Train residual correction models on 2025 data
=======================================================================
Produces:
  1. bias_table[tier, step] — per-tier per-step mean bias
  2. rf_model.pkl — Random Forest residual predictor
  3. ridge_H.pkl — Direct multi-step Ridge models (H=1,3,6)
  4. val_weights.json — optimal ensemble weights
"""

import numpy as np, pandas as pd, os, json, pickle, warnings
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CLEAN_CSV = os.path.join(OUTPUT_DIR, "clean_data.csv")

T1_THR, T2_THR = 0.05, 0.15

_cstr_path = os.path.join(OUTPUT_DIR, "cstr_final_best.json")
if os.path.exists(_cstr_path):
    with open(_cstr_path) as f:
        _cstr = json.load(f)
    A_T1 = _cstr.get("A_T1", 400)
    A_T2 = _cstr.get("A_T2", 250)
    A_T3 = _cstr.get("A_T3", 30)
else:
    A_T1, A_T2, A_T3 = 400, 250, 30

# ==============================================================
# 1. Load data
# ==============================================================
def load_2025():
    df = pd.read_csv(CLEAN_CSV)
    arr = {
        "FILT": df["FILT_NTU"].values.astype(float),
        "NTU": df["NTU"].values.astype(float),
        "CW": df["CW_WELL_LEVEL"].values.astype(float),
        "Q": df["TW_FLOW"].values.astype(float),
        "RW_NTU": df["RW_NTU"].values.astype(float),
        "RL": df["RIVER_LEVEL"].values.astype(float),
    }
    time_vals = pd.to_numeric(df["TIME"], errors="coerce").fillna(700).astype(int)
    arr["hour"] = (time_vals // 100) % 24
    arr["day_sin"] = np.sin(2 * np.pi * pd.to_datetime(df["DATE"]).dt.dayofyear / 365)
    arr["day_cos"] = np.cos(2 * np.pi * pd.to_datetime(df["DATE"]).dt.dayofyear / 365)
    return arr

# ==============================================================
# 2. Recursive CSTR with bias correction framework
# ==============================================================
def get_tier(ft):
    return 1 if ft <= T1_THR else (2 if ft <= T2_THR else 3)

def get_A(tier):
    return [A_T1, A_T2, A_T3][tier - 1]

def cstr_one_step(prev_ntu, ft, cw_t, q_t, bias=0.0):
    tier = get_tier(ft)
    A0 = get_A(tier)
    Hv = max(cw_t, 0.1); Qv = max(q_t, 1.0)
    theta = max(A0 * Hv / Qv, 0.02)
    beta = np.clip(np.exp(-2.0 / theta), 0.001, 0.999)
    return np.clip(beta * prev_ntu + (1.0 - beta) * ft + bias, 0, None)

def run_recursive(filt_seq, cw_seq, q_seq, ntu_init, bias_table):
    """Run recursive CSTR with step-dependent bias correction."""
    n = len(filt_seq)
    pred = np.zeros(n)
    pred[0] = ntu_init
    for i in range(1, n):
        tier = get_tier(filt_seq[i])
        bias = bias_table.get(tier, {}).get(i, 0.0)
        pred[i] = cstr_one_step(pred[i-1], filt_seq[i], cw_seq[i], q_seq[i], bias)
    return pred

# ==============================================================
# 3. Train bias table
# ==============================================================
def train_bias_table(data):
    """Run recursive CSTR with NO bias, record per-(tier,step) mean bias."""
    n = len(data["NTU"])
    # Store errors per (tier, step)
    errors = {(t, s): [] for t in [1, 2, 3] for s in range(1, 13)}

    for day_start in range(0, n - 24, 12):
        # Sort data from hour=1 to hour=23
        day_slice = slice(day_start, day_start + 12)
        pred = np.zeros(12)
        pred[0] = data["NTU"][day_start]  # start from true NTU at 1:00
        for step in range(1, 12):
            idx = day_start + step
            ft = data["FILT"][idx]
            tier = get_tier(ft)
            pred[step] = cstr_one_step(pred[step-1], ft, data["CW"][idx], data["Q"][idx])
            err = data["NTU"][idx] - pred[step]
            errors[(tier, step)].append(err)

    bias_table = {1: {}, 2: {}, 3: {}}
    for (t, s), vals in errors.items():
        if len(vals) > 5:
            bias_table[t][s] = float(np.mean(vals))
        else:
            bias_table[t][s] = 0.0
    return bias_table

# ==============================================================
# 4. Train RF residual model
# ==============================================================
def train_rf_residual(data, bias_table):
    """Train RF to predict recursive error from features."""
    n = len(data["NTU"])
    X_list, y_list = [], []

    for day_start in range(6, n - 24, 12):
        pred = np.zeros(12)
        pred[0] = data["NTU"][day_start]
        for step in range(1, 12):
            idx = day_start + step
            ft = data["FILT"][idx]; tier = get_tier(ft)
            pred[step] = cstr_one_step(pred[step-1], ft, data["CW"][idx], data["Q"][idx],
                                       bias_table[tier].get(step, 0.0))
            err = data["NTU"][idx] - pred[step]
            # Features: [step, tier, FILT, CW, Q, NTU_prev, hour_cos, hour_sin, day_cos]
            feat = [
                step, tier, ft, data["CW"][idx], data["Q"][idx],
                pred[step-1], data["hour"][idx],
                np.cos(2 * np.pi * data["hour"][idx] / 24),
                np.sin(2 * np.pi * data["hour"][idx] / 24),
                data["day_cos"][idx], data["day_sin"][idx],
            ]
            X_list.append(feat)
            y_list.append(err)

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"\n  RF training samples: {len(X)}")

    rf = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=10,
                               random_state=42, n_jobs=-1)
    rf.fit(X, y)
    y_pred = rf.predict(X)
    r2 = r2_score(y, y_pred)
    print(f"  RF in-sample R2: {r2:.4f}")

    # Feature importance
    feat_names = ["step", "tier", "FILT", "CW", "Q", "NTU_prev",
                  "hour", "hour_cos", "hour_sin", "day_cos", "day_sin"]
    fi = sorted(zip(feat_names, rf.feature_importances_), key=lambda x: -x[1])
    print(f"  Top-5 features:")
    for n, v in fi[:5]:
        print(f"    {n}: {v:.4f}")

    return rf, feat_names

# ==============================================================
# 5. Train direct multi-step Ridge models
# ==============================================================
def train_direct_models(data):
    """Train Ridge models for direct H-step prediction."""
    models = {}
    results = {}
    for H in [1, 3, 6]:
        X_list, y_list = [], []
        for t in range(6, len(data["NTU"]) - H):
            ft = data["FILT"][t+H]
            feat = [
                data["NTU"][t], data["FILT"][t], data["FILT"][t+H],
                data["CW"][t+H], data["Q"][t+H],
                data["hour"][t+H],
                np.cos(2 * np.pi * data["hour"][t+H] / 24),
                np.sin(2 * np.pi * data["hour"][t+H] / 24),
                data["day_cos"][t+H], data["day_sin"][t+H],
            ]
            X_list.append(feat); y_list.append(data["NTU"][t+H])

        X = np.array(X_list); y = np.array(y_list)
        valid = ~np.isnan(y) & (y > 0)
        X, y = X[valid], y[valid]

        # TS-CV
        tscv = TimeSeriesSplit(n_splits=5)
        r2s, rms = [], []
        for tr, va in tscv.split(X):
            m = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]).fit(X[tr], y[tr])
            p = m.predict(X[va])
            r2s.append(r2_score(y[va], p))
            rms.append(np.sqrt(mean_squared_error(y[va], p)))
            models[f"Ridge_H{H}"] = m

        print(f"\n  Direct H={H}: CV R2={np.mean(r2s):.4f}+-{np.std(r2s):.4f}, RMSE={np.mean(rms):.4f}")
        results[H] = {"r2": round(float(np.mean(r2s)), 4), "rmse": round(float(np.mean(rms)), 4)}

    return models, results

# ==============================================================
# 6. Cross-validate ensemble weights
# ==============================================================
def find_ensemble_weights(data, bias_table, rf_model, direct_models):
    """Find optimal ensemble weights via TS-CV on 2025."""
    n = len(data["NTU"])
    all_preds = {method: [] for method in ["base", "rf", "direct1", "direct3"]}
    all_true = []

    tscv = TimeSeriesSplit(n_splits=5)
    fold_weights = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(np.arange(n))):
        va_start = max(va_idx[0], 11)  # need 12-step window
        va_end = va_idx[-1]
        fold_preds = {m: [] for m in all_preds}
        fold_true = []

        for day_start in range(va_start, va_end - 12, 12):
            true_ntu = data["NTU"][day_start + 3:day_start + 10]  # 7:00-19:00
            if np.any(np.isnan(true_ntu)) or np.any(true_ntu <= 0):
                continue

            # Initialize from previous day's mean
            tr_data = data["NTU"][:day_start]
            init_val = np.mean(tr_data[~np.isnan(tr_data)]) if len(tr_data) > 0 else data["NTU"][0]

            # Base CSTR (bias-corrected)
            pred_b = np.zeros(12)
            pred_b[0] = init_val
            for s in range(1, 12):
                idx = day_start + s
                ft = data["FILT"][idx]; tier = get_tier(ft)
                pred_b[s] = cstr_one_step(pred_b[s-1], ft, data["CW"][idx], data["Q"][idx],
                                         bias_table[tier].get(s, 0.0))
            fold_preds["base"].extend(pred_b[3:10].tolist())

            # RF corrected
            pred_rf = np.zeros(12); pred_rf[0] = init_val
            for s in range(1, 12):
                idx = day_start + s
                ft = data["FILT"][idx]; tier = get_tier(ft)
                pred_rf[s] = cstr_one_step(pred_rf[s-1], ft, data["CW"][idx], data["Q"][idx],
                                          bias_table[tier].get(s, 0.0))
                if s >= 1:
                    feat = np.array([[s, tier, ft, data["CW"][idx], data["Q"][idx],
                                      pred_rf[s-1], data["hour"][idx],
                                      np.cos(2*np.pi*data["hour"][idx]/24),
                                      np.sin(2*np.pi*data["hour"][idx]/24),
                                      data["day_cos"][idx], data["day_sin"][idx]]])
                    delta = rf_model.predict(feat)[0]
                    pred_rf[s] = np.clip(pred_rf[s] + delta, 0, None)
            fold_preds["rf"].extend(pred_rf[3:10].tolist())

            # Direct models
            for H, key in [(1, "direct1"), (3, "direct3")]:
                for q3_step, s in enumerate([3, 4, 5, 6, 7, 8, 9]):
                    t_h = day_start + s
                    if t_h + H < n:
                        feat = np.array([[data["NTU"][t_h], data["FILT"][t_h], data["FILT"][t_h+H],
                                          data["CW"][t_h+H], data["Q"][t_h+H],
                                          data["hour"][t_h+H],
                                          np.cos(2*np.pi*data["hour"][t_h+H]/24),
                                          np.sin(2*np.pi*data["hour"][t_h+H]/24),
                                          data["day_cos"][t_h+H], data["day_sin"][t_h+H]]])
                        pred_h = direct_models[f"Ridge_H{H}"].predict(feat)[0]
                    else:
                        pred_h = fold_preds["base"][-1] if fold_preds["base"] else 0
                    fold_preds[key].append(pred_h)

            fold_true.extend(true_ntu.tolist())

        if len(fold_true) > 10:
            # Find optimal weights via grid search
            best_w, best_r2 = None, -999
            for w1 in np.arange(0, 1.1, 0.2):
                for w2 in np.arange(0, 1.1 - w1, 0.2):
                    w3 = 1 - w1 - w2
                    if w3 < 0: continue
                    ens = w1 * np.array(fold_preds["base"]) + \
                          w2 * np.array(fold_preds["rf"]) + \
                          w3 * np.array(fold_preds["direct1"]) if len(fold_preds["direct1"]) == len(fold_preds["base"]) else \
                          np.array(fold_preds["base"])
                    r2_e = r2_score(fold_true, ens)
                    if r2_e > best_r2:
                        best_r2 = r2_e; best_w = (w1, w2, w3)
            fold_weights.append({"fold": fold, "w_base": best_w[0], "w_rf": best_w[1],
                                 "w_direct": best_w[2], "r2": best_r2})

    avg_w = {k: np.mean([f[k] for f in fold_weights]) for k in ["w_base", "w_rf", "w_direct"]}
    print(f"\n  Optimal ensemble weights:")
    print(f"    w_base = {avg_w['w_base']:.3f}")
    print(f"    w_rf   = {avg_w['w_rf']:.3f}")
    print(f"    w_direct = {avg_w['w_direct']:.3f}")
    print(f"    Avg fold R2 = {np.mean([f['r2'] for f in fold_weights]):.4f}")

    return avg_w, fold_weights

# ==============================================================
# 7. Main
# ==============================================================
def main():
    print("=" * 70)
    print("  Residual Correction Model Training (2025)")
    print("=" * 70)

    data = load_2025()
    print(f"  Loaded {len(data['NTU'])} samples.")

    # ---- 1. Bias table ----
    print(f"\n{'='*60}")
    print(f"  Phase 1: Per-tier per-step bias table")
    print(f"{'='*60}")
    bias_table = train_bias_table(data)
    for t in [1, 2, 3]:
        vals = [bias_table[t].get(s, 0) for s in range(1, 13)]
        print(f"  T{t}: {[f'{v:+.4f}' for v in vals]}")

    # ---- 2. RF residual ----
    print(f"\n{'='*60}")
    print(f"  Phase 2: RF residual model")
    print(f"{'='*60}")
    rf_model, feat_names = train_rf_residual(data, bias_table)

    # ---- 3. Direct models ----
    print(f"\n{'='*60}")
    print(f"  Phase 3: Direct multi-step models")
    print(f"{'='*60}")
    direct_models, direct_results = train_direct_models(data)

    # ---- 4. Ensemble weights ----
    print(f"\n{'='*60}")
    print(f"  Phase 4: Cross-validation ensemble weights")
    print(f"{'='*60}")
    avg_w, fold_weights = find_ensemble_weights(data, bias_table, rf_model, direct_models)

    # ---- 5. Baseline comparison ----
    print(f"\n{'='*60}")
    print(f"  Phase 5: Baseline comparison on 2025")
    print(f"{'='*60}")

    # Naive lag-1 baseline
    ntu = data["NTU"]
    valid = ~np.isnan(ntu)
    lag1_pred = np.roll(ntu, 1)
    lag1_pred[0] = ntu[0]
    r2_lag1 = r2_score(ntu[valid], lag1_pred[valid])
    rmse_lag1 = np.sqrt(mean_squared_error(ntu[valid], lag1_pred[valid]))
    print(f"  Lag-1 (naive): R2={r2_lag1:.4f}, RMSE={rmse_lag1:.4f}")

    # One-step CSTR
    pred_1s = np.zeros(len(ntu)); pred_1s[0] = ntu[0]
    for t in range(1, len(ntu)):
        pred_1s[t] = cstr_one_step(ntu[t-1], data["FILT"][t], data["CW"][t], data["Q"][t])
    r2_1s = r2_score(ntu[valid], pred_1s[valid])
    rmse_1s = np.sqrt(mean_squared_error(ntu[valid], pred_1s[valid]))
    print(f"  CSTR one-step: R2={r2_1s:.4f}, RMSE={rmse_1s:.4f}")

    # ---- Save ----
    model_dir = os.path.join(OUTPUT_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, "bias_table.json"), "w") as f:
        json.dump(bias_table, f, indent=2)

    with open(os.path.join(model_dir, "rf_model.pkl"), "wb") as f:
        pickle.dump(rf_model, f)

    for k, v in direct_models.items():
        with open(os.path.join(model_dir, f"{k}.pkl"), "wb") as f:
            pickle.dump(v, f)

    train_summary = {
        "bias_table": bias_table,
        "rf_feature_names": feat_names,
        "direct_models": {k: direct_results[int(k.split("_H")[1])] for k in direct_models},
        "ensemble_weights": avg_w,
        "fold_weights": fold_weights,
        "baseline": {"lag1": {"r2": round(r2_lag1, 4), "rmse": round(rmse_lag1, 4)},
                     "cstr_1step": {"r2": round(r2_1s, 4), "rmse": round(rmse_1s, 4)},
                     "cstr_recursive_bias_corrected": {"ensemble_r2": round(np.mean([f['r2'] for f in fold_weights]), 4)}},
    }
    with open(os.path.join(model_dir, "train_summary.json"), "w") as f:
        json.dump(train_summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Models saved to {model_dir}/")
    print(f"  Train summary saved.")
    print(f"\n[DONE]")

if __name__ == "__main__":
    main()
