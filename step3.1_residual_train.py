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
# 4b. Train physical-embedding RF (direct NTU prediction)
# ==============================================================
def train_physical_rf(data, bias_table):
    """RF predicting NTU directly with CSTR_pred as a feature."""
    n = len(data["NTU"])
    X_list, y_list = [], []
    for day_start in range(6, n - 24, 12):
        f_day = data["FILT"][day_start:day_start+12]
        cw_day = data["CW"][day_start:day_start+12]
        q_day = data["Q"][day_start:day_start+12]
        ntu_day = data["NTU"][day_start:day_start+12]
        qi = [3,4,5,6,7,8,9]
        f_q3 = np.array([f_day[i] for i in qi])
        cw_q3 = np.array([cw_day[i] for i in qi])
        q_q3 = np.array([q_day[i] for i in qi])
        t_q3 = np.array([ntu_day[i] for i in qi])
        if np.any(np.isnan(t_q3)): continue
        # Recursive CSTR
        cs = np.zeros(7); cs[0] = data["NTU"][day_start] if day_start > 0 else data["NTU"][0]
        for i in range(1, 7):
            t = 1 if f_q3[i] <= 0.05 else (2 if f_q3[i] <= 0.15 else 3)
            cs[i] = cstr_one_step(cs[i-1], f_q3[i], cw_q3[i], q_q3[i], bias_table[t].get(i, 0))
        dc = data["day_cos"][day_start]; ds = data["day_sin"][day_start]
        for i in range(7):
            feat = [i, get_tier(f_q3[i]), f_q3[i], cw_q3[i], q_q3[i],
                    cs[i], cs[max(0,i-1)],
                    [7,9,11,13,15,17,19][i],
                    np.cos(2*np.pi*[7,9,11,13,15,17,19][i]/24),
                    np.sin(2*np.pi*[7,9,11,13,15,17,19][i]/24),
                    dc, ds, np.mean(f_q3[:i+1]) if i > 0 else f_q3[0]]
            X_list.append(feat); y_list.append(t_q3[i])
    X = np.array(X_list); y = np.array(y_list)
    valid = ~np.isnan(y) & (y > 0)
    X, y = X[valid], y[valid]
    print(f"\n  Physical RF training samples: {len(X)}")
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5,
                               random_state=42, n_jobs=-1)
    rf.fit(X, y)
    y_pred = rf.predict(X)
    print(f"  Physical RF in-sample R2: {r2_score(y, y_pred):.4f}")
    feat_names = ["step","tier","FILT","CW","Q","CSTR_pred","CSTR_prev",
                  "hour","hour_cos","hour_sin","day_cos","day_sin","FILT_mean"]
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
def build_q3_features(data, day_start, bias_table, rf_res=None, rf_phy=None):
    """Build Q3 prediction window features and run all models.
    Returns (true_ntu_7h, base_pred, res_rf_pred, phy_rf_pred, direct_pred)."""
    f_day = data["FILT"][day_start:day_start+12]
    cw_day = data["CW"][day_start:day_start+12]
    q_day = data["Q"][day_start:day_start+12]
    ntu_day = data["NTU"][day_start:day_start+12]
    qi = [3,4,5,6,7,8,9]
    f_q3 = np.array([f_day[i] for i in qi])
    cw_q3 = np.array([cw_day[i] for i in qi])
    q_q3 = np.array([q_day[i] for i in qi])
    t_q3 = np.array([ntu_day[i] for i in qi])
    true_ntu = t_q3.copy()
    # Init
    tr_data = data["NTU"][:day_start]
    init = np.mean(tr_data[~np.isnan(tr_data)]) if len(tr_data) > 0 else data["NTU"][0]
    # Recursive CSTR
    cs = np.zeros(7); cs[0] = init
    for i in range(1, 7):
        t = get_tier(f_q3[i]); b = bias_table[t].get(i, 0)
        cs[i] = cstr_one_step(cs[i-1], f_q3[i], cw_q3[i], q_q3[i], b)
    # Base = CSTR
    base_pred = cs.copy()
    if rf_res is None:
        return true_ntu, base_pred, None, None, None
    # Residual RF
    pr = np.zeros(7); pr[0] = init
    for i in range(1, 7):
        t = get_tier(f_q3[i]); b = bias_table[t].get(i, 0)
        pr[i] = cstr_one_step(pr[i-1], f_q3[i], cw_q3[i], q_q3[i], b)
    rf_feats = np.array([[i, get_tier(f_q3[i]), f_q3[i], cw_q3[i], q_q3[i],
                          pr[i-1], [7,9,11,13,15,17,19][i],
                          np.cos(2*np.pi*[7,9,11,13,15,17,19][i]/24),
                          np.sin(2*np.pi*[7,9,11,13,15,17,19][i]/24),
                          data["day_cos"][day_start], data["day_sin"][day_start]]
                         for i in range(1, 7)])
    pr[1:] = np.clip(pr[1:] + rf_res.predict(rf_feats), 0, None)
    res_rf_pred = pr.copy()
    # Physical-embedding RF
    dc = data["day_cos"][day_start]; ds = data["day_sin"][day_start]
    phy_feats = np.array([[i, get_tier(f_q3[i]), f_q3[i], cw_q3[i], q_q3[i],
                           cs[i], cs[max(0,i-1)],
                           [7,9,11,13,15,17,19][i],
                           np.cos(2*np.pi*[7,9,11,13,15,17,19][i]/24),
                           np.sin(2*np.pi*[7,9,11,13,15,17,19][i]/24),
                           dc, ds, np.mean(f_q3[:i+1]) if i > 0 else f_q3[0]]
                          for i in range(7)])
    phy_pred = np.clip(rf_phy.predict(phy_feats), 0, None)
    return true_ntu, base_pred, res_rf_pred, phy_pred, None

def find_ensemble_weights(data, bias_table, rf_res, rf_phy, direct_models):
    """Find optimal ensemble weights via TS-CV on 2025.
    Tests: base, res_rf, phy_rf, res_rf+phy_rf blend, +direct."""
    n = len(data["NTU"])
    tscv = TimeSeriesSplit(n_splits=5)
    fold_results = []
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(np.arange(n))):
        va_start = max(va_idx[0], 11); va_end = va_idx[-1]
        p_base, p_res, p_phy, p_direct, targets = [], [], [], [], []
        for day_start in range(va_start, va_end - 12, 12):
            t, b, r, p, _ = build_q3_features(data, day_start, bias_table, rf_res, rf_phy)
            if len(t) == 0: continue
            targets.extend(t.tolist()); p_base.extend(b.tolist())
            p_res.extend(r.tolist()); p_phy.extend(p.tolist())
        y = np.array(targets); v = ~np.isnan(y) & (y > 0)
        y = y[v]; pb = np.array(p_base)[v]; pr = np.array(p_res)[v]; pp = np.array(p_phy)[v]
        # Find best blend weight between res_rf and phy_rf
        best_w, best_r2 = 0.5, -999
        for w in np.arange(0, 1.01, 0.05):
            pe = w * pr + (1-w) * pp
            r2e = r2_score(y, pe)
            if r2e > best_r2: best_r2 = r2e; best_w = w
        pe_best = best_w * pr + (1-best_w) * pp
        fold_results.append({"fold": fold, "w_res": best_w, "w_phy": 1-best_w,
                             "r2_res": r2_score(y, pr), "r2_phy": r2_score(y, pp),
                             "r2_blend": best_r2})
        print(f"  Fold {fold}: best w_res={best_w:.2f}, "
              f"res_r2={r2_score(y,pr):.4f} phy_r2={r2_score(y,pp):.4f} blend_r2={best_r2:.4f}")

    avg_w_res = np.mean([f["w_res"] for f in fold_results])
    avg_r2_res = np.mean([f["r2_res"] for f in fold_results])
    avg_r2_phy = np.mean([f["r2_phy"] for f in fold_results])
    avg_r2_blend = np.mean([f["r2_blend"] for f in fold_results])
    print(f"\n  Average: w_res={avg_w_res:.3f}, "
          f"res_r2={avg_r2_res:.4f}, phy_r2={avg_r2_phy:.4f}, blend_r2={avg_r2_blend:.4f}")
    return {"w_res": round(avg_w_res, 3), "w_phy": round(1-avg_w_res, 3),
            "r2_res": round(avg_r2_res, 4), "r2_phy": round(avg_r2_phy, 4),
            "r2_blend": round(avg_r2_blend, 4)}, fold_results

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
    rf_res, feat_names = train_rf_residual(data, bias_table)

    # ---- 3. Physical-embedding RF ----
    print(f"\n{'='*60}")
    print(f"  Phase 3: Physical-embedding RF")
    print(f"{'='*60}")
    rf_phy, phy_feat_names = train_physical_rf(data, bias_table)

    # ---- 4. Direct models ----
    print(f"\n{'='*60}")
    print(f"  Phase 4: Direct multi-step models")
    print(f"{'='*60}")
    direct_models, direct_results = train_direct_models(data)

    # ---- 5. Dual-RF blend weights ----
    print(f"\n{'='*60}")
    print(f"  Phase 5: Dual-RF blend weight via CV")
    print(f"{'='*60}")
    blend_w, fold_results = find_ensemble_weights(data, bias_table, rf_res, rf_phy, direct_models)

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

    with open(os.path.join(model_dir, "rf_residual.pkl"), "wb") as f:
        pickle.dump(rf_res, f)
    with open(os.path.join(model_dir, "rf_physical.pkl"), "wb") as f:
        pickle.dump(rf_phy, f)

    for k, v in direct_models.items():
        with open(os.path.join(model_dir, f"{k}.pkl"), "wb") as f:
            pickle.dump(v, f)

    train_summary = {
        "bias_table": bias_table,
        "rf_residual_feature_names": feat_names,
        "rf_physical_feature_names": phy_feat_names,
        "direct_models": {k: direct_results[int(k.split("_H")[1])] for k in direct_models},
        "dual_rf_blend": blend_w,
        "fold_results": fold_results,
        "baseline": {"lag1": {"r2": round(r2_lag1, 4), "rmse": round(rmse_lag1, 4)},
                     "cstr_1step": {"r2": round(r2_1s, 4), "rmse": round(rmse_1s, 4)},
                     "dual_rf_blend_cv": {"r2": round(blend_w["r2_blend"], 4)}},
    }
    with open(os.path.join(model_dir, "train_summary.json"), "w") as f:
        json.dump(train_summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Models saved to {model_dir}/")
    print(f"  Train summary saved.")
    print(f"\n[DONE]")

if __name__ == "__main__":
    main()
