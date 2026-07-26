"""
step3.3_validation.py — 5-fold TS-CV validation of all methods on 2025
======================================================================
"""

import numpy as np, pandas as pd, os, json, pickle, warnings
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
CLEAN_CSV = os.path.join(OUTPUT_DIR, "clean_data.csv")
T1_THR, T2_THR = 0.05, 0.15
A_T1, A_T2, A_T3 = 400, 250, 30
HOURS_2H = [7, 9, 11, 13, 15, 17, 19]

def load_models():
    with open(os.path.join(MODEL_DIR, "bias_table.json")) as f:
        bt = json.load(f)
    bt = {int(k): {int(kk): vv for kk, vv in v.items()} for k, v in bt.items()}
    with open(os.path.join(MODEL_DIR, "rf_model.pkl"), "rb") as f:
        rf = pickle.load(f)
    dm = {}
    for k in ["Ridge_H1"]:
        fp = os.path.join(MODEL_DIR, f"{k}.pkl")
        if os.path.exists(fp):
            with open(fp, "rb") as f:
                dm[k] = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "train_summary.json")) as f:
        s = json.load(f)
    return bt, rf, dm, s

def get_tier(ft):
    return 1 if ft <= T1_THR else (2 if ft <= T2_THR else 3)

def cstr_2h(p, ft, cw, q, bias=0):
    t = get_tier(ft); A0 = [A_T1, A_T2, A_T3][t-1]
    th = max(A0 * max(cw, 0.1) / max(q, 1.0), 0.02)
    b = np.clip(np.exp(-2.0 / th), 0.001, 0.999)
    return np.clip(b * p + (1.0 - b) * ft + bias, 0, None)

def main():
    bt, rf, dm, s = load_models()
    w = s["ensemble_weights"]
    df = pd.read_csv(CLEAN_CSV)
    ntu = df["NTU"].values.astype(float); filt = df["FILT_NTU"].values.astype(float)
    cw = df["CW_WELL_LEVEL"].values.astype(float); q = df["TW_FLOW"].values.astype(float)
    date = pd.to_datetime(df["DATE"]); doy = date.dt.dayofyear.values
    n = len(ntu); tscv = TimeSeriesSplit(n_splits=5); fold_results = []
    for fi, (tr, va) in enumerate(tscv.split(np.arange(n))):
        va_start = max(va[0], 11); va_end = va[-1]
        meth = {"base": [], "rf": [], "direct": [], "ensemble": []}; targs = []
        for ds in range(va_start, va_end - 12, 12):
            f_day = filt[ds:ds+12]; cw_day = cw[ds:ds+12]; q_day = q[ds:ds+12]
            ntu_day = ntu[ds:ds+12]
            qi = [3, 4, 5, 6, 7, 8, 9]
            f_q3 = np.array([f_day[i] for i in qi]); cw_q3 = np.array([cw_day[i] for i in qi])
            q_q3 = np.array([q_day[i] for i in qi]); t_q3 = np.array([ntu_day[i] for i in qi])
            if np.any(np.isnan(t_q3)): continue
            init = np.mean(ntu[tr][~np.isnan(ntu[tr])])
            dc, ds = np.cos(2*np.pi*doy[ds]/365), np.sin(2*np.pi*doy[ds]/365)
            pb = np.zeros(7); pb[0] = init
            for i in range(1, 7):
                t = get_tier(f_q3[i]); b = bt.get(t, {}).get(i, 0)
                pb[i] = cstr_2h(pb[i-1], f_q3[i], cw_q3[i], q_q3[i], b)
            meth["base"].extend(pb.tolist())
            pr = np.zeros(7); pr[0] = init
            for i in range(1, 7):
                t = get_tier(f_q3[i]); b = bt.get(t, {}).get(i, 0)
                pr[i] = cstr_2h(pr[i-1], f_q3[i], cw_q3[i], q_q3[i], b)
            rff = np.array([[i, get_tier(f_q3[i]), f_q3[i], cw_q3[i], q_q3[i],
                             pr[i-1], HOURS_2H[i], np.cos(2*np.pi*HOURS_2H[i]/24),
                             np.sin(2*np.pi*HOURS_2H[i]/24), dc, ds] for i in range(1, 7)])
            pr[1:] = np.clip(pr[1:] + rf.predict(rff), 0, None)
            meth["rf"].extend(pr.tolist())
            pd_ = pb.copy()
            if "Ridge_H1" in dm:
                fts = np.array([[pb[i], f_q3[i], f_q3[i], cw_q3[i], q_q3[i],
                                 HOURS_2H[i], np.cos(2*np.pi*HOURS_2H[i]/24),
                                 np.sin(2*np.pi*HOURS_2H[i]/24), dc, ds] for i in range(7)])
                pd_ = np.clip(dm["Ridge_H1"].predict(fts), 0, None)
            meth["direct"].extend(pd_.tolist())
            meth["ensemble"].extend((w["w_base"]*pb + w["w_rf"]*pr + w["w_direct"]*pd_).tolist())
            targs.extend(t_q3.tolist())
        targs = np.array(targs); v = ~np.isnan(targs) & (targs > 0)
        fm = {"fold": fi, "n": v.sum()}
        for name, vals in meth.items():
            a, b2 = np.array(vals)[v], targs[v]
            fm[name] = {"r2": r2_score(b2, a), "rmse": float(np.sqrt(mean_squared_error(b2, a)))}
        fold_results.append(fm)

    print(f"CV Results:\n{'Method':<15} {'R2':<20} {'RMSE':<20}")
    for name in ["base", "rf", "direct", "ensemble"]:
        r2s = [f[name]["r2"] for f in fold_results]
        rms = [f[name]["rmse"] for f in fold_results]
        print(f"{name:<15} {np.mean(r2s):.4f}+-{np.std(r2s):.4f}  {np.mean(rms):.4f}+-{np.std(rms):.4f}")
    pred_1s = np.zeros(n); pred_1s[0] = ntu[0]
    for t in range(1, n):
        pred_1s[t] = cstr_2h(ntu[t-1], filt[t], cw[t], q[t])
    v = ~np.isnan(ntu); nv = ntu[v]; pv = pred_1s[v]
    print(f"{'one-step CSTR':<15} {r2_score(nv, pv):.4f}  {np.sqrt(mean_squared_error(nv, pv)):.4f}")
    with open(os.path.join(MODEL_DIR, "validation_results.json"), "w") as f:
        json.dump(fold_results, f, indent=2, default=str)

if __name__ == "__main__":
    main()
