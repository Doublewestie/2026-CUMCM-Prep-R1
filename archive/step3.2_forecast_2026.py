"""
step3.2_forecast_2026.py — Q3 2026 forecast via physical-embedding RF
=====================================================================
Loads bias_table + rf_physical -> FILT reconstruction -> 2h prediction -> 1h interp
"""

import numpy as np, pandas as pd, os, json, pickle, warnings
from scipy.interpolate import CubicSpline
from step0_config import DATA_DIR_2026 as DATA_2026
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
CLEAN_CSV = os.path.join(OUTPUT_DIR, "clean_data.csv")

T1_THR, T2_THR = 0.05, 0.15; A_T1, A_T2, A_T3 = 400, 250, 30
Q_HOURS = list(range(7, 20)); HOURS_2H = [7, 9, 11, 13, 15, 17, 19]
N_MC = 200

def load_models():
    with open(os.path.join(MODEL_DIR, "bias_table.json")) as f:
        bt = json.load(f)
    bt = {int(k): {int(kk): vv for kk, vv in v.items()} for k, v in bt.items()}
    with open(os.path.join(MODEL_DIR, "rf_physical.pkl"), "rb") as f:
        rf = pickle.load(f)
    return bt, rf

def load_feb():
    fp = os.path.join(DATA_2026, "2026年2月.xls")
    df = pd.read_excel(fp, engine="xlrd")
    df.columns = [c.strip().replace(".", "_").replace(" ", "_") for c in df.columns]
    rn = {"FILT__NTU": "FILT_NTU", "C/W_WELL_LEVEL": "CW_WELL_LEVEL",
          "T/W_FLOW": "TW_FLOW", "R/W_NTU": "RW_NTU"}
    df.rename(columns={k: v for k, v in rn.items() if k in df.columns}, inplace=True)
    df["hour"] = (pd.to_numeric(df["TIME"], errors="coerce").fillna(700).astype(int) // 100)
    return df.sort_values("hour").reset_index(drop=True)

def load_2025_date(td):
    df = pd.read_csv(CLEAN_CSV)
    df["DATE"] = pd.to_datetime(df["DATE"])
    m = df["DATE"].dt.strftime("%Y-%m-%d") == f"2025-{td}"
    if m.sum() == 0: return None
    s = df[m].copy()
    s["hour"] = (pd.to_numeric(s["TIME"], errors="coerce").fillna(700).astype(int) // 100)
    return s.sort_values("hour")

def get_tier(ft): return 1 if ft <= T1_THR else (2 if ft <= T2_THR else 3)
def cstr_2h(p, ft, cw, q, bias=0):
    t = get_tier(ft); A0 = [A_T1, A_T2, A_T3][t-1]
    th = max(A0 * max(cw, 0.1) / max(q, 1.0), 0.02)
    b = np.clip(np.exp(-2.0 / th), 0.001, 0.999)
    return np.clip(b * p + (1.0 - b) * ft + bias, 0, None)

def reconstruct_filt(feb_df, dl):
    h2 = feb_df["hour"].values.astype(float)
    if dl == "Feb1":
        f2 = feb_df["FILT_NTU"].values.astype(float)
    else:
        md = "02-10" if dl == "Feb10" else "02-20"
        d25 = load_2025_date(md)
        if d25 is not None:
            b = np.nanmean(feb_df["FILT_NTU"].values - d25["FILT_NTU"].values)
            f2 = np.clip(d25["FILT_NTU"].values + (0 if np.isnan(b) else b), 0.01, None)
        else:
            f2 = feb_df["FILT_NTU"].values.astype(float)
    cs = CubicSpline(HOURS_2H, np.array([f2[list(h2).index(float(h))] for h in HOURS_2H]), bc_type="natural")
    return np.clip(cs(Q_HOURS), 0, None)

def predict_2h(bt, rf, f2h, c2h, q2h, init, dc, ds):
    n = len(f2h)
    cs = np.zeros(n); cs[0] = init
    for i in range(1, n):
        t = get_tier(f2h[i]); cs[i] = cstr_2h(cs[i-1], f2h[i], c2h[i], q2h[i], bt[t].get(i, 0))
    phy_feats = np.array([[i, get_tier(f2h[i]), f2h[i], c2h[i], q2h[i], cs[i], cs[max(0,i-1)],
                           HOURS_2H[i], np.cos(2*np.pi*HOURS_2H[i]/24),
                           np.sin(2*np.pi*HOURS_2H[i]/24), dc, ds,
                           np.mean(f2h[:i+1]) if i > 0 else f2h[0]] for i in range(n)])
    pe = np.clip(rf.predict(phy_feats), 0, None)
    mc = np.zeros((N_MC, n)); mc[:, 0] = np.clip(init + np.random.normal(0, 0.08, N_MC), 0.01, None)
    for i in range(1, n):
        mc[:, i] = 0.9 * mc[:, i-1] + 0.1 * f2h[i] + bt[get_tier(f2h[i])].get(i, 0)
        mc[:, i] = np.clip(mc[:, i] + np.random.normal(0, {1:0.08,2:0.12,3:0.20}.get(get_tier(f2h[i]),0.1), N_MC), 0, None)
    return {"ensemble": pe, "base": cs, "p50": np.median(mc, 0),
            "p5": np.percentile(mc, 5, 0), "p95": np.percentile(mc, 95, 0)}

def main():
    bt, rf = load_models()
    feb_df = load_feb()
    jan_df = pd.read_excel(os.path.join(DATA_2026, "2026年1月.xls"), engine="xlrd")
    jan_ntu_mean = jan_df["NTU"].mean()
    h2 = feb_df["hour"].values.astype(float)
    cw2 = np.array([feb_df["CW_WELL_LEVEL"].values.astype(float)[list(h2).index(float(h))] for h in HOURS_2H])
    q2 = np.array([feb_df["TW_FLOW"].values.astype(float)[list(h2).index(float(h))] for h in HOURS_2H])
    all_rows = []
    for dl, dt in [("Feb1","2026-02-01"),("Feb10","2026-02-10"),("Feb20","2026-02-20")]:
        f1h = reconstruct_filt(feb_df, dl)
        f2h_vals = np.array([f1h[Q_HOURS.index(h)] for h in HOURS_2H])
        init = jan_ntu_mean if dl == "Feb1" else load_2025_date("02-10" if dl=="Feb10" else "02-20")["NTU"].mean()
        doy = {"Feb1":1,"Feb10":10,"Feb20":20}[dl]
        dc, ds = np.cos(2*np.pi*doy/365), np.sin(2*np.pi*doy/365)
        r2 = predict_2h(bt, rf, f2h_vals, cw2, q2, init, dc, ds)
        r = {}
        for k in ["base","ensemble","p50","p5","p95"]:
            r[k] = np.clip(CubicSpline(HOURS_2H, r2[k], bc_type="natural")(Q_HOURS), 0, None)
        print(f"\n=== {dl} init={init:.4f} ===")
        for i in range(len(Q_HOURS)):
            print(f"  {Q_HOURS[i]:02d}:00 f={f1h[i]:.4f} ens={r['ensemble'][i]:.4f} [{r['p5'][i]:.4f},{r['p95'][i]:.4f}]")
            all_rows.append({"date": dt, "time": f"{Q_HOURS[i]:02d}:00", "FILT": round(f1h[i], 4),
                             "NTU_ensemble": round(r["ensemble"][i], 4),
                             "NTU_P5": round(r["p5"][i], 4), "NTU_P95": round(r["p95"][i], 4)})
    pd.DataFrame(all_rows).to_csv(os.path.join(OUTPUT_DIR, "q3_final_predictions.csv"), index=False)
    print(f"\nSaved.")

if __name__ == "__main__":
    main()
