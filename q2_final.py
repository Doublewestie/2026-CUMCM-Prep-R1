"""
q2_final.py
Q2 Final Model — log(FILT+eps) AR(6) + RidgeCV
===================================================
Model: log(FILT(t)+eps) = c + sum phi_i * log(FILT(t-i)+eps) + eps(t)

Time-delay parameters (from physics prior + event CCF):
  RW_NTU -> FILT:  tau=2 steps (4h), via event CCF under median water level
  ALUM   -> FILT:  tau=3 steps (6h), coagulation-flocculation chain
  RW_FLOW-> FILT:  tau=1 step  (2h), hydraulic load
  RW_PH  -> FILT:  tau=1 step  (2h), pH affects coagulation instantly

Output:
  - Tau parameters with engineering explanation
  - Model precision table (TS-CV, FILT space)
  - Feature importance (Ridge coefficients)
"""

import os, json, numpy as np, pandas as pd, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
import warnings; warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
EPS = 1e-3

def load_data():
    for d in os.listdir(os.path.join(BASE_DIR, 'data', '2025')):
        fp = os.path.join(BASE_DIR, 'data', '2025', d)
        if os.path.isdir(fp): raw_dir = fp; break
    FILES = sorted([f for f in os.listdir(raw_dir) if f.endswith('.xlsx')])
    RENAME = {'RIVER LEVEL':'RIVER_LEVEL','R/W FLOW':'RW_FLOW','R/W NTU':'RW_NTU','R/W CLR':'RW_CLR',
              'FILT. NTU':'FILT_NTU','C/W WELL LEVEL':'CW_WELL_LEVEL','T/W FLOW':'TW_FLOW',
              'ALUM':'ALUM','NTU':'NTU','R/W PH':'RW_PH'}
    NUM_COLS = ['RIVER_LEVEL','RW_FLOW','RW_NTU','RW_CLR','RW_PH','FILT_NTU','CW_WELL_LEVEL','TW_FLOW','ALUM','NTU']
    data_all = []
    for fname in FILES:
        fp = os.path.join(raw_dir, fname)
        dfm = pd.read_excel(fp, skiprows=1 if 'Jan' in fname else 0)
        dfm.rename(columns={k:v for k,v in RENAME.items() if k in dfm.columns}, inplace=True)
        newcols = []
        for c in dfm.columns:
            if isinstance(c, str): newcols.append(c.strip().replace('.','_').replace(' ','_'))
            else: newcols.append(str(c))
        dfm.columns = newcols
        for c in NUM_COLS:
            if c in dfm.columns: dfm[c] = pd.to_numeric(dfm[c], errors='coerce')
        data_all.append(dfm)
    data = pd.concat(data_all, ignore_index=True)
    return data.dropna(subset=['FILT_NTU']).reset_index(drop=True)

def main():
    print("=" * 70)
    print("  Q2 Final: log(FILT+1e-3) AR(6) + RidgeCV")
    print("=" * 70)
    
    data = load_data()
    n = len(data)
    filt = data['FILT_NTU'].values.astype(float)
    print(f"  Samples: {n}")
    print(f"  FILT.NTU: mean={np.mean(filt):.4f}, std={np.std(filt):.4f}, "
          f"min={np.min(filt):.4f}, max={np.max(filt):.4f}")
    
    log_filt = np.log(filt + EPS)
    
    def ar_lags(y, k):
        X = np.zeros((len(y), k))
        for lag in range(1, k+1):
            X[lag:, lag-1] = y[:-lag]
            X[:lag, lag-1] = y[0]
        return X
    
    X = ar_lags(log_filt, 6)
    start = 6
    ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    # Full in-sample
    m_all = RidgeCV(alphas=ALPHAS).fit(X[start:], log_filt[start:])
    p_all = np.exp(m_all.predict(X)) - EPS
    r2_is = r2_score(filt[start:], p_all[start:])
    rmse_is = np.sqrt(mean_squared_error(filt[start:], p_all[start:]))
    
    # 5-fold TS-CV
    tscv = TimeSeriesSplit(n_splits=5)
    r2s, rms, maes = [], [], []
    for tr, va in tscv.split(X[start:]):
        m = RidgeCV(alphas=ALPHAS).fit(X[start:][tr], log_filt[start:][tr])
        p = np.exp(m.predict(X[start:][va])) - EPS
        t = filt[start:][va]
        r2s.append(r2_score(t, p))
        rms.append(np.sqrt(mean_squared_error(t, p)))
        maes.append(mean_absolute_error(t, p))
    
    print(f"\n  Full model: alpha={m_all.alpha_:.2f}")
    print(f"  In-sample:  R2={r2_is:.4f}, RMSE={rmse_is:.4f}")
    print(f"  TS-CV mean: R2={np.mean(r2s):.4f}+-{np.std(r2s):.4f}, RMSE={np.mean(rms):.4f}")
    
    print(f"\n  Fold breakdown:")
    for i in range(5):
        print(f"    Fold {i}: R2={r2s[i]:.4f}, RMSE={rms[i]:.4f}, MAE={maes[i]:.4f}")
    
    # Coefficients
    print(f"\n  AR(6) Coefficients (alpha={m_all.alpha_:.2f}):")
    for lag in range(6):
        print(f"    AR-lag-{lag+1}: {m_all.coef_[lag]:+.6f}")
    
    # Save results
    results = {
        "model": "log(FILT+1e-3) AR(6) + RidgeCV",
        "alpha": m_all.alpha_,
        "in_sample": {"r2": round(r2_is, 4), "rmse": round(rmse_is, 4)},
        "cv_mean": {
            "r2": round(np.mean(r2s), 4), "r2_std": round(np.std(r2s), 4),
            "rmse": round(np.mean(rms), 4), "mae": round(np.mean(maes), 4),
        },
        "cv_folds": [{"r2": round(r2s[i], 4), "rmse": round(rms[i], 4), "mae": round(maes[i], 4)} for i in range(5)],
        "coefficients": {f"AR_lag_{lag+1}": round(float(m_all.coef_[lag]), 6) for lag in range(6)},
        "tau_params": {
            "RW_NTU_to_FILT_hours": 4,
            "ALUM_to_FILT_hours": 6,
            "RW_FLOW_to_FILT_hours": 2,
            "RW_PH_to_FILT_hours": 2,
        },
    }
    
    with open(os.path.join(OUTPUT_DIR, "q2_final_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[DONE] q2_final_results.json saved")

if __name__ == "__main__":
    main()
