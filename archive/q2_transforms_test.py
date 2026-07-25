"""
q2_transforms_test.py — Piecewise quadratic transforms for extreme FILT
Clean version: predict ΔFILT (transformed), evaluate on ΔFILT (original).
"""
import os, numpy as np, pandas as pd, warnings
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data', '2025')
data_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
data_dir = os.path.join(DATA_DIR, data_dirs[0])
FILES = sorted([f for f in os.listdir(data_dir) if f.endswith('.xlsx')])
RENAME = {'RIVER LEVEL':'RIVER_LEVEL','R/W FLOW':'RW_FLOW','R/W NTU':'RW_NTU','R/W CLR':'RW_CLR','FILT. NTU':'FILT_NTU','C/W WELL LEVEL':'CW_WELL_LEVEL','T/W FLOW':'TW_FLOW','ALUM':'ALUM','NTU':'NTU'}
NUM_COLS = ['RIVER_LEVEL','RW_FLOW','RW_NTU','RW_CLR','FILT_NTU','CW_WELL_LEVEL','TW_FLOW','ALUM','NTU']
dfs = []
for f in FILES:
    fp = os.path.join(data_dir, f)
    df = pd.read_excel(fp, skiprows=1 if 'Jan' in f else 0)
    df.rename(columns={k:v for k,v in RENAME.items() if k in df.columns}, inplace=True)
    for c in NUM_COLS:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    dfs.append(df)
data = pd.concat(dfs, ignore_index=True)
data = data.dropna(subset=['FILT_NTU']).reset_index(drop=True)
for c in ['RIVER_LEVEL','RW_NTU','RW_CLR','RW_FLOW','ALUM']:
    med = data[c].median();
    data[c].fillna(med if not pd.isna(med) else 0, inplace=True)

FILT = data['FILT_NTU'].values
n = len(data)
ntu_p95 = np.percentile(data['RW_NTU'].dropna(), 95)
river_p97 = np.percentile(data['RIVER_LEVEL'].dropna(), 97)
extreme = ((data['RW_NTU'] > ntu_p95) | (data['RIVER_LEVEL'] > river_p97)).astype(int)
print(f"N={n}  Normal={n-extreme.sum()}  Extreme={extreme.sum()}")

# ============================================================
# Transforms + mapping table
# ============================================================
print("\nTransform mapping:")
print(f"{'FILT':>8} {'ident':>8} {'QuadA':>8} {'QuadB':>8}")
for v in [0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.30, 0.50, 1.0, 3.0, 5.0, 9.80]:
    a = v + max(0, v - 0.05)**2
    b = v + max(0, v - 0.08)**2
    print(f"{v:>8.2f} {v:>8.2f} {a:>8.2f} {b:>8.2f}")

# ============================================================
# Evaluate: train AR(6) on ΔFILT (transformed), test on ΔFILT (original)
# ============================================================
def compute_dy(FILT_raw, tf_func):
    """Transform then compute 1st difference"""
    y = tf_func(FILT_raw)
    return np.diff(y, prepend=y[0])

def compute_metrics(y_true, y_pred, mask=None):
    if mask is not None and mask.sum() < 5:
        return {'r2': np.nan, 'rmse': np.nan}
    s = slice(None) if mask is None else mask
    r2 = r2_score(y_true[s], y_pred[s]) if len(y_true[s]) > 1 else np.nan
    rmse = np.sqrt(mean_squared_error(y_true[s], y_pred[s]))
    return {'r2': r2, 'rmse': rmse, 'n': s if isinstance(s, slice) else mask.sum()}

def evaluate(name, tf_func, to_filt_func=None):
    """Full CV + in-sample evaluation"""
    dy = compute_dy(FILT, tf_func)
    dy_actual = np.diff(FILT, prepend=FILT[0])
    
    # Build lag features: 6 lags of Δy (transformed)
    X = np.zeros((n, 6))
    for lag in range(1, 7):
        X[lag:, lag-1] = dy[:-lag]
        X[:lag, lag-1] = dy[0]
    
    # In-sample
    theta = np.linalg.lstsq(np.column_stack([np.ones(n), X]), dy, rcond=None)[0]
    pred_dy = np.column_stack([np.ones(n), X]) @ theta
    
    # Convert Δy_pred to ΔFILT_pred for evaluation
    # pred_dy is Δy, we need to convert back
    # If no inversion needed (identity), pred_dy is already ΔFILT
    if to_filt_func is None:
        delta_pred = pred_dy
    else:
        # Approximate: use derivative of transform
        # For inverse: dFILT/dy ≈ 1 / (1 + 2*max(0, FILT-θ))
        y_pred = np.zeros(n)
        y_pred[0] = tf_func(FILT[0])
        for t in range(1, n):
            y_pred[t] = y_pred[t-1] + pred_dy[t]
        filt_approx = np.array([to_filt_func(np.array([yp]))[0] for yp in y_pred])
        delta_pred = np.diff(filt_approx, prepend=filt_approx[0])
    
    is_metrics = {
        'all': compute_metrics(dy_actual, delta_pred),
        'norm': compute_metrics(dy_actual, delta_pred, extreme == 0),
        'ext': compute_metrics(dy_actual, delta_pred, extreme == 1),
    }
    
    # Also compute level metrics for extreme zone
    if to_filt_func is not None:
        y_pred_ts = np.zeros(n)
        y_pred_ts[0] = tf_func(FILT[0])
        for t in range(1, n):
            y_pred_ts[t] = y_pred_ts[t-1] + pred_dy[t]
        filt_level = np.array([to_filt_func(np.array([yp]))[0] for yp in y_pred_ts])
        lvl_ext = compute_metrics(FILT, filt_level, extreme == 1)
    else:
        lvl_ext = compute_metrics(FILT, np.zeros(n) + np.mean(FILT), extreme == 1)
        y_pred_ts = FILT
    
    # 5-fold CV
    tscv = TimeSeriesSplit(n_splits=5)
    cv_r2_all, cv_r2_ext = [], []
    
    for tr, va in tscv.split(np.arange(n)):
        dy_tr = dy[tr]; dy_va = dy[va]
        X_tr = np.column_stack([np.ones(len(tr))] + [np.roll(dy_tr, l) for l in range(1, 7)])
        for lag in range(1, 7):
            X_tr[:lag, lag] = dy_tr[0]
        
        theta_cv = np.linalg.lstsq(X_tr, dy_tr, rcond=None)[0]
        X_va = np.column_stack([np.ones(len(va))] + [np.roll(dy_va, l) for l in range(1, 7)])
        for lag in range(1, 7):
            X_va[:lag, lag] = dy_va[0]
        
        pred_dy_va = X_va @ theta_cv
        
        # Convert to ΔFILT
        if to_filt_func is None:
            delta_va = pred_dy_va
        else:
            y_va_pred = np.zeros(len(va))
            y_va_pred[0] = tf_func(FILT[va[0]])
            for t in range(1, len(va)):
                y_va_pred[t] = y_va_pred[t-1] + pred_dy_va[t]
            filt_va = np.array([to_filt_func(np.array([yp]))[0] for yp in y_va_pred])
            delta_va = np.diff(filt_va, prepend=filt_va[0])
        
        dy_va_act = np.diff(FILT[va], prepend=FILT[va[0]])
        ext_va = extreme[va] == 1
        cv_r2_all.append(r2_score(dy_va_act, delta_va) if len(dy_va_act) > 1 else np.nan)
        cv_r2_ext.append(r2_score(dy_va_act[ext_va], delta_va[ext_va]) if ext_va.sum() > 5 else np.nan)
    
    return is_metrics, lvl_ext, cv_r2_all, cv_r2_ext

# Run all schemes
print(f"\n{'='*75}")
print(f"  IN-SAMPLE RESULTS (DeltaFILT R2)")
print(f"{'='*75}")
print(f"{'Scheme':<25} {'All':>8} {'Normal':>8} {'Extreme':>8} {'Lvl_Ext_R2':>10} {'Lvl_Ext_RMSE':>13}")
print(f"{'-'*72}")

results = []
for name, tf_func, inv_func in [
    ('Baseline (identity)', lambda x: x, None),
    ('log1p', lambda x: np.log1p(x), None),
    ('sqrt', lambda x: np.sqrt(x), None),
    ('Quad A (0.05)', lambda x: x + np.maximum(0, x-0.05)**2,
     lambda y: np.where(y <= 0.05, y, np.sqrt(y + 0.0025) - 0.45)),
    ('Quad B (0.08)', lambda x: x + np.maximum(0, x-0.08)**2,
     lambda y: np.where(y <= 0.08, y, np.sqrt(y + 0.0016) - 0.44)),
]:
    is_m, lvl_ext, cv_a, cv_e = evaluate(name, tf_func, inv_func)
    results.append((name, is_m, lvl_ext, cv_a, cv_e))
    print(f"{name:<25} {is_m['all']['r2']:>8.4f} {is_m['norm']['r2']:>8.4f} {is_m['ext']['r2']:>8.4f} {lvl_ext['r2']:>10.4f} {lvl_ext['rmse']:>13.4f}")

print(f"\n{'='*75}")
print(f"  5-FOLD CV (DeltaFILT R2)")
print(f"{'='*75}")
print(f"{'Scheme':<25} {'CV R2_all':>10} {'CV R2_ext':>10}")
print(f"{'-'*47}")
for name, is_m, lvl_ext, cv_a, cv_e in results:
    m_all = np.nanmean(cv_a) if cv_a else np.nan
    m_ext = np.nanmean([x for x in cv_e if not np.isnan(x)]) if cv_e else np.nan
    print(f"{name:<25} {m_all:>10.4f} {m_ext:>10.4f}")
