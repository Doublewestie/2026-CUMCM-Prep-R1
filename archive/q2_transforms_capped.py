"""
q2_transforms_capped.py — Quadratic + cap transforms for extreme FILT
Variants:
  A: y = x + min((x-θ)², C*(x-θ))     (linear cap)
  B: y = x + C * tanh((x-θ)² / K)     (tanh cap)
θ=0.05, test C ∈ {0.5, 1, 2, 5}
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
RENAME = {'RIVER LEVEL':'RIVER_LEVEL','R/W FLOW':'RW_FLOW','R/W NTU':'RW_NTU',
    'R/W CLR':'RW_CLR','FILT. NTU':'FILT_NTU','C/W WELL LEVEL':'CW_WELL_LEVEL',
    'T/W FLOW':'TW_FLOW','ALUM':'ALUM','NTU':'NTU'}
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
    med = data[c].median(); data[c].fillna(med if not pd.isna(med) else 0, inplace=True)

FILT = data['FILT_NTU'].values; n = len(data)
ntu_p95 = np.percentile(data['RW_NTU'].dropna(), 95)
river_p97 = np.percentile(data['RIVER_LEVEL'].dropna(), 97)
extreme = ((data['RW_NTU'] > ntu_p95) | (data['RIVER_LEVEL'] > river_p97)).astype(int)

print(f"N={n}  Normal={n-extreme.sum()}  Extreme={extreme.sum()}")

# ============================================================
# Transform definitions
# ============================================================
THETA = 0.05

# Identity (baseline)
def ident(x): return x
def inv_ident(y): return y

# Linear cap: y = x + min((x-θ)², C*(x-θ))
def make_lin_cap(C):
    def forward(x):
        excess = np.maximum(0, x - THETA)
        quad = excess ** 2
        linear = C * excess
        return x + np.where(quad < linear, quad, linear)
    def inverse(y):
        res = np.zeros_like(y)
        for i, v in enumerate(y):
            if v <= THETA:
                res[i] = v
            else:
                # Try quad branch: v = x + (x-θ)²
                x_quad = (np.sqrt(4*v + 4*THETA*THETA + 4*THETA + 1) - 2*THETA - 1) / 2
                excess = x_quad - THETA
                if excess >= 0 and excess*excess <= C*excess:
                    res[i] = x_quad
                else:
                    # Linear branch: v = x + C*(x-θ) = x*(1+C) - C*θ
                    res[i] = (v + C * THETA) / (1 + C)
        return res
    return forward, inverse

# Tanh cap: y = x + C * tanh((x-θ)² / K)
def make_tanh_cap(C, K=1.0):
    def forward(x):
        excess = np.maximum(0, x - THETA)
        return x + C * np.tanh(excess ** 2 / K)
    def inverse(y):
        res = np.zeros_like(y)
        for i, v in enumerate(y):
            if v <= THETA:
                res[i] = v
            else:
                # Numerical inverse using simple iteration
                x0 = v
                for _ in range(20):
                    ex = max(0, x0 - THETA)
                    f = x0 + C * np.tanh(ex**2 / K) - v
                    jac = 1 + C * (1 - np.tanh(ex**2 / K)**2) * (2*ex / K) if ex > 0 else 1
                    if abs(jac) < 1e-10: jac = 1
                    x0 = x0 - f / jac
                res[i] = max(0.03, x0)
        return res
    return forward, inverse

# ============================================================
# Evaluate
# ============================================================
def evaluate(name, tf, inv_func):
    dy_actual = np.diff(FILT, prepend=FILT[0])
    dy = np.diff(tf(FILT), prepend=tf(FILT[0]))
    
    # AR(6) on Δy
    X = np.zeros((n, 6))
    for lag in range(1, 7):
        X[lag:, lag-1] = dy[:-lag]
        X[:lag, lag-1] = dy[0]
    theta = np.linalg.lstsq(np.column_stack([np.ones(n), X]), dy, rcond=None)[0]
    pred_dy = np.column_stack([np.ones(n), X]) @ theta
    
    # Convert Δy_pred back to ΔFILT_pred
    y_pred = np.zeros(n)
    y_pred[0] = tf(FILT[0])
    for t in range(1, n):
        y_pred[t] = y_pred[t-1] + pred_dy[t]
    filt_pred = inv_func(y_pred)
    delta_pred = np.diff(filt_pred, prepend=filt_pred[0])
    
    # Metrics
    def metrics(y_true, y_p, mask=None):
        if mask is not None and mask.sum() < 5:
            return {'r2': np.nan, 'rmse': np.nan}
        s = slice(None) if mask is None else mask
        r2 = r2_score(y_true[s], y_p[s]) if len(y_true[s]) > 1 else 0
        rmse = np.sqrt(mean_squared_error(y_true[s], y_p[s]))
        return {'r2': r2, 'rmse': rmse}
    
    return {
        'delta_all': metrics(dy_actual, delta_pred),
        'delta_norm': metrics(dy_actual, delta_pred, extreme == 0),
        'delta_ext': metrics(dy_actual, delta_pred, extreme == 1),
        'level_ext': metrics(FILT, filt_pred, extreme == 1),
    }

# ============================================================
# Run all tests
# ============================================================
# Show transform table
print("\nTransform mapping (FILT -> y):")
print(f"{'FILT':>8}", end='')
for C in [0.5, 1, 2]:
    print(f"{'Lin'+str(C):>8}{'Tanh'+str(C):>8}", end='')
print()
for v in [0.03, 0.05, 0.08, 0.15, 0.30, 0.50, 1.0, 3.0, 5.0, 9.8]:
    print(f"{v:>8.2f}", end='')
    for C in [0.5, 1, 2]:
        print(f"{v+min(max(0,v-0.05)**2, C*max(0,v-0.05)):>8.2f}", end='')
        print(f"{v+C*np.tanh(max(0,v-0.05)**2):>8.2f}", end='')
    print()

schemes = [('Identity', ident, inv_ident)]

for C in [0.5, 1, 2, 5]:
    tf_lin, inv_lin = make_lin_cap(C)
    schemes.append((f'LinCap C={C}', tf_lin, inv_lin))
    
    tf_tanh, inv_tanh = make_tanh_cap(C)
    schemes.append((f'TanhCap C={C}', tf_tanh, inv_tanh))

print(f"\n{'='*80}")
print(f"  EVALUATION RESULTS")
print(f"{'='*80}")
print(f"{'Scheme':<18} {'Delta All':>12} {'Delta Norm':>12} {'Delta Ext':>12} {'Level Ext':>12}")
print(f"{'':<18} {'R2':>6} {'RMSE':>6} {'R2':>6} {'RMSE':>6} {'R2':>6} {'RMSE':>6} {'R2':>6} {'RMSE':>6}")
print(f"{'-'*80}")

for name, tf, inv in schemes:
    r = evaluate(name, tf, inv)
    print(f"{name:<18} "
          f"{r['delta_all']['r2']:>6.4f} {r['delta_all']['rmse']:>6.4f} "
          f"{r['delta_norm']['r2']:>6.4f} {r['delta_norm']['rmse']:>6.4f} "
          f"{r['delta_ext']['r2']:>6.4f} {r['delta_ext']['rmse']:>6.4f} "
          f"{r['level_ext']['r2']:>6.4f} {r['level_ext']['rmse']:>6.4f}")
