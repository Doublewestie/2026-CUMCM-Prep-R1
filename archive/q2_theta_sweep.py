"""
q2_theta_sweep.py — Optimal FILT threshold sweep for Q2 two-zone model
"""
import os, numpy as np, pandas as pd, warnings
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data', '2025')

# === 1. Load data ===
data_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
data_dir = os.path.join(DATA_DIR, data_dirs[0])
FILES = sorted([f for f in os.listdir(data_dir) if f.endswith('.xlsx')])
RENAME = {
    'RIVER LEVEL':'RIVER_LEVEL','R/W FLOW':'RW_FLOW','R/W NTU':'RW_NTU',
    'R/W CLR':'RW_CLR','R/W PH':'RW_PH','FILT. NTU':'FILT_NTU',
    'C/W WELL LEVEL':'CW_WELL_LEVEL','T/W FLOW':'TW_FLOW',
    'ALUM':'ALUM','NTU':'NTU','CLR':'CLR','PH':'PH',
}
NUM_COLS = ['RIVER_LEVEL','RW_FLOW','RW_NTU','RW_CLR','RW_PH',
            'FILT_NTU','CW_WELL_LEVEL','TW_FLOW','ALUM','NTU','CLR','PH']

dfs = []
for f in FILES:
    fp = os.path.join(data_dir, f)
    df = pd.read_excel(fp, skiprows=1 if 'Jan' in f else 0)
    df.rename(columns={k:v for k,v in RENAME.items() if k in df.columns}, inplace=True)
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    dfs.append(df)
data = pd.concat(dfs, ignore_index=True)
data = data.dropna(subset=['FILT_NTU']).reset_index(drop=True)

# === 2. Features ===
data['eta_coag'] = np.clip((data['RW_NTU'] - data['FILT_NTU']) / (data['RW_NTU'] + 1e-6), 0.5, 1.0)
data['RIVER_HI'] = (data['RIVER_LEVEL'] > 5.5).astype(int)
data['NTU_HI']   = (data['RW_NTU'] > 80).astype(int)
data['DANGER']   = ((data['RIVER_LEVEL'] > 5.5) & (data['RW_NTU'] > 80)).astype(int)
data['R_x_N']    = data['RIVER_LEVEL'] * data['RW_NTU'] / 100.0

for lag in [1,2,3,4,5,6]:
    data[f'FILT_lag{lag}'] = data['FILT_NTU'].shift(lag)

data = data.dropna(subset=['FILT_lag1','FILT_lag2','FILT_lag3',
                            'FILT_lag4','FILT_lag5','FILT_lag6']).reset_index(drop=True)
n = len(data)

# Fill any NaN
for c in data.columns:
    if data[c].dtype in [np.float64, np.int64]:
        data[c].fillna(data[c].median() if data[c].notna().any() else 0, inplace=True)

# Fill any NaN in FILT too
FILT = np.nan_to_num(data['FILT_NTU'].values, nan=0.0)

# Build feature matrices (numpy arrays, position-indexed)
ar_cols = ['FILT_lag1','FILT_lag2','FILT_lag3','FILT_lag4','FILT_lag5','FILT_lag6']
enh_cols = ar_cols + ['eta_coag','RIVER_LEVEL','RW_NTU','RW_CLR',
                       'RIVER_HI','NTU_HI','DANGER','R_x_N']

# Fill NaN in feature matrices explicitly
for c in ar_cols + enh_cols:
    if data[c].isna().any():
        data[c].fillna(0.0, inplace=True)

X_ar = np.nan_to_num(data[ar_cols].values, nan=0.0)
X_enh = np.nan_to_num(data[enh_cols].values, nan=0.0)

print(f"Loaded: {n} rows, theta sweep starting...\n")

# === 3. Sweep ===
theta_range = np.arange(0.03, 0.31, 0.01)
N_SPLITS = 5
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

results = []

for theta in theta_range:
    # Accumulate predictions across all folds
    pred_A_all = np.zeros(n)
    pred_B_all = np.zeros(n)
    pred_C_all = np.zeros(n)

    for tr, va in tscv.split(data):
        tr_filt = FILT[tr]
        va_filt = FILT[va]
        tr_X_ar = X_ar[tr]; va_X_ar = X_ar[va]
        tr_X_enh = X_enh[tr]; va_X_enh = X_enh[va]

        # Split training data by theta
        tr_comfort = tr_filt <= theta
        tr_stress = tr_filt > theta

        # Split validation data by theta
        va_comfort = va_filt <= theta
        va_stress = va_filt > theta

        # ---- Model A: baseline ----
        comfort_mean = np.mean(tr_filt)  # ALWAYS safe fallback
        # Stress: AR(6) on all training data
        ar_A = Ridge(alpha=1.0).fit(tr_X_ar, tr_filt)
        pred_A = np.zeros(len(va))
        pred_A[va_comfort] = comfort_mean
        if va_stress.sum() > 0:
            pred_A[va_stress] = ar_A.predict(va_X_ar[va_stress])
        pred_A_all[va] = pred_A

        # ---- Model B: pure AR with zone-specific training ----
        pred_B = np.zeros(len(va))
        if va_comfort.sum() > 0:
            if tr_comfort.sum() > 0:
                pred_B[va_comfort] = np.mean(tr_filt[tr_comfort])
            else:
                pred_B[va_comfort] = comfort_mean
        if va_stress.sum() > 0:
            if tr_stress.sum() > 20:
                ar_stress = Ridge(alpha=1.0).fit(tr_X_ar[tr_stress], tr_filt[tr_stress])
                pred_B[va_stress] = ar_stress.predict(va_X_ar[va_stress])
            elif tr_stress.sum() > 0:
                pred_B[va_stress] = np.mean(tr_filt[tr_stress])
            else:
                pred_B[va_stress] = comfort_mean
        pred_B_all[va] = pred_B

        # ---- Model C: enhanced stress zone ----
        pred_C = np.zeros(len(va))
        pred_C[va_comfort] = pred_B[va_comfort]  # same comfort zone as B
        if va_stress.sum() > 0:
            if tr_stress.sum() > 20:
                enh_model = Ridge(alpha=1.0).fit(tr_X_enh[tr_stress], tr_filt[tr_stress])
                pred_C[va_stress] = enh_model.predict(va_X_enh[va_stress])
            elif tr_stress.sum() > 0:
                pred_C[va_stress] = np.mean(tr_filt[tr_stress])
            else:
                pred_C[va_stress] = comfort_mean
        pred_C_all[va] = pred_C

    # After all folds, compute metrics
    comfort_n = (FILT <= theta).sum()
    stress_n = (FILT > theta).sum()

    for model_name, preds in [('A_baseline', pred_A_all), ('B_pure_ar', pred_B_all), ('C_enhanced', pred_C_all)]:
        rmse = np.sqrt(mean_squared_error(FILT, preds))
        r2 = r2_score(FILT, preds)
        results.append({
            'theta': theta, 'model': model_name,
            'comfort_n': comfort_n, 'stress_n': stress_n,
            'rmse': rmse, 'r2': r2
        })

# === 4. Print results ===
print(f"{'theta':>6} {'comfort':>8} {'stress':>7} || {'A_baseline':>12} || {'B_pure_ar':>12} || {'C_enhanced':>12}")
print(f"{'':>6} {'n':>8} {'n':>7} || {'rmse':>6} {'r2':>6} || {'rmse':>6} {'r2':>6} || {'rmse':>6} {'r2':>6}")
print("-" * 82)

best_rmse = float('inf')
best_theta = None

for theta in theta_range:
    rows = [r for r in results if abs(r['theta'] - theta) < 0.005]
    a, b, c = rows[0], rows[1], rows[2]
    if c['rmse'] < best_rmse:
        best_rmse = c['rmse']
        best_theta = theta
    mark = ' ***' if c['rmse'] < 0.36 else ''
    print(f"{theta:>6.2f} {a['comfort_n']:>8d} {a['stress_n']:>7d} || {a['rmse']:>6.4f} {a['r2']:>6.4f} || {b['rmse']:>6.4f} {b['r2']:>6.4f} || {c['rmse']:>6.4f} {c['r2']:>6.4f}{mark}")

# Summary
print()
print("=" * 60)
print(f"  Best theta = {best_theta:.2f},  RMSE(C) = {best_rmse:.4f}")
print(f"  Global AR(6) reference: RMSE=0.3650, R2=0.519")
print()

# Show optimal theta result
best_rows = [r for r in results if abs(r['theta'] - best_theta) < 0.005]
for r in best_rows:
    print(f"  {r['model']:<15}  RMSE={r['rmse']:.4f}  R2={r['r2']:.4f}")

df_results = pd.DataFrame(results)
out_path = os.path.join(BASE, 'output', 'q2_theta_sweep.csv')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df_results.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f"\n  Saved to {out_path}")
