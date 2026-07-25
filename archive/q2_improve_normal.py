"""
q2_improve_normal.py — Explore improvements for normal/stable zone FILT prediction
Scheme 2: Inverse CSTR (use NTU to back-estimate FILT)
Scheme 3: Time pattern features (hour, day, season)
"""
import os, numpy as np, pandas as pd, warnings
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data', '2025')
data_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
data_dir = os.path.join(DATA_DIR, data_dirs[0])
FILES = sorted([f for f in os.listdir(data_dir) if f.endswith('.xlsx')])

dfs = []
RENAME = {'RIVER LEVEL':'RIVER_LEVEL','R/W FLOW':'RW_FLOW','R/W NTU':'RW_NTU',
    'R/W CLR':'RW_CLR','FILT. NTU':'FILT_NTU','C/W WELL LEVEL':'CW_WELL_LEVEL',
    'T/W FLOW':'TW_FLOW','ALUM':'ALUM','NTU':'NTU'}
NUM_COLS = ['RIVER_LEVEL','RW_FLOW','RW_NTU','RW_CLR','FILT_NTU',
            'CW_WELL_LEVEL','TW_FLOW','ALUM','NTU']
for f in FILES:
    fp = os.path.join(data_dir, f)
    df = pd.read_excel(fp, skiprows=1 if 'Jan' in f else 0)
    df.rename(columns={k:v for k,v in RENAME.items() if k in df.columns}, inplace=True)
    for c in NUM_COLS:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    dfs.append(df)
data = pd.concat(dfs, ignore_index=True)
data = data.dropna(subset=['FILT_NTU']).reset_index(drop=True)

# Fill NaN on raw features
for c in ['RIVER_LEVEL','RW_NTU','RW_CLR','RW_FLOW','ALUM','CW_WELL_LEVEL','TW_FLOW','NTU']:
    if c in data.columns:
        med = data[c].median()
        data[c] = data[c].fillna(med if not pd.isna(med) else 0)

# Build time features from TIME column
def parse_time(x):
    try:
        return int(x)
    except:
        return 700  # default
data['hour_int'] = data['TIME'].apply(parse_time)
data['hour'] = (data['hour_int'] // 100) % 24
data['night'] = ((data['hour'] >= 22) | (data['hour'] <= 5)).astype(int)
data['morning'] = ((data['hour'] >= 6) & (data['hour'] <= 10)).astype(int)
data['midday'] = ((data['hour'] >= 11) & (data['hour'] <= 15)).astype(int)
data['evening'] = ((data['hour'] >= 16) & (data['hour'] <= 21)).astype(int)
data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)

# Season (from month in filename path not available, use day_sin/cos from Q1 convention)
# Approximate: each file is 1 month
data['month'] = None
for f in FILES:
    months = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
              'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
    for abbr, num in months.items():
        if abbr in f:
            mask = data.index[data.index.isin(data.index)]
            break
# Simpler approach: just use the row index to estimate season?
# Actually the data is concatenated monthly, so we can map by file index
starts = []
for f in FILES:
    fp = os.path.join(DATA_DIR, data_dirs[0], f)
    df_tmp = pd.read_excel(fp, skiprows=1 if 'Jan' in f else 0)
    starts.append(len(df_tmp))

# Actually let me just create a month column by file position
cumsum = 0
data['month'] = 0
for i, f in enumerate(FILES):
    fp = os.path.join(DATA_DIR, data_dirs[0], f)
    df_tmp = pd.read_excel(fp, skiprows=1 if 'Jan' in f else 0)
    n_rows = len(df_tmp)
    data.loc[data.index[cumsum:cumsum + n_rows], 'month'] = (i % 12) + 1
    cumsum += n_rows

data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

# Now have a clean dataset with time features
FILT = data['FILT_NTU'].values
n = len(data)

# ============================================================
# SCHEME 2: Inverse CSTR
# ============================================================
print(f"{'='*80}")
print(f"  SCHEME 2: Inverse CSTR (FILT from NTU)")
print(f"{'='*80}")

# Compute β₂ from CSTR formula
# β₂(t) = exp(-2h / θ), θ = A · CW_WELL_LEVEL(t-1) / TW_FLOW(t-1)
DELTA_T = 2.0
cw = data['CW_WELL_LEVEL'].values
tw = data['TW_FLOW'].values
ntu = data['NTU'].values

# Test different A values
print(f"\n  Inverse CSTR: FILT_est(t) = (NTU(t) - b2*NTU(t-1)) / (1-b2)")
print(f"  Stable zone defined as: FILT <= 0.08")
stable = FILT <= 0.08
print(f"  Stable zone: {stable.sum()}/{n}")

for A in [30, 50, 100, 141.3, 250, 400, 30*400]:
    theta = A * cw / np.maximum(tw, 0.1)
    b2 = np.exp(-DELTA_T / np.maximum(theta, 0.01))
    ntu_lag1 = np.roll(ntu, 1); ntu_lag1[0] = ntu[0]
    filt_inv = np.zeros(n)
    for t in range(n):
        b2t = b2[t]
        if b2t >= 0.999: b2t = 0.999
        if b2t <= 0.01: b2t = 0.01
        filt_inv[t] = (ntu[t] - b2t * ntu_lag1[t]) / (1 - b2t)
    # Clip to sensible range
    filt_inv = np.clip(filt_inv, 0, 20)
    
    # Evaluate
    for zone_name, mask in [('All', slice(None)), ('Stable', stable)]:
        actual = FILT[mask]
        pred = filt_inv[mask]
        r2 = r2_score(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        if isinstance(mask, slice):
            print(f"  A={A:>5.1f}  All:            R2={r2:.4f} RMSE={rmse:.4f}")
        else:
            print(f"  A={A:>5.1f}  Stable:         R2={r2:.4f} RMSE={rmse:.4f}")

# Multi-A: use tier-specific A based on FILT level
filt_est_tier = np.zeros(n)
for i in range(n):
    if FILT[i] <= 0.05:
        A_tier = 400
    elif FILT[i] <= 0.15:
        A_tier = 250
    else:
        A_tier = 30
    theta = A_tier * cw[i] / max(tw[i], 0.1)
    b2t = np.exp(-DELTA_T / max(theta, 0.01))
    b2t = np.clip(b2t, 0.01, 0.999)
    filt_est_tier[i] = (ntu[i] - b2t * ntu_lag1[i]) / (1 - b2t)
filt_est_tier = np.clip(filt_est_tier, 0, 20)
print(f"  A=per_tier      All:            R2={r2_score(FILT, filt_est_tier):.4f} RMSE={np.sqrt(mean_squared_error(FILT, filt_est_tier)):.4f}")
print(f"  A=per_tier      Stable:         R2={r2_score(FILT[stable], filt_est_tier[stable]):.4f} RMSE={np.sqrt(mean_squared_error(FILT[stable], filt_est_tier[stable])):.4f}")

# Compare with AR(6) on FILT levels (not delta) as fair baseline
ar_lags = np.zeros((n, 6))
for lag in range(1, 7):
    ar_lags[lag:, lag-1] = FILT[:-lag]
    ar_lags[:lag, lag-1] = FILT[0]
ar_model = Ridge(alpha=1.0).fit(ar_lags, FILT)
ar_pred = ar_model.predict(ar_lags)
print(f"\n  AR(6) baseline (level prediction):")
print(f"  AR(6)  All:            R2={r2_score(FILT, ar_pred):.4f} RMSE={np.sqrt(mean_squared_error(FILT, ar_pred)):.4f}")
print(f"  AR(6)  Stable:         R2={r2_score(FILT[stable], ar_pred[stable]):.4f} RMSE={np.sqrt(mean_squared_error(FILT[stable], ar_pred[stable])):.4f}")

# Combine: use inverse CSTR when it's better, AR(6) otherwise
combined = np.where(filt_est_tier - FILT > 0.3, ar_pred,  # if inv CSTR is way off, use AR
                   np.where(FILT > 0.3, filt_est_tier, ar_pred))  # for high FILT, use inv CSTR
print(f"\n  Combined (invCSTR+AR6):")
print(f"  Combined All:       R2={r2_score(FILT, combined):.4f} RMSE={np.sqrt(mean_squared_error(FILT, combined)):.4f}")
print(f"  Combined Stable:    R2={r2_score(FILT[stable], combined[stable]):.4f} RMSE={np.sqrt(mean_squared_error(FILT[stable], combined[stable])):.4f}")

# ============================================================
# SCHEME 3: Time pattern features
# ============================================================
print(f"\n{'='*80}")
print(f"  SCHEME 3: Time pattern features")
print(f"{'='*80}")

# Check ΔFILT by hour
delta_f = np.diff(FILT, prepend=FILT[0])
print(f"\n  ΔFILT by hour of day:")
print(f"  {'Hour':>6}{'Mean':>8}{'Std':>8}{'N':>8}")
for h in sorted(data['hour'].unique()):
    mask = data['hour'] == h
    if mask.sum() < 10: continue
    d = delta_f[mask]
    print(f"{h:>6}{d.mean():>8.4f}{d.std():>8.4f}{mask.sum():>8d}")

print(f"\n  ΔFILT by time period:")
for period_name in ['night','morning','midday','evening']:
    mask = data[period_name] == 1
    d = delta_f[mask]
    print(f"  {period_name:<10} mean={d.mean():.4f} std={d.std():.4f} n={mask.sum()}")

# Check ΔFILT by month (season)
print(f"\n  ΔFILT by month:")
print(f"  {'Month':>6}{'Mean':>8}{'Std':>8}{'N':>8}")
for m in sorted(data['month'].unique()):
    mask = data['month'] == m
    if mask.sum() < 10: continue
    d = delta_f[mask]
    print(f"{m:>6}{d.mean():>8.4f}{d.std():>8.4f}{mask.sum():>8d}")

# Build time-enhanced AR model
df_model = pd.DataFrame({
    'target': delta_f,
    'f1': delta_f,  # will shift
    'f2': delta_f,
    'f3': delta_f,
    'f4': delta_f,
    'f5': delta_f,
    'f6': delta_f,
})
data['FILT_lag1'] = FILT
data['FILT_lag2'] = np.roll(FILT, 1); data['FILT_lag3'] = np.roll(FILT, 2)
data['FILT_lag4'] = np.roll(FILT, 3); data['FILT_lag5'] = np.roll(FILT, 4); data['FILT_lag6'] = np.roll(FILT, 5)
time_feats = ['hour_sin','hour_cos','month_sin','month_cos','night','morning','midday','evening']

# Baseline: AR(6) only
X_base = np.column_stack([np.ones(n)] + [np.roll(delta_f, l) for l in range(1, 7)])
for lag in range(6):
    X_base[:lag+1, lag+1] = delta_f[0]
theta_base = np.linalg.lstsq(X_base, delta_f, rcond=None)[0]
r2_base_all = r2_score(delta_f, X_base @ theta_base)

# AR(6) + time features
time_vals = data[time_feats].values
X_time = np.column_stack([np.ones(n)] + [np.roll(delta_f, l) for l in range(1, 7)] + [time_vals])
for lag in range(6):
    X_time[:lag+1, lag+1] = delta_f[0]
theta_time = np.linalg.lstsq(X_time, delta_f, rcond=None)[0]
pred_time = X_time @ theta_time
r2_time_all = r2_score(delta_f, pred_time)

print(f"\n  AR(6) baseline R2(Delta) = {r2_base_all:.4f}")
print(f"  AR(6)+time R2(Delta)     = {r2_time_all:.4f}")
print(f"  Time features improvement: {r2_time_all - r2_base_all:+.4f}")

# Per zone
for mask, name in [(stable, 'Stable'), (~stable, 'Unstable')]:
    r2_b = r2_score(delta_f[mask], (X_base @ theta_base)[mask])
    r2_t = r2_score(delta_f[mask], pred_time[mask])
    print(f"  {name}: AR6 R2={r2_b:.4f}  AR6+time R2={r2_t:.4f}  delta={r2_t-r2_b:+.4f}")

# Feature importance (Ridge coefficients for time features)
time_ridge = Ridge(alpha=1.0).fit(np.column_stack([np.roll(delta_f, l) for l in range(1, 7)] + [time_vals]), delta_f)
print(f"\n  Time feature coefficients:")
for name, coef in zip(['ar_lag1','ar_lag2','ar_lag3','ar_lag4','ar_lag5','ar_lag6'] + time_feats, 
                       time_ridge.coef_):
    print(f"  {name:<15} {coef:+.6f}")
