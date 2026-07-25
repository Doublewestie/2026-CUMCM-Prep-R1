"""
q2_balance_test.py — Test the balance detector feature from remote Q1 commit
Balance = (RIVER_LEVEL - RL_med) * (TW_FLOW - Q_med)
  > 0: same direction (stable clearwell, A=100)
  < 0: opposite (stressed clearwell, A=20)

Check if this helps Q2 tier prediction or AR(6) improvement.
"""
import os, numpy as np, pandas as pd, warnings
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score
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
for c in ['RIVER_LEVEL','RW_FLOW','RW_NTU','RW_CLR','RW_FLOW','ALUM','CW_WELL_LEVEL','TW_FLOW','NTU']:
    if c in data.columns:
        med = data[c].median()
        data[c] = data[c].fillna(med if not pd.isna(med) else 0)

FILT = data['FILT_NTU'].values; n = len(data)

# ============================================================
# 1. Balance detector definition
# ============================================================
RL_med = np.median(data['RIVER_LEVEL'].dropna())
Q_med = np.median(data['TW_FLOW'].dropna())
RL = data['RIVER_LEVEL'].values
Q = data['TW_FLOW'].values

balance = (RL - RL_med) * (Q - Q_med)
balance_sign = np.sign(balance)  # -1, 0, +1

print(f"Balance detector: RL_med={RL_med:.2f}, Q_med={Q_med:.1f}")
print(f"  Balance > 0 (same direction): {(balance>0).sum():>6d} ({((balance>0).sum()/n*100):.1f}%)")
print(f"  Balance < 0 (opposite):       {(balance<0).sum():>6d} ({((balance<0).sum()/n*100):.1f}%)")
print(f"  Balance = 0:                  {(balance==0).sum():>6d}")

# ============================================================
# 2. Does balance sign predict FILT zone?
# ============================================================
print(f"\n{'='*70}")
print(f"  2. BALANCE SIGN vs FILT ZONE")
print(f"{'='*70}")

t1 = FILT <= 0.05
t2 = (FILT > 0.05) & (FILT <= 0.15)
t3 = FILT > 0.15

print(f"{'Zone':>6} {'All':>8} {'Bal>0':>8} {'Bal<0':>8} {'Bal=0':>8}")
print(f"{'-'*38}")
for zone_name, zone_mask in [('T1', t1), ('T2', t2), ('T3', t3)]:
    n_all = zone_mask.sum()
    n_pos = (zone_mask & (balance>0)).sum()
    n_neg = (zone_mask & (balance<0)).sum()
    n_zero = (zone_mask & (balance==0)).sum()
    print(f"{zone_name:>6} {n_all:>8d} {n_pos:>8d} {n_neg:>8d} {n_zero:>8d}")

# T3 rate by balance sign
for bs, name in [(1, 'Balance>0'), (-1, 'Balance<0'), (0, 'Balance=0')]:
    mask = balance_sign == bs
    if mask.sum() < 10: continue
    t3r = t3[mask].mean()
    print(f"  {name}: T3 rate = {t3r:.1%} (vs baseline {t3.mean():.1%})  n={mask.sum()}")

# Balance as T2+/T3 classifier (AUC)
auc_t3 = roc_auc_score(t3.astype(int), balance)
auc_t2 = roc_auc_score((FILT>0.05).astype(int), balance)
print(f"  Balance AUC for T3: {auc_t3:.4f}")
print(f"  Balance AUC for T2+: {auc_t2:.4f}")

# Compare with RIVER_LEVEL alone
auc_river_t3 = roc_auc_score(t3.astype(int), RL)
auc_river_t2 = roc_auc_score((FILT>0.05).astype(int), RL)
print(f"  RIVER_LEVEL AUC for T3: {auc_river_t3:.4f}")
print(f"  RIVER_LEVEL AUC for T2+: {auc_river_t2:.4f}")

# ============================================================
# 3. Balance balance sign × extreme warning
# ============================================================
print(f"\n{'='*70}")
print(f"  3. BALANCE × EXTREME WARNING (joint effect)")
print(f"{'='*70}")

ntu_p95 = np.percentile(data['RW_NTU'].dropna(), 95)
river_p97 = np.percentile(data['RIVER_LEVEL'].dropna(), 97)
extreme_ntu = data['RW_NTU'] > ntu_p95
extreme_river = RL > river_p97
extreme_flag = (extreme_ntu | extreme_river).values

# Extreme only: split by balance sign
ext_only = extreme_flag & (balance_sign != 0)
for bs_val, bs_name in [(1, 'Bal>0'), (-1, 'Bal<0')]:
    ex_bs = extreme_flag & (balance_sign == bs_val)
    if ex_bs.sum() < 5: continue
    t3r = t3[ex_bs].mean()
    t2pr = (FILT[ex_bs] > 0.05).mean()
    print(f"  Extreme+{bs_name}: T3={t3r:.0%}  T2+={t2pr:.0%}  n={ex_bs.sum()}")

# Extreme + balance > 0 = extreme but stable condition
# Extreme + balance < 0 = extreme AND stressed condition
for cond, name in [(extreme_flag, 'Extreme only'),
                    (extreme_flag & (balance > 0), 'Extreme + Bal>0'),
                    (extreme_flag & (balance < 0), 'Extreme + Bal<0')]:
    if cond.sum() < 5: continue
    t3r = t3[cond].mean()
    t2pr = (FILT[cond] > 0.05).mean()
    print(f"  {name:>30}: T3={t3r:.0%}  T2+={t2pr:.0%}  n={cond.sum()}")

# ============================================================
# 4. Balance as AR(6) feature
# ============================================================
print(f"\n{'='*70}")
print(f"  4. BALANCE in AR(6) + DeltaFILT prediction")
print(f"{'='*70}")

# Lags
FILT_lags = np.zeros((n, 6))
for lag in range(1, 7):
    FILT_lags[lag:, lag-1] = FILT[:-lag]
    FILT_lags[:lag, lag-1] = FILT[0]

# Delta
delta_filt = np.diff(FILT, prepend=FILT[0])

# Baseline: AR(6) on delta lags
X_dlags = np.zeros((n, 6))
for lag in range(1, 7):
    X_dlags[lag:, lag-1] = delta_filt[:-lag]
    X_dlags[:lag, lag-1] = delta_filt[0]

X_base = np.column_stack([np.ones(n), X_dlags])
theta_b = np.linalg.lstsq(X_base, delta_filt, rcond=None)[0]
r2_base = r2_score(delta_filt, X_base @ theta_b)

# AR(6) + balance
X_bal = np.column_stack([np.ones(n), X_dlags, balance])
theta_bl = np.linalg.lstsq(X_bal, delta_filt, rcond=None)[0]
r2_bal = r2_score(delta_filt, X_bal @ theta_bl)
print(f"  AR(6) delta baseline:             R2={r2_base:.4f}")
print(f"  AR(6) delta + balance:            R2={r2_bal:.4f}")
print(f"  Improvement:                      +{r2_bal-r2_base:.4f}")

# Per zone
for mask, name in [(FILT<=0.15, 'Normal/Stable'), (FILT>0.15, 'T3 zone')]:
    r2b = r2_score(delta_filt[mask], (X_base @ theta_b)[mask])
    r2bl = r2_score(delta_filt[mask], (X_bal @ theta_bl)[mask])
    print(f"  {name}: base={r2b:.4f}  +bal={r2bl:.4f}  delta={r2bl-r2b:+.4f}")

# AR(6) on FILT levels + balance
X_lev = np.column_stack([np.ones(n), FILT_lags])
X_levb = np.column_stack([np.ones(n), FILT_lags, balance])
t_l = np.linalg.lstsq(X_lev, FILT, rcond=None)[0]
t_lb = np.linalg.lstsq(X_levb, FILT, rcond=None)[0]
r2_l = r2_score(FILT, X_lev @ t_l)
r2_lb = r2_score(FILT, X_levb @ t_lb)
print(f"\n  AR(6) level baseline:             R2={r2_l:.4f}")
print(f"  AR(6) level + balance:            R2={r2_lb:.4f}")
print(f"  Improvement:                      +{r2_lb-r2_l:.4f}")
