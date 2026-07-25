"""
q2_tau_align_check.py — Compare raw vs τ-aligned features for extreme-value prediction
====================================================================================
τ values (from Q1 softmax + event CCF):
  RW_NTU → FILT:    τ=2 steps (4h)
  RIVER_LEVEL → FILT: τ=1 step  (2h)  (hydraulic)
  RW_CLR → FILT:    τ=2 steps (4h)  (synced with NTU)
  ALUM → FILT:      τ=2 steps (4h)
"""
import os, numpy as np, pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import json, warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data', '2025')

# ============================================================
# 1. LOAD
# ============================================================
data_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
data_dir = os.path.join(DATA_DIR, data_dirs[0])
FILES = sorted([f for f in os.listdir(data_dir) if f.endswith('.xlsx')])
RENAME = {
    'RIVER LEVEL':'RIVER_LEVEL','R/W FLOW':'RW_FLOW','R/W NTU':'RW_NTU',
    'R/W CLR':'RW_CLR','R/W PH':'RW_PH','FILT. NTU':'FILT_NTU',
    'C/W WELL LEVEL':'CW_WELL_LEVEL','T/W FLOW':'TW_FLOW','ALUM':'ALUM','NTU':'NTU',
}
NUM_COLS = ['RIVER_LEVEL','RW_FLOW','RW_NTU','RW_CLR','RW_PH','FILT_NTU','CW_WELL_LEVEL','TW_FLOW','ALUM','NTU']
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
    if c in data.columns:
        med = data[c].median()
        data[c] = data[c].fillna(med if not pd.isna(med) else 0)

n = len(data)
FILT = data['FILT_NTU'].values

# ============================================================
# 2. Create τ-aligned features
# ============================================================
data['NTU_tau2'] = data['RW_NTU'].shift(2).fillna(data['RW_NTU'].median())  # 4h lag
data['RIVER_tau1'] = data['RIVER_LEVEL'].shift(1).fillna(data['RIVER_LEVEL'].median())  # 2h lag
data['CLR_tau2'] = data['RW_CLR'].shift(2).fillna(data['RW_CLR'].median())
data['ALUM_tau2'] = data['ALUM'].shift(2).fillna(data['ALUM'].median())
data['NTUxCLR_tau2'] = data['NTU_tau2'] * data['CLR_tau2']
data['RIVxNTU_tau'] = data['RIVER_tau1'] * data['NTU_tau2']  # hybrid: river 2h + ntu 4h

# ============================================================
# 3. Compare raw vs aligned: precision at each θ
# ============================================================
theta_list = [0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
pct_list = [90, 95, 97, 99]

# Define feature pairs to compare: (raw, aligned, name)
pairs = [
    ('RW_NTU', 'NTU_tau2', 'RW_NTU'),
    ('RIVER_LEVEL', 'RIVER_tau1', 'RIVER_LEVEL'),
    ('RW_CLR', 'CLR_tau2', 'RW_CLR'),
    ('NTUxCLR_tau2', None, 'LOAD_tau'),
    ('RIVxNTU_tau', None, 'RIVxNTU_tau'),
]

# Also raw combos for comparison
data['LOAD_raw'] = data['RW_NTU'] * data['RW_CLR']
data['RIVxNTU_raw'] = data['RIVER_LEVEL'] * data['RW_NTU']

print(f"{'='*110}")
print(f"  TAU ALIGNMENT EFFECT: Raw vs τ-aligned extreme-value prediction")
print(f"{'='*110}")

print(f"{'Feature':<18} {'Pct':>4} {'τ_align':>8} ||", end='')
for th in theta_list:
    print(f' θ={th:<5}', end='')
print()
print("-"*110)

def rule_precision(feat_vals, true_vals, pct, theta):
    thr = np.percentile(feat_vals, pct)
    mask = feat_vals > thr
    if mask.sum() < 10:
        return None, thr, mask.sum()
    y = (true_vals > theta).astype(int)
    return y[mask].mean(), thr, mask.sum()

for feat_name, aligned_name, base_name in pairs:
    raw_vals = data[base_name].values if base_name in data.columns else None
    tau_vals = data[aligned_name].values if aligned_name else None
    
    for pct in pct_list:
        line = f"{feat_name:<18} P{pct:>3d} {'tau':>8} ||"
        for theta in theta_list:
            p, thr, n_trig = rule_precision(tau_vals, FILT, pct, theta) if tau_vals is not None else (None, 0, 0)
            if p is not None and n_trig >= 10:
                line += f' {p:>6.1%}({n:<3d})'
            else:
                line += f'    N/A   '
        
        # If there's a raw version, show raw too
        if raw_vals is not None:
            line += f'  | raw: '
            for theta in theta_list:
                p_r, _, n_r = rule_precision(raw_vals, FILT, pct, theta)
                if p_r is not None and n_r >= 10:
                    delta = (p - p_r) if p is not None else 0
                    line += f' {p_r:>6.1%}({n_r:<3d})'
                else:
                    line += f'    N/A   '
        
        if tau_vals is not None:
            thr = np.percentile(tau_vals, pct)
            n_trig = (tau_vals > thr).sum()
            line += f'  || trigger_n={n_trig} thr={thr:.0f}'
        print(line)

# ============================================================
# 4. CV comparison: raw vs τ-aligned RW_NTU at θ=0.05
# ============================================================
print(f"\n{'='*110}")
print(f"  CV COMPARISON: RW_NTU raw vs τ-aligned, θ=0.05")
print(f"{'='*110}")

tscv = TimeSeriesSplit(n_splits=5)

for feat_name, feat_raw_col, feat_tau_col in [('RW_NTU', 'RW_NTU', 'NTU_tau2')]:
    raw_vals = data[feat_raw_col].values
    tau_vals = data[feat_tau_col].values
    
    print(f"\n  {feat_name}:")
    print(f"{'Fold':>5} {'raw_P95':>8} {'raw_prec':>9} {'raw_rec':>8} {'raw_N':>5} | {'tau_P95':>8} {'tau_prec':>9} {'tau_rec':>8} {'tau_N':>5} {'improve':>8}")
    print("-"*75)
    
    raw_results = []
    tau_results = []
    
    for fold, (tr, va) in enumerate(tscv.split(data)):
        # Learn P95 on training
        raw_thr = np.percentile(raw_vals[tr], 95)
        tau_thr = np.percentile(tau_vals[tr], 95)
        
        # Apply on validation
        raw_pred = raw_vals[va] > raw_thr
        tau_pred = tau_vals[va] > tau_thr
        
        va_true = (FILT[va] > 0.05).astype(int)
        base_rate = va_true.mean()
        
        raw_prec = va_true[raw_pred].mean() if raw_pred.sum() > 0 else 0
        raw_rec = va_true[raw_pred].sum() / max(va_true.sum(), 1)
        tau_prec = va_true[tau_pred].mean() if tau_pred.sum() > 0 else 0
        tau_rec = va_true[tau_pred].sum() / max(va_true.sum(), 1)
        
        imp = tau_prec - raw_prec
        print(f"  Fold{fold:>2d} {raw_thr:>8.1f} {raw_prec:>8.1%} {raw_rec:>8.1%} {raw_pred.sum():>5d} | {tau_thr:>8.1f} {tau_prec:>8.1%} {tau_rec:>8.1%} {tau_pred.sum():>5d} {imp:>+8.1%}")
        
        raw_results.append(raw_prec)
        tau_results.append(tau_prec)
    
    print(f"  {'Mean':>5} {'':>8} {np.mean(raw_results):>8.1%} {'':>8} {'':>5} | {'':>8} {np.mean(tau_results):>8.1%} {'':>8} {'':>5} {np.mean(tau_results)-np.mean(raw_results):>+8.1%}")

# ============================================================
# 5. Check: what if we combine raw + tau aligned?
# ============================================================
print(f"\n{'='*110}")
print(f"  TAU WINDOW COMBINATION: NTU at t-2, t-3 (4h, 6h)")
print(f"{'='*110}")

# NTU weighted alignment: use the softmax weights from Q1
# τ₁ weights: [0.006, 0.035, 0.362, 0.361, 0.063, 0.092, 0.081]
softmax_weights = [0.006, 0.035, 0.362, 0.361, 0.063, 0.092, 0.081]
for pct in [95, 97, 99]:
    print(f"\n  P{pct} threshold:")
    print(f"{'θ':>6} {'raw_prec':>9} {'tau2_prec':>10} {'tau3_prec':>10} {'softmax_prec':>12} {'N_tau2':>6}")
    
    ntu_raw = data['RW_NTU'].values
    ntu_t2 = data['NTU_tau2'].values
    ntu_t3 = data['RW_NTU'].shift(3).fillna(data['RW_NTU'].median()).values
    
    # Softmax: weighted sum of lag 0-6
    ntu_aligned = np.zeros(n)
    for lag, w in enumerate(softmax_weights):
        if lag == 0:
            ntu_aligned += w * ntu_raw
        else:
            ntu_aligned += w * data['RW_NTU'].shift(lag).fillna(data['RW_NTU'].median()).values
    
    for theta in [0.05, 0.08, 0.12, 0.15]:
        y = (FILT > theta).astype(int)
        vals_list = [(ntu_raw, 'raw'), (ntu_t2, 't-2'), (ntu_t3, 't-3'), (ntu_aligned, 'softmax')]
        precs = {}
        
        for vals, name in vals_list:
            thr = np.percentile(vals, pct)
            mask = vals > thr
            if mask.sum() >= 10:
                precs[name] = {'prec': y[mask].mean(), 'n': mask.sum()}
        
        line = f"θ={theta:<5.2f}"
        for name in ['raw', 't-2', 't-3', 'softmax']:
            if name in precs:
                line += f' {precs[name]["prec"]:>8.1%}  '
            else:
                line += f'    N/A    '
        n_t2 = precs.get('t-2', {}).get('n', 0)
        line += f'  N={n_t2:>4d}'
        
        # Show improvement
        if 'raw' in precs and 't-2' in precs:
            line += f'  Δ={precs["t-2"]["prec"]-precs["raw"]["prec"]:+5.1%}'
        print(line)
