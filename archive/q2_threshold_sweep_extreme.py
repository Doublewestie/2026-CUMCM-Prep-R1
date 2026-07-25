"""
q2_threshold_sweep_extreme.py — Find optimal FILT threshold for extreme-value rules
=================================================================================
Goal: For extreme input features (RIVER>P95, NTU×CLR>P95, etc.), find the FILT
threshold θ where the warning system achieves stable prediction accuracy.
"""
import os, numpy as np, pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score
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
    'C/W WELL LEVEL':'CW_WELL_LEVEL','T/W FLOW':'TW_FLOW','ALUM':'ALUM','NTU':'NTU','CLR':'CLR',
}
NUM_COLS = ['RIVER_LEVEL','RW_FLOW','RW_NTU','RW_CLR','RW_PH','FILT_NTU','CW_WELL_LEVEL','TW_FLOW','ALUM','NTU','CLR']
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
n = len(data)

# Fill NaN
for c in ['RIVER_LEVEL','RW_NTU','RW_CLR','RW_FLOW','ALUM']:
    if c in data.columns:
        med = data[c].median()
        data[c] = data[c].fillna(med if not pd.isna(med) else 0)

FILT = data['FILT_NTU'].values

# Derived features
data['LOAD'] = data['RW_NTU'] * data['RW_CLR']
data['NTU_DOSE'] = data['RW_NTU'] / (data['ALUM'] + 1e-4)
data['RIVxNTU'] = data['RIVER_LEVEL'] * data['RW_NTU']

feat_pool = {
    'RIVER_LEVEL': data['RIVER_LEVEL'].values,
    'RW_NTU':      data['RW_NTU'].values,
    'RW_CLR':      data['RW_CLR'].values,
    'LOAD':        data['LOAD'].values,
    'NTU_DOSE':    data['NTU_DOSE'].values,
    'RIVxNTU':     data['RIVxNTU'].values,
}

# ============================================================
# 2. Define warning rules (from previous analysis + variations)
# ============================================================
# We'll define a set of candidate warning rules at various percentiles
candidate_rules = []

for feat_name, vals in feat_pool.items():
    for pct in [85, 88, 90, 92, 95, 97, 99]:
        thr = np.percentile(vals, pct)
        candidate_rules.append({
            'name': f'{feat_name}>P{pct}(>{thr:.0f})',
            'feat': feat_name,
            'threshold': thr,
            'pct': pct,
        })

# Also add combination rules
for pct in [90, 95]:
    r_thr = np.percentile(feat_pool['RIVER_LEVEL'], pct)
    n_thr = np.percentile(feat_pool['RW_NTU'], pct)
    l_thr = np.percentile(feat_pool['LOAD'], pct)
    
    candidate_rules.append({
        'name': f'RIVER>P{pct}_AND_NTU>P{pct}',
        'rule_type': 'and',
        'feats': ['RIVER_LEVEL', 'RW_NTU'],
        'thresholds': [r_thr, n_thr],
    })
    candidate_rules.append({
        'name': f'RIVER>P{pct}_AND_LOAD>P{pct}',
        'rule_type': 'and',
        'feats': ['RIVER_LEVEL', 'LOAD'],
        'thresholds': [r_thr, l_thr],
    })

# Evaluate each rule's "is true" vector
def rule_applies(rule):
    if 'rule_type' in rule and rule['rule_type'] == 'and':
        mask = np.ones(n, dtype=bool)
        for feat_name, thr in zip(rule['feats'], rule['thresholds']):
            mask &= (feat_pool[feat_name] > thr)
        return mask
    else:
        return feat_pool[rule['feat']] > rule['threshold']

# ============================================================
# 3. Sweep FILT thresholds for each rule
# ============================================================
theta_candidates = [0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.0]

print(f"{'='*120}")
print(f"  EXTREME-RULE THRESHOLD SWEEP: For each rule, find best FILT threshold")
print(f"{'='*120}")
print(f"{'Rule':<45} {'N_trig':>6}  |", end='')
for th in theta_candidates:
    print(f" {'thr='+str(th):>10}", end='')
print()

# For each rule and each threshold, compute metrics
best_rules = []

for rule in candidate_rules:
    pred = rule_applies(rule)
    n_trig = pred.sum()
    if n_trig < 10:
        continue
    
    name_str = rule['name']
    print(f"\n{name_str:<45} {n_trig:>6d}  | ", end='')
    
    for th_idx, theta in enumerate(theta_candidates):
        y_true = (FILT > theta).astype(int)
        base_rate = y_true.mean()
        
        # When rule triggers, what's the FILT>theta rate?
        trig_rate = y_true[pred].mean() if n_trig > 0 else 0
        precision_val = trig_rate
        recall_val = y_true[pred].sum() / max(y_true.sum(), 1)
        f1 = 2 * precision_val * recall_val / (precision_val + recall_val + 1e-8)
        
        # For precision (trig_rate), mark if it significantly exceeds base_rate
        gain = precision_val - base_rate
        
        if gain > 0.05 and precision_val > 0.60:
            mark = f'{precision_val:.0%}'.rjust(8) + '.'
        else:
            mark = f'{precision_val:.0%}'.rjust(8) + ' '
        
        print(f'{mark} ', end='')

# ============================================================
# 4. Detailed evaluation for the best 3 single rules at each threshold
# ============================================================
print(f"\n\n{'='*120}")
print(f"  TOP 3 SINGLE RULES: Detail at each FILT threshold")
print(f"{'='*120}")

# Pick top-performing rules
top_single = ['RIVER_LEVEL>P95', 'RW_NTU>P95', 'LOAD>P95']
top_single_full = [r for r in candidate_rules if any(t in r['name'] for t in top_single)][:3]

for rule in top_single_full:
    pred = rule_applies(rule)
    n_trig = pred.sum()
    print(f"\n  Rule: {rule['name']}  (N_trig={n_trig})")
    print(f"{'FILT_thr':<10} {'Base%':>8} {'Trig%':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Gain':>8} {'Guarantee':>12}")
    print(f"{'-'*70}")
    
    for theta in theta_candidates:
        y_true = (FILT > theta).astype(int)
        base = y_true.mean()
        p = y_true[pred].mean()
        r = y_true[pred].sum() / max(y_true.sum(), 1)
        f1 = 2*p*r/(p+r+1e-8)
        g = p - base
        
        # Guarantee: if rule triggers, we're "X% sure FILT > θ"
        guarantee_level = f'>θ={p:.0%}"'
        print(f"θ={theta:<5.2f} {base:>7.1%} {p:>7.1%} {p:>7.1%} {r:>7.1%} {f1:>7.3f} {g:>+7.1%} {guarantee_level:>12}")

# ============================================================
# 5. CV validation: for each rule + theta, compute CV precision
# ============================================================
print(f"\n{'='*120}")
print(f"  CV VALIDATION (5-fold TimeSeriesSplit)")
print(f"{'='*120}")

tscv = TimeSeriesSplit(n_splits=5)

for rule in top_single_full:
    rule_name = rule['name']
    feat_name = rule.get('feat', '')
    pct_val = rule.get('pct', 0)
    
    print(f"\n  Rule: {rule_name}")
    print(f"{'FILT_thr':<10} {'CV_Prec':>8} {'CV_Rec':>8} {'CV_F1':>8} {'N_trig_CV':>10}")
    print(f"{'-'*50}")
    
    for theta in [0.05, 0.08, 0.12, 0.15, 0.20, 0.30]:
        cv_precs = []; cv_recs = []; cv_n = []
        
        for tr, va in tscv.split(data):
            # Learn threshold on training set
            tr_vals = feat_pool[feat_name][tr]
            tr_thr = np.percentile(tr_vals, pct_val)
            
            # Apply on validation set
            va_pred = feat_pool[feat_name][va] > tr_thr
            n_va = va_pred.sum()
            
            if n_va < 5:
                continue
            
            va_true = (FILT[va] > theta).astype(int)
            cv_precs.append(precision_score(va_true, va_pred, zero_division=0))
            cv_recs.append(recall_score(va_true, va_pred, zero_division=0))
            cv_n.append(n_va)
        
        if cv_precs:
            print(f"θ={theta:<5.2f} {np.mean(cv_precs):>7.1%} {np.mean(cv_recs):>7.1%} {np.mean(cv_precs)+2e-9:>7.3f} {int(np.mean(cv_n)):>10d}")

# ============================================================
# 6. Summary recommendation
# ============================================================
print(f"\n{'='*120}")
print(f"  SUMMARY: RECOMMENDED RULES")
print(f"{'='*120}")

# For each FILT threshold, find the best rule
for theta in [0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
    print(f"\n  FILT threshold: θ={theta:.2f}")
    best_f1, best_name = 0, ''
    for rule in candidate_rules:
        pred = rule_applies(rule)
        n_trig = pred.sum()
        if n_trig < 20: continue
        y_true = (FILT > theta).astype(int)
        p = y_true[pred].mean()
        r = y_true[pred].sum() / max(y_true.sum(), 1)
        f1v = 2*p*r/(p+r+1e-8)
        if p > 0.70 and r > 0.01 and f1v > best_f1:
            best_f1 = f1v
            best_name = f"{rule['name']} (P={p:.0%}, R={r:.1%})"
    if best_name:
        print(f"    Best rule: {best_name}")

# Save
out = {
    'theta_candidates': theta_candidates,
    'results': []
}
for rule in candidate_rules:
    pred = rule_applies(rule)
    n_trig = pred.sum()
    if n_trig < 10: continue
    for theta in theta_candidates:
        y_true = (FILT > theta).astype(int)
        out['results'].append({
            'rule': rule['name'],
            'n_trig': int(n_trig),
            'theta': theta,
            'base_rate': float(y_true.mean()),
            'precision': float(y_true[pred].mean()),
            'recall': float(y_true[pred].sum() / max(y_true.sum(), 1)),
        })
out_dir = os.path.join(BASE, 'output')
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'q2_extreme_sweep.json'), 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved to output/q2_extreme_sweep.json")
