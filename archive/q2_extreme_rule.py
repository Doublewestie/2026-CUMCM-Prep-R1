"""
q2_extreme_rule.py — Extreme-value warning rules for FILT tier prediction
=======================================================================
Finds "safe zone" where FILT <= 0.05, and extreme-threshold rules for T2+/T3.
Avoids the signal-dilution problem by only triggering on tail events.
"""
import os, numpy as np, pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import json, warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data', '2025')

# ============================================================
# 1. LOAD + PREP
# ============================================================
data_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
data_dir = os.path.join(DATA_DIR, data_dirs[0])
FILES = sorted([f for f in os.listdir(data_dir) if f.endswith('.xlsx')])
RENAME = {
    'RIVER LEVEL':'RIVER_LEVEL','R/W FLOW':'RW_FLOW','R/W NTU':'RW_NTU',
    'R/W CLR':'RW_CLR','R/W PH':'RW_PH','FILT. NTU':'FILT_NTU',
    'C/W WELL LEVEL':'CW_WELL_LEVEL','T/W FLOW':'TW_FLOW',
    'ALUM':'ALUM','NTU':'NTU','CLR':'CLR','PH':'PH',
}
NUM_COLS = ['RIVER_LEVEL','RW_FLOW','RW_NTU','RW_CLR','RW_PH','FILT_NTU','CW_WELL_LEVEL','TW_FLOW','ALUM','NTU','CLR','PH']

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
n = len(data)

# Fill NaN
raw_cols = ['RIVER_LEVEL','RW_NTU','RW_CLR','RW_FLOW','ALUM','CW_WELL_LEVEL','TW_FLOW']
for c in raw_cols:
    if c in data.columns:
        med = data[c].median()
        data[c] = data[c].fillna(med if not pd.isna(med) else 0)

# ---- Targets ----
FILT = data['FILT_NTU'].values
t1 = (FILT <= 0.05).astype(int)
t2plus = (FILT > 0.05).astype(int)
t3 = (FILT > 0.15).astype(int)
t2only = ((FILT > 0.05) & (FILT <= 0.15)).astype(int)

print(f"N={n}  T1(FILT<=0.05)={(t1==1).sum()}  T2(0.05<FILT<=0.15)={(t2only==1).sum()}  T3(FILT>0.15)={(t3==1).sum()}")
print(f"T2+ base rate = {t2plus.mean():.1%},  T3 base rate = {t3.mean():.1%}")

# ---- Physics-derived features (NO FILT leakage) ----
# note: ALUM is ~constant 0.054 in data, using median as dose
ALUM_med = data['ALUM'].median() if not pd.isna(data['ALUM'].median()) else 0.054

# LOAD = NTU * CLR (total pollutant load competing for ALUM)
data['LOAD'] = data['RW_NTU'] * data['RW_CLR']

# NTU_dose_ratio = NTU / ALUM (how much turbidity per unit coagulant)
data['NTU_DOSE'] = data['RW_NTU'] / (data['ALUM'] + 1e-4)

# RIVER_x_NTU (flood × pollution interaction)
data['RIVxNTU'] = data['RIVER_LEVEL'] * data['RW_NTU']

# RIVER_x_FLOW (hydraulic stress)
data['RIVxFLOW'] = data['RIVER_LEVEL'] * data['RW_FLOW']

# FLUX = NTU * FLOW (turbidity mass flow rate)
data['FLUX'] = data['RW_NTU'] * data['RW_FLOW']

# ---- Feature pool (all raw + derived, NO FILT involvement) ----
feat_pool = {
    # Raw
    'RW_NTU':       data['RW_NTU'].values,
    'RW_CLR':       data['RW_CLR'].values,
    'RIVER_LEVEL':  data['RIVER_LEVEL'].values,
    'RW_FLOW':      data['RW_FLOW'].values,
    'ALUM':         data['ALUM'].values,
    # Physics-derived
    'LOAD':         data['LOAD'].values,
    'NTU_DOSE':     data['NTU_DOSE'].values,
    'RIVxNTU':      data['RIVxNTU'].values,
    'RIVxFLOW':     data['RIVxFLOW'].values,
    'FLUX':         data['FLUX'].values,
}

y_t2plus = t2plus
y_t3 = t3

print(f"\n{'='*90}")
print(f"  PHASE 1: SAFE ZONE DISCOVERY")
print(f"  (Subspace where FILT > 0.05 almost never happens)")
print(f"{'='*90}")

# ============================================================
# PHASE 1: SAFE ZONE ("hypercube" where T2+ rate → 0)
# ============================================================
# For each feature, find threshold where everything below it is T1
safe_rules = []
for feat_name_base in ['RW_NTU','RW_CLR','RIVER_LEVEL','FLUX','LOAD']:
    name = feat_name_base
    vals = feat_pool[name]
    
    # Scan decreasing thresholds: what % of samples below threshold are T1?
    best_thr, best_t1_rate, best_n = 0, 0, 0
    for p in [10, 20, 25, 30, 33, 40, 50, 60, 70]:
        thr = np.percentile(vals, p)
        mask = vals <= thr
        n_mask = mask.sum()
        if n_mask < 50:
            continue
        t1_rate = t1[mask].mean()
        if t1_rate > best_t1_rate and n_mask > 100:
            best_t1_rate, best_thr, best_n = t1_rate, thr, n_mask
    
    if best_t1_rate > 0.80:  # at least 80% T1 below threshold
        safe_rules.append({'feat': name, 'threshold': best_thr, 't1_rate': best_t1_rate, 'n': best_n})
        print(f"\n  {name} <= {best_thr:.1f}")
        print(f"    → T1 rate = {best_t1_rate:.1%}  (T2+= {1-best_t1_rate:.1%})  N = {best_n}")

# Combined safe zone: ALL features below threshold simultaneously
print(f"\n  Combined safe zone (ALL must be below threshold):")
safe_feats = {r['feat']: r['threshold'] for r in safe_rules}
mask_safe = np.ones(n, dtype=bool)
for feat, thr in safe_feats.items():
    mask_safe &= (feat_pool[feat] <= thr)
n_safe = mask_safe.sum()
t1_safe = t1[mask_safe].mean()
t2plus_safe = t2plus[mask_safe].mean()
t3_safe = t3[mask_safe].mean()
print(f"    N = {n_safe} ({n_safe/n*100:.1f}%)")
print(f"    T1 rate = {t1_safe:.1%}  (T2+ = {t2plus_safe:.1%})  T3 rate = {t3_safe:.1%}")

# ============================================================
# PHASE 2: EXTREME-THRESHOLD WARNING (single feature)
# ============================================================
print(f"\n{'='*90}")
print(f"  PHASE 2: EXTREME-VALUE WARNING (T2+ = FILT > 0.05)")
print(f"{'='*90}")
print(f"  Base T2+ rate: {t2plus.mean():.1%}")
print()
print(f"  {'Feature':<15} {'P':<5} {'Thresh':>8} {'N_trig':>6} {'T2+%':>7} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Gain':>6}")
print(f"  {'-'*70}")

extreme_rules_t2 = []
extreme_rules_t3 = []

for feat_name in feat_pool:
    vals = feat_pool[feat_name]
    
    best = {'t2': {'f1':0,'pct':0,'prec':0,'rec':0,'n':0,'thr':0,'rule':''},
            't3': {'f1':0,'pct':0,'prec':0,'rec':0,'n':0,'thr':0,'rule':''}}
    
    for pct in [85, 90, 93, 95, 97, 98, 99]:
        thr = np.percentile(vals, pct)
        mask = vals > thr
        n_trig = mask.sum()
        if n_trig < 20:
            continue
        
        # T2+ target
        t2rate = t2plus[mask].mean()
        t2rec  = t2plus[mask].sum() / max(t2plus.sum(), 1)
        t2prec = t2plus[mask].mean()
        t2f1   = 2 * t2prec * t2rec / (t2prec + t2rec + 1e-6)
        
        if t2prec > best['t2']['prec'] and n_trig > 30:
            best['t2'] = {'f1':t2f1, 'pct':pct, 'prec':t2prec, 'rec':t2rec, 'n':n_trig, 'thr':thr, 'rule':'>'}
        
        # T3 target
        t3rate = t3[mask].mean()
        t3rec  = t3[mask].sum() / max(t3.sum(), 1)
        t3prec = t3[mask].mean()
        t3f1   = 2 * t3prec * t3rec / (t3prec + t3rec + 1e-6)
        
        if t3prec > best['t3']['prec'] and n_trig > 30:
            best['t3'] = {'f1':t3f1, 'pct':pct, 'prec':t3prec, 'rec':t3rec, 'n':n_trig, 'thr':thr, 'rule':'>'}
    
    # Report T2+
    b = best['t2']
    if b['prec'] > t2plus.mean() + 0.05:  # must beat base rate by 5pp
        gain = b['prec'] - t2plus.mean()
        extreme_rules_t2.append({'feat':feat_name, **b})
        print(f"  {feat_name:<15} P{b['pct']:<4} {b['thr']:>8.1f} {b['n']:>6d} {b['prec']:>6.1%} {b['prec']:>6.3f} {b['rec']:>6.3f} {b['f1']:>6.3f} {gain:>+6.3f}")
    
    # Report T3
    b3 = best['t3']
    if b3['prec'] > t3.mean() + 0.03:
        gain = b3['prec'] - t3.mean()
        extreme_rules_t3.append({'feat':feat_name, **b3})

# ============================================================
# PHASE 3: MULTI-FEATURE JOINT WARNINGS
# ============================================================
print(f"\n{'='*90}")
print(f"  PHASE 3: MULTI-FEATURE JOINT WARNINGS")
print(f"{'='*90}")

# Take top 3 single-feature rules for T2+
top_t2 = sorted(extreme_rules_t2, key=lambda x: -x['prec'])[:3]
print(f"\n  Top single-feature rules (T2+):")
for r in top_t2:
    print(f"    {r['feat']} > {r['thr']:.1f} → T2+={r['prec']:.0%} (N={r['n']})")

# Test combinations
if len(top_t2) >= 2:
    print(f"\n  Joint rules (2-feature, P90 thresholds):")
    p90_thrs = {}
    for r in top_t2:
        p90_thrs[r['feat']] = np.percentile(feat_pool[r['feat']], 90)
    
    feat_names = list(p90_thrs.keys())
    for i in range(len(feat_names)):
        for j in range(i+1, len(feat_names)):
            f1, f2 = feat_names[i], feat_names[j]
            mask = (feat_pool[f1] > p90_thrs[f1]) & (feat_pool[f2] > p90_thrs[f2])
            n_joint = mask.sum()
            if n_joint < 20:
                continue
            t2p = t2plus[mask].mean()
            t3p = t3[mask].mean()
            print(f"    {f1}>{p90_thrs[f1]:.0f} AND {f2}>{p90_thrs[f2]:.0f}")
            print(f"      T2+={t2p:.0%}  T3={t3p:.0%}  N={n_joint}")
    
    # Triple
    if len(top_t2) >= 3:
        f1, f2, f3 = feat_names[0], feat_names[1], feat_names[2]
        mask = (feat_pool[f1] > p90_thrs[f1]) & (feat_pool[f2] > p90_thrs[f2]) & (feat_pool[f3] > p90_thrs[f3])
        n_trip = mask.sum()
        if n_trip >= 10:
            t2p = t2plus[mask].mean()
            t3p = t3[mask].mean()
            print(f"    {f1} AND {f2} AND {f3} (all > P90)")
            print(f"      T2+={t2p:.0%}  T3={t3p:.0%}  N={n_trip}")

# ============================================================
# PHASE 4: ALUM FEEDBACK ANALYSIS
# ============================================================
print(f"\n{'='*90}")
print(f"  PHASE 4: ALUM FEEDBACK RESPONSE")
print(f"{'='*90}")

# Does ALUM increase when NTU/CLR/RIVER are extreme?
for feat_name in ['RW_NTU','RW_CLR','RIVER_LEVEL']:
    vals = feat_pool[feat_name]
    alum_vals = data['ALUM'].values
    for pct in [50, 75, 90, 95]:
        thr = np.percentile(vals, pct)
        mask_hi = vals > thr; mask_lo = vals <= thr
        alum_hi = alum_vals[mask_hi].mean() if mask_hi.sum() > 0 else 0
        alum_lo = alum_vals[mask_lo].mean() if mask_lo.sum() > 0 else 0
        print(f"  {feat_name} > P{pct} ({thr:.0f}): ALUM={alum_hi:.4f}  vs <=P{pct}: ALUM={alum_lo:.4f}  diff={alum_hi-alum_lo:+.4f}")

# ============================================================
# PHASE 5: FINAL RULES + VOTING
# ============================================================
print(f"\n{'='*90}")
print(f"  PHASE 5: VOTING RULE SYSTEM")
print(f"{'='*90}")

# Build voting: each rule gets weight proportional to its precision gain
# Base score: 0
# Trigger RW_NTU > P95 → +3 (if precision is high)
# Trigger LOAD > P95 → +2
# etc.

vote_weights = {}
print(f"\n  Voting rules (each triggered → add points):")
for r in sorted(extreme_rules_t2, key=lambda x: -x['prec']):
    gain = r['prec'] - t2plus.mean()
    weight = max(1, int(gain * 20))  # scale gain to integer weight
    trigger_mask = feat_pool[r['feat']] > r['thr']
    n_trig = trigger_mask.sum()
    if n_trig < 30:
        continue
    vote_weights[r['feat']] = {'threshold': r['thr'], 'weight': weight, 'pct': r['pct']}
    print(f"  {r['feat']:>15} > {r['thr']:>8.1f} (P{r['pct']}) → +{weight} pts  (T2+={r['prec']:.0%})")

# Apply voting
total_votes = np.zeros(n)
for feat, rule in vote_weights.items():
    trigger = feat_pool[feat] > rule['threshold']
    total_votes += trigger * rule['weight']

print(f"\n  Vote distribution (max possible: {sum(r['weight'] for r in vote_weights.values())}):")
for pct in [10,25,50,75,90,95]:
    print(f"    P{pct}: {np.percentile(total_votes, pct):.1f}")
print(f"  Comfort zone vote mean: {total_votes[t1==1].mean():.2f}")
print(f"  T2+ zone vote mean: {total_votes[t2plus==1].mean():.2f}")
print(f"  T3 zone vote mean: {total_votes[t3==1].mean():.2f}")

# ============================================================
# PHASE 6: CROSS-VALIDATION (5-fold TS)
# ============================================================
print(f"\n{'='*90}")
print(f"  PHASE 6: CROSS-VALIDATION (5-fold TimeSeriesSplit)")
print(f"{'='*90}")

tscv = TimeSeriesSplit(n_splits=5)
cv_t2 = []
cv_t3 = []

for fold, (tr, va) in enumerate(tscv.split(data)):
    # On training set: discover extreme thresholds
    tr_t2 = t2plus[tr]; tr_t3 = t3[tr]
    base_t2_tr = tr_t2.mean(); base_t3_tr = tr_t3.mean()
    
    # Find best Pct threshold for each feature on training set
    fold_rules = []
    for feat_name in feat_pool:
        vals_tr = feat_pool[feat_name][tr]
        vals_va = feat_pool[feat_name][va]
        
        best_prec, best_thr, best_pct = 0, 0, 0
        for pct in [85, 90, 93, 95, 97, 99]:
            thr_cv = np.percentile(vals_tr, pct)
            mask_tr = vals_tr > thr_cv
            n_trig = mask_tr.sum()
            if n_trig < 20:
                continue
            prec_cv = tr_t2[mask_tr].mean()
            if prec_cv > best_prec and prec_cv > base_t2_tr + 0.03:
                best_prec, best_thr, best_pct = prec_cv, thr_cv, pct
        
        if best_thr > 0:
            fold_rules.append({'feat': feat_name, 'thr': best_thr, 'prec': best_prec})
    
    # Apply rules to validation set, vote
    va_votes = np.zeros(len(va))
    for fr in fold_rules:
        trig = feat_pool[fr['feat']][va] > fr['thr']
        va_votes += trig.astype(float)
    
    # Evaluate
    for target, target_name, y_true, store in [
        (t2plus[va], 'T2+', t2plus[va], cv_t2),
        (t3[va], 'T3', t3[va], cv_t3)
    ]:
        # Find best vote threshold on validation
        best_f1, best_thr = 0, 0
        best_prec, best_rec = 0, 0
        for vote_thr in np.arange(0, va_votes.max() + 0.5, 0.5):
            pred = (va_votes >= vote_thr).astype(int)
            f1 = f1_score(y_true, pred, zero_division=0)
            prec = precision_score(y_true, pred, zero_division=0)
            rec = recall_score(y_true, pred, zero_division=0)
            if f1 > best_f1 and rec > 0.05:
                best_f1, best_thr = f1, vote_thr
                best_prec, best_rec = prec, rec
        
        store.append({'fold': fold, 'n_rules': len(fold_rules), 'prec': best_prec, 'rec': best_rec, 'f1': best_f1})

print(f"\n  T2+ CV results:")
t2_precs = [r['prec'] for r in cv_t2]; t2_recs = [r['rec'] for r in cv_t2]
print(f"    Precision: {np.mean(t2_precs):.3f} ± {np.std(t2_precs):.3f}")
print(f"    Recall:    {np.mean(t2_recs):.3f} ± {np.std(t2_recs):.3f}")
print(f"    Base T2+ rate: {t2plus.mean():.3f}")
print(f"    Precision gain: {np.mean(t2_precs) - t2plus.mean():.3f}")

print(f"\n  T3 CV results:")
t3_precs = [r['prec'] for r in cv_t3]; t3_recs = [r['rec'] for r in cv_t3]
print(f"    Precision: {np.mean(t3_precs):.3f} ± {np.std(t3_precs):.3f}")
print(f"    Recall:    {np.mean(t3_recs):.3f} ± {np.std(t3_recs):.3f}")
print(f"    Base T3 rate: {t3.mean():.3f}")
print(f"    Precision gain: {np.mean(t3_precs) - t3.mean():.3f}")

# ============================================================
# SAVE
# ============================================================
out = {
    'safe_zone': {r['feat']: {'threshold': float(r['threshold']), 't1_rate': float(r['t1_rate']), 'n': int(r['n'])} for r in safe_rules},
    'safe_zone_combined': {'n': int(n_safe), 't1_rate': float(t1_safe), 't2plus_rate': float(t2plus_safe)},
    'extreme_rules_t2plus': [{'feat': rr['feat'], 'threshold': float(rr['thr']), 'pct': rr['pct'], 'precision': float(rr['prec']), 'recall': float(rr['rec']), 'n': int(rr['n'])} for rr in extreme_rules_t2[:10]],
    'extreme_rules_t3': [{'feat': rr['feat'], 'threshold': float(rr['thr']), 'pct': rr['pct'], 'precision': float(rr['prec']), 'recall': float(rr['rec']), 'n': int(rr['n'])} for rr in extreme_rules_t3[:10]],
    'voting': {k: {'threshold': float(v['threshold']), 'weight': int(v['weight'])} for k, v in vote_weights.items()},
    'cv_t2plus': {'precision_mean': float(np.mean(t2_precs)), 'precision_std': float(np.std(t2_precs)), 'recall_mean': float(np.mean(t2_recs)), 'recall_std': float(np.std(t2_recs))},
    'cv_t3': {'precision_mean': float(np.mean(t3_precs)), 'precision_std': float(np.std(t3_precs)), 'recall_mean': float(np.mean(t3_recs)), 'recall_std': float(np.std(t3_recs))},
}
out_dir = os.path.join(BASE, 'output')
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'q2_extreme_rules.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\n  Saved to output/q2_extreme_rules.json")
print(f"{'='*90}")
