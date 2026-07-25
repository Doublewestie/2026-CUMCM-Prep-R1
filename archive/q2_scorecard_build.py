"""
q2_scorecard_build.py — Scorecard v2: Single-rule features + voting ensemble
=======================================================================
Improved: uses per-feature optimal threshold rules instead of quantile bins.
"""
import os, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
import json, warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data', '2025')
THETA = 0.04

# ============================================================
# 1. LOAD + FEATURE ENGINEERING
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

# Label
data['stress'] = (data['FILT_NTU'] > THETA).astype(int)
N_COMF = (data['stress'] == 0).sum()
N_STRESS = (data['stress'] == 1).sum()
BASE_STRESS_RATE = data['stress'].mean()
print(f"N={len(data)}  Comfort={N_COMF}  Stress={N_STRESS}  BaseStressRate={BASE_STRESS_RATE:.1%}")

# Fill NaN in raw features
raw_cols = ['RIVER_LEVEL','RW_NTU','RW_CLR','RW_FLOW','ALUM','CW_WELL_LEVEL','TW_FLOW']
for c in raw_cols:
    if c in data.columns:
        med = data[c].median()
        data[c] = data[c].fillna(med if not pd.isna(med) else 0)

# Derived features
data['eta_coag'] = np.clip((data['RW_NTU']-data['FILT_NTU'])/(data['RW_NTU']+1e-6), 0.5, 1.0)

# Interaction features (physics-driven)
data['LOAD']          = data['RW_NTU'] * data['RW_CLR']
data['STRESS_RATIO']  = data['RW_NTU'] / (data['ALUM'] + 1e-4)
data['FLUX']          = data['RW_NTU'] * data['RW_FLOW']
data['RIVER_x_NTU']   = data['RIVER_LEVEL'] * data['RW_NTU']
data['RIVER_x_FLOW']  = data['RIVER_LEVEL'] * data['RW_FLOW']
data['eta_gap']       = 1.0 - data['eta_coag']  # removal insufficiency
data['LOAD_x_RIVER']  = data['LOAD'] * data['RIVER_LEVEL']  # load amplified by river

# ============================================================
# 2. FEATURE POOL + SINGLE-RULE SCAN
# ============================================================
feature_pool = {
    'RIVER_LEVEL':     data['RIVER_LEVEL'].values,
    'RW_NTU':          data['RW_NTU'].values,
    'RW_CLR':          data['RW_CLR'].values,
    'RW_FLOW':         data['RW_FLOW'].values,
    'ALUM':            data['ALUM'].values,
    'LOAD':            data['LOAD'].values,
    'STRESS_RATIO':    data['STRESS_RATIO'].values,
    'FLUX':            data['FLUX'].values,
    'RIVER_x_NTU':     data['RIVER_x_NTU'].values,
    'RIVER_x_FLOW':    data['RIVER_x_FLOW'].values,
    'LOAD_x_RIVER':    data['LOAD_x_RIVER'].values,
}
# Binary features
feature_pool['NTU_HI']   = (data['RW_NTU'] > 80).astype(int)
feature_pool['RIVER_HI'] = (data['RIVER_LEVEL'] > 5.5).astype(int)
feature_pool['CLR_HI']   = (data['RW_CLR'] > 472).astype(int)
feature_pool['DANGER']   = ((data['RIVER_LEVEL'] > 5.5) & (data['RW_NTU'] > 80)).astype(int)

y = data['stress'].values
n = len(data)

# Scan each feature for best single threshold
print(f"\n{'='*85}")
print(f"  SINGLE-FEATURE RULE EVALUATION (threshold scan)")
print(f"{'='*85}")
print(f"{'Feature':<18} {'BestThr':>9} {'Rule':>8} {'Stress%':>8} {'AUC':>7} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N_rule':>6}")
print("-"*85)

rule_results = []
for name, vals in feature_pool.items():
    best_f1, best_thr, best_rule = 0, 0, '>'
    best_prec, best_rec, best_auc = 0, 0, 0.5
    best_stress_pct = BASE_STRESS_RATE

    # For binary features, threshold is fixed
    n_uniq = len(np.unique(vals))
    if n_uniq <= 2:
        for rule_dir in ['>', '<=']:
            pred = (vals > 0.5).astype(int) if rule_dir == '>' else (vals <= 0.5).astype(int)
            f1 = f1_score(y, pred, zero_division=0)
            prec = precision_score(y, pred, zero_division=0)
            rec = recall_score(y, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1; best_prec = prec; best_rec = rec; best_rule = rule_dir
                best_stress_pct = y[pred == 1].mean() if pred.sum() > 0 else 0
        try:
            best_auc = roc_auc_score(y, vals)
        except:
            best_auc = 0.5
    else:
        # Scan thresholds at percentiles
        for p in range(20, 95, 5):
            thr = np.percentile(vals, p)
            for rule_dir in ['>', '<=']:
                pred = (vals > thr).astype(int) if rule_dir == '>' else (vals <= thr).astype(int)
                f1 = f1_score(y, pred, zero_division=0)
                prec = precision_score(y, pred, zero_division=0)
                rec = recall_score(y, pred, zero_division=0)
                if f1 > best_f1 and prec > 0.66:  # require precision > 2/3
                    best_f1 = f1; best_prec = prec; best_rec = rec
                    best_thr = thr; best_rule = rule_dir
                    best_stress_pct = y[pred == 1].mean() if pred.sum() > 0 else 0
        try:
            best_auc = roc_auc_score(y, vals)
        except:
            best_auc = 0.5

    rule_pred = (vals > best_thr).astype(int) if best_rule == '>' else (vals <= best_thr).astype(int)
    n_rule = rule_pred.sum()

    if best_f1 > 0:
        bar = '#' * int(best_f1 * 20)
        print(f"{name:<18} {best_thr:>9.1f} {best_rule:>5}   {best_thr:>4.0f} {best_stress_pct:>7.1%} {best_auc:>7.4f} {best_prec:>6.3f} {best_rec:>6.3f} {best_f1:>6.3f} {n_rule:>6d} {bar}")
        rule_results.append({
            'name': name, 'threshold': best_thr, 'rule': best_rule,
            'auc': best_auc, 'precision': best_prec, 'recall': best_rec,
            'f1': best_f1, 'n_triggered': n_rule, 'stress_pct': best_stress_pct
        })

# ============================================================
# 3. FEATURE SELECTION (top by AUC, non-redundant)
# ============================================================
rule_results.sort(key=lambda x: -x['auc'])
selected = []
used_names = set()
for r in rule_results:
    if r['name'] in used_names:
        continue
    
    # Check correlation with already-selected features
    redundant = False
    for s in selected:
        if s['name'] in feature_pool and r['name'] in feature_pool:
            corr = np.corrcoef(feature_pool[r['name']], feature_pool[s['name']])[0, 1]
            if abs(corr) > 0.85:
                redundant = True
                break
    
    if not redundant:
        selected.append(r)
        used_names.add(r['name'])
        if len(selected) >= 6:
            break

print(f"\n{'='*85}")
print(f"  SELECTED SCORECARD FEATURES (AUC-sorted, |r|<0.85)")
print(f"{'='*85}")
# Build combined feature matrix for CV evaluation
selected_names = [s['name'] for s in selected]
X_sel = np.column_stack([feature_pool[name] for name in selected_names])
print(f"  Features: {selected_names}")

# ============================================================
# 4. SCORECARD CONSTRUCTION
# ============================================================
print(f"\n{'='*85}")
print(f"  SCORECARD (each rule triggers +1 point)")
print(f"{'='*85}")
print(f"{'Feature':<18} {'Rule':>20} {'Stress%':>8} {'Score':>6}")
print("-"*60)

def make_scorecard(data, feature_pool, selected):
    """Build scorecard: each rule adds 1 point when triggered"""
    scorecard = []
    for s in selected:
        name = s['name']
        thr = s['threshold']
        rule_dir = s['rule']
        vals = feature_pool[name]
        if rule_dir == '>':
            pred = (vals > thr).astype(int)
            rule_str = f'> {thr:.1f}'
        else:
            pred = (vals <= thr).astype(int)
            rule_str = f'<= {thr:.1f}'
        stress_when_true = y[pred == 1].mean() if pred.sum() > 0 else 0
        scorecard.append({'name': name, 'rule': rule_str, 'threshold': float(thr), 'direction': rule_dir})
        print(f"  {name:<18} {rule_str:>20} {stress_when_true:>7.1%}    +1")
    return scorecard

scorecard = make_scorecard(data, feature_pool, selected)

# Apply scorecard
total_scores = np.zeros(n)
for s in selected:
    vals = feature_pool[s['name']]
    thr = s['threshold']
    if s['rule'] == '>':
        total_scores += (vals > thr).astype(int)
    else:
        total_scores += (vals <= thr).astype(int)

print(f"\n  Max possible score: {len(selected)}")
print(f"  Score distribution:")
for p, label in [(10,'P10'),(25,'P25'),(50,'P50'),(75,'P75'),(90,'P90'),(95,'P95')]:
    print(f"    {label}: {np.percentile(total_scores, p):.1f}")

# Find optimal total score threshold
best_f1_score, best_score_thr = 0, 0
for thr in range(len(selected) + 1):
    pred = (total_scores >= thr).astype(int)
    f1 = f1_score(y, pred, zero_division=0)
    if f1 > best_f1_score:
        best_f1_score, best_score_thr = f1, thr

print(f"\n  Optimal total score threshold: >= {best_score_thr}")
pred_final = (total_scores >= best_score_thr).astype(int)
acc = accuracy_score(y, pred_final)
prec = precision_score(y, pred_final, zero_division=0)
rec = recall_score(y, pred_final, zero_division=0)
f1 = f1_score(y, pred_final, zero_division=0)
auc_score = roc_auc_score(y, total_scores)

print(f"\n  Performance (in-sample):")
print(f"    Accuracy:  {acc:.4f}")
print(f"    Precision: {prec:.4f}")
print(f"    Recall:    {rec:.4f}")
print(f"    F1:        {f1:.4f}")
print(f"    AUC:       {auc_score:.4f}")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, pred_final)
print(f"    Confusion: TP={cm[1,1]} FP={cm[0,1]} TN={cm[0,0]} FN={cm[1,0]}")

# Score distribution by actual zone
print(f"\n{'='*85}")
print(f"  SCORE DISTRIBUTION BY ZONE")
print(f"{'='*85}")
for label, mask in [('Comfort (<=0.04)', y == 0), ('Stress  (>0.04)', y == 1)]:
    s = total_scores[mask]
    hist = [f"{i}:{(s==i).mean()*100:.0f}%" for i in range(len(selected)+1)]
    print(f"\n  {label}: mean={s.mean():.2f}")
    print(f"    Score histogram: {' | '.join(hist)}")

# ============================================================
# 5. CROSS-VALIDATION (5-fold TimeSeriesSplit)
# ============================================================
print(f"\n{'='*85}")
print(f"  CROSS-VALIDATION (5-fold TimeSeriesSplit)")
print(f"{'='*85}")

tscv = TimeSeriesSplit(n_splits=5)
cv_scores = np.zeros(n)
cv_total = np.zeros(n)

for tr, va in tscv.split(data):
    y_tr, y_va = y[tr], y[va]
    
    # Re-scan thresholds on training set only
    va_scores = np.zeros(len(va))
    for s in selected:
        name = s['name']
        vals_tr = feature_pool[name][tr]
        vals_va = feature_pool[name][va]
        
        # Find best threshold on training set
        best_f1_cv, best_thr_cv, best_dir_cv = 0, 0, '>'
        for p in range(20, 95, 5):
            thr_cv = np.percentile(vals_tr, p)
            for rdir in ['>', '<=']:
                pred_cv = (vals_tr > thr_cv).astype(int) if rdir == '>' else (vals_tr <= thr_cv).astype(int)
                f1_cv = f1_score(y_tr, pred_cv, zero_division=0)
                if f1_cv > best_f1_cv:
                    best_f1_cv = f1_cv
                    best_thr_cv = thr_cv
                    best_dir_cv = rdir
        
        if best_dir_cv == '>':
            va_scores += (vals_va > best_thr_cv).astype(int)
        else:
            va_scores += (vals_va <= best_thr_cv).astype(int)
    
    cv_total[va] = va_scores

# CV AUC
cv_auc = roc_auc_score(y, cv_total)
# Find optimal CV threshold
best_cv_f1, best_cv_thr = 0, 0
for thr in range(len(selected) + 1):
    pred_cv = (cv_total >= thr).astype(int)
    f1_cv = f1_score(y, pred_cv, zero_division=0)
    if f1_cv > best_cv_f1:
        best_cv_f1, best_cv_thr = f1_cv, thr

cv_pred = (cv_total >= best_cv_thr).astype(int)
cv_acc = accuracy_score(y, cv_pred)
cv_prec = precision_score(y, cv_pred, zero_division=0)
cv_rec = recall_score(y, cv_pred, zero_division=0)
cv_f1 = f1_score(y, cv_pred, zero_division=0)

print(f"  CV Threshold: >= {best_cv_thr}")
print(f"  CV Accuracy:  {cv_acc:.4f}")
print(f"  CV Precision: {cv_prec:.4f}")
print(f"  CV Recall:    {cv_rec:.4f}")
print(f"  CV F1:        {cv_f1:.4f}")
print(f"  CV AUC:       {cv_auc:.4f}")

# ============================================================
# 6. SAVE
# ============================================================
out_dir = os.path.join(BASE, 'output')
os.makedirs(out_dir, exist_ok=True)
result = {
    'theta': THETA,
    'n_comfort': int(N_COMF), 'n_stress': int(N_STRESS),
    'base_stress_rate': float(BASE_STRESS_RATE),
    'n_rules': len(selected),
    'scorecard': [{'feature': s['name'], 'rule': s['rule'], 'threshold': float(s['threshold']), 'direction': s['direction']} for s in scorecard],
    'optimal_score_threshold': best_score_thr,
    'in_sample': {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc_score},
    'cv': {'threshold': best_cv_thr, 'accuracy': cv_acc, 'precision': cv_prec, 'recall': cv_rec, 'f1': cv_f1, 'auc': cv_auc},
    'selected_features': selected_names,
    'all_rules': [{k: float(v) if isinstance(v, (np.floating, float, np.integer)) else v for k, v in rr.items() if k != 'name'} for rr in rule_results],
}
with open(os.path.join(out_dir, 'q2_scorecard_metrics.json'), 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"\n  Saved to output/q2_scorecard_metrics.json")
print(f"{'='*85}")
