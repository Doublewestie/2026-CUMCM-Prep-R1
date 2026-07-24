"""
step1.0_tier_classifier.py — Three-Tier Classifier (C1 + C2)
=============================================================
Input: clean_data.csv
Output: tier_labels.npy, tier_classifier.pkl, tier_params.json
"""
import numpy as np, pandas as pd, os, json, joblib, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from q1_data_utils import load_clean_data, add_tier_labels
from step0_config import *
warnings.filterwarnings("ignore")

def build_features_for_classifier(df):
    feats = pd.DataFrame(index=df.index)
    for c in ["RW_NTU","RW_FLOW","RIVER_LEVEL","RW_CLR","ALUM","CW_WELL_LEVEL"]:
        if c in df.columns:
            feats[c] = df[c].values
    feats["FILT_NTU_lag1"] = df["FILT_NTU"].shift(1).fillna(df["FILT_NTU"].median())
    feats["FILT_NTU_lag3"] = df["FILT_NTU"].shift(3).fillna(df["FILT_NTU"].median())
    feats["day_sin"] = df["day_sin"].values
    feats["day_cos"] = df["day_cos"].values
    feats["month_sin"] = df["month_sin"].values
    feats["month_cos"] = df["month_cos"].values
    return feats.fillna(feats.median())

def main():
    print("=" * 60)
    print("  step1.0 — Three-Tier Classifier")
    print("=" * 60)
    df = load_clean_data()
    df = add_tier_labels(df)
    y = df["tier"].values
    X = build_features_for_classifier(df)
    n = len(df)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    # C1: FILT > 0.05? (T1 vs T2+T3)
    y_c1 = (y > 1).astype(int)
    c1_probs = np.zeros(n)
    c1_accs = []
    for tr, va in tscv.split(X):
        m = LogisticRegression(max_iter=1000, class_weight="balanced")
        m.fit(X.iloc[tr], y_c1[tr])
        c1_probs[va] = m.predict_proba(X.iloc[va])[:, 1]
        acc = accuracy_score(y_c1[va], m.predict(X.iloc[va]))
        c1_accs.append(acc)
    c1_pred = (c1_probs > 0.5).astype(int)
    print(f"\n[C1] FILT > 0.05?  CV Acc: {np.mean(c1_accs):.4f}+-{np.std(c1_accs):.4f}")
    print(f"      Precision/Recall (class 1):")
    print(f"      {precision_recall_fscore_support(y_c1, c1_pred, average='binary')}")

    # C2: FILT > 0.15? (T2 vs T3, only on samples where y>1)
    y_c2 = (y > 2).astype(int)  # full length, only used for mask_c2
    mask_c2 = y > 1
    c2_probs = np.full(n, 0.5)
    c2_accs = []
    for tr, va in tscv.split(X):
        tr_set = set(tr)
        va_set = set(va)
        train_idx = sorted([i for i in tr_set if mask_c2[i]])
        val_idx = sorted([i for i in va_set if mask_c2[i]])
        if len(train_idx) < 10 or len(val_idx) < 5:
            continue
        m2 = LogisticRegression(max_iter=1000, class_weight="balanced")
        m2.fit(X.iloc[train_idx], y_c2[train_idx])
        pred = m2.predict_proba(X.iloc[val_idx])[:, 1]
        for idx, p in zip(val_idx, pred):
            c2_probs[idx] = p
        acc = accuracy_score(y_c2[val_idx], m2.predict(X.iloc[val_idx]))
        c2_accs.append(acc)
    c2_pred = np.zeros(n, dtype=int)
    c2_pred[mask_c2] = (c2_probs[mask_c2] > 0.5).astype(int)
    if c2_accs:
        print(f"\n[C2] FILT > 0.15? CV Acc: {np.mean(c2_accs):.4f}+-{np.std(c2_accs):.4f}")

    # Final tier prediction
    y_pred = np.ones(n, dtype=int)
    y_pred[c1_pred == 1] = 2
    y_pred[(c1_pred == 1) & (c2_pred == 1)] = 3

    overall_acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred, labels=[1,2,3])
    print(f"\n[Overall] 3-Tier Accuracy: {overall_acc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    per_class = precision_recall_fscore_support(y, y_pred, labels=[1,2,3])
    print(f"Per-class precision: {per_class[0].round(3)}")
    print(f"Per-class recall:    {per_class[1].round(3)}")

    # Save
    np.save(OUT_TIER_LABELS, y)
    result = {
        "thresholds": TIER_THRESHOLDS,
        "c1_mean_acc": float(np.mean(c1_accs)),
        "c2_mean_acc": float(np.mean(c2_accs)) if c2_accs else 0,
        "overall_acc": float(overall_acc),
        "confusion_matrix": cm.tolist(),
        "tier_counts": {f"T{i+1}": int((y==i+1).sum()) for i in range(3)},
        "tier_pcts": {f"T{i+1}": round((y==i+1).sum()/n*100, 1) for i in range(3)},
    }
    with open(OUT_TIER_PARAMS, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[DONE] {OUT_TIER_PARAMS}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
