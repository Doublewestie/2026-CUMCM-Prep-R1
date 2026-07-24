"""
step1.4_feature_importance.py — T3 Feature Importance (SHAP + Permutation)
============================================================================
Only on T3 (>0.15) subset, with focused physical feature set.
"""
import numpy as np, pandas as pd, os, warnings
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
from q1_data_utils import load_clean_data, add_tier_labels
from step0_config import *
warnings.filterwarnings("ignore")

def build_t3_features(df):
    """Focused feature set for T3 factor analysis"""
    feats = pd.DataFrame(index=df.index)
    # Core physical variables
    feats["RW_NTU"] = df["RW_NTU"]
    feats["RW_FLOW"] = df["RW_FLOW"]
    feats["RIVER_LEVEL"] = df["RIVER_LEVEL"]
    feats["ALUM"] = df["ALUM"]
    feats["CLR"] = df["CLR"]
    feats["CW_WELL_LEVEL"] = df["CW_WELL_LEVEL"]
    feats["TW_FLOW"] = df["TW_FLOW"]
    feats["PH"] = df["PH"]
    feats["RW_CLR"] = df["RW_CLR"]
    # Derived physical
    feats["eta_coag"] = (df["RW_NTU"] - df["FILT_NTU"]) / (df["RW_NTU"] + 1e-6)
    feats["phi_alum"] = df["ALUM"] / (df["RW_NTU"] + 1e-6)
    # Lag (limited)
    feats["RW_NTU_lag3"] = df["RW_NTU"].shift(3)
    feats["RW_FLOW_lag3"] = df["RW_FLOW"].shift(3)
    # Agg (limited)
    feats["RW_NTU_mean6"] = df["RW_NTU"].rolling(6, min_periods=1).mean()
    feats["FILT_NTU_mean6"] = df["FILT_NTU"].rolling(6, min_periods=1).mean()
    # Time
    feats["day_sin"] = df["day_sin"]
    feats["day_cos"] = df["day_cos"]
    return feats.fillna(0)

def main():
    print("=" * 60)
    print("  step1.4 — T3 Feature Importance")
    print("=" * 60)

    df = load_clean_data()
    df = add_tier_labels(df)
    mask = df["tier"] == 3
    sub = df[mask].copy()
    print(f"\n  T3 samples: {len(sub)}")

    X = build_t3_features(sub)
    y = sub["NTU"].values  # use raw NTU for interpretability
    names = X.columns.tolist()

    print(f"  Features: {X.shape[1]}")

    # XGBoost
    xgb = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1,
                       random_state=42, verbosity=0)
    xgb.fit(X.values, y)

    # SHAP
    print("\n  Computing SHAP...")
    import shap
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X.values)
    phi_j = np.abs(shap_values).mean(axis=0)
    phi_norm = phi_j / (phi_j.sum() + 1e-10)

    # Permutation
    print("  Computing Permutation Importance (×10)...")
    perm = permutation_importance(xgb, X.values, y, n_repeats=10,
                                  random_state=42, n_jobs=-1,
                                  scoring="neg_root_mean_squared_error")
    I_j = np.maximum(perm.importances_mean, 0)
    I_norm = I_j / (I_j.sum() + 1e-10)

    # Robust fusion
    robust = np.sqrt(phi_norm * I_norm)

    df_imp = pd.DataFrame({
        "feature": names, "shap_phi": phi_j.round(6), "shap_norm": phi_norm.round(4),
        "perm_I": I_j.round(6), "perm_norm": I_norm.round(4),
        "robust": robust.round(4)
    }).sort_values("robust", ascending=False).reset_index(drop=True)
    df_imp["rank"] = range(1, len(df_imp) + 1)

    df_imp.to_csv(OUT_TIER_FACTOR, index=False, encoding="utf-8-sig")

    print(f"\n  {'Rank':<6} {'Feature':<22} {'SHAP_norm':<10} {'Perm_norm':<10} {'Robust':<8}")
    print(f"  {'-'*56}")
    for _, r in df_imp.head(15).iterrows():
        print(f"  {r['rank']:<6.0f} {r['feature']:<22} {r['shap_norm']:<10.4f} {r['perm_norm']:<10.4f} {r['robust']:<8.4f}")

    print(f"\n[DONE] {OUT_TIER_FACTOR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
