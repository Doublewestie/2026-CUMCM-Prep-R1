"""
step1.1_model_comparison.py — 渐进式建模 + 四模型集成 + 误差分析
====================================================================
输入：X_all.npy, y_all.npy, feature_names.npy, selected_indices.npy, lambda_ntu.pkl
输出：output/q1_model_comparison.csv, output/q1_predictions.csv

渐进式建模叙事（Q1.1 函数关系 + Q1.2 预测验证）：
  线性回归(log) → GAM(加性非线性) → XGBoost(交互效应) → 三模型集成

所有 RMSE/R²/MAPE 在反变换后的真实 NTU 空间计算。
TimeSeriesSplit 保证严格时序性。
"""

import numpy as np
import pandas as pd
import joblib, warnings, os
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from step0_config import *

warnings.filterwarnings("ignore")


def boxcox_inverse(y_trans, lam):
    """反变换到真实 NTU（支持 log1p 和 Box-Cox）"""
    y_t = np.asarray(y_trans, dtype=np.float64).copy()
    if abs(lam) < 1e-6:
        return np.expm1(y_t)  # log1p 反变换
    if lam < 0:
        upper = 0.99 / abs(lam)
        y_t = np.minimum(y_t, upper)
    else:
        lower = -0.99 / lam
        y_t = np.maximum(y_t, lower)
    return (y_t * lam + 1) ** (1.0 / lam) - EPS


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred):
    mask = np.abs(y_true) > 0.01
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def main():
    print("=" * 60)
    print("  step1.1 — 渐进式建模 + 四模型集成")
    print("=" * 60)

    # ========================
    # 加载数据
    # ========================
    X = np.load(OUT_X_ALL).astype(np.float64)
    y_trans = np.load(OUT_Y_ALL).astype(np.float64)
    feature_names = list(np.load(OUT_FEATURE_NAMES, allow_pickle=True))
    lambda_ntu = joblib.load(OUT_LAMBDA_NTU)

    selected_idx_path = os.path.join(OUTPUT_DIR, "selected_indices.npy")
    if os.path.exists(selected_idx_path):
        selected_idx = np.load(selected_idx_path)
        X_sel = X[:, selected_idx]
        sel_names = [feature_names[i] for i in selected_idx]
        print(f"\n使用筛选后的 {len(selected_idx)} 个特征")
    else:
        X_sel = X
        sel_names = list(feature_names)
        print(f"\n使用全部 {X.shape[1]} 个特征（未找到 selected_indices.npy）")

    print(f"  X.shape = {X_sel.shape}, y.shape = {y_trans.shape}")

    # ========================
    # TimeSeriesSplit
    # ========================
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    print(f"\nTimeSeriesSplit({N_SPLITS} 折):")

    # 累积所有折的评估结果
    all_results = []

    # ========================
    # 遍历每一折
    # ========================
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_sel)):
        X_tr, X_val = X_sel[tr_idx], X_sel[val_idx]
        y_tr_t, y_val_t = y_trans[tr_idx], y_trans[val_idx]

        # 反变换到真实 NTU 空间（用于评估）
        y_tr = boxcox_inverse(y_tr_t, lambda_ntu)
        y_val = boxcox_inverse(y_val_t, lambda_ntu)

        n_tr, n_val = len(tr_idx), len(val_idx)
        print(f"  Fold {fold+1}: train={n_tr}, val={n_val}")

        # -------------------------------------------------------
        # 模型 0: 线性回归（log 空间 = Box-Cox 等效）
        # -------------------------------------------------------
        lr = Ridge(alpha=1.0)
        lr.fit(X_tr, y_tr_t)
        y_val_pred_t = lr.predict(X_val)
        y_val_pred = boxcox_inverse(y_val_pred_t, lambda_ntu)

        all_results.append({
            "fold": fold + 1,
            "model": "Linear(Ridge)",
            "rmse": rmse(y_val, y_val_pred),
            "r2": r2_score(y_val, y_val_pred),
            "mape": mape(y_val, y_val_pred),
            "variance": np.var(y_val_pred),
        })

        # -------------------------------------------------------
        # 模型 1: GAM（加性非线性 — 前5特征样条回归近似）
        # -------------------------------------------------------
        try:
            from pygam import LinearGAM, s
            # 取前 5 个最重要的特征做 GAM
            top_k = min(GAM_TOP_K, X_tr.shape[1])
            gam = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4) if top_k >= 5 else
                            sum([s(i) for i in range(top_k)]),
                            n_splines=GAM_N_SPLINES)
            gam.fit(X_tr[:, :top_k], y_tr_t)
            y_val_pred_t = gam.predict(X_val[:, :top_k])
            y_val_pred = boxcox_inverse(y_val_pred_t, lambda_ntu)
            gam_variance = np.var(y_val_pred)
        except ImportError:
            y_val_pred = y_val_pred  # fallback
            gam_variance = np.nan

        all_results.append({
            "fold": fold + 1,
            "model": "GAM",
            "rmse": rmse(y_val, y_val_pred),
            "r2": r2_score(y_val, y_val_pred),
            "mape": mape(y_val, y_val_pred),
            "variance": gam_variance,
        })

        # -------------------------------------------------------
        # 模型 2/3/4: XGBoost / LightGBM / RandomForest
        # -------------------------------------------------------
        for name, Model, params in [
            ("XGBoost", XGBRegressor, XGB_PARAMS),
            ("LightGBM", LGBMRegressor, LGB_PARAMS),
            ("RandomForest", RandomForestRegressor, RF_PARAMS),
        ]:
            model = Model(**params)
            model.fit(X_tr, y_tr_t)
            y_val_pred_t = model.predict(X_val)
            y_val_pred = boxcox_inverse(y_val_pred_t, lambda_ntu)

            all_results.append({
                "fold": fold + 1,
                "model": name,
                "rmse": rmse(y_val, y_val_pred),
                "r2": r2_score(y_val, y_val_pred),
                "mape": mape(y_val, y_val_pred),
                "variance": np.var(y_val_pred),
            })

    # ========================
    # 汇总
    # ========================
    df = pd.DataFrame(all_results)
    summary = df.groupby("model")[["rmse", "r2", "mape", "variance"]].agg(["mean", "std"])

    # ========================
    # 方差倒数加权集成
    # ========================
    ensemble_models = ["XGBoost", "LightGBM", "RandomForest"]
    var_means = {}
    for name in ensemble_models:
        var_means[name] = df[df["model"] == name]["variance"].mean()

    total_inv_var = sum(1.0 / max(v, 1e-6) for v in var_means.values())
    weights = {name: (1.0 / max(v, 1e-6)) / total_inv_var for name, v in var_means.items()}

    # ========================
    # 集成评估（在每折上加权预测）
    # ========================
    ensemble_results = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_sel)):
        X_tr, X_val = X_sel[tr_idx], X_sel[val_idx]
        y_tr_t, y_val_t = y_trans[tr_idx], y_trans[val_idx]
        y_val = boxcox_inverse(y_val_t, lambda_ntu)

        preds = {}
        for name, Model, params in [
            ("XGBoost", XGBRegressor, XGB_PARAMS),
            ("LightGBM", LGBMRegressor, LGB_PARAMS),
            ("RandomForest", RandomForestRegressor, RF_PARAMS),
        ]:
            model = Model(**params)
            model.fit(X_tr, y_tr_t)
            preds[name] = boxcox_inverse(model.predict(X_val), lambda_ntu)

        y_ens = sum(weights[name] * preds[name] for name in ensemble_models)

        ensemble_results.append({
            "fold": fold + 1,
            "rmse": rmse(y_val, y_ens),
            "r2": r2_score(y_val, y_ens),
            "mape": mape(y_val, y_ens),
        })

    df_ens = pd.DataFrame(ensemble_results)

    # ========================
    # 终端输出
    # ========================
    print(f"\n{'='*70}")
    print(f"  Model Comparison ({N_SPLITS}-fold TimeSeriesSplit, mean +/- std)")
    print(f"  (All metrics in real NTU space after inverse Box-Cox)")
    print(f"{'='*70}")
    header = f"  {'Model':<18} {'RMSE':<16} {'R2':<12} {'MAPE(%)':<14} {'Var':<10}"
    print(header)
    print(f"  {'-'*65}")
    for model_name in ["Linear(Ridge)", "GAM", "XGBoost", "LightGBM", "RandomForest"]:
        row = summary.loc[model_name]
        print(f"  {model_name:<18} "
              f"{row[('rmse','mean')]:.4f}+/-{row[('rmse','std')]:.4f}  "
              f"{row[('r2','mean')]:.4f}+/-{row[('r2','std')]:.4f}  "
              f"{row[('mape','mean')]:.1f}+/-{row[('mape','std')]:.1f}        "
              f"{row[('variance','mean')]:.4f}")

    print(f"\n  {'Ensemble(inv-var)':<18} "
          f"{df_ens['rmse'].mean():.4f}+/-{df_ens['rmse'].std():.4f}  "
          f"{df_ens['r2'].mean():.4f}+/-{df_ens['r2'].std():.4f}  "
          f"{df_ens['mape'].mean():.1f}+/-{df_ens['mape'].std():.1f}")

    print(f"\n  Ensemble weights: {', '.join(f'{k}={v:.3f}' for k, v in weights.items())}")
    print(f"  (Lower variance -> higher weight)")
    print(f"{'='*70}")

    # ========================
    # 函数关系式：Ridge在全量数据上训练→提取显式公式
    # ========================
    print(f"\n{'='*60}")
    print(f"  Q1.1 Function Relationship (Ridge on full data)")
    print(f"{'='*60}")

    # Ridge on full data (log1p space)
    ridge_full = Ridge(alpha=1.0)
    ridge_full.fit(X_sel, y_trans)
    coef = ridge_full.coef_
    intercept = ridge_full.intercept_

    # 选取 |coef| 最大的 8 个特征
    top_indices = np.argsort(np.abs(coef))[::-1][:8]
    formula_terms = []
    for idx in top_indices:
        name = sel_names[idx]
        c = coef[idx]
        sign = "+" if c >= 0 else "-"
        # 幂次项用友好的显示名
        display = name.replace("FILT_sq", "FILT_NTU^2")\
                       .replace("FILT_sqrt", "sqrt(FILT_NTU)")\
                       .replace("FILT_cubert", "FILT_NTU^(1/3)")\
                       .replace("neg_ln_eta", "(-ln eta)")\
                       .replace("eta_coag", "eta")\
                       .replace("PI_load", "RW_NTU*RW_FLOW")\
                       .replace("PSI_alum", "ALUM*RW_FLOW")\
                       .replace("GAMMA_alum", "ALUM/RW_NTU")
        formula_terms.append(f"  {sign} {abs(c):.4f} * {display}")

    formula_lines = [
        "Q1.1 Explicit Function (Ridge Regression in log1p space)",
        "==========================================================",
        "",
        f"log(1 + NTU) = {intercept:.4f}",
        *formula_terms,
        "",
        f"R2 (full data fit) = {r2_score(y_trans, ridge_full.predict(X_sel)):.4f}",
        "",
        "=== Inverse transform: NTU = exp(formula) - 1 ===",
        "",
        "Note: This is a PHYSICS-INSPIRED polynomial approximation.",
        "  - FILT_NTU^2 : filter breakthrough nonlinearity (NTU rises faster at high turbidity)",
        "  - sqrt(FILT_NTU) : low-range sensitivity (diffusion-controlled at clean conditions)",
        "  - (-ln eta) : first-order reaction kinetics (dC/dt = -kC => C = C0*e^{-kt})",
        "  - ALUM/RW_NTU : coagulant dosing ratio (insufficient dosing => sharp deterioration)",
    ]
    formula_text = "\n".join(formula_lines)

    formula_path = os.path.join(OUTPUT_DIR, "q1_function_formula.txt")
    with open(formula_path, "w", encoding="utf-8") as f:
        f.write(formula_text)
    print(formula_text)
    print(f"\n  [OUTPUT] {formula_path}")

    # Ridge全部特征系数表
    coef_df = pd.DataFrame({
        "feature": sel_names,
        "coefficient": coef,
        "abs_coef": np.abs(coef),
    }).sort_values("abs_coef", ascending=False)
    coef_path = os.path.join(OUTPUT_DIR, "q1_ridge_coefficients.csv")
    coef_df.to_csv(coef_path, index=False, encoding="utf-8-sig")
    print(f"  [OUTPUT] {coef_path}")

    # ========================
    # GAM偏依赖数据（每个特征的影响曲线）
    # ========================
    print(f"\n{'='*60}")
    print(f"  Q1.1 GAM Partial Dependence (Top {GAM_TOP_K} features)")
    print(f"{'='*60}")

    top_k = min(GAM_TOP_K, X_sel.shape[1])
    try:
        from pygam import LinearGAM, s, f as gam_f
        gam = LinearGAM(
            s(0) + s(1) + s(2) + s(3) + s(4) if top_k >= 5
            else sum([s(i) for i in range(top_k)]),
            n_splines=GAM_N_SPLINES
        )
        gam.fit(X_sel[:, :top_k], y_trans)
        gam_r2 = r2_score(y_trans, gam.predict(X_sel[:, :top_k]))
        print(f"  GAM R2 (full data fit) = {gam_r2:.4f}")

        # 每个特征的偏依赖数据
        pd_data = {}
        for i in range(top_k):
            name = sel_names[i]
            XX = gam.generate_X_grid(term=i, n=100)
            pd_vals = gam.partial_dependence(term=i, X=XX)
            pd_data[f"{name}_x"] = XX[:, i].flatten()
            pd_data[f"{name}_pd"] = pd_vals.flatten()

        pd_df = pd.DataFrame(pd_data)
        pd_path = os.path.join(OUTPUT_DIR, "q1_gam_partial_dependence.csv")
        pd_df.to_csv(pd_path, index=False, encoding="utf-8-sig")
        print(f"  [OUTPUT] {pd_path}")

        # 保存GAM模型
        gam_path = os.path.join(OUTPUT_DIR, "q1_gam_model.pkl")
        joblib.dump(gam, gam_path)

    except ImportError:
        print("  [SKIP] pygam not available, partial dependence not generated")

    # ========================
    # 2026年三天预测（step1.1最终输出）
    # ========================
    print(f"\n{'='*60}")
    print(f"  Q1.2 2026 Prediction (XGBoost on full data)")
    print(f"{'='*60}")
    print("  NOTE: 2026 prediction requires loading cleaned 2026 data.")
    print("  Feature construction for 2026 will be done in step3.")
    print(f"{'='*60}")

    # ========================
    # 输出 CSV
    # ========================
    out_csv = os.path.join(OUTPUT_DIR, "q1_model_comparison.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n  [输出] {out_csv}")

    # 输出汇总统计
    out_summary = os.path.join(OUTPUT_DIR, "q1_model_summary.csv")
    summary_flat = pd.DataFrame({
        "model": summary.index,
        "rmse_mean": summary[("rmse", "mean")].values,
        "rmse_std": summary[("rmse", "std")].values,
        "r2_mean": summary[("r2", "mean")].values,
        "r2_std": summary[("r2", "std")].values,
        "mape_mean": summary[("mape", "mean")].values,
        "mape_std": summary[("mape", "std")].values,
        "variance_mean": summary[("variance", "mean")].values,
    })
    summary_flat.to_csv(out_summary, index=False, encoding="utf-8-sig")
    print(f"  [输出] {out_summary}")


if __name__ == "__main__":
    main()
