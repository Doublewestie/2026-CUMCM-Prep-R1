"""
q2_final.py
Q2 最终模型 — 负反馈解耦 + LRV信号放大 + 自回归建模
=======================================================
三步走:
  1. 负反馈解耦: ALUM(t) = f(RW, FILT, 季节, 时间) → e_alum
  2. LRV变换: LRV = log10(RW / FILT) 放大信号
  3. LRV建模: LRV = AR + e_alum + RIVER_LEVEL + TW_FLOW + 季节

输出:
  - Q2答案: 时滞参数 + 影响力描述 + 精度验证表
  - 解耦验证: e_alum vs FILT互相关对比
  - LRV分析: 分布/时序/自相关
  - 最终精度: 还原FILT空间的RMSE/R2
"""

import os, json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import lightgbm as lgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
EPS = 1e-10
TSCV_SPLITS = 5


def load_data():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "clean_data.csv")).dropna(subset=["RW_NTU", "FILT_NTU", "NTU"]).copy()
    df["hour"] = (df["TIME"] / 100).astype(int)
    df = df.reset_index(drop=True)
    n = len(df)
    out = {
        "rw": df["RW_NTU"].values.astype(np.float64),
        "filt": df["FILT_NTU"].values.astype(np.float64),
        "out": df["NTU"].values.astype(np.float64),
        "alum": df["ALUM"].values.astype(np.float64),
        "rl": df["RIVER_LEVEL"].values.astype(np.float64),
        "cwl": df["CW_WELL_LEVEL"].values.astype(np.float64),
        "twf": df["TW_FLOW"].values.astype(np.float64),
        "rwf": df["RW_FLOW"].values.astype(np.float64),
        "rwclr": df["RW_CLR"].values.astype(np.float64),
        "month": df["MONTH"].values.astype(np.int32),
        "hour": df["hour"].values.astype(np.int32),
    }
    out["df"] = df
    out["n"] = n
    return out


def make_lag_feats(arr, lags):
    feats = []
    for lag in lags:
        s = np.roll(arr, lag)
        s[:lag] = np.nan
        feats.append(s)
    return np.column_stack(feats)


# ==============================
# 1. 负反馈解耦
# ==============================
def decouple_alum(data):
    """拟合操作员加矾行为: ALUM(t) = f(RW_NTU×lags, FILT×lags, 时间, 季节)
       返回 e_alum = ALUM - ALUM_pred 即为解耦后的真实扰动信号"""
    n, hour, month = data["n"], data["hour"], data["month"]
    feats, names = [], []

    for lag in [0, 1, 2, 3]:
        s = np.roll(data["rw"], lag); s[:lag] = np.nan
        feats.append(s); names.append(f"RW_lag{lag}")
    for lag in [0, 1, 2]:
        s = np.roll(data["filt"], lag); s[:lag] = np.nan
        feats.append(s); names.append(f"FILT_lag{lag}")
    s = np.roll(data["rwclr"], 0); feats.append(s); names.append("RW_CLR")
    drw = np.zeros(n); drw[1:] = np.diff(data["rw"])
    feats.append(drw); names.append("delta_RW")
    feats.append(np.sin(2 * np.pi * hour / 24)); names.append("hour_sin")
    feats.append(np.cos(2 * np.pi * hour / 24)); names.append("hour_cos")
    feats.append(np.sin(2 * np.pi * month / 12)); names.append("month_sin")
    feats.append(np.cos(2 * np.pi * month / 12)); names.append("month_cos")

    X = np.column_stack(feats)
    y = data["alum"].copy()
    valid = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y)
    Xv, yv = X[valid], y[valid]

    y_pred = np.full_like(yv, np.nan)
    tscv = TimeSeriesSplit(n_splits=TSCV_SPLITS)
    for tr, vl in tscv.split(Xv):
        m = lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                               random_state=42, verbose=-1)
        m.fit(Xv[tr], yv[tr])
        y_pred[vl] = m.predict(Xv[vl])

    e_alum = np.full(n, np.nan)
    e_alum[valid] = yv - y_pred

    # 解耦前后对比
    raw_cc, dec_cc = [], []
    for d in range(7):
        a_lag = np.roll(data["alum"], d)[d:n]
        e_lag = np.roll(e_alum, d)[d:n]
        f_slice = data["filt"][d:n]
        mask_raw = ~(np.isnan(a_lag) | np.isnan(f_slice))
        mask_dec = ~(np.isnan(e_lag) | np.isnan(f_slice))
        r1 = abs(pearsonr(a_lag[mask_raw], f_slice[mask_raw])[0]) if mask_raw.sum() > 10 else np.nan
        r2 = abs(pearsonr(e_lag[mask_dec], f_slice[mask_dec])[0]) if mask_dec.sum() > 10 else np.nan
        raw_cc.append(r1); dec_cc.append(r2)

    return e_alum, y_pred, valid, raw_cc, dec_cc


# ==============================
# 2. LRV 变换
# ==============================
def compute_lrv(rw, filt):
    return np.log10(np.maximum(rw, 0.01) / np.maximum(filt, EPS))


# ==============================
# 3. LRV建模 + FILT还原评估
# ==============================
def build_dataset(lrv, e_alum, data):
    """构造LRV建模完整特征矩阵"""
    n = data["n"]
    feats, names = [], []

    for lag in range(1, 7):
        s = np.roll(lrv, lag); s[:lag] = np.nan
        feats.append(s); names.append(f"LRV_lag{lag}")
    for lag in range(4):
        s = np.roll(e_alum, lag); s[:lag] = np.nan
        feats.append(s); names.append(f"eAlum_lag{lag}")
    feats.append(data["rl"]); names.append("RIVER_LEVEL")
    feats.append(data["twf"]); names.append("TW_FLOW")
    feats.append(np.sin(2 * np.pi * data["hour"] / 24)); names.append("hour_sin")
    feats.append(np.cos(2 * np.pi * data["hour"] / 24)); names.append("hour_cos")
    feats.append(np.sin(2 * np.pi * data["month"] / 12)); names.append("month_sin")
    feats.append(np.cos(2 * np.pi * data["month"] / 12)); names.append("month_cos")

    X = np.column_stack(feats)
    valid = ~np.any(np.isnan(X), axis=1) & ~np.isnan(lrv)
    return X[valid], lrv[valid], valid, names


def _clone_model(model):
    try:
        from sklearn.base import clone
        return clone(model)
    except:
        return model.__class__(**model.get_params())

def train_and_evaluate(model, model_name, X, y, valid_indices, data):
    """TimeSeriesSplit训练, 还原FILT空间评估"""
    n_splits = min(TSCV_SPLITS, len(X) // 20)
    if n_splits < 2:
        return None, None
    tscv = TimeSeriesSplit(n_splits=n_splits)
    yp = np.full_like(y, np.nan)
    print(f"  [{model_name}] {n_splits}-fold TS-CV ...", end=" ", flush=True)
    for tr, vl in tscv.split(X):
        m = _clone_model(model)
        m.fit(X[tr], y[tr])
        yp[vl] = m.predict(X[vl])

    # 还原FILT: 用valid_indices定位到原始数据
    rw_v = data["rw"][valid_indices]
    y_valid = np.where(np.isnan(yp), np.nan, y)[:len(yp)]
    filt_t = rw_v / (10 ** np.clip(y, 0, 10) + EPS)
    filt_p = rw_v / (10 ** np.clip(yp, 0, 10) + EPS)
    ok = ~np.isnan(filt_p) & ~np.isnan(filt_t) & (filt_p >= 0) & (filt_t >= 0)
    ft, fp = filt_t[ok], filt_p[ok]
    rmse = np.sqrt(np.mean((fp - ft) ** 2))
    r2 = 1 - np.sum((fp - ft) ** 2) / (np.sum((ft - np.mean(ft)) ** 2) + EPS)
    mae = np.mean(np.abs(fp - ft))
    print(f"RMSE={rmse:.4f} R2={r2:.4f} MAE={mae:.4f}")
    return {"rmse": rmse, "r2": r2, "mae": mae}, (ft, fp)


# ==============================
# 4. 主流程
# ==============================
def main():
    print("=" * 60)
    print("  Q2 Final: 负反馈解耦 + LRV建模")
    print("=" * 60)

    data = load_data()
    n = data["n"]
    print(f"  样本数: {n}")
    print(f"  RW_NTU: [{data['rw'].min():.0f}, {data['rw'].max():.0f}]")
    print(f"  FILT.NTU: [{data['filt'].min():.3f}, {data['filt'].max():.2f}]")

    # ===== 1. 解耦 =====
    print("\n" + "=" * 50)
    print("  Phase 1: 负反馈解耦")
    print("=" * 50)
    e_alum, alum_pred, dec_valid, raw_cc, dec_cc = decouple_alum(data)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    lags_h = np.arange(7) * 2
    axes[0].bar(lags_h, raw_cc, color="steelblue", alpha=0.7, width=1.5, label="raw ALUM")
    axes[0].bar(lags_h, dec_cc, color="darkorange", alpha=0.7, width=1.5, label="e_alum(解耦后)")
    axes[0].set_xlabel("lag (h)"); axes[0].set_ylabel("|corr| with FILT.NTU")
    axes[0].set_title("ALUM vs FILT: 解耦前后对比")
    axes[0].legend()
    axes[1].scatter(data["filt"][::20], data["alum"][::20], alpha=0.3, s=2, c="steelblue", label="raw ALUM")
    axes[1].scatter(data["filt"][::20], e_alum[::20], alpha=0.3, s=2, c="darkorange", label="e_alum")
    axes[1].set_xlabel("FILT.NTU"); axes[1].set_ylabel("ALUM / e_alum")
    axes[1].set_title("解耦散点对比 (x20降采样)")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "q2_decoupling.png"), dpi=150)
    plt.close()

    print(f"  原始 ALUM vs FILT max|r| = {max(raw_cc):.4f}")
    print(f"  解耦 e_alum vs FILT max|r| = {max(dec_cc):.4f}")
    print(f"  提升: {max(dec_cc) - max(raw_cc):+.4f}")
    print(f"  e_alum 最优lag = {np.argmax(dec_cc)}步 ({np.argmax(dec_cc)*2}h)")

    # ===== 2. LRV变换 =====
    print("\n" + "=" * 50)
    print("  Phase 2: LRV信号放大")
    print("=" * 50)
    lrv = compute_lrv(data["rw"], data["filt"])
    lrv_lag1 = pearsonr(lrv[:-1], lrv[1:])[0]
    filt_lag1 = pearsonr(data["filt"][:-1], data["filt"][1:])[0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(lrv, bins=100, color="steelblue", alpha=0.7, edgecolor="white")
    axes[0, 0].axvline(x=np.median(lrv), color="red", linestyle="--", label=f"median={np.median(lrv):.2f}")
    axes[0, 0].set_xlabel("LRV"); axes[0, 0].legend()

    axes[0, 1].scatter(data["filt"][::20], lrv[::20], alpha=0.2, s=2, c="steelblue")
    axes[0, 1].axvline(x=0.3, color="red", linestyle=":", alpha=0.5)
    axes[0, 1].set_xlabel("FILT.NTU"); axes[0, 1].set_ylabel("LRV")

    axes[1, 0].plot(lrv, color="steelblue", linewidth=0.4)
    axes[1, 0].axhline(y=np.median(lrv), color="red", linestyle="--")
    axes[1, 0].set_ylabel("LRV"); axes[1, 0].set_title("LRV时序 (全年)")

    ac = np.correlate(lrv - np.nanmean(lrv), lrv - np.nanmean(lrv), mode="full")
    ac = ac / ac.max()
    ac = ac[len(lrv) - 1:]
    axes[1, 1].plot(ac[:50], color="steelblue", linewidth=1)
    axes[1, 1].axhline(y=filt_lag1, color="gray", linestyle=":", label=f"FILT lag1={filt_lag1:.3f}")
    axes[1, 1].axhline(y=lrv_lag1, color="red", linestyle="--", label=f"LRV lag1={lrv_lag1:.3f}")
    axes[1, 1].set_xlabel("lag"); axes[1, 1].set_ylabel("autocorr"); axes[1, 1].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "q2_lrv_analysis.png"), dpi=150)
    plt.close()

    print(f"  FILT: lag1 autocorr={filt_lag1:.3f}  std={np.std(data['filt']):.4f}")
    print(f"  LRV:  lag1 autocorr={lrv_lag1:.3f}  std={np.std(lrv[~np.isnan(lrv)]):.4f}")

    # ===== 3. LRV建模 =====
    print("\n" + "=" * 50)
    print("  Phase 3: LRV建模 + FILT还原评估")
    print("=" * 50)
    X, y, valid_idx, feat_names = build_dataset(lrv, e_alum, data)
    valid_indices = np.arange(data["n"])[valid_idx]
    print(f"  LRV建模有效样本: {len(X)}")

    # Ridge
    ridge = Ridge(alpha=1.0, random_state=42)
    r_ridge, pred_data = train_and_evaluate(ridge, "Ridge", X, y, valid_indices, data)

    # LightGBM
    lgbm = lgb.LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.03,
                              random_state=42, verbose=-1, subsample=0.8, colsample_bytree=0.8,
                              reg_alpha=0.01, reg_lambda=0.01)
    r_lgbm, _ = train_and_evaluate(lgbm, "LightGBM", X, y, valid_indices, data)

    # Baseline: 不用LRV+解耦, 直接用AR(6) on FILT
    print(f"\n  基线 (from step2.2):")
    print(f"    AR(6) on FILT: RMSE=0.365  R2=0.519  (7参数)")

    # ===== 输出答案 =====
    print(f"\n{'='*60}")
    print("  Q2 Answer")
    print(f"{'='*60}")
    print(f"  1.1 FILT.NTU dynamic model")
    print(f"  Using LRV=log10(RW/FILT) + negative feedback decoupling e_alum.")
    print(f"  LRV(t) = sum a_k*LRV(t-k) + sum b_j*e_alum(t-j) + c1*RIVER_LEVEL(t) + c2*TW_FLOW(t) + eps")
    print(f"")
    print(f"  1.2 Time delay parameters:")
    print(f"    RW_NTU -> FILT: AR lag1 auto={lrv_lag1:.3f} (buffered by 99.7% removal)")
    print(f"    ALUM -> FILT:   opt lag={np.argmax(dec_cc):d} step(s) ({np.argmax(dec_cc)*2}h)")
    print(f"    e_alum max|r| = {max(dec_cc):.4f} (after decoupling)")
    print(f"")
    print(f"  2. Model validation (restored to FILT.NTU space):")

    # 打印对比表
    models_info = [
        ("LRV+Ridge (本模型)", r_ridge),
        ("LRV+LightGBM (本模型)", r_lgbm),
    ]
    print(f"  {'模型':<35s} {'RMSE':>8s} {'R2':>8s} {'MAE':>8s}")
    print(f"  {'-'*60}")
    for name, res in models_info:
        if res:
            print(f"  {name:<35s} {res['rmse']:>8.4f} {res['r2']:>8.4f} {res['mae']:>8.4f}")
    print(f"  {'AR(6) on FILT (基线)':<35s} {'0.3646':>8s} {'0.5193':>8s} {'0.1055':>8s}")

    # 变量重要性
    if r_ridge and r_lgbm:
        ridge_model = Ridge(alpha=1.0, random_state=42)
        ridge_model.fit(X, y)
        ridge_imp = sorted(zip(feat_names, np.abs(ridge_model.coef_)), key=lambda x: -x[1])

        lgbm_model = lgb.LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.03,
                                         random_state=42, verbose=-1)
        lgbm_model.fit(X, y)
        lgb_imp = sorted(zip(feat_names, lgbm_model.feature_importances_), key=lambda x: -x[1])

        print(f"\n  变量重要性 (Ridge |coef| 前10):")
        for name, c in ridge_imp[:10]:
            print(f"    {name:<15s}  {c:.4f}")
        print(f"\n  变量重要性 (LightGBM 前10):")
        for name, c in lgb_imp[:10]:
            print(f"    {name:<15s}  {c:.0f}")

    # 保存预测值
    if pred_data:
        filt_t, filt_p = pred_data
        pd.DataFrame({"true_FILT": filt_t, "pred_FILT_LRV_Ridge": filt_p}).to_csv(
            os.path.join(OUTPUT_DIR, "q2_final_predictions.csv"), index=False, encoding="utf-8-sig")

    # 保存JSON结果
    results = {}
    for name, res in models_info:
        if res:
            results[name] = res

    with open(os.path.join(OUTPUT_DIR, "q2_final_results.json"), "w", encoding="utf-8") as f:
        json.dump({
            "decoupling": {
                "raw_ALUM_maxr": max(raw_cc),
                "e_alum_maxr": max(dec_cc),
                "best_lag_steps": int(np.argmax(dec_cc)),
                "best_lag_hours": int(np.argmax(dec_cc) * 2),
            },
            "LRV": {
                "lag1_autocorr": round(lrv_lag1, 3),
                "FILT_lag1_autocorr": round(filt_lag1, 3),
                "std": round(np.nanstd(lrv), 3),
                "FILT_std": round(np.nanstd(data["filt"]), 3),
            },
            "models": {k: v for k, v in results.items()},
            "baseline_AR6": {"rmse": 0.3646, "r2": 0.5193},
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[Q2 Final] Done → q2_final_results.json, q2_final_predictions.csv")


if __name__ == "__main__":
    main()
