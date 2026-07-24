"""
step2.0_time_delay_estimation.py
Q2 时滞参数估计 — MIC + TE + CCF 三方法交叉验证
=====================================================
输入: clean_data.csv (原始清洗后数据)
输出: tau_params.json, tau_analysis.csv, tau_comparison.csv
       figures/tau_combined.png, figures/tau_seasonal.png

方法:
  CCF  — Pearson 互相关 (线性基线)
  MIC  — 最大信息系数 (非线性统计依赖)
  TE   — 传递熵 (因果方向性) + Surrogate 检验 (p < 0.05)
  TE_seg — TE 按季节分段(高浊季 / 低浊季 / 过渡季)

融合: d* = argmax MIC(d) × I[TE_pval(d) < 0.05]
ALUM: 代理线性模型扫参 d ∈ {0..6} → 取 argmin RMSE
"""

import os, json, sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, entropy
from scipy.signal import correlate
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ==============================
# 配置
# ==============================
MAX_LAG = 6            # 候选时滞 0~6 步 (0~12h)
N_SHUFFLE = 50         # Surrogate 检验洗牌次数
N_BINS = 15            # TE 离散化分箱数
N_SPLITS = 5           # ALUM 扫参交叉验证折数
EPS = 1e-6

INPUT_VARS = ["RW_NTU", "RW_FLOW", "RW_PH"]
VAR_LABELS = {"RW_NTU": "R/W NTU → FILT.NTU",
              "RW_FLOW": "R/W FLOW → FILT.NTU",
              "RW_PH": "R/W PH → FILT.NTU",
              "ALUM": "ALUM → FILT.NTU"}

SEASON_MAP = {
    "high":  [6, 7, 8],                  # 高浊季 (7-9月, 0-indexed)
    "low":   [0, 1, 2, 11],              # 低浊季 (1-3月, 12月, 0-indexed)
    "trans": [3, 4, 5, 9, 10],           # 过渡季 (4-6月, 10-11月, 0-indexed)
}


# ==============================
# 数据加载
# ==============================
def load_data():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "clean_data.csv"))
    cols_need = ["RW_NTU", "RW_FLOW", "RW_PH", "ALUM", "FILT_NTU", "MONTH"]
    for c in cols_need:
        if c not in df.columns:
            raise KeyError(f"clean_data.csv 缺少列: {c}")
    df = df[cols_need].copy()
    df = df.dropna()
    x_all = {}
    for c in ["RW_NTU", "RW_FLOW", "RW_PH", "ALUM"]:
        x_all[c] = df[c].values.astype(np.float64)
    y = df["FILT_NTU"].values.astype(np.float64)
    months = df["MONTH"].values.astype(np.int32)
    return x_all, y, months


# ==============================
# 交叉相关 CCF
# ==============================
def calc_ccf(x, y, max_lag):
    scores = np.zeros(max_lag + 1)
    n = len(y)
    for d in range(max_lag + 1):
        if d >= n - 1:
            scores[d] = -1
            continue
        valid = ~np.isnan(x[d:]) & ~np.isnan(y[:n - d])
        if np.sum(valid) < 10:
            scores[d] = -1
            continue
        r, _ = pearsonr(x[d:][valid], y[:n - d][valid])
        scores[d] = abs(r)
    return scores


# ==============================
# 最大信息系数 MIC (minepy 无依赖备选)
# ==============================
def _try_minepy():
    try:
        from minepy import MINE
        return True
    except ImportError:
        return False

_HAS_MINEPY = _try_minepy()

def _mic_minepy(x, y):
    from minepy import MINE
    m = MINE(alpha=0.6, c=15)
    m.compute_score(x, y)
    return m.mic()

def _mic_fallback(x, y):
    n = len(x)
    x_bins = max(3, min(int(np.sqrt(n)), 50))
    y_bins = max(3, min(int(np.sqrt(n)), 50))
    h2d, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
    joint = h2d / np.sum(h2d)
    px = np.sum(joint, axis=1)
    py = np.sum(joint, axis=0)
    mi = 0.0
    for i in range(x_bins):
        for j in range(y_bins):
            if joint[i, j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (px[i] * py[j] + 1e-12) + 1e-12)
    mic = mi / np.log2(min(x_bins, y_bins) + 1e-12)
    return max(0, min(1, mic))

def calc_mic(x, y, max_lag):
    scores = np.zeros(max_lag + 1)
    n = len(y)
    for d in range(max_lag + 1):
        if d >= n - 1:
            continue
        xx = x[d:].copy()
        yy = y[:n - d].copy()
        valid = ~(np.isnan(xx) | np.isnan(yy))
        xx, yy = xx[valid], yy[valid]
        if len(xx) < 20:
            continue
        if _HAS_MINEPY:
            scores[d] = _mic_minepy(xx, yy)
        else:
            scores[d] = _mic_fallback(xx, yy)
    return scores


# ==============================
# 传递熵 TE
# ==============================
def _digitize(data, n_bins):
    pcts = np.linspace(0, 100, n_bins + 1)[1:-1]
    edges = np.percentile(data[~np.isnan(data)], pcts)
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.linspace(np.nanmin(data), np.nanmax(data), n_bins + 1)[1:-1]
    return np.digitize(data, edges)

def _entropy_1d(binned, n_states):
    counts = np.bincount(binned[~np.isnan(binned)], minlength=n_states + 1)[1:]
    counts = counts.astype(np.float64)
    total = np.sum(counts)
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def _cond_entropy(b_y_given, b_x_given, n_y_states, n_x_states):
    valid = ~(np.isnan(b_y_given) | np.isnan(b_x_given))
    b_y = b_y_given[valid].astype(np.int32)
    b_x = b_x_given[valid].astype(np.int32)
    flat_idx = b_x * (n_y_states + 1) + b_y
    counts = np.bincount(flat_idx, minlength=(n_x_states + 1) * (n_y_states + 1))
    ce = 0.0
    for xi in range(n_x_states):
        row_counts = counts[xi * (n_y_states + 1):(xi + 1) * (n_y_states + 1)][1:]
        row_total = np.sum(row_counts)
        if row_total == 0:
            continue
        row_probs = row_counts.astype(np.float64) / row_total
        row_probs = row_probs[row_probs > 0]
        ce += (row_total / len(b_y)) * (-np.sum(row_probs * np.log2(row_probs)))
    return ce

def calc_te(x, y, max_lag, n_bins=N_BINS, n_shuffle=N_SHUFFLE):
    scores = np.zeros(max_lag + 1)
    pvals = np.ones(max_lag + 1)
    n = len(y)
    for d in range(max_lag + 1):
        if d + 2 >= n:
            continue
        len_valid = n - d - 1
        y_future = y[1 + d:n].copy()
        y_past = y[0:n - 1 - d].copy()
        x_past = x[0:len_valid].copy()
        valid = ~(np.isnan(y_future) | np.isnan(y_past) | np.isnan(x_past))
        if np.sum(valid) < 10:
            continue
        b_y_f = _digitize(y_future[valid], n_bins)
        b_y_p = _digitize(y_past[valid], n_bins)
        b_x = _digitize(x_past[valid], n_bins)
        n_y_states = n_bins
        base_ce = _cond_entropy(b_y_f, b_y_p, n_y_states, n_y_states)
        if base_ce <= 0:
            continue
        te_ce = _cond_entropy(b_y_f, b_x, n_y_states, n_bins)
        te_raw = base_ce - te_ce
        scores[d] = max(0, te_raw)
        te_shuffled = np.zeros(n_shuffle)
        idx_valid = np.where(valid)[0]
        for s in range(n_shuffle):
            x_shuf = x_past.copy()
            np.random.shuffle(x_shuf)
            b_xs = _digitize(x_shuf[idx_valid], n_bins)
            te_s_ce = _cond_entropy(b_y_f, b_xs, n_y_states, n_bins)
            te_shuffled[s] = max(0, base_ce - te_s_ce)
        pvals[d] = np.mean(te_shuffled >= te_raw) if n_shuffle > 0 else 1.0
    return scores, pvals


# ==============================
# 季节分段 TE
# ==============================
def calc_te_seasonal(x, y, months, max_lag, n_bins=N_BINS, n_shuffle=N_SHUFFLE):
    results = {}
    for season_name, season_months in SEASON_MAP.items():
        mask = np.isin(months, season_months)
        if np.sum(mask) < 50:
            results[season_name] = None
            continue
        x_s = x[mask].copy()
        y_s = y[mask].copy()
        te_scores, te_pvals = calc_te(x_s, y_s, max_lag, n_bins, min(n_shuffle, 200))
        results[season_name] = {"te": te_scores.tolist(), "pval": te_pvals.tolist()}
    return results


# ==============================
# ALUM 扫参 (代理线性模型)
# ==============================
def sweep_alum(x_alum, y_filt, x_others_aligned, max_lag):
    best_d, best_rmse, results = 0, float("inf"), {}
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    for d in range(max_lag + 1):
        x_alum_lag = np.roll(x_alum, d)
        x_alum_lag[:d] = np.nan
        X_input = np.column_stack([x_alum_lag] + list(x_others_aligned.values()))
        valid = ~np.any(np.isnan(X_input), axis=1)
        X_v, y_v = X_input[valid], y_filt[valid].reshape(-1, 1)
        if len(X_v) < 50:
            results[d] = float("inf")
            continue
        rmses = []
        for tr_idx, vl_idx in tscv.split(X_v):
            if len(tr_idx) < 10 or len(vl_idx) < 5:
                continue
            lr = LinearRegression()
            lr.fit(X_v[tr_idx], y_v[tr_idx])
            pred = lr.predict(X_v[vl_idx])
            rmse = np.sqrt(np.mean((pred.ravel() - y_v[vl_idx].ravel()) ** 2))
            rmses.append(rmse)
        avg_rmse = np.mean(rmses) if rmses else float("inf")
        results[d] = avg_rmse
        if avg_rmse < best_rmse:
            best_rmse, best_d = avg_rmse, d
    return best_d, best_rmse, results


# ==============================
# 融合决策
# ==============================
def fuse_delay(ccf, mic, te_scores, te_pvals, max_lag):
    ccf_norm = (ccf - np.min(ccf)) / (np.max(ccf) - np.min(ccf) + EPS)
    mic_norm = (mic - np.min(mic)) / (np.max(mic) - np.min(mic) + EPS)
    te_sig = np.array([te_pvals[d] < 0.05 for d in range(max_lag + 1)], dtype=float)
    any_sig = np.any(te_sig > 0)
    if any_sig:
        mic_weighted = mic_norm * te_sig
        d_best = int(np.argmax(mic_weighted))
        method = "MIC×I[TE_pval<0.05]"
    else:
        best_fallback = int(np.argmax(ccf))
        mic_weighted = ccf_norm
        d_best = best_fallback
        method = "CCF(CCF回退·TE无显著lag)"
    return {
        "ccf": ccf.tolist(),
        "mic": mic.tolist(),
        "te": te_scores.tolist(),
        "te_pval": te_pvals.tolist(),
        "mic_weighted": mic_weighted.tolist(),
        "best_d": d_best,
        "best_hours": d_best * 2,
        "method": method,
        "any_te_significant": bool(any_sig),
    }


# ==============================
# 主流程
# ==============================
def main():
    log_path = os.path.join(OUTPUT_DIR, "step2.0_progress.txt")
    def log(msg, end="\n"):
        print(msg, end=end, flush=True)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(msg + end)

    log("[step2.0] 加载数据...")
    x_all, y, months = load_data()
    n = len(y)
    print(f"  样本数: {n}, 月份范围: {months.min()}-{months.max()}")

    results = {}
    tau_params = {}
    x_others_aligned = {}

    for vi, var in enumerate(INPUT_VARS):
        print(f"\n[step2.0] 分析 {var} → FILT.NTU ...", flush=True)
        x = x_all[var]

        x_log = np.log1p(x)
        y_log = np.log1p(y)

        print(f"  计算 CCF...", end=" ", flush=True)
        ccf = calc_ccf(x_log, y_log, MAX_LAG)
        print(f"完成", flush=True)
        print(f"  计算 MIC...", end=" ", flush=True)
        mic = calc_mic(x_log, y_log, MAX_LAG)
        print(f"完成", flush=True)
        print(f"  计算 TE ({N_SHUFFLE}次洗牌)...", end=" ", flush=True)
        te_scores, te_pvals = calc_te(x_log, y_log, MAX_LAG)
        print(f"完成", flush=True)

        fused = fuse_delay(ccf, mic, te_scores, te_pvals, MAX_LAG)
        d_best = fused["best_d"]
        tau_params[var] = d_best

        te_seasonal = calc_te_seasonal(x_log, y_log, months, MAX_LAG)

        results[var] = {
            "ccf": ccf.tolist(),
            "mic": mic.tolist(),
            "te": te_scores.tolist(),
            "te_pval": te_pvals.tolist(),
            "fused": fused,
            "te_seasonal": te_seasonal,
        }

        x_aligned = np.roll(x_log, d_best)
        x_aligned[:d_best] = np.nan
        x_others_aligned[var] = x_aligned

        print(f"  CCF 最优 lag={np.argmax(ccf)}, r={np.max(ccf):.4f}")
        print(f"  MIC 最优 lag={np.argmax(mic)}, val={np.max(mic):.4f}")
        print(f"  TE  最优 lag={np.argmax(te_scores)}, val={np.max(te_scores):.4f}")
        sig_lags = [d for d in range(MAX_LAG + 1) if te_pvals[d] < 0.05]
        print(f"  TE 显著 lag: {sig_lags}")
        print(f"  融合 → d* = {d_best} ({d_best * 2}h)")

    print("\n[step2.0] ALUM 扫参...")
    x_alum_log = np.log1p(x_all["ALUM"])
    d_alum, rmse_alum, sweep_results = sweep_alum(
        x_alum_log, y_log, x_others_aligned, MAX_LAG)
    tau_params["ALUM"] = d_alum
    results["ALUM_sweep"] = {str(k): v for k, v in sweep_results.items()}
    print(f"  ALUM 最优 d* = {d_alum} ({d_alum * 2}h), RMSE = {rmse_alum:.4f}")

    # ==============================
    # 精度对比表
    # ==============================
    rows = []
    for var in INPUT_VARS:
        r = results[var]
        # TE 不分段
        d_te_nonseg = int(np.argmax(r["te"]))

        # TE 分段 → 取各段最优 lag 的众数
        seasonal_ds = []
        for sn, sres in r["te_seasonal"].items():
            if sres is not None:
                seasonal_ds.append(int(np.argmax(sres["te"])))
        d_te_seg = int(np.round(np.median(seasonal_ds))) if seasonal_ds else d_te_nonseg

        # 交叉相关
        d_ccf = int(np.argmax(r["ccf"]))
        # MIC
        d_mic = int(np.argmax(r["mic"]))
        # 融合
        d_fused = r["fused"]["best_d"]

        for method_name, d_val in [("CCF(互相关)", d_ccf),
                                    ("MIC", d_mic),
                                    ("TE(不分段)", d_te_nonseg),
                                    ("TE(分季)", d_te_seg),
                                    ("MIC+TE融合", d_fused)]:
            rows.append({
                "变量": VAR_LABELS.get(var, var),
                "方法": method_name,
                "最优lag(步)": d_val,
                "最优lag(h)": d_val * 2,
            })

    df_cmp = pd.DataFrame(rows)
    df_cmp.to_csv(os.path.join(OUTPUT_DIR, "tau_comparison.csv"), index=False, encoding="utf-8-sig")
    print("\n[step2.0] 精度对比表 → output/tau_comparison.csv")
    print(df_cmp.to_string(index=False))

    # ==============================
    # 保存
    # ==============================
    tau_out = {
        k: {"steps": int(v), "hours": int(v * 2)}
        for k, v in tau_params.items()
    }
    for var in INPUT_VARS:
        tau_out[var]["ccf_best"] = int(np.argmax(results[var]["ccf"]))
        tau_out[var]["mic_best"] = int(np.argmax(results[var]["mic"]))
        tau_out[var]["te_best"] = int(np.argmax(results[var]["te"]))
        sig_lags = [d for d in range(MAX_LAG + 1) if results[var]["te_pval"][d] < 0.05]
        tau_out[var]["te_significant_lags"] = sig_lags
    tau_out["ALUM"]["method"] = "sweep_linear_proxy"
    tau_out["ALUM"]["sweep_rmse"] = {str(k): float(v) for k, v in sweep_results.items()}

    with open(os.path.join(OUTPUT_DIR, "tau_params.json"), "w", encoding="utf-8") as f:
        json.dump(tau_out, f, indent=2, ensure_ascii=False)
    print("\n[step2.0] tau_params.json 已保存")

    # ==============================
    # 分析表格
    # ==============================
    analysis_rows = []
    for var in INPUT_VARS:
        r = results[var]
        for d in range(MAX_LAG + 1):
            analysis_rows.append({
                "变量": var,
                "lag(步)": d,
                "lag(h)": d * 2,
                "CCF": round(r["ccf"][d], 4),
                "MIC": round(r["mic"][d], 4),
                "TE": round(r["te"][d], 4),
                "TE_pval": round(r["te_pval"][d], 4),
                "TE_显著": "是" if r["te_pval"][d] < 0.05 else "否",
                "融合得分": round(r["fused"]["mic_weighted"][d], 4),
            })
    df_analysis = pd.DataFrame(analysis_rows)
    df_analysis.to_csv(os.path.join(OUTPUT_DIR, "tau_analysis.csv"), index=False, encoding="utf-8-sig")
    print("[step2.0] tau_analysis.csv 已保存")

    # ==============================
    # 图1: 三变量三方法九宫格
    # ==============================
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for row_i, var in enumerate(INPUT_VARS):
        r = results[var]
        lags = np.arange(MAX_LAG + 1)
        lags_h = lags * 2

        ax_ccf = axes[row_i, 0]
        ax_ccf.bar(lags_h, r["ccf"], color="steelblue", alpha=0.8)
        ax_ccf.axvline(x=np.argmax(r["ccf"]) * 2, color="red", linestyle="--", label=f"best={np.argmax(r['ccf'])*2}h")
        ax_ccf.set_title(f"{var} — CCF")
        ax_ccf.set_xlabel("lag (h)")
        ax_ccf.set_ylabel("|r|")
        ax_ccf.legend(fontsize=7)

        ax_mic = axes[row_i, 1]
        ax_mic.bar(lags_h, r["mic"], color="seagreen", alpha=0.8)
        ax_mic.axvline(x=np.argmax(r["mic"]) * 2, color="red", linestyle="--", label=f"best={np.argmax(r['mic'])*2}h")
        ax_mic.set_title(f"{var} — MIC")
        ax_mic.set_xlabel("lag (h)")
        ax_mic.set_ylabel("MIC")
        ax_mic.legend(fontsize=7)

        ax_te = axes[row_i, 2]
        colors = ["forestgreen" if r["te_pval"][d] < 0.05 else "lightgray" for d in range(MAX_LAG + 1)]
        ax_te.bar(lags_h, r["te"], color=colors, alpha=0.8)
        ax_te.axvline(x=r["fused"]["best_d"] * 2, color="red", linestyle="--", label=f"融合={r['fused']['best_d']*2}h")
        ax_te.set_title(f"{var} — TE (绿=显著)")
        ax_te.set_xlabel("lag (h)")
        ax_te.set_ylabel("TE (bits)")
        ax_te.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "tau_combined.png"), dpi=150)
    plt.close()
    print("[step2.0] figures/tau_combined.png 已保存")

    # ==============================
    # 图2: TE 季节分段对比
    # ==============================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    season_colors = {"high": "darkred", "low": "steelblue", "trans": "darkorange"}
    for col_i, var in enumerate(INPUT_VARS):
        ax = axes[col_i]
        r = results[var]
        lags_h = np.arange(MAX_LAG + 1) * 2
        # 不分段
        ax.plot(lags_h, r["te"], "k-o", linewidth=2, markersize=6, label="全年TE")
        # 分段
        for sn in ["high", "low", "trans"]:
            sres = r["te_seasonal"].get(sn)
            if sres is not None:
                ax.plot(lags_h, sres["te"], "s--", color=season_colors[sn],
                        linewidth=1.5, markersize=5, label=f"{sn}季")
        ax.set_title(var)
        ax.set_xlabel("lag (h)")
        ax.set_ylabel("TE (bits)")
        ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "tau_seasonal.png"), dpi=150)
    plt.close()
    print("[step2.0] figures/tau_seasonal.png 已保存")

    # ==============================
    # 图3: ALUM 扫参
    # ==============================
    fig, ax = plt.subplots(figsize=(8, 4))
    ds = list(sweep_results.keys())
    rmses = [sweep_results[d] for d in ds]
    ax.bar([d * 2 for d in ds], rmses, color="mediumpurple", alpha=0.8)
    ax.axvline(x=d_alum * 2, color="red", linestyle="--", label=f"best d*={d_alum} ({d_alum*2}h)")
    ax.set_xlabel("lag (h)")
    ax.set_ylabel("代理模型 RMSE")
    ax.set_title("ALUM 时滞扫参 (线性代理)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "tau_alum_sweep.png"), dpi=150)
    plt.close()
    print("[step2.0] figures/tau_alum_sweep.png 已保存")

    print("\n[step2.0] 完成.")


if __name__ == "__main__":
    main()
