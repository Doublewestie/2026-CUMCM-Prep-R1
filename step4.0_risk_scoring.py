"""
step4.0_risk_scoring.py — 三维风险评分 + 熵权法
=================================================
融合 Q1/Q2 发现的三维评分:
  f₁ 幅度 — 分区归一化(Q2 θ=0.15) + 月度P99滚动(Q1)
  f₂ 时长 — 分区T_half + CSTR β₂惯性折扣(Q1)
  f₃ 趋势 — 滞后对齐差分(Q2 τ) + η_coag加权(Q1)

输出: q4_risk_scores.csv, q4_omega_weights.json
"""

import os, json, warnings
import numpy as np
import pandas as pd
from step0_config import *

warnings.filterwarnings("ignore")
EPS = 1e-10


def load_q2_params():
    fp = os.path.join(OUTPUT_DIR, "step2_final_results.json")
    if not os.path.exists(fp):
        print("  [WARN] step2_final_results.json not found, using defaults")
        return {"tau_params": {"RW_NTU_to_FILT_hours": 4}}
    with open(fp, encoding="utf-8") as f:
        return json.load(f)


def load_cstr_params():
    fp = os.path.join(OUTPUT_DIR, "cstr_final_best.json")
    if not os.path.exists(fp):
        print("  [WARN] cstr_final_best.json not found, using defaults")
        return {"A_T1": 400, "A_T2": 250, "A_T3": 30,
                "A_T3_rule": {"A_same": 100, "A_diff": 20},
                "RL_med": 6.09, "Q_med": 44.0}
    with open(fp, encoding="utf-8") as f:
        return json.load(f)


def compute_per_sample_beta2(filt_ntu, cw_well, tw_flow, river_level, cstr):
    """Compute CSTR beta2 per sample using actual per-tier A + Balance Detector."""
    n = len(filt_ntu)
    beta2 = np.full(n, 0.95)
    T1_THR, T2_THR = 0.05, 0.15
    A_T1 = cstr.get("A_T1", 400)
    A_T2 = cstr.get("A_T2", 250)
    A_T3 = cstr.get("A_T3", 30)
    rule = cstr.get("A_T3_rule", {})
    A_same = rule.get("A_same", 100)
    A_diff = rule.get("A_diff", 20)
    RL_med = cstr.get("RL_med", 6.09)
    Q_med = cstr.get("Q_med", 44.0)

    for t in range(1, n):
        H = max(cw_well[t - 1], 0.1)
        Qv = max(tw_flow[t - 1], 1.0)
        ft = filt_ntu[t]
        if ft <= T1_THR:
            A0 = A_T1
        elif ft <= T2_THR:
            A0 = A_T2
        else:
            rv = river_level[t]
            if not np.isnan(rv):
                A0 = A_same if (rv - RL_med) * (tw_flow[t] - Q_med) > 0 else A_diff
            else:
                A0 = A_T3
        theta = A0 * H / Qv
        theta = max(theta, 0.02)
        beta2[t] = np.clip(np.exp(-2.0 / theta), 0.001, 0.999)
    return beta2


def compute_f1_amplitude(ntu, filt_ntu, month, theta=Q4_THETA):
    """f₁: 分区归一化超标幅度 + 月度P99滚动"""
    p99_monthly = {}
    for m in range(1, 13):
        mask = month == m
        if mask.sum() > 10:
            p99_monthly[m] = np.percentile(ntu[mask], 99)
        else:
            p99_monthly[m] = np.percentile(ntu, 99)
    raw = np.maximum(0, ntu - Q4_NTU_LIMIT)
    p99_global = np.percentile(raw, 99)
    p99_stress = np.percentile(raw[filt_ntu >= theta], 99) if (filt_ntu >= theta).sum() > 10 else p99_global
    f1 = np.zeros_like(ntu)
    for i in range(len(ntu)):
        m = int(month[i]) if not np.isnan(month[i]) else 6
        m = max(1, min(12, m))
        if filt_ntu[i] < theta:
            f1[i] = min(1.0, ntu[i] / Q4_NTU_LIMIT)
        else:
            denom = max(p99_stress, EPS)
            f1[i] = min(1.0, raw[i] / denom)
    return f1


def compute_f2_duration(ntu, filt_ntu, beta2_arr, theta=Q4_THETA):
    """f₂: 分区T_half + CSTR β₂惯性折扣 (per-sample beta2 from actual CSTR model)"""
    f2 = np.zeros_like(ntu)
    exceed = (ntu > Q4_NTU_LIMIT).astype(float)
    n = len(ntu)
    d_run = 0
    for i in range(n):
        if exceed[i] > 0.5:
            d_run += 1
        else:
            d_run = 0
        if filt_ntu[i] < theta:
            gamma = np.log(2) / Q4_T_HALF_COMFORT
            d_eff = d_run
        else:
            gamma = np.log(2) / Q4_T_HALF_STRESS
            d_eff = d_run
        if i > 0 and filt_ntu[i] < theta and ntu[i] > Q4_NTU_LIMIT:
            d_eff = d_run * beta2_arr[i]
        f2[i] = 1 - np.exp(-gamma * d_eff)
    return f2


def compute_f3_trend(ntu, filt_ntu, eta_coag, tau_steps=Q4_F3_TAU_REF):
    """f₃: 滞后对齐差分 + η_coag加权"""
    n = len(ntu)
    f3 = np.zeros(n)
    for i in range(tau_steps, n):
        ref = filt_ntu[i - tau_steps]
        diff = ntu[i] - ref
        f3[i] = max(0, diff)
        if eta_coag is not None and not np.isnan(eta_coag[i]):
            eta_drop = max(0, -(eta_coag[i] - eta_coag[i - 1]))
            f3[i] *= (1 + Q4_ALPHA_ETA * eta_drop)
    p99 = np.percentile(f3, 99) or 1.0
    return np.minimum(1.0, f3 / max(p99, EPS))


def compute_eta_coag(rw_ntu, filt_ntu):
    """混凝去除效率 η_coag = (RW_NTU - FILT_NTU) / RW_NTU"""
    denom = np.maximum(rw_ntu, EPS)
    return (rw_ntu - filt_ntu) / denom


def entropy_weight(X):
    """熵权法: X is (n, d) array, return weights w (d,)"""
    n, d = X.shape
    p = X / np.maximum(X.sum(axis=0, keepdims=True), EPS)
    entropy = -np.sum(p * np.log(np.maximum(p, EPS)), axis=0) / np.log(n)
    info = 1 - entropy
    w = info / max(info.sum(), EPS)
    return w


def main():
    print("=" * 60)
    print("  step4.0 — 三维风险评分 + 熵权法")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUT_Q4_FIG_DIR, exist_ok=True)

    print("\n[1/5] 加载数据...")
    df = pd.read_csv(OUT_CLEAN_DATA)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["NTU", "FILT_NTU"]).reset_index(drop=True)
    n = len(df)
    print(f"  样本数: {n}")

    ntu = df["NTU"].values.astype(float)
    filt = df["FILT_NTU"].values.astype(float)
    rw_ntu = df["RW_NTU"].values.astype(float)
    month = df["DATE"].dt.month.values.astype(float)

    print("\n[2/5] 加载 CSTR/Q2 参数 + 计算 η_coag...")
    cstr = load_cstr_params()
    cw_well = df["CW_WELL_LEVEL"].values.astype(float)
    tw_flow = df["TW_FLOW"].values.astype(float)
    rl = df["RIVER_LEVEL"].values.astype(float)
    beta2 = compute_per_sample_beta2(filt, cw_well, tw_flow, rl, cstr)
    print(f"  beta2 per-tier: T1 median={np.median(beta2[filt<=0.05]):.3f}, "
          f"T2 median={np.median(beta2[(filt>0.05)&(filt<=0.15)]):.3f}, "
          f"T3 median={np.median(beta2[filt>0.15]):.3f}")
    eta = compute_eta_coag(rw_ntu, filt)

    print("\n[3/5] 三维评分计算...")
    f1 = compute_f1_amplitude(ntu, filt, month)
    f2 = compute_f2_duration(ntu, filt, beta2)
    f3 = compute_f3_trend(ntu, filt, eta)

    print(f"  f1幅度: mean={f1.mean():.4f} std={f1.std():.4f}")
    print(f"  f2时长: mean={f2.mean():.4f} std={f2.std():.4f}")
    print(f"  f3趋势: mean={f3.mean():.4f} std={f3.std():.4f}")

    print("\n[4/5] 熵权法赋权...")
    X = np.column_stack([f1, f2, f3])
    w = entropy_weight(X)
    s_risk = X @ w

    print(f"  熵权: w1={w[0]:.4f} w2={w[1]:.4f} w3={w[2]:.4f}")
    print(f"  S_risk: mean={s_risk.mean():.4f} std={s_risk.std():.4f} "
          f"P90={np.percentile(s_risk, 90):.4f}")

    print("\n[5/5] 保存输出...")
    score_df = df[["DATE", "NTU", "FILT_NTU"]].copy()
    score_df["MONTH"] = month
    score_df["ZONE"] = np.where(filt < Q4_THETA, "comfort", "stress")
    score_df["f1"] = f1
    score_df["f2"] = f2
    score_df["f3"] = f3
    score_df["S_risk"] = s_risk
    score_df["HAS_EVENT"] = (ntu > Q4_NTU_LIMIT).astype(int)
    score_df["ETA_COAG"] = eta
    score_df.to_csv(OUT_Q4_RISK_SCORES, index=False, encoding="utf-8-sig")

    weights = {
        "f1_amplitude": round(w[0], 6),
        "f2_duration": round(w[1], 6),
        "f3_trend": round(w[2], 6),
        "method": "entropy_weight",
    }
    with open(OUT_Q4_WEIGHTS, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2, ensure_ascii=False)

    print(f"  [DONE] {OUT_Q4_RISK_SCORES}")
    print(f"  [DONE] {OUT_Q4_WEIGHTS}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
