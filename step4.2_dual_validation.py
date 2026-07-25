"""
step4.2_dual_validation.py — FCE + Bootstrap + 事件回溯验证
================================================================
1. FCE模糊综合评价 (分区隶属度) vs Jenks → Kappa (分层报告)
2. Bootstrap 1000次 → 断点95% CI (分区独立)
3. 事件回溯验证: 超标捕获率/虚警率/提前预警步数/等级单调性
"""

import os, json, warnings
import numpy as np
import pandas as pd
from step0_config import *

warnings.filterwarnings("ignore")
EPS = 1e-10


def jenks_1d(values, n_classes):
    """Jenks自然断点: 一维DP(预计算SSD加速), 返回 (n_classes-1) 个断点"""
    v = np.sort(values)
    n = len(v)
    if n < n_classes or n_classes < 2:
        return []
    cum_sum = np.zeros(n + 1)
    cum_sq = np.zeros(n + 1)
    for i in range(n):
        cum_sum[i + 1] = cum_sum[i] + v[i]
        cum_sq[i + 1] = cum_sq[i] + v[i] * v[i]

    def ssd(i, j):
        if j <= i:
            return 0.0
        s = cum_sum[j] - cum_sum[i]
        sq = cum_sq[j] - cum_sq[i]
        m = s / (j - i)
        return sq - 2 * m * s + (j - i) * m * m

    mat_d = np.full((n + 1, n_classes + 1), float("inf"))
    mat_b = np.zeros((n + 1, n_classes + 1), dtype=int)

    for i in range(1, n + 1):
        mat_d[i][1] = ssd(0, i)
        mat_b[i][1] = 1

    for k in range(2, n_classes + 1):
        for i in range(1, n + 1):
            best = float("inf")
            best_j = 1
            for j in range(k - 1, i):
                val = mat_d[j][k - 1] + ssd(j, i)
                if val < best:
                    best = val
                    best_j = j
            mat_d[i][k] = best
            mat_b[i][k] = best_j

    breaks = []
    bk = n
    for k in range(n_classes, 1, -1):
        bk = mat_b[bk][k]
        if bk > 0 and bk < n:
            breaks.append(v[bk])
    breaks.sort()
    return breaks


def triangle_membership(value, left, peak, right):
    """三角隶属度: left→0, peak→1, right→0"""
    if value <= left or value >= right:
        return 0.0
    if value <= peak:
        return (value - left) / max(peak - left, EPS)
    else:
        return (right - value) / max(right - peak, EPS)


def fce_grade(s_risk, is_stress, breaks_comfort, breaks_stress):
    """FCE分级: 三角隶属度, 断点作为分级锚点"""
    if is_stress:
        # 应力区: 使用应力区断点 + 0边界/1.0上界
        p = [0.0] + list(breaks_stress) + [1.0]
    else:
        p = [0.0] + list(breaks_comfort) + [1.0]
    # 4个三角形的(peak, left, right)
    peaks = [(p[0] + p[1]) / 2, (p[1] + p[2]) / 2,
             (p[2] + p[3]) / 2, (p[3] + 1.0) / 2]
    lefts = [p[0], p[0], p[1], p[2]]
    rights = [p[1], p[2], p[3], p[3]]
    mus = [triangle_membership(s_risk, lefts[i], peaks[i], rights[i])
           for i in range(4)]
    best = np.argmax(mus)
    return best + 1, mus


def compute_kappa(y_true, y_pred, n_classes=4):
    """Cohen's Kappa"""
    cm = np.zeros((n_classes, n_classes), dtype=float)
    for t, p in zip(y_true, y_pred):
        cm[int(t) - 1, int(p) - 1] += 1.0
    po = np.trace(cm) / max(cm.sum(), EPS)
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    pe = (row_sum @ col_sum) / max(cm.sum() ** 2, EPS)
    kappa = (po - pe) / max(1 - pe, EPS)
    return kappa, cm


def bootstrap_jenks(values, n_iter=30, n_classes=3, subsample=400):
    """Bootstrap重采样(子采样加速)估计断点CI"""
    all_breaks = []
    n = len(values)
    sample_n = min(n, subsample)
    for _ in range(n_iter):
        idx = np.random.randint(0, n, sample_n)
        sample = values[idx]
        brk = jenks_1d(sample, n_classes)
        if len(brk) == n_classes - 1:
            all_breaks.append(brk)
    if not all_breaks:
        return {}
    arr = np.array(all_breaks)
    result = {}
    for i in range(arr.shape[1]):
        col = arr[:, i]
        result[f"break_{i+1}"] = {
            "mean": round(float(col.mean()), 6),
            "std": round(float(col.std()), 6),
            "ci_low": round(float(np.percentile(col, 2.5)), 6),
            "ci_high": round(float(np.percentile(col, 97.5)), 6),
        }
    return result


def event_backtest(df):
    """事件回溯验证"""
    n = len(df)
    ntu = df["NTU"].values
    filt = df["FILT_NTU"].values
    rw_ntu = df["RW_NTU"].values if "RW_NTU" in df.columns else None
    grades = df["GRADE"].values
    s_risk = df["S_risk"].values
    eta = df["ETA_COAG"].values if "ETA_COAG" in df.columns else None

    # 事件标签
    event = np.zeros(n, dtype=int)
    for i in range(n):
        if ntu[i] > Q4_NTU_LIMIT:
            event[i] = 4  # A级: 超标
        elif filt[i] >= Q4_THETA:
            event[i] = 3  # B级: 滤后恶化
        elif rw_ntu is not None and rw_ntu[i] > np.percentile(rw_ntu, 95) and eta is not None and i > 0 and not np.isnan(eta[i]) and not np.isnan(eta[i - 1]) and eta[i] < eta[i - 1] * 0.95:
            event[i] = 2  # C级: 原水冲击
        else:
            event[i] = 1  # 无事件

    # 混淆矩阵: event vs grade
    cm = np.zeros((4, 4), dtype=float)
    for e, g in zip(event, grades):
        cm[e - 1, g - 1] += 1.0

    results = {}
    # 超标捕获率
    exceed_mask = event == 4
    if exceed_mask.sum() > 0:
        captured = ((grades >= 3) & exceed_mask).sum()
        results["exceed_capture_rate"] = round(float(captured / max(exceed_mask.sum(), EPS)), 4)
    else:
        results["exceed_capture_rate"] = -1.0

    # 正常时虚警率
    normal_mask = event == 1
    if normal_mask.sum() > 0:
        false_alarm = (normal_mask & (grades >= 3)).sum()
        results["false_alarm_rate"] = round(float(false_alarm / max(normal_mask.sum(), EPS)), 4)
    else:
        results["false_alarm_rate"] = -1.0

    # 等级单调性
    grade_means_by_event = [float(grades[event == e].mean()) for e in [1, 2, 3, 4]]
    monotonic = all(grade_means_by_event[i] <= grade_means_by_event[i + 1] for i in range(3))
    results["grade_monotonic"] = monotonic
    results["grade_mean_by_event"] = {f"E{e+1}": round(grade_means_by_event[e], 4) for e in range(4)}

    # 提前预警步数: 首次grade≥3在NTU超标前
    lead_steps = []
    i = 0
    while i < n:
        if event[i] == 4:
            start = i
            for lookback in range(min(12, i), 0, -1):
                if grades[i - lookback] >= 3:
                    lead_steps.append(lookback)
                    break
            i += 1
            while i < n and event[i] == 4:
                i += 1
        else:
            i += 1
    results["avg_lead_steps"] = round(float(np.mean(lead_steps)), 2) if lead_steps else 0.0
    results["lead_steps_list"] = lead_steps[:10]

    results["event_distribution"] = {f"E{e}": int((event == e).sum()) for e in [1, 2, 3, 4]}
    results["confusion_matrix"] = cm.tolist()
    return results


def main():
    print("=" * 60)
    print("  step4.2 — 双重验证 + 事件回溯")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[1/5] 加载数据...")
    df = pd.read_csv(OUT_Q4_RISK_SCORES)
    if "GRADE" not in df.columns:
        print("  [ERR] GRADE not found, run step4.1 first")
        return
    s_risk = df["S_risk"].values
    zone = df["ZONE"].values
    grades = df["GRADE"].values
    n = len(df)
    print(f"  样本: {n}")

    mask_comfort = zone == "comfort"
    mask_stress = ~mask_comfort

    with open(OUT_Q4_BREAKS_COMF, encoding="utf-8") as f:
        bc = json.load(f)["breaks"]
    with open(OUT_Q4_BREAKS_STR, encoding="utf-8") as f:
        bs = json.load(f)["breaks"]

    print("\n[2/5] FCE模糊综合评价...")
    fce_grades = np.zeros(n, dtype=int)
    for i in range(n):
        fce_grades[i], _ = fce_grade(s_risk[i], zone[i] == "stress", bc, bs)

    fce_agreement = (fce_grades == grades).mean()
    kappa_all, cm_all = compute_kappa(grades, fce_grades)
    print(f"  FCE-Jenks一致率: {fce_agreement:.4f}")
    print(f"  Kappa(全量):     {kappa_all:.4f}")

    kappa_comfort, cm_comfort = compute_kappa(grades[mask_comfort], fce_grades[mask_comfort])
    kappa_stress, cm_stress = compute_kappa(grades[mask_stress], fce_grades[mask_stress])
    print(f"  Kappa(舒适区):   {kappa_comfort:.4f}  (n={mask_comfort.sum()})")
    print(f"  Kappa(应力区):   {kappa_stress:.4f}  (n={mask_stress.sum()})")

    print("\n[3/5] Bootstrap 稳定性...")
    boot_comfort = bootstrap_jenks(s_risk[mask_comfort], Q4_BOOTSTRAP_N, 3)
    boot_stress = bootstrap_jenks(s_risk[mask_stress], Q4_BOOTSTRAP_N, 3)
    if boot_comfort:
        for k, v in boot_comfort.items():
            print(f"  舒适区 {k}: mean={v['mean']:.4f} CI=[{v['ci_low']:.4f}, {v['ci_high']:.4f}]")
    if boot_stress:
        for k, v in boot_stress.items():
            print(f"  应力区 {k}: mean={v['mean']:.4f} CI=[{v['ci_low']:.4f}, {v['ci_high']:.4f}]")

    print("\n[4/5] 事件回溯验证...")
    backtest = event_backtest(df)
    print(f"  超标捕获率:      {backtest['exceed_capture_rate']:.2%}")
    print(f"  正常虚警率:      {backtest['false_alarm_rate']:.2%}")
    print(f"  等级单调性:      {'PASS' if backtest['grade_monotonic'] else 'FAIL'}")
    print(f"  平均提前预警步:  {backtest['avg_lead_steps']:.1f}")

    print("\n[5/5] 保存输出...")
    kappa_report = {
        "kappa_all": round(kappa_all, 4),
        "kappa_comfort": round(kappa_comfort, 4) if not np.isnan(kappa_comfort) else -1,
        "kappa_stress": round(kappa_stress, 4) if not np.isnan(kappa_stress) else -1,
        "fce_agreement": round(fce_agreement, 4),
        "confusion_matrix_all": cm_all.tolist(),
        "n_comfort": int(mask_comfort.sum()),
        "n_stress": int(mask_stress.sum()),
    }
    with open(OUT_Q4_KAPPA, "w", encoding="utf-8") as f:
        json.dump(kappa_report, f, indent=2, ensure_ascii=False)

    boot_result = {"comfort": boot_comfort, "stress": boot_stress}
    with open(OUT_Q4_BOOTSTRAP.replace(".csv", ".json"), "w", encoding="utf-8") as f:
        json.dump(boot_result, f, indent=2, ensure_ascii=False)

    with open(OUT_Q4_BACKTEST, "w", encoding="utf-8") as f:
        json.dump(backtest, f, indent=2, ensure_ascii=False, default=str)

    print(f"  [DONE] {OUT_Q4_KAPPA}")
    print(f"  [DONE] bootstrap report")
    print(f"  [DONE] {OUT_Q4_BACKTEST}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
