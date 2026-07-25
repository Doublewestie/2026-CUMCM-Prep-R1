"""
step4.4_predict_2026.py — 2026年3月逐日风险分类
=================================================
使用Q4模型(从2025训练)对2026年3月12条数据做四级风险分类。
"""

import os, json, warnings
import numpy as np
import pandas as pd
from step0_config import *

warnings.filterwarnings("ignore")
EPS = 1e-10


def find_2026_march():
    """定位2026年3月Excel文件"""
    for f in os.listdir(DATA_DIR_2026):
        fp = os.path.join(DATA_DIR_2026, f)
        if os.path.isfile(fp) and ("3" in f or "Mar" in f or "03" in f):
            return fp
    return None


def load_2026_march():
    """加载并清洗2026年3月数据"""
    fp = find_2026_march()
    if fp is None:
        print("  [ERR] Cannot find 2026 March data")
        return None
    print(f"  Found: {os.path.basename(fp)}")
    df = pd.read_excel(fp)

    rename = {"TIME ": "TIME", "RIVER LEVEL": "RIVER_LEVEL",
              "R/W PUMP DUTY": "RW_PUMP_DUTY", "R/W FLOW": "RW_FLOW",
              "R/W NTU": "RW_NTU", "R/W CLR": "RW_CLR",
              "R/W PH": "RW_PH", "FILT. NTU": "FILT_NTU",
              "C/W WELL LEVEL": "CW_WELL_LEVEL", "T/W FLOW": "TW_FLOW",
              "T/W PUMP DUTY": "TW_PUMP_DUTY"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Convert numeric
    for c in ["RW_FLOW", "RW_NTU", "RW_CLR", "FILT_NTU", "CW_WELL_LEVEL",
              "TW_FLOW", "ALUM", "NTU"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Parse pump counts
    if "TW_PUMP_DUTY" in df.columns:
        df["tw_pump_count"] = df["TW_PUMP_DUTY"].apply(
            lambda x: len(set(str(x).replace("+", ",").split(","))) if pd.notna(x) else 0)
        df = df.drop(columns=["TW_PUMP_DUTY"])

    df = df.dropna(subset=["NTU", "FILT_NTU"]).reset_index(drop=True)
    df["MONTH"] = 3
    print(f"  {len(df)} rows after cleaning")
    return df


def promote_risk_for_stress(zone, ntu):
    """应力区f₁自动上调(溶解Q2控制余量发现)"""
    pass  # 全部舒适区, 不需提升


def compute_f1_amplitude_2026(ntu, filt, theta=Q4_THETA):
    """f₁: 分区归一化"""
    exceed = np.maximum(0, ntu - Q4_NTU_LIMIT)
    p99_stress_path = os.path.join(OUTPUT_DIR, "..", "output")
    risk_csv = os.path.join(OUTPUT_DIR, "q4_risk_scores.csv")
    if os.path.exists(risk_csv):
        train = pd.read_csv(risk_csv)["f1"]
        p99_stress = np.percentile(train, 99)
    else:
        p99_stress = 1.0
    f1 = np.zeros_like(ntu)
    for i in range(len(ntu)):
        if filt[i] < theta:
            f1[i] = min(1.0, ntu[i] / Q4_NTU_LIMIT)
        else:
            f1[i] = min(1.0, exceed[i] / max(p99_stress, EPS))
    return f1


def compute_f2_duration_2026(ntu, n_samples):
    """f₂: 持续时长(需连续超标才累积, 2026数据稀疏所以每个样本独立)"""
    exceed = (ntu > Q4_NTU_LIMIT).astype(float)
    f2 = np.zeros(n_samples)
    for i in range(n_samples):
        if exceed[i] > 0.5:
            gamma = np.log(2) / Q4_T_HALF_COMFORT
            f2[i] = 1 - np.exp(-gamma * 1)
        else:
            f2[i] = 0.0
    return f2


def compute_f3_trend_2026(ntu, filt):
    """f₃: 恶化趋势(无历史上下文时简化为NTU相对FILT的增量)"""
    n = len(ntu)
    f3 = np.zeros(n)
    for i in range(1, n):
        diff = ntu[i] - filt[i - 1]
        f3[i] = max(0, diff)
    p99_path = os.path.join(OUTPUT_DIR, "q4_risk_scores.csv")
    if os.path.exists(p99_path):
        train_f3 = pd.read_csv(p99_path)["f3"].values
        p99 = np.percentile(train_f3, 99) or 1.0
    else:
        p99 = 1.0
    return np.minimum(1.0, f3 / max(p99, EPS))


def main():
    print("=" * 60)
    print("  step4.4 — 2026年3月风险分类")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[1/5] 加载2026年3月数据...")
    df = load_2026_march()
    if df is None or len(df) == 0:
        print("  [ERR] No data loaded")
        return

    ntu = df["NTU"].values.astype(float)
    filt = df["FILT_NTU"].values.astype(float)
    n = len(df)
    print(f"  NTU: {[f'{v:.3f}' for v in ntu]}")
    print(f"  FILT: {[f'{v:.3f}' for v in filt]}")

    print("\n[2/5] 加载Q4模型参数...")
    with open(OUT_Q4_WEIGHTS, encoding="utf-8") as f:
        weights = json.load(f)
    w1, w2, w3 = (weights.get(k, 0.3) for k in
                  ["f1_amplitude", "f2_duration", "f3_trend"])
    print(f"  weights: w1={w1:.4f}, w2={w2:.4f}, w3={w3:.4f}")

    with open(OUT_Q4_BREAKS_COMF, encoding="utf-8") as f:
        breaks_comfort = json.load(f)["breaks"]
    print(f"  comfort breaks: {[round(b, 4) for b in breaks_comfort]}")

    print("\n[3/5] 计算风险评分...")
    zone = np.where(filt < Q4_THETA, "comfort", "stress")
    print(f"  分区: {list(zone)}")

    f1 = compute_f1_amplitude_2026(ntu, filt)
    f2 = compute_f2_duration_2026(ntu, n)
    f3 = compute_f3_trend_2026(ntu, filt)
    s_risk = w1 * f1 + w2 * f2 + w3 * f3

    print(f"  f1: {[f'{v:.4f}' for v in f1]}")
    print(f"  f2: {[f'{v:.4f}' for v in f2]}")
    print(f"  f3: {[f'{v:.4f}' for v in f3]}")
    print(f"  S_risk: {[f'{v:.4f}' for v in s_risk]}")

    print("\n[4/5] 分级...")
    def map_grades(value, breaks_comfort, breaks_stress, is_stress):
        if is_stress:
            if len(breaks_stress) < 2:
                return 2 if value < 0.33 else (3 if value < 0.66 else 4)
            if value <= breaks_stress[0]:
                return 2
            elif value <= breaks_stress[1]:
                return 3
            else:
                return 4
        else:
            if len(breaks_comfort) < 2:
                return 1 if value < 0.15 else (2 if value < 0.30 else 3)
            if value <= breaks_comfort[0]:
                return 1
            elif value <= breaks_comfort[1]:
                return 2
            else:
                return 3
    with open(OUT_Q4_BREAKS_STR, encoding="utf-8") as f:
        breaks_stress = json.load(f)["breaks"]
    grades = np.array([
        map_grades(s_risk[i], breaks_comfort, breaks_stress, zone[i] == "stress")
        for i in range(n)
    ])
    print(f"  Grades: {list(grades)}")

    print("\n[5/5] 输出结果...")
    result = df[["TIME", "NTU", "FILT_NTU"]].copy()
    result["ZONE"] = zone
    result["f1"] = [round(v, 4) for v in f1]
    result["f2"] = [round(v, 4) for v in f2]
    result["f3"] = [round(v, 4) for v in f3]
    result["S_risk"] = [round(v, 4) for v in s_risk]
    result["GRADE"] = grades

    # Update existing excel
    excel_path = OUT_Q4_EXCEL
    if os.path.exists(excel_path):
        with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a",
                            if_sheet_exists="replace") as writer:
            result.to_excel(writer, sheet_name="2026_march", index=False)
    else:
        result.to_excel(excel_path, sheet_name="2026_march", index=False,
                        engine="openpyxl")

    print(f"\n  === 2026年3月风险分类结果 ===")
    print(f"  {'TIME':>6s} {'NTU':>6s} {'FILT':>6s} {'Zone':>9s} {'S_risk':>8s} {'Grade':>6s}")
    print(f"  {'-'*42}")
    for i in range(n):
        t = int(result.iloc[i]["TIME"])
        print(f"  {t:6d} {ntu[i]:6.3f} {filt[i]:6.3f} {zone[i]:>9s} "
              f"{s_risk[i]:8.4f} {grades[i]:6d}")

    grade_counts = pd.Series(grades).value_counts().sort_index()
    print(f"\n  等级分布:")
    for g in [1, 2, 3, 4]:
        print(f"    Grade {g}: {grade_counts.get(g, 0)} 条")

    print(f"\n  [DONE] 结果已写入 {excel_path} [2026_march]")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
