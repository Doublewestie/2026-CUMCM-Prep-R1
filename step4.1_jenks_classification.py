"""
step4.1_jenks_classification.py — 分区独立 Jenks → 四级映射
==============================================================
舒适区 S_risk → Jenks(3级) → {安全, 关注, 预警}
应力区 S_risk → Jenks(3级) → {低危, 中危, 高危}
校准映射 → 统一四级 {1,2,3,4}
"""

import os, json
import numpy as np
import pandas as pd
from step0_config import *

EPS = 1e-10


def jenks_1d(values, n_classes):
    """Jenks自然断点: 一维DP, 返回 (n_classes-1) 个断点"""
    v = np.sort(values)
    n = len(v)
    if n < n_classes or n_classes < 2:
        return []
    mat_d = np.full((n + 1, n_classes + 1), 0.0)
    mat_b = np.zeros((n + 1, n_classes + 1), dtype=int)

    def ssd(i, j):
        if j <= i:
            return 0.0
        sl = v[i:j]
        m = sl.mean()
        return ((sl - m) ** 2).sum()

    for i in range(1, n + 1):
        mat_d[i][1] = ssd(0, i)
        mat_b[i][1] = 1

    for k in range(2, n_classes + 1):
        mat_d[1][k] = 1.0
        mat_b[1][k] = 1
        for i in range(2, n + 1):
            best = float("inf")
            best_j = 1
            for j in range(max(1, k - 1), i):
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


def map_grades(value, breaks_comfort, breaks_stress, is_stress):
    """将分区等级映射到统一四级"""
    if is_stress:
        # 应力区3级 → 统一2/3/4
        if len(breaks_stress) < 2:
            if value < 0.33:
                return 2
            elif value < 0.66:
                return 3
            else:
                return 4
        if value <= breaks_stress[0]:
            return 2
        elif value <= breaks_stress[1]:
            return 3
        else:
            return 4
    else:
        # 舒适区3级 → 统一1/2/3
        if len(breaks_comfort) < 2:
            if value < 0.15:
                return 1
            elif value < 0.30:
                return 2
            else:
                return 3
        if value <= breaks_comfort[0]:
            return 1
        elif value <= breaks_comfort[1]:
            return 2
        else:
            return 3


def compute_mapping_table(grades_comfort, grades_stress, s_risk, zone):
    """计算校准映射表: 分区等级→统一等级的NTU超标率锚定"""
    pass


def main():
    print("=" * 60)
    print("  step4.1 — 分区独立 Jenks → 四级映射")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[1/4] 加载风险评分...")
    df = pd.read_csv(OUT_Q4_RISK_SCORES)
    s_risk = df["S_risk"].values
    zone = df["ZONE"].values
    n = len(df)
    print(f"  总样本: {n}")

    mask_comfort = zone == "comfort"
    mask_stress = ~mask_comfort
    n_c, n_s = mask_comfort.sum(), mask_stress.sum()
    print(f"  舒适区: {n_c} ({100*n_c/n:.1f}%)  应力区: {n_s} ({100*n_s/n:.1f}%)")

    print("\n[2/4] 分区独立 Jenks...")
    breaks_comfort = jenks_1d(s_risk[mask_comfort], 3)
    breaks_stress = jenks_1d(s_risk[mask_stress], 3)
    print(f"  舒适区断点: {[round(b, 4) for b in breaks_comfort]}")
    print(f"  应力区断点: {[round(b, 4) for b in breaks_stress]}")

    print("\n[3/4] 校准映射 → 统一四级...")
    grades = np.zeros(n, dtype=int)
    for i in range(n):
        grades[i] = map_grades(s_risk[i], breaks_comfort, breaks_stress,
                               zone[i] == "stress")

    dist = [int((grades == g).sum()) for g in [1, 2, 3, 4]]
    print(f"  等级分布: 1={dist[0]}({100*dist[0]/n:.1f}%) "
          f"2={dist[1]}({100*dist[1]/n:.1f}%) "
          f"3={dist[2]}({100*dist[2]/n:.1f}%) "
          f"4={dist[3]}({100*dist[3]/n:.1f}%)")

    zone_x_grade = {}
    for z, zl in [("comfort", "comfort"), ("stress", "stress")]:
        zmask = zone == zl
        for g in [1, 2, 3, 4]:
            key = f"{z}_grade{g}"
            zone_x_grade[key] = int((zmask & (grades == g)).sum())

    print("\n[4/4] 保存输出...")
    np.save(OUT_Q4_GRADES, grades)

    breaks_result = {
        "theta": Q4_THETA,
        "n_comfort": int(n_c),
        "n_stress": int(n_s),
        "comfort_breaks": [round(b, 6) for b in breaks_comfort],
        "stress_breaks": [round(b, 6) for b in breaks_stress],
        "grade_distribution": {
            "grade1": int(dist[0]), "grade2": int(dist[1]),
            "grade3": int(dist[2]), "grade4": int(dist[3]),
        },
        "zone_x_grade": zone_x_grade,
    }

    with open(OUT_Q4_BREAKS_COMF, "w", encoding="utf-8") as f:
        json.dump({"breaks": [round(b, 6) for b in breaks_comfort],
                    "n": int(n_c)}, f, indent=2, ensure_ascii=False)

    with open(OUT_Q4_BREAKS_STR, "w", encoding="utf-8") as f:
        json.dump({"breaks": [round(b, 6) for b in breaks_stress],
                    "n": int(n_s)}, f, indent=2, ensure_ascii=False)

    # Save to scores df
    df["GRADE"] = grades
    df.to_csv(OUT_Q4_RISK_SCORES, index=False, encoding="utf-8-sig")

    print(f"  [DONE] {OUT_Q4_GRADES}")
    print(f"  [DONE] {OUT_Q4_BREAKS_COMF}")
    print(f"  [DONE] {OUT_Q4_BREAKS_STR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
