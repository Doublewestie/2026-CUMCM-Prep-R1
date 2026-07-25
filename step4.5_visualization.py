"""
step4.5_visualization.py — Q4 可视化 + Excel 输出
====================================================
7张图 + 5 sheet Excel
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from step0_config import *

EPS = 1e-10
os.makedirs(OUT_Q4_FIG_DIR, exist_ok=True)


def make_time_series_with_zone(df):
    """图1: 风险热力图 + 分区背景着色"""
    df = df.sort_values("DATE").reset_index(drop=True)
    x = np.arange(len(df))
    fig, ax1 = plt.subplots(figsize=(20, 6))

    ax1.fill_between(x, 0, 1, where=df["ZONE"].values == "comfort",
                      color="lightgreen", alpha=0.3, transform=ax1.get_xaxis_transform(),
                      label="Comfort Zone")
    ax1.fill_between(x, 0, 1, where=df["ZONE"].values == "stress",
                      color="lightcoral", alpha=0.3, transform=ax1.get_xaxis_transform(),
                      label="Stress Zone")

    color_map = {1: "green", 2: "gold", 3: "orange", 4: "red"}
    for g in [1, 2, 3, 4]:
        mask = df["GRADE"].values == g
        ax1.scatter(x[mask], df["S_risk"].values[mask], c=color_map[g],
                    s=8, alpha=0.6, label=f"Grade {g}", zorder=5)

    ax1.set_xlabel("Time step"); ax1.set_ylabel("S_risk")
    ax1.set_title("Risk Score Time Series with Zone Background")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_Q4_FIG_DIR, "q4_risk_timeseries.png"), dpi=300)
    plt.close()
    print("  [FIG1] risk_timeseries.png")


def make_grade_pie(df):
    """图2: 等级占比饼图(G3/G4过小, 用图例避免重叠)"""
    counts = df["GRADE"].value_counts().sort_index()
    sizes = [counts.get(i, 0) for i in [1, 2, 3, 4]]
    colors = ["green", "gold", "orange", "red"]
    labels = [f"Grade {i}" for i in [1, 2, 3, 4]]
    explode = [0, 0, 0.05, 0.1]

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, colors=colors, autopct="%1.1f%%",
        startangle=90, explode=explode, pctdistance=0.75,
        textprops={"fontsize": 9})
    for at in autotexts:
        at.set_fontsize(8)

    ax.legend(wedges, [f"{l} ({s})" for l, s in zip(labels, sizes)],
              title="Grade", loc="center left", bbox_to_anchor=(0.85, 0, 0.5, 1),
              fontsize=9)
    ax.set_title("Risk Grade Distribution", fontsize=12)

    fig.savefig(os.path.join(OUT_Q4_FIG_DIR, "q4_grade_pie.png"), dpi=300,
                bbox_inches="tight")
    plt.close()
    print("  [FIG2] grade_pie.png")


def make_transition_matrix(df):
    """图3: 状态转移矩阵热力图"""
    grades = df.sort_values("DATE")["GRADE"].values
    n = len(grades)
    cm = np.zeros((4, 4), dtype=float)
    for i in range(1, n):
        cm[int(grades[i - 1]) - 1, int(grades[i]) - 1] += 1.0
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_pct, cmap="YlOrRd", vmin=0, vmax=1)

    for i in range(4):
        for j in range(4):
            val = cm_pct[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val > 0.5 else "black", fontsize=12)

    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(["G1", "G2", "G3", "G4"])
    ax.set_yticklabels(["G1", "G2", "G3", "G4"])
    ax.set_xlabel("To Grade"); ax.set_ylabel("From Grade")
    ax.set_title("State Transition Matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_Q4_FIG_DIR, "q4_transition_matrix.png"), dpi=300)
    plt.close()
    print("  [FIG3] transition_matrix.png")


def make_zone_grade_heatmap(df):
    """图4: 分区 vs 等级联合分布热力图"""
    heat = np.zeros((2, 4), dtype=float)
    for gi, g in enumerate([1, 2, 3, 4]):
        heat[0, gi] = ((df["ZONE"] == "comfort") & (df["GRADE"] == g)).sum()
        heat[1, gi] = ((df["ZONE"] == "stress") & (df["GRADE"] == g)).sum()

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(heat, cmap="Blues", aspect="auto")

    for i in range(2):
        for j in range(4):
            val = int(heat[i, j])
            ax.text(j, i, str(val), ha="center", va="center",
                    color="white" if val > heat.max() * 0.6 else "black")

    ax.set_xticks(range(4)); ax.set_yticks(range(2))
    ax.set_xticklabels(["G1", "G2", "G3", "G4"])
    ax.set_yticklabels(["Comfort", "Stress"])
    ax.set_title("Zone × Grade Joint Distribution")
    fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_Q4_FIG_DIR, "q4_zone_grade_heatmap.png"), dpi=300)
    plt.close()
    print("  [FIG4] zone_grade_heatmap.png")


def make_dimension_contribution(df):
    """图5: 各维度贡献堆叠面积图"""
    df = df.sort_values("DATE").reset_index(drop=True)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(20, 5))

    f1 = df["f1"].values * w1 if "f1" in df.columns else df["f1"].values
    f2 = df["f2"].values * w2 if "f2" in df.columns else df["f2"].values
    f3 = df["f3"].values * w3 if "f3" in df.columns else df["f3"].values

    ax.stackplot(x, f1, f2, f3, labels=["f1 Amplitude", "f2 Duration", "f3 Trend"],
                 colors=["steelblue", "darkorange", "seagreen"], alpha=0.8)

    ax.set_xlabel("Time step"); ax.set_ylabel("Weighted contribution")
    ax.set_title("Dimension Contributions Over Time")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_Q4_FIG_DIR, "q4_dimension_contribution.png"), dpi=300)
    plt.close()
    print("  [FIG5] dimension_contribution.png")


def make_event_confusion(backtest):
    """图6: 事件回溯混淆矩阵"""
    cm = np.array(backtest["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="YlOrRd")

    for i in range(4):
        for j in range(4):
            val = int(cm[i, j])
            ax.text(j, i, str(val), ha="center", va="center",
                    color="white" if val > cm.max() * 0.6 else "black")

    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(["G1", "G2", "G3", "G4"])
    ax.set_yticklabels(["None", "Raw shock", "Filt stress", "Exceed"])
    ax.set_xlabel("Risk Grade"); ax.set_ylabel("Event Type")
    ax.set_title(f"Event Backtest\nCapture={backtest['exceed_capture_rate']:.0%}, "
                 f"FA={backtest['false_alarm_rate']:.0%}")
    fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_Q4_FIG_DIR, "q4_event_confusion.png"), dpi=300)
    plt.close()
    print("  [FIG6] event_confusion.png")


def make_ntu_vs_risk(df):
    """图7: NTU + S_risk 双轴图"""
    df = df.sort_values("DATE").reset_index(drop=True)
    x = np.arange(len(df))
    fig, ax1 = plt.subplots(figsize=(20, 6))

    ax1.plot(x, df["NTU"].values, "b-", lw=0.8, alpha=0.7, label="NTU")
    ax1.axhline(y=Q4_NTU_LIMIT, color="gray", ls="--", lw=1, alpha=0.5)
    ax1.set_ylabel("NTU", color="b"); ax1.tick_params(axis="y", labelcolor="b")

    ax2 = ax1.twinx()
    ax2.plot(x, df["S_risk"].values, "r-", lw=1.2, label="S_risk")
    ax2.fill_between(x, 0, df["S_risk"].values, color="red", alpha=0.1)
    ax2.set_ylabel("S_risk", color="r"); ax2.tick_params(axis="y", labelcolor="r")
    ax2.set_ylim(-0.05, 1.05)

    ax1.set_xlabel("Time step"); ax1.set_title("NTU and Risk Score")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_Q4_FIG_DIR, "q4_ntu_vs_risk.png"), dpi=300)
    plt.close()
    print("  [FIG7] ntu_vs_risk.png")


def export_excel(df, backtest):
    """输出Excel: 6个sheet (含2025年3月), 并保留已有的2026_march sheet"""
    df_out = df.sort_values("DATE").reset_index(drop=True)
    df_out["DATE"] = pd.to_datetime(df_out["DATE"])
    cols = ["DATE", "NTU", "FILT_NTU", "ZONE", "f1", "f2", "f3", "S_risk", "GRADE"]
    cols = [c for c in cols if c in df_out.columns]

    # Load existing 2026_march if present
    existing_2026 = None
    if os.path.exists(OUT_Q4_EXCEL):
        try:
            existing_2026 = pd.read_excel(OUT_Q4_EXCEL, sheet_name="2026_march")
        except Exception:
            pass

    with pd.ExcelWriter(OUT_Q4_EXCEL, engine="openpyxl") as writer:
        df_out[cols].to_excel(writer, sheet_name="daily_detail", index=False)

        summary = df_out.groupby("GRADE").agg(
            count=("GRADE", "count"),
            mean_NTU=("NTU", "mean"),
            p90_NTU=("NTU", lambda x: np.percentile(x, 90)),
        ).reset_index()
        summary.columns = ["Grade", "Days", "Mean_NTU", "P90_NTU"]
        summary.to_excel(writer, sheet_name="grade_summary", index=False)

        cross = df_out.groupby(["ZONE", "GRADE"]).size().unstack(fill_value=0)
        cross.to_excel(writer, sheet_name="zone_grade_cross")

        cm = np.array(backtest.get("confusion_matrix", [[0]*4]*4))
        cm_df = pd.DataFrame(cm, index=["None", "RawShock", "FiltStress", "Exceed"],
                             columns=["G1", "G2", "G3", "G4"])
        cm_df.to_excel(writer, sheet_name="event_confusion")

        grade_arr = df_out["GRADE"].values
        n = len(grade_arr)
        tm = np.zeros((4, 4))
        for i in range(1, n):
            tm[int(grade_arr[i - 1]) - 1, int(grade_arr[i]) - 1] += 1
        tm_df = pd.DataFrame(tm, index=["G1", "G2", "G3", "G4"],
                             columns=["G1", "G2", "G3", "G4"])
        tm_df.to_excel(writer, sheet_name="transition_matrix")

        # 2025年3月
        mar25 = df_out[(df_out["DATE"].dt.year == 2025) & (df_out["DATE"].dt.month == 3)]
        if len(mar25) > 0:
            mar25[cols].to_excel(writer, sheet_name="2025_march", index=False)
            g1 = int((mar25["GRADE"] == 1).sum())
            g2 = int((mar25["GRADE"] == 2).sum())
            g3 = int((mar25["GRADE"] == 3).sum())
            g4 = int((mar25["GRADE"] == 4).sum())
            total = len(mar25)
            print(f"  2025年3月: Grade1={g1}({100*g1/total:.1f}%) "
                  f"Grade2={g2}({100*g2/total:.1f}%) "
                  f"Grade3={g3}({100*g3/total:.1f}%) "
                  f"Grade4={g4}({100*g4/total:.1f}%)")

        # Re-add 2026_march if existed
        if existing_2026 is not None:
            existing_2026.to_excel(writer, sheet_name="2026_march", index=False)

    print(f"  [EXCEL] {OUT_Q4_EXCEL}")


def main():
    print("=" * 60)
    print("  step4.5 — Q4 可视化 + Excel")
    print("=" * 60)

    os.makedirs(OUT_Q4_FIG_DIR, exist_ok=True)

    print("\n[1/2] 加载数据...")
    df = pd.read_csv(OUT_Q4_RISK_SCORES)
    if "GRADE" not in df.columns or "ZONE" not in df.columns:
        print("  [ERR] GRADE/ZONE not found, run step4.0, step4.1 first")
        return

    global w1, w2, w3
    if os.path.exists(OUT_Q4_WEIGHTS):
        with open(OUT_Q4_WEIGHTS, encoding="utf-8") as f:
            wt = json.load(f)
        w1, w2, w3 = wt.get("f1_amplitude", 0.4), wt.get("f2_duration", 0.3), wt.get("f3_trend", 0.3)
    else:
        w1, w2, w3 = 0.4, 0.3, 0.3

    print("\n[2/2] 生成图表...")
    make_time_series_with_zone(df)
    make_grade_pie(df)
    make_transition_matrix(df)
    make_zone_grade_heatmap(df)
    make_dimension_contribution(df)

    backtest = {}
    if os.path.exists(OUT_Q4_BACKTEST):
        with open(OUT_Q4_BACKTEST, encoding="utf-8") as f:
            backtest = json.load(f)
    make_event_confusion(backtest)
    make_ntu_vs_risk(df)

    print("\n  导出Excel...")
    export_excel(df, backtest)

    print(f"\n  [DONE] 全部输出至 {OUT_Q4_FIG_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
