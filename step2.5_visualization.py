"""
step2.5_visualization.py
Q2 统一出图 — 汇总时滞分析 + 模型预测 + 消融对比 + 注意力热力图
=================================================================
输入: output/q2_predictions.csv (由 step2.1 产生)
       output/q2_ablation.csv
       output/q2_baseline_comparison.csv
       output/q2_attention_weights.npy
       output/tau_analysis.csv
输出: figures/q2_pred_vs_actual.png
       figures/q2_dl_vs_baseline_summary.png
       figures/q2_summary_table.txt
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def plot_pred_vs_actual():
    pred_file = os.path.join(OUTPUT_DIR, "q2_predictions.csv")
    if not os.path.exists(pred_file):
        print("[step2.5] q2_predictions.csv 未找到，跳过预测vs实际图")
        return

    df = pd.read_csv(pred_file)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(df.iloc[:, 1], df.iloc[:, 0], alpha=0.4, s=8, c="steelblue")
    lim_min = min(df.iloc[:, 1].min(), df.iloc[:, 0].min())
    lim_max = max(df.iloc[:, 1].max(), df.iloc[:, 0].max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1, label="y=x")
    ax.set_xlabel("真实 FILT.NTU")
    ax.set_ylabel("预测 FILT.NTU")
    ax.set_title("TCN — 预测 vs 实际 (FILT.NTU)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "q2_pred_vs_actual.png"), dpi=150)
    plt.close()
    print("[step2.5] figures/q2_pred_vs_actual.png 已保存")


def plot_dl_vs_baseline():
    ablation_file = os.path.join(OUTPUT_DIR, "q2_ablation.csv")
    baseline_file = os.path.join(OUTPUT_DIR, "q2_baseline_comparison.csv")

    models, rmses, r2s = [], [], []

    if os.path.exists(ablation_file):
        df = pd.read_csv(ablation_file)
        full = df[df["config"] == "完整模型"]
        if len(full) > 0:
            models.append("完整模型(TCN)")
            rmses.append(full["rmse"].values[0])
            r2s.append(full["r2"].values[0])

        gr = df[df["config"] == "GRU替代TCN"]
        if len(gr) > 0:
            models.append("TCN→GRU")
            rmses.append(gr["rmse"].values[0])
            r2s.append(gr["r2"].values[0])

    if os.path.exists(baseline_file):
        df_b = pd.read_csv(baseline_file)
        for _, row in df_b.iterrows():
            name = row.get("模型", row.get("model_name", "未知"))
            models.append(name)
            rmses.append(float(row.get("RMSE", row.get("rmse", 0))))
            r2s.append(float(row.get("R2", row.get("r2", 0))))

    if len(models) == 0:
        print("[step2.5] 无对比数据，跳过 DL vs baseline 图")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = ["darkorange" if "TCN" in m else "steelblue" for m in models]

    axes[0].bar(range(len(models)), rmses, color=colors, alpha=0.85)
    axes[0].set_xticks(range(len(models)))
    axes[0].set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("深度学习 vs 基准模型 — RMSE")

    axes[1].bar(range(len(models)), r2s, color=colors, alpha=0.85)
    axes[1].set_xticks(range(len(models)))
    axes[1].set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    axes[1].set_ylabel("R²")
    axes[1].set_title("深度学习 vs 基准模型 — R²")

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "q2_dl_vs_baseline_summary.png"), dpi=150)
    plt.close()
    print("[step2.5] figures/q2_dl_vs_baseline_summary.png 已保存")


def write_summary():
    lines = ["Q2 模型总结",
             "=" * 60, ""]

    ablation_file = os.path.join(OUTPUT_DIR, "q2_ablation.csv")
    if os.path.exists(ablation_file):
        df = pd.read_csv(ablation_file)
        lines.append("消融实验对比:")
        lines.append("-" * 40)
        full = df[df["config"] == "完整模型"]
        if len(full) > 0:
            lines.append(f"  完整模型(TCN+物理Loss+PINN): RMSE={full['rmse'].values[0]:.4f} R²={full['r2'].values[0]:.4f}")
        for _, row in df.iterrows():
            if row["config"] != "完整模型":
                lines.append(f"  {row['config']}: RMSE={row['rmse']:.4f} R²={row['r2']:.4f}")
        lines.append("")

    baseline_file = os.path.join(OUTPUT_DIR, "q2_baseline_comparison.csv")
    if os.path.exists(baseline_file):
        df_b = pd.read_csv(baseline_file)
        lines.append("基准模型对比:")
        lines.append("-" * 40)
        for _, row in df_b.iterrows():
            name = row.get("模型", row.get("model_name", "未知"))
            rmse = row.get("RMSE", row.get("rmse", "-"))
            r2 = row.get("R2", row.get("r2", "-"))
            lines.append(f"  {name}: RMSE={rmse} R²={r2}")
        lines.append("")

    tau_file = os.path.join(OUTPUT_DIR, "tau_params.json")
    if os.path.exists(tau_file):
        import json
        with open(tau_file, "r") as f:
            tau = json.load(f)
        lines.append("最优时滞参数:")
        lines.append("-" * 40)
        for v in ["RW_NTU", "RW_FLOW", "RW_PH", "ALUM"]:
            tv = tau[v]
            lines.append(f"  {v}: d*={tv['steps']}步 ({tv['hours']}h)")
        lines.append("")

    text = "\n".join(lines)
    with open(os.path.join(OUTPUT_DIR, "q2_summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print("[step2.5] output/q2_summary.txt 已保存")
    print("\n" + text)


def main():
    print("[step2.5] Q2 统一可视化")

    plot_pred_vs_actual()
    plot_dl_vs_baseline()
    write_summary()

    print("\n[step2.5] 完成.")


if __name__ == "__main__":
    main()
