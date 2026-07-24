"""
step2.1_tcn_dynamic_model.py
Q2 滤后水浊度动态时滞模型 — TCN + 时滞注意力 + 物理EmbeddedLoss + PINN正则
===================================================================================
双模式:
  --mode physical : 按 tau_params_physical.json 物理先验对齐输入
  --mode adaptive : 将各变量的 lag 0-3 堆叠为特征, TCN 注意力自学习时滞
  --mode both     : 先后跑两个模式并保存对比

用法:
  python step2.1_tcn_dynamic_model.py --mode both
"""

import os, json, sys, argparse, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_IN = 31
HIDDEN_DIM = 64
TCN_LAYERS = 4
KERNEL_SIZE = 3
DILATIONS = [1, 2, 4, 8]
DROPOUT = 0.1
BATCH_SIZE = 128
MAX_EPOCHS = 500
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 30
N_SPLITS = 5

LAMBDA_SMOOTH = 0.1
LAMBDA_UPPER = 0.5
LAMBDA_PINN = 0.05
HUBER_DELTA = 1.0
EPS = 1e-6

INPUT_VARS = ["RW_NTU", "RW_FLOW", "RW_PH", "ALUM"]
AUTOREG_LAGS = 6
ADAPTIVE_LAGS = 4


# ==============================
# 数据加载与对齐
# ==============================
def load_raw(clean_csv):
    df = pd.read_csv(clean_csv).dropna(subset=["FILT_NTU"])
    y_raw = df["FILT_NTU"].values.astype(np.float64)
    x_raw = {v: df[v].values.astype(np.float64) for v in INPUT_VARS}
    n = len(y_raw)
    y_log = np.log1p(y_raw)
    x_log = {v: np.log1p(x_raw[v]) for v in INPUT_VARS}
    x_rw_raw = x_raw["RW_NTU"].copy()
    return y_log, y_raw, x_log, x_rw_raw


def build_physical(y_log, x_log, x_rw_raw, tau_params):
    """物理对齐模式: 各变量按 tau 平移后作为单维特征"""
    aligned = {}
    for v in INPUT_VARS:
        d_star = tau_params[v]["steps"]
        x_shifted = np.roll(x_log[v], d_star)
        x_shifted[:d_star] = np.nan
        aligned[v] = x_shifted

    x_rw_aligned = np.roll(x_rw_raw, tau_params["RW_NTU"]["steps"])
    x_rw_aligned[:tau_params["RW_NTU"]["steps"]] = np.nan

    X_list = [aligned[v][:, None] for v in INPUT_VARS]
    for lag in range(1, AUTOREG_LAGS + 1):
        y_ar = np.roll(y_log, lag); y_ar[:lag] = np.nan
        X_list.append(y_ar[:, None])

    X_full = np.column_stack(X_list)
    return _pack_sequences(X_full, y_log, x_rw_aligned), X_full.shape[1]


def build_adaptive(y_log, x_log, x_rw_raw):
    """自适应模式: 每个输入变量取 lag 0..3, 堆叠为多维特征, TCN自选"""
    X_list = []
    for v in INPUT_VARS:
        for lag in range(ADAPTIVE_LAGS):
            x_shifted = np.roll(x_log[v], lag)
            x_shifted[:lag] = np.nan
            X_list.append(x_shifted[:, None])

    for lag in range(1, AUTOREG_LAGS + 1):
        y_ar = np.roll(y_log, lag); y_ar[:lag] = np.nan
        X_list.append(y_ar[:, None])

    X_full = np.column_stack(X_list)
    x_rw_aligned = x_rw_raw.copy()
    return _pack_sequences(X_full, y_log, x_rw_aligned), X_full.shape[1]


def _pack_sequences(X_full, y_log, x_rw_ntu_raw):
    n = len(y_log)
    seqs_x, seqs_y, seqs_w = [], [], []
    for t in range(T_IN, n):
        x_window = X_full[t - T_IN:t]
        y_target = y_log[t]
        w_upper = x_rw_ntu_raw[t]
        if np.any(np.isnan(x_window)) or np.isnan(y_target) or np.isnan(w_upper):
            continue
        seqs_x.append(x_window)
        seqs_y.append(y_target)
        seqs_w.append(w_upper)
    return (np.stack(seqs_x).astype(np.float32),
            np.array(seqs_y, dtype=np.float32),
            np.array(seqs_w, dtype=np.float32))


# ==============================
# TCN 模型
# ==============================
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=self.padding)
        nn.utils.weight_norm(self.conv)

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :out.shape[2] - self.padding]


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.conv = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.conv(x); out = self.relu(out); out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        u = torch.tanh(self.W(h))
        scores = self.v(u).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights.unsqueeze(-1) * h, dim=1)
        return context, weights


class Q2Model(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.tcn_blocks = nn.ModuleList()
        for i in range(TCN_LAYERS):
            self.tcn_blocks.append(
                TCNBlock(input_dim if i == 0 else hidden_dim, hidden_dim,
                         KERNEL_SIZE, DILATIONS[i], DROPOUT))
        self.attention = TemporalAttention(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.k_param = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        h = x.permute(0, 2, 1)
        for block in self.tcn_blocks:
            h = block(h)
        h = h.permute(0, 2, 1)
        context, attn_weights = self.attention(h)
        return self.fc_out(context), attn_weights


# ==============================
# 损失函数
# ==============================
def huber_loss(y_pred, y_true, delta=HUBER_DELTA):
    diff = y_pred.squeeze() - y_true
    abs_diff = torch.abs(diff)
    mask = abs_diff <= delta
    return torch.where(mask, 0.5 * diff ** 2, delta * (abs_diff - 0.5 * delta)).mean()


def compute_loss(y_pred, y_true, x_rw_ntu_raw, k_param):
    loss = huber_loss(y_pred, y_true)
    y_pred_s = y_pred.squeeze()
    if len(y_pred_s) > 1:
        loss += LAMBDA_SMOOTH * torch.mean(torch.abs(y_pred_s[1:] - y_pred_s[:-1]))
    y_pred_real = torch.expm1(y_pred_s)
    loss += LAMBDA_UPPER * torch.mean(F.relu(y_pred_real - x_rw_ntu_raw.float()))
    rw_log = torch.tensor(np.log1p(np.maximum(x_rw_ntu_raw.cpu().numpy(), 0)),
                          dtype=torch.float32).to(y_pred_s.device)
    loss += LAMBDA_PINN * F.mse_loss(y_pred_s, rw_log - torch.abs(k_param))
    return loss


# ==============================
# 训练与评估
# ==============================
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for bx, by, bw in loader:
        bx, by, bw = bx.to(DEVICE), by.to(DEVICE), bw.to(DEVICE)
        optimizer.zero_grad()
        y_hat, _ = model(bx)
        loss = compute_loss(y_hat, by, bw, model.k_param)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(bx)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_mse, total_mae, total_viol = 0.0, 0.0, 0
    preds, trues, attns_list = [], [], []
    for bx, by, bw in loader:
        bx, by, bw = bx.to(DEVICE), by.to(DEVICE), bw.to(DEVICE)
        y_hat, attn_w = model(bx)
        y_hat_s = y_hat.squeeze()
        total_mse += F.mse_loss(y_hat_s, by, reduction="sum").item()
        total_mae += F.l1_loss(y_hat_s, by, reduction="sum").item()
        preds.append(y_hat_s.cpu().numpy())
        trues.append(by.cpu().numpy())
        attns_list.append(attn_w.cpu().numpy())
        total_viol += np.sum(np.expm1(y_hat_s.cpu().numpy()) > bw.cpu().numpy())
    n = len(loader.dataset)
    rmse = np.sqrt(total_mse / n)
    mae = total_mae / n
    yp, yt = np.concatenate(preds), np.concatenate(trues)
    yp_r, yt_r = np.expm1(yp), np.expm1(yt)
    ss_res = np.sum((yp_r - yt_r) ** 2)
    ss_tot = np.sum((yt_r - np.mean(yt_r)) ** 2)
    r2 = 1 - ss_res / (ss_tot + EPS)
    return rmse, r2, mae, total_viol / n, attns_list, yp_r, yt_r


def train_model(X, Y, W, input_dim, mode_name):
    print(f"\n  [{mode_name}] input_dim={input_dim} seqs={len(X)}", flush=True)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_metrics, all_attns, all_preds, all_trues = [], [], [], []
    best_state = None
    best_rmse = float("inf")

    for fold, (tr_idx, vl_idx) in enumerate(tscv.split(X)):
        X_tr, Y_tr, W_tr = X[tr_idx], Y[tr_idx], W[tr_idx]
        X_vl, Y_vl, W_vl = X[vl_idx], Y[vl_idx], W[vl_idx]
        tr_ldr = DataLoader(TensorDataset(
            torch.FloatTensor(X_tr), torch.FloatTensor(Y_tr),
            torch.FloatTensor(W_tr)), batch_size=BATCH_SIZE, shuffle=False)
        vl_ldr = DataLoader(TensorDataset(
            torch.FloatTensor(X_vl), torch.FloatTensor(Y_vl),
            torch.FloatTensor(W_vl)), batch_size=BATCH_SIZE, shuffle=False)

        model = Q2Model(input_dim).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)

        best_fold_rmse, patience_cnt = float("inf"), 0
        for _ in range(MAX_EPOCHS):
            train_epoch(model, tr_ldr, opt)
            rmse_v, _, _, _, _, _, _ = evaluate(model, vl_ldr)
            sch.step(rmse_v)
            if rmse_v < best_fold_rmse:
                best_fold_rmse = rmse_v
                best_fold_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

        model.load_state_dict(best_fold_state)
        rmse_f, r2_f, mae_f, viol_f, attns_f, preds_f, trues_f = evaluate(model, vl_ldr)
        fold_metrics.append({"fold": fold, "rmse": rmse_f, "r2": r2_f, "mae": mae_f,
                             "violation_rate": viol_f, "k": model.k_param.item()})
        all_attns.extend(attns_f)
        all_preds.append(preds_f)
        all_trues.append(trues_f)
        if rmse_f < best_rmse:
            best_rmse = rmse_f
            best_state = best_fold_state
        print(f"    Fold{fold}: RMSE={rmse_f:.4f} R2={r2_f:.4f} k={model.k_param.item():.3f}", flush=True)

    avg = {k: np.mean([f[k] for f in fold_metrics]) for k in ["rmse", "r2", "mae", "violation_rate"]}
    avg["k_pinn"] = np.mean([f["k"] for f in fold_metrics])
    avg["mode"] = mode_name
    return avg, best_state, all_attns, np.concatenate(all_preds), np.concatenate(all_trues)


# ==============================
# 主流程
# ==============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["physical", "adaptive", "both"], default="both")
    args = parser.parse_args()

    clean_csv = os.path.join(OUTPUT_DIR, "clean_data.csv")
    y_log, y_raw, x_log, x_rw_raw = load_raw(clean_csv)

    results = {}
    states = {}

    for run_mode in (["physical", "adaptive"] if args.mode == "both" else [args.mode]):
        if run_mode == "physical":
            tau_file = os.path.join(OUTPUT_DIR, "tau_params_physical.json")
            if not os.path.exists(tau_file):
                print("[step2.1] tau_params_physical.json not found, skip physical mode")
                continue
            with open(tau_file, encoding="utf-8") as f:
                tau = json.load(f)
            print(f"\n[step2.1] ==== 物理先验对齐模式 ====", flush=True)
            (X, Y, W), in_dim = build_physical(y_log, x_log, x_rw_raw, tau)
            mode_label = "physical_prior"
        else:
            print(f"\n[step2.1] ==== 自适应时滞模式 ====", flush=True)
            (X, Y, W), in_dim = build_adaptive(y_log, x_log, x_rw_raw)
            mode_label = "adaptive_delay"

        avg, state, attns, preds, trues = train_model(X, Y, W, in_dim, mode_label)
        results[mode_label] = avg
        states[mode_label] = state

        torch.save(state, os.path.join(OUTPUT_DIR, f"tcn_model_{mode_label}.pt"))
        pd.DataFrame({"pred_FILT_NTU": preds, "true_FILT_NTU": trues}).to_csv(
            os.path.join(OUTPUT_DIR, f"q2_predictions_{mode_label}.csv"), index=False)

        if attns:
            all_a = np.concatenate([a for a in attns if a.size > 0], axis=0) if attns else np.array([])
            if all_a.size > 0:
                avg_attn = np.mean(all_a, axis=0)
                np.save(os.path.join(OUTPUT_DIR, f"q2_attention_{mode_label}.npy"), all_a)
                fig, ax = plt.subplots(figsize=(12, 3))
                lags_h = (np.arange(len(avg_attn))[::-1]) * 2
                ax.bar(lags_h, avg_attn, color="steelblue", alpha=0.8)
                ax.set_xlabel("lag (h)"); ax.set_ylabel("avg attention")
                ax.set_title(f"Attention — {mode_label}")
                plt.tight_layout()
                fig.savefig(os.path.join(FIG_DIR, f"q2_attention_{mode_label}.png"), dpi=150)
                plt.close()

    # ==============================
    # 对比输出
    # ==============================
    if len(results) == 2:
        print(f"\n{'='*65}")
        print(f"  物理先验 vs 自适应时滞 对比")
        print(f"{'='*65}")
        r_phys = results["physical_prior"]
        r_adap = results["adaptive_delay"]
        print(f"  {'Metric':<15s} {'物理先验对齐':>15s} {'自适应时滞':>15s} {'Delta':>15s}")
        print(f"  {'-'*60}")
        for k in ["rmse", "r2", "mae"]:
            d = r_adap[k] - r_phys[k] if k != "r2" else r_adap[k] - r_phys[k]
            sign = "+" if k == "r2" else "+"
            print(f"  {k.upper():<15s} {r_phys[k]:>15.4f} {r_adap[k]:>15.4f} {d:>+15.4f}")
        print(f"  {'违规率':<15s} {r_phys['violation_rate']:>15.4f} {r_adap['violation_rate']:>15.4f}")
        print(f"  {'k_PINN':<15s} {r_phys['k_pinn']:>15.4f} {r_adap['k_pinn']:>15.4f}")

        better = "物理先验" if r_phys["r2"] > r_adap["r2"] else "自适应时滞"
        print(f"\n  结论: {better}模式 R2 更高, 物理先验R2={r_phys['r2']:.4f} vs 自适应R2={r_adap['r2']:.4f}")

        # 保存对比表
        df_cmp = pd.DataFrame([
            {"模式": "物理先验对齐", "RMSE": r_phys["rmse"], "R2": r_phys["r2"],
             "MAE": r_phys["mae"], "viol_rate": r_phys["violation_rate"],
             "k_pinn": r_phys["k_pinn"], "input_dim": 10},
            {"模式": "自适应时滞", "RMSE": r_adap["rmse"], "R2": r_adap["r2"],
             "MAE": r_adap["mae"], "viol_rate": r_adap["violation_rate"],
             "k_pinn": r_adap["k_pinn"], "input_dim": 22},
        ])
        df_cmp.to_csv(os.path.join(OUTPUT_DIR, "q2_mode_comparison.csv"), index=False, encoding="utf-8-sig")

        # 柱状图
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        colors = ["darkorange", "steelblue"]
        modes = ["物理先验对齐", "自适应时滞"]
        axes[0].bar(modes, [r_phys["rmse"], r_adap["rmse"]], color=colors, alpha=0.85)
        axes[0].set_ylabel("RMSE"); axes[0].set_title("RMSE")
        axes[1].bar(modes, [r_phys["r2"], r_adap["r2"]], color=colors, alpha=0.85)
        axes[1].set_ylabel("R2"); axes[1].set_title("R2")
        plt.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "q2_mode_comparison.png"), dpi=150)
        plt.close()
        print("[step2.1] figures/q2_mode_comparison.png saved")

    elif len(results) == 1:
        mode = list(results.keys())[0]
        avg = results[mode]
        print(f"\n[step2.1] {mode}: RMSE={avg['rmse']:.4f} R2={avg['r2']:.4f} MAE={avg['mae']:.4f}")

        df_m = pd.DataFrame([{"model": mode, "rmse": avg["rmse"], "r2": avg["r2"],
                              "mae": avg["mae"], "violation_rate": avg["violation_rate"],
                              "k_pinn": avg["k_pinn"]}])
        df_m.to_csv(os.path.join(OUTPUT_DIR, "q2_metrics.csv"), index=False, encoding="utf-8-sig")

    print("\n[step2.1] Done.")


if __name__ == "__main__":
    main()
