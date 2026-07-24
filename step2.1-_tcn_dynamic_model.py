"""
step2.1_tcn_dynamic_model.py
Q2 滤后水浊度动态时滞模型 — TCN + 时滞注意力 + 物理EmbeddedLoss + PINN正则
===================================================================================
输入: clean_data.csv, tau_params.json
输出: tcn_model.pt, q2_predictions.csv, q2_metrics.csv
       q2_attention_weights.npy, q2_pinn_k.npy
       figures/q2_loss_curve.png, figures/q2_attention_avg.png

消融实验统一在 step5.0 中执行。
"""

import os, json, sys
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


# ==============================
# 数据加载与对齐
# ==============================
def load_and_align(clean_csv, tau_params):
    df = pd.read_csv(clean_csv)
    df = df.dropna(subset=["FILT_NTU"])
    n = len(df)

    y_raw = df["FILT_NTU"].values.astype(np.float64)
    x_raw_all = {}
    for v in INPUT_VARS:
        x_raw_all[v] = df[v].values.astype(np.float64)

    y_log = np.log1p(y_raw)
    x_log_all = {v: np.log1p(x_raw_all[v]) for v in INPUT_VARS}

    aligned = {}
    for v in INPUT_VARS:
        d_star = tau_params[v]["steps"]
        x_shifted = np.roll(x_log_all[v], d_star)
        x_shifted[:d_star] = np.nan
        aligned[v] = x_shifted

    x_rw_ntu_raw = np.roll(x_raw_all["RW_NTU"], tau_params["RW_NTU"]["steps"])
    x_rw_ntu_raw[:tau_params["RW_NTU"]["steps"]] = np.nan

    return y_log, y_raw, aligned, x_rw_ntu_raw


def build_sequences(y_log, aligned, x_rw_ntu_raw, t_in=T_IN, ar_lags=AUTOREG_LAGS):
    n = len(y_log)
    X_list = [aligned[v][:, None] for v in INPUT_VARS]
    for lag in range(1, ar_lags + 1):
        y_ar = np.roll(y_log, lag)
        y_ar[:lag] = np.nan
        X_list.append(y_ar[:, None])

    X_full = np.column_stack(X_list)
    input_dim = X_full.shape[1]

    seqs_x, seqs_y, seqs_w = [], [], []
    for t in range(t_in, n):
        x_window = X_full[t - t_in:t]
        y_target = y_log[t]
        w_upper = x_rw_ntu_raw[t]
        if np.any(np.isnan(x_window)) or np.isnan(y_target) or np.isnan(w_upper):
            continue
        seqs_x.append(x_window)
        seqs_y.append(y_target)
        seqs_w.append(w_upper)

    return (np.stack(seqs_x).astype(np.float32),
            np.array(seqs_y, dtype=np.float32),
            np.array(seqs_w, dtype=np.float32),
            input_dim)


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
        out = self.conv(x)
        out = self.relu(out)
        out = self.dropout(out)
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
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, layers=TCN_LAYERS,
                 kernel_size=KERNEL_SIZE, dilations=DILATIONS, dropout=DROPOUT):
        super().__init__()
        self.tcn_blocks = nn.ModuleList()
        for i in range(layers):
            self.tcn_blocks.append(
                TCNBlock(input_dim if i == 0 else hidden_dim, hidden_dim,
                         kernel_size, dilations[i], dropout))
        self.attention = TemporalAttention(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.k_param = nn.Parameter(torch.tensor(0.5))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if not any(p is m for p in self.parameters() if hasattr(m, 'is_weight_norm')):
                    continue
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = x.permute(0, 2, 1)
        for block in self.tcn_blocks:
            h = block(h)
        h = h.permute(0, 2, 1)
        context, attn_weights = self.attention(h)
        y_hat = self.fc_out(context)
        return y_hat, attn_weights


# ==============================
# 损失函数
# ==============================
def huber_loss(y_pred, y_true, delta=HUBER_DELTA):
    diff = y_pred.squeeze() - y_true
    abs_diff = torch.abs(diff)
    mask = abs_diff <= delta
    loss = torch.where(mask, 0.5 * diff ** 2, delta * (abs_diff - 0.5 * delta))
    return loss.mean()


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
        y_real = np.expm1(y_hat_s.cpu().numpy())
        x_raw = bw.cpu().numpy()
        total_viol += np.sum(y_real > x_raw)
    n = len(loader.dataset)
    rmse = np.sqrt(total_mse / n)
    mae = total_mae / n
    y_all_pred = np.concatenate(preds)
    y_all_true = np.concatenate(trues)
    yp_r, yt_r = np.expm1(y_all_pred), np.expm1(y_all_true)
    ss_res = np.sum((yp_r - yt_r) ** 2)
    ss_tot = np.sum((yt_r - np.mean(yt_r)) ** 2)
    r2 = 1 - ss_res / (ss_tot + EPS)
    viol_rate = total_viol / n
    return rmse, r2, mae, viol_rate, attns_list, yp_r, yt_r


# ==============================
# 主流程
# ==============================
def main():
    clean_csv = os.path.join(OUTPUT_DIR, "clean_data.csv")
    tau_file = os.path.join(OUTPUT_DIR, "tau_params.json")
    if not os.path.exists(tau_file):
        print("[step2.1] 请先运行 step2.0_time_delay_estimation.py")
        sys.exit(1)

    with open(tau_file, "r", encoding="utf-8") as f:
        tau_full = json.load(f)

    y_log, y_raw, aligned, x_rw_raw = load_and_align(clean_csv, tau_full)
    X, Y, W, input_dim = build_sequences(y_log, aligned, x_rw_raw)
    print(f"[step2.1] 序列数={len(X)} 输入维度={input_dim}")

    # --- 训练 ---
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_metrics, all_fold_attns, all_fold_preds, all_fold_trues = [], [], [], []
    best_model_state = None
    best_overall_rmse = float("inf")

    for fold, (tr_idx, vl_idx) in enumerate(tscv.split(X)):
        X_tr, Y_tr, W_tr = X[tr_idx], Y[tr_idx], W[tr_idx]
        X_vl, Y_vl, W_vl = X[vl_idx], Y[vl_idx], W[vl_idx]
        tr_loader = DataLoader(TensorDataset(
            torch.FloatTensor(X_tr), torch.FloatTensor(Y_tr),
            torch.FloatTensor(W_tr)), batch_size=BATCH_SIZE, shuffle=False)
        vl_loader = DataLoader(TensorDataset(
            torch.FloatTensor(X_vl), torch.FloatTensor(Y_vl),
            torch.FloatTensor(W_vl)), batch_size=BATCH_SIZE, shuffle=False)

        model = Q2Model(input_dim).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        best_rmse, patience_cnt = float("inf"), 0
        for epoch in range(MAX_EPOCHS):
            train_loss = train_epoch(model, tr_loader, optimizer)
            rmse_val, _, _, _, _, _, _ = evaluate(model, vl_loader)
            scheduler.step(rmse_val)
            if rmse_val < best_rmse:
                best_rmse = rmse_val
                best_fold_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

        model.load_state_dict(best_fold_state)
        rmse_f, r2_f, mae_f, viol_f, attns_f, preds_f, trues_f = evaluate(model, vl_loader)
        fold_metrics.append({"fold": fold, "rmse": rmse_f, "r2": r2_f, "mae": mae_f,
                             "violation_rate": viol_f, "k_pinn": model.k_param.item()})
        all_fold_attns.extend(attns_f)
        all_fold_preds.append(preds_f)
        all_fold_trues.append(trues_f)
        if rmse_f < best_overall_rmse:
            best_overall_rmse = rmse_f
            best_model_state = best_fold_state
        print(f"  Fold{fold}: RMSE={rmse_f:.4f} R2={r2_f:.4f} MAE={mae_f:.4f} "
              f"viol={viol_f:.3f} k={model.k_param.item():.3f}")

    # --- 保存模型 ---
    best_model = Q2Model(input_dim)
    best_model.load_state_dict(best_model_state)
    torch.save(best_model.state_dict(), os.path.join(OUTPUT_DIR, "tcn_model.pt"))
    np.save(os.path.join(OUTPUT_DIR, "q2_pinn_k.npy"),
            np.array([m["k_pinn"] for m in fold_metrics]))
    print("[step2.1] tcn_model.pt, q2_pinn_k.npy 已保存")

    # --- 保存预测 ---
    all_preds_cat = np.concatenate(all_fold_preds)
    all_trues_cat = np.concatenate(all_fold_trues)
    df_pred = pd.DataFrame({"pred_FILT_NTU": all_preds_cat, "true_FILT_NTU": all_trues_cat})
    df_pred.to_csv(os.path.join(OUTPUT_DIR, "q2_predictions.csv"), index=False, encoding="utf-8-sig")

    # --- 保存指标 ---
    avg_metrics = {k: np.mean([f[k] for f in fold_metrics]) for k in
                   ["rmse", "r2", "mae", "violation_rate"]}
    avg_metrics["model"] = "TCN_full"
    avg_metrics["k_pinn"] = np.mean([f["k_pinn"] for f in fold_metrics])
    pd.DataFrame([avg_metrics]).to_csv(
        os.path.join(OUTPUT_DIR, "q2_metrics.csv"), index=False, encoding="utf-8-sig")
    print(f"\n[step2.1] Complete model: RMSE={avg_metrics['rmse']:.4f} R2={avg_metrics['r2']:.4f} "
          f"MAE={avg_metrics['mae']:.4f} viol={avg_metrics['violation_rate']:.4f}")

    # --- 注意力热力图 ---
    if all_fold_attns:
        avg_attn = np.mean(np.concatenate(all_fold_attns, axis=0), axis=0)
        np.save(os.path.join(OUTPUT_DIR, "q2_attention_weights.npy"),
                np.concatenate(all_fold_attns, axis=0))

        fig, ax = plt.subplots(figsize=(12, 3))
        lags_h = (np.arange(len(avg_attn))[::-1]) * 2
        ax.bar(lags_h, avg_attn, color="steelblue", alpha=0.8)
        for v in INPUT_VARS:
            d_tag = tau_full[v]["steps"]
            ax.axvline(x=d_tag * 2, color="red", linestyle="--", alpha=0.6,
                       label=f"{v}={d_tag*2}h")
        ax.set_xlabel("lag (h)")
        ax.set_ylabel("平均注意力权重")
        ax.set_title("时滞注意力 — 平均权重分布")
        ax.legend(fontsize=7)
        plt.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "q2_attention_avg.png"), dpi=150)
        plt.close()
        print("[step2.1] figures/q2_attention_avg.png 已保存")

    # --- 预测vs实际 ---
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(all_trues_cat, all_preds_cat, alpha=0.4, s=6, c="steelblue")
    lim_min = min(all_trues_cat.min(), all_preds_cat.min())
    lim_max = max(all_trues_cat.max(), all_preds_cat.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1)
    ax.set_xlabel("真实 FILT.NTU")
    ax.set_ylabel("预测 FILT.NTU")
    ax.set_title(f"TCN 预测 vs 实际 (RMSE={avg_metrics['rmse']:.4f}, R²={avg_metrics['r2']:.4f})")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "q2_pred_vs_actual.png"), dpi=150)
    plt.close()
    print("[step2.1] figures/q2_pred_vs_actual.png 已保存")

    print("\n[step2.1] 完成.")


if __name__ == "__main__":
    main()
