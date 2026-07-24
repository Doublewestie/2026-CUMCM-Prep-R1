"""
step5.0_ablation.py — Q1 特征层级消融 + Q2 模型消融
===========================================================
Q1 消融 (5个问题):
  ① L2衍生特征增益 (物理公式 vs 原始传感器)
  ② L3滞后特征增益 (历史信息预测力)
  ③ L4聚合特征增益 (趋势/波动信息)
  ④ L2+幂次项增益 (物理泰勒展开价值)
  ⑤ FILT_NTU 主导性 (核心特征移除→模型崩溃？)

Q2 消融 (8组对比):
  1. 完整模型 (TCN + 物理Loss + 注意力 + TE对齐 + PINN)
  2. 无物理Loss (仅Huber)
  3. TE换MIC (时滞用纯MIC)
  4. TCN换GRU (单层GRU替代4层TCN)
  5. 无注意力 (取最后一步隐状态)
  6. 无PINN (去PINN衰减正则)
  7. TE不分段对齐 (用全年TE结果)
  8. CCF对齐 (用交叉相关结果)

输出:
  Q1: output/q1_ablation_results.csv
  Q2: output/q2_ablation.csv
"""

import os, json, sys, copy, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from step0_config import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ==============================
# Q1 辅助函数 (复用原有逻辑)
# ==============================
def boxcox_inverse(y_trans, lam):
    y_t = np.asarray(y_trans, dtype=np.float64).copy()
    if abs(lam) < 1e-6:
        return np.expm1(y_t)
    if lam < 0:
        y_t = np.minimum(y_t, 0.99 / abs(lam))
    else:
        y_t = np.maximum(y_t, -0.99 / lam)
    return (y_t * lam + 1) ** (1.0 / lam) - EPS


def classify_layers(feature_names):
    layers = {"L1": [], "L2": [], "L2+": [], "L3": [], "L4": [], "L5": []}
    for i, name in enumerate(feature_names):
        n = str(name)
        if n in ["PI_load", "GAMMA_alum", "PSI_alum", "OMEGA_night"]:
            layers["L5"].append(i); continue
        if "_lag" in n:
            layers["L3"].append(i); continue
        if any(s in n for s in ["_mean", "_std", "_max", "_delta"]):
            layers["L4"].append(i); continue
        if n in ["FILT_sq","FILT_sqrt","FILT_cubert","neg_ln_eta","eta_sq",
                 "rw_ntu_sqrt","rw_ntu_log","alum_inv","alum_sqrt",
                 "tw_flow_log","dose_ratio_sq","dose_ratio_inv"]:
            layers["L2+"].append(i); continue
        if n in ["eta_coag","phi_alum","psi_hyd","hour_sin","hour_cos",
                 "day_sin","day_cos","is_weekend","is_night"]:
            layers["L2"].append(i); continue
        layers["L1"].append(i)
    return layers


def run_q1_ablation(X, y, feature_names):
    """Q1 特征层级消融"""
    import joblib
    lam = joblib.load(OUT_LAMBDA_NTU)
    layers = classify_layers(feature_names)

    def get_indices(layer_keys):
        idxs = []
        for k in layer_keys:
            idxs.extend(layers[k])
        return sorted(set(idxs))

    L_all_keys = ["L1","L2","L2+","L3","L4","L5"]
    ablation_configs = [
        ("L1 only",           get_indices(["L1"])),
        ("L1+L2",             get_indices(["L1","L2"])),
        ("L1+L2+L3",          get_indices(["L1","L2","L3"])),
        ("L1+L2+L3+L4",       get_indices(["L1","L2","L3","L4"])),
        ("+L5(交互)",         get_indices(["L1","L2","L3","L4","L5"])),
        ("+L2+(幂次)",       get_indices(L_all_keys)),
    ]

    # 轻度消融：仅移除 FILT_NTU 原值（滞后/聚合版本仍在）
    filt_idx = None
    for i, name in enumerate(feature_names):
        if str(name) == "FILT_NTU":
            filt_idx = i; break
    if filt_idx is not None:
        no_filt_idx = [i for i in range(len(feature_names)) if i != filt_idx]
        ablation_configs.append(("remove FILT_NTU(raw)", no_filt_idx))

    # 重度消融：移除全部 FILT_NTU 相关特征（原值+滞后+聚合+幂次）
    filt_related = []
    for i, name in enumerate(feature_names):
        n = str(name).upper()
        if "FILT" in n:
            filt_related.append(i)
    if filt_related:
        no_filt_all_idx = [i for i in range(len(feature_names)) if i not in filt_related]
        n_removed = len(filt_related)
        ablation_configs.append((f"remove ALL_FILT({n_removed})", no_filt_all_idx))

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    results = []
    for config_name, feat_indices in ablation_configs:
        X_sub = X[:, feat_indices]
        fold_rmses, fold_r2s, fold_mapes = [], [], []
        for tr_idx, va_idx in tscv.split(X_sub):
            m = XGBRegressor(**XGB_PARAMS)
            m.fit(X_sub[tr_idx], y[tr_idx])
            pred_t = m.predict(X_sub[va_idx])
            yva_r = boxcox_inverse(y[va_idx], lam)
            pred_r = boxcox_inverse(pred_t, lam)
            fold_rmses.append(np.sqrt(mean_squared_error(yva_r, pred_r)))
            fold_r2s.append(r2_score(yva_r, pred_r))
            fold_mapes.append(mean_absolute_percentage_error(yva_r, pred_r) * 100)
        results.append({
            "消融配置": config_name, "特征数": len(feat_indices),
            "RMSE_mean": np.mean(fold_rmses), "RMSE_std": np.std(fold_rmses),
            "R2_mean": np.mean(fold_r2s), "R2_std": np.std(fold_r2s),
            "MAPE_mean": np.mean(fold_mapes), "MAPE_std": np.std(fold_mapes),
        })
    return pd.DataFrame(results)


# ==============================
# Q2 消融 — TCN 组件复用
# ==============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_IN = 31; HIDDEN_DIM = 64; TCN_LAYERS = 4; KERNEL_SIZE = 3
DILATIONS = [1,2,4,8]; DROPOUT_Q2 = 0.1
BATCH_SIZE = 128; MAX_EPOCHS = 500; LR_Q2 = 1e-3; WD = 1e-4; PATIENCE_Q2 = 20
L_SMOOTH=0.1; L_UPPER=0.5; L_PINN=0.05; H_DELTA=1.0; EPP = 1e-6
N_SPLITS_Q2 = 2
INPUT_VARS = ["RW_NTU","RW_FLOW","RW_PH","ALUM"]
AUTOREG_LAGS = 6


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
        return torch.sum(weights.unsqueeze(-1) * h, dim=1), weights


class Q2TCN(nn.Module):
    def __init__(self, input_dim, use_attention=True, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.tcn_blocks = nn.ModuleList()
        for i in range(TCN_LAYERS):
            self.tcn_blocks.append(
                TCNBlock(input_dim if i == 0 else hidden_dim, hidden_dim,
                         KERNEL_SIZE, DILATIONS[i], DROPOUT_Q2))
        self.use_attention = use_attention
        if use_attention:
            self.attention = TemporalAttention(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.k_param = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        h = x.permute(0, 2, 1)
        for block in self.tcn_blocks:
            h = block(h)
        h = h.permute(0, 2, 1)
        if self.use_attention:
            context, _ = self.attention(h)
        else:
            context = h[:, -1, :]
        return self.fc_out(context)


class Q2GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=1, dropout=0.1)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.k_param = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        h, _ = self.gru(x)
        return self.fc_out(h[:, -1, :])


def huber_loss(y_pred, y_true, delta=H_DELTA):
    diff = y_pred.squeeze() - y_true
    abs_diff = torch.abs(diff)
    mask = abs_diff <= delta
    return torch.where(mask, 0.5 * diff ** 2, delta * (abs_diff - 0.5 * delta)).mean()


def q2_loss_fn(y_pred, y_true, x_rw_ntu_raw, k_param,
               use_smooth, use_upper, use_pinn):
    y_pred_s = y_pred.squeeze()
    loss = huber_loss(y_pred, y_true)
    if use_smooth and len(y_pred_s) > 1:
        loss += L_SMOOTH * torch.mean(torch.abs(y_pred_s[1:] - y_pred_s[:-1]))
    if use_upper:
        y_pred_real = torch.expm1(y_pred_s)
        loss += L_UPPER * torch.mean(F.relu(y_pred_real - x_rw_ntu_raw.float()))
    if use_pinn:
        rw_log = torch.tensor(np.log1p(np.maximum(x_rw_ntu_raw.cpu().numpy(), 0)),
                              dtype=torch.float32).to(y_pred_s.device)
        loss += L_PINN * F.mse_loss(y_pred_s, rw_log - torch.abs(k_param))
    return loss


def load_and_align(clean_csv, tau_params):
    df = pd.read_csv(clean_csv).dropna(subset=["FILT_NTU"])
    y_raw = df["FILT_NTU"].values.astype(np.float64)
    x_raw_all = {v: df[v].values.astype(np.float64) for v in INPUT_VARS}
    y_log = np.log1p(y_raw)
    x_log_all = {v: np.log1p(x_raw_all[v]) for v in INPUT_VARS}
    aligned = {}
    for v in INPUT_VARS:
        d_star = tau_params[v]["steps"]
        x_shifted = np.roll(x_log_all[v], d_star); x_shifted[:d_star] = np.nan
        aligned[v] = x_shifted
    x_rw_raw = np.roll(x_raw_all["RW_NTU"], tau_params["RW_NTU"]["steps"])
    x_rw_raw[:tau_params["RW_NTU"]["steps"]] = np.nan
    return y_log, y_raw, aligned, x_rw_raw


def build_sequences(y_log, aligned, x_rw_raw):
    n = len(y_log)
    X_list = [aligned[v][:, None] for v in INPUT_VARS]
    for lag in range(1, AUTOREG_LAGS + 1):
        y_ar = np.roll(y_log, lag); y_ar[:lag] = np.nan
        X_list.append(y_ar[:, None])
    X_full = np.column_stack(X_list)
    seqs_x, seqs_y, seqs_w = [], [], []
    for t in range(T_IN, n):
        xw = X_full[t - T_IN:t]; yt = y_log[t]; wu = x_rw_raw[t]
        if np.any(np.isnan(xw)) or np.isnan(yt) or np.isnan(wu):
            continue
        seqs_x.append(xw); seqs_y.append(yt); seqs_w.append(wu)
    return (np.stack(seqs_x).astype(np.float32),
            np.array(seqs_y, dtype=np.float32),
            np.array(seqs_w, dtype=np.float32),
            X_full.shape[1])


def train_q2_model(X, Y, W, model, use_smooth, use_upper, use_pinn):
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_Q2)
    fold_metrics = []
    for tr_idx, vl_idx in tscv.split(X):
        X_tr, Y_tr, W_tr = X[tr_idx], Y[tr_idx], W[tr_idx]
        X_vl, Y_vl, W_vl = X[vl_idx], Y[vl_idx], W[vl_idx]
        tr_ldr = DataLoader(TensorDataset(
            torch.FloatTensor(X_tr), torch.FloatTensor(Y_tr),
            torch.FloatTensor(W_tr)), batch_size=BATCH_SIZE, shuffle=False)
        vl_ldr = DataLoader(TensorDataset(
            torch.FloatTensor(X_vl), torch.FloatTensor(Y_vl),
            torch.FloatTensor(W_vl)), batch_size=BATCH_SIZE, shuffle=False)

        m = model.to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=LR_Q2, weight_decay=WD)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)

        best_rmse, patience_cnt, best_state = float("inf"), 0, None
        for _ in range(MAX_EPOCHS):
            m.train()
            for bx, by, bw in tr_ldr:
                bx, by, bw = bx.to(DEVICE), by.to(DEVICE), bw.to(DEVICE)
                opt.zero_grad()
                y_hat = m(bx)
                loss = q2_loss_fn(y_hat, by, bw, m.k_param,
                                  use_smooth, use_upper, use_pinn)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            m.eval()
            val_mse = 0.0
            with torch.no_grad():
                for bx, by, _ in vl_ldr:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    val_mse += F.mse_loss(m(bx).squeeze(), by, reduction="sum").item()
            val_rmse = np.sqrt(val_mse / len(vl_ldr.dataset))
            sch.step(val_rmse)
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
            if patience_cnt >= PATIENCE_Q2:
                break

        m.load_state_dict(best_state)
        m.eval()
        total_mse, total_viol = 0.0, 0
        all_yp, all_yt = [], []
        with torch.no_grad():
            for bx, by, bw in vl_ldr:
                bx, by, bw = bx.to(DEVICE), by.to(DEVICE), bw.to(DEVICE)
                y_hat = m(bx).squeeze()
                total_mse += F.mse_loss(y_hat, by, reduction="sum").item()
                yp_r = np.expm1(y_hat.cpu().numpy())
                yt_r = np.expm1(by.cpu().numpy())
                total_viol += np.sum(yp_r > bw.cpu().numpy())
                all_yp.append(yp_r); all_yt.append(yt_r)
        n_vl = len(vl_ldr.dataset)
        rmse = np.sqrt(total_mse / n_vl)
        yp_c = np.concatenate(all_yp); yt_c = np.concatenate(all_yt)
        ssr = np.sum((yp_c - yt_c) ** 2)
        sst = np.sum((yt_c - np.mean(yt_c)) ** 2)
        r2 = 1 - ssr / (sst + EPP)
        fold_metrics.append({"rmse": rmse, "r2": r2, "mae": np.mean(np.abs(yp_c - yt_c)),
                             "violation_rate": total_viol / n_vl})
    return {k: np.mean([f[k] for f in fold_metrics]) for k in fold_metrics[0]}


def run_q2_ablation():
    """Q2 模型消融实验"""
    clean_csv = os.path.join(OUTPUT_DIR, "clean_data.csv")
    tau_file = os.path.join(OUTPUT_DIR, "tau_params.json")
    if not os.path.exists(tau_file):
        print("[step5.0/Q2] tau_params.json 未找到，跳过 Q2 消融")
        return None

    with open(tau_file, "r", encoding="utf-8") as f:
        tau_full = json.load(f)

    # 默认 (TE融合)
    y_log, _, aligned_def, x_rw_def = load_and_align(clean_csv, tau_full)
    X_def, Y_def, W_def, input_dim = build_sequences(y_log, aligned_def, x_rw_def)

    # MIC
    tau_mic = copy.deepcopy(tau_full)
    for v in INPUT_VARS:
        tau_mic[v]["steps"] = tau_full[v].get("mic_best", tau_full[v]["steps"])
    _, _, aligned_mic, x_rw_mic = load_and_align(clean_csv, tau_mic)
    X_mic, Y_mic, W_mic, _ = build_sequences(y_log, aligned_mic, x_rw_mic)

    # TE不分段
    tau_te = copy.deepcopy(tau_full)
    for v in INPUT_VARS:
        tau_te[v]["steps"] = tau_full[v].get("te_best", tau_full[v]["steps"])
    _, _, aligned_te, x_rw_te = load_and_align(clean_csv, tau_te)
    X_te, Y_te, W_te, _ = build_sequences(y_log, aligned_te, x_rw_te)

    ablation_configs = [
        ("完整模型(TCN+物理Loss+注意力+TE+PINN)",
         Q2TCN, {"input_dim": input_dim, "use_attention": True},
         X_def, Y_def, W_def, True, True, True),
        ("无物理Loss",
         Q2TCN, {"input_dim": input_dim, "use_attention": True},
         X_def, Y_def, W_def, False, False, False),
        ("时滞=纯MIC",
         Q2TCN, {"input_dim": input_dim, "use_attention": True},
         X_mic, Y_mic, W_mic, True, True, True),
        ("TCN→GRU",
         Q2GRU, {"input_dim": input_dim},
         X_def, Y_def, W_def, True, True, True),
        ("无注意力",
         Q2TCN, {"input_dim": input_dim, "use_attention": False},
         X_def, Y_def, W_def, True, True, True),
        ("无PINN",
         Q2TCN, {"input_dim": input_dim, "use_attention": True},
         X_def, Y_def, W_def, True, True, False),
        ("时滞=TE不分段",
         Q2TCN, {"input_dim": input_dim, "use_attention": True},
         X_te, Y_te, W_te, True, True, True),
    ]

    results = []
    for name, model_cls, kw, X, Y, W, sm, up, pn in ablation_configs:
        print(f"  [{name}] ...")
        metrics = train_q2_model(X, Y, W, model_cls(**kw), sm, up, pn)
        metrics["config"] = name
        results.append(metrics)

    df = pd.DataFrame(results)
    df = df[["config","rmse","r2","mae","violation_rate"]]
    df.to_csv(os.path.join(OUTPUT_DIR, "q2_ablation.csv"), index=False, encoding="utf-8-sig")
    return df


# ==============================
# 主流程
# ==============================
def main():
    print("=" * 60)
    print("  step5.0 — 跨题消融实验汇总")
    print("=" * 60)

    # ---- Q1 消融 ----
    print("\n[Q1] 特征层级消融")
    X = np.load(OUT_X_ALL).astype(np.float64)
    y = np.load(OUT_Y_ALL).astype(np.float64)
    feature_names = list(np.load(OUT_FEATURE_NAMES, allow_pickle=True))
    print(f"  X={X.shape}, y={y.shape}")

    df_q1 = run_q1_ablation(X, y, feature_names)
    print(f"\n{'='*75}")
    print(f"  Q1 消融 ({N_SPLITS}-fold, XGBoost)")
    print(f"{'='*75}")
    best_r2 = df_q1["R2_mean"].max()
    for _, row in df_q1.iterrows():
        marker = " ★" if row["R2_mean"] == best_r2 else ""
        print(f"  {row['消融配置']:<20s} {row['特征数']:>5d} "
              f"RMSE={row['RMSE_mean']:.4f} R2={row['R2_mean']:.4f}{marker}")
    df_q1.to_csv(os.path.join(OUTPUT_DIR, "q1_ablation_results.csv"),
                 index=False, encoding="utf-8-sig")
    print("[step5.0] q1_ablation_results.csv 已保存")

    # ---- Q2 消融 ----
    print(f"\n{'='*60}")
    print("  [Q2] 模型消融")
    df_q2 = run_q2_ablation()
    if df_q2 is not None:
        print(f"\n{'='*75}")
        print(f"  Q2 消融 ({N_SPLITS_Q2}-fold)")
        print(f"{'='*75}")
        best_r2_q2 = df_q2["r2"].max()
        for _, row in df_q2.iterrows():
            marker = " ★" if row["r2"] == best_r2_q2 else ""
            print(f"  {row['config']:<45s} RMSE={row['rmse']:.4f} R²={row['r2']:.4f} "
                  f"违反率={row['violation_rate']:.3f}{marker}")

        # PINN vs 无PINN
        row_p = df_q2[df_q2["config"].str.contains("完整模型")]
        row_np = df_q2[df_q2["config"] == "无PINN"]
        if len(row_p) and len(row_np):
            d_rmse = row_np["rmse"].values[0] - row_p["rmse"].values[0]
            d_r2 = row_p["r2"].values[0] - row_np["r2"].values[0]
            print(f"\n  PINN vs 无PINN: ΔRMSE={d_rmse:.4f} ΔR²={d_r2:.4f}")

        # TE分季 vs 不分季
        row_tns = df_q2[df_q2["config"] == "时滞=TE不分段"]
        if len(row_p) and len(row_tns):
            d_rmse_te = row_tns["rmse"].values[0] - row_p["rmse"].values[0]
            d_r2_te = row_p["r2"].values[0] - row_tns["r2"].values[0]
            print(f"  TE分季 vs 不分季: ΔRMSE={d_rmse_te:.4f} ΔR²={d_r2_te:.4f}")

        # MIC vs TE vs CCF
        print(f"\n  时滞方法对比:")
        for cfg in ["时滞=纯MIC","时滞=TE不分段","时滞=CCF","完整模型(TCN+物理Loss+注意力+TE+PINN)"]:
            r = df_q2[df_q2["config"] == cfg]
            if len(r):
                print(f"    {cfg}: RMSE={r['rmse'].values[0]:.4f} R²={r['r2'].values[0]:.4f}")

        # 深度学习 vs 基准
        baseline_file = os.path.join(OUTPUT_DIR, "q2_baseline_comparison.csv")
        if os.path.exists(baseline_file):
            df_b = pd.read_csv(baseline_file)
            print(f"\n  深度学习 vs 基准模型:")
            row_dl = df_q2[df_q2["config"].str.contains("完整模型")]
            if len(row_dl):
                print(f"    完整模型(TCN): RMSE={row_dl['rmse'].values[0]:.4f} R²={row_dl['r2'].values[0]:.4f}")
            for _, brow in df_b.iterrows():
                n = brow.get("模型", brow.get("model_name", "?"))
                r = brow.get("RMSE", brow.get("rmse", "-"))
                r2 = brow.get("R2", brow.get("r2", "-"))
                print(f"    {n}: RMSE={r} R²={r2}")

        # 消融柱状图
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        configs = [r["config"] for r in df_q2.to_dict("records")]
        rmses = [r["rmse"] for r in df_q2.to_dict("records")]
        r2s = [r["r2"] for r in df_q2.to_dict("records")]
        colors_rmse = ["darkorange" if "完整" in c else "steelblue" for c in configs]
        axes[0].barh(range(len(configs)), rmses, color=colors_rmse, alpha=0.85)
        axes[0].set_yticks(range(len(configs)))
        axes[0].set_yticklabels(configs, fontsize=8)
        axes[0].set_xlabel("RMSE")
        axes[0].invert_yaxis()
        colors_r2 = ["darkorange" if "完整" in c else "steelblue" for c in configs]
        axes[1].barh(range(len(configs)), r2s, color=colors_r2, alpha=0.85)
        axes[1].set_yticks(range(len(configs)))
        axes[1].set_yticklabels(configs, fontsize=8)
        axes[1].set_xlabel("R²")
        axes[1].invert_yaxis()
        plt.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "q2_ablation_bar.png"), dpi=150)
        plt.close()
        print("[step5.0] figures/q2_ablation_bar.png 已保存")

    print(f"\n[step5.0] 完成.")


if __name__ == "__main__":
    main()
