"""
step3.9_diagnostics.py — Q3 Routing: Formula & Loss Diagnostics
=================================================================
Diagnostic experiments:
  1. CSTR chain per-step residual bias (mean, std, skew)
  2. Feature-error correlation at prediction time (5:00)
  3. Residual learning: pred = CSTR + NN_correction(features, step)
  4. Log-space loss: MSE(log(NTU+eps), log(pred+eps))
  5. Huber loss: less sensitive to high-NTU outliers

Models tested:
  A. Blend:      pred = w*persist + (1-w)*cstr  (current, baseline)
  B. Residual:   pred = cstr + delta_nn          (new)
  C. Residual+Log: pred = exp(log_cstr + delta_nn_log) (new)

Output: results/step3.9_diagnostics.json
"""

import os, sys, json, warnings
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
warnings.filterwarnings("ignore")
torch.manual_seed(42)

EPS = 1e-3
T1_THR, T2_THR = 0.05, 0.15
A_T1, A_T2, A_SAME, A_DIFF = 400, 250, 100, 20
RL_MED, Q_MED = 8.0, 48
N_SPLITS = 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CLEAN_CSV = os.path.join(BASE_DIR, "output", "clean_data.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    import pandas as pd
    df = pd.read_csv(CLEAN_CSV)
    n = len(df)
    doy = pd.to_datetime(df["DATE"]).dt.dayofyear.values
    hraw = df["TIME"].values.astype(np.float64)
    hour = np.clip((hraw / 100).astype(int) % 24, 0, 23)
    return {"FILT": df["FILT_NTU"].values.astype(float),
            "NTU": df["NTU"].values.astype(float),
            "CW": df["CW_WELL_LEVEL"].values.astype(float),
            "Q": df["TW_FLOW"].values.astype(float),
            "RL": np.nan_to_num(df["RIVER_LEVEL"].values.astype(float), nan=8.0),
            "doy": doy, "hour": hour, "n": n}


def cstr_step(prev_ntu, ft, cw_prev, q_prev, rl_curr):
    if ft <= T1_THR:        A0 = A_T1
    elif ft <= T2_THR:       A0 = A_T2
    else:
        bal = 1 if (rl_curr - RL_MED) * (q_prev - Q_MED) > 0 else 0
        A0 = A_SAME if bal else A_DIFF
    theta = max(A0 * max(cw_prev, 0.1) / max(q_prev, 1.0), 0.02)
    beta = np.clip(np.exp(-2.0 / theta), 0.001, 0.999)
    return beta * prev_ntu + (1.0 - beta) * ft


def build_cstr_chain(data, day_start):
    cs = np.zeros(7)
    cs[0] = data["NTU"][day_start]
    for i in range(1, 7):
        ft = data["FILT"][day_start + 3 + i]
        cw_prev = data["CW"][day_start + 2 + i]
        q_prev = data["Q"][day_start + 3 + i]
        rl_curr = data["RL"][day_start + 3 + i]
        cs[i] = cstr_step(cs[i - 1], ft, cw_prev, q_prev, rl_curr)
    return cs


def extract_features(data, day_start):
    """Features KNOWN at 5:00 prediction time."""
    f5 = data["FILT"][day_start + 2]
    n5 = data["NTU"][day_start + 2]
    n1 = data["NTU"][day_start]
    cw5 = data["CW"][day_start + 2]
    q5 = data["Q"][day_start + 2]
    rl5 = data["RL"][day_start + 2]
    log_hq = np.log(max(cw5, 0.1) / max(q5, 1.0))
    month = (data["doy"][day_start] - 1) / 365.0

    # Also: overnight trend, FILT delta
    f1 = data["FILT"][day_start]
    dfilt = f5 - f1
    dntu = n5 - n1

    return np.array([f5, n5, n1, dntu, dfilt, cw5, q5, rl5, log_hq,
                      np.sin(2 * np.pi * month), np.cos(2 * np.pi * month)], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
# Diagnostic 1: CSTR chain per-step residual
# ═══════════════════════════════════════════════════════════════
def diagnostic_residuals(data):
    print("\n" + "=" * 65)
    print("  DIAGNOSTIC 1: CSTR Chain Per-Step Residual")
    print("=" * 65)
    n = data["n"]
    all_days = list(range(6, n - 30, 12))
    labels = ["07:00", "09:00", "11:00", "13:00", "15:00", "17:00", "19:00"]
    all_res = {i: [] for i in range(7)}

    for ds in all_days:
        if ds + 12 > n: continue
        cs = build_cstr_chain(data, ds)
        for i in range(7):
            true = data["NTU"][ds + 3 + i]
            all_res[i].append(cs[i] - true)

    print(f"  {'Step':<8} {'Mean err':>10} {'Std err':>10} {'RMSE':>10} "
          f"{'Skew':>8} {'Q95 err':>10}")
    print(f"  {'─'*58}")
    for i in range(7):
        r = np.array(all_res[i])
        print(f"  {labels[i]:<8} {r.mean():10.4f} {r.std():10.4f} "
              f"{np.sqrt(np.mean(r**2)):10.4f} {float(np.mean(r**3)/max(r.std()**3, 1e-6)):8.2f} "
              f"{np.percentile(np.abs(r), 95):10.4f}")

    # Overall bias decomposition
    all_r = np.concatenate([np.array(all_res[i]) for i in range(7)])
    bias_total = np.mean(all_r) ** 2
    var_total = np.var(all_r)
    print(f"\n  MSE decomposition: bias2={bias_total:.6f}  var={var_total:.6f}  "
          f"RMSE={np.sqrt(bias_total + var_total):.4f}")
    print(f"  Bias share: {bias_total/(bias_total+var_total)*100:.1f}%")

    # By FILT tier
    for tier_name, lo, hi in [("T1", 0, 0.05), ("T2", 0.05, 0.15), ("T3", 0.15, 999)]:
        mask = (data["FILT"] >= lo) & ((data["FILT"] < hi) if hi < 999 else (data["FILT"] >= lo))
        valid = all_days.copy()  # simplified — compute per-day tier
        tier_res = []
        for ds in all_days:
            if ds + 12 > n: continue
            ft = data["FILT"][ds + 2]
            if lo <= ft < hi or (hi >= 999 and ft >= lo):
                cs = build_cstr_chain(data, ds)
                for i in range(7):
                    tier_res.append(cs[i] - data["NTU"][ds + 3 + i])
        if tier_res:
            tr = np.array(tier_res)
            print(f"  {tier_name}: n={len(tr)}  mean_err={tr.mean():.4f}  rmse={np.sqrt(np.mean(tr**2)):.4f}")

    return all_res


# ═══════════════════════════════════════════════════════════════
# Diagnostic 2: Feature-error correlation
# ═══════════════════════════════════════════════════════════════
def diagnostic_feature_correlation(data):
    print("\n" + "=" * 65)
    print("  DIAGNOSTIC 2: 5:00 Features vs CSTR Error Correlation")
    print("=" * 65)
    fnames = ["FILT_5am", "NTU_5am", "NTU_1am", "dNTU", "dFILT",
              "CW_5am", "Q_5am", "RL_5am", "log_HQ", "m_sin", "m_cos"]
    n = data["n"]
    all_days = list(range(6, n - 30, 12))

    feat_list = []
    err_list = []
    for ds in all_days:
        if ds + 12 > n: continue
        feats = extract_features(data, ds)
        cs = build_cstr_chain(data, ds)
        for i in range(7):
            err = cs[i] - data["NTU"][ds + 3 + i]
            feat_list.append(feats)
            err_list.append(err)

    F = np.array(feat_list)
    E = np.array(err_list)
    print(f"  Samples: {len(E)}")

    for j, name in enumerate(fnames):
        corr = np.corrcoef(F[:, j], E)[0, 1]
        bar = "|" * max(0, min(20, int(abs(corr) * 40))) + ("+" if corr > 0 else "-")
        print(f"  {name:<14}  corr={corr:+.4f}  {bar}")

    # R2 if we only had 5:00 features to predict CSTR error?
    from sklearn.linear_model import RidgeCV
    m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]).fit(F, E)
    pred_e = m.predict(F)
    r2_e = 1 - np.sum((E - pred_e)**2) / np.sum((E - E.mean())**2)
    print(f"\n  Ridge on 5:00 features -> CSTR error R2 = {r2_e:.4f}")
    print(f"  [If close to 0: NN has nothing to learn from 5:00 features]")


# ═══════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════
class BlendModel(nn.Module):
    """Model A: Blend weights (current approach)"""
    def __init__(self, n_feat, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 14))
    def forward(self, x):
        return torch.softmax(self.net(x).view(-1, 7, 2), dim=-1)


class ResidualModel(nn.Module):
    """Model B: Learn correction delta to CSTR prediction.
       pred = clip(CSTR + delta, 0, inf)
       delta = NN(features, step_idx)
    """
    def __init__(self, n_feat, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 7))
    def forward(self, x):
        bs = x.shape[0]
        deltas = torch.zeros(bs, 7, device=x.device)
        for i in range(7):
            step_feat = torch.full((bs, 1), i / 6.0, device=x.device)
            xi = torch.cat([x, step_feat], dim=-1)
            deltas[:, i] = self.net(xi)[:, i]
        return deltas


class LogResidualModel(nn.Module):
    """Model C: Correction in log-space.
       log_pred = log(CSTR+eps) + delta_log
       pred = exp(log_pred) - eps
    """
    def __init__(self, n_feat, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 7))
    def forward(self, x):
        bs = x.shape[0]
        deltas = torch.zeros(bs, 7, device=x.device)
        for i in range(7):
            step_feat = torch.full((bs, 1), i / 6.0, device=x.device)
            xi = torch.cat([x, step_feat], dim=-1)
            deltas[:, i] = self.net(xi)[:, i]
        return torch.clamp(deltas, -2, 2)


# ═══════════════════════════════════════════════════════════════
# Data builder
# ═══════════════════════════════════════════════════════════════
def build_dataset(data):
    n = data["n"]
    all_days = list(range(6, n - 30, 12))
    feats, cstr_chains, n5_vals, targets = [], [], [], []
    for ds in all_days:
        if ds + 12 > n: continue
        feats.append(extract_features(data, ds))
        cs = build_cstr_chain(data, ds)
        cstr_chains.append(cs)
        n5_vals.append(data["NTU"][ds + 2])
        targets.append([data["NTU"][ds + 3 + i] for i in range(7)])
    F = np.array(feats, dtype=np.float32)
    # Normalize
    F_mean, F_std = F.mean(axis=0), F.std(axis=0) + 1e-6
    F = (F - F_mean) / F_std
    return (torch.tensor(F, device=DEVICE),
            torch.tensor(np.array(cstr_chains, dtype=np.float32), device=DEVICE),
            torch.tensor(np.array(n5_vals, dtype=np.float32), device=DEVICE),
            torch.tensor(np.array(targets, dtype=np.float32), device=DEVICE),
            F_mean, F_std)


# ═══════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════
def train_model(model_class, feat_t, cstr_t, n5_t, y_t, tr_idx, va_idx,
                n_feat, hidden, loss_type="mse", epochs=3000, lr=0.003):
    model = model_class(n_feat, hidden).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    X_tr, cs_tr, n5_tr, y_tr = feat_t[tr_idx], cstr_t[tr_idx], n5_t[tr_idx], y_t[tr_idx]
    bs = X_tr.shape[0]

    eps_log = 0.005  # for log-space stability
    for ep in range(epochs):
        model.train()
        if isinstance(model, BlendModel):
            w = model(X_tr)  # (bs, 7, 2)
            pred = w[:, :, 0] * n5_tr.unsqueeze(-1) + w[:, :, 1] * cs_tr
        elif isinstance(model, ResidualModel):
            delta = model(X_tr)
            pred = torch.clamp(cs_tr + delta, 0, None)
        else:  # LogResidualModel
            delta_log = model(X_tr)
            log_cs = torch.log(torch.clamp(cs_tr, min=eps_log))
            log_pred = log_cs + delta_log
            pred = torch.exp(torch.clamp(log_pred, max=5)) - eps_log

        if loss_type == "mse":
            loss = nn.MSELoss()(pred, y_tr)
        elif loss_type == "huber":
            loss = nn.SmoothL1Loss()(pred, y_tr)
        elif loss_type == "log":
            y_log = torch.log(torch.clamp(y_tr, min=eps_log))
            p_log = torch.log(torch.clamp(pred, min=eps_log))
            loss = nn.MSELoss()(y_log, p_log) * 10
        else:
            loss = nn.MSELoss()(pred, y_tr)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    with torch.no_grad():
        if isinstance(model, BlendModel):
            w_all = model(feat_t)
            p_all = w_all[:, :, 0] * n5_t.unsqueeze(-1) + w_all[:, :, 1] * cstr_t
        elif isinstance(model, ResidualModel):
            p_all = torch.clamp(cstr_t + model(feat_t), 0, None)
        else:
            delta_all = model(feat_t)
            log_cs_all = torch.log(torch.clamp(cstr_t, min=eps_log))
            p_all = torch.exp(torch.clamp(log_cs_all + delta_all, max=5)) - eps_log
    return p_all


def evaluate_routing(data, va_day_indices):
    """Original if-else routing on validation days."""
    n = data["n"]
    all_days = list(range(6, n - 30, 12))
    valid_days = [all_days[i] for i in va_day_indices]
    preds, ys = [], []
    for ds in valid_days:
        cs = build_cstr_chain(data, ds)
        n5 = data["NTU"][ds + 2]
        n1 = data["NTU"][ds]
        f5 = data["FILT"][ds + 2]
        if f5 < 0.05 and abs(n5 - n1) < 0.02:
            p = [n5] * 7
        elif f5 >= 0.15 or abs(n5 - n1) >= 0.05:
            if f5 >= 1.0:
                p = np.clip(cs, 0, None)
            else:
                p = np.clip(n5 + 0.25 * (cs - n5), 0, None)
        else:
            p = np.clip(0.34 * cs + 0.66 * n5, 0, None)
        preds.extend(p)
        ys.extend([data["NTU"][ds + 3 + i] for i in range(7)])
    return r2_score(ys, preds)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    print(f"Device: {DEVICE}")
    print("=" * 65)
    print("  Q3: Formula & Loss Diagnostics")
    print("=" * 65)

    data = load_data()
    n_days = len(list(range(6, data["n"] - 30, 12)))
    print(f"  Valid days: {n_days}")

    # ── Diagnostic 1: Residuals ───────────────────────────────
    diagnostic_residuals(data)

    # ── Diagnostic 2: Feature correlation ─────────────────────
    diagnostic_feature_correlation(data)

    # ── Build dataset ─────────────────────────────────────────
    feat_t, cstr_t, n5_t, y_t, fmean_arr, fstd_arr = build_dataset(data)
    n_feat = feat_t.shape[1]
    n_total = feat_t.shape[0]
    print(f"\n  Dataset: {n_total} days x {n_feat} features x 7 steps")

    # ── TS-CV: compare architectures ──────────────────────────
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    day_range = np.arange(n_total)

    results = {"routing": [], "cstr_only": [], "persist_only": [],
               "blend_mse": [], "residual_mse": [], "residual_huber": [], "residual_log": []}

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(day_range)):
        n_tr = len(tr_idx)
        hidden = 8 if n_tr < 100 else 16
        ep = 2500 if n_tr < 100 else 3000
        print(f"\n  Fold {fold}: train={n_tr} val={len(va_idx)} "
              f"[hidden={hidden}, epochs={ep}]")

        # Baselines
        r2_route = evaluate_routing(data, va_idx)
        results["routing"].append(r2_route)

        y_va = y_t[va_idx].cpu().numpy().ravel()
        cs_pure = cstr_t[va_idx].cpu().numpy().ravel()
        ps_pure = np.repeat(n5_t[va_idx].cpu().numpy(), 7)
        results["cstr_only"].append(r2_score(y_va, cs_pure))
        results["persist_only"].append(r2_score(y_va, ps_pure))

        # A: Blend + MSE
        pA = train_model(BlendModel, feat_t, cstr_t, n5_t, y_t, tr_idx, va_idx,
                         n_feat, hidden, "mse", ep)
        r2A = r2_score(y_va, pA[va_idx].cpu().numpy().ravel())
        results["blend_mse"].append(r2A)

        # B: Residual + MSE
        pB = train_model(ResidualModel, feat_t, cstr_t, n5_t, y_t, tr_idx, va_idx,
                         n_feat, hidden, "mse", ep)
        r2B = r2_score(y_va, pB[va_idx].cpu().numpy().ravel())
        results["residual_mse"].append(r2B)

        # C: Residual + Huber
        pC = train_model(ResidualModel, feat_t, cstr_t, n5_t, y_t, tr_idx, va_idx,
                         n_feat, hidden, "huber", ep)
        r2C = r2_score(y_va, pC[va_idx].cpu().numpy().ravel())
        results["residual_huber"].append(r2C)

        # D: LogResidual + log loss
        pD = train_model(LogResidualModel, feat_t, cstr_t, n5_t, y_t, tr_idx, va_idx,
                         n_feat, hidden, "log", ep)
        r2D = r2_score(y_va, pD[va_idx].cpu().numpy().ravel())
        results["residual_log"].append(r2D)

        print(f"  routing={r2_route:.4f}  cstr={r2_score(y_va, cs_pure):.4f}  "
              f"persist={r2_score(y_va, ps_pure):.4f}")
        print(f"  blendMSE={r2A:.4f}  residMSE={r2B:.4f}  "
              f"residHuber={r2C:.4f}  residLog={r2D:.4f}")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  FINAL COMPARISON (CV mean R2)")
    print(f"{'='*65}")
    for key, vals in results.items():
        print(f"  {key:<20}  {np.mean(vals):7.4f} +- {np.std(vals):7.4f}  "
              f"folds=[{', '.join(f'{v:.3f}' for v in vals)}]")

    best_key = max(results, key=lambda k: np.mean(results[k]))
    print(f"\n  Best: {best_key} (R2={np.mean(results[best_key]):.4f})")
    print(f"  vs routing: {np.mean(results[best_key]) - np.mean(results['routing']):+.4f}")

    save = {k: {"mean": round(float(np.mean(v)), 4),
                 "std": round(float(np.std(v)), 4),
                 "folds": [round(float(f), 4) for f in v]}
            for k, v in results.items()}
    save["best"] = best_key
    out = os.path.join(RESULTS_DIR, "step3.9_diagnostics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(save, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out}")


if __name__ == "__main__":
    main()
