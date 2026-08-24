"""
step3.9_learnable_routing.py — Q3: NN-learned blending weights
================================================================
Replaces four-layer if-else routing with NN-learned continuous blending.

Architecture:
  Physics:  persist_pred = NTU(5am)                  (constant anchor)
            cstr_pred    = CSTR chain (tier-A beta)   (transfer function)
  ML:       weights = softmax(NN(features_at_5am))    (7 step x 2 signals)
            pred[h] = w_persist * persist + w_cstr * cstr_pred[h]

Baselines:
  - Original four-layer if-else routing  (R2=0.617 CV Oracle)
  - Pure persistence                     (R2 negative)
  - Pure CSTR chain                      (no routing)
  - NN-learned blending                  (this experiment)

Output: results/step3.9_routing_verification.json
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
C_TH, ALPHA_B, GAMMA_W = 1.0, 0.34, 0.25
N_SPLITS = 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CLEAN_CSV = os.path.join(BASE_DIR, "output", "clean_data.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════
def load_data():
    import pandas as pd
    df = pd.read_csv(CLEAN_CSV)
    n = len(df)
    doy = pd.to_datetime(df["DATE"]).dt.dayofyear.values
    hour_raw = df["TIME"].values.astype(np.float64)
    hour = np.clip((hour_raw / 100).astype(int) % 24, 0, 23)
    return {
        "FILT": df["FILT_NTU"].values.astype(float),
        "NTU": df["NTU"].values.astype(float),
        "CW": df["CW_WELL_LEVEL"].values.astype(float),
        "Q": df["TW_FLOW"].values.astype(float),
        "RL": np.nan_to_num(df["RIVER_LEVEL"].values.astype(float), nan=8.0),
        "doy": doy, "hour": hour, "n": n,
    }


# ═══════════════════════════════════════════════════════════════
# Physics: CSTR Chain (tier-A beta, fixed)
# ═══════════════════════════════════════════════════════════════
def cstr_step(prev_ntu, ft, cw_prev, q_prev, rl_curr):
    if ft <= T1_THR:
        A0 = A_T1
    elif ft <= T2_THR:
        A0 = A_T2
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


# ═══════════════════════════════════════════════════════════════
# Features at prediction time (5:00)
# ═══════════════════════════════════════════════════════════════
def make_features(data):
    """Per-day features at 5:00 (ds+2). Returns (n_days, n_feat) tensor."""
    n = data["n"]
    all_days = list(range(6, n - 30, 12))
    feats = []
    for ds in all_days:
        if ds + 12 > n:
            continue
        f5 = data["FILT"][ds + 2]
        n5 = data["NTU"][ds + 2]
        n1 = data["NTU"][ds]
        cw5 = data["CW"][ds + 2]
        q5 = data["Q"][ds + 2]
        rl5 = data["RL"][ds + 2]
        log_hq = np.log(max(cw5, 0.1) / max(q5, 1.0))
        month = (data["doy"][ds] - 1) / 365.0
        month_sin = np.sin(2 * np.pi * month)
        month_cos = np.cos(2 * np.pi * month)
        hour = data["hour"][ds + 2]
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        feats.append([f5, n5, n1, n5 - n1, cw5, q5, rl5, log_hq,
                       month_sin, month_cos, hour_sin, hour_cos])
    feats = np.array(feats, dtype=np.float32)
    # Normalize
    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True) + 1e-6
    feats_norm = (feats - mean) / std
    return torch.tensor(feats_norm, dtype=torch.float32, device=DEVICE), mean, std


# ═══════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════
class BlendRouter(nn.Module):
    """NN learns per-step blending weights between CSTR and persistence.

    Input:  features at 5am (batch, n_feat)
    Output: weights (batch, 7, 2) — [w_persist, w_cstr] for steps 0..6
    """

    def __init__(self, n_feat, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 14),  # 7 steps * 2 weights
        )

    def forward(self, x):
        raw = self.net(x).view(-1, 7, 2)
        return torch.softmax(raw, dim=-1)  # sum to 1 per step


class BlendRouter3Way(nn.Module):
    """NN learns per-step 3-way blending: [persist, CSTR, blend].

    blend = alpha * CSTR + (1-alpha) * persist (like original Type B)
    """

    def __init__(self, n_feat, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 21),  # 7 steps * 3 weights
        )

    def forward(self, x):
        raw = self.net(x).view(-1, 7, 3)
        return torch.softmax(raw, dim=-1)


# ═══════════════════════════════════════════════════════════════
# Day Data Builder
# ═══════════════════════════════════════════════════════════════
def build_day_tensors(data):
    """Precompute CSTR chain + targets for all valid days. Returns tensors."""
    n = data["n"]
    all_days = list(range(6, n - 30, 12))
    cstr_chains = []
    ntu_5am_vals = []
    targets_all = []

    for ds in all_days:
        if ds + 12 > n:
            continue
        cs = build_cstr_chain(data, ds)
        cstr_chains.append(cs)  # (7,)
        ntu_5am_vals.append(data["NTU"][ds + 2])
        targets_all.append([data["NTU"][ds + 3 + i] for i in range(7)])

    cstr_chains = np.array(cstr_chains, dtype=np.float32)       # (n_days, 7)
    ntu_5am_vals = np.array(ntu_5am_vals, dtype=np.float32)     # (n_days,)
    targets_all = np.array(targets_all, dtype=np.float32)        # (n_days, 7)

    return (torch.tensor(cstr_chains, device=DEVICE),
            torch.tensor(ntu_5am_vals, device=DEVICE),
            torch.tensor(targets_all, device=DEVICE))


# ═══════════════════════════════════════════════════════════════
# Original Four-Layer Routing (baseline)
# ═══════════════════════════════════════════════════════════════
def classify_day(ft_5am, n1, n5):
    if ft_5am < 0.05 and abs(n5 - n1) < 0.02:
        return "A"
    elif ft_5am >= 0.15 or abs(n5 - n1) >= 0.05:
        return "C"
    return "B"

def four_layer_routing(cs_chain, n5, ft_5am, n1):
    dy = classify_day(ft_5am, n1, n5)
    preds = np.zeros(7)
    for i in range(7):
        if dy == "A":
            preds[i] = n5
        elif dy == "C":
            if ft_5am >= C_TH:
                preds[i] = np.clip(cs_chain[i], 0, None)
            else:
                preds[i] = np.clip(n5 + GAMMA_W * (cs_chain[i] - n5), 0, None)
        else:
            preds[i] = np.clip(ALPHA_B * cs_chain[i] + (1 - ALPHA_B) * n5, 0, None)
    return preds


# ═══════════════════════════════════════════════════════════════
# Training + Evaluation
# ═══════════════════════════════════════════════════════════════
def evaluate_baselines(data, day_indices):
    """Evaluate four-layer routing and pure persistence on given day indices."""
    n = data["n"]
    all_days = list(range(6, n - 30, 12))
    valid_days = [all_days[i] for i in day_indices]

    preds_routing = []
    preds_persist = []
    preds_cstr = []
    y_all = []

    for ds in valid_days:
        cs = build_cstr_chain(data, ds)
        n5 = data["NTU"][ds + 2]
        f5 = data["FILT"][ds + 2]
        n1 = data["NTU"][ds]
        targets = [data["NTU"][ds + 3 + i] for i in range(7)]

        preds_routing.extend(four_layer_routing(cs, n5, f5, n1))
        preds_persist.extend([n5] * 7)
        preds_cstr.extend(np.clip(cs, 0, None))
        y_all.extend(targets)

    r2_routing = r2_score(y_all, preds_routing)
    r2_persist = r2_score(y_all, preds_persist)
    r2_cstr = r2_score(y_all, preds_cstr)
    return r2_routing, r2_persist, r2_cstr


def train_nn_routing(model, feat_tensor, cstr_t, n5_t, targets_t, tr_idx, epochs=3000):
    """Train NN blending model. tr_idx indexes into day list."""
    model = model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=0.003)

    X_tr = feat_tensor[tr_idx]           # (n_train, n_feat)
    cs_tr = cstr_t[tr_idx]               # (n_train, 7)
    n5_tr = n5_t[tr_idx]                 # (n_train,)
    y_tr = targets_t[tr_idx]             # (n_train, 7)

    for epoch in range(epochs):
        model.train()
        weights = model(X_tr)            # (n_train, 7, 2)
        w_persist = weights[:, :, 0]     # (n_train, 7)
        w_cstr = weights[:, :, 1]        # (n_train, 7)
        persist_pred = n5_tr.unsqueeze(-1).expand(-1, 7)  # (n_train, 7)
        pred = w_persist * persist_pred + w_cstr * cs_tr
        loss = nn.MSELoss()(pred, y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        weights_va = model(feat_tensor)
        w_p_va = weights_va[:, :, 0]
        w_c_va = weights_va[:, :, 1]
        persist_va = n5_t.unsqueeze(-1).expand(-1, 7)
        pred_va = w_p_va * persist_va + w_c_va * cstr_t
    return pred_va, y_tr if tr_idx is None else targets_t


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    print(f"Device: {DEVICE}")
    print("=" * 65)
    print("  Q3: NN-Blended Routing — Verification")
    print("=" * 65)

    data = load_data()
    n = data["n"]
    all_days_list = list(range(6, n - 30, 12))
    n_days = len(all_days_list)
    print(f"  Days: {n_days}, Samples: {n}")

    feat_tensor, f_mean, f_std = make_features(data)
    cstr_t, n5_t, targets_t = build_day_tensors(data)
    n_feat = feat_tensor.shape[1]
    print(f"  Features: {n_feat}, Days: {cstr_t.shape[0]}, "
          f"Targets: {targets_t.shape}")

    # ── TS-CV: 5-fold day-level ────────────────────────────────
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    day_range = np.arange(n_days)

    fold_r2_routing, fold_r2_persist, fold_r2_cstr = [], [], []
    fold_r2_nn2, fold_r2_nn3 = [], []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(day_range)):
        n_tr = len(tr_idx)
        print(f"\n{'─'*55}")
        print(f"  Fold {fold}: train_days={n_tr} val_days={len(va_idx)}")

        # Baselines
        r2_route, r2_pers, r2_cs = evaluate_baselines(data, va_idx)
        fold_r2_routing.append(r2_route)
        fold_r2_persist.append(r2_pers)
        fold_r2_cstr.append(r2_cs)
        print(f"  Routing={r2_route:.4f}  Persist={r2_pers:.4f}  CSTR={r2_cs:.4f}")

        # Adaptive capacity: fewer params when data is scarce
        hidden = 8 if n_tr < 100 else 16
        epochs = 2000 if n_tr < 100 else 3000
        if n_tr < 80:
            epochs = 4000
        print(f"  NN config: hidden={hidden}, epochs={epochs}")

        # NN 2-way blending
        model2 = BlendRouter(n_feat, hidden)
        pred_va, _ = train_nn_routing(model2, feat_tensor, cstr_t, n5_t, targets_t,
                                       tr_idx, epochs=epochs)
        y_va = targets_t[va_idx].cpu().numpy()
        p2_va = pred_va[va_idx].cpu().numpy()
        r2_2 = r2_score(y_va.ravel(), p2_va.ravel())
        fold_r2_nn2.append(r2_2)

        # NN 3-way blending
        model3 = BlendRouter3Way(n_feat, hidden)
        pred3_va, _ = train_nn_routing(model3, feat_tensor, cstr_t, n5_t, targets_t,
                                        tr_idx, epochs=epochs)
        p3_va = pred3_va[va_idx].cpu().numpy()
        r2_3 = r2_score(y_va.ravel(), p3_va.ravel())
        fold_r2_nn3.append(r2_3)

        print(f"  NN-2way={r2_2:.4f}  NN-3way={r2_3:.4f}")

    # ── Summary ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Method':<32} {'R2 CV':>7}  {'+-std':>7}")
    print(f"  {'─'*50}")
    print(f"  {'Four-layer if-else (original)':<32} {np.mean(fold_r2_routing):7.4f}  {np.std(fold_r2_routing):7.4f}")
    print(f"  {'Pure CSTR chain (no routing)':<32} {np.mean(fold_r2_cstr):7.4f}  {np.std(fold_r2_cstr):7.4f}")
    print(f"  {'Pure persistence':<32} {np.mean(fold_r2_persist):7.4f}  {np.std(fold_r2_persist):7.4f}")
    print(f"  {'  '}")
    print(f"  {'NN 2-way (persist+CSTR blend)':<32} {np.mean(fold_r2_nn2):7.4f}  {np.std(fold_r2_nn2):7.4f}")
    print(f"  {'NN 3-way (persist+CSTR+blend)':<32} {np.mean(fold_r2_nn3):7.4f}  {np.std(fold_r2_nn3):7.4f}")

    best_nn = max(np.mean(fold_r2_nn2), np.mean(fold_r2_nn3))
    best_routing = np.mean(fold_r2_routing)
    print(f"\n  Delta (NN best - if-else): {best_nn - best_routing:+.4f}")
    if best_nn > best_routing:
        print(f"  *** NN routing BEATS if-else routing! ***")
    else:
        print(f"  NN routing does NOT beat if-else routing.")

    # Save
    result = {
        "model": "Q3 NN-Blended Routing",
        "cv_results": {
            "routing_ifelse": {"r2": round(float(np.mean(fold_r2_routing)), 4),
                                "std": round(float(np.std(fold_r2_routing)), 4),
                                "folds": [round(float(f), 4) for f in fold_r2_routing]},
            "cstr_only": {"r2": round(float(np.mean(fold_r2_cstr)), 4),
                           "std": round(float(np.std(fold_r2_cstr)), 4),
                           "folds": [round(float(f), 4) for f in fold_r2_cstr]},
            "persist_only": {"r2": round(float(np.mean(fold_r2_persist)), 4),
                              "std": round(float(np.std(fold_r2_persist)), 4),
                              "folds": [round(float(f), 4) for f in fold_r2_persist]},
            "nn_2way": {"r2": round(float(np.mean(fold_r2_nn2)), 4),
                         "std": round(float(np.std(fold_r2_nn2)), 4),
                         "folds": [round(float(f), 4) for f in fold_r2_nn2]},
            "nn_3way": {"r2": round(float(np.mean(fold_r2_nn3)), 4),
                         "std": round(float(np.std(fold_r2_nn3)), 4),
                         "folds": [round(float(f), 4) for f in fold_r2_nn3]},
        },
        "verdict": "NN_BETTER" if best_nn > best_routing else "ROUTING_BETTER",
    }
    out_path = os.path.join(RESULTS_DIR, "step3.9_routing_verification.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
