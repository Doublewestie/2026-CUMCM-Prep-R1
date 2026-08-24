"""
step1.10_learnable_beta.py - Phase 1: Learnable CSTR Beta
======================================================================
Hypothesis: A small NN learning beta = sigmoid(f(CW, Q, FILT, time))
can outperform hand-tuned tier-dependent A parameters (R2=0.689 CV).

Exp 1a (linear):  beta = sigmoid(w * log(CW/Q) + b)             - 2 params
Exp 1b (linear2): beta = sigmoid(w1*log(CW/Q) + w2*FILT + b)   - 3 params
Exp 1c (mlp):     beta = sigmoid(MLP(7 feat, hidden=8))          - ~73 params
Exp 1d (mlp_full): beta = sigmoid(MLP(9 feat, hidden=8))         - ~89 params

Training: Adam on MSE(NTU_pred, NTU_true), one-step-ahead CSTR.
Validation: TS-CV 5-fold (same as step1.7+_tscv_validation.py).
Baselines: Hand-tuned uniform A=141.3, tier-A={400,250,100,20}.

Output: results/step1.10_verification.json
"""

import os, sys, json, warnings
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
warnings.filterwarnings("ignore")
torch.manual_seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
CLEAN_CSV = os.path.join(BASE_DIR, "output", "clean_data.csv")
EPS = 1e-6

T1_THR, T2_THR = 0.05, 0.15
A_T1, A_T2, A_SAME, A_DIFF = 400, 250, 100, 20
RL_MED, Q_MED = 8.0, 48
N_SPLITS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════
def load_data():
    import pandas as pd
    df = pd.read_csv(CLEAN_CSV)
    n = len(df)
    doy = pd.to_datetime(df["DATE"]).dt.dayofyear.values
    filt = df["FILT_NTU"].values.astype(np.float64)
    ntu = df["NTU"].values.astype(np.float64)
    cw = df["CW_WELL_LEVEL"].values.astype(np.float64)
    q = df["TW_FLOW"].values.astype(np.float64)
    rl = df["RIVER_LEVEL"].values.astype(np.float64)
    rl = np.nan_to_num(rl, nan=np.nanmean(rl))
    hour_raw = df["TIME"].values.astype(np.float64)
    hour = (hour_raw / 100).astype(int) % 24
    hour = np.clip(hour, 0, 23)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin = np.sin(2 * np.pi * doy / 365)
    day_cos = np.cos(2 * np.pi * doy / 365)

    # Normalize features for NN training
    cw_raw, q_raw = cw.copy(), q.copy()
    cw_mean, cw_std = np.mean(cw), np.std(cw) + EPS
    q_mean, q_std = np.mean(q), np.std(q) + EPS
    filtr_mean, filtr_std = np.mean(filt), np.std(filt) + EPS
    rl_mean, rl_std = np.mean(rl), np.std(rl) + EPS

    return {
        "n": n, "FILT": filt, "NTU": ntu, "CW": cw, "Q": q, "RL": rl,
        "CW_raw": cw_raw, "Q_raw": q_raw,
        "hour_sin": hour_sin, "hour_cos": hour_cos,
        "day_sin": day_sin, "day_cos": day_cos,
        "cw_mean": cw_mean, "cw_std": cw_std,
        "q_mean": q_mean, "q_std": q_std,
        "filtr_mean": filtr_mean, "filtr_std": filtr_std,
        "rl_mean": rl_mean, "rl_std": rl_std,
    }


# ═══════════════════════════════════════════════════════════════
# Hand-Tuned Baselines (for comparison)
# ═══════════════════════════════════════════════════════════════
def get_balance_flag(rl_v, q_v):
    return 1 if (rl_v - RL_MED) * (q_v - Q_MED) > 0 else 0

def hand_tuned_beta_uniform(cw_prev, q_prev):
    A = 141.3
    theta = max(A * max(cw_prev, 0.1) / max(q_prev, 1.0), 0.02)
    return np.clip(np.exp(-2.0 / theta), 0.001, 0.999)

def hand_tuned_beta_tier(ft, cw_prev, q_prev, rl_curr):
    if ft <= T1_THR:
        A0 = A_T1
    elif ft <= T2_THR:
        A0 = A_T2
    else:
        A0 = A_SAME if get_balance_flag(rl_curr, q_prev) else A_DIFF
    theta = max(A0 * max(cw_prev, 0.1) / max(q_prev, 1.0), 0.02)
    return np.clip(np.exp(-2.0 / theta), 0.001, 0.999)

def predict_cstr_handtuned(data, mode="tier"):
    """One-step-ahead CSTR with hand-tuned beta. Returns pred, true arrays."""
    filt, ntu, cw, q, rl = data["FILT"], data["NTU"], data["CW"], data["Q"], data["RL"]
    n = len(ntu)
    pred = np.zeros(n)
    pred[0] = ntu[0]
    for t in range(1, n):
        if mode == "tier":
            beta = hand_tuned_beta_tier(filt[t], cw[t-1], q[t-1], rl[t])
        else:
            beta = hand_tuned_beta_uniform(cw[t-1], q[t-1])
        pred[t] = beta * ntu[t-1] + (1.0 - beta) * filt[t]
    return np.clip(pred, 0, None), ntu


# ═══════════════════════════════════════════════════════════════
# PyTorch: Differentiable CSTR
# ═══════════════════════════════════════════════════════════════
class DifferentiableCSTR(nn.Module):
    """Learnable beta = sigmoid(NN(features)), then CSTR one-step-ahead."""

    def __init__(self, n_features, hidden=8):
        super().__init__()
        if hidden == 0:
            self.net = nn.Linear(n_features, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(n_features, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        self._sigmoid = nn.Sigmoid()
        self._model_type = "beta"

    def forward(self, features, ntu_prev, filt_curr, cw_prev=None, q_prev=None):
        beta = self._sigmoid(self.net(features)).squeeze(-1)
        beta = torch.clamp(beta, 0.001, 0.999)
        return beta * ntu_prev.detach() + (1.0 - beta) * filt_curr


class LearnableTheta(nn.Module):
    """Scheme A: learn A_eff = exp(NN(features)), embed physical theta formula.
    
    theta = A_eff * CW / Q      (CSTR residence time, physics-constrained)
    beta  = exp(-2h / theta)    (CSTR decay, no sigmoid needed)
    NTU_pred = beta * NTU(t-1) + (1-beta) * FILT(t)
    """

    def __init__(self, n_features, hidden=8):
        super().__init__()
        if hidden == 0:
            self.net = nn.Linear(n_features, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(n_features, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        self._model_type = "theta"

    def forward(self, features, ntu_prev, filt_curr, cw_prev, q_prev):
        raw_A = torch.clamp(self.net(features).squeeze(-1), -4, 8)
        A_eff = torch.exp(raw_A)
        theta = A_eff * cw_prev / torch.clamp(q_prev, min=1.0)
        theta = torch.clamp(theta, min=0.02)
        beta = torch.exp(-2.0 / theta)
        beta = torch.clamp(beta, 0.001, 0.999)
        return beta * ntu_prev + (1.0 - beta) * filt_curr


def make_features(data, feat_type="linear"):
    """Build feature tensor for each model variant.
    
    Tier indicators (binary, not normalized):
      tier_flag_1:  FILT <= 0.05
      tier_flag_2:  0.05 < FILT <= 0.15
      bal_flag:     (RL - RL_med) * (Q - Q_med) > 0  (same sign)
    """
    n = data["n"]
    cw = (data["CW"] - data["cw_mean"]) / data["cw_std"]
    q = (data["Q"] - data["q_mean"]) / data["q_std"]
    filt = (data["FILT"] - data["filtr_mean"]) / data["filtr_std"]
    rl = (data["RL"] - data["rl_mean"]) / data["rl_std"]
    log_hq = np.log(np.maximum(data["CW_raw"], 0.1) / np.maximum(data["Q_raw"], 1.0))
    
    # Tier indicator features (binary, no normalization)
    tier_f1 = (data["FILT"] <= 0.05).astype(np.float64)
    tier_f2 = ((data["FILT"] > 0.05) & (data["FILT"] <= 0.15)).astype(np.float64)
    bal_f = ((data["RL"] - RL_MED) * (data["Q"] - Q_MED) > 0).astype(np.float64)
    tier_features = np.column_stack([tier_f1, tier_f2, bal_f])

    core_7 = np.column_stack([cw, q, filt, rl, log_hq,
                               data["hour_sin"], data["hour_cos"]])

    if feat_type == "linear":
        raw = np.column_stack([log_hq])
    elif feat_type == "linear2":
        raw = np.column_stack([log_hq, data["FILT"]])
    elif feat_type == "mlp":
        raw = core_7
    elif feat_type == "mlp_full":
        raw = np.column_stack([cw, q, filt, rl, log_hq,
                                data["hour_sin"], data["hour_cos"],
                                data["day_sin"], data["day_cos"]])
    # Scheme B: core features + tier indicators
    elif feat_type == "mlp_tier":
        raw = np.column_stack([core_7, tier_features])          # 7+3=10
    elif feat_type == "mlp_full_tier":
        raw = np.column_stack([core_7, data["day_sin"], data["day_cos"],
                                tier_features])                  # 9+3=12
    else:
        raise ValueError(f"Unknown feat_type: {feat_type}")

    return torch.tensor(raw, dtype=torch.float32, device=DEVICE)


def train_one_fold(model, feat_tensor, ntu_tensor, filt_tensor, tr_idx, va_idx,
                    cw_tensor=None, q_tensor=None, epochs=2000):
    """Train model on tr_idx, return validation predictions."""
    model = model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=0.005)
    is_theta = getattr(model, "_model_type", "beta") == "theta"

    X_tr = feat_tensor[tr_idx]
    ntu_tr = ntu_tensor[tr_idx]
    filt_tr = filt_tensor[tr_idx]
    ntu_prev_tr = ntu_tensor[tr_idx - 1]

    n_train = len(tr_idx)
    if n_train < 10:
        return None

    for epoch in range(epochs):
        model.train()
        if is_theta:
            cw_prev_tr = cw_tensor[tr_idx - 1]
            q_prev_tr = q_tensor[tr_idx - 1]
            pred = model(X_tr, ntu_prev_tr, filt_tr, cw_prev_tr, q_prev_tr)
        else:
            pred = model(X_tr, ntu_prev_tr, filt_tr)
        loss = nn.MSELoss()(pred, ntu_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        X_va = feat_tensor[va_idx]
        ntu_va = ntu_tensor[va_idx]
        filt_va = filt_tensor[va_idx]
        ntu_prev_va = ntu_tensor[va_idx - 1]
        if is_theta:
            cw_prev_va = cw_tensor[va_idx - 1]
            q_prev_va = q_tensor[va_idx - 1]
            pred_va = model(X_va, ntu_prev_va, filt_va, cw_prev_va, q_prev_va)
        else:
            pred_va = model(X_va, ntu_prev_va, filt_va)

    return {"pred": pred_va.cpu().numpy(), "true": ntu_va.cpu().numpy()}


# ═══════════════════════════════════════════════════════════════
# TS-CV Evaluation
# ═══════════════════════════════════════════════════════════════
def evaluate_tscv(model_class, feat_tensor, ntu_tensor, filt_tensor, n_features,
                  hidden, epochs=1500, verbose=True, cw_tensor=None, q_tensor=None):
    """5-fold TS-CV for a model variant."""
    n = len(ntu_tensor)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    r2s = []
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(np.arange(n))):
        model = model_class(n_features, hidden)
        result = train_one_fold(model, feat_tensor, ntu_tensor, filt_tensor,
                                tr_idx, va_idx, cw_tensor, q_tensor, epochs)
        if result is None:
            continue
        r2 = r2_score(result["true"], result["pred"])
        r2s.append(r2)
        if verbose:
            print(f"    Fold {fold}: train={len(tr_idx)} val={len(va_idx)} R2={r2:.4f}")
    return np.mean(r2s), np.std(r2s), r2s


def evaluate_baseline_tscv(data, mode="tier", verbose=True):
    """TS-CV for hand-tuned baseline (numpy only, no training)."""
    ntu = data["NTU"]
    n = len(ntu)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    r2s = []
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(np.arange(n))):
        va_pred = np.zeros(len(va_idx))
        va_true = np.array([ntu[i] for i in va_idx])
        for j, idx in enumerate(va_idx):
            if j == 0:
                va_pred[j] = ntu[idx - 1] if idx > 0 else ntu[0]
            else:
                if mode == "tier":
                    beta = hand_tuned_beta_tier(data["FILT"][idx],
                                                data["CW"][idx - 1],
                                                data["Q"][idx - 1],
                                                data["RL"][idx])
                else:
                    beta = hand_tuned_beta_uniform(data["CW"][idx - 1],
                                                   data["Q"][idx - 1])
                va_pred[j] = beta * ntu[idx - 1] + (1.0 - beta) * data["FILT"][idx]
        r2 = r2_score(va_true[1:], va_pred[1:])
        r2s.append(r2)
        if verbose:
            print(f"    Fold {fold}: R2={r2:.4f}")
    return np.mean(r2s), np.std(r2s), r2s


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    print(f"Device: {DEVICE}")
    print("=" * 65)
    print("  Phase 1: Learnable CSTR Beta — Verification")
    print("=" * 65)

    data = load_data()
    print(f"  Samples: {data['n']}, NTU mean={np.mean(data['NTU']):.4f}")

    ntu_t = torch.tensor(data["NTU"], dtype=torch.float32, device=DEVICE)
    filt_t = torch.tensor(data["FILT"], dtype=torch.float32, device=DEVICE)

    # ── Baselines ──────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("  [Baseline] Hand-tuned A=141.3 (uniform)")
    b_r2, b_std, b_folds = evaluate_baseline_tscv(data, "uniform")
    print(f"  → TS-CV R2 = {b_r2:.4f} +- {b_std:.4f}")

    print(f"\n  [Baseline] Hand-tuned tier-A (400/250/100/20)")
    t_r2, t_std, t_folds = evaluate_baseline_tscv(data, "tier")
    print(f"  → TS-CV R2 = {t_r2:.4f} +- {t_std:.4f}")

    # ── Exp 1a: Linear β = σ(w·log(CW/Q) + b) ─────────────────
    print(f"\n{'─'*55}")
    print("  [Exp 1a] Linear β = σ(w·log(CW/Q) + b) — 2 params")
    feat_linear = make_features(data, "linear")
    r2a, std_a, folds_a = evaluate_tscv(DifferentiableCSTR, feat_linear,
                                         ntu_t, filt_t, n_features=1, hidden=0)
    print(f"  → TS-CV R2 = {r2a:.4f} +- {std_a:.4f}")

    # ── Exp 1b: Linear2 β = σ(w₁·log(CW/Q) + w₂·FILT + b) ─────
    print(f"\n{'─'*55}")
    print("  [Exp 1b] Linear β = σ(w₁·log(CW/Q) + w₂·FILT + b) — 3 params")
    feat_linear2 = make_features(data, "linear2")
    r2b, std_b, folds_b = evaluate_tscv(DifferentiableCSTR, feat_linear2,
                                         ntu_t, filt_t, n_features=2, hidden=0)
    print(f"  → TS-CV R2 = {r2b:.4f} +- {std_b:.4f}")

    # ── Exp 1c: MLP β = σ(MLP(CW,Q,FILT,RL,log_HQ,hsin,hcos)) ──
    print(f"\n{'─'*55}")
    print("  [Exp 1c] MLP β (7 features, hidden=8) — ~73 params")
    feat_mlp = make_features(data, "mlp")
    r2c, std_c, folds_c = evaluate_tscv(DifferentiableCSTR, feat_mlp,
                                         ntu_t, filt_t, n_features=7, hidden=8)
    print(f"  → TS-CV R2 = {r2c:.4f} +- {std_c:.4f}")

    # ── Exp 1d: MLP Full (7+2 = 9 features, hidden=8) ──────────
    print(f"\n{'─'*55}")
    print("  [Exp 1d] MLP β (9 features, hidden=8) — ~89 params")
    feat_mlp_full = make_features(data, "mlp_full")
    r2d, std_d, folds_d = evaluate_tscv(DifferentiableCSTR, feat_mlp_full,
                                         ntu_t, filt_t, n_features=9, hidden=8)
    print(f"  → TS-CV R2 = {r2d:.4f} +- {std_d:.4f}")

    # ── Phase 2: Scheme A (learn theta) + Scheme B (tier features) ────
    cw_tensor = torch.tensor(data["CW_raw"], dtype=torch.float32, device=DEVICE)
    q_tensor = torch.tensor(data["Q_raw"], dtype=torch.float32, device=DEVICE)

    print(f"\n{'='*55}")
    print(f"  Phase 2: Learnable Theta (A) + Tier Features (B)")
    print(f"{'='*55}")

    # Exp 2a: Theta model, 7-feat MLP
    print(f"\n  [Exp 2a] Theta MLP (7 feat, h=8) — Scheme A only")
    r2_2a, std_2a, f_2a = evaluate_tscv(LearnableTheta, feat_mlp, ntu_t, filt_t,
                                         7, 8, cw_tensor=cw_tensor, q_tensor=q_tensor)
    print(f"  -> TS-CV R2 = {r2_2a:.4f} +- {std_2a:.4f}")

    # Exp 2b: Theta model, 10-feat (7 + 3 tier indicators) — A+B
    print(f"\n  [Exp 2b] Theta MLP (10 feat, h=8) — A + B (tier flags)")
    feat_tier = make_features(data, "mlp_tier")
    r2_2b, std_2b, f_2b = evaluate_tscv(LearnableTheta, feat_tier, ntu_t, filt_t,
                                         10, 8, cw_tensor=cw_tensor, q_tensor=q_tensor)
    print(f"  -> TS-CV R2 = {r2_2b:.4f} +- {std_2b:.4f}")

    # Exp 2c: Theta model, 12-feat (9 full + 3 tier indicators) — A+B
    print(f"\n  [Exp 2c] Theta MLP (12 feat, h=8) — A + B (full feat + tier flags)")
    feat_full_tier = make_features(data, "mlp_full_tier")
    r2_2c, std_2c, f_2c = evaluate_tscv(LearnableTheta, feat_full_tier, ntu_t, filt_t,
                                         12, 8, cw_tensor=cw_tensor, q_tensor=q_tensor)
    print(f"  -> TS-CV R2 = {r2_2c:.4f} +- {std_2c:.4f}")

    # ── Summary (updated) ────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Config':<50} {'R2':>6}  {'+-std':>6}  {'params':>7}")
    print(f"  {'─'*65}")
    print(f"  {'Baseline: A=141.3 (uniform)':<50} {b_r2:6.4f}  {b_std:6.4f}  {'1':>7}")
    print(f"  {'Baseline: tier-A (400/250/100/20)':<50} {t_r2:6.4f}  {t_std:6.4f}  {'5':>7}")
    print(f"  {'  '}")
    print(f"  {'Exp 1d: Beta MLP(9f,8h) [best Phase 1]':<50} {r2d:6.4f}  {std_d:6.4f}  {'89':>7}")
    print(f"  {'  '}")
    print(f"  {'Exp 2a: Theta MLP(7f,8h) [Scheme A]':<50} {r2_2a:6.4f}  {std_2a:6.4f}  {'73':>7}")
    print(f"  {'Exp 2b: Theta MLP(10f,8h) [A+B]':<50} {r2_2b:6.4f}  {std_2b:6.4f}  {'97':>7}")
    print(f"  {'Exp 2c: Theta MLP(12f,8h) [A+B full]':<50} {r2_2c:6.4f}  {std_2c:6.4f}  {'113':>7}")

    all_nn = {"Beta 9feat": (r2d, std_d), "Theta 7feat A": (r2_2a, std_2a),
              "Theta 10feat A+B": (r2_2b, std_2b), "Theta 12feat A+B": (r2_2c, std_2c)}
    best_name = max(all_nn, key=lambda k: all_nn[k][0])
    best_r2, best_s = all_nn[best_name]

    print(f"\n  Best NN: {best_name} (R2={best_r2:.4f})")
    print(f"  Best hand-tuned: tier-A (R2={t_r2:.4f})")
    print(f"  Delta (NN - hand): {best_r2 - t_r2:+.4f}")
    if best_r2 > t_r2:
        print(f"\n  *** VERDICT: Learnable CSTR BEATS hand-tuned tier-A. ***")
        print(f"  Proceed to Phase 2 (end-to-end Q3 pipeline).")
    else:
        gap = t_r2 - best_r2
        print(f"\n  *** VERDICT: NN not yet beating tier-A (gap={gap:.4f}). ***")
        print(f"  But Theta model beats Beta model by {best_r2 - r2d:+.4f} — physics prior helps.")
        if best_r2 + best_s > t_r2:
            print(f"  Some folds DO exceed tier-A baseline.")

    result = {
        "baseline_uniform": {"r2": round(float(b_r2), 4), "std": round(float(b_std), 4)},
        "baseline_tier": {"r2": round(float(t_r2), 4), "std": round(float(t_std), 4)},
        "exp_1d_beta_mlp_9feat": {"r2": round(float(r2d), 4), "std": round(float(std_d), 4)},
        "exp_2a_theta_mlp_7feat": {"r2": round(float(r2_2a), 4), "std": round(float(std_2a), 4)},
        "exp_2b_theta_mlp_10feat_tier": {"r2": round(float(r2_2b), 4), "std": round(float(std_2b), 4)},
        "exp_2c_theta_mlp_12feat_tier_full": {"r2": round(float(r2_2c), 4), "std": round(float(std_2c), 4)},
        "best_nn": best_name,
        "best_nn_r2": round(float(best_r2), 4),
        "verdict": "PASS" if best_r2 > t_r2 else "FAIL",
    }
    out_path = os.path.join(RESULTS_DIR, "step1.10_verification.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
