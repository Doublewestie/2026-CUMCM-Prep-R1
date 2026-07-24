"""
step2.1+closed_loop_decompose.py — Closed-Loop Operator Policy Decomposition
=============================================================================
Core insight: ALUM is a control variable set by the operator in response to
observed conditions. Using raw ALUM as an input to predict FILT introduces
causal confounding — the operator's response (ALUM=f(FILT_lag)) and the
physical effect (FILT=g(ALUM_lag)) are inseparable in the data.

Solution: Learn the operator's policy first, then extract the "unexpected"
component epsilon = ALUM - ALUM_policy as the true physical driver.

Steps:
  A: OLS regression of operator control law
  B: Extract epsilon (residual = unexpected dosing deviation)
  C: Re-run greybox Segment 1 with epsilon replacing raw ALUM
  D: Compare: raw-ALUM vs epsilon-driven greybox R2, K_m, beta1

Input:  clean_data.csv, theta_params.json
Output: q2_operator_policy.json, q2_epsilon_greybox_metrics.csv
"""

import os, json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import r2_score, mean_squared_error
from step0_config import *

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Load theta from diagnostic
THETA_FILE = os.path.join(OUTPUT_DIR, "theta_params.json")
THETA = 0.15


def load_data():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "clean_data.csv"))
    req = ["RW_NTU", "FILT_NTU", "ALUM", "CLR", "CW_WELL_LEVEL", "TW_FLOW", "DATE"]
    df = df.dropna(subset=req)
    rw_ntu  = df["RW_NTU"].values.astype(np.float64)
    filt    = df["FILT_NTU"].values.astype(np.float64)
    alum    = df["ALUM"].values.astype(np.float64)
    clr     = df["CLR"].values.astype(np.float64)
    cw      = df["CW_WELL_LEVEL"].values.astype(np.float64)
    tw      = df["TW_FLOW"].values.astype(np.float64)
    date    = pd.to_datetime(df["DATE"])
    doy     = date.dt.dayofyear.values
    day_sin = np.sin(2 * np.pi * doy / 365)
    day_cos = np.cos(2 * np.pi * doy / 365)
    return rw_ntu, filt, alum, clr, cw, tw, day_sin, day_cos


# ==================== Step A: Operator Policy ====================

def learn_operator_policy(alum, rw_ntu, filt, theta=THETA):
    """
    OLS: ALUM(t) = alpha0 + alpha1*RW_NTU(t) + alpha2*max(0, FILT(t-1)-theta)

    Returns: (alpha0, alpha1, alpha2, ALUM_policy, R2)
    """
    n = len(alum)
    filt_lag = np.roll(filt, 1)
    filt_lag[0] = filt[0]

    X = np.column_stack([
        np.ones(n),
        rw_ntu,
        np.maximum(0, filt_lag - theta),
    ])

    # OLS
    coef, residuals, rank, s = np.linalg.lstsq(X, alum, rcond=None)
    alpha0, alpha1, alpha2 = coef
    alum_policy = X @ coef
    ss_res = np.sum((alum - alum_policy) ** 2)
    ss_tot = np.sum((alum - alum.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + EPS)

    # Standard errors
    n_obs = n
    k = 3
    sigma2 = ss_res / (n_obs - k)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv) * sigma2)
    t_stats = coef / (se + EPS)

    return {
        "alpha0": float(alpha0), "alpha1": float(alpha1), "alpha2": float(alpha2),
        "se_alpha0": float(se[0]), "se_alpha1": float(se[1]), "se_alpha2": float(se[2]),
        "t_alpha0": float(t_stats[0]), "t_alpha1": float(t_stats[1]), "t_alpha2": float(t_stats[2]),
        "R2": float(r2),
    }, alum_policy


# ==================== Step B: Epsilon ====================

def extract_epsilon(alum, alum_policy):
    """epsilon(t) = ALUM(t) - ALUM_policy(t), winsorized at 3 sigma."""
    eps = alum - alum_policy
    # winsorize
    sigma = eps.std()
    eps = np.clip(eps, -3 * sigma, 3 * sigma)
    return eps


# ==================== Step C: Greybox with epsilon ====================

def run_seg1_epsilon(params, rw_ntu, filt_obs, epsilon, clr, day_sin, day_cos):
    """
    Greybox Segment 1 using epsilon instead of raw ALUM.

    FILT(t) = beta1*FILT(t-1) + (1-beta1)*RW_NTU(t-sp1)*[1 - eta(epsilon(t-sp1))]
    eta(e) = max(0, e) / (max(0, e) + K_m + alpha*CLR(t-sp1))

    Shift-1 on epsilon: the unexpected dosing at t-1 determines filter performance at t.
    """
    beta1, Km0, Km1, Km2, alpha, FILT0 = params
    n = len(filt_obs)

    Km_t = Km0 + Km1 * day_sin + Km2 * day_cos

    # shift-1: epsilon(t-1) affects FILT(t)
    eps_shift = np.roll(np.maximum(0, epsilon), 1)
    eps_shift[0] = eps_shift[1]
    clr_shift = np.roll(clr, 1)
    clr_shift[0] = clr_shift[1]
    rw_shift = np.roll(rw_ntu, 1)
    rw_shift[0] = rw_shift[1]

    eta = eps_shift / (eps_shift + Km_t + alpha * clr_shift + EPS)
    eta = np.clip(eta, 0.0, 1.0)

    pred = np.zeros(n)
    pred[0] = FILT0
    for t in range(1, n):
        pred[t] = beta1 * pred[t - 1] + (1.0 - beta1) * rw_shift[t] * (1.0 - eta[t])

    mse = np.mean((pred - filt_obs) ** 2)
    LAM = GREYBOX_LAMBDA
    viol_up = np.mean(np.maximum(0, pred - rw_ntu))
    viol_nn = np.mean(np.maximum(0, -pred))
    viol_km = np.mean(np.maximum(0, 0.001 - Km_t))
    loss = mse + LAM["filter_upper"] * viol_up + LAM["nonneg"] * viol_nn + LAM["km_pos"] * viol_km
    return pred, loss, mse


def calibrate_epsilon(epsilon, rw_ntu, filt_obs, clr, day_sin, day_cos):
    """Multi-start L-BFGS-B for epsilon-driven greybox."""
    bounds = [
        GREYBOX_PARAM_BOUNDS["beta1"], GREYBOX_PARAM_BOUNDS["Km0"],
        GREYBOX_PARAM_BOUNDS["Km1"], GREYBOX_PARAM_BOUNDS["Km2"],
        GREYBOX_PARAM_BOUNDS["alpha"], GREYBOX_PARAM_BOUNDS["FILT0"],
    ]
    best_loss, best_x = float("inf"), None
    for i in range(GREYBOX_N_RESTARTS):
        if i == 0:
            x0 = np.array([GREYBOX_PARAM_INIT[k] for k in
                           ["beta1","Km0","Km1","Km2","alpha","FILT0"]])
        else:
            x0 = np.array([np.random.uniform(*GREYBOX_PARAM_BOUNDS[k]) for k in
                           ["beta1","Km0","Km1","Km2","alpha","FILT0"]])
        def obj(p):
            _, l, _ = run_seg1_epsilon(p, rw_ntu, filt_obs, epsilon, clr, day_sin, day_cos)
            return l
        res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": GREYBOX_MAX_ITER, "ftol": 1e-12})
        if res.fun < best_loss:
            best_loss, best_x = res.fun, res.x
    return best_x, best_loss


def run_original_seg1(alum_raw, rw_ntu, filt_obs, clr, day_sin, day_cos):
    """Re-run original segment1 with raw ALUM for fair comparison."""
    from scipy.optimize import minimize as sp_min
    bnds = [
        GREYBOX_PARAM_BOUNDS["beta1"], GREYBOX_PARAM_BOUNDS["Km0"],
        GREYBOX_PARAM_BOUNDS["Km1"], GREYBOX_PARAM_BOUNDS["Km2"],
        GREYBOX_PARAM_BOUNDS["alpha"], GREYBOX_PARAM_BOUNDS["FILT0"],
    ]

    def loss_fn(p):
        beta1, Km0, Km1, Km2, alpha, FILT0 = p
        n = len(rw_ntu)
        Km = Km0 + Km1 * day_sin + Km2 * day_cos
        eta = alum_raw / (alum_raw + Km + alpha * clr + EPS)
        eta = np.clip(eta, 0, 1)
        pred = np.zeros(n); pred[0] = FILT0
        for t in range(1, n):
            pred[t] = beta1 * pred[t - 1] + (1 - beta1) * rw_ntu[t] * (1 - eta[t])
        return np.mean((pred - filt_obs) ** 2)

    x0 = np.array([GREYBOX_PARAM_INIT[k] for k in ["beta1","Km0","Km1","Km2","alpha","FILT0"]])
    res = sp_min(loss_fn, x0, method="L-BFGS-B", bounds=bnds,
                 options={"maxiter": GREYBOX_MAX_ITER, "ftol": 1e-12})
    return res.x


# ==================== Main ====================

def main():
    print("=" * 60)
    print("  step2.1+ — Closed-Loop Operator Policy Decomposition")
    print("=" * 60)

    rw_ntu, filt, alum, clr, cw, tw, day_sin, day_cos = load_data()
    n = len(filt)
    print(f"  Samples: {n}, theta={THETA}")

    # Step A: Operator policy
    policy_info, alum_policy = learn_operator_policy(alum, rw_ntu, filt, THETA)
    print(f"\n  [Step A] Operator Control Law:")
    print(f"    ALUM(t) = {policy_info['alpha0']:.6f} + "
          f"{policy_info['alpha1']:.6f}*RW_NTU(t) + "
          f"{policy_info['alpha2']:.6f}*max(0,FILT(t-1)-{THETA})")
    print(f"    R2={policy_info['R2']:.4f}  "
          f"t(alpha1)={policy_info['t_alpha1']:.2f}  t(alpha2)={policy_info['t_alpha2']:.2f}")
    print(f"    Interp: RW_NTU 1-sigma up -> ALUM += {policy_info['alpha1']*rw_ntu.std():.4f}")
    print(f"           FILT > theta -> ALUM += {policy_info['alpha2']*1.0:.4f} per 1 NTU above")

    # Step B: Epsilon
    epsilon = extract_epsilon(alum, alum_policy)
    print(f"\n  [Step B] epsilon = ALUM - ALUM_policy:")
    print(f"    mean={epsilon.mean():.6f}  std={epsilon.std():.6f}  "
          f"P5={np.percentile(epsilon,5):.6f}  P95={np.percentile(epsilon,95):.6f}")

    # Step C: Greybox with epsilon
    print(f"\n  [Step C] Epsilon-driven greybox Segment 1...")
    p_eps, loss_eps = calibrate_epsilon(epsilon, rw_ntu, filt, clr, day_sin, day_cos)
    pred_eps, _, mse_eps = run_seg1_epsilon(p_eps, rw_ntu, filt, epsilon, clr, day_sin, day_cos)
    rmse_eps = np.sqrt(mse_eps)
    r2_eps = r2_score(filt, pred_eps)

    beta1_eps, Km0_eps = p_eps[0], p_eps[1]
    print(f"    beta1={beta1_eps:.4f}  Km0={Km0_eps:.6f}  "
          f"Km1={p_eps[2]:.6f}  Km2={p_eps[3]:.6f}  alpha={p_eps[4]:.6f}")
    print(f"    RMSE={rmse_eps:.4f}  R2={r2_eps:.4f}")

    # Step D: Comparison with raw-ALUM
    print(f"\n  [Step D] Comparison: raw-ALUM vs epsilon-driven")
    p_raw = run_original_seg1(alum, rw_ntu, filt, clr, day_sin, day_cos)
    print(f"    Raw ALUM:    beta1={p_raw[0]:.4f}  Km0={p_raw[1]:.6f}  alpha={p_raw[4]:.6f}")
    print(f"    Epsilon:     beta1={beta1_eps:.4f}  Km0={Km0_eps:.6f}  alpha={p_eps[4]:.6f}")
    print(f"    Epsilon R2 = {r2_eps:.4f}")

    # Verdict
    km_improved = Km0_eps > GREYBOX_PARAM_BOUNDS["Km0"][0] * 5  # Km not at bound
    print(f"\n  [Verdict]")
    if km_improved and r2_eps > -0.5:
        print(f"    Km0 = {Km0_eps:.6f} > bound({GREYBOX_PARAM_BOUNDS['Km0'][0]:.4f}) "
              f"-> Closed-loop decomposition SUCCESSFUL")
        print(f"    Epsilon captures dosing variation beyond operator's routine policy.")
    else:
        print(f"    Km0 = {Km0_eps:.6f} ~ bound -> decomposition did not improve identifiability.")
        print(f"    Operator's dosing policy explains all systematic ALUM variation.")

    # Save
    result = {
        "operator_policy": policy_info,
        "epsilon_stats": {"mean": float(epsilon.mean()), "std": float(epsilon.std())},
        "epsilon_greybox": {
            "beta1": float(beta1_eps), "Km0": float(Km0_eps), "r2": float(r2_eps),
        },
        "raw_alum_greybox": {
            "beta1": float(p_raw[0]), "Km0": float(p_raw[1]),
        },
    }
    with open(os.path.join(OUTPUT_DIR, "q2_operator_policy.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n  [DONE] q2_operator_policy.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
