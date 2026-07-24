"""
step1.7+_tier_froude.py — Per-tier A with Froude dead-zone modulation
=======================================================================
Refinement over step1.7: A is no longer global but tier-dependent.

  A_eff(t) = A_tier(FILT[t]) / (1 + k_Fr * Q[t-1] / H[t-1]^(3/2))

  A_tier = { A_T1  if FILT[t] <= 0.05
             A_T2  if 0.05 < FILT[t] <= 0.15
             A_T3  if FILT[t] > 0.15           }

Three-phase scan:
  Phase 1 (k_Fr=0):       independently optimize A_T1, A_T2, A_T3  (18 evals)
  Phase 2 (fix best As):   scan k_Fr                                 (5 evals)
  Phase 3 (fix best k_Fr): backtrack optimize A_T1, A_T2, A_T3       (18 evals)
  Total: 41
"""

import os, json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
EPS = 1e-6

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
CLEAN_CSV = os.path.join(OUTPUT_DIR, "clean_data.csv")

T1_THR = 0.05
T2_THR = 0.15


def load_data():
    df = pd.read_csv(CLEAN_CSV)
    df["DATE"] = pd.to_datetime(df["DATE"])
    return {
        "FILT": df["FILT_NTU"].values.astype(np.float64),
        "NTU":  df["NTU"].values.astype(np.float64),
        "CW":   df["CW_WELL_LEVEL"].values.astype(np.float64),
        "Q":    df["TW_FLOW"].values.astype(np.float64),
        "RW_NTU": df["RW_NTU"].values.astype(np.float64),
    }


def predict_tier_froude(data, A_T1, A_T2, A_T3, k_Fr, gamma_eta=0.0):
    filt = data["FILT"]
    ntu  = data["NTU"]
    cw   = data["CW"]
    q    = data["Q"]
    n = len(ntu)
    pred = np.zeros(n)
    if n == 0:
        return pred
    pred[0] = ntu[0]
    for t in range(1, n):
        H = max(cw[t - 1], 0.1)
        Qv = max(q[t - 1], 1.0)
        # tier-dependent base area
        ft = filt[t]
        if ft <= T1_THR:
            A0 = A_T1
        elif ft <= T2_THR:
            A0 = A_T2
        else:
            A0 = A_T3
        # Froude dead-zone modulation
        if k_Fr > 0:
            A_eff = A0 / (1.0 + k_Fr * Qv / H ** 1.5)
        else:
            A_eff = A0
        theta = A_eff * H / Qv
        theta = max(theta, 0.02)
        beta = np.exp(-2.0 / theta)
        beta = np.clip(beta, 0.001, 0.999)
        pred[t] = beta * ntu[t - 1] + (1.0 - beta) * filt[t]
    if gamma_eta != 0:
        eta_coag = (data.get("RW_NTU", np.zeros(len(filt))) - filt) / (data.get("RW_NTU", np.ones(len(filt))) + EPS)
        eta_mean = eta_coag.mean()
        pred = pred + gamma_eta * (eta_coag - eta_mean)
    return np.clip(pred, 0.0, np.inf)


def eval_config(data, A_T1, A_T2, A_T3, k_Fr, gamma_eta=0.0):
    pred = predict_tier_froude(data, A_T1, A_T2, A_T3, k_Fr, gamma_eta)
    ntu  = data["NTU"]
    filt = data["FILT"]
    res = {}
    ssr = np.sum((ntu - pred) ** 2)
    sst = np.sum((ntu - ntu.mean()) ** 2)
    res["R2_all"]   = round(float(1 - ssr / (sst + EPS)), 4)
    res["RMSE_all"] = round(float(np.sqrt(np.mean((ntu - pred) ** 2))), 4)

    for tid, tname, mask in [
        (1, "T1", filt <= T1_THR),
        (2, "T2", (filt > T1_THR) & (filt <= T2_THR)),
        (3, "T3", filt > T2_THR),
    ]:
        if mask.sum() < 10:
            res[f"R2_{tname}"] = None
        else:
            ssr = np.sum((ntu[mask] - pred[mask]) ** 2)
            sst = np.sum((ntu[mask] - ntu[mask].mean()) ** 2)
            res[f"R2_{tname}"] = round(float(1 - ssr / (sst + EPS)), 4)

    ext = filt > 0.5
    if ext.sum() >= 10:
        ssr = np.sum((ntu[ext] - pred[ext]) ** 2)
        sst = np.sum((ntu[ext] - ntu[ext].mean()) ** 2)
        res["R2_ext"] = round(float(1 - ssr / (sst + EPS)), 4)
        res["n_ext"]  = int(ext.sum())
    else:
        res["R2_ext"] = None
        res["n_ext"]  = 0

    res["A_T1"] = A_T1; res["A_T2"] = A_T2; res["A_T3"] = A_T3; res["k_Fr"] = k_Fr
    return res


# ================================================================
# Main
# ================================================================
def main():
    print("=" * 70)
    print("  step1.7_final — Per-Tier CSTR (Final)")
    print("=" * 70)

    data = load_data()
    print(f"\n  Loaded {len(data['NTU'])} samples.")

    rows = []

    A1_grid = [200, 300, 400, 500, 700, 1000]
    A2_grid = [100, 150, 200, 250, 300, 400]
    A3_grid = [20, 30, 50, 70, 100, 150]
    kF_grid = [0, 0.01, 0.02, 0.03, 0.04]

    # ---- Baseline ----
    r_bl = eval_config(data, 141.3, 141.3, 141.3, 0.0)
    print(f"\n  [Baseline] A_all=141.3, k_Fr=0  →  "
          f"R2={r_bl['R2_all']}  T1={r_bl['R2_T1']}  T2={r_bl['R2_T2']}  T3={r_bl['R2_T3']}  ext={r_bl['R2_ext']}")

    # ==================== PHASE 1: independent per-tier A (k_Fr=0) ====================
    print(f"\n{'='*70}")
    print("  Phase 1: per-tier A (k_Fr = 0)")
    print(f"{'='*70}")

    best_p1 = {}  # {tier: (best_A, best_R2)}

    for tier_label, tier_idx, A_grid in [
        ("T1", 0, A1_grid),
        ("T2", 1, A2_grid),
        ("T3", 2, A3_grid),
    ]:
        print(f"\n  [{tier_label}] scanning A...")
        best_r2, best_a = -999, 0
        # baseline A for other tiers = 141.3 (neutral, minimizes interference)
        defaults = {0: 141.3, 1: 141.3, 2: 141.3}
        for a_val in A_grid:
            defaults[tier_idx] = a_val
            r = eval_config(data, defaults[0], defaults[1], defaults[2], 0.0)
            marker = ""
            tier_r2 = r[f"R2_{tier_label}"]
            full_r2 = r["R2_all"]
            if tier_r2 is not None and tier_r2 > best_r2:
                best_r2 = tier_r2; best_a = a_val; marker = " ***"
            print(f"    A_{tier_label}={a_val:5.0f}  "
                  f"R2_{tier_label}={tier_r2}  R2_all={full_r2}{marker}")
        best_p1[tier_idx] = (best_a, best_r2)
        print(f"  [Best A_{tier_label}] = {best_a} (R2_{tier_label}={best_r2:.4f})")

    A1_best, A2_best, A3_best = best_p1[0][0], best_p1[1][0], best_p1[2][0]
    r_p1 = eval_config(data, A1_best, A2_best, A3_best, 0.0)
    r_p1["config"] = "Phase1_best"
    rows.append(r_p1)
    print(f"\n  [Phase 1 combined] A1={A1_best} A2={A2_best} A3={A3_best} k_Fr=0  →  "
          f"R2_all={r_p1['R2_all']}  T1={r_p1['R2_T1']}  T2={r_p1['R2_T2']}  T3={r_p1['R2_T3']}  ext={r_p1['R2_ext']}")

    # ==================== PHASE 2: scan k_Fr with fixed As ====================
    print(f"\n{'='*70}")
    print(f"  Phase 2: k_Fr scan (A1={A1_best}, A2={A2_best}, A3={A3_best})")
    print(f"{'='*70}")

    best_kfr, best_k_r2 = 0.0, -999
    for kf in kF_grid:
        r = eval_config(data, A1_best, A2_best, A3_best, kf)
        r["config"] = f"P2_kF={kf:.2f}"
        rows.append(r)
        marker = ""
        if r["R2_all"] is not None and r["R2_all"] > best_k_r2:
            best_k_r2 = r["R2_all"]; best_kfr = kf; marker = " ***"
        print(f"    k_Fr={kf:.2f}  "
              f"R2_all={r['R2_all']}  T1={r['R2_T1']}  T2={r['R2_T2']}  T3={r['R2_T3']}  ext={r['R2_ext']}{marker}")
    print(f"  [Best k_Fr] = {best_kfr:.2f} (R2_all={best_k_r2:.4f})")

    # ==================== PHASE 3: backtrack optimize As with best k_Fr ====================
    print(f"\n{'='*70}")
    print(f"  Phase 3: backtrack A (k_Fr = {best_kfr:.2f})")
    print(f"{'='*70}")

    best_p3 = {}
    for tier_label, tier_idx, A_grid in [
        ("T1", 0, A1_grid),
        ("T2", 1, A2_grid),
        ("T3", 2, A3_grid),
    ]:
        print(f"\n  [{tier_label}] backtracking A...")
        best_r2, best_a = -999, 0
        defaults = {0: A1_best, 1: A2_best, 2: A3_best}
        for a_val in A_grid:
            defaults[tier_idx] = a_val
            r = eval_config(data, defaults[0], defaults[1], defaults[2], best_kfr)
            tier_r2 = r[f"R2_{tier_label}"]
            if tier_r2 is not None and tier_r2 > best_r2:
                best_r2 = tier_r2; best_a = a_val
            marker = " ***" if a_val == best_a else ""
            # only print if different from phase 1 best
        best_p3[tier_idx] = (best_a, best_r2)
        print(f"  [Best A_{tier_label}] = {best_a} (R2_{tier_label}={best_r2:.4f})")

    A1_final, A2_final, A3_final = best_p3[0][0], best_p3[1][0], best_p3[2][0]
    r_p3 = eval_config(data, A1_final, A2_final, A3_final, best_kfr)
    r_p3["config"] = "Phase3_final"
    rows.append(r_p3)
    print(f"\n  [Phase 3 final] A1={A1_final} A2={A2_final} A3={A3_final} k_Fr={best_kfr:.2f}  →  "
          f"R2_all={r_p3['R2_all']}  T1={r_p3['R2_T1']}  T2={r_p3['R2_T2']}  T3={r_p3['R2_T3']}  ext={r_p3['R2_ext']}")

    # ==================== PHASE 4: eta_coag correction term (use true eta) ====================
    print(f"\n{'='*70}")
    print(f"  Phase 4: eta_coag correction: NTU += gamma * (eta - eta_mean)")
    print(f"  (A1={A1_final}, A2={A2_final}, A3={A3_final}, k_Fr={best_kfr:.2f})")
    print(f"{'='*70}")

    best_g_eta, best_g_r2 = 0.0, r_p3["R2_all"]
    for g in [-0.05, -0.02, -0.01, 0, 0.01, 0.02, 0.05, 0.10]:
        r_g = eval_config(data, A1_final, A2_final, A3_final, best_kfr, gamma_eta=g)
        r_g["config"] = f"P4_gEta={g:.2f}"
        rows.append(r_g)
        m = " ***" if r_g["R2_all"] > best_g_r2 else ""
        if r_g["R2_all"] > best_g_r2:
            best_g_r2 = r_g["R2_all"]; best_g_eta = g
        print(f"    gamma={g:+.2f}  "
              f"R2_all={r_g['R2_all']}  T1={r_g['R2_T1']}  T3={r_g['R2_T3']}  ext={r_g['R2_ext']}{m}")
    print(f"  [Best gamma_eta] = {best_g_eta:+.2f} (R2_all={best_g_r2:.4f}, dR2={best_g_r2 - r_p3['R2_all']:+.4f})")

    # Adopt best gamma if dR2 meaningful
    gamma_final = best_g_eta if (best_g_r2 - r_p3['R2_all']) > 0.002 else 0.0
    r_final = eval_config(data, A1_final, A2_final, A3_final, best_kfr, gamma_eta=gamma_final)
    r_final["config"] = "FINAL"
    rows.append(r_final)
    print(f"  [Final gamma] = {gamma_final:+.2f}")
    if gamma_final != 0:
        print(f"  eta correction active: dR2_all={best_g_r2 - r_p3['R2_all']:+.4f}")

    # ==================== Phase 4b: loading correction (RW_FLOW * RW_NTU) ====================
    rw_ntu = data["RW_NTU"]
    rw_flow = data.get("RW_FLOW", None)
    if rw_flow is None:
        rw_flow = pd.read_csv(CLEAN_CSV)["RW_FLOW"].values.astype(np.float64)
        data["RW_FLOW"] = rw_flow
    loading_raw = rw_flow * rw_ntu
    # Normalize loading to avoid numerical issues
    L_mean = loading_raw.mean()
    L_std  = loading_raw.std()

    print(f"\n{'='*70}")
    print(f"  Phase 4b: loading correction (RW_FLOW*RW_NTU normalized)")
    print(f"  (A1={A1_final}, A2={A2_final}, A3={A3_final})")
    print(f"{'='*70}")

    def eval_loading(gamma_load):
        pred = predict_tier_froude(data, A1_final, A2_final, A3_final, k_Fr=0)
        if gamma_load != 0:
            pred = pred + gamma_load * (loading_raw - L_mean) / max(L_std, EPS)
        pred = np.clip(pred, 0, np.inf)
        ntu_v = data["NTU"]; filt_v = data["FILT"]
        ssr = np.sum((ntu_v-pred)**2); sst = np.sum((ntu_v-ntu_v.mean())**2)
        r2_all = 1-ssr/(sst+EPS)
        for tk, mk in [("T1",filt_v<=0.05),("T3",filt_v>0.15),("ext",filt_v>0.5)]:
            pass
        m_t3 = filt_v>0.15; m_ex = filt_v>0.5
        ssr3 = np.sum((ntu_v[m_t3]-pred[m_t3])**2); sst3 = np.sum((ntu_v[m_t3]-ntu_v[m_t3].mean())**2); r2t3 = 1-ssr3/(sst3+EPS)
        ssre = np.sum((ntu_v[m_ex]-pred[m_ex])**2); sste = np.sum((ntu_v[m_ex]-ntu_v[m_ex].mean())**2); r2ex = 1-ssre/(sste+EPS)
        return round(r2_all,4), round(r2t3,4), round(r2ex,4), ssr

    best_gL, best_r2L, base_r2L = 0.0, 0.0, 0.0
    for gL in [0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]:
        r2a, r2t3, r2ex, _ = eval_loading(gL)
        m = ""
        if gL == 0: base_r2L = r2a
        if r2a > best_r2L: best_r2L = r2a; best_gL = gL; m = " ***"
        print(f"    gL={gL:.2f}  R2_all={r2a:.4f}  T3={r2t3:.4f}  ext={r2ex:.4f}{m}")
    print(f"  [Best gL] = {best_gL:.2f} (dR2={best_r2L-base_r2L:+.4f})")
    print(f"  [Gate: dR2={best_r2L-base_r2L:+.4f} %s 0.002] %s" %
          (">" if best_r2L-base_r2L > 0.002 else "<=",
           "ADOPT" if best_r2L-base_r2L > 0.002 else "SKIP"))

    # ==================== Phase 4c: RW_NTU correction ====================
    R_mean = rw_ntu.mean(); R_std = rw_ntu.std()
    print(f"\n{'='*70}")
    print(f"  Phase 4c: RW_NTU correction (normalized)")
    print(f"  (A1={A1_final}, A2={A2_final}, A3={A3_final})")
    print(f"{'='*70}")

    def eval_rw(gamma_rw):
        pred = predict_tier_froude(data, A1_final, A2_final, A3_final, k_Fr=0)
        if gamma_rw != 0:
            pred = pred + gamma_rw * (rw_ntu - R_mean) / max(R_std, EPS)
        pred = np.clip(pred, 0, np.inf)
        ntu_v = data["NTU"]; ssr = np.sum((ntu_v-pred)**2); sst = np.sum((ntu_v-ntu_v.mean())**2)
        r2a = 1-ssr/(sst+EPS)
        m_t3 = data["FILT"]>0.15; m_ex = data["FILT"]>0.5
        ssr3 = np.sum((ntu_v[m_t3]-pred[m_t3])**2); sst3 = np.sum((ntu_v[m_t3]-ntu_v[m_t3].mean())**2); r2t3 = 1-ssr3/(sst3+EPS)
        ssre = np.sum((ntu_v[m_ex]-pred[m_ex])**2); sste = np.sum((ntu_v[m_ex]-ntu_v[m_ex].mean())**2); r2ex = 1-ssre/(sste+EPS)
        return round(r2a,4), round(r2t3,4), round(r2ex,4)

    best_gR, best_r2R, base_r2R = 0.0, 0.0, 0.0
    for gR in [0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]:
        r2a, r2t3, r2ex = eval_rw(gR)
        m = ""
        if gR == 0: base_r2R = r2a
        if r2a > best_r2R: best_r2R = r2a; best_gR = gR; m = " ***"
        print(f"    gR={gR:.2f}  R2_all={r2a:.4f}  T3={r2t3:.4f}  ext={r2ex:.4f}{m}")
    print(f"  [Best gR] = {best_gR:.2f} (dR2={best_r2R-base_r2R:+.4f})")
    print(f"  [Gate: dR2={best_r2R-base_r2R:+.4f} %s 0.002] %s" %
          (">" if best_r2R-base_r2R > 0.002 else "<=",
           "ADOPT" if best_r2R-base_r2R > 0.002 else "SKIP"))

    # ==================== Save ====================
    df_out = pd.DataFrame(rows)
    col_order = ["config", "A_T1", "A_T2", "A_T3", "k_Fr",
                 "R2_all", "RMSE_all", "R2_T1", "R2_T2", "R2_T3", "R2_ext", "n_ext"]
    df_out = df_out[[c for c in col_order if c in df_out.columns]]
    csv_path = os.path.join(OUTPUT_DIR, "cstr_final_ablation.csv")
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[DONE] {csv_path}")

    best_path = os.path.join(OUTPUT_DIR, "cstr_final_best.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump({"A_T1": A1_final, "A_T2": A2_final, "A_T3": A3_final,
                   "k_Fr": best_kfr, "gamma_eta": gamma_final,
                   "R2_all": r_final["R2_all"],
                   "R2_T1": r_final["R2_T1"], "R2_T2": r_final["R2_T2"],
                   "R2_T3": r_final["R2_T3"], "R2_ext": r_final["R2_ext"]}, f, indent=2)
    print(f"[DONE] {best_path}")

    # ==================== Summary ====================
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Baseline (A=141.3,k=0):           "
          f"R2={r_bl['R2_all']}   T1={r_bl['R2_T1']}  T3={r_bl['R2_T3']}  ext={r_bl['R2_ext']}")
    print(f"  Phase 1 (分tier,k=0):             "
          f"R2={r_p1['R2_all']}   T1={r_p1['R2_T1']}  T3={r_p1['R2_T3']}  ext={r_p1['R2_ext']}")
    print(f"  Phase 3 (tier+Fr,k={best_kfr:.2f}):   "
          f"R2={r_p3['R2_all']}   T1={r_p3['R2_T1']}  T3={r_p3['R2_T3']}  ext={r_p3['R2_ext']}")
    print(f"  FINAL (+eta_correction g={gamma_final:+.2f}):  "
          f"R2={r_final['R2_all']}   T1={r_final['R2_T1']}  T3={r_final['R2_T3']}  ext={r_final['R2_ext']}")
    print(f"  dR2 (baseline→final):             "
          f"full={r_final['R2_all'] - r_bl['R2_all']:+.4f}  "
          f"T1={r_final['R2_T1'] - r_bl['R2_T1']:+.4f}  "
          f"T3={r_final['R2_T3'] - r_bl['R2_T3']:+.4f}  "
          f"ext={r_final['R2_ext'] - r_bl['R2_ext']:+.4f}")
    print(f"  Final params: A1={A1_final}  A2={A2_final}  A3={A3_final}  "
          f"k_Fr={best_kfr:.2f}  gamma_eta={gamma_final:+.2f}")

    # ==================== Figure ====================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    configs = ["Baseline", "Phase1", "Phase3"]
    metrics = {"R2_all": "steelblue", "R2_T1": "seagreen", "R2_T3": "darkorange", "R2_ext": "firebrick"}
    for (label, color) in [("R2_all", "steelblue"), ("R2_T3", "darkorange"), ("R2_ext", "firebrick")]:
        # Left: timeline bars
        pass  # skip for simplicity

    # Simple grouped bar
    ax = axes[0]
    x = np.arange(3)
    w = 0.2
    vals = [
        [r_bl["R2_all"], r_p1["R2_all"], r_p3["R2_all"]],
        [r_bl["R2_T3"],  r_p1["R2_T3"],  r_p3["R2_T3"]],
        [r_bl["R2_T1"],  r_p1["R2_T1"],  r_p3["R2_T1"]],
        [r_bl["R2_ext"], r_p1["R2_ext"], r_p3["R2_ext"]],
    ]
    names = ["R2_all", "R2_T3", "R2_T1", "R2_ext(>0.5)"]
    colors = ["steelblue", "darkorange", "seagreen", "firebrick"]
    for i, (vs, nm, cl) in enumerate(zip(vals, names, colors)):
        ax.bar(x + (i - 1.5) * w, vs, w, label=nm, color=cl, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(["Baseline", "Phase1\n(tier A)", "Phase3\n(tier+Fr)"], fontsize=8)
    ax.set_title("Step1.7+ Ablation Summary"); ax.legend(fontsize=7)

    # Middle: A per tier
    ax = axes[1]
    ax.bar(["T1", "T2", "T3"], [A1_best, A2_best, A3_best], color=["seagreen", "gold", "darkorange"], alpha=0.7, label="Phase1 A")
    ax.bar(["T1", "T2", "T3"], [A1_final, A2_final, A3_final], color=["seagreen", "gold", "darkorange"], alpha=0.4, hatch="//", label="Phase3 A")
    ax.set_ylabel("A_eff"); ax.set_title("Per-Tier A"); ax.legend(fontsize=7)

    # Right: k_Fr sensitivity
    ax = axes[2]
    phase2_rows = [r for r in rows if r["config"].startswith("P2_")]
    kfs = [r["k_Fr"] for r in phase2_rows]
    r2s = [r["R2_all"] for r in phase2_rows]
    ax.plot(kfs, r2s, "o-", color="steelblue", ms=8)
    ax.set_xlabel("k_Fr"); ax.set_ylabel("R2_all")
    ax.set_title("k_Fr Sensitivity"); ax.axhline(y=r_bl["R2_all"], color="gray", ls="--", lw=0.8)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "cstr_final.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[DONE] {fig_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
