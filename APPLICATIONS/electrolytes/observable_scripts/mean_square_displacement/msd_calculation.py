#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSD & diffusion (Na and P as PF6 tracer) across multiple systems.
- Discard first 100 ps (equilibration).
- Compute MSD and fit D using the following 900 ps window (τ = 0..900 ps).
- Save ONE combined PNG (2×3 subplots) and ONE CSV.

Assumes 50 fs between saved frames if frame.info['time'] missing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ase.io import read
from ase.geometry import find_mic
from pathlib import Path
from tqdm import tqdm
import pickle
# ─────────────── user: solvents & trajectories ───────────────────────────────
solvents = ["dme_w_hessian","dme_wo_hessian"] 
distilled_trajs = ["/home/yuejian/project/MLFF-distill/ablate_distillation/ablation_md_simulation_10ns/md_omol_naotf_dme_s1p1_omol_10/md_omol_naotf_dme_s1p1_omol_10.traj",
                   "/home/yuejian/project/MLFF-distill/ablate_distillation/ablation_md_simulation_10ns/md_omol_naotf_dme_s1p1_omol_undistill/md_omol_naotf_dme_s1p1_omol_undistill.traj"]
traj_info = list(zip(distilled_trajs, solvents))  # (path, label)
analyze_first_n_frames = 800000 # [500000,600000,700000,800000,900000,1000000]

# Equilibration trim and frame timing
EQ_TIME_PS   = 100.0   # discard first 100 ps
dt_fallback  = 0.05    # ps (50 fs) if frame.info['time'] is absent

# Output root
out_dir = Path("/home/yuejian/project/MLFF-distill/ablate_distillation/msd/10ns_intermediate/10ns_out/"+f"{analyze_first_n_frames}")
out_dir.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {out_dir}")
# ───────────────────── helper functions ──────────────────────────────────────
def build_time_array(frames, dt_fallback=0.05):
    """Use frame.info['time'] if available; else uniform spacing (ps)."""
    have_time = all(('time' in f.info) for f in frames)
    if have_time:
        t = np.array([float(f.info['time']) for f in frames], dtype=float)
        t -= t[0]
    else:
        t = np.arange(len(frames), dtype=float) * float(dt_fallback)
    return t

def unwrap_positions(frames, sel_idx):
    """
    Unwrap positions across PBC step-to-step (NPT-safe via MIC).
    Returns (T, M, 3) in Å.
    """
    T = len(frames)
    M = len(sel_idx)
    pos_unwrap = np.zeros((T, M, 3), dtype=float)
    pos_unwrap[0] = frames[0].get_positions()[sel_idx]
    print("Unwrapping positions...")
    for t in tqdm(range(1, T)):
        curr = frames[t].get_positions()[sel_idx]
        prev = frames[t-1].get_positions()[sel_idx]
        cell = frames[t].get_cell()
        pbc  = frames[t].get_pbc()
        disp_mic, _ = find_mic(curr - prev, cell, pbc=pbc)
        pos_unwrap[t] = pos_unwrap[t-1] + disp_mic
    return pos_unwrap

def msd_time_origin(unwrapped):
    """
    Time-origin averaged MSD over atoms and time origins.
    Returns (msd, se) with shapes (T,).
    """
    T, M, _ = unwrapped.shape
    max_lag = T - 1
    msd = np.zeros(max_lag + 1, dtype=float)
    var = np.zeros_like(msd)
    print("Computing MSD...")
    for lag in tqdm(range(max_lag + 1)):
        d = unwrapped[lag:] - unwrapped[:T-lag]      # (T-lag, M, 3)
        dr2 = np.sum(d**2, axis=2).reshape(-1)       # flatten
        msd[lag] = dr2.mean()
        var[lag] = dr2.var(ddof=1) / max(1, dr2.size)
    se = np.sqrt(var)
    return msd, se

def round_sig(x, n=4):
    """Round to n significant figures and return float (for nice CSV)."""
    if x == 0 or not np.isfinite(x):
        return float(x)
    return float(f"{x:.{n}g}")

# ───────────────────── aggregate plotting setup ──────────────────────────────
n_sys = len(traj_info)
ncols = 3
nrows = (n_sys + ncols - 1) // ncols
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7), squeeze=False)

rows = []  # CSV rows

# ───────────────────── main loop ─────────────────────────────────────────────
for idx, (traj_path, solv) in enumerate(traj_info):
    ax = axes[idx // ncols, idx % ncols]

    # Read FULL trajectory (1 ns expected)
    frames = read(traj_path, index=f":{analyze_first_n_frames}")
    # subsample frames for every 5 steps
    frames = frames[::5]
    print(f"Subsampled frames for every 5 steps")
    print(f"Read and subsampled {len(frames)} frames")
    print(f"Analyzing first {analyze_first_n_frames} frames")
    if len(frames) < 5:
        print(f"[WARN] {solv}: too few frames, skipping.")
        ax.set_visible(False)
        continue

    # Build time array and locate 100 ps cut
    times_ps = build_time_array(frames, dt_fallback=dt_fallback)
    dt_ps = float(np.median(np.diff(times_ps)))
    if not (dt_ps > 0):
        print(f"[WARN] {solv}: non-increasing time stamps; skipping.")
        ax.set_visible(False)
        continue

    # Index at which time >= 100 ps
    i_eq = int(np.searchsorted(times_ps, EQ_TIME_PS, side="left"))
    if i_eq >= len(frames) - 5:
        print(f"[WARN] {solv}: not enough frames after 100 ps; skipping.")
        ax.set_visible(False)
        continue

    # Use ONLY frames after equilibration for MSD/D (τ = 0..~900 ps)
    frames_post = frames[i_eq:]
    # Species selection (from first post-eq frame)
    symbols0 = frames_post[0].get_chemical_symbols()
    na_idx = [i for i, s in enumerate(symbols0) if s == "Na"]
    p_idx  = [i for i, s in enumerate(symbols0) if s == "P"]

    # Unwrap on the post-eq window (fresh origin at 100 ps)
    pos_na = unwrap_positions(frames_post, na_idx)
    pos_p  = unwrap_positions(frames_post, p_idx)

    # MSDs on post-eq window; τ measured from 100 ps point
    msd_na, se_na = msd_time_origin(pos_na)
    msd_p,  se_p  = msd_time_origin(pos_p)
    tau = np.arange(msd_na.size, dtype=float) * dt_ps  # ps, τ ∈ [0, ~900]

    # save msd to dictionary
    msd_dict = {
        "frames_total": len(frames),
        "msd_na": msd_na,
        "se_na": se_na,
        "msd_p": msd_p,
        "se_p": se_p,
        "tau": tau,
        "dt_ps": dt_ps,
        "EQ_TIME_PS": EQ_TIME_PS
    }
    with open(out_dir / f"msd_dict_{solv}.pkl", "wb") as f:
        pickle.dump(msd_dict, f)

    # Fit D on entire 0..900 ps post-eq window (you can set a tau_min to skip ballistic)
    tau_fit_min_ps = 0.0  # e.g., set to 5–20 ps if you want to exclude ballistic
    fit_start = int(np.searchsorted(tau, tau_fit_min_ps, side="left"))
    slope_na, intercept_na = np.polyfit(tau[fit_start:], msd_na[fit_start:], 1)
    slope_p,  intercept_p  = np.polyfit(tau[fit_start:], msd_p[fit_start:],  1)

    D_na_A2_ps = slope_na / 6.0
    D_p_A2_ps  = slope_p  / 6.0
    # 1 Å^2/ps = 1e-4 cm^2/s
    D_na = D_na_A2_ps * 1e-4
    D_p  = D_p_A2_ps  * 1e-4

    # Store for CSV (rounded to 4 sig figs)
    rows.append({
        "solvent": solv,
        "frames_total": len(frames),
        "dt_ps": round_sig(dt_ps, 4),
        "equilibration_ps": EQ_TIME_PS,
        "window_ps": round_sig(tau[-1], 4),
        "D_Na_A2_per_ps": round_sig(D_na_A2_ps, 4),
        "D_Na_cm2_per_s": round_sig(D_na, 4),
        "D_P_A2_per_ps":  round_sig(D_p_A2_ps, 4),
        "D_P_cm2_per_s":  round_sig(D_p, 4),
    })

    # ─────────── subplot ───────────
    ax.plot(tau, msd_na, lw=1.6, label="Na MSD")
    ax.plot(tau, msd_p,  lw=1.6, label="P MSD")
    ax.fill_between(tau, msd_na - se_na, msd_na + se_na, alpha=0.20)
    ax.fill_between(tau, msd_p  - se_p,  msd_p  + se_p,  alpha=0.20)

    fit_na = intercept_na + slope_na * tau
    fit_p  = intercept_p  + slope_p  * tau
    ax.plot(tau[fit_start:], fit_na[fit_start:], linestyle="--", lw=1.2, label="Na fit")
    ax.plot(tau[fit_start:], fit_p[fit_start:],  linestyle="--", lw=1.2, label="P fit")

    ax.set_title(f"NaPF6 — {solv}")
    ax.set_xlabel("τ since 100 ps (ps)")
    ax.set_ylabel("MSD (Å$^2$)")
    ax.set_xlim(0.0, 1000.0)  # per feedback: show 0 → 1000 ps on x-axis
    ax.grid(True, linestyle=":")
    info = (f"D(Na)={D_na_A2_ps:.3f} Å²/ps\n"
            f"      ={D_na:.2e} cm²/s\n"
            f"D(P) ={D_p_A2_ps:.3f} Å²/ps\n"
            f"      ={D_p:.2e} cm²/s")
    ax.text(0.98, 0.02, info, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))
    ax.legend(frameon=False, fontsize=8)

# Hide any unused subplots (if any)
for j in range(n_sys, nrows * ncols):
    axes[j // ncols, j % ncols].set_visible(False)

fig.suptitle("MSD & Diffusion — post-eq window (100→1000 ps) — 50 fs/frame", fontsize=14)
fig.tight_layout(rect=(0, 0.03, 1, 0.97))
combined_png = out_dir / "msd_all_systems.png"
fig.savefig(combined_png, dpi=300)
plt.close(fig)
print(f"Saved combined plot → {combined_png}")

# ─────────── one summary CSV ───────────
if rows:
    df = pd.DataFrame(rows)
    csv_path = out_dir / "diffusion_coefficients_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")
    print(df.to_string(index=False))
else:
    print("No results written (no frames or no trajectories).")
