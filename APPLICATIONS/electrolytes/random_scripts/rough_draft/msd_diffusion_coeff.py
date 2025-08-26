#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSD and diffusion coefficients for selected species (Na and P).
- Proper time-origin averaged MSD(t) with PBC unwrapping (NPT-safe).
- Linear fit on the last half of time lags to estimate D via MSD = 6 D t.
- Saves per-system PNG plots and a CSV summary at the end.

Notes:
- "P" is used as the PF6– tracer (central atom). If you prefer true anion COM,
  say the word and I’ll switch it to PF6 COM tracking.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ase.io import read
from ase.geometry import find_mic
from tqdm import tqdm
from pathlib import Path

# ─────────────── user: trajectories & labels ─────────────────────────────────
traj_info = [
    ("/Volumes/drive_n1/nitesh_projs_Y2/LBL_proj/Battery_project_lbl/bat_esra/1M_NaPF6_DME_uma/NPT_sims/md_omol_re3.traj","s1"),
    ("/Volumes/drive_n1/nitesh_projs_Y2/LBL_proj/Battery_project_lbl/bat_esra/1M_NaPF6_DME_uma/NPT_sims/md_omol_re5_small_new_1p1.traj","s1p1"),
    ("/Volumes/drive_n1/nitesh_projs_Y2/LBL_proj/Battery_project_lbl/bat_esra/1M_NaPF6_DME_uma/NPT_sims/md_omol_re5_medium_new_1p1.traj", "m1p1"),
]

# Read last N frames (skip the final 100 if you like; change as needed)
read_slice = "-2000:"

# Fallback frame time step if frame.info['time'] missing (ps) - ps between each frame. for you saving every 50 fs, this will be 0.05 ps 
dt_fallback = 0.05

# ───────────────────── helper functions ──────────────────────────────────────
def build_time_array(frames, dt_fallback=0.01):
    # Use frame.info['time'] if present; otherwise fallback to uniform spacing.
    t = []
    have_time = all(('time' in f.info) for f in frames)
    if have_time:
        for f in frames:
            t.append(float(f.info['time']))
        t = np.asarray(t)
        # ensure monotonic & start at zero
        t = t - t[0]
    else:
        t = np.arange(len(frames)) * float(dt_fallback)
    return t

def unwrap_positions(frames, sel_idx):
    """
    Returns unwrapped positions of selected atoms across frames.
    Shape: (T, M, 3) in Å.
    Handles variable cell (NPT) using MIC step-to-step.
    """
    T = len(frames)
    M = len(sel_idx)
    pos_unwrap = np.zeros((T, M, 3), dtype=float)

    # t=0 reference
    pos_unwrap[0] = frames[0].get_positions()[sel_idx]

    # step-wise MIC increments
    for t in range(1, T):
        curr = frames[t].get_positions()[sel_idx]
        prev = frames[t-1].get_positions()[sel_idx]
        cell = frames[t].get_cell()
        pbc  = frames[t].get_pbc()

        disp_mic, _ = find_mic(curr - prev, cell, pbc=pbc)  # (M,3)
        pos_unwrap[t] = pos_unwrap[t-1] + disp_mic
    return pos_unwrap

def msd_time_origin(unwrapped):
    """
    Time-origin averaged MSD over atoms *and* time origins.
    unwrapped: (T, M, 3)
    Returns:
      tau (ps indices), msd (Å^2), stderr (Å^2) for visual bands.
    """
    T, M, _ = unwrapped.shape
    max_lag = T - 1
    msd = np.zeros(max_lag + 1, dtype=float)
    var = np.zeros_like(msd)

    # center-of-mass removal is not applied (we want tracer diffusion)
    # vectorized is tricky for varying lag; loop on lag is fine for ~2000 frames
    for lag in range(0, max_lag + 1):
        # displacements for all time-origins that fit this lag
        # shape: (T-lag, M, 3)
        d = unwrapped[lag:] - unwrapped[:T-lag]
        dr2 = np.sum(d**2, axis=2)  # (T-lag, M)
        vals = dr2.reshape(-1)
        msd[lag] = vals.mean()
        # standard error of mean for shading
        var[lag] = vals.var(ddof=1) / max(1, (vals.size))
    se = np.sqrt(var)
    return msd, se

# ───────────────────── main loop ─────────────────────────────────────────────
rows = []
out_dir = Path("msd_plots")
out_dir.mkdir(exist_ok=True)

for traj_path, system in traj_info:
    frames = read(traj_path, index=read_slice)
    if len(frames) < 5:
        print(f"[WARN] {system}: too few frames, skipping.")
        continue

    times_ps = build_time_array(frames, dt_fallback=dt_fallback)
    dt_ps = np.median(np.diff(times_ps))
    assert dt_ps > 0, "Non-increasing time stamps."

    # pick species
    symbols0 = frames[0].get_chemical_symbols()
    na_idx = [i for i, s in enumerate(symbols0) if s == "Na"]
    p_idx  = [i for i, s in enumerate(symbols0) if s == "P"]  # PF6 tracer via P

    # unwrap trajectories
    print(f"{system}: unwrapping Na ...")
    pos_na = unwrap_positions(frames, na_idx)  # (T, Nna, 3)
    print(f"{system}: unwrapping P ...")
    pos_p  = unwrap_positions(frames, p_idx)   # (T, Np,  3)

    # time-origin averaged MSDs
    print(f"{system}: computing MSDs ...")
    msd_na, se_na = msd_time_origin(pos_na)
    msd_p,  se_p  = msd_time_origin(pos_p)

    # time array for MSD lags
    tau = np.arange(msd_na.size, dtype=float) * dt_ps  # ps

    # fit D on the last half of lags (avoid ballistic regime)
    start = msd_na.size // 2
    # Robust guard: require at least ~50 points to fit
    start = min(start, max(1, msd_na.size - 50))

    slope_na, intercept_na = np.polyfit(tau[start:], msd_na[start:], 1)
    slope_p,  intercept_p  = np.polyfit(tau[start:], msd_p[start:],  1)

    D_na_A2_ps = slope_na / 6.0
    D_p_A2_ps  = slope_p  / 6.0
    # convert to cm^2/s: 1 Å^2/ps = 1e-4 cm^2/s
    D_na = D_na_A2_ps * 1e-4
    D_p  = D_p_A2_ps  * 1e-4

    print(f"\n[{system}] D(Na) = {D_na_A2_ps:.4f} Å²/ps  → {D_na:.4e} cm²/s")
    print(f"[{system}] D(P)  = {D_p_A2_ps:.4f} Å²/ps  → {D_p:.4e} cm²/s\n")

    rows.append({
        "system": system,
        "frames_used": len(frames),
        "dt_ps": dt_ps,
        "D_Na_A2_per_ps": D_na_A2_ps,
        "D_Na_cm2_per_s": D_na,
        "D_P_A2_per_ps":  D_p_A2_ps,
        "D_P_cm2_per_s":  D_p
    })

    # ─────────── plot ───────────
    plt.figure(figsize=(4.2, 3.2))
    # MSD
    plt.plot(tau, msd_na, lw=1.8, label="Na MSD")
    plt.plot(tau, msd_p,  lw=1.8, label="P MSD")
    # simple shaded 1σ error bands (time-origin SEM)
    plt.fill_between(tau, msd_na - se_na, msd_na + se_na, alpha=0.25)
    plt.fill_between(tau, msd_p  - se_p,  msd_p  + se_p,  alpha=0.25)

    # show fitted line segments (last-half)
    fit_na = intercept_na + slope_na * tau
    fit_p  = intercept_p  + slope_p  * tau
    plt.plot(tau[start:], fit_na[start:], linestyle="--", lw=1.2, label="Na fit")
    plt.plot(tau[start:], fit_p[start:],  linestyle="--", lw=1.2, label="P fit")

    plt.xlabel("τ (ps)")
    plt.ylabel("MSD (Å$^2$)")
    plt.title(f"MSD: {system}")
    plt.legend(frameon=False)
    plt.tight_layout()
    out_png = out_dir / f"msd_{system}.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved plot → {out_png}")

# ─────────── summary CSV ───────────
if rows:
    df = pd.DataFrame(rows)
    df.to_csv("diffusion_coefficients_summary.csv", index=False)
    print("\nWrote diffusion_coefficients_summary.csv")
    print(df.to_string(index=False))
else:
    print("No results written (no frames or no trajectories).")
