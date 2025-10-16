#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSD & diffusion (Na and P as PF6 tracer) across multiple systems.
- Discard first 100 ps (equilibration).
- Compute MSD and fit D using the following 900 ps window (τ = 0..900 ps).
- Save ONE combined PNG (2×3 subplots) and ONE CSV.

Assumes 50 fs between saved frames if frame.info['time'] missing.

OVERALL WORKFLOW:
================
1. Read MD trajectory for each solvent system
2. Build time array and identify equilibration cutoff (100 ps)
3. Select atoms of interest (Na+ ions and P atoms from PF6- anions)
4. Unwrap positions across periodic boundaries (remove PBC jumps)
5. Compute MSD using time-origin averaging: MSD(τ) = <|r(t+τ) - r(t)|²>
6. Fit diffusion coefficient D from Einstein relation: MSD(τ) = 6Dτ + C
7. Store results in CSV with both Å²/ps and cm²/s units
8. Plot MSD vs lag time with linear fit overlay for all systems

PHYSICS:
========
- MSD measures how far atoms diffuse over time lag τ
- In diffusive regime, MSD grows linearly with time: MSD ∝ τ
- Einstein relation gives: D = slope/6 (in 3D)
- D is the diffusion coefficient (mobility measure)
- Higher D = faster diffusion = more mobile ions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ase.io import read
from ase.geometry import find_mic
from pathlib import Path
from tqdm import tqdm

# ─────────────── user: solvents & trajectories ───────────────────────────────
solvents = ["diglyme", "dme", "PC", "TGDME"]
distilled_trajs = ["/global/homes/y/yuejian/project/MLFF-distill/yuejian/MD_naotf/md_omol_naotf_diglyme_1m_s1p1/md_omol_naotf_diglyme_1m_s1p1.traj",
         "/global/homes/y/yuejian/project/MLFF-distill/yuejian/MD_naotf/md_omol_naotf_dme_s1p1_omol/md_omol_naotf_dme_s1p1_omol.traj",
         "/global/homes/y/yuejian/project/MLFF-distill/yuejian/MD_naotf/md_omol_naotf_pc_1m_s1p1/md_omol_naotf_pc_1m_s1p1.traj",
         "/global/homes/y/yuejian/project/MLFF-distill/yuejian/MD_naotf/md_omol_naotf_tgdme_1m_s1p1/md_omol_naotf_tgdme_1m_s1p1.traj"]
traj_info = list(zip(distilled_trajs, solvents))  # (path, label)

# Equilibration trim and frame timing
EQ_TIME_PS   = 100.0   # discard first 100 ps
dt_fallback  = 0.01    # ps (10 fs) if frame.info['time'] is absent

# Output root
out_dir = Path("/global/homes/y/yuejian/project/MLFF-distill/yuejian/MD_naotf/msd")
out_dir.mkdir(parents=True, exist_ok=True)

# ───────────────────── helper functions ──────────────────────────────────────

def build_time_array(frames, dt_fallback=0.05):
    """
    Build time array for trajectory frames.
    
    - If frames have 'time' info: extract and normalize to start at 0
    - Otherwise: assume uniform spacing (dt_fallback ps between frames)
    Returns: time array in picoseconds
    """
    have_time = all(('time' in f.info) for f in frames)
    if have_time:
        t = np.array([float(f.info['time']) for f in frames], dtype=float)
        t -= t[0]  # normalize so first frame is at t=0
    else:
        t = np.arange(len(frames), dtype=float) * float(dt_fallback)
    return t

def unwrap_positions(frames, sel_idx):
    """
    Unwrap atomic positions across periodic boundaries.
    
    Purpose: MD trajectories wrap atoms back into the box when they cross boundaries.
    For MSD calculation, we need continuous trajectories without jumps.
    
    Method: 
    - Start with first frame positions
    - For each subsequent frame, compute displacement using minimum image convention (MIC)
    - Add displacement to previous unwrapped position
    - This handles variable cell (NPT) simulations correctly
    
    Args:
        frames: list of ASE atoms objects
        sel_idx: indices of atoms to unwrap (e.g., all Na atoms)
    
    Returns: (T, M, 3) array of unwrapped positions in Å, where T=frames, M=atoms
    """
    T = len(frames)
    M = len(sel_idx)
    pos_unwrap = np.zeros((T, M, 3), dtype=float)
    pos_unwrap[0] = frames[0].get_positions()[sel_idx]  # first frame as-is
    for t in tqdm(range(1, T)):
        curr = frames[t].get_positions()[sel_idx]
        prev = frames[t-1].get_positions()[sel_idx]
        cell = frames[t].get_cell()
        pbc  = frames[t].get_pbc()
        disp_mic, _ = find_mic(curr - prev, cell, pbc=pbc)  # MIC displacement
        pos_unwrap[t] = pos_unwrap[t-1] + disp_mic  # accumulate unwrapped position
    return pos_unwrap

def msd_time_origin(unwrapped):
    """
    Compute mean squared displacement (MSD) using time-origin averaging.
    
    Physics: MSD(τ) = <|r(t+τ) - r(t)|²>
    where <...> averages over all time origins t and all atoms.
    
    Method:
    - For each lag time τ (0 to T-1):
      * Compute displacement vectors between all pairs (t, t+τ)
      * Square and sum to get squared displacements
      * Average over all time origins and atoms
      * Also compute variance for error bars (standard error)
    
    Args:
        unwrapped: (T, M, 3) array of unwrapped positions
    
    Returns:
        msd: MSD values at each lag time, shape (T,)
        se: standard error of MSD, shape (T,)
    """
    T, M, _ = unwrapped.shape
    max_lag = T - 1
    msd = np.zeros(max_lag + 1, dtype=float)
    var = np.zeros_like(msd)
    for lag in tqdm(range(max_lag + 1)):
        d = unwrapped[lag:] - unwrapped[:T-lag]      # (T-lag, M, 3) displacement vectors
        dr2 = np.sum(d**2, axis=2).reshape(-1)       # squared distance, flattened
        msd[lag] = dr2.mean()                        # average over all origins & atoms
        var[lag] = dr2.var(ddof=1) / max(1, dr2.size)  # variance for error bar
    se = np.sqrt(var)  # standard error
    return msd, se

def round_sig(x, n=4):
    """
    Round number to n significant figures for clean CSV output.
    
    Examples:
        1234.567 → 1235 (4 sig figs)
        0.001234 → 0.001234 (4 sig figs)
    """
    if x == 0 or not np.isfinite(x):
        return float(x)
    return float(f"{x:.{n}g}")

# ───────────────────── aggregate plotting setup ──────────────────────────────
# Create a 2D grid of subplots (3 columns) to show all systems together
n_sys = len(traj_info)
ncols = 3
nrows = (n_sys + ncols - 1) // ncols  # ceiling division to fit all systems
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7), squeeze=False)

rows = []  # store results for CSV output

# ───────────────────── main loop ─────────────────────────────────────────────
# Process each trajectory and compute MSD/diffusion for Na and P atoms
for idx, (traj_path, solv) in enumerate(traj_info):
    ax = axes[idx // ncols, idx % ncols]  # get subplot for this system

    # STEP 1: Read trajectory
    frames = read(traj_path, index=":")
    if len(frames) < 5:
        print(f"[WARN] {solv}: too few frames, skipping.")
        ax.set_visible(False)
        continue

    # STEP 2: Build time array and find equilibration cutoff
    times_ps = build_time_array(frames, dt_fallback=dt_fallback)
    dt_ps = float(np.median(np.diff(times_ps)))  # median timestep between frames
    if not (dt_ps > 0):
        print(f"[WARN] {solv}: non-increasing time stamps; skipping.")
        ax.set_visible(False)
        continue

    # Find frame index where time >= 100 ps (equilibration cutoff)
    i_eq = int(np.searchsorted(times_ps, EQ_TIME_PS, side="left"))
    if i_eq >= len(frames) - 5:
        print(f"[WARN] {solv}: not enough frames after 100 ps; skipping.")
        ax.set_visible(False)
        continue

    # STEP 3: Select frames after equilibration and identify atoms
    # Only use post-equilibration frames for MSD calculation (τ = 0..~900 ps)
    frames_post = frames[i_eq:]
    
    # Find indices of Na and P atoms (P is used as PF6 tracer)
    symbols0 = frames_post[0].get_chemical_symbols()
    na_idx = [i for i, s in enumerate(symbols0) if s == "Na"]
    p_idx  = [i for i, s in enumerate(symbols0) if s == "P"]

    # STEP 4: Unwrap positions (remove periodic boundary jumps)
    # This creates continuous trajectories starting from the 100 ps point
    pos_na = unwrap_positions(frames_post, na_idx)
    pos_p  = unwrap_positions(frames_post, p_idx)

    # STEP 5: Compute MSD using time-origin averaging
    # τ (tau) is measured from the 100 ps point (frame 0 of post-eq window)
    msd_na, se_na = msd_time_origin(pos_na)
    msd_p,  se_p  = msd_time_origin(pos_p)
    tau = np.arange(msd_na.size, dtype=float) * dt_ps  # lag times in ps, τ ∈ [0, ~900]

    # STEP 6: Fit diffusion coefficient D from MSD(τ)
    # Einstein relation: MSD(τ) = 6Dτ + C (in 3D)
    # Slope of MSD vs τ gives 6D, so D = slope/6
    tau_fit_min_ps = 0.0  # start fitting from τ=0 (can set to 5-20 ps to exclude ballistic regime)
    fit_start = int(np.searchsorted(tau, tau_fit_min_ps, side="left"))
    slope_na, intercept_na = np.polyfit(tau[fit_start:], msd_na[fit_start:], 1)  # linear fit
    slope_p,  intercept_p  = np.polyfit(tau[fit_start:], msd_p[fit_start:],  1)

    # Convert slope to diffusion coefficient
    D_na_A2_ps = slope_na / 6.0  # D in Å²/ps
    D_p_A2_ps  = slope_p  / 6.0
    # Convert to cm²/s (1 Å²/ps = 1e-4 cm²/s)
    D_na = D_na_A2_ps * 1e-4
    D_p  = D_p_A2_ps  * 1e-4

    # STEP 7: Store results for CSV output (rounded to 4 significant figures)
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

    # STEP 8: Create subplot for this system
    # Plot MSD vs τ with error bars (shaded regions)
    ax.plot(tau, msd_na, lw=1.6, label="Na MSD")
    ax.plot(tau, msd_p,  lw=1.6, label="P MSD")
    ax.fill_between(tau, msd_na - se_na, msd_na + se_na, alpha=0.20)  # error bars
    ax.fill_between(tau, msd_p  - se_p,  msd_p  + se_p,  alpha=0.20)

    # Overlay linear fits to show diffusion regime
    fit_na = intercept_na + slope_na * tau
    fit_p  = intercept_p  + slope_p  * tau
    ax.plot(tau[fit_start:], fit_na[fit_start:], linestyle="--", lw=1.2, label="Na fit")
    ax.plot(tau[fit_start:], fit_p[fit_start:],  linestyle="--", lw=1.2, label="P fit")

    # Format subplot
    ax.set_title(f"NaPF6 — {solv}")
    ax.set_xlabel("τ since 100 ps (ps)")
    ax.set_ylabel("MSD (Å$^2$)")
    ax.set_xlim(0.0, 1000.0)  # show 0 → 1000 ps on x-axis
    ax.grid(True, linestyle=":")
    
    # Add text box with diffusion coefficients
    info = (f"D(Na)={D_na_A2_ps:.3f} Å²/ps\n"
            f"      ={D_na:.2e} cm²/s\n"
            f"D(P) ={D_p_A2_ps:.3f} Å²/ps\n"
            f"      ={D_p:.2e} cm²/s")
    ax.text(0.98, 0.02, info, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))
    ax.legend(frameon=False, fontsize=8)

# Hide any unused subplots (if we have fewer systems than subplot slots)
for j in range(n_sys, nrows * ncols):
    axes[j // ncols, j % ncols].set_visible(False)

# Save combined figure with all systems
fig.suptitle("MSD & Diffusion — post-eq window (100→1000 ps) — 50 fs/frame", fontsize=14)
fig.tight_layout(rect=(0, 0.03, 1, 0.97))
combined_png = out_dir / "msd_all_systems.png"
fig.savefig(combined_png, dpi=300)
plt.close(fig)
print(f"Saved combined plot → {combined_png}")

# ─────────── Save summary CSV ───────────
# Write all diffusion coefficients to a single CSV file
if rows:
    df = pd.DataFrame(rows)
    csv_path = out_dir / "diffusion_coefficients_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")
    print(df.to_string(index=False))
else:
    print("No results written (no frames or no trajectories).")
