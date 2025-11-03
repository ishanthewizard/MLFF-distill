#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Na self-diffusion from VACF (Green–Kubo) for six NaPF6 solvents.
Outputs under: /projects/beye/iamin/observables/distilled_vacf

Per solvent:
  - VACF_<SOLVENT>_Na.csv                     (time_ps, VACF_A2fs2)
  - GK_integral_<SOLVENT>_Na.csv              (time_ps, GK_int_A2_per_fs, GK_int_SI_m2_s)

Combined:
  - D_summary.csv
  - VACF_and_GK_Na_all.png  (top: VACF overlays; bottom: GK integral overlays)

WHAT THIS SCRIPT DOES:
This script calculates the self-diffusion coefficient of Na+ ions in different solvents using the 
Green-Kubo method. The Green-Kubo method relates diffusion to the velocity autocorrelation function (VACF).
The diffusion coefficient D is calculated as: D = (1/3) * ∫₀^∞ ⟨v(0)·v(t)⟩ dt
where ⟨v(0)·v(t)⟩ is the velocity autocorrelation function.
"""

# Import necessary libraries for molecular dynamics analysis
from pathlib import Path
import numpy as np
import pandas as pd
from ase.io.trajectory import Trajectory  # For reading trajectory files
from ase.geometry import find_mic         # For minimum image convention (periodic boundaries)
import matplotlib.pyplot as plt          # For plotting results
from tqdm import tqdm                    # For progress bars

# ----------------------------- CONFIGURATION SECTION -----------------------------
# This section defines all the parameters and file paths used in the analysis
# Output directory where results will be saved
OUT_DIR = Path("/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/vacf/10ns_dt50fs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# List of solvent names and corresponding trajectory file paths
# Each trajectory contains MD simulation data for Na+ ions in different solvents
SOLVENTS = ["dme_distill", "dme_wo_hessian"]
TRAJS = ["/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/ablate_distillation/ablation_md_simulation_10ns/md_omol_naotf_dme_s1p1_omol_10/md_omol_naotf_dme_s1p1_omol_10.traj",
"/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/ablate_distillation/ablation_md_simulation_10ns/md_omol_naotf_dme_s1p1_omol_undistill/md_omol_naotf_dme_s1p1_omol_undistill.traj"]

# The atomic species we're analyzing (sodium ions)
SPECIES = "Na"

# Time step parameters
# Trajectories sampled every 10 fs
DT_FS_FALLBACK = 10  # fs

# Optional cropping parameters
TMAX_PS = None  # Maximum time to analyze (None = use full trajectory)

# VACF analysis parameters
TAIL_FIT = False         # Whether to fit exponential tail to VACF (usually False)
TAIL_WINDOW_PS = 1.5     # Window size for tail fitting if enabled

# ----------------------------- HELPER FUNCTIONS -----------------------------
# These functions perform the core calculations needed for VACF analysis

def unwrap_positions(frames):
    """
    PURPOSE: Unwrap atomic positions across periodic boundaries to get true trajectories
    
    WHY NEEDED: In MD simulations with periodic boundary conditions, atoms that cross 
    the simulation box boundary appear to "jump" to the opposite side. This function
    corrects for this by tracking the cumulative displacement using the minimum image 
    convention (MIC).
    
    INPUT: frames - List of ASE Atoms objects from trajectory
    OUTPUT: R - Array of shape [T, N, 3] containing unwrapped positions in Å
             T = number of time frames, N = number of atoms, 3 = x,y,z coordinates
    """
    T = len(frames)
    R = np.empty((T, len(frames[0]), 3), dtype=float)
    # Start with wrapped positions from first frame
    R[0] = frames[0].get_positions(wrap=True)
    
    # For each subsequent frame, calculate displacement and unwrap
    for i in tqdm(range(1, T), desc="Unwrapping", leave=False):
        r_prev = R[i-1]  # Previous unwrapped positions
        r_now  = frames[i].get_positions(wrap=True)  # Current wrapped positions
        cell   = frames[i].get_cell()  # Simulation box dimensions
        pbc    = frames[i].get_pbc()   # Periodic boundary conditions
        
        # Calculate displacement using minimum image convention
        disp, _ = find_mic(r_now - r_prev, cell, pbc=pbc)
        R[i] = r_prev + disp  # Add displacement to get unwrapped position
    return R

def estimate_velocities_from_positions(R, dt_fs):
    """
    PURPOSE: Calculate velocities from position data using numerical differentiation
    
    WHY NEEDED: Some trajectory files don't store velocity information, so we need to 
    estimate velocities from position changes over time.
    
    METHOD: Uses central difference formula for most points, forward/backward difference 
    for endpoints. This gives more accurate velocities than simple forward differences.
    
    INPUT: R - Position array [T, N, 3], dt_fs - time step in femtoseconds
    OUTPUT: V - Velocity array [T, N, 3] in Å/fs
    """
    V = np.zeros_like(R)
    # Central difference for interior points (more accurate)
    V[1:-1] = (R[2:] - R[:-2]) / (2.0 * dt_fs)
    # Forward difference for first point
    V[0]    = (R[1] - R[0])   / dt_fs
    # Backward difference for last point
    V[-1]   = (R[-1] - R[-2]) / dt_fs
    return V

def vacf_fft_multi(V, stride=1, max_lag=None):
    """
    PURPOSE: Calculate Velocity Autocorrelation Function (VACF) using Fast Fourier Transform
    
    WHAT IS VACF: The VACF measures how much a particle's velocity at time t correlates 
    with its velocity at time 0. It decays from 1 (perfect correlation) to 0 (no correlation)
    as time increases. The integral of VACF gives the diffusion coefficient.
    
    METHOD: Uses FFT for efficiency - much faster than direct correlation calculation.
    The Wiener-Khinchin theorem states that autocorrelation = inverse FFT of power spectrum.
    
    INPUT: V - Velocity array [T, N, 3] in Å/fs
           stride - Skip frames for efficiency (default 1)
           max_lag - Maximum time lag to calculate (default: all)
    OUTPUT: vacf - VACF values in Å^2/fs^2, averaged over all atoms and directions
    """
    V = V[::stride]  # Apply stride (skip frames if needed)
    T, N, D = V.shape  # T=time frames, N=atoms, D=dimensions (3)
    if max_lag is None:
        max_lag = T - 1

    # Remove per-particle drift (center-of-mass motion)
    # This ensures we're measuring thermal motion, not bulk flow
    Vm = V - V.mean(axis=0, keepdims=True)
    X  = Vm.reshape(T, N*D)  # Flatten to [T, N*3] for FFT

    # Calculate power spectrum using FFT
    nfft = 1 << (2*T - 1).bit_length()  # Next power of 2 for efficient FFT
    F    = np.fft.rfft(X, n=nfft, axis=0)  # Forward FFT
    S    = F * np.conjugate(F)  # Power spectrum
    ac   = np.fft.irfft(S, n=nfft, axis=0).real[:T]  # Inverse FFT → autocorrelation
    
    # Unbiased normalization (account for decreasing number of samples at longer lags)
    ac  /= (np.arange(T, 0, -1)[:, None])  # unbiased normalization
    vacf = ac.mean(axis=1)[:max_lag+1]  # Average over all atoms/directions
    return vacf

def first_zero_crossing(y):
    """
    PURPOSE: Find the first point where a function crosses zero (goes from positive to negative)
    
    WHY NEEDED: In Green-Kubo analysis, we typically integrate the VACF only up to its first 
    zero crossing, as this is where the correlation becomes negligible and noise dominates.
    
    INPUT: y - Array of function values
    OUTPUT: Index of first zero crossing, or None if no crossing found
    """
    for i in range(1, len(y)):
        if y[i-1] > 0.0 and y[i] <= 0.0:
            return i
    return None

def cumulative_trapz(y, dx):
    """
    PURPOSE: Calculate cumulative integral using trapezoidal rule
    
    WHY NEEDED: The Green-Kubo formula requires integrating the VACF from 0 to t.
    This function calculates D(t) = (1/3) * ∫₀ᵗ VACF(τ) dτ for all time points.
    
    METHOD: Uses trapezoidal rule for numerical integration, which is more accurate 
    than rectangular rule for smooth functions like VACF.
    
    INPUT: y - Function values to integrate, dx - spacing between points
    OUTPUT: Cumulative integral values
    """
    out = np.zeros_like(y, dtype=float)
    if len(y) < 2:
        return out
    # Trapezoidal rule: ∫f(x)dx ≈ Σ[0.5*(f[i]+f[i-1])*dx]
    out[1:] = np.cumsum(0.5*(y[1:] + y[:-1]) * dx)
    return out

def fit_exponential_tail(t, y):
    """
    PURPOSE: Fit exponential decay to the tail of VACF for improved diffusion calculation
    
    WHY NEEDED: Sometimes VACF doesn't decay to zero cleanly due to noise or insufficient 
    sampling. This function fits an exponential tail A*exp(-t/τ) to extend the integration
    and get a more accurate diffusion coefficient.
    
    METHOD: Linear least squares fit to log(y) = log(A) - t/τ
    
    INPUT: t - time points in ps, y - VACF values
    OUTPUT: (A, tau_ps) - amplitude and decay time, or None if fit fails
    """
    mask = y > 0  # Only fit positive values
    if mask.sum() < 5:  # Need at least 5 points for reliable fit
        return None
    t_pos = t[mask]
    y_pos = y[mask]
    logy  = np.log(y_pos)
    
    # Linear system: log(y) = log(A) - t/τ
    # Matrix form: [1, -t] * [log(A), 1/τ] = log(y)
    A = np.vstack([np.ones_like(t_pos), -t_pos]).T
    sol, *_ = np.linalg.lstsq(A, logy, rcond=None)
    logA, inv_tau = sol
    
    if inv_tau <= 0:  # Check for physical decay (positive tau)
        return None
    return np.exp(logA), 1.0 / inv_tau

# ----------------------------- MAIN PROCESSING FUNCTION -----------------------------
# This function processes one solvent system and calculates its diffusion coefficient

def process_one(solvent, traj_path):
    """
    PURPOSE: Process a single solvent system to calculate Na+ diffusion coefficient
    
    WORKFLOW:
    1. Load trajectory frames
    2. Extract Na+ atom positions
    3. Unwrap positions across periodic boundaries
    4. Calculate velocities (from file or estimate from positions)
    5. Compute VACF using FFT
    6. Integrate VACF to get diffusion coefficient
    7. Save results to CSV files
    
    INPUT: solvent - name of solvent, traj_path - path to trajectory file
    OUTPUT: Dictionary with analysis results and data for plotting
    """
    
    # STEP 1: Load trajectory frames
    with Trajectory(traj_path) as tr:
        frames = [at.copy() for at in tr]  # Copy to avoid memory issues
    if len(frames) < 8:
        raise RuntimeError(f"{solvent}: not enough frames")

    # STEP 2: Identify Na+ atoms in the system
    sym = np.array(frames[0].get_chemical_symbols())  # Get atomic symbols
    idx = np.where(sym == SPECIES)[0]  # Find indices of Na atoms
    if idx.size == 0:
        raise RuntimeError(f"{solvent}: no {SPECIES} atoms found")

    # STEP 3: Set up time parameters
    times_fs = None  # Not used in this version
    dt_fs    = DT_FS_FALLBACK  # Time step for velocity estimation

    # STEP 4: Optional cropping by physical time (if TMAX_PS is set)
    if TMAX_PS is not None and times_fs is not None:
        T_keep = int(min(len(frames), np.floor(TMAX_PS * 1e3 / dt_fs)))
        frames = frames[:T_keep]
        times_fs = times_fs[:T_keep]

    # STEP 5: Unwrap positions and extract velocities
    R = unwrap_positions(frames)[:, idx, :]  # Unwrap positions for Na atoms only
    
    # Try to get velocities from trajectory file first
    try:
        V_all = np.stack([at.get_velocities() for at in frames], axis=0)[:, idx, :]
        if np.isnan(V_all).any():  # Check for invalid velocities
            raise ValueError
        V = V_all  # Use velocities from file (in Å/fs)
    except Exception:
        # If velocities not available or invalid, estimate from positions
        V = estimate_velocities_from_positions(R, dt_fs)

    # STEP 6: Calculate VACF using FFT method
    vacf = vacf_fft_multi(V, stride=1)  # VACF in Å^2/fs^2
    dt_eff_fs = dt_fs  # Effective time step
    t_ps = np.arange(len(vacf)) * dt_eff_fs * 1e-3  # Time axis in picoseconds

    # STEP 7: Find first zero crossing of VACF
    iz = first_zero_crossing(vacf)
    if iz is None:
        iz = len(vacf) - 1  # Use last point if no zero crossing found
    t_zero_ps = float(t_ps[iz])

    # STEP 8: Calculate Green-Kubo integral (diffusion coefficient)
    # D = (1/3) * ∫₀ᵗ VACF(τ) dτ
    GK_A2_per_fs = (1.0/3.0) * cumulative_trapz(vacf, dx=dt_eff_fs)   # Å^2/fs
    GK_SI_m2_s   = GK_A2_per_fs * 1e-5                                # m^2/s (Å^2/fs → m^2/s)
    D_base_SI    = GK_SI_m2_s[iz]  # Diffusion coefficient at zero crossing
    note = "integral to first zero crossing"
    use_tail = False

    # STEP 9: Optional exponential tail fitting (usually disabled)
    if TAIL_FIT and iz < len(vacf) - 5:
        t0 = t_ps[iz]  # Start of tail region
        w  = float(TAIL_WINDOW_PS)  # Window size for fitting
        mask = (t_ps >= t0) & (t_ps <= min(t0 + w, t_ps[-1]))
        fit = fit_exponential_tail(t_ps[mask] - t0, vacf[mask])
        if fit is not None:
            A, tau_ps = fit
            # Add tail contribution to diffusion coefficient
            # VACF units Å^2/fs^2; tau in ps → fs
            D_tail_SI = (1.0/3.0) * A * (tau_ps * 1e3) * 1e-5
            D_base_SI += D_tail_SI
            note += f" + exp tail ({w:.2f} ps)"
            use_tail = True
        else:
            note += " (tail fit failed)"

    # STEP 10: Save results to CSV files
    pd.DataFrame({"time_ps": t_ps, "VACF_A2fs2": vacf}).to_csv(
        OUT_DIR / f"VACF_{solvent}_{SPECIES}.csv", index=False)
    pd.DataFrame({
        "time_ps": t_ps,
        "GK_int_A2_per_fs": GK_A2_per_fs,
        "GK_int_SI_m2_s": GK_SI_m2_s
    }).to_csv(OUT_DIR / f"GK_integral_{solvent}_{SPECIES}.csv", index=False)

    # STEP 11: Return results dictionary for summary and plotting
    return {
        "solvent": solvent,                    # Solvent name
        "dt_fs": dt_eff_fs,                   # Time step used
        "n_frames": len(frames),               # Number of trajectory frames
        f"n_{SPECIES}": int(idx.size),         # Number of Na+ atoms
        "t_max_ps": float(t_ps[-1]),          # Maximum time analyzed
        "t_zero_ps": float(t_zero_ps),         # Time of first VACF zero crossing
        "use_tail": use_tail,                  # Whether exponential tail was used
        "D_SI_m2_s": float(D_base_SI),        # Diffusion coefficient in m²/s
        "D_cm2_s": float(D_base_SI * 1e4),    # Diffusion coefficient in cm²/s
        "note": note,                          # Description of calculation method
        "t_ps": t_ps,                         # Time axis for plotting
        "vacf": vacf,                         # VACF values for plotting
        "GK_SI_m2_s": GK_SI_m2_s,            # GK integral values for plotting
    }

# ----------------------------- MAIN EXECUTION AND PLOTTING -----------------------------
# This section runs the analysis for all solvents and creates summary plots

if __name__ == "__main__":
    """
    MAIN EXECUTION BLOCK:
    1. Process each solvent system
    2. Collect results
    3. Save summary CSV
    4. Create combined plots
    """
    
    # Process all solvent systems
    results = []
    for s, p in zip(SOLVENTS, TRAJS):
        try:
            print(f"Processing {s} ...")
            res = process_one(s, p)
            results.append(res)
            print(f"  D = {res['D_cm2_s']:.4e} cm^2/s  ({res['note']})")
        except Exception as e:
            print(f"[WARN] {s}: {e}")

    if results:
        # STEP 1: Save summary CSV with diffusion coefficients
        df = pd.DataFrame([{k: v for k, v in r.items()
                            if k not in ("t_ps", "vacf", "GK_SI_m2_s")} for r in results])
        df.to_csv(OUT_DIR / "D_summary.csv", index=False)
        print(f"Saved {OUT_DIR / 'D_summary.csv'}")

        # STEP 2: Create combined plots showing VACF and GK integrals for all solvents
        plt.figure(figsize=(8.5, 7.0))

        # TOP PLOT: VACF curves for all solvents (first 2 ps only)
        ax1 = plt.subplot(2, 1, 1)
        for r in results:
            mask = r["t_ps"] <= 2.0  # Only show first 2 ps for clarity
            ax1.plot(r["t_ps"][mask], r["vacf"][mask], lw=1.6, label=r["solvent"])
        ax1.set_xlim(0, 2.0)
        ax1.set_xlabel("time (ps)")
        ax1.set_ylabel(r"VACF (Å$^2$/fs$^2$)")
        ax1.set_title(f"Na VACF — {len(results)} systems (first 2 ps)")
        ax1.grid(True, linestyle=":")
        ax1.legend(ncol=3, fontsize=8)

        # BOTTOM PLOT: Green-Kubo integral curves (diffusion coefficient vs time)
        ax2 = plt.subplot(2, 1, 2)
        for r in results:
            pos = r["GK_SI_m2_s"] > 0  # Only plot positive values
            ax2.plot(r["t_ps"][pos], r["GK_SI_m2_s"][pos], lw=1.6, label=r["solvent"])
        ax2.set_yscale("log")  # Log scale for better visualization
        ax2.set_xlabel("time (ps)")
        ax2.set_ylabel(r"$D(t)$ from GK integral (m$^2$/s)")
        ax2.set_title("Green–Kubo integral (positive part; log scale)")
        ax2.grid(True, linestyle=":")

        # Save the combined plot
        plt.tight_layout()
        out_png = OUT_DIR / "VACF_and_GK_Na_all.png"
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"Saved combined plot → {out_png}")
    else:
        print("No successful results.")
