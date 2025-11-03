#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel RDF comparison: equilibrated vs unequilibrated trajectories.
Computes Na–O and Na–F RDFs (g(r), n(r), PMF), saves CSVs, and
plots both trajectories on the same axes.
"""

import os
from math import pi
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from ase.io.trajectory import Trajectory
from ase.geometry import find_mic

# ───────────────────────────── Config ─────────────────────────────
# Configuration section: Define file paths, physical parameters, and analysis settings

# # Dictionary mapping trajectory labels to their file paths
# # Contains equilibrated, unequilibrated, and UMA (reference) trajectories
# traj_paths = {
#     "pert_10": "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/ablate_distillation/ablation_md_simulation/md_omol_naotf_pc_1m_s1p1_10/md_omol_naotf_pc_1m_s1p1_10.traj",
#     "w_o hessian": "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/ablate_distillation/ablation_md_simulation/md_omol_naotf_pc_1m_s1p1_undistill/md_omol_naotf_pc_1m_s1p1_undistill.traj",
# }
# # Output directory for all RDF results (CSVs and plots)
# out_root = Path("/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/ablate_distillation/rdf_output/ablate_hessian_cols")

# traj_paths = {
#     "pc": "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/ablate_distillation/ablation_md_simulation/md_omol_naotf_pc_1m_s1p1_10/md_omol_naotf_pc_1m_s1p1_10.traj",
#     "tgdme": "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/ablate_distillation/ablation_md_simulation/md_omol_naotf_tgdme_1m_s1p1_10/md_omol_naotf_tgdme_1m_s1p1_10.traj",
#     "dme": "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/ablate_distillation/ablation_md_simulation/md_omol_naotf_dme_s1p1_omol_10/md_omol_naotf_dme_s1p1_omol_10.traj",
#     "diglyme": "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/ablate_distillation/ablation_md_simulation/md_omol_naotf_diglyme_1m_s1p1_10/md_omol_naotf_diglyme_1m_s1p1_10.traj",
#     "w_o hessian": "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/ablate_distillation/ablation_md_simulation/md_omol_naotf_pc_1m_s1p1_undistill/md_omol_naotf_pc_1m_s1p1_undistill.traj",
#     "uma": "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/ablate_distillation/ablation_md_simulation/uma_naotf_pc_1m_s1p1/uma_naotf_pc_1m_s1p1.traj"
# }
# # Output directory for all RDF results (CSVs and plots)
# out_root = Path("/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/ablate_distillation/rdf_output/ablate_wwo_hessian_500ps")

# Physical constants and RDF parameters
T = 323  # Temperature in Kelvin
RkJ = 8.314462618e-3  # Gas constant in kJ/(mol·K)
r_max, dr = 15.0, 0.05  # Maximum distance (Å) and bin width (Å) for RDF calculation
bins = np.arange(0.0, r_max + dr, dr)  # Distance bins for histogram
r_mid = 0.5 * (bins[:-1] + bins[1:])  # Midpoint of each bin for plotting
shell_vol = 4.0 / 3.0 * pi * (bins[1:]**3 - bins[:-1]**3)  # Volume of each spherical shell

# Frame timing parameters (hard-coded)
frame_dt_ps = 0.01  # Time between saved frames: 10 fs = 0.01 ps
skip_ps = 50.0  # Skip first 50 ps of trajectory (equilibration period)
start_frame = int(skip_ps / frame_dt_ps)  # Frame index to start analysis (5000 frames)

# ───────────────────────────── Helpers ─────────────────────────────
def compute_rdf(traj_file: str, cation: str, partner: str, stride: int = 10, first_n_frames: int = None) -> pd.DataFrame:
    """
    Compute radial distribution function (RDF) for cation-partner pairs from trajectory.
    
    Args:
        traj_file: Path to ASE trajectory file
        cation: Chemical symbol of cation (e.g., "Na")
        partner: Chemical symbol of partner atom (e.g., "O", "F")
        stride: Frame stride for sampling (default: 10)
    
    Returns:
        DataFrame with columns: r_A (distance), g_r (RDF), n_r (coordination number), w_r_kJmol (PMF)
    """
    # Load trajectory and validate
    traj = Trajectory(traj_file, mode="r")

    if first_n_frames is not None:
        traj = traj[:first_n_frames]

    if len(traj) == 0:
        raise ValueError(f"Empty trajectory: {traj_file}")

    # Initialize accumulators for RDF calculation
    hist = np.zeros(len(r_mid))  # Distance histogram
    n_cat_total = 0.0  # Total cation count across all frames
    n_part_total = 0.0  # Total partner count across all frames
    vol_sum = 0.0  # Sum of cell volumes
    n_frames = 0  # Number of processed frames

    # Process trajectory frames: skip first 50 ps, then sample with stride
    for at in tqdm(traj[start_frame::stride], desc=f"Computing RDF for {cation}-{partner}"):
        
        # Get atomic symbols and find indices of cation and partner atoms
        syms = at.get_chemical_symbols()
        idx_cat = [i for i, s in enumerate(syms) if s == cation]
        idx_part = [i for i, s in enumerate(syms) if s == partner]
        if not idx_cat or not idx_part:
            continue

        # Get positions and cell information
        pos = at.get_positions()
        pos_cat, pos_part = pos[idx_cat], pos[idx_part]
        cell, pbc = at.get_cell(), at.get_pbc()

        # For each cation, compute distances to all partner atoms
        for rc in pos_cat:
            # Use minimum image convention for periodic boundaries
            disp, _ = find_mic(pos_part - rc, cell, pbc)
            d = np.linalg.norm(disp, axis=1)  # Distances
            hist += np.histogram(d, bins=bins)[0]  # Add to histogram

        # Accumulate counts and volume for normalization
        n_cat_total += len(idx_cat)
        n_part_total += len(idx_part)
        vol_sum += at.get_volume()
        n_frames += 1


    # Validate that we processed some frames
    if n_frames == 0:
        raise ValueError(f"No frames used after skipping {skip_ps} ps in {traj_file}")

    # Calculate averages for normalization
    n_cat_avg = n_cat_total / n_frames
    n_part_avg = n_part_total / n_frames
    vol_avg = vol_sum / n_frames

    # Compute RDF quantities
    rho_part = n_part_avg / vol_avg  # Average partner density
    counts_per_cat = hist / n_cat_total  # Normalized histogram
    g_r = counts_per_cat / (rho_part * shell_vol)  # RDF g(r)
    n_r = np.cumsum(counts_per_cat)  # Coordination number n(r)

    # Compute potential of mean force (PMF) w(r) = -kT ln(g(r))
    with np.errstate(divide="ignore", invalid="ignore"):
        w_r = -RkJ * T * np.log(g_r)
    w_r[~np.isfinite(w_r)] = np.nan  # Handle log(0) and log(negative)
    # Shift PMF to zero at large distances (last 10% of data)
    tail = max(1, int(0.9 * len(w_r)))
    w_r -= np.nanmean(w_r[tail:])

    return pd.DataFrame({"r_A": r_mid, "g_r": g_r, "n_r": n_r, "w_r_kJmol": w_r})


def save_and_plot(pair: str, results: dict, out_root: Path):
    """
    Save RDF data as CSV files and create comparison plots.
    
    Args:
        pair: Chemical symbol of partner atom (e.g., "O", "F")
        results: Dictionary mapping trajectory labels to RDF DataFrames
    """
    # Create output directory for this atom pair
    out_dir = out_root / pair
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save each trajectory's RDF data as CSV file
    for label, df in results.items():
        print(f"Saving {label} RDF data to {out_dir / f'RDF_{label}.csv'}")
        df.to_csv(out_dir / f"RDF_{label}.csv", index=False)
    print(f"RDF data saved to {out_dir}")
    # # Create comparison plots for all three RDF metrics
    # metrics = [
    #     ("g_r", r"$g(r)$"),  # Radial distribution function
    #     ("n_r", r"$n(r)$"),  # Coordination number
    #     ("w_r_kJmol", r"$w(r)$ (kJ mol$^{-1}$)"),  # Potential of mean force
    # ]
    
    # # Create 3-panel figure for side-by-side comparison
    # fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # r = results["equilibrated"]["r_A"].values  # Distance axis (same for all)

    # # Plot each metric on separate subplot
    # for ax, (col, ylabel) in zip(axes, metrics):
    #     # Overlay all trajectory types on same axes
    #     for label, df in results.items():
    #         ax.plot(r, df[col], label=label, linewidth=2)
    #     ax.set_title(col)  # Metric name as title
    #     ax.set_xlabel("r (Å)")
    #     ax.set_ylabel(ylabel)
    #     ax.set_xlim(0, r_max)  # Set x-axis range
    #     ax.grid(True, linestyle=":")
    #     ax.legend()

    # # Add overall title and save high-resolution plot
    # fig.suptitle(f"Na–{pair} RDF Comparison", fontsize=14, fontweight="bold")
    # fig.tight_layout(rect=(0, 0, 1, 0.95))  # Leave space for suptitle
    # fig.savefig(out_dir / f"Na-{pair}_comparison.png", dpi=300)
    # plt.close(fig)  # Free memory


def worker(label: str, traj: str, partner: str, cation: str = "Na", first_n_frames: int = None) -> tuple:
    """
    Worker function for parallel RDF computation.
    Called by ProcessPoolExecutor to compute RDF for one trajectory-partner combination.
    
    Args:
        label: Trajectory label (e.g., "equilibrated", "unequilibrated", "uma")
        traj: Path to trajectory file
        partner: Partner atom symbol (e.g., "O", "F")
    
    Returns:
        Tuple of (label, partner, DataFrame) for result collection
    """
    df = compute_rdf(traj, cation=cation, partner=partner, first_n_frames=first_n_frames)
    return label, partner, df


def main(cation: str = "Na",traj_paths: dict = None, out_root: Path = None, first_n_frames: int = None):
    """
    Main function to compute RDFs for multiple trajectories and atom pairs.
    Uses parallel processing to compute RDFs efficiently.
    """
    # Parallel processing setup
    max_workers = 16  # Number of parallel processes
    tasks = []  # List to store submitted tasks
    results = {"O": {}, "F": {}}  # Results organized by partner atom

    # Serial version for debug (parallel code commented out)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        # Create tasks for all combinations of partner atoms and trajectories
        for partner in ["O", "F"]:  # Oxygen and Fluorine partners
            for label, traj in traj_paths.items():  # All trajectory types
                tasks.append(ex.submit(worker, label, traj, partner, cation=cation, first_n_frames=first_n_frames))
    
        # Collect results as they complete, with progress bar
        for fut in tqdm(as_completed(tasks), total=len(tasks), desc="Computing RDFs"):
            label, partner, df = fut.result()
            results[partner][label] = df  # Store in nested dictionary

    # # Serial loop for debug
    # for partner in ["F"]:  # Oxygen and Fluorine partners
    #     for label, traj in tqdm(traj_paths.items(), desc=f"Computing RDFs for {partner}"):
    #         _, _, df = worker(label, traj, partner)
    #         results[partner][label] = df  # Store in nested dictionary
    # breakpoint()
    # Generate output files and plots for each partner atom
    for partner in ["O", "F"]:
        print(f"Saving and plotting {partner} RDFs")
        save_and_plot(partner, results[partner], out_root)

    print(f"Done. Outputs under: {out_root}")


# ───────────────────────────── Main ─────────────────────────────
if __name__ == "__main__":
    traj_paths = {
        "pert_10": "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/ablate_distillation/ablation_md_simulation/md_omol_naotf_pc_1m_s1p1_10/md_omol_naotf_pc_1m_s1p1_10.traj",
        "w_o hessian": "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/ablate_distillation/ablation_md_simulation/md_omol_naotf_pc_1m_s1p1_undistill/md_omol_naotf_pc_1m_s1p1_undistill.traj",
    }
    out_root = Path("/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/ablate_distillation/rdf_output/ablate_hessian_cols")
    main(cation="Na",traj_paths=traj_paths, out_root=out_root)