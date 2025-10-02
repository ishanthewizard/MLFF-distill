#!/usr/bin/env python3
"""
Density analysis for ASE trajectories
-------------------------------------

For each *.traj file in the working directory:
  • Pick the LAST 1 000 frames, stepping backwards every 100 MD steps.
  • Compute the average density (g cm⁻³) and its sample standard deviation.
  • Write one line per trajectory to simulation_density_results.csv.

Requires: ASE, NumPy, pandas
"""

import os
import numpy as np
import pandas as pd
from ase.io import Trajectory
from tqdm import tqdm
# ------------------------------------------------------------------
# Physical constants
AMU_TO_KG       = 1.66053906660e-27   # kg per atomic mass unit
ANGSTROM3_TO_M3 = 1e-30               # Å³ → m³

# Sampling parameters
N_SNAPSHOTS = 1_000   # how many snapshots to analyse
STRIDE      = 10     # gap (in MD steps / frames) between snapshots
# ------------------------------------------------------------------


def last_snapshots(traj, n=N_SNAPSHOTS, stride=STRIDE):
    """
    Return the final `n` frames of `traj`, sampled every `stride` steps
    working backwards from the end.

    Example with n=3, stride=2, len=10  -->  frames 8, 6, 4.
    """
    tail   = traj[-n * stride:]   # slice the last n*stride frames (or whole traj)
    frames = tail[::stride]       # walk through that tail with the desired stride
    return frames[-n:]            # ensure at most n frames are returned


def compute_density_stats(traj):
    """
    Calculate mean and sample standard deviation of density (g cm⁻³)
    using the specified snapshot selection.
    """
    frames    = last_snapshots(traj)
    densities = []

    for atoms in tqdm(frames):
        mass_kg   = atoms.get_masses().sum() * AMU_TO_KG
        volume_m3 = atoms.get_volume()       * ANGSTROM3_TO_M3
        densities.append((mass_kg / volume_m3) / 1000.0)  # convert kg/m³ → g/cm³

    densities = np.asarray(densities)
    return densities.mean(), densities.std(ddof=1)        # sample SD (ddof=1)


# ------------------------------------------------------------------
if __name__ == "__main__":
    results = []
    root_path = "/projects/beye/iamin/distillation_project/escaip_trajs"
    solvents = ["dme"]
    traj_paths = [os.path.join(root_path, f"naotf_{solvent}.traj") for solvent in solvents ]

    for fname in traj_paths:
        try:
            traj = Trajectory(fname)
            mean_rho, sd_rho = compute_density_stats(traj)
            results.append({
                "system": fname[:-5],                 # strip ".traj"
                "average_density_g_cm3": float(f"{mean_rho:.4g}"),
                "std_dev": float(f"{sd_rho:.4g}")
            })
        except Exception as exc:
            print(f"[WARN] {fname}: {exc}")

    df = pd.DataFrame(results)
    plot_dir = '/projects/beye/iamin/distillation_project/escaip_trajs/observables/'
    df.to_csv(f"{plot_dir}/simulation_density_results_naotf", index=False)
    print("✅  Written simulation_density_results.csv with std_dev column.")
