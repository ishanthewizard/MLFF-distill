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
from ase.io import Trajectory
from tqdm import tqdm
# Physical constants
AMU_TO_KG       = 1.66053906660e-27   # kg per atomic mass unit
ANGSTROM3_TO_M3 = 1e-30               # Å³ → m³

# Sampling parameters
N_SNAPSHOTS = 1000   # how many snapshots to analyse
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
    import argparse
    parser = argparse.ArgumentParser(description="Compute density from ASE trajectories")
    parser.add_argument("--trajs", nargs='+', required=True, help="List of trajectory files")
    parser.add_argument("--out-dir", required=True, help="Output directory for the CSV")
    parser.add_argument("--note", type=str, default="", help="Optional note to attach to the CSV filename")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = []

    def extract_system_name(filename):
        name = os.path.basename(filename).replace(".traj", "")
        if name.startswith("md_omol_"):
            name = name[len("md_omol_"):]
        parts = name.split('_', 1)
        salt = parts[0] if len(parts) > 0 else name
        rest = parts[1] if len(parts) > 1 else ""
        
        cation = "Na" if salt.startswith("na") else "Li" if salt.startswith("li") else "Unknown"
        anion_raw = salt[2:] if cation in ["Na", "Li"] else salt
        anion = "PF6" if anion_raw == "pf6" else "OTf" if anion_raw == "otf" else anion_raw
        
        if rest.startswith("propylene_carbonate"):
            solvent = "propylene_carbonate"
        else:
            solvent = rest.split('_')[0] if rest else "Unknown"
            
        return f"{cation} - {anion} - {solvent}"

    for fname in args.trajs:
        try:
            print(f"Processing {fname}...")
            sys_name = extract_system_name(fname)
            traj = Trajectory(fname)
            mean_rho, sd_rho = compute_density_stats(traj)
            results.append({
                "system": sys_name,
                "average_density_g_cm3": float(f"{mean_rho:.4g}"),
                "std_dev": float(f"{sd_rho:.4g}")
            })
        except Exception as exc:
            print(f"[WARN] {fname}: {exc}")

    if results:
        import csv
        filename = f"simulation_density_results_{args.note}.csv" if args.note else "simulation_density_results.csv"
        out_csv = os.path.join(args.out_dir, filename)
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["system", "average_density_g_cm3", "std_dev"])
            writer.writeheader()
            writer.writerows(results)
        print(f"✅  Written {out_csv} with {len(results)} rows.")
    else:
        print("No valid results computed.")
