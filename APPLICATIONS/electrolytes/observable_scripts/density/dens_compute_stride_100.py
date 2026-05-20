#!/usr/bin/env python3
"""
Density analysis for ASE trajectories
-------------------------------------

Inputs:
  • --trajs: One or more ASE .traj files.
  • --dt: Time spacing between consecutive trajectory frames, with units
    (fs/ps/ns), e.g. 100fs or 0.1ps.
  • --start-time: Time to start density analysis, with units (fs/ps/ns),
    e.g. 2ns.
  • --n-frames (optional, default 1000): Maximum number of consecutive
    frames to analyze starting from --start-time.
  • --out-dir: Directory where the output CSV is written.
  • --note (optional): Suffix used in output filename.

Selection rule:
  • start_idx = ceil(start_time / dt)
  • analyze frames in [start_idx, start_idx + n_frames), truncated if
    trajectory ends earlier.

Outputs:
  • CSV file in --out-dir:
      - simulation_density_results.csv
      - simulation_density_results_<note>.csv (if --note is provided)
  • CSV columns:
      - system
      - average_density_g_cm3
      - std_dev
  • One row per trajectory file.

Requires: ASE, NumPy, pandas
"""

import os
import re
import numpy as np
from ase.io import Trajectory
from tqdm import tqdm
# Physical constants
AMU_TO_KG       = 1.66053906660e-27   # kg per atomic mass unit
ANGSTROM3_TO_M3 = 1e-30               # Å³ → m³

# Sampling defaults
N_SNAPSHOTS = 1000   # how many snapshots to analyse
# ------------------------------------------------------------------


def parse_time_to_fs(time_str):
    """
    Parse a time string with units (fs, ps, ns) to femtoseconds.

    Examples:
        "100fs" -> 100.0
        "2ps"   -> 2000.0
        "2ns"   -> 2_000_000.0
    """
    value = time_str.strip().lower()
    unit_factors = {"fs": 1.0, "ps": 1e3, "ns": 1e6}
    for unit, factor in unit_factors.items():
        if value.endswith(unit):
            return float(value[:-len(unit)]) * factor
    raise ValueError(
        f"Unsupported time format '{time_str}'. Use values like 100fs, 2ps, 2ns."
    )


def select_frame_range(traj, dt_fs, start_time_fs, n=N_SNAPSHOTS):
    """
    Return frame index range for up to `n` consecutive frames from `start_time_fs`.

    Args:
        traj: ASE trajectory object.
        dt_fs: Time step between consecutive frames in femtoseconds.
        start_time_fs: Start time for density calculation in femtoseconds.
        n: Number of frames to include.
    """
    if dt_fs <= 0:
        raise ValueError(f"dt must be positive, got {dt_fs}.")
    if start_time_fs < 0:
        raise ValueError(f"start_time must be non-negative, got {start_time_fs}.")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}.")

    start_idx = int(np.ceil(start_time_fs / dt_fs))
    if start_idx >= len(traj):
        raise ValueError(
            f"Start frame index {start_idx} is beyond trajectory length {len(traj)}."
        )

    end_idx = min(start_idx + n, len(traj))
    return start_idx, end_idx


def compute_density_stats(traj, dt_fs, start_time_fs, n=N_SNAPSHOTS):
    """
    Calculate mean and sample standard deviation of density (g cm⁻³)
    using a user-selected time range.
    """
    start_idx, end_idx = select_frame_range(
        traj, dt_fs=dt_fs, start_time_fs=start_time_fs, n=n
    )
    densities = []

    for frame_idx in tqdm(range(start_idx, end_idx), desc="Density frames"):
        atoms = traj[frame_idx]
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
    parser.add_argument(
        "--dt",
        type=str,
        required=True,
        help="Time between consecutive frames (e.g., 100fs, 0.1ps).",
    )
    parser.add_argument(
        "--start-time",
        type=str,
        required=True,
        help="Start time for density analysis (e.g., 2ns).",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=N_SNAPSHOTS,
        help=f"Number of frames to analyze from start time (default: {N_SNAPSHOTS}).",
    )
    parser.add_argument("--note", type=str, default="", help="Optional note to attach to the CSV filename")
    args = parser.parse_args()

    dt_fs = parse_time_to_fs(args.dt)
    start_time_fs = parse_time_to_fs(args.start_time)

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

    def extract_conditions_from_path(path):
        concentration = "Unknown"
        temperature = "Unknown"
        normalized_path = os.path.normpath(path)
        path_parts = normalized_path.split(os.sep)

        # Concentration can appear as 1M, 0.5M, or 0_1M in directory names.
        conc_match = re.search(r"(\d+(?:[._]\d+)?)M", normalized_path)
        if conc_match:
            concentration = f"{conc_match.group(1).replace('_', '.')}M"

        for part in path_parts:
            temp_match = re.fullmatch(r"(\d+)(?:_2)?K", part)
            if temp_match:
                temperature = f"{temp_match.group(1)}K"
                break

        return concentration, temperature

    for fname in args.trajs:
        try:
            print(f"Processing {fname}...")
            sys_name = extract_system_name(fname)
            concentration, temperature = extract_conditions_from_path(fname)
            with Trajectory(fname, "r") as traj_reader:
                mean_rho, sd_rho = compute_density_stats(
                    traj_reader, dt_fs=dt_fs, start_time_fs=start_time_fs, n=args.n_frames
                )
            results.append({
                "traj_file": fname,
                "system": sys_name,
                "concentration": concentration,
                "temperature": temperature,
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
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "traj_file",
                    "system",
                    "concentration",
                    "temperature",
                    "average_density_g_cm3",
                    "std_dev",
                ],
            )
            writer.writeheader()
            writer.writerows(results)
        print(f"✅  Written {out_csv} with {len(results)} rows.")
    else:
        print("No valid results computed.")
