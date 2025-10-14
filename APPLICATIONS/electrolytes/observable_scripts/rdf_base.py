#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base RDF computation script for a single trajectory.

Usage:
    python rdf_base.py <traj_path> <cation_atom> <partner_atom> [output_csv]

Example:
    python rdf_base.py /path/to/trajectory.traj Na O output.csv
    python rdf_base.py /path/to/trajectory.traj Na F  # saves as Na-F_rdf.csv

This script computes radial distribution function (RDF), coordination number n(r), 
and potential of mean force (PMF) w(r) for a given cation-partner atom pair.
"""

import argparse
import sys
from math import pi
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io.trajectory import Trajectory
from ase.geometry import find_mic
from tqdm import tqdm

# ─────────────────────────── Global constants ───────────────────────────
T = 298.15                       # K
RkJ = 8.314462618e-3             # kJ mol^-1 K^-1
r_max, dr = 15.0, 0.05           # Å
bins = np.arange(0.0, r_max + dr, dr)
r_mid = 0.5 * (bins[:-1] + bins[1:])
shell_vol = 4.0 / 3.0 * pi * (bins[1:]**3 - bins[:-1]**3)


def resolve_traj_path(traj_path: str) -> str:
    """
    Accepts either a .traj file path or a directory containing .traj files.
    Returns a concrete .traj path (most recently modified if multiple).
    Raises FileNotFoundError if not found.
    """
    pth = Path(traj_path)
    if pth.is_file() and pth.suffix == ".traj":
        return str(pth)
    if pth.is_dir():
        trajs = sorted(pth.glob("*.traj"), key=lambda x: x.stat().st_mtime, reverse=True)
        if trajs:
            return str(trajs[0])
    # try with a common default filename inside the directory
    candidate = pth / "traj.traj"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(f"No .traj file found for: {traj_path}")


def compute_rdf_for_pair(traj_path: str, cation_atom: str, partner_atom: str):
    """
    Stream trajectory and accumulate histogram for cation-partner distances.
    Returns DataFrame with r_A, g_r, n_r, w_r_kJmol.
    
    Args:
        traj_path: Path to trajectory file or directory
        cation_atom: Symbol of cation atom (e.g., 'Na')
        partner_atom: Symbol of partner atom (e.g., 'O', 'F')
    
    Returns:
        pandas.DataFrame with columns: r_A, g_r, n_r, w_r_kJmol
    """
    traj_file = resolve_traj_path(traj_path)
    traj = Trajectory(traj_file, mode="r")
    
    if len(traj) == 0:
        raise ValueError(f"Empty trajectory: {traj_file}")

    print(f"Computing RDF for {cation_atom}-{partner_atom} from {len(traj)} frames...")
    print(f"Using trajectory: {traj_file}")
    
    hist = np.zeros(len(r_mid), dtype=np.float64)
    n_cat_total = 0.0
    n_part_total = 0.0
    vol_sum = 0.0
    n_frames_used = 0

    # Process every 10th frame for efficiency
    for at in tqdm(traj[::10], desc="Processing frames"):
        syms = at.get_chemical_symbols()
        idx_cat = [i for i, s in enumerate(syms) if s == cation_atom]
        idx_part = [i for i, s in enumerate(syms) if s == partner_atom]
        
        if not idx_cat or not idx_part:
            continue

        pos = at.get_positions()
        pos_cat = pos[idx_cat]
        pos_part = pos[idx_part]
        cell, pbc = at.get_cell(), at.get_pbc()

        # accumulate pair distances (minimum image)
        for rc in pos_cat:
            disp, _ = find_mic(pos_part - rc, cell, pbc)
            d = np.linalg.norm(disp, axis=1)
            hist += np.histogram(d, bins=bins)[0]

        n_cat_total += float(len(idx_cat))
        n_part_total += float(len(idx_part))
        vol_sum += float(at.get_volume())
        n_frames_used += 1

    if n_frames_used == 0:
        raise ValueError("No usable frames found in trajectory.")

    n_cat_avg = n_cat_total / n_frames_used
    n_part_avg = n_part_total / n_frames_used
    vol_avg = vol_sum / n_frames_used

    print(f"Average counts: {cation_atom}={n_cat_avg:.2f}, {partner_atom}={n_part_avg:.2f}")
    print(f"Average volume: {vol_avg:.2f} Å³")
    print(f"Frames used: {n_frames_used}")

    if n_cat_avg <= 0 or n_part_avg <= 0 or vol_avg <= 0:
        raise ValueError(f"Invalid counts/volume ({cation_atom}={n_cat_avg}, {partner_atom}={n_part_avg}, V={vol_avg}).")

    rho_partner = n_part_avg / vol_avg  # Å^-3
    counts_per_cat_per_frame = hist / n_cat_total  # normalize by total number of cations across frames
    g_r = counts_per_cat_per_frame / (rho_partner * shell_vol)

    # coordination number: cumulative neighbors per cation
    n_r = np.cumsum(counts_per_cat_per_frame)

    # PMF
    with np.errstate(divide="ignore", invalid="ignore"):
        w_r = -RkJ * T * np.log(g_r)
    w_r[~np.isfinite(w_r)] = np.nan
    tail = max(1, int(0.9 * len(w_r)))
    w_r = w_r - np.nanmean(w_r[tail:])

    df = pd.DataFrame({
        "r_A": r_mid,
        "g_r": g_r,
        "n_r": n_r,
        "w_r_kJmol": w_r
    })
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Compute RDF for cation-partner atom pair from trajectory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/trajectory.traj Na O output.csv
  %(prog)s /path/to/trajectory.traj Na F  # saves as Na-F_rdf.csv
  %(prog)s /path/to/directory Na O  # finds .traj file in directory
        """
    )
    
    parser.add_argument("traj_path", help="Path to trajectory file or directory containing .traj file")
    parser.add_argument("cation", help="Cation atom symbol (e.g., Na)")
    parser.add_argument("partner", help="Partner atom symbol (e.g., O, F)")
    parser.add_argument("output", nargs="?", help="Output CSV file (default: {cation}-{partner}_rdf.csv)")
    
    args = parser.parse_args()
    
    # Set default output filename if not provided
    if args.output is None:
        args.output = f"{args.cation}-{args.partner}_rdf.csv"
    
    try:
        # Compute RDF
        df = compute_rdf_for_pair(args.traj_path, args.cation, args.partner)
        
        # Save to CSV
        output_path = Path(args.output)
        df.to_csv(output_path, index=False)
        
        print(f"\nRDF computation completed!")
        print(f"Output saved to: {output_path.absolute()}")
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"r range: {df['r_A'].min():.2f} - {df['r_A'].max():.2f} Å")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
