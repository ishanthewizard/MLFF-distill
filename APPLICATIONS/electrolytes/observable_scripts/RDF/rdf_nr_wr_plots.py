#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute distilled RDF CSVs for Na–O and Na–F with parallel workers,
then plot UMA vs Distilled curves per solvent, pair (O/F), and observable.

Output folder layout (under distilld_csv_folder):
  <base>/
    O/
      csv/                       # distilled CSVs for Na–O
      g_r/                       # per-solvent g(r) plots (UMA vs distilled, MAE in legend)
      n_r/
      pmf/
    F/
      csv/                       # distilled CSVs for Na–F
      g_r/
      n_r/
      pmf/

For each solvent (DME, DG, DMC, TGDME, PC, THF), and for each pair (Na–O, Na–F),
we save distilled CSVs named: RDF_<SOLVENT>_Na-<PAIR>.csv
Columns: r_A, g_r, n_r, w_r_kJmol
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

# ─────────────────────────── User-provided inputs ────────────────────────────
solvents = ["DME", "DG", "DMC", "TGDME", "PC", "THF"]
distilled_trajs = [f"/projects/beye/iamin/trajs/napf6_{s}_1ns.traj" for s in solvents]
uma_csvs_na_F = [f"/projects/beye/iamin/observables/uma_csvs/rdf_csvs_na_F/RDF_{s}_Na-F.csv" for s in solvents]
uma_csvs_na_O = [f"/projects/beye/iamin/observables/uma_csvs/rdf_csvs_na_o/RDF_{s}_Na-O.csv" for s in solvents]

# NOTE: variable name as provided (typo preserved intentionally)
distilld_csv_folder = "/projects/beye/iamin/observables/distilled_csvs"

# ─────────────────────────── Global constants/bins ───────────────────────────
T = 298.15                       # K
RkJ = 8.314462618e-3             # kJ mol^-1 K^-1
r_max, dr = 15.0, 0.05           # Å
bins = np.arange(0.0, r_max + dr, dr)
r_mid = 0.5 * (bins[:-1] + bins[1:])
shell_vol = 4.0 / 3.0 * pi * (bins[1:]**3 - bins[:-1]**3)

# ───────────────────────────── Utility helpers ───────────────────────────────
def resolve_traj_path(p: str) -> str:
    """
    Accepts either a .traj file path or a directory containing .traj files.
    Returns a concrete .traj path (most recently modified if multiple).
    Raises FileNotFoundError if not found.
    """
    pth = Path(p)
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
    raise FileNotFoundError(f"No .traj file found for: {p}")

def mae(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return float("nan")
    return float(np.nanmean(np.abs(a[mask] - b[mask])))

# ───────────────────── RDF computation (single trajectory) ───────────────────
def compute_rdf_for_pair(traj_path: str, cation_atom: str, partner_atom: str):
    """
    Stream trajectory and accumulate histogram for cation-partner distances.
    Returns DataFrame with r_A, g_r, n_r, w_r_kJmol.
    """
    traj = Trajectory(traj_path, mode="r")
    if len(traj) == 0:
        raise ValueError(f"Empty trajectory: {traj_path}")

    hist = np.zeros(len(r_mid), dtype=np.float64)
    n_cat_total = 0.0
    n_part_total = 0.0
    vol_sum = 0.0
    n_frames_used = 0

    for at in tqdm(traj[::10]):
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

    if n_cat_avg <= 0 or n_part_avg <= 0 or vol_avg <= 0:
        raise ValueError(f"Invalid counts/volume (Na={n_cat_avg}, X={n_part_avg}, V={vol_avg}).")

    rho_partner = n_part_avg / vol_avg  # Å^-3
    counts_per_cat_per_frame = hist / (n_cat_total)  # normalize by total number of cations across frames
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

def _worker_compute_and_save(traj_in: str, pair_atom: str, out_csv: str):
    """
    Worker function: compute RDF for Na–<pair_atom> and save CSV to out_csv.
    Returns out_csv on success.
    """
    traj_file = resolve_traj_path(traj_in)
    df = compute_rdf_for_pair(traj_file, cation_atom="Na", partner_atom=pair_atom)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return str(out_path)

# ───────────────────────── Batch compute (parallel) ──────────────────────────
def batch_compute_distilled_csvs(solvent_list, traj_list, base_out_dir: Path, max_workers: int = 12):
    """
    For both Na–O and Na–F, compute distilled CSVs in parallel.
    Saves to:
      base_out_dir / 'O' / 'csv' / RDF_<SOLVENT>_Na-O.csv
      base_out_dir / 'F' / 'csv' / RDF_<SOLVENT>_Na-F.csv
    """
    pair_map = {"O": "O", "F": "F"}

    for pair_key, partner in pair_map.items():
        out_csv_dir = base_out_dir / pair_key / "csv"
        out_csv_dir.mkdir(parents=True, exist_ok=True)

        tasks = []
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for solv, traj in zip(solvent_list, traj_list):
                out_csv = out_csv_dir / f"RDF_{solv}_Na-{pair_key}.csv"
                tasks.append(ex.submit(_worker_compute_and_save, traj, partner, str(out_csv)))

            for _ in tqdm(as_completed(tasks), total=len(tasks), desc=f"Computing distilled CSVs Na–{pair_key}"):
                pass  # we just show progress; errors will raise on result() if needed

        # force exception propagation if any
        for t in tasks:
            _ = t.result()

# ─────────────────────────────── Plotting utils ──────────────────────────────
def load_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    df = pd.read_csv(path)
    need = {"r_A", "g_r", "n_r", "w_r_kJmol"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} missing columns; found {df.columns.tolist()}, need {sorted(need)}")
    return df

def align_on_r(xr: np.ndarray, y_old_r: np.ndarray, y_old: np.ndarray) -> np.ndarray:
    """
    Interpolate y_old (defined on y_old_r) onto xr.
    """
    # Ensure finite for interpolation bounds
    mask = np.isfinite(y_old_r) & np.isfinite(y_old)
    if np.sum(mask) < 2:
        return np.full_like(xr, np.nan, dtype=float)
    return np.interp(xr, y_old_r[mask], y_old[mask], left=np.nan, right=np.nan)

def make_plots_combined(solvent_list, uma_paths_F, uma_paths_O, base_out_dir: Path, dpi: int = 300):
    """
    For each solvent, create a single 6-panel figure:
      rows = pair atoms: [Na–O, Na–F]
      cols = metrics: [g(r), n(r), PMF w(r)]
    Saves PNGs under:
      base_out_dir / "combined_plots" / NaPF6-<SOLVENT>_six_plots.png
    """
    # Build mapping solvent -> UMA csv for each pair
    uma_map = {
        "F": {s: Path(p) for s, p in zip(solvent_list, uma_paths_F)},
        "O": {s: Path(p) for s, p in zip(solvent_list, uma_paths_O)},
    }

    # metric name -> (column, human title, y-label)
    metrics = [
        ("g_r", "g(r)", r"$g(r)$"),
        ("n_r", "n(r)", r"$n(r)$"),
        ("w_r_kJmol", "w(r)", r"$w(r)$ (kJ mol$^{-1}$)"),
    ]
    pairs = [("O", "Na–O"), ("F", "Na–F")]

    out_root = base_out_dir / "combined_plots"
    out_root.mkdir(parents=True, exist_ok=True)

    for solv in solvent_list:
        # Load distilled CSVs for both pairs
        dist_paths = {
            pk: base_out_dir / pk / "csv" / f"RDF_{solv}_Na-{pk}.csv"
            for pk, _ in pairs
        }
        dist_dfs = {pk: load_csv_safe(path) for pk, path in dist_paths.items()}

        # Load UMA CSVs for both pairs
        uma_dfs = {pk: load_csv_safe(uma_map[pk][solv]) for pk, _ in pairs}

        # Prepare figure: 2 rows (O,F) × 3 cols (g_r, n_r, pmf)
        fig, axes = plt.subplots(
            nrows=2, ncols=3, figsize=(12, 7), sharex="col"
        )

        for r_idx, (pair_key, pair_label) in enumerate(pairs):
            dist_df = dist_dfs[pair_key]
            uma_df = uma_dfs[pair_key]

            # r-grid from distilled; align UMA to it
            r = np.asarray(dist_df["r_A"].values, dtype=float)
            uma_r = np.asarray(uma_df["r_A"].values, dtype=float)

            uma_aligned = {
                "g_r": align_on_r(r, uma_r, np.asarray(uma_df["g_r"].values, dtype=float)),
                "n_r": align_on_r(r, uma_r, np.asarray(uma_df["n_r"].values, dtype=float)),
                "w_r_kJmol": align_on_r(r, uma_r, np.asarray(uma_df["w_r_kJmol"].values, dtype=float)),
            }

            for c_idx, (col, mtitle, ylabel) in enumerate(metrics):
                ax = axes[r_idx, c_idx]
                y_dist = np.asarray(dist_df[col].values, dtype=float)
                y_uma = uma_aligned[col]

                the_mae = mae(y_uma, y_dist)

                # Plot UMA (dashed) and Distilled (solid)
                ax.plot(r, y_uma, linestyle="--", linewidth=2.0, label="UMA")
                ax.plot(r, y_dist, linestyle="-", linewidth=2.0, label=f"Distilled (MAE={the_mae:.3f})")

                # Titles/labels/grid
                ax.set_title(f"{pair_label} — {mtitle}", fontsize=11)
                ax.set_ylabel(ylabel)
                ax.set_xlim(0.0, r_max)
                if col != "w_r_kJmol":
                    ax.set_ylim(bottom=0)
                ax.grid(True, linestyle=":")
                ax.legend(frameon=True, fontsize=9)

                # Only bottom row gets x-labels to reduce clutter
                if r_idx == len(pairs) - 1:
                    ax.set_xlabel("r (Å)")

        fig.suptitle(f"NaPF6 — {solv}", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=(0, 0.03, 1, 0.96))

        out_png = out_root / f"NaPF6-{solv}_six_plots.png"
        fig.savefig(out_png, dpi=dpi)
        plt.close(fig)

# ────────────────────────────────── Main ─────────────────────────────────────
if __name__ == "__main__":
    base_out = Path(distilld_csv_folder)

    # 1) Compute distilled CSVs in parallel for both Na–O and Na–F
    # batch_compute_distilled_csvs(
    #     solvent_list=solvents,
    #     traj_list=distilled_trajs,
    #     base_out_dir=base_out,
    #     max_workers=12,
    # )

    # 2) Generate per-solvent plots (UMA vs Distilled) with MAE in legend
    make_plots_combined(
        solvent_list=solvents,
        uma_paths_F=uma_csvs_na_F,
        uma_paths_O=uma_csvs_na_O,
        base_out_dir=base_out,
    )

    print(f"Done. Outputs under: {base_out}")
