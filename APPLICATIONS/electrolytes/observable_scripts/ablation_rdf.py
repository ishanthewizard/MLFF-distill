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
traj_paths = {
    "equilibrated": "/projects/beye/iamin/distillation_project/distilled_trajs/naotf_dme_equilibrated.traj",
    "unequilibrated": "/projects/beye/iamin/distillation_project/distilled_trajs/naotf_dme_UNequilibrated.traj",
    "uma": "/projects/beye/iamin/distillation_project/500ps_data_ckpt_323K/md_omol_naotf_dme_s1p1_omol/md_omol_naotf_dme_s1p1_omol.traj"
}
out_root = Path("/projects/beye/iamin/distillation_project/distilled_trajs/observables")

T = 298.15
RkJ = 8.314462618e-3
r_max, dr = 15.0, 0.05
bins = np.arange(0.0, r_max + dr, dr)
r_mid = 0.5 * (bins[:-1] + bins[1:])
shell_vol = 4.0 / 3.0 * pi * (bins[1:]**3 - bins[:-1]**3)

# Frame timing (hard-coded)
frame_dt_ps = 0.01  # 10 fs = 0.01 ps between saved frames
skip_ps = 50.0
start_frame = int(skip_ps / frame_dt_ps)  # 5000 frames

# ───────────────────────────── Helpers ─────────────────────────────
def compute_rdf(traj_file: str, cation: str, partner: str, stride: int = 10) -> pd.DataFrame:
    traj = Trajectory(traj_file, mode="r")
    if len(traj) == 0:
        raise ValueError(f"Empty trajectory: {traj_file}")

    hist = np.zeros(len(r_mid))
    n_cat_total = 0.0
    n_part_total = 0.0
    vol_sum = 0.0
    n_frames = 0

    # Skip first 50 ps, then stride
    for at in traj[start_frame::stride]:
        syms = at.get_chemical_symbols()
        idx_cat = [i for i, s in enumerate(syms) if s == cation]
        idx_part = [i for i, s in enumerate(syms) if s == partner]
        if not idx_cat or not idx_part:
            continue

        pos = at.get_positions()
        pos_cat, pos_part = pos[idx_cat], pos[idx_part]
        cell, pbc = at.get_cell(), at.get_pbc()

        for rc in pos_cat:
            disp, _ = find_mic(pos_part - rc, cell, pbc)
            d = np.linalg.norm(disp, axis=1)
            hist += np.histogram(d, bins=bins)[0]

        n_cat_total += len(idx_cat)
        n_part_total += len(idx_part)
        vol_sum += at.get_volume()
        n_frames += 1

    if n_frames == 0:
        raise ValueError(f"No frames used after skipping {skip_ps} ps in {traj_file}")

    n_cat_avg = n_cat_total / n_frames
    n_part_avg = n_part_total / n_frames
    vol_avg = vol_sum / n_frames

    rho_part = n_part_avg / vol_avg
    counts_per_cat = hist / n_cat_total
    g_r = counts_per_cat / (rho_part * shell_vol)
    n_r = np.cumsum(counts_per_cat)

    with np.errstate(divide="ignore", invalid="ignore"):
        w_r = -RkJ * T * np.log(g_r)
    w_r[~np.isfinite(w_r)] = np.nan
    tail = max(1, int(0.9 * len(w_r)))
    w_r -= np.nanmean(w_r[tail:])

    return pd.DataFrame({"r_A": r_mid, "g_r": g_r, "n_r": n_r, "w_r_kJmol": w_r})


def save_and_plot(pair: str, results: dict):
    out_dir = out_root / pair
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save CSVs
    for label, df in results.items():
        df.to_csv(out_dir / f"RDF_{label}.csv", index=False)

    # Plot comparisons
    metrics = [
        ("g_r", r"$g(r)$"),
        ("n_r", r"$n(r)$"),
        ("w_r_kJmol", r"$w(r)$ (kJ mol$^{-1}$)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    r = results["equilibrated"]["r_A"].values

    for ax, (col, ylabel) in zip(axes, metrics):
        for label, df in results.items():
            ax.plot(r, df[col], label=label, linewidth=2)
        ax.set_title(col)
        ax.set_xlabel("r (Å)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, r_max)
        ax.grid(True, linestyle=":")
        ax.legend()

    fig.suptitle(f"Na–{pair} RDF Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / f"Na-{pair}_comparison.png", dpi=300)
    plt.close(fig)


def worker(label: str, traj: str, partner: str) -> tuple:
    df = compute_rdf(traj, cation="Na", partner=partner)
    return label, partner, df


# ───────────────────────────── Main ─────────────────────────────
if __name__ == "__main__":
    max_workers = 8
    tasks = []
    results = {"O": {}, "F": {}}

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for partner in ["O", "F"]:
            for label, traj in traj_paths.items():
                tasks.append(ex.submit(worker, label, traj, partner))

        for fut in tqdm(as_completed(tasks), total=len(tasks), desc="Computing RDFs"):
            label, partner, df = fut.result()
            results[partner][label] = df

    for partner in ["O", "F"]:
        save_and_plot(partner, results[partner])

    print(f"Done. Outputs under: {out_root}")
