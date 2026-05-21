#!/usr/bin/env python
"""Kinetic energy, potential energy, total energy, and temperature
time-series plots for student model trajectories.

One figure per system × student with four stacked subplots:
  PE (eV), KE (eV), E_total (eV), T (K)  vs simulation time (ns).

Raw samples shown as faint scatter; thick line is a rolling window average.

env: /global/homes/y/yuejian/project/MLFF-distill/yuejian/envs/fairchemV2_new
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from ase.io.trajectory import Trajectory

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.linestyle": ":"})

# ── Config ────────────────────────────────────────────────────────────────────
STUDENT_DT_FS  = 100.0          # fs per frame
N_SAMPLE       = 3000           # frames to sample (strided) for plotting
ROLL_FRAC      = 0.05           # rolling average width as fraction of N_SAMPLE

OUTPUT_DIR = Path("/global/homes/y/yuejian/project/MLFF-distill/yuejian/3d_turbulence/rdf/sliding_window")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Paths (same as sliding-window script) ─────────────────────────────────────
STUDENT_ROOT = Path("/global/cfs/cdirs/m5024/distillation_project/results/diffusivity_main_results_ckpt/other_fix_run")
MICRO    = STUDENT_ROOT / "micro_trained_on_all_concentration_50ps_window"
ORIGINAL = STUDENT_ROOT / "original_trained_on_1M_only_100ps_window"

SYSTEMS = [
    {
        "name": "NaPF6/DME 0.1M",
        "conc_subpath": "20ns_solvent_0_1M",
        "temp_subpaths": {"micro": "", "original": "298K"},
        "system_dir": "md_omol_napf6_dme_re1",
        "traj_name": "md_omol_napf6_dme_re1.traj",
    },
    {
        "name": "NaOTf/DME 0.1M",
        "conc_subpath": "20ns_solvent_0_1M",
        "temp_subpaths": {"micro": "", "original": "298K"},
        "system_dir": "md_omol_naotf_dme_s1p1_omol",
        "traj_name": "md_omol_naotf_dme_s1p1_omol.traj",
    },
    {
        "name": "LiPF6/DME 0.5M",
        "conc_subpath": "20ns_solvent_solute_0.5M",
        "temp_subpaths": {"micro": "298_2K", "original": "298_2K"},
        "system_dir": "md_omol_lipf6_pfactor_0.1_1fs_mask_t",
        "traj_name": "md_omol_lipf6_pfactor_0.1_1fs_mask_t.traj",
    },
    {
        "name": "NaPF6/DME 0.5M",
        "conc_subpath": "20ns_solvent_solute_0.5M",
        "temp_subpaths": {"micro": "298_2K", "original": "298_2K"},
        "system_dir": "md_omol_napf6_dme_re1",
        "traj_name": "md_omol_napf6_dme_re1.traj",
    },
    {
        "name": "NaOTf/DME 1M",
        "conc_subpath": "20ns_solute_solvent_1M",
        "temp_subpaths": {"micro": "298K", "original": "298K"},
        "system_dir": "naotf_dme",
        "traj_name": "naotf_dme.traj",
    },
    {
        "name": "NaPF6/DME 1M",
        "conc_subpath": "20ns_solute_solvent_1M",
        "temp_subpaths": {"micro": "298K", "original": "298K"},
        "system_dir": "napf6_dme",
        "traj_name": "napf6_dme.traj",
    },
]

STUDENT_MODELS = {
    "Micro student":    (MICRO,    "micro"),
    "Original student": (ORIGINAL, "original"),
}
MODEL_COLORS = {
    "Micro student":    "#ff7f0e",
    "Original student": "#2ca02c",
}


def _build_path(root, conc_subpath, temp_subpath, system_dir, traj_name):
    p = root / conc_subpath
    if temp_subpath:
        p = p / temp_subpath
    return p / system_dir / traj_name


for sys in SYSTEMS:
    sys["student_paths"] = {}
    for model_label, (root, key) in STUDENT_MODELS.items():
        temp = sys["temp_subpaths"][key]
        sys["student_paths"][model_label] = _build_path(
            root, sys["conc_subpath"], temp, sys["system_dir"], sys["traj_name"]
        )

missing = []
for sys in SYSTEMS:
    for model, path in sys["student_paths"].items():
        if not path.exists():
            missing.append(f"  {sys['name']} / {model}: {path}")
if missing:
    print("WARNING – missing trajectories:")
    print("\n".join(missing))
else:
    print(f"All {len(SYSTEMS) * 2} student trajectory files found.\n")


# ── Rolling average ───────────────────────────────────────────────────────────

def rolling_mean(x, w):
    """Rolling average with min_periods=1 so edges use available data, no zero-pad artifact."""
    w = max(1, int(w))
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()


# ── Extract energies ──────────────────────────────────────────────────────────

def extract_energies(traj_path, n_sample=N_SAMPLE, dt_fs=STUDENT_DT_FS):
    traj   = Trajectory(str(traj_path), mode="r")
    n_tot  = len(traj)
    stride = max(1, n_tot // n_sample)
    indices = range(0, n_tot, stride)

    times, pe, ke, etot, temp = [], [], [], [], []
    for idx in tqdm(indices, desc="  frames", leave=False):
        at = traj[idx]
        if at.calc is None or "momenta" not in at.arrays:
            continue
        times.append(idx * dt_fs * 1e-6)          # fs → ns
        pe.append(at.calc.results["energy"])
        ke.append(at.get_kinetic_energy())
        etot.append(at.calc.results["energy"] + at.get_kinetic_energy())
        temp.append(at.get_temperature())

    traj.close()
    return (np.array(times), np.array(pe), np.array(ke),
            np.array(etot), np.array(temp))


# ── Plot & save ───────────────────────────────────────────────────────────────

for sys in SYSTEMS:
    sys_name = sys["name"]

    for model_label, traj_path in sys["student_paths"].items():
        if not traj_path.exists():
            print(f"SKIP (missing): {sys_name} / {model_label}")
            continue

        print(f"\n{sys_name}  |  {model_label}")
        color = MODEL_COLORS[model_label]

        times, pe, ke, etot, temp = extract_energies(traj_path)
        roll_w = max(3, int(len(times) * ROLL_FRAC))

        fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
        fig.suptitle(
            f"{sys_name}  |  {model_label}\nEnergy & Temperature vs Time",
            fontsize=13, fontweight="bold",
        )

        datasets = [
            (pe,   r"$E_{pot}$ (eV)",   "PE"),
            (ke,   r"$E_{kin}$ (eV)",   "KE"),
            (etot, r"$E_{tot}$ (eV)",   "Total E"),
            (temp, r"$T$ (K)",           "Temperature"),
        ]

        for ax, (vals, ylabel, label) in zip(axes, datasets):
            ax.scatter(times, vals, s=1.5, alpha=0.25, color=color, rasterized=True)
            ax.plot(times, rolling_mean(vals, roll_w),
                    color=color, lw=1.6, label=f"{label} (roll avg)")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=9, loc="upper right")

            mean_v = vals.mean()
            std_v  = vals.std()
            ax.axhline(mean_v, color="black", lw=0.8, ls="--", alpha=0.6)
            ax.set_ylim(mean_v - 5 * std_v, mean_v + 5 * std_v)

            # annotate mean ± std in corner
            ax.text(0.01, 0.05,
                    f"mean={mean_v:.4g}  std={std_v:.3g}",
                    transform=ax.transAxes, fontsize=8,
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

        axes[-1].set_xlabel("Simulation time (ns)")
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        safe_sys   = sys_name.replace("/", "_").replace(" ", "_")
        safe_model = model_label.replace(" ", "_")
        out_png    = OUTPUT_DIR / f"energy_{safe_sys}_{safe_model}.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_png}")

print("\nAll done.")
