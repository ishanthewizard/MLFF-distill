#!/usr/bin/env python
"""RDF & Density Comparison: Student Models vs. UMA Teacher

Compares radial distribution functions (g(r), n(r)) and densities between:
- UMA teacher (reference)
- Micro student (trained on all concentrations, 50 ps window)
- Original student (trained on 1M only, 100 ps window)

UMA teacher trajs are 1 ns (1 fs/frame); student trajs are 20 ns (100 fs/frame).
All models use the same absolute time window: skip first 0.1 ns, analyse 0.9 ns,
sampling 1000 frames via strided indexing to avoid loading entire files into memory.

Plots are saved to OUTPUT_DIR.
"""

import os
from math import pi
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from ase.io.trajectory import Trajectory
from ase.geometry import find_mic

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.linestyle": ":"})

# === Config ===
N_FRAMES = 3000
SKIP_NS = 0.1    # ns to skip (equilibration)
WINDOW_NS = 0.9  # ns window used for analysis
R_MAX = 10.0
DR = 0.05
MAX_WORKERS = min(32, os.cpu_count())

# Frame intervals in fs
UMA_DT_FS = 10.0
STUDENT_DT_FS = 100.0

OUTPUT_DIR = Path("/global/homes/y/yuejian/project/MLFF-distill/yuejian/3d_turbulence/rdf")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Directory roots ===
STUDENT_ROOT = Path("/global/cfs/cdirs/m5024/distillation_project/results/diffusivity_main_results_ckpt/other_fix_run")
TEACHER_ROOT = Path("/global/cfs/cdirs/m5024/distillation_project/data/raw_data_from_UMA_simulation/other_temperature_for_comparing_with_student_model")

MICRO = STUDENT_ROOT / "micro_trained_on_all_concentration_50ps_window"
ORIGINAL = STUDENT_ROOT / "original_trained_on_1M_only_100ps_window"

SYSTEMS = [
    {
        "name": "NaPF6/DME 0.1M",
        "conc_subpath": "20ns_solvent_0_1M",
        "temp_subpaths": {"teacher": "298K", "micro": "", "original": "298K"},
        "system_dir": "md_omol_napf6_dme_re1",
        "traj_name": "md_omol_napf6_dme_re1.traj",
        "rdf_pairs": [("Na", "O"), ("Na", "F")],
    },
    {
        "name": "NaOTf/DME 0.1M",
        "conc_subpath": "20ns_solvent_0_1M",
        "temp_subpaths": {"teacher": "298K", "micro": "", "original": "298K"},
        "system_dir": "md_omol_naotf_dme_s1p1_omol",
        "traj_name": "md_omol_naotf_dme_s1p1_omol.traj",
        "rdf_pairs": [("Na", "S"), ("Na", "O")],
    },
    {
        "name": "LiPF6/DME 0.5M",
        "conc_subpath": "20ns_solvent_solute_0.5M",
        "temp_subpaths": {"teacher": "298_2K", "micro": "298_2K", "original": "298_2K"},
        "system_dir": "md_omol_lipf6_pfactor_0.1_1fs_mask_t",
        "traj_name": "md_omol_lipf6_pfactor_0.1_1fs_mask_t.traj",
        "rdf_pairs": [("Li", "O"), ("Li", "F")],
    },
    {
        "name": "NaPF6/DME 0.5M",
        "conc_subpath": "20ns_solvent_solute_0.5M",
        "temp_subpaths": {"teacher": "298_2K", "micro": "298_2K", "original": "298_2K"},
        "system_dir": "md_omol_napf6_dme_re1",
        "traj_name": "md_omol_napf6_dme_re1.traj",
        "rdf_pairs": [("Na", "O"), ("Na", "F")],
    },
    {
        "name": "NaOTf/DME 1M",
        "conc_subpath": "20ns_solute_solvent_1M",
        "temp_subpaths": {"teacher": "298K", "micro": "298K", "original": "298K"},
        "system_dir": "naotf_dme",
        "traj_name": "naotf_dme.traj",
        "rdf_pairs": [("Na", "S"), ("Na", "O")],
    },
    {
        "name": "NaPF6/DME 1M",
        "conc_subpath": "20ns_solute_solvent_1M",
        "temp_subpaths": {"teacher": "298K", "micro": "298K", "original": "298K"},
        "system_dir": "napf6_dme",
        "traj_name": "napf6_dme.traj",
        "rdf_pairs": [("Na", "O"), ("Na", "F")],
    },
]

MODEL_ROOTS = {
    "UMA (teacher)": TEACHER_ROOT,
    "Micro student": MICRO,
    "Original student": ORIGINAL,
}
MODEL_KEY_MAP = {
    "UMA (teacher)": "teacher",
    "Micro student": "micro",
    "Original student": "original",
}
MODEL_COLORS = {
    "UMA (teacher)": "#1f77b4",
    "Micro student": "#ff7f0e",
    "Original student": "#2ca02c",
}
MODEL_ORDER = ["UMA (teacher)", "Micro student", "Original student"]
MODEL_DT_FS = {
    "UMA (teacher)": UMA_DT_FS,
    "Micro student": STUDENT_DT_FS,
    "Original student": STUDENT_DT_FS,
}


def _build_path(root, conc_subpath, temp_subpath, system_dir, traj_name):
    p = root / conc_subpath
    if temp_subpath:
        p = p / temp_subpath
    return p / system_dir / traj_name


# Build and validate all paths
for sys in SYSTEMS:
    sys["paths"] = {}
    for model_label, root in MODEL_ROOTS.items():
        key = MODEL_KEY_MAP[model_label]
        temp = sys["temp_subpaths"][key]
        sys["paths"][model_label] = _build_path(
            root, sys["conc_subpath"], temp, sys["system_dir"], sys["traj_name"]
        )

missing = []
for sys in SYSTEMS:
    for model, path in sys["paths"].items():
        if not path.exists():
            missing.append(f"  {sys['name']} / {model}: {path}")

if missing:
    print("WARNING - missing trajectories:")
    print("\n".join(missing))
else:
    print(f"All {len(SYSTEMS) * 3} trajectory files found.")


# === Core computation ===

def sample_frames(traj_path, dt_fs, n_frames=N_FRAMES):
    traj = Trajectory(str(traj_path), mode="r")
    start = int(SKIP_NS * 1e6 / dt_fs)
    end = start + int(WINDOW_NS * 1e6 / dt_fs)
    end = min(end, len(traj))
    stride = max(1, (end - start) // n_frames)
    return traj[start:end:stride], traj


def compute_rdf(traj_path, cation, partner, dt_fs, n_frames=N_FRAMES, r_max=R_MAX, dr=DR):
    bins = np.arange(0.0, r_max + dr, dr)
    r_mid = 0.5 * (bins[:-1] + bins[1:])
    shell_vol = 4.0 / 3.0 * pi * (bins[1:] ** 3 - bins[:-1] ** 3)

    hist = np.zeros(len(r_mid))
    n_cat_total = 0.0
    n_part_total = 0.0
    vol_sum = 0.0
    frame_count = 0

    frames, traj = sample_frames(traj_path, dt_fs, n_frames)
    try:
        for at in frames:
            syms = at.get_chemical_symbols()
            idx_cat = [i for i, s in enumerate(syms) if s == cation]
            idx_part = [i for i, s in enumerate(syms) if s == partner]
            if not idx_cat or not idx_part:
                continue

            pos = at.get_positions()
            cell, pbc = at.get_cell(), at.get_pbc()

            for rc in pos[idx_cat]:
                disp, _ = find_mic(pos[idx_part] - rc, cell, pbc)
                d = np.linalg.norm(disp, axis=1)
                hist += np.histogram(d, bins=bins)[0]

            n_cat_total += len(idx_cat)
            n_part_total += len(idx_part)
            vol_sum += at.get_volume()
            frame_count += 1
    finally:
        traj.close()

    if frame_count == 0:
        raise ValueError(f"No valid frames in {traj_path}")

    rho_part = (n_part_total / frame_count) / (vol_sum / frame_count)
    counts_per_cat = hist / n_cat_total
    g_r = counts_per_cat / (rho_part * shell_vol)
    n_r = np.cumsum(counts_per_cat)

    return pd.DataFrame({"r": r_mid, "g_r": g_r, "n_r": n_r})


def compute_density(traj_path, dt_fs, n_frames=N_FRAMES):
    AMU_TO_KG = 1.66053906660e-27
    ANG3_TO_M3 = 1e-30

    frames, traj = sample_frames(traj_path, dt_fs, n_frames)
    densities = []
    try:
        for at in frames:
            mass_kg = float(np.sum(at.get_masses())) * AMU_TO_KG
            vol_m3 = float(at.get_volume()) * ANG3_TO_M3
            densities.append((mass_kg / vol_m3) / 1000.0)
    finally:
        traj.close()

    densities = np.array(densities)
    return densities.mean(), densities.std(ddof=1)


# === Workers ===

def _rdf_worker(args):
    sys_name, model_name, traj_path, cation, partner, dt_fs = args
    df = compute_rdf(traj_path, cation, partner, dt_fs)
    return sys_name, model_name, f"{cation}-{partner}", df


def _density_worker(args):
    sys_name, model_name, traj_path, dt_fs = args
    mean, std = compute_density(traj_path, dt_fs)
    return sys_name, model_name, mean, std


# === Compute RDFs ===

rdf_tasks = []
for sys in SYSTEMS:
    for model_name, traj_path in sys["paths"].items():
        if not traj_path.exists():
            continue
        dt_fs = MODEL_DT_FS[model_name]
        for cation, partner in sys["rdf_pairs"]:
            rdf_tasks.append((sys["name"], model_name, str(traj_path), cation, partner, dt_fs))

print(f"\nComputing {len(rdf_tasks)} RDFs with {MAX_WORKERS} workers...")

rdf_results = {}

with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(_rdf_worker, task) for task in rdf_tasks]
    for fut in tqdm(as_completed(futures), total=len(futures), desc="RDFs"):
        sys_name, model_name, pair_label, df = fut.result()
        rdf_results.setdefault(sys_name, {}).setdefault(pair_label, {})[model_name] = df

print("RDF computation done.")

# === Plot RDFs ===

for sys in SYSTEMS:
    sys_name = sys["name"]
    if sys_name not in rdf_results:
        continue

    pairs = list(rdf_results[sys_name].keys())
    n_pairs = len(pairs)

    fig, axes = plt.subplots(n_pairs, 2, figsize=(12, 4 * n_pairs), squeeze=False)
    fig.suptitle(f"{sys_name} — RDF Comparison", fontsize=14, fontweight="bold")

    for row, pair_label in enumerate(pairs):
        models_data = rdf_results[sys_name][pair_label]

        for model_name in MODEL_ORDER:
            if model_name not in models_data:
                continue
            df = models_data[model_name]
            color = MODEL_COLORS[model_name]
            axes[row, 0].plot(df["r"], df["g_r"], label=model_name, color=color, lw=1.5)
            axes[row, 1].plot(df["r"], df["n_r"], label=model_name, color=color, lw=1.5)

        axes[row, 0].set_ylabel(f"{pair_label}  $g(r)$")
        axes[row, 0].set_xlim(0, R_MAX)
        axes[row, 1].set_ylabel(f"{pair_label}  $n(r)$")
        axes[row, 1].set_xlim(0, R_MAX)
        axes[row, 0].legend(fontsize=9)

    for ax in axes[-1]:
        ax.set_xlabel(r"$r$ (Å)")

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    safe_name = sys_name.replace("/", "_").replace(" ", "_")
    out_path = OUTPUT_DIR / f"rdf_{safe_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

# === Compute Densities ===

density_tasks = []
for sys in SYSTEMS:
    for model_name, traj_path in sys["paths"].items():
        if not traj_path.exists():
            continue
        dt_fs = MODEL_DT_FS[model_name]
        density_tasks.append((sys["name"], model_name, str(traj_path), dt_fs))

print(f"\nComputing {len(density_tasks)} densities with {MAX_WORKERS} workers...")

density_rows = []
with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(_density_worker, task) for task in density_tasks]
    for fut in tqdm(as_completed(futures), total=len(futures), desc="Densities"):
        sys_name, model_name, mean, std = fut.result()
        density_rows.append({"System": sys_name, "Model": model_name, "Density (g/cm³)": mean, "Std (g/cm³)": std})

density_df = pd.DataFrame(density_rows)
density_df = density_df.sort_values(["System", "Model"]).reset_index(drop=True)
print("\nDensity results:")
print(density_df.to_string(index=False))

# Save density table as CSV
csv_path = OUTPUT_DIR / "density_comparison.csv"
density_df.to_csv(csv_path, index=False)
print(f"\nSaved density table: {csv_path}")

# === Plot Densities ===

systems_order = [s["name"] for s in SYSTEMS]
n_systems = len(systems_order)
bar_width = 0.25

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(n_systems)

for i, model_name in enumerate(MODEL_ORDER):
    means, stds = [], []
    for sys_name in systems_order:
        row = density_df[(density_df["System"] == sys_name) & (density_df["Model"] == model_name)]
        if len(row) > 0:
            means.append(row["Density (g/cm³)"].values[0])
            stds.append(row["Std (g/cm³)"].values[0])
        else:
            means.append(0)
            stds.append(0)
    ax.bar(
        x + i * bar_width, means, bar_width,
        yerr=stds, label=model_name, color=MODEL_COLORS[model_name],
        capsize=3, edgecolor="black", linewidth=0.5,
    )

ax.set_xticks(x + bar_width)
ax.set_xticklabels(systems_order, rotation=30, ha="right")
ax.set_ylabel(r"Density (g/cm$^3$)")
ax.set_title("Density Comparison: Student Models vs. UMA Teacher", fontweight="bold")
ax.legend()
fig.tight_layout()

density_plot_path = OUTPUT_DIR / "density_comparison.png"
fig.savefig(density_plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {density_plot_path}")

print("\nAll done.")
