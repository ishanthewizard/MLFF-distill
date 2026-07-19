#!/usr/bin/env python3
"""Compare cell-size stability: anisotropic NPT vs isotropic MTK NPT.

Anisotropic NPT: reads first 5 ns from the live .traj file.
Isotropic NPT: loads pre-computed cell_size_teacher.npz (already ~4 ns).
Produces a merged 4-panel plot (a, b, c, volume vs time).
"""
import sys
from pathlib import Path

import numpy as np

_OBS = Path(__file__).resolve().parent
if str(_OBS) not in sys.path:
    sys.path.insert(0, str(_OBS))

from cell_size.compute import extract_cell_timeseries
from cell_size.plot import plot_cell_timeseries

# ── paths ──────────────────────────────────────────────────────────────────────
ANISO_TRAJ = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5024/distillation_project/results/diffusivity_main_results_20ns_final"
    "/micro_acas_50ps/20ns_solvent_0_1M/298K"
    "/md_omol_napf6_dme_re1/md_omol_napf6_dme_re1.traj"
)
ISO_NPZ = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/yuejian/electrolyte_application/uma_density_isotropic_mtk_npt"
    "/analysis/20260604_140021_msd_cell_size_energy"
    "/NaPF6_DME_298K_0.1M/cell_size_teacher.npz"
)
OUTPUT_DIR = Path(
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/yuejian/electrolyte_application/uma_density_isotropic_mtk_npt"
    "/analysis/cell_size_comparison_iso_vs_aniso"
)

MAX_NS = 5.0   # first 5 ns from anisotropic trajectory
DT_FS  = 100.0  # frame interval for the anisotropic trajectory

# ── load anisotropic NPT (first 5 ns) ─────────────────────────────────────────
print("Extracting anisotropic NPT cell sizes (first 5 ns) …")
aniso = extract_cell_timeseries(
    ANISO_TRAJ,
    dt_fs=DT_FS,
    n_sample=2000,
    max_ns=MAX_NS,
)
print(f"  sampled {len(aniso['times_ns'])} frames, "
      f"t=[{aniso['times_ns'][0]:.3f}, {aniso['times_ns'][-1]:.3f}] ns")

# ── load isotropic NPT (pre-computed npz) ─────────────────────────────────────
print("Loading isotropic MTK NPT cell sizes …")
_npz = np.load(ISO_NPZ)
iso = {k: _npz[k] for k in ("times_ns", "a", "b", "c", "volume")}
# truncate to MAX_NS for a fair comparison
mask = iso["times_ns"] <= MAX_NS
iso = {k: v[mask] for k, v in iso.items()}
print(f"  loaded {len(iso['times_ns'])} frames, "
      f"t=[{iso['times_ns'][0]:.3f}, {iso['times_ns'][-1]:.3f}] ns")

# ── merge and plot ─────────────────────────────────────────────────────────────
times_and_cells = {
    "Isotropic NPT (MTK)":  iso,
    "Anisotropic NPT":       aniso,
}
model_colors = {
    "Isotropic NPT (MTK)": "#2ca02c",   # green → stable
    "Anisotropic NPT":      "#d62728",   # red
}

out_path = plot_cell_timeseries(
    times_and_cells,
    sys_name="NaPF6/DME 0.1M 298K — NPT barostat comparison",
    output_dir=OUTPUT_DIR,
    model_colors=model_colors,
    roll_window_ns=0.2,
)
print(f"\nSaved → {out_path}")
