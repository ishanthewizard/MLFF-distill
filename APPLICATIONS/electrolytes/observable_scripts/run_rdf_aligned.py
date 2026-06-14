#!/usr/bin/env python3
"""RDF comparison with strict timestep alignment.

Both trajectories are sampled at identical physical timestamps (100 fs stride,
0.5–0.95 ns window), so every frame pair is at exactly the same simulation time.

UMA  : 10  fs/frame → every 10th frame  (indices 50000–95000 step 10)
NVT  : 100 fs/frame → every 1st  frame  (indices  5000– 9500 step  1)
"""
import sys
from math import pi
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io.trajectory import Trajectory
from ase.geometry import find_mic

# ── paths ─────────────────────────────────────────────────────────────────────
UMA_TRAJ = Path(
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5024/distillation_project/data/raw_data_from_UMA_simulation"
    "/other_temperature_for_comparing_with_student_model"
    "/20ns_solvent_0_1M/298K/md_omol_naotf_dme_s1p1_omol"
    "/md_omol_naotf_dme_s1p1_omol.traj"
)
NVT_TRAJ = Path(
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/yuejian/electrolyte_application/distort_temp_md_nvt/simulations/nvt/nose_hoover"
    "/na_otf_dme_298K_0.1M_teacher/na_otf_dme_298K_0.1M_teacher.traj"
)
OUT_DIR = Path(
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/yuejian/electrolyte_application/distort_temp_md_nvt/analysis/nvt_ckpt_1_5"
    "/rdf_uma_vs_nvt"
)

# ── alignment parameters ───────────────────────────────────────────────────────
ALIGNED_DT_FS = 1000.0       # effective dt for both (fs)
START_NS      = 0.5
END_NS        = 0.95
UMA_DT_FS     = 10.0
NVT_DT_FS     = 100.0

RDF_PAIRS = [("Na", "S"), ("Na", "O")]
R_MAX, DR = 10.0, 0.05

COLORS = {"UMA": "#1f77b4", "NVT_teacher": "#ff7f0e"}

# ── frame index ranges ─────────────────────────────────────────────────────────
uma_start  = int(START_NS * 1e6 / UMA_DT_FS)        # 50000
uma_end    = int(END_NS   * 1e6 / UMA_DT_FS)         # 95000
uma_stride = int(ALIGNED_DT_FS / UMA_DT_FS)          # 10

nvt_start  = int(START_NS * 1e6 / NVT_DT_FS)        # 5000
nvt_end    = int(END_NS   * 1e6 / NVT_DT_FS)         # 9500
nvt_stride = int(ALIGNED_DT_FS / NVT_DT_FS)          # 1

print(f"UMA  frames {uma_start}:{uma_end}:{uma_stride}  "
      f"→ {len(range(uma_start, uma_end, uma_stride))} frames")
print(f"NVT  frames {nvt_start}:{nvt_end}:{nvt_stride}  "
      f"→ {len(range(nvt_start, nvt_end, nvt_stride))} frames")


# ── RDF helpers ───────────────────────────────────────────────────────────────
def _rdf_hist(frames, cation, partner, bins):
    r_mid = 0.5 * (bins[:-1] + bins[1:])
    hist = np.zeros(len(r_mid), dtype=np.float64)
    n_cat = n_part = vol_sum = frame_count = 0.0
    for at in frames:
        syms = at.get_chemical_symbols()
        ic = [i for i, s in enumerate(syms) if s == cation]
        ip = [i for i, s in enumerate(syms) if s == partner]
        if not ic or not ip:
            continue
        pos  = at.get_positions()
        cell = at.get_cell()
        pbc  = at.get_pbc()
        for rc in pos[ic]:
            disp, _ = find_mic(pos[ip] - rc, cell, pbc)
            hist += np.histogram(np.linalg.norm(disp, axis=1), bins=bins)[0]
        n_cat    += len(ic)
        n_part   += len(ip)
        vol_sum  += at.get_volume()
        frame_count += 1
    return hist, n_cat, n_part, vol_sum, frame_count


def compute_rdf(frames, cation, partner):
    bins = np.arange(0.0, R_MAX + DR, DR)
    hist, n_cat, n_part, vol_sum, fc = _rdf_hist(frames, cation, partner, bins)
    if fc == 0:
        raise ValueError("No valid frames")
    r_mid      = 0.5 * (bins[:-1] + bins[1:])
    shell_vol  = 4.0 / 3.0 * pi * (bins[1:] ** 3 - bins[:-1] ** 3)
    rho        = (n_part / fc) / (vol_sum / fc)
    counts_cat = hist / n_cat
    g_r        = counts_cat / (rho * shell_vol)
    n_r        = np.cumsum(counts_cat)
    return r_mid, g_r, n_r


# ── load frames ───────────────────────────────────────────────────────────────
print("\nLoading UMA frames …")
with Trajectory(str(UMA_TRAJ), mode="r") as t:
    uma_frames = list(t[uma_start:uma_end:uma_stride])
print(f"  loaded {len(uma_frames)} frames")

print("Loading NVT frames …")
with Trajectory(str(NVT_TRAJ), mode="r") as t:
    nvt_frames = list(t[nvt_start:nvt_end:nvt_stride])
print(f"  loaded {len(nvt_frames)} frames")

assert len(uma_frames) == len(nvt_frames), \
    f"Frame count mismatch: UMA={len(uma_frames)} NVT={len(nvt_frames)}"
print(f"\nBoth trajectories: {len(uma_frames)} frames "
      f"@ {ALIGNED_DT_FS:.0f} fs stride, {START_NS}–{END_NS} ns")

# ── compute RDFs ──────────────────────────────────────────────────────────────
results = {}
for model, frames in [("UMA", uma_frames), ("NVT_teacher", nvt_frames)]:
    results[model] = {}
    for cat, partner in RDF_PAIRS:
        lbl = f"{cat}-{partner}"
        print(f"  RDF {model} {lbl} …", end=" ", flush=True)
        r, g, n = compute_rdf(frames, cat, partner)
        results[model][lbl] = (r, g, n)
        print("done")

# ── plot ──────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
n_pairs = len(RDF_PAIRS)
fig, axes = plt.subplots(n_pairs, 2, figsize=(13, 4 * n_pairs))
if n_pairs == 1:
    axes = axes[np.newaxis, :]

fig.suptitle(
    f"NaOTf/DME 298K 0.1M — RDF comparison (aligned @ {ALIGNED_DT_FS:.0f} fs, "
    f"{START_NS}–{END_NS} ns)",
    fontsize=12, fontweight="bold",
)

for row, (cat, partner) in enumerate(RDF_PAIRS):
    lbl = f"{cat}-{partner}"
    ax_g, ax_n = axes[row]
    for model in ("UMA", "NVT_teacher"):
        r, g, n = results[model][lbl]
        c = COLORS[model]
        ax_g.plot(r, g, color=c, lw=1.5, label=model)
        ax_n.plot(r, n, color=c, lw=1.5, label=model)
    ax_g.set_xlabel("r (Å)")
    ax_g.set_ylabel("g(r)")
    ax_g.set_title(f"{lbl}  g(r)")
    ax_g.legend(fontsize=9)
    ax_g.grid(True, ls=":")
    ax_n.set_xlabel("r (Å)")
    ax_n.set_ylabel("n(r)")
    ax_n.set_title(f"{lbl}  n(r)")
    ax_n.legend(fontsize=9)
    ax_n.grid(True, ls=":")

fig.tight_layout(rect=(0, 0, 1, 0.95))
out = OUT_DIR / "rdf_aligned_uma_vs_nvt_naotf_298K_0p1M.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out}")

# also save CSV
for model in ("UMA", "NVT_teacher"):
    rows = []
    for cat, partner in RDF_PAIRS:
        lbl = f"{cat}-{partner}"
        r, g, n = results[model][lbl]
        df = pd.DataFrame({"r": r, "g_r": g, "n_r": n, "pair": lbl})
        rows.append(df)
    pd.concat(rows).to_csv(OUT_DIR / f"rdf_aligned_{model}.csv", index=False)
print("CSVs saved.")
