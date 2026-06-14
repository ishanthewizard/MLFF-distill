#!/usr/bin/env python3
"""
Compute density from 9-10 ns window of MD trajectories and make parity plots.

Two model types:
  - Student (OMol2): ASE .traj files, dt=0.1 ps (every 100 steps at 1 fs)
  - OPLS (GROMACS): .xtc + .tpr files

Five systems at 1 M, 298 K:
  naotf_dme, naotf_diglyme, napf6_dme, napf6_diglyme, napf6_pc

Output:
  density_results.csv      -- computed densities
  density_parity_plot.png  -- parity plot: Student+OPLS vs Exp
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from ase.io.trajectory import Trajectory as AseTraj
import MDAnalysis as mda

matplotlib.rcParams.update({"font.size": 12})

# ── paths ─────────────────────────────────────────────────────────────────────
BASE = Path("/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project")
STUDENT_BASE = BASE / "results/diffusivity_main_results_20ns_final/original_100ps/20ns_solute_solvent_1M/298K"
OPLS_BASE    = BASE / "results/opls_baseline/tpr_files_1M"
XLSX_PATH    = BASE / "experiment_data/diffusivity/Updated data of density, viscosity, and diffusivity.xlsx"

OUT_DIR = Path(__file__).resolve().parent / "density_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── system definitions ────────────────────────────────────────────────────────
# (label, student_dir, student_traj_stem, opls_stem)
# DEGDME = Diglyme
SYSTEMS = [
    ("NaOTf-DME",     "naotf_dme",          "naotf_dme",         "npt_1M_naotf_dme"),
    ("NaOTf-Diglyme", "naotf_diglyme",       "naotf_diglyme",     "npt_1M_naotf_diglyme"),
    ("NaPF6-DME",     "napf6_dme",           "napf6_dme",         "npt_1M_napf6_dme"),
    ("NaPF6-Diglyme", "md_omol_napf6_diglyme_pfactor_0.1_1fs", "md_omol_napf6_diglyme_pfactor_0.1_1fs", "npt_1M_napf6_diglyme"),
    ("NaPF6-PC",      "md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1",
                      "md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1",
                      "npt_1M_napf6_pc"),
]

# Experimental densities (g/mL) from 298.2 K sheet of the xlsx
EXP_DENSITY = {
    "NaOTf-DME":     0.9799,
    "NaOTf-Diglyme": 1.04709,
    "NaPF6-DME":     0.99502,
    "NaPF6-Diglyme": 1.05868,
    "NaPF6-PC":      1.29088,
}

# AMU to g/mL conversion for mass_amu / vol_A3
# 1 amu = 1.66054e-24 g, 1 Å³ = 1e-24 mL
AMU_A3_TO_G_ML = 1.66054


# ── density calculators ───────────────────────────────────────────────────────

def density_from_student_traj(traj_path: Path, t_start_ns: float = 9.0, t_end_ns: float = 10.0,
                               dt_ps: float = 0.1, n_sample: int = 500) -> float:
    """Compute mean density (g/mL) from ASE trajectory in [t_start_ns, t_end_ns]."""
    i_start = int(t_start_ns * 1000 / dt_ps)
    i_end   = int(t_end_ns   * 1000 / dt_ps)
    stride  = max(1, (i_end - i_start) // n_sample)

    frame_idxs = list(range(i_start, i_end, stride))
    densities = []

    with AseTraj(str(traj_path)) as trj:
        n_total = len(trj)
        # Get total mass once from frame 0
        f0 = trj[0]
        total_mass_amu = f0.get_masses().sum()

        for idx in frame_idxs:
            if idx >= n_total:
                break
            f = trj[idx]
            vol_A3 = f.get_volume()
            densities.append(total_mass_amu * AMU_A3_TO_G_ML / vol_A3)

    if not densities:
        raise RuntimeError(f"No frames loaded from {traj_path.name} in [{t_start_ns}, {t_end_ns}] ns")
    return float(np.mean(densities))


def density_from_opls_traj(tpr_path: Path, xtc_path: Path,
                            t_start_ns: float = 9.0, t_end_ns: float = 10.0,
                            n_sample: int = 500) -> float:
    """Compute mean density (g/mL) from GROMACS xtc+tpr in [t_start_ns, t_end_ns]."""
    u = mda.Universe(str(tpr_path), str(xtc_path))
    dt_ps = float(u.trajectory.dt)  # ps per frame
    total_mass_amu = float(u.atoms.masses.sum())

    t_start_ps = t_start_ns * 1000.0
    t_end_ps   = t_end_ns   * 1000.0
    i_start = int(t_start_ps / dt_ps)
    i_end   = int(t_end_ps   / dt_ps)
    n_frames = len(u.trajectory)
    i_start = min(i_start, n_frames - 1)
    i_end   = min(i_end,   n_frames)

    stride = max(1, (i_end - i_start) // n_sample)
    frame_idxs = list(range(i_start, i_end, stride))

    densities = []
    for idx in frame_idxs:
        ts = u.trajectory[idx]
        vol_A3 = float(ts.volume)
        densities.append(total_mass_amu * AMU_A3_TO_G_ML / vol_A3)

    if not densities:
        raise RuntimeError(f"No frames loaded from {xtc_path.name} in [{t_start_ns}, {t_end_ns}] ns")
    return float(np.mean(densities))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    results = []

    for label, student_dir, student_stem, opls_stem in SYSTEMS:
        print(f"\n{'='*60}")
        print(f"System: {label}")

        # Student model
        student_traj = STUDENT_BASE / student_dir / f"{student_stem}.traj"
        if not student_traj.exists():
            print(f"  [WARN] student traj not found: {student_traj}")
            d_student = float("nan")
        else:
            print(f"  Student traj: {student_traj.name}")
            d_student = density_from_student_traj(student_traj)
            print(f"  Student density: {d_student:.4f} g/mL")

        # OPLS
        tpr = OPLS_BASE / f"{opls_stem}.tpr"
        xtc = OPLS_BASE / f"{opls_stem}.xtc"
        if not tpr.exists() or not xtc.exists():
            print(f"  [WARN] OPLS files not found: {tpr.name} / {xtc.name}")
            d_opls = float("nan")
        else:
            print(f"  OPLS traj: {xtc.name}")
            d_opls = density_from_opls_traj(tpr, xtc)
            print(f"  OPLS density: {d_opls:.4f} g/mL")

        d_exp = EXP_DENSITY[label]
        print(f"  Exp density:    {d_exp:.4f} g/mL")

        results.append({
            "system": label,
            "density_student_g_mL": d_student,
            "density_opls_g_mL": d_opls,
            "density_exp_g_mL": d_exp,
        })

    df = pd.DataFrame(results)
    csv_path = OUT_DIR / "density_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n\nSaved CSV: {csv_path}")
    print(df.to_string(index=False))

    # ── parity plot ───────────────────────────────────────────────────────────
    _make_parity_plot(df, OUT_DIR / "density_parity_plot.png")


MARKERS = ["o", "s", "^", "D", "v"]
COLORS_STUDENT = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
COLORS_OPLS    = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]


def _make_parity_plot(df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 6))

    all_vals = []

    for i, row in df.iterrows():
        x_exp = row["density_exp_g_mL"]
        marker = MARKERS[i % len(MARKERS)]
        color  = COLORS_STUDENT[i % len(COLORS_STUDENT)]
        label  = row["system"]

        if np.isfinite(row["density_student_g_mL"]):
            ax.scatter(x_exp, row["density_student_g_mL"],
                       marker=marker, color=color, s=80, zorder=3,
                       label=f"{label} (Student)")
            all_vals.extend([x_exp, row["density_student_g_mL"]])

        if np.isfinite(row["density_opls_g_mL"]):
            ax.scatter(x_exp, row["density_opls_g_mL"],
                       marker=marker, color=color, s=80, zorder=3,
                       facecolors="none", linewidths=1.5,
                       label=f"{label} (OPLS)")
            all_vals.extend([x_exp, row["density_opls_g_mL"]])

    if all_vals:
        lo = min(all_vals) * 0.97
        hi = max(all_vals) * 1.03
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, zorder=1, label="y = x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    ax.set_xlabel("Experimental density (g/mL)", fontsize=13)
    ax.set_ylabel("Simulated density (g/mL)", fontsize=13)
    ax.set_title("Density parity plot — 1 M, 298 K\n(9–10 ns window)", fontsize=13)
    ax.set_aspect("equal")

    # Custom legend: separate blocks for Student and OPLS
    handles, labels = ax.get_legend_handles_labels()
    # Re-order: first all student, then parity line, then OPLS
    student_h = [(h, l) for h, l in zip(handles, labels) if "Student" in l]
    opls_h    = [(h, l) for h, l in zip(handles, labels) if "OPLS" in l]
    parity_h  = [(h, l) for h, l in zip(handles, labels) if "y = x" in l]

    legend_handles = [h for h, l in student_h + opls_h + parity_h]
    legend_labels  = [l for h, l in student_h + opls_h + parity_h]
    ax.legend(legend_handles, legend_labels, fontsize=8, loc="upper left",
              frameon=True, framealpha=0.9)

    # Add a note: filled = Student, open = OPLS
    ax.text(0.98, 0.02,
            "Filled marker = Student (OMol2)\nOpen marker = OPLS",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved parity plot: {out_path}")


if __name__ == "__main__":
    main()
