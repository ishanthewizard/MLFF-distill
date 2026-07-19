#!/usr/bin/env python
"""Potential energy + cell size (+ density/T/P) from the stored GROMACS .edr files.

The OPLS 1M NPT production runs store energies every nstenergy=100 steps
(dt=1 fs) -> one .edr record every 100 fs.  The compressed .xtc holds no
energies, and eval.py's `energy` analysis explicitly skips GROMACS, so we read
the *stored* values straight from the .edr with pyedr:

  - Potential  (kJ/mol)   -> potential energy time series + equilibrium mean
  - Volume     (nm^3)     -> cell volume
  - Box-X/Y/Z  (nm)       -> box edge lengths (cubic here)
  - Density    (kg/m^3)   -> mass density
  - Temperature (K), Pressure (bar) -> sanity checks

Per system we write:
  <out>/<name>_edr_timeseries.npz   downsampled series (for re-plotting)
  <out>/<name>_edr.png              4-panel time series (PE / box edge / density / T)
and a combined:
  <out>/edr_summary.csv             post-eq means +/- std per system
  <out>/edr_potential_energy.png    PE overlay (per-atom) across systems
  <out>/edr_cell_volume.png         volume overlay across systems

Means/std are computed on the FULL post-`eq_cut_ns` record; plots are
downsampled to ~4000 points for speed.

Env: fairchemV2_new (has pyedr).
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pyedr

SYSTEMS = [
    # (name, salt, solvent_label)
    ("npt_1M_naotf_diglyme", "NaOTf",  "Diglyme"),
    ("npt_1M_naotf_dme",     "NaOTf",  "DME"),
    ("npt_1M_napf6_diglyme", "NaPF6",  "Diglyme"),
    ("npt_1M_napf6_dme",     "NaPF6",  "DME"),
    ("npt_1M_napf6_pc",      "NaPF6",  "PC"),
]

COLORS = {
    "npt_1M_naotf_diglyme": "#1f77b4",
    "npt_1M_naotf_dme":     "#ff7f0e",
    "npt_1M_napf6_diglyme": "#2ca02c",
    "npt_1M_napf6_dme":     "#d62728",
    "npt_1M_napf6_pc":      "#9467bd",
}


def rolling(x, w):
    if w <= 1 or w >= len(x):
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = (c[w:] - c[:-w]) / w
    pad = np.full(w - 1, out[0])
    return np.concatenate([pad, out])


def downsample(t, y, npts=4000):
    if len(t) <= npts:
        return t, y
    idx = np.linspace(0, len(t) - 1, npts).astype(int)
    return t[idx], y[idx]


def analyze_one(edr_path, name, salt, solvent, n_atoms_map, out_dir, eq_cut_ns=2.0):
    d = pyedr.edr_to_dict(str(edr_path))
    t_ps = np.asarray(d["Time"])
    t_ns = t_ps / 1000.0
    pe = np.asarray(d["Potential"])                 # kJ/mol
    vol = np.asarray(d["Volume"])                   # nm^3
    box = np.asarray(d.get("Box-X", np.cbrt(vol)))  # nm
    dens = np.asarray(d["Density"])                 # kg/m^3
    temp = np.asarray(d["Temperature"])             # K
    pres = np.asarray(d["Pressure"])                # bar

    natoms = n_atoms_map[name]
    mask = t_ns >= eq_cut_ns
    if mask.sum() < 10:
        mask = np.ones_like(t_ns, dtype=bool)

    def ms(a):
        return float(np.mean(a[mask])), float(np.std(a[mask]))

    pe_m, pe_s = ms(pe)
    vol_m, vol_s = ms(vol)
    box_m, box_s = ms(box)
    dens_m, dens_s = ms(dens)
    temp_m, temp_s = ms(temp)
    pres_m, pres_s = ms(pres)

    summary = {
        "system": name, "salt": salt, "solvent": solvent, "n_atoms": natoms,
        "traj_ns": float(t_ns[-1]), "n_records": int(len(t_ns)),
        "dt_ps": float(t_ps[1] - t_ps[0]), "eq_cut_ns": eq_cut_ns,
        "PE_kJ_mol_mean": pe_m, "PE_kJ_mol_std": pe_s,
        "PE_per_atom_kJ_mol_mean": pe_m / natoms, "PE_per_atom_kJ_mol_std": pe_s / natoms,
        "Volume_nm3_mean": vol_m, "Volume_nm3_std": vol_s,
        "BoxEdge_nm_mean": box_m, "BoxEdge_nm_std": box_s,
        "BoxEdge_A_mean": box_m * 10.0, "BoxEdge_A_std": box_s * 10.0,
        "Density_g_mL_mean": dens_m / 1000.0, "Density_g_mL_std": dens_s / 1000.0,
        "Temperature_K_mean": temp_m, "Temperature_K_std": temp_s,
        "Pressure_bar_mean": pres_m, "Pressure_bar_std": pres_s,
    }

    # downsampled timeseries for storage + plotting
    td, _ = downsample(t_ns, t_ns)
    np.savez_compressed(
        out_dir / f"{name}_edr_timeseries.npz",
        time_ns=downsample(t_ns, t_ns)[1],
        potential_kJ_mol=downsample(t_ns, pe)[1],
        volume_nm3=downsample(t_ns, vol)[1],
        box_edge_nm=downsample(t_ns, box)[1],
        density_g_mL=downsample(t_ns, dens)[1] / 1000.0,
        temperature_K=downsample(t_ns, temp)[1],
        pressure_bar=downsample(t_ns, pres)[1],
        n_atoms=natoms, eq_cut_ns=eq_cut_ns,
    )

    # per-system 4-panel figure
    w = max(1, int(round(1000.0 / (t_ps[1] - t_ps[0]))))   # ~1 ns rolling
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    color = COLORS.get(name, "#1f77b4")
    panels = [
        ("Potential energy per atom (kJ/mol)", pe / natoms, axes[0, 0]),
        ("Box edge (Å)", box * 10.0, axes[0, 1]),
        ("Density (g/mL)", dens / 1000.0, axes[1, 0]),
        ("Temperature (K)", temp, axes[1, 1]),
    ]
    for title, y, ax in panels:
        tp, yp = downsample(t_ns, y)
        ax.plot(tp, yp, lw=0.4, alpha=0.35, color=color)
        _, yr = downsample(t_ns, rolling(y, w))
        ax.plot(tp, yr, lw=1.6, color=color, label="1 ns rolling")
        ax.axvline(eq_cut_ns, color="grey", ls="--", lw=1, label=f"eq cut {eq_cut_ns} ns")
        ax.set_xlabel("time (ns)")
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"{name}  ({salt} / {solvent}, 1 M, NPT 298 K, {t_ns[-1]:.1f} ns)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / f"{name}_edr.png", dpi=140)
    plt.close(fig)

    return summary, (t_ns, pe / natoms, vol, box * 10.0, dens / 1000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tpr-dir", default="/global/homes/y/yuejian/project/MLFF-distill/"
                    "m5024/distillation_project/results/opls_baseline/tpr_files_1M")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eq-cut-ns", type=float, default=2.0)
    args = ap.parse_args()

    tpr_dir = Path(args.tpr_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # atom counts (from TPR probe); used for per-atom PE normalisation
    n_atoms_map = {
        "npt_1M_naotf_diglyme": 2292,
        "npt_1M_naotf_dme": 2153,
        "npt_1M_napf6_diglyme": 2275,
        "npt_1M_napf6_dme": 2136,
        "npt_1M_napf6_pc": 2268,
    }

    summaries = []
    series = {}
    for name, salt, solvent in SYSTEMS:
        edr = tpr_dir / f"{name}.edr"
        if not edr.exists():
            print(f"[EDR] SKIP {name}: missing {edr}")
            continue
        print(f"[EDR] {name}: reading {edr.name} ...", flush=True)
        s, ser = analyze_one(edr, name, salt, solvent, n_atoms_map, out, args.eq_cut_ns)
        summaries.append(s)
        series[name] = (ser, salt, solvent)
        print(f"[EDR] {name}: {s['traj_ns']:.1f} ns  PE/atom={s['PE_per_atom_kJ_mol_mean']:.3f} kJ/mol  "
              f"box={s['BoxEdge_A_mean']:.3f} A  rho={s['Density_g_mL_mean']:.4f} g/mL  "
              f"T={s['Temperature_K_mean']:.1f} K  P={s['Pressure_bar_mean']:.1f} bar", flush=True)

    if not summaries:
        print("No EDR files processed.")
        return

    df = pd.DataFrame(summaries)
    csv_path = out / "edr_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[EDR] wrote {csv_path}")

    # overlay: potential energy per atom
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, ((t_ns, pe_pa, vol, box_A, dens), salt, solvent) in series.items():
        tp, yp = downsample(t_ns, rolling(pe_pa, max(1, int(len(t_ns) / 500))))
        ax.plot(tp, yp, lw=1.3, color=COLORS.get(name), label=f"{salt}/{solvent}")
    ax.axvline(args.eq_cut_ns, color="grey", ls="--", lw=1)
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("Potential energy per atom (kJ/mol)")
    ax.set_title("OPLS 1 M NPT — potential energy (stored, 1 ns rolling)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "edr_potential_energy.png", dpi=140)
    plt.close(fig)

    # overlay: cell volume
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, ((t_ns, pe_pa, vol, box_A, dens), salt, solvent) in series.items():
        tp, yp = downsample(t_ns, rolling(vol, max(1, int(len(t_ns) / 500))))
        ax.plot(tp, yp, lw=1.3, color=COLORS.get(name), label=f"{salt}/{solvent}")
    ax.axvline(args.eq_cut_ns, color="grey", ls="--", lw=1)
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("Cell volume (nm³)")
    ax.set_title("OPLS 1 M NPT — cell volume (stored, 1 ns rolling)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "edr_cell_volume.png", dpi=140)
    plt.close(fig)

    print(f"[EDR] wrote overlay figures to {out}")
    # echo table
    cols = ["system", "salt", "solvent", "traj_ns", "PE_per_atom_kJ_mol_mean",
            "BoxEdge_A_mean", "Volume_nm3_mean", "Density_g_mL_mean",
            "Temperature_K_mean", "Pressure_bar_mean"]
    print("\n" + df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
