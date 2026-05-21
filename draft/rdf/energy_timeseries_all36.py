#!/usr/bin/env python
"""KE, PE, total energy, and temperature time-series for all 36 trajectories.

Auto-discovers every *.traj under both model directories:
  - original_trained_on_1M_only_100ps_window   (18 traj)
  - micro_trained_on_all_concentration_50ps_window  (18 traj)

One figure per trajectory with four stacked subplots:
  PE (eV), KE (eV), E_total (eV), T (K) vs simulation time (ns).
Raw samples as faint scatter; thick line is a rolling average.
Also saves per-frame numerical data as .npz.

Output directory:
  /global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/analysis/energy_timeseries/

env: /global/homes/y/yuejian/project/MLFF-distill/yuejian/envs/fairchemV2_new
"""

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
DT_FS      = 100.0    # fs per frame
N_SAMPLE   = 3000     # frames to sample (strided)
ROLL_FRAC  = 0.05     # rolling average width as fraction of N_SAMPLE

OUTPUT_BASE = Path(
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/analysis"
    "/energy_timeseries"
)
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# ── Model roots ───────────────────────────────────────────────────────────────
_BASE = Path(
    "/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project"
    "/results/diffusivity_main_results_ckpt/other_fix_run"
)
MODEL_ROOTS = {
    "original": _BASE / "original_trained_on_1M_only_100ps_window",
    "micro":    _BASE / "micro_trained_on_all_concentration_50ps_window",
}
MODEL_COLORS = {"original": "#2ca02c", "micro": "#ff7f0e"}


# ── Auto-discovery ────────────────────────────────────────────────────────────

def discover_trajs(model_root: Path, model_label: str):
    entries = []
    for traj_path in sorted(model_root.rglob("*.traj")):
        rel   = traj_path.relative_to(model_root)
        parts = rel.parts
        if len(parts) == 3:
            conc, sys_dir, _ = parts
            temp = ""
        elif len(parts) == 4:
            conc, temp, sys_dir, _ = parts
        else:
            continue
        human = f"{sys_dir.replace('_', ' ')} | {conc} {temp}".strip()
        entries.append({
            "model":     model_label,
            "traj_path": traj_path,
            "sys_dir":   sys_dir,
            "conc":      conc,
            "temp":      temp,
            "name":      human,
            "slug":      f"{model_label}__{conc}__{temp}__{sys_dir}".replace(
                         ".", "p").replace("/", "_"),
        })
    return entries


all_entries = []
for label, root in MODEL_ROOTS.items():
    all_entries.extend(discover_trajs(root, label))

print(f"Discovered {len(all_entries)} trajectories total.")
for e in all_entries:
    status = "OK" if e["traj_path"].exists() else "MISSING"
    print(f"  [{status}] {e['model']:10s}  {e['conc']:30s}  {e['temp']:8s}  {e['sys_dir']}")


# ── Rolling average ───────────────────────────────────────────────────────────

def rolling_mean(x, w):
    w = max(1, int(w))
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()


# ── Extract energies from trajectory ──────────────────────────────────────────

def extract_energies(traj_path, n_sample=N_SAMPLE, dt_fs=DT_FS):
    traj    = Trajectory(str(traj_path), mode="r")
    n_tot   = len(traj)
    stride  = max(1, n_tot // n_sample)
    indices = range(0, n_tot, stride)

    times, pe, ke, etot, temp = [], [], [], [], []
    for idx in tqdm(indices, desc="  frames", leave=False):
        at = traj[idx]
        if at.calc is None or "energy" not in at.calc.results:
            continue
        times.append(idx * dt_fs * 1e-6)           # fs → ns
        pe.append(at.calc.results["energy"])
        ke.append(at.get_kinetic_energy())
        etot.append(at.calc.results["energy"] + at.get_kinetic_energy())
        temp.append(at.get_temperature())

    traj.close()
    return (np.array(times), np.array(pe), np.array(ke),
            np.array(etot), np.array(temp))


# ── Main loop ─────────────────────────────────────────────────────────────────

for entry in all_entries:
    traj_path = entry["traj_path"]
    if not traj_path.exists():
        print(f"SKIP (missing): {entry['name']}")
        continue

    sys_name  = entry["name"]
    slug      = entry["slug"]
    model_lbl = entry["model"]
    color     = MODEL_COLORS[model_lbl]

    out_dir = OUTPUT_BASE / model_lbl
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{model_lbl.upper()}  |  {sys_name}")

    times, pe, ke, etot, temp = extract_energies(traj_path)
    if len(times) == 0:
        print("  No frames with energy data — skipping.")
        continue

    roll_w = max(3, int(len(times) * ROLL_FRAC))

    # save numerical data
    npz_path = out_dir / f"{slug}__energy.npz"
    np.savez_compressed(
        npz_path,
        times=times, pe=pe, ke=ke, etot=etot, temp=temp
    )
    print(f"  Saved data: {npz_path}")

    # print summary statistics
    print(f"  {'Quantity':12s}  {'mean':>12s}  {'std':>10s}  {'min':>12s}  {'max':>12s}")
    for label, vals in [("PE (eV)", pe), ("KE (eV)", ke),
                         ("Etot (eV)", etot), ("T (K)", temp)]:
        print(f"  {label:12s}  {vals.mean():12.4g}  {vals.std():10.3g}  "
              f"{vals.min():12.4g}  {vals.max():12.4g}")

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    fig.suptitle(
        f"{model_lbl.upper()}  |  {sys_name}\nEnergy & Temperature vs Time",
        fontsize=12, fontweight="bold",
    )

    datasets = [
        (pe,   r"$E_{pot}$ (eV)",  "PE"),
        (ke,   r"$E_{kin}$ (eV)",  "KE"),
        (etot, r"$E_{tot}$ (eV)",  "Total E"),
        (temp, r"$T$ (K)",          "Temperature"),
    ]

    for ax, (vals, ylabel, lbl) in zip(axes, datasets):
        ax.scatter(times, vals, s=1.5, alpha=0.25, color=color, rasterized=True)
        ax.plot(times, rolling_mean(vals, roll_w),
                color=color, lw=1.6, label=f"{lbl} (roll avg)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9, loc="upper right")

        mean_v = vals.mean()
        std_v  = vals.std()
        ax.axhline(mean_v, color="black", lw=0.8, ls="--", alpha=0.6)
        ax.set_ylim(mean_v - 5 * std_v, mean_v + 5 * std_v)
        ax.text(0.01, 0.05,
                f"mean={mean_v:.4g}  std={std_v:.3g}",
                transform=ax.transAxes, fontsize=8,
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    axes[-1].set_xlabel("Simulation time (ns)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_png = out_dir / f"{slug}__energy.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_png}")

print("\nAll done.")
