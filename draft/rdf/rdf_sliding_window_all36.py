#!/usr/bin/env python
"""Sliding-window RDF and n(r) analysis for all 36 trajectories.

Auto-discovers every *.traj under both model directories:
  - original_trained_on_1M_only_100ps_window   (18 traj)
  - micro_trained_on_all_concentration_50ps_window  (18 traj)

For each trajectory × RDF pair:
  - Divides the 20 ns trajectory into overlapping windows
    (0.9 ns wide, sliding every 1 ns, skipping the first 0.1 ns)
  - Computes g(r) and n(r) per window
  - Plots: overlaid curves (blue→red = early→late) + std band
  - Suggests equilibration cutoff (std/mean < 5 %)
  - Saves numerical data as .npz

Output directory:
  /global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/analysis/rdf_sliding_window/

env: /global/homes/y/yuejian/project/MLFF-distill/yuejian/envs/fairchemV2_new
"""

from math import pi
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from ase.io.trajectory import Trajectory
from ase.geometry import find_mic

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.linestyle": ":"})

# ── Config ────────────────────────────────────────────────────────────────────
WINDOW_NS           = 0.9
SLIDE_NS            = 1.0
SKIP_NS             = 0.1
TOTAL_NS            = 20.0
N_FRAMES_PER_WINDOW = 900

R_MAX = 10.0
DR    = 0.05

DT_FS           = 100.0   # fs per frame
STD_THRESH_FRAC = 0.05

OUTPUT_BASE = Path(
    "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/analysis"
    "/rdf_sliding_window"
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


# ── RDF-pair inference from directory name ────────────────────────────────────

def get_rdf_pairs(sys_dir: str):
    n = sys_dir.lower()
    if "lipf6" in n:
        return [("Li", "O"), ("Li", "F")]
    if "naotf" in n or n.startswith("naotf"):
        return [("Na", "S"), ("Na", "O")]
    # napf6 / napf6_* / md_omol_napf6_*
    if "napf6" in n or n.startswith("napf6"):
        return [("Na", "O"), ("Na", "F")]
    # fallback
    return [("Na", "O")]


# ── Auto-discovery of all trajectories ───────────────────────────────────────

def discover_trajs(model_root: Path, model_label: str):
    """Walk model_root and return list of entry dicts."""
    entries = []
    for traj_path in sorted(model_root.rglob("*.traj")):
        rel   = traj_path.relative_to(model_root)
        parts = rel.parts       # (conc, [temp,] sys_dir, traj_file)
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
            "rdf_pairs": get_rdf_pairs(sys_dir),
            # unique slug for file naming
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


# ── Window definitions ────────────────────────────────────────────────────────

def get_windows(dt_fs=DT_FS, total_ns=TOTAL_NS,
                skip_ns=SKIP_NS, window_ns=WINDOW_NS, slide_ns=SLIDE_NS):
    windows = []
    t = skip_ns
    while t + window_ns <= total_ns + 1e-9:
        t_end   = min(t + window_ns, total_ns)
        f_start = int(round(t     * 1e6 / dt_fs))
        f_end   = int(round(t_end * 1e6 / dt_fs))
        windows.append((t, t_end, f_start, f_end))
        t += slide_ns
    return windows


# ── RDF for one window ────────────────────────────────────────────────────────

def compute_rdf_window(traj, f_start, f_end, cation, partner,
                       n_frames=N_FRAMES_PER_WINDOW, r_max=R_MAX, dr=DR):
    bins     = np.arange(0.0, r_max + dr, dr)
    r_mid    = 0.5 * (bins[:-1] + bins[1:])
    shell_v  = 4.0 / 3.0 * pi * (bins[1:] ** 3 - bins[:-1] ** 3)
    length   = f_end - f_start
    stride   = max(1, length // n_frames)
    indices  = range(f_start, f_end, stride)
    hist     = np.zeros(len(r_mid))
    n_cat = n_part = vol_sum = frame_count = 0.0

    for idx in indices:
        at   = traj[idx]
        syms = at.get_chemical_symbols()
        ic   = [i for i, s in enumerate(syms) if s == cation]
        ip   = [i for i, s in enumerate(syms) if s == partner]
        if not ic or not ip:
            continue
        pos  = at.get_positions()
        cell = at.get_cell()
        pbc  = at.get_pbc()
        for rc in pos[ic]:
            disp, _ = find_mic(pos[ip] - rc, cell, pbc)
            d = np.linalg.norm(disp, axis=1)
            hist += np.histogram(d, bins=bins)[0]
        n_cat      += len(ic)
        n_part     += len(ip)
        vol_sum    += at.get_volume()
        frame_count += 1

    if frame_count == 0:
        return r_mid, np.zeros(len(r_mid)), np.zeros(len(r_mid))

    rho             = (n_part / frame_count) / (vol_sum / frame_count)
    counts_per_cat  = hist / n_cat
    g_r             = counts_per_cat / (rho * shell_v)
    n_r             = np.cumsum(counts_per_cat)
    return r_mid, g_r, n_r


def first_peak(r, g_r, r_min=1.5):
    mask  = r >= r_min
    idx   = np.argmax(g_r[mask])
    return float(r[mask][idx]), float(g_r[mask][idx])


# ── Main loop ─────────────────────────────────────────────────────────────────

windows = get_windows()
print(f"\nSliding windows: {len(windows)}  ({WINDOW_NS} ns wide, {SLIDE_NS} ns step, "
      f"skip {SKIP_NS} ns)\n")

for entry in all_entries:
    traj_path = entry["traj_path"]
    if not traj_path.exists():
        print(f"SKIP (missing): {entry['name']}")
        continue

    sys_name  = entry["name"]
    slug      = entry["slug"]
    model_lbl = entry["model"]

    # output sub-dir per model
    out_dir = OUTPUT_BASE / model_lbl
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  {model_lbl.upper()}  |  {sys_name}")
    print(f"  {traj_path}")
    print(f"{'='*70}")

    n_total_frames = len(Trajectory(str(traj_path), mode="r"))

    for cation, partner in entry["rdf_pairs"]:
        pair_label = f"{cation}-{partner}"
        print(f"\n  Pair: {pair_label}")

        all_gr, all_nr = [], []
        r_mid = None
        summary_rows = []

        traj = Trajectory(str(traj_path), mode="r")

        for w_idx, (t_start, t_end, f_start, f_end) in enumerate(
            tqdm(windows, desc=f"    windows [{pair_label}]", leave=False)
        ):
            f_end = min(f_end, n_total_frames)
            if f_end <= f_start:
                continue
            r, g, n = compute_rdf_window(traj, f_start, f_end, cation, partner)
            if r_mid is None:
                r_mid = r
            all_gr.append(g)
            all_nr.append(n)

            pk_r, pk_h = first_peak(r, g)
            pk_idx     = np.argmax(g[r >= 1.5]) + np.searchsorted(r, 1.5)
            after_peak = g[pk_idx:]
            min_idx    = (np.argmin(after_peak) + pk_idx) if len(after_peak) else pk_idx
            summary_rows.append({
                "Window":        w_idx + 1,
                "Start (ns)":    t_start,
                "End (ns)":      t_end,
                "Peak r (Å)":    round(pk_r, 3),
                "Peak g(r)":     round(pk_h, 3),
                "Shell cut (Å)": round(float(r[min_idx]), 3),
                "CN":            round(float(n[min_idx]), 3),
            })

        traj.close()

        if not all_gr:
            print("    No windows — skipping.")
            continue

        all_gr   = np.array(all_gr)
        all_nr   = np.array(all_nr)
        mean_gr  = all_gr.mean(axis=0)
        std_gr   = all_gr.std(axis=0, ddof=1)
        mean_nr  = all_nr.mean(axis=0)
        std_nr   = all_nr.std(axis=0, ddof=1)
        n_win    = all_gr.shape[0]

        # save numerical data
        npz_path = out_dir / f"{slug}__{pair_label}.npz"
        np.savez_compressed(
            npz_path,
            r_mid=r_mid, all_gr=all_gr, all_nr=all_nr,
            mean_gr=mean_gr, std_gr=std_gr,
            mean_nr=mean_nr, std_nr=std_nr,
            window_starts=np.array([w[0] for w in windows[:n_win]]),
        )
        print(f"  Saved data: {npz_path}")

        # summary table
        df_sum = pd.DataFrame(summary_rows)
        print("\n  Window summary:")
        print(df_sum.to_string(index=False))

        # equilibration cutoff
        cutoff_ns = None
        for w in range(n_win):
            seg = all_gr[w:]
            if seg.shape[0] < 2:
                break
            seg_std  = seg.std(axis=0, ddof=1)
            seg_mean = seg.mean(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                rr = np.nanmean(np.where(seg_mean > 0.1, seg_std / seg_mean, np.nan))
            if rr < STD_THRESH_FRAC:
                cutoff_ns = windows[w][0]
                break
        if cutoff_ns is not None:
            print(f"\n  Equilibration cutoff: start from {cutoff_ns:.1f} ns "
                  f"(std/mean < {STD_THRESH_FRAC*100:.0f}%)")
        else:
            print(f"\n  Equilibration cutoff: std/mean never < "
                  f"{STD_THRESH_FRAC*100:.0f}% — may not be equilibrated.")

        cmap      = cm.get_cmap("coolwarm", n_win)
        t_starts  = np.array([w[0] for w in windows[:n_win]])

        # ── g(r) plot ──────────────────────────────────────────────────────────
        fig, (ax_rdf, ax_std) = plt.subplots(
            2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [3, 1.2]}
        )
        fig.suptitle(
            f"{model_lbl.upper()}  |  {sys_name}  |  {pair_label}\n"
            f"Sliding window g(r)  ({WINDOW_NS} ns window, {SLIDE_NS} ns step)",
            fontsize=11, fontweight="bold",
        )
        for wi, g in enumerate(all_gr):
            ax_rdf.plot(r_mid, g, color=cmap(wi / max(n_win - 1, 1)),
                        lw=0.8, alpha=0.85)
        ax_rdf.plot(r_mid, mean_gr, color="black", lw=1.8, ls="--", label="mean")
        ax_rdf.set_ylabel(r"$g(r)$")
        ax_rdf.set_xlim(0, R_MAX)
        ax_rdf.legend(fontsize=9)
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=t_starts[0], vmax=t_starts[-1])
        )
        sm.set_array([])
        fig.colorbar(sm, ax=ax_rdf, pad=0.01).set_label("Window start (ns)")

        ax_std.fill_between(r_mid, 0, std_gr, alpha=0.4, color="steelblue", label="std")
        ax_std.plot(r_mid, std_gr, color="steelblue", lw=1.2)
        ax_std.set_ylabel(r"$\sigma[g(r)]$")
        ax_std.set_xlabel(r"$r$ (Å)")
        ax_std.set_xlim(0, R_MAX)
        ax_std.legend(fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.93))

        out_gr = out_dir / f"{slug}__gr__{pair_label}.png"
        fig.savefig(out_gr, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_gr}")

        # ── n(r) plot ──────────────────────────────────────────────────────────
        fig2, (ax_nr, ax_nstd) = plt.subplots(
            2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [3, 1.2]}
        )
        fig2.suptitle(
            f"{model_lbl.upper()}  |  {sys_name}  |  {pair_label}\n"
            f"Sliding window n(r)  ({WINDOW_NS} ns window, {SLIDE_NS} ns step)",
            fontsize=11, fontweight="bold",
        )
        for wi, nw in enumerate(all_nr):
            ax_nr.plot(r_mid, nw, color=cmap(wi / max(n_win - 1, 1)),
                       lw=0.8, alpha=0.85)
        ax_nr.plot(r_mid, mean_nr, color="black", lw=1.8, ls="--", label="mean")
        ax_nr.set_ylabel(r"$n(r)$")
        ax_nr.set_xlim(0, R_MAX)
        ax_nr.legend(fontsize=9)
        sm2 = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=t_starts[0], vmax=t_starts[-1])
        )
        sm2.set_array([])
        fig2.colorbar(sm2, ax=ax_nr, pad=0.01).set_label("Window start (ns)")

        ax_nstd.fill_between(r_mid, 0, std_nr, alpha=0.4, color="darkorange", label="std")
        ax_nstd.plot(r_mid, std_nr, color="darkorange", lw=1.2)
        ax_nstd.set_ylabel(r"$\sigma[n(r)]$")
        ax_nstd.set_xlabel(r"$r$ (Å)")
        ax_nstd.set_xlim(0, R_MAX)
        ax_nstd.legend(fontsize=9)
        fig2.tight_layout(rect=(0, 0, 1, 0.93))

        out_nr = out_dir / f"{slug}__nr__{pair_label}.png"
        fig2.savefig(out_nr, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved: {out_nr}")

print("\nAll done.")
