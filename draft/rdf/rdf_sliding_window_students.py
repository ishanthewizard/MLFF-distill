#!/usr/bin/env python
"""Sliding-window RDF analysis for student model trajectories.

For each system × student × RDF pair, divides the 20 ns trajectory into
overlapping windows (0.9 ns wide, sliding every 1 ns) and computes g(r)
per window.  Plots:
  - top subplot : all g(r) curves overlaid, blue→red (early→late), with colorbar
  - bottom subplot: std(g(r)) as shaded band across windows

Prints a summary table of first-peak position/height per window and suggests
an equilibration cutoff where the std band drops below 5 % of the mean.

env: /global/homes/y/yuejian/project/MLFF-distill/yuejian/envs/fairchemV2_new
"""

import os
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
WINDOW_NS   = 0.9    # window width (ns)
SLIDE_NS    = 1.0    # slide step  (ns)
SKIP_NS     = 0.1    # skip from start (equilibration)
TOTAL_NS    = 20.0   # total trajectory length
N_FRAMES_PER_WINDOW = 900   # frames sampled per window

R_MAX = 10.0
DR    = 0.05

STUDENT_DT_FS = 100.0   # fs per frame

STD_THRESH_FRAC = 0.05  # 5 % of mean → equilibrated

OUTPUT_DIR = Path("/global/homes/y/yuejian/project/MLFF-distill/yuejian/3d_turbulence/rdf/sliding_window")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
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
        "rdf_pairs": [("Na", "O"), ("Na", "F")],
    },
    {
        "name": "NaOTf/DME 0.1M",
        "conc_subpath": "20ns_solvent_0_1M",
        "temp_subpaths": {"micro": "", "original": "298K"},
        "system_dir": "md_omol_naotf_dme_s1p1_omol",
        "traj_name": "md_omol_naotf_dme_s1p1_omol.traj",
        "rdf_pairs": [("Na", "S"), ("Na", "O")],
    },
    {
        "name": "LiPF6/DME 0.5M",
        "conc_subpath": "20ns_solvent_solute_0.5M",
        "temp_subpaths": {"micro": "298_2K", "original": "298_2K"},
        "system_dir": "md_omol_lipf6_pfactor_0.1_1fs_mask_t",
        "traj_name": "md_omol_lipf6_pfactor_0.1_1fs_mask_t.traj",
        "rdf_pairs": [("Li", "O"), ("Li", "F")],
    },
    {
        "name": "NaPF6/DME 0.5M",
        "conc_subpath": "20ns_solvent_solute_0.5M",
        "temp_subpaths": {"micro": "298_2K", "original": "298_2K"},
        "system_dir": "md_omol_napf6_dme_re1",
        "traj_name": "md_omol_napf6_dme_re1.traj",
        "rdf_pairs": [("Na", "O"), ("Na", "F")],
    },
    {
        "name": "NaOTf/DME 1M",
        "conc_subpath": "20ns_solute_solvent_1M",
        "temp_subpaths": {"micro": "298K", "original": "298K"},
        "system_dir": "naotf_dme",
        "traj_name": "naotf_dme.traj",
        "rdf_pairs": [("Na", "S"), ("Na", "O")],
    },
    {
        "name": "NaPF6/DME 1M",
        "conc_subpath": "20ns_solute_solvent_1M",
        "temp_subpaths": {"micro": "298K", "original": "298K"},
        "system_dir": "napf6_dme",
        "traj_name": "napf6_dme.traj",
        "rdf_pairs": [("Na", "O"), ("Na", "F")],
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


# Build paths
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


# ── Window definitions ────────────────────────────────────────────────────────

def get_windows(dt_fs=STUDENT_DT_FS, total_ns=TOTAL_NS,
                skip_ns=SKIP_NS, window_ns=WINDOW_NS, slide_ns=SLIDE_NS):
    """Return list of (start_ns, end_ns, frame_start, frame_end) for each window."""
    windows = []
    t = skip_ns
    while t + window_ns <= total_ns + 1e-9:
        t_end = min(t + window_ns, total_ns)
        f_start = int(round(t     * 1e6 / dt_fs))
        f_end   = int(round(t_end * 1e6 / dt_fs))
        windows.append((t, t_end, f_start, f_end))
        t += slide_ns
    return windows


# ── RDF for one window ────────────────────────────────────────────────────────

def compute_rdf_window(traj, f_start, f_end, cation, partner,
                       n_frames=N_FRAMES_PER_WINDOW, r_max=R_MAX, dr=DR):
    bins    = np.arange(0.0, r_max + dr, dr)
    r_mid   = 0.5 * (bins[:-1] + bins[1:])
    shell_v = 4.0 / 3.0 * pi * (bins[1:] ** 3 - bins[:-1] ** 3)

    length  = f_end - f_start
    stride  = max(1, length // n_frames)
    indices = range(f_start, f_end, stride)

    hist        = np.zeros(len(r_mid))
    n_cat_total = 0.0
    n_part_total = 0.0
    vol_sum     = 0.0
    frame_count = 0

    for idx in indices:
        at   = traj[idx]
        syms = at.get_chemical_symbols()
        idx_cat  = [i for i, s in enumerate(syms) if s == cation]
        idx_part = [i for i, s in enumerate(syms) if s == partner]
        if not idx_cat or not idx_part:
            continue

        pos  = at.get_positions()
        cell = at.get_cell()
        pbc  = at.get_pbc()

        for rc in pos[idx_cat]:
            disp, _ = find_mic(pos[idx_part] - rc, cell, pbc)
            d = np.linalg.norm(disp, axis=1)
            hist += np.histogram(d, bins=bins)[0]

        n_cat_total  += len(idx_cat)
        n_part_total += len(idx_part)
        vol_sum      += at.get_volume()
        frame_count  += 1

    if frame_count == 0:
        return r_mid, np.zeros(len(r_mid)), np.zeros(len(r_mid))

    rho_part       = (n_part_total / frame_count) / (vol_sum / frame_count)
    counts_per_cat = hist / n_cat_total
    g_r            = counts_per_cat / (rho_part * shell_v)
    n_r            = np.cumsum(counts_per_cat)
    return r_mid, g_r, n_r


def first_peak(r, g_r, r_min=1.5):
    mask = r >= r_min
    idx  = np.argmax(g_r[mask])
    r_arr  = r[mask]
    g_arr  = g_r[mask]
    return float(r_arr[idx]), float(g_arr[idx])


# ── Per-system, per-model, per-pair analysis ──────────────────────────────────

windows = get_windows()
print(f"Sliding windows: {len(windows)} windows of {WINDOW_NS} ns, step {SLIDE_NS} ns")
print(f"  first window: {windows[0][0]:.1f}–{windows[0][1]:.1f} ns")
print(f"  last  window: {windows[-1][0]:.1f}–{windows[-1][1]:.1f} ns\n")

for sys in SYSTEMS:
    sys_name = sys["name"]
    for model_label, traj_path in sys["student_paths"].items():
        if not traj_path.exists():
            print(f"SKIP (missing): {sys_name} / {model_label}")
            continue

        print(f"\n{'='*60}")
        print(f"  {sys_name}  |  {model_label}")
        print(f"  {traj_path}")
        print(f"{'='*60}")

        n_total_frames = len(Trajectory(str(traj_path), mode="r"))

        for cation, partner in sys["rdf_pairs"]:
            pair_label = f"{cation}-{partner}"
            print(f"\n  Pair: {pair_label}")

            all_gr   = []    # shape: (n_windows, n_bins)
            all_nr   = []
            r_mid    = None
            summary_rows = []

            # open a fresh handle per pair to avoid stale file descriptor
            traj = Trajectory(str(traj_path), mode="r")

            for w_idx, (t_start, t_end, f_start, f_end) in enumerate(
                tqdm(windows, desc=f"    windows", leave=False)
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
                # coordination number at first-shell cutoff (first minimum after first peak)
                pk_idx = np.argmax(g[r >= 1.5]) + np.searchsorted(r, 1.5)
                after_peak = g[pk_idx:]
                r_after    = r[pk_idx:]
                min_idx    = np.argmin(after_peak) + pk_idx if len(after_peak) else pk_idx
                cn_cutoff  = float(n[min_idx])
                summary_rows.append({
                    "Window": w_idx + 1,
                    "Start (ns)":   t_start,
                    "End (ns)":     t_end,
                    "Peak r (Å)":   round(pk_r, 3),
                    "Peak g(r)":    round(pk_h, 3),
                    "Shell cut (Å)": round(float(r[min_idx]), 3),
                    "CN":           round(cn_cutoff, 3),
                })

            traj.close()

            if not all_gr:
                print("    No windows computed – skipping.")
                continue

            all_gr    = np.array(all_gr)          # (n_windows, n_bins)
            all_nr    = np.array(all_nr)
            mean_gr   = all_gr.mean(axis=0)
            std_gr    = all_gr.std(axis=0, ddof=1)
            mean_nr   = all_nr.mean(axis=0)
            std_nr    = all_nr.std(axis=0, ddof=1)
            n_windows = all_gr.shape[0]

            # ── Summary table ──
            df_sum = pd.DataFrame(summary_rows)
            print("\n  Window summary (g(r) peak + coordination number):")
            print(df_sum.to_string(index=False))

            # ── Equilibration cutoff ──
            # find first window where cumulative std (over remaining windows) < threshold
            cutoff_ns = None
            for w in range(n_windows):
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
                print(f"\n  Equilibration cutoff suggestion: start analysis from {cutoff_ns:.1f} ns "
                      f"(std/mean < {STD_THRESH_FRAC*100:.0f}% from that window onward)")
            else:
                print(f"\n  Equilibration cutoff: std/mean never drops below {STD_THRESH_FRAC*100:.0f}% "
                      f"– trajectory may not be fully equilibrated for this pair.")

            # ── Plot ──
            cmap      = cm.get_cmap("coolwarm", n_windows)
            t_starts  = np.array([w[0] for w in windows[:n_windows]])

            fig, (ax_rdf, ax_std) = plt.subplots(
                2, 1, figsize=(9, 8),
                gridspec_kw={"height_ratios": [3, 1.2]},
            )
            fig.suptitle(
                f"{sys_name}  |  {model_label}  |  {pair_label}\n"
                f"Sliding window RDF  ({WINDOW_NS} ns window, {SLIDE_NS} ns step)",
                fontsize=12, fontweight="bold",
            )

            # top: overlaid windows
            for w_idx, g in enumerate(all_gr):
                color = cmap(w_idx / max(n_windows - 1, 1))
                ax_rdf.plot(r_mid, g, color=color, lw=0.8, alpha=0.85)

            ax_rdf.plot(r_mid, mean_gr, color="black", lw=1.8, ls="--", label="mean")
            ax_rdf.set_ylabel(r"$g(r)$")
            ax_rdf.set_xlim(0, R_MAX)
            ax_rdf.legend(fontsize=9)

            # colorbar
            sm = plt.cm.ScalarMappable(
                cmap=cmap,
                norm=plt.Normalize(vmin=t_starts[0], vmax=t_starts[-1]),
            )
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax_rdf, pad=0.01)
            cb.set_label("Window start time (ns)")

            # bottom: std band
            ax_std.fill_between(r_mid, 0, std_gr, alpha=0.4, color="steelblue", label="std")
            ax_std.plot(r_mid, std_gr, color="steelblue", lw=1.2)
            ax_std.set_ylabel(r"$\sigma[g(r)]$")
            ax_std.set_xlabel(r"$r$ (Å)")
            ax_std.set_xlim(0, R_MAX)
            ax_std.legend(fontsize=9)

            fig.tight_layout(rect=(0, 0, 1, 0.93))

            safe_sys   = sys_name.replace("/", "_").replace(" ", "_")
            safe_model = model_label.replace(" ", "_")
            out_png    = OUTPUT_DIR / f"sliding_{safe_sys}_{safe_model}_{pair_label}.png"
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {out_png}")

            # ── n(r) plot ──
            fig2, (ax_nr, ax_nstd) = plt.subplots(
                2, 1, figsize=(9, 8),
                gridspec_kw={"height_ratios": [3, 1.2]},
            )
            fig2.suptitle(
                f"{sys_name}  |  {model_label}  |  {pair_label}\n"
                f"Sliding window coordination number  ({WINDOW_NS} ns window, {SLIDE_NS} ns step)",
                fontsize=12, fontweight="bold",
            )

            for w_idx, nw in enumerate(all_nr):
                color = cmap(w_idx / max(n_windows - 1, 1))
                ax_nr.plot(r_mid, nw, color=color, lw=0.8, alpha=0.85)

            ax_nr.plot(r_mid, mean_nr, color="black", lw=1.8, ls="--", label="mean")
            ax_nr.set_ylabel(r"$n(r)$")
            ax_nr.set_xlim(0, R_MAX)
            ax_nr.legend(fontsize=9)

            sm2 = plt.cm.ScalarMappable(
                cmap=cmap,
                norm=plt.Normalize(vmin=t_starts[0], vmax=t_starts[-1]),
            )
            sm2.set_array([])
            cb2 = fig2.colorbar(sm2, ax=ax_nr, pad=0.01)
            cb2.set_label("Window start time (ns)")

            ax_nstd.fill_between(r_mid, 0, std_nr, alpha=0.4, color="darkorange", label="std")
            ax_nstd.plot(r_mid, std_nr, color="darkorange", lw=1.2)
            ax_nstd.set_ylabel(r"$\sigma[n(r)]$")
            ax_nstd.set_xlabel(r"$r$ (Å)")
            ax_nstd.set_xlim(0, R_MAX)
            ax_nstd.legend(fontsize=9)

            fig2.tight_layout(rect=(0, 0, 1, 0.93))
            out_nr_png = OUTPUT_DIR / f"sliding_nr_{safe_sys}_{safe_model}_{pair_label}.png"
            fig2.savefig(out_nr_png, dpi=150, bbox_inches="tight")
            plt.close(fig2)
            print(f"  Saved: {out_nr_png}")

print("\nAll done.")
