#!/usr/bin/env python3
"""Parity plots: three model groups vs experiment, from All_data - main_filtered.csv."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

CSV_PATH = Path("/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/csv/All_data - main_filtered.csv")
OUT_DIR  = CSV_PATH.parent

GROUPS = ["100ns original", "micro all conc"]
Y_LABELS = ["100ns original", "micro all conc"]

COL_MAP = {
    "cation": {
        "50ns student":   "293K student cation diffusivity 50ns",
        "100ns original": "293K student cation diffusivity 100ns",
        "micro all conc": "293K micro all concentration cation",
        "exp":            "exp student cation diffusivity",
    },
    "anion": {
        "50ns student":   "293K student anion diffusivity 50ns",
        "100ns original": "293K student anion diffusivity 100ns",
        "micro all conc": "293K micro all concentration anion",
        "exp":            "exp anion diffusivity",
    },
    "solvent": {
        "50ns student":   "293K student solvent diffusivity 50ns",
        "100ns original": "293K student solvent diffusivity 100ns",
        "micro all conc": "293K micro all concentration solvent",
        "exp":            "exp solvent diffusivity",
    },
}

SPECIES_COLORS = {"cation": "C0", "anion": "C1", "solvent": "C2"}
OUTLIER_THRESHOLD = 500.0


def collect_points(df, group):
    x_vals, y_vals, sp_vals = [], [], []
    for sp in ["cation", "anion", "solvent"]:
        x = pd.to_numeric(df[COL_MAP[sp]["exp"]], errors="coerce")
        y = pd.to_numeric(df[COL_MAP[sp][group]], errors="coerce")
        valid = x.notna() & y.notna()
        x_vals.extend(x[valid].tolist())
        y_vals.extend(y[valid].tolist())
        sp_vals.extend([sp] * int(valid.sum()))
    return np.array(x_vals), np.array(y_vals), np.array(sp_vals)


def plot_panel(ax, x, y, colors, ylabel, show_legend=False, loglog=False):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0) & (x <= OUTLIER_THRESHOLD) & (y <= OUTLIER_THRESHOLD)
    x, y, colors = x[mask], y[mask], np.asarray(colors)[mask]

    for sp in ["cation", "anion", "solvent"]:
        m = colors == sp
        if m.any():
            ax.scatter(x[m], y[m], s=60, alpha=0.85, c=SPECIES_COLORS[sp],
                       label=sp, edgecolor="k", linewidth=0.5)

    all_vals = np.concatenate([x, y]) if len(x) else np.array([0.1, 1.0])
    if loglog:
        lo = all_vals[all_vals > 0].min() * 0.7
        hi = all_vals.max() * 1.5
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        lo = max(0.0, all_vals.min() * 0.9)
        hi = all_vals.max() * 1.1 if all_vals.max() > 0 else 1.0
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, zorder=0)

    if len(x) >= 2:
        r, _ = pearsonr(np.log10(x) if loglog else x, np.log10(y) if loglog else y)
        mae  = np.mean(np.abs(y - x))
        rmse = np.sqrt(np.mean((y - x) ** 2))
        label = "r(log)" if loglog else "r"
        ax.text(0.05, 0.95, f"{label} = {r:.3f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))

    ax.set_xlabel("experiment  (×10⁻¹⁰ m²/s)", fontsize=11)
    ax.set_ylabel(f"{ylabel}  (×10⁻¹⁰ m²/s)", fontsize=11)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    if show_legend:
        ax.legend(loc="lower right", fontsize=10)


def make_figure(df, loglog=False):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    for idx, (group, ylabel) in enumerate(zip(GROUPS, Y_LABELS)):
        x, y, colors = collect_points(df, group)
        plot_panel(axes[idx], x, y, colors, ylabel, show_legend=(idx == 0), loglog=loglog)

    scale_tag = "log-log" if loglog else "linear"
    fig.suptitle(f"Diffusivity parity: model vs experiment  (cation · anion · solvent)  [{scale_tag}]",
                 fontsize=13)
    plt.tight_layout()

    suffix = "_loglog" if loglog else ""
    out_path = OUT_DIR / f"parity_all_groups_vs_exp{suffix}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


def main():
    df = pd.read_csv(CSV_PATH)
    make_figure(df, loglog=False)
    make_figure(df, loglog=True)


if __name__ == "__main__":
    main()
