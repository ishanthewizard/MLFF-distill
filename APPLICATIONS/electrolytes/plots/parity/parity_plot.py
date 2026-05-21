#!/usr/bin/env python3
"""Parity plots for 0.5M dataset with two point-selection modes."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "main - 0.5M.csv"
OUT_DIR = SCRIPT_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)

GROUPS = ["50ps student", "100ps student"]
X_AXIS_LABEL = "experiment"
Y_AXIS_LABELS = ["293K 50ps student", "293K 100ps student"]

COL_MAP = {
    "cation": {
        "50ps student": "293K 50ps student cation diffusivity",
        "100ps student": "293K 100ps student cation diffusivity(17ns)",
        "exp": "exp student cation diffusivity",
    },
    "anion": {
        "50ps student": "293K 50ps student anion diffusivity",
        "100ps student": "293K 100ps student anion diffusivity (17ns)",
        "exp": "exp anion diffusivity",
    },
    "solvent": {
        "50ps student": "293K 50ps student solvent diffusivity",
        "100ps student": "293K 100ps student solvent diffusivity (17ns)",
        "exp": "exp solvent diffusivity",
    },
}


def plot_pairwise_parity(ax, x, y, xlabel, ylabel, colors):
    outlier_threshold = 1000.0

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    colors = np.asarray(colors)

    mask = np.isfinite(x) & np.isfinite(y) & (x <= outlier_threshold) & (y <= outlier_threshold)
    x_clean = x[mask]
    y_clean = y[mask]
    c_clean = colors[mask]

    for sp, col in [("cation", "C0"), ("anion", "C1"), ("solvent", "C2")]:
        m = c_clean == sp
        if m.any():
            ax.scatter(
                x_clean[m],
                y_clean[m],
                s=60,
                alpha=0.8,
                c=col,
                label=sp,
                edgecolor="k",
                linewidth=0.5,
            )

    all_vals = np.concatenate([x_clean, y_clean]) if len(x_clean) else np.array([0.0, 1.0])
    lo = max(0, all_vals.min() * 0.95)
    hi = all_vals.max() * 1.05 if all_vals.max() > 0 else 1
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, zorder=0)

    if len(x_clean) >= 2:
        r, _ = pearsonr(x_clean, y_clean)
        mae = np.mean(np.abs(y_clean - x_clean))
        rmse = np.sqrt(np.mean((y_clean - x_clean) ** 2))
        stats_text = f"r = {r:.3f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}"
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
        )

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def collect_points(df, group, row_mask):
    x_vals, y_vals, sp_vals = [], [], []
    for sp in ["cation", "anion", "solvent"]:
        x = pd.to_numeric(df.loc[row_mask, COL_MAP[sp]["exp"]], errors="coerce")
        y = pd.to_numeric(df.loc[row_mask, COL_MAP[sp][group]], errors="coerce")
        valid = x.notna() & y.notna()
        x_vals.extend(x[valid].tolist())
        y_vals.extend(y[valid].tolist())
        sp_vals.extend([sp] * int(valid.sum()))
    return np.array(x_vals), np.array(y_vals), np.array(sp_vals)


def make_figure(data_by_group, out_name):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes = np.atleast_1d(axes).flatten()

    for idx, group in enumerate(GROUPS):
        x, y, colors = data_by_group[group]
        plot_pairwise_parity(
            axes[idx],
            x=x,
            y=y,
            xlabel=X_AXIS_LABEL,
            ylabel=Y_AXIS_LABELS[idx],
            colors=colors,
        )
        if idx == 0:
            axes[idx].legend(loc="lower right", fontsize=12)

    fig.suptitle("Cation + Anion + Solvent diffusivity (×10⁻¹⁰ m²/s)", fontsize=14)
    plt.tight_layout()
    out_path = OUT_DIR / out_name
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main():
    df = pd.read_csv(CSV_PATH)

    # Plot 1: shared part only (true intersection between groups and experiment)
    shared_masks = []
    for sp in ["cation", "anion", "solvent"]:
        exp_col = pd.to_numeric(df[COL_MAP[sp]["exp"]], errors="coerce")
        g0_col = pd.to_numeric(df[COL_MAP[sp][GROUPS[0]]], errors="coerce")
        g1_col = pd.to_numeric(df[COL_MAP[sp][GROUPS[1]]], errors="coerce")
        shared_masks.append(exp_col.notna() & g0_col.notna() & g1_col.notna())
    shared_row_mask = shared_masks[0] | shared_masks[1] | shared_masks[2]

    data_shared = {g: collect_points(df, g, shared_row_mask) for g in GROUPS}
    make_figure(data_shared, "parity_li_shared_only.png")

    # Plot 2: each group uses all points it has
    all_mask = pd.Series(True, index=df.index)
    data_all = {g: collect_points(df, g, all_mask) for g in GROUPS}
    make_figure(data_all, "parity_all_available.png")


if __name__ == "__main__":
    main()
