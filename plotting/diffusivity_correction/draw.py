#!/usr/bin/env python3
"""Parity plots for diffusivity correction study."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "main - correction.csv"
OUT_DIR = SCRIPT_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)

# Groups: methods vs experiment
METHODS = [
    "student before correction",
    "student after correction",
    "OPLS",
]

X_AXIS_LABEL = "experiment"
Y_AXIS_LABELS = [
    "293K 100ps student",
    "293K 100ps student corrected",
    "OPLS",
]

# Column mapping by species
COL_MAP = {
    "cation": {
        "student before correction": "293K 100ps student cation diffusivity(17ns)",
        "student after correction": "293K 100ps student cation diffusivity(17ns)correction",
        "OPLS": "OPLS cation diffusivity",
        "exp": "exp student cation diffusivity",
    },
    "anion": {
        "student before correction": "293K 100ps student anion diffusivity (17ns)",
        "student after correction": "293K 100ps student anion diffusivity (17ns)correction",
        "OPLS": "OPLS anion diffusivity",
        "exp": "exp anion diffusivity",
    },
    "solvent": {
        "student before correction": "293K 100ps student solvent diffusivity (17ns)",
        "student after correction": "293K 100ps student solvent diffusivity (17ns)correction",
        "OPLS": "OPLS student solvent diffusivity",
        "exp": "exp solvent diffusivity",
    },
}


def plot_pairwise_parity(ax, x, y, xlabel, ylabel, colors=None):
    """Scatter plot with Pearson r, MAE, RMSE and parity line."""
    OUTLIER_THRESHOLD = 1000.0

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x <= OUTLIER_THRESHOLD) & (y <= OUTLIER_THRESHOLD)
    x_clean = x[mask]
    y_clean = y[mask]

    if colors is not None:
        c = np.asarray(colors)[mask]
        for sp, col in [("cation", "C0"), ("anion", "C1"), ("solvent", "C2")]:
            m = c == sp
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
    else:
        ax.scatter(x_clean, y_clean, s=60, alpha=0.8, edgecolor="k", linewidth=0.5)

    all_vals = np.concatenate([x_clean, y_clean])
    lo = max(0, all_vals.min() * 0.95) if all_vals.size else 0
    hi = all_vals.max() * 1.05 if all_vals.size else 1
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


def main():
    df = pd.read_csv(CSV_PATH)

    # Merge cation, anion, solvent into one vector for each group
    data = {}
    for group in METHODS + ["exp"]:
        vals = []
        for species in ["cation", "anion", "solvent"]:
            col = COL_MAP[species][group]
            v = pd.to_numeric(df[col], errors="coerce")
            vals.extend(v.tolist())
        data[group] = np.array(vals)

    colors = np.array([sp for sp in ["cation", "anion", "solvent"] for _ in range(len(df))])

    n_methods = len(METHODS)
    ncols = 3
    nrows = (n_methods + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, method in enumerate(METHODS):
        ax = axes_flat[idx]
        plot_pairwise_parity(
            ax,
            x=data["exp"],
            y=data[method],
            xlabel=X_AXIS_LABEL,
            ylabel=Y_AXIS_LABELS[idx] if idx < len(Y_AXIS_LABELS) else method,
            colors=colors,
        )
        if idx == 0:
            ax.legend(loc="lower right", fontsize=12)

    for idx in range(n_methods, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Cation + Anion + Solvent diffusivity (×10⁻¹⁰ m²/s)", fontsize=14)
    plt.tight_layout()

    out_path = OUT_DIR / "parity_before_after_opls.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
