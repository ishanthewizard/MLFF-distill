#!/usr/bin/env python3
"""Parity plot: simulated vs experimental ionic conductivity.

Reads one or more ``conductivity_parity.csv`` files (as produced by
``compute.py``), keeps only rows that have an experimental conductivity
value, and makes a log-log parity plot of
``sim_conductivity_onsager_uS_cm`` vs ``exp_conductivity_uS_cm``.

Each input csv is plotted with its own color/label so multiple sources
(e.g. OPLS npt, OPLS nvt, PAINN) can be overlaid on the same axes.  MAE
(in log10 space) and Spearman rank correlation are reported per-csv and
overall.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def _log_lims(vals, pad_factor=3.0):
    finite = [v for v in vals if v is not None and np.isfinite(v) and v > 0]
    if not finite:
        return 1e-2, 1e2
    return min(finite) / pad_factor, max(finite) * pad_factor


def _stats(exp_vals, sim_vals):
    exp_vals = np.asarray(exp_vals, dtype=float)
    sim_vals = np.asarray(sim_vals, dtype=float)
    mask = np.isfinite(exp_vals) & np.isfinite(sim_vals) & (exp_vals > 0) & (sim_vals > 0)
    if mask.sum() < 2:
        return np.nan, np.nan, int(mask.sum())
    log_mae = float(np.mean(np.abs(np.log10(sim_vals[mask]) - np.log10(exp_vals[mask]))))
    rho, _ = spearmanr(exp_vals[mask], sim_vals[mask])
    return log_mae, float(rho), int(mask.sum())


def plot_parity(csv_paths, labels=None, output=None, sim_col="sim_conductivity_onsager_uS_cm"):
    csv_paths = [Path(p) for p in csv_paths]
    if labels is None:
        labels = [p.stem for p in csv_paths]
    assert len(labels) == len(csv_paths)

    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p)
        df = df[df["exp_conductivity_uS_cm"].notna() & (df["exp_conductivity_uS_cm"] > 0)].copy()
        dfs.append(df)

    all_vals = []
    for df in dfs:
        all_vals += list(df["exp_conductivity_uS_cm"].dropna())
        all_vals += list(df[sim_col].dropna())
    lo, hi = _log_lims(all_vals)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="1:1", zorder=1)

    cmap = plt.get_cmap("tab10")
    text_lines = []
    all_exp, all_sim = [], []
    for i, (df, label) in enumerate(zip(dfs, labels)):
        color = cmap(i % 10)
        ax.scatter(df["exp_conductivity_uS_cm"], df[sim_col],
                   color=color, s=60, alpha=0.8, edgecolors="gray",
                   linewidths=0.5, label=label, zorder=3)
        log_mae, rho, n = _stats(df["exp_conductivity_uS_cm"], df[sim_col])
        text_lines.append(f"{label} (n={n}): logMAE={log_mae:.2f}, Spearman={rho:.2f}")
        all_exp += list(df["exp_conductivity_uS_cm"])
        all_sim += list(df[sim_col])

    log_mae_all, rho_all, n_all = _stats(all_exp, all_sim)
    text_lines.append(f"all (n={n_all}): logMAE={log_mae_all:.2f}, Spearman={rho_all:.2f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Experimental conductivity (uS/cm)", fontsize=11)
    ax.set_ylabel("MD Onsager conductivity (uS/cm)", fontsize=11)
    ax.set_title("Ionic conductivity: MD vs experiment", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.text(0.97, 0.03, "\n".join(text_lines), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    fig.tight_layout()
    if output is None:
        output = Path(csv_paths[0]).parent / "parity_conductivity.png"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot] saved {output}")
    for line in text_lines:
        print(f"  {line}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csvs", nargs="+", help="conductivity_parity.csv file(s)")
    parser.add_argument("--labels", nargs="+", default=None,
                         help="Labels for each csv (default: filename stem)")
    parser.add_argument("--output", default=None, help="Output PNG path")
    parser.add_argument("--sim-col", default="sim_conductivity_onsager_uS_cm",
                         choices=["sim_conductivity_onsager_uS_cm", "sim_conductivity_NE_uS_cm"])
    args = parser.parse_args()
    plot_parity(args.csvs, labels=args.labels, output=args.output, sim_col=args.sim_col)


if __name__ == "__main__":
    main()
