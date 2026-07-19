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


def _safe(s):
    """Filesystem-safe version of a name: replace / and spaces with _."""
    return str(s).replace("/", "_").replace(" ", "_")


def plot_collective_vacf(times, acf_pp, acf_pm, acf_mm, system, model, out_dir,
                         dim=1):
    """Collective velocity ACF (++, +-, --) for one Cartesian component.

    Diagnostic figure produced by the mdcraft conductivity backend
    (``compute.run_onsager_conductivity_mdcraft``).  Saves a PNG into
    ``out_dir`` and returns its path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    ax.plot(times, acf_pp[:, dim], label="+ +")
    ax.plot(times, acf_pm[:, dim], label="+ -")
    ax.plot(times, acf_mm[:, dim], label="- -")
    ax.axhline(0, color="grey", lw=0.6)
    if len(times) > 1:
        ax.set_xlim(0, times[min(50, len(times) - 1)])
    ax.legend()
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Correlation function")
    ax.set_title(f"{system} / {model}: collective velocity ACF "
                 f"({'xyz'[dim]}-component)")
    fig.tight_layout()

    out = out_dir / f"conductivity_mdcraft_vacf_{_safe(model)}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out}")
    return out


def plot_cross_displacement(t, cross, pairs, s, e, kappa_uScm,
                            fit_start_ns, fit_stop_ns, system, model, out_dir):
    """Collective cross-displacement <dR_i . dR_j> vs lag, with linear fits.

    Diagnostic figure produced by the mdcraft conductivity backend.  ``cross``
    is the (3, nt) raw collective displacement (++, +-, --); ``pairs`` are the
    mdcraft ``results.pairs``; ``s``/``e`` bound the diffusive fit window.
    Saves a PNG into ``out_dir`` and returns its path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_lbl = {(0, 0): "++  (cation-cation)", (0, 1): "+-  (cation-anion)",
                (1, 1): "--  (anion-anion)"}
    colors = ["tab:blue", "tab:green", "tab:red"]

    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    for k, pr in enumerate(pairs):
        y = cross[k]
        ax.plot(t, y, lw=1.5, color=colors[k % 3],
                label=pair_lbl.get(tuple(pr), str(pr)))
        coef = np.polyfit(t[s:e], y[s:e], 1)          # linear fit over window
        ax.plot(t[s:e], np.polyval(coef, t[s:e]), "--", lw=1.2, color="k")
    ax.axvspan(t[s], t[e - 1], color="grey", alpha=0.12)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("t (ps)")
    ax.set_ylabel(r"collective $\langle \Delta R_i \cdot \Delta R_j \rangle$  ($\AA^2$)")
    ax.set_title(rf"{system} / {model}: cross-displacement "
                 rf"($\kappa$={kappa_uScm:.0f} $\mu$S/cm)")
    ax.legend(fontsize=8,
              title=f"fit {fit_start_ns}-{fit_stop_ns} ns (dashed = linear fit)")
    fig.tight_layout()

    out = out_dir / f"conductivity_mdcraft_cross_displacement_{_safe(model)}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out}")
    return out


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
