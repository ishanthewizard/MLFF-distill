#!/usr/bin/env python3
"""Generic group parity plot across multiple "*_parity.csv" files.

Each input csv is expected to have one column with experimental values
and one (or more) columns with simulated values, plus enough metadata to
identify points (e.g. system/source/ensemble). Rows where the
experimental column is missing/non-positive are dropped before plotting.

Each csv is plotted with its own color/label so multiple sources (e.g.
OPLS npt, OPLS nvt, PAINN tf32) can be overlaid on the same log-log axes.
MAE (in log10 space) and Spearman rank correlation are reported per-csv
and overall.

This is intentionally observable-agnostic: it just needs an
``exp_col`` / ``sim_col`` pair present in every csv (e.g.
``exp_conductivity_uS_cm`` / ``sim_conductivity_onsager_uS_cm`` for
conductivity, or analogous columns for density/viscosity parity csvs).

Per-property defaults (column names, axis labels, log/linear scale) are
defined in ``properties.py`` and selected via ``--property``; pass
``--exp-col``/``--sim-col``/etc. to override or to plot a property not
yet listed there.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from properties import PROPERTIES


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


def plot_group_parity(
    csv_paths,
    exp_col,
    sim_col,
    labels=None,
    output=None,
    xlabel=None,
    ylabel=None,
    title=None,
    log_scale=True,
    annotate_meta=False,
):
    """Overlay parity scatter from multiple csvs and report MAE/Spearman.

    Parameters
    ----------
    csv_paths : list of paths to "*_parity.csv" files (same schema)
    exp_col   : column name with experimental values
    sim_col   : column name with simulated values
    labels    : legend labels per csv (default: filename stem)
    output    : output PNG path (default: alongside first csv)
    xlabel, ylabel, title : plot labels (sensible defaults if None)
    log_scale : use log-log axes (recommended for conductivity/diffusivity)
    annotate_meta : if True and a single csv has cation/anion/solvent/
        concentration/temperature_K columns, color points by
        cation-anion-solvent system (with a legend) and annotate each
        point with its concentration/temperature.

    Returns
    -------
    Path to saved PNG file.
    """
    csv_paths = [Path(p) for p in csv_paths]
    if labels is None:
        labels = [p.stem for p in csv_paths]
    assert len(labels) == len(csv_paths)

    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p)
        df = df[df[exp_col].notna() & (df[exp_col] > 0)].copy()
        dfs.append(df)

    all_vals = []
    for df in dfs:
        all_vals += list(df[exp_col].dropna())
        all_vals += list(df[sim_col].dropna())

    fig, ax = plt.subplots(figsize=(7, 7))

    if log_scale:
        lo, hi = _log_lims(all_vals)
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        finite = [v for v in all_vals if np.isfinite(v)]
        lo, hi = (0, max(finite) * 1.05) if finite else (0, 1)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="1:1", zorder=1)

    # Two naming schemes seen across parity csvs: conductivity_parity_*.csv
    # uses cation/anion/solvent/concentration, diffusivity_with_exp_all.csv
    # uses cat_symbol/anion_symbol/solvent_symbol/concentration_M.
    meta_col_sets = [
        ("cation", "anion", "solvent", "concentration", "temperature_K"),
        ("cat_symbol", "anion_symbol", "solvent_symbol", "concentration_M", "temperature_K"),
    ]
    meta_cols = None
    if annotate_meta and len(dfs) == 1:
        for cols in meta_col_sets:
            if all(c in dfs[0].columns for c in cols):
                meta_cols = cols
                break
    use_meta = meta_cols is not None

    cmap = plt.get_cmap("tab10")
    text_lines = []
    all_exp, all_sim = [], []
    if use_meta:
        df = dfs[0]
        c_cat, c_an, c_sol, c_conc, c_temp = meta_cols
        sys_keys = df[c_cat] + "-" + df[c_an] + "-" + df[c_sol]
        uniq_sys = sorted(sys_keys.unique())
        sys_cmap = plt.get_cmap("tab10" if len(uniq_sys) <= 10 else "tab20")
        sys_color = {s: sys_cmap(i % sys_cmap.N) for i, s in enumerate(uniq_sys)}
        for _, row in df.iterrows():
            sys_name = f"{row[c_cat]}-{row[c_an]}-{row[c_sol]}"
            point_label = f"{sys_name}, {row[c_conc]:g}M, {row[c_temp]:.0f}K"
            ax.scatter(row[exp_col], row[sim_col],
                       color=sys_color[sys_name], s=60, alpha=0.8,
                       edgecolors="gray", linewidths=0.5, label=point_label, zorder=3)
        log_mae, rho, n = _stats(df[exp_col], df[sim_col])
        text_lines.append(f"{labels[0]} (n={n}): logMAE={log_mae:.2f}, Spearman={rho:.2f}")
        all_exp += list(df[exp_col])
        all_sim += list(df[sim_col])
    else:
        for i, (df, label) in enumerate(zip(dfs, labels)):
            color = cmap(i % 10)
            ax.scatter(df[exp_col], df[sim_col],
                       color=color, s=60, alpha=0.8, edgecolors="gray",
                       linewidths=0.5, label=label, zorder=3)
            log_mae, rho, n = _stats(df[exp_col], df[sim_col])
            text_lines.append(f"{label} (n={n}): logMAE={log_mae:.2f}, Spearman={rho:.2f}")
            all_exp += list(df[exp_col])
            all_sim += list(df[sim_col])

    log_mae_all, rho_all, n_all = _stats(all_exp, all_sim)
    text_lines.append(f"all (n={n_all}): logMAE={log_mae_all:.2f}, Spearman={rho_all:.2f}")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel or f"Experimental {exp_col}", fontsize=11)
    ax.set_ylabel(ylabel or f"MD {sim_col}", fontsize=11)
    ax.set_title(title or "MD vs experiment", fontsize=12)
    if use_meta:
        ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  borderaxespad=0.0, ncol=1)
    else:
        ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.text(0.97, 0.03, "\n".join(text_lines), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    fig.tight_layout()
    if output is None:
        output = Path(csv_paths[0]).parent / "group_parity.png"
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
    parser.add_argument("csvs", nargs="+", help="*_parity.csv file(s)")
    parser.add_argument("--property", choices=sorted(PROPERTIES), default=None,
                         help="Look up exp/sim columns and labels from properties.py")
    parser.add_argument("--exp-col", default=None, help="Experimental value column name (overrides --property)")
    parser.add_argument("--sim-col", default=None, help="Simulated value column name (overrides --property)")
    parser.add_argument("--labels", nargs="+", default=None,
                         help="Labels for each csv (default: filename stem)")
    parser.add_argument("--output", default=None, help="Output PNG path")
    parser.add_argument("--xlabel", default=None)
    parser.add_argument("--ylabel", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--linear", action="store_true", help="Force linear axes instead of log-log")
    args = parser.parse_args()

    cfg = PROPERTIES.get(args.property, {}) if args.property else {}
    exp_col = args.exp_col or cfg.get("exp_col")
    sim_col = args.sim_col or cfg.get("sim_col")
    if exp_col is None or sim_col is None:
        parser.error("--exp-col/--sim-col are required unless --property is given")
    log_scale = cfg.get("log_scale", True)
    if args.linear:
        log_scale = False

    plot_group_parity(
        args.csvs, exp_col, sim_col, labels=args.labels, output=args.output,
        xlabel=args.xlabel or cfg.get("xlabel"),
        ylabel=args.ylabel or cfg.get("ylabel"),
        title=args.title or cfg.get("title"),
        log_scale=log_scale,
    )


if __name__ == "__main__":
    main()
