"""One-row conductivity (Onsager) parity figure: PAINN vs exp | OPLS vs exp.

Reads a ``conductivity_all.csv`` that already carries the OPLS column
``opls_sigma_onsager_mS_cm``. Two panels share one legend:

  * PAINN vs experiment -- per system: replica MEAN dot + [min, max] bar.
  * OPLS  vs experiment -- one dot per system.

Both use the Onsager conductivity (mS/cm) on y, exp_conductivity_mS_cm on x.
Points are colored per (salt, solvent, concentration, temperature); the CSV has
no concentration column so concentration is parsed from the system name.

Usage:
  python plot_conductivity_parity_row.py <conductivity_all.csv> [--output-dir <dir>]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from parity_plot import _stats, _log_lims

EXP_COL = "exp_conductivity_mS_cm"
AXIS = r"Onsager conductivity (mS/cm)"

# (panel source, value column, draws replica min..max bars)
PANELS = [
    ("PAINN", "sigma_onsager_mS_cm", True),
    ("OPLS", "opls_sigma_onsager_mS_cm", False),
]


def _conc(name):
    m = re.search(r"_(\d+(?:_\d+)?)M_\d+K", name)
    return float(m.group(1).replace("_", ".")) if m else float("nan")


def _label(row):
    return (f"{row['cat_symbol']}-{row['anion_symbol']}-{row['solvent_symbol']}, "
            f"{_conc(row['system']):g}M, {float(row['T_K']):.0f}K")


def _system_rows(df, val_col):
    """One row per system: exp value + mean/min/max of val_col across replicas."""
    df = df[df[EXP_COL].notna() & (df[EXP_COL] > 0)]
    rows = []
    for system, sub in df.groupby("system", sort=False):
        exp = float(sub[EXP_COL].iloc[0])
        if not np.isfinite(exp) or exp <= 0:
            continue
        vals = sub[val_col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        rows.append({
            "exp": exp, "mean": float(np.mean(vals)),
            "lo": float(np.min(vals)), "hi": float(np.max(vals)),
            "label": _label(sub.iloc[0]),
        })
    return pd.DataFrame(rows)


def plot_row(csv_path, output, log_scale):
    df = pd.read_csv(csv_path)

    labels = sorted({_label(r) for _, r in df.iterrows()})
    cmap = plt.get_cmap("tab20" if len(labels) > 10 else "tab10")
    color_of = {lbl: cmap(i % cmap.N) for i, lbl in enumerate(labels)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax, (src, val_col, show_bars) in zip(axes, PANELS):
        agg = _system_rows(df, val_col)

        vals = list(agg["exp"]) + list(agg["lo"]) + list(agg["hi"]) + list(agg["mean"])
        if log_scale:
            lo, hi = _log_lims(vals)
            ax.set_xscale("log"); ax.set_yscale("log")
        else:
            finite = [v for v in vals if np.isfinite(v)]
            lo, hi = (0, max(finite) * 1.05) if finite else (0, 1)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, zorder=1)

        for _, r in agg.iterrows():
            c = color_of[r["label"]]
            if show_bars:
                yerr = [[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]]
                ax.errorbar(r["exp"], r["mean"], yerr=yerr, fmt="o", ms=7,
                            color=c, ecolor=c, elinewidth=1.3, capsize=3,
                            capthick=1.3, alpha=0.9, markeredgecolor="gray",
                            markeredgewidth=0.5, zorder=3)
            else:
                ax.scatter(r["exp"], r["mean"], color=c, s=60, alpha=0.9,
                           edgecolors="gray", linewidths=0.5, zorder=3)

        log_mae, rho, n = _stats(agg["exp"], agg["mean"])
        ax.text(0.97, 0.03, f"n={n}\nlogMAE={log_mae:.2f}\nSpearman={rho:.2f}",
                transform=ax.transAxes, va="bottom", ha="right", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel(f"Experimental {AXIS}", fontsize=10)
        ax.set_ylabel(f"{src} {AXIS}", fontsize=10)
        bar = " (replica mean +/- min..max)" if show_bars else ""
        ax.set_title(f"{src} vs experiment{bar}", fontsize=12)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)

    handles = [Line2D([0], [0], linestyle="--", color="k", lw=1.0, label="1:1")]
    handles += [Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color_of[l],
                       markeredgecolor="gray", markersize=7, label=l) for l in labels]
    fig.legend(handles=handles, fontsize=7.5, loc="center left",
               bbox_to_anchor=(0.99, 0.5), borderaxespad=0.0)

    scale = "log" if log_scale else "linear"
    fig.suptitle(f"Onsager conductivity: model vs experiment ({scale})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.99, 1))
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {output.name}")


def run(csv_path, output_dir=None):
    csv_path = Path(csv_path)
    output_dir = Path(output_dir) if output_dir else csv_path.parent
    for log_scale, suffix in [(True, "log"), (False, "linear")]:
        plot_row(csv_path, output_dir / f"conductivity_parity_row_onsager_{suffix}.png", log_scale)
    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--output-dir", default=None)
    a = p.parse_args()
    run(a.csv, a.output_dir)
