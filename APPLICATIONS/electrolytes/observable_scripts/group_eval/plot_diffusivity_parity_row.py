"""Single-row (cation | anion | solvent) diffusivity parity figures.

Builds two 1x3 parity figures from a ``diffusivity_with_exp_all.csv`` that
already carries OPLS columns (``opls_D_{cat,ani,sol}_1e-10_m2s``):

  * PAINN vs experiment  -- per system: replica MEAN dot + [min, max] bar,
                            using the finite-size-corrected MD diffusivity.
  * OPLS  vs experiment  -- one dot per system (single OPLS value/system).

Same conventions as plot_diffusivity_parity_mean.py: cation/anion panels keep
only concentration_M >= 0.5 (too few ions at low conc), solvent uses all
systems; points are colored per (salt, solvent, concentration, temperature)
with one shared legend to the right of the row.

Usage:
  python plot_diffusivity_parity_row.py <diffusivity_with_exp_all.csv> [--output-dir <dir>]
"""
import argparse
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

# (panel title, exp column, PAINN sim column, OPLS column, min concentration_M)
SPECIES = [
    ("Cation",  "exp_D_cation_1e-10_m2s",  "D_cat_corrected_1e-10_m2s", "opls_D_cat_1e-10_m2s", 0.5),
    ("Anion",   "exp_D_anion_1e-10_m2s",   "D_ani_corrected_1e-10_m2s", "opls_D_ani_1e-10_m2s", 0.5),
    ("Solvent", "exp_D_solvent_1e-10_m2s", "D_sol_corrected_1e-10_m2s", "opls_D_sol_1e-10_m2s", 0.0),
]

AXIS = r"D ($\times10^{-10}$ m$^2$/s)"


def _label(row):
    return (f"{row['cat_symbol']}-{row['anion_symbol']}-{row['solvent_symbol']}, "
            f"{float(row['concentration_M']):g}M, {float(row['temperature_K']):.0f}K")


def _system_rows(df, exp_col, val_col, min_conc):
    """One row per system: experimental value + mean/min/max of val_col across replicas."""
    df = df[df[exp_col].notna() & (df[exp_col] > 0)]
    if min_conc > 0:
        df = df[df["concentration_M"] >= min_conc]
    rows = []
    for system, sub in df.groupby("system", sort=False):
        exp = float(sub[exp_col].iloc[0])
        if not np.isfinite(exp) or exp <= 0:
            continue
        vals = sub[val_col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        f = sub.iloc[0]
        rows.append({
            "exp": exp, "mean": float(np.mean(vals)),
            "lo": float(np.min(vals)), "hi": float(np.max(vals)),
            "label": _label(f),
        })
    return pd.DataFrame(rows)


def plot_row(csv_path, mode, output, log_scale):
    """mode: 'painn' (replica mean +/- min..max) or 'opls' (single value/system)."""
    df = pd.read_csv(csv_path)

    # shared color map over every system that appears in any panel
    labels = sorted({_label(r) for _, r in df.iterrows()})
    cmap = plt.get_cmap("tab20" if len(labels) > 10 else "tab10")
    color_of = {lbl: cmap(i % cmap.N) for i, lbl in enumerate(labels)}

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
    src = "PAINN" if mode == "painn" else "OPLS"
    show_bars = mode == "painn"

    for ax, (title, exp_col, painn_col, opls_col, min_conc) in zip(axes, SPECIES):
        val_col = painn_col if mode == "painn" else opls_col
        agg = _system_rows(df, exp_col, val_col, min_conc)

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
        conc_note = "conc>=0.5M" if min_conc > 0 else "all conc"
        ax.text(0.97, 0.03, f"n={n}, {conc_note}\nlogMAE={log_mae:.2f}\nSpearman={rho:.2f}",
                transform=ax.transAxes, va="bottom", ha="right", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel(f"Experimental {AXIS}", fontsize=10)
        ax.set_ylabel(f"{src} {AXIS}", fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)

    handles = [Line2D([0], [0], linestyle="--", color="k", lw=1.0, label="1:1")]
    handles += [Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color_of[l],
                       markeredgecolor="gray", markersize=7, label=l) for l in labels]
    fig.legend(handles=handles, fontsize=7.5, loc="center left",
               bbox_to_anchor=(0.99, 0.5), borderaxespad=0.0)

    scale = "log" if log_scale else "linear"
    bar = " (replica mean +/- min..max, finite-size corrected)" if show_bars else ""
    fig.suptitle(f"{src} vs experiment - diffusivity ({scale}){bar}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.99, 1))
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {output.name}")


def run(csv_path, output_dir=None):
    csv_path = Path(csv_path)
    output_dir = Path(output_dir) if output_dir else csv_path.parent
    for mode in ("painn", "opls"):
        for log_scale, suffix in [(True, "log"), (False, "linear")]:
            plot_row(csv_path, mode,
                     output_dir / f"diffusivity_parity_row_{mode}_vs_exp_{suffix}.png",
                     log_scale=log_scale)
    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--output-dir", default=None)
    a = p.parse_args()
    run(a.csv, a.output_dir)
