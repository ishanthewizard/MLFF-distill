"""Direct model-vs-model self-diffusivity comparison: PaiNN (FP32, NVT) vs
OPLS-AA, no experiment axis.  x = OPLS raw D, y = PaiNN raw D (raw PBC for both --
OPLS has no Yeh-Hummer correction, so raw-vs-raw is apples-to-apples).

  2 rows x 3 cols.  Columns = cation / anion / solvent.
  Row 1 = all systems;  Row 2 = 1.0 M systems only.

Encoding matches the parity figures: solvent -> colour, salt -> marker,
concentration -> fill (0.1 M hollow).  Axes are square with identical x/y limits
per column so the dashed 1:1 line is a true diagonal.

    python plot_painn_vs_opls_diff.py --merged <PAINN merged> --opls <OPLS group_results>
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_group_transport_parity import (
    load_painn_diff, load_opls_diff, legend_panel,
    SOLV_COLOR, SALT_MARKER, _salt_solv, _fnum, D_UNIT,
)


def _merge_models(painn, opls):
    for df in (painn, opls):
        df["temperature_K"] = df["temperature_K"].round(1)
    return painn.merge(opls, on=["system_id", "concentration_M", "temperature_K"],
                       how="inner")


def _col_bounds(m, k):
    means, ups = [], []
    for _, r in m.iterrows():
        for mc, sc in [(f"opls_{k}_mean", f"opls_{k}_std"),
                       (f"painn_{k}_mean", f"painn_{k}_std")]:
            y, s = _fnum(r[mc]), _fnum(r[sc])
            if np.isfinite(y) and y > 0:
                means.append(y)
                ups.append(y + s if (np.isfinite(s) and s > 0) else y)
    if not means:
        return 0.1, 10.0
    return min(means) / 1.4, max(ups) * 1.4


def draw_mm_panel(ax, m, k, title, lo, hi, show_xlabel, ylabel=None):
    edge = max(lo, 1e-6)
    ax.fill_between([edge, hi], [edge / 2, hi / 2], [edge * 2, hi * 2],
                    color="0.8", alpha=0.30, lw=0, zorder=0)
    ax.plot([lo, hi], [lo, hi], color="0.45", ls="--", lw=1.3, zorder=1)

    ratios = []
    for _, r in m.iterrows():
        salt, solv = _salt_solv(r.system_id)
        if solv not in SOLV_COLOR or salt not in SALT_MARKER:
            continue
        c = SOLV_COLOR[solv]
        x, xs = _fnum(r[f"opls_{k}_mean"]), _fnum(r[f"opls_{k}_std"])
        y, ys = _fnum(r[f"painn_{k}_mean"]), _fnum(r[f"painn_{k}_std"])
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        filled = float(r.concentration_M) != 0.1
        ratios.append(y / x)

        def _err(v, s):  # clip lower whisker so it stays positive on a log axis
            if not (np.isfinite(s) and s > 0):
                return None
            return np.array([[min(s, 0.9 * v)], [s]])
        xe, ye = _err(x, xs), _err(y, ys)
        if xe is not None or ye is not None:
            ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="none", ecolor="0.4",
                        elinewidth=1.1, capsize=3.0, alpha=0.8, zorder=2.5)
        ax.scatter(x, y, s=120, marker=SALT_MARKER[salt],
                   facecolor=c if filled else "white", edgecolor=c,
                   linewidths=1.9, alpha=0.95, zorder=3)

    if ratios:
        gm = float(np.exp(np.mean(np.log(ratios))))
        ax.text(0.96, 0.05, f"PaiNN/OPLS\n{gm:.2f}×  (n={len(ratios)})",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5,
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.4",
                facecolor="white", edgecolor="0.55", linewidth=1.3, alpha=0.9), zorder=5)
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi))
    ax.set_aspect("equal")
    if show_xlabel:
        ax.set_xlabel(f"OPLS-AA  ({D_UNIT})")
    if ylabel:
        ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=8)
    ax.grid(True, which="major", ls=":", lw=0.7, alpha=0.55)
    ax.grid(True, which="minor", ls=":", lw=0.5, alpha=0.20)
    ax.tick_params(which="both", direction="in", top=True, right=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--opls", type=Path, required=True)
    ap.add_argument("--diff-mode", choices=["raw", "corrected"], default="corrected",
                    help="raw PBC or Yeh-Hummer corrected D for BOTH models (default corrected)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    m = _merge_models(load_painn_diff(args.merged, args.diff_mode),
                      load_opls_diff(args.opls, args.diff_mode))
    rows = [("all systems", m),
            ("1.0 M only", m[np.abs(m["concentration_M"] - 1.0) < 0.01].copy())]
    species = [("cat", r"Cation  $D^{+}$"), ("ani", r"Anion  $D^{-}$"),
               ("sol", r"Solvent  $D^{0}$")]

    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 15, "axes.labelsize": 12.5,
        "axes.linewidth": 0.9, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig = plt.figure(figsize=(15.5, 10.6))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.05],
                          left=0.075, right=0.995, top=0.90, bottom=0.075,
                          wspace=0.28, hspace=0.16)
    bounds = {k: _col_bounds(m, k) for k, _ in species}  # shared per-column (both rows)
    letters = iter("abcdef")
    for i, (rlab, sub) in enumerate(rows):
        for j, (k, title) in enumerate(species):
            ax = fig.add_subplot(gs[i, j])
            lo, hi = bounds[k]
            draw_mm_panel(ax, sub, k, title if i == 0 else "", lo, hi,
                          show_xlabel=(i == 1),
                          ylabel=(f"PaiNN (FP32)\n({D_UNIT})" if j == 0 else None))
            if j == 0:
                ax.text(0.05, 0.95, rlab, transform=ax.transAxes, va="top",
                        ha="left", fontsize=12, fontweight="bold", color="0.20",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor="0.6", linewidth=1.1, alpha=0.9))
            ax.annotate(next(letters), xy=(0, 1), xycoords="axes fraction",
                        xytext=(-46, 8), textcoords="offset points", fontsize=15,
                        fontweight="bold", va="bottom", ha="left")
    legend_panel(fig.add_subplot(gs[:, 3]), show_opls=False, col_a=0.05,
                 col_b=0.24, tx=0.36)

    dmode = "Yeh-Hummer corrected" if args.diff_mode == "corrected" else "raw PBC"
    fig.suptitle(f"Self-diffusivity: PaiNN (FP32, NVT) vs OPLS-AA  —  {dmode}, "
                 "298 K (+ LiPF$_6$/DME 273/323 K),  marker = system",
                 fontsize=13.5, fontweight="bold", y=0.965)
    out = args.out or (args.merged / "figure_diffusivity_painn_vs_opls_2row.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    print(f"saved -> {out}  and  {out.with_suffix('.pdf')}")
    plt.close(fig)


if __name__ == "__main__":
    main()
