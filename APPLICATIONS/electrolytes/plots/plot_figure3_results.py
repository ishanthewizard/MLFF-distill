"""Figure 2 (results): a single 2x3 parity figure for the distilled PaiNN models
vs experiment at 298 K.

  Row 1 : self-diffusivity parity for cation / anion / solvent -- per-condition
          Yeh-Hummer-corrected means with +/-1 s.d. error bars over the 4 replicas
          (the PBC -> inf shift stems are omitted to keep the panels uncluttered).
  Row 2 : ionic conductivity parity (Onsager/cluster, same legend); a viscosity
          panel left as a TODO placeholder; and the shared legend.

Diffusivity is read from ``diffusivity_by_system.csv`` (exp01 root). Conductivity
is read from ``src/analysis/observables/from_yue/conductivity_all.csv`` (per
replica) and aggregated to per-(system, concentration) mean +/- s.d. at 298 K.

    python -m src.analysis.plotting.plot_figure2_results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

ROOT = Path("/pscratch/sd/i/ishan_a/electrolytes_data/experiments/exp01_mixed_noise_all_systems")
REPO = Path(__file__).resolve().parents[3]
COND_CSV = REPO / "src/analysis/observables/from_yue/conductivity_all.csv"
OUT_DIR = REPO / "paper"

# encoding: solvent -> colour (vivid), salt -> marker shape, concentration -> fill
# (0.1 M hollow, else filled).
SOLV_COLOR = {"dme": "#E68310", "diglyme": "#11A579", "pc": "#E73F74", "tgdme": "#3969AC"}
SOLV_LABEL = {"dme": "DME", "diglyme": "DEGDME", "pc": "PC", "tgdme": "TEGDME"}
SALT_MARKER = {"napf6": "o", "naotf": "s", "lipf6": "^"}
SALT_LABEL = {"napf6": "NaPF$_6$", "naotf": "NaOTf", "lipf6": "LiPF$_6$"}

SPECIES = [  # (Yeh-Hummer-corrected mean, replica-std, exp, title)
    ("D_cation_inf_mean",  "D_cation_inf_std",  "D_cation_exp",  r"Cation self-diffusivity  $D^{+}$"),
    ("D_anion_inf_mean",   "D_anion_inf_std",   "D_anion_exp",   r"Anion self-diffusivity  $D^{-}$"),
    ("D_solvent_inf_mean", "D_solvent_inf_std", "D_solvent_exp", r"Solvent self-diffusivity  $D^{0}$"),
]
D_UNIT = r"$10^{-10}\,\mathrm{m^2\,s^{-1}}$"
S_UNIT = r"$\mathrm{mS\,cm^{-1}}$"


def _fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _bounds(m, mean_col, std_col, exp_col, pbc_col=None):
    vals = []
    for _, r in m.iterrows():
        cand = [_fnum(r[exp_col]), _fnum(r[mean_col])]
        if pbc_col:
            cand.append(_fnum(r[pbc_col]))
        ys, sd = _fnum(r[mean_col]), _fnum(r[std_col])
        if np.isfinite(ys) and np.isfinite(sd):
            cand += [ys - sd, ys + sd]
        vals += [v for v in cand if np.isfinite(v) and v > 0]
    return min(vals), max(vals)


def draw_panel(ax, m, mean_col, std_col, exp_col, title, xlabel, ylabel=None,
               pbc_col=None, log=True):
    """One parity panel, styled identically to plot_replica_parity."""
    vmin, vmax = _bounds(m, mean_col, std_col, exp_col, pbc_col)
    lo, hi = (vmin / 1.4, vmax * 1.4) if log else (0.0, vmax * 1.08)
    edge = max(lo, 1e-6)
    ax.fill_between([edge, hi], [edge / 2, hi / 2], [edge * 2, hi * 2],
                    color="0.8", alpha=0.30, lw=0, zorder=0)
    ax.plot([lo, hi], [lo, hi], color="0.45", ls="--", lw=1.3, zorder=1)

    devs = []
    for _, r in m.iterrows():
        salt, solv = r.system_id.split("_")[0], r.system_id.split("_")[1]
        c = SOLV_COLOR[solv]
        xe, ys, sd = _fnum(r[exp_col]), _fnum(r[mean_col]), _fnum(r[std_col])
        if not np.isfinite(xe) or not np.isfinite(ys):
            continue
        devs.append(abs(ys - xe) / xe)
        if pbc_col:                                   # Yeh-Hummer shift stem
            yr = _fnum(r[pbc_col])
            if np.isfinite(yr):
                xd = xe / 1.055
                ax.plot([xd, xd], [yr, ys], color=c, lw=1.5,
                        alpha=0.55, solid_capstyle="round", zorder=2)
                ax.scatter(xd, yr, s=22, marker="_", color=c,
                           linewidths=1.5, alpha=0.7, zorder=2.1)
        if np.isfinite(sd) and sd > 0:                # +/-1 s.d. over replicas
            ax.errorbar(xe, ys, yerr=sd, fmt="none", ecolor="0.35", elinewidth=1.2,
                        capsize=3.5, capthick=1.2, alpha=0.85, zorder=2.6)
        filled = float(r.concentration_M) != 0.1      # 0.1 M open, else filled
        ax.scatter(xe, ys, s=120, marker=SALT_MARKER[salt],
                   facecolor=c if filled else "white",
                   edgecolor=c, linewidths=1.9, alpha=0.95, zorder=3)

    if devs:
        ax.text(0.05, 0.95, f"MAPD = {100*np.mean(devs):.0f}%\n(n = {len(devs)})",
                transform=ax.transAxes, ha="left", va="top", fontsize=10.5,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="0.7", alpha=0.85), zorder=5)
    if log:
        ax.set(xscale="log", yscale="log")
    ax.set(xlim=(lo, hi), ylim=(lo, hi))
    ax.set_aspect("equal")
    ax.set_xlabel(f"Experiment  ({xlabel})")
    if ylabel:
        ax.set_ylabel(f"Simulation  ({ylabel})")
    ax.set_title(title, fontweight="bold", pad=8)
    ax.grid(True, which="major", ls=":", lw=0.7, alpha=0.55)
    ax.grid(True, which="minor", ls=":", lw=0.5, alpha=0.20)
    ax.tick_params(which="both", direction="in", top=True, right=True)


def todo_panel(ax, title, xlabel, ylabel):
    """Placeholder panel for a not-yet-computed observable."""
    ax.fill_between([1e-2, 1e1], [5e-3, 5e0], [2e-2, 2e1], color="0.8", alpha=0.18, lw=0)
    ax.plot([1e-2, 1e1], [1e-2, 1e1], color="0.6", ls="--", lw=1.2, alpha=0.5)
    ax.set(xscale="log", yscale="log", xlim=(1e-2, 1e1), ylim=(1e-2, 1e1))
    ax.set_aspect("equal")
    ax.set_xlabel(f"Experiment  ({xlabel})")
    ax.set_ylabel(f"Simulation  ({ylabel})")
    ax.set_title(title, fontweight="bold", pad=8, color="0.4")
    ax.grid(True, which="major", ls=":", lw=0.7, alpha=0.35)
    ax.tick_params(which="both", direction="in", top=True, right=True, colors="0.5")
    ax.text(0.5, 0.5, "TODO\nshear viscosity\n(Green–Kubo)", transform=ax.transAxes,
            ha="center", va="center", fontsize=15, color="0.55", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="0.7",
                      ls="--", alpha=0.9))


# systems enumerated in the legend, grouped by solvent; each entry is
# (salt, solvent, has_0.1M).  napf6_tgdme has no 0.1 M run.
LEGEND_SYSTEMS = [
    ("napf6", "dme", True),     ("naotf", "dme", True),
    ("napf6", "diglyme", True), ("naotf", "diglyme", True),
    ("napf6", "pc", True),      ("naotf", "pc", True),
    ("napf6", "tgdme", False),  ("naotf", "tgdme", True),
]


def legend_panel(ax):
    """Per-system legend: each salt/solvent in its own colour across two
    concentration columns (0.1 M hollow, 1.0 M filled), plus the 0.5 M LiPF6
    baseline and a compact guides block. Built by hand for full control."""
    ax.axis("off")
    tA = ax.transAxes
    col_a, col_b, tx = 0.06, 0.19, 0.30      # 0.1 M col / 1.0 M col / label x
    ms, lw = 9.0, 1.7

    def marker(x, y, mk, color, filled):
        ax.scatter(x, y, marker=mk, s=ms ** 2,
                   facecolors=color if filled else "white", edgecolors=color,
                   linewidths=lw, transform=tA, clip_on=False, zorder=5)

    def label(y, text):
        ax.text(tx, y, text, va="center", ha="left", fontsize=10, transform=tA)

    # column headers
    ax.text(col_a, 0.985, "0.1 M", ha="center", va="center", fontsize=10,
            fontweight="bold", transform=tA)
    ax.text(col_b, 0.985, "1.0 M", ha="center", va="center", fontsize=10,
            fontweight="bold", transform=tA)

    y, dy, gap = 0.92, 0.058, 0.016
    prev_solv = None
    for salt, solv, has01 in LEGEND_SYSTEMS:
        if prev_solv is not None and solv != prev_solv:
            y -= gap
        prev_solv = solv
        c, mk = SOLV_COLOR[solv], SALT_MARKER[salt]
        if has01:
            marker(col_a, y, mk, c, filled=False)
        marker(col_b, y, mk, c, filled=True)
        label(y, f"{SALT_LABEL[salt]} in {SOLV_LABEL[solv]}")
        y -= dy

    # 0.5 M LiPF6 baseline -- its own separated block
    y -= dy * 0.5 + gap
    ax.text(col_b, y, "0.5 M", ha="center", va="center", fontsize=10,
            fontweight="bold", transform=tA)
    y -= dy * 0.9
    marker(col_b, y, SALT_MARKER["lipf6"], SOLV_COLOR["dme"], filled=True)
    label(y, f"{SALT_LABEL['lipf6']} in {SOLV_LABEL['dme']}")
    y -= dy

    # compact guides
    y -= gap * 1.6
    ax.plot([col_a - 0.03, col_a + 0.07], [y, y], ls="--", color="0.45", lw=1.4,
            transform=tA, clip_on=False)
    label(y, "1:1  ($y=x$)")
    y -= dy
    ax.add_patch(Rectangle((col_a - 0.03, y - 0.014), 0.10, 0.028, facecolor="0.8",
                           alpha=0.55, lw=0, transform=tA, clip_on=False))
    label(y, r"within $\times 2$")
    y -= dy
    ax.plot([col_a + 0.02, col_a + 0.02], [y - 0.016, y + 0.016], color="0.4",
            lw=1.4, transform=tA, clip_on=False)
    ax.plot([col_a - 0.005, col_a + 0.045], [y + 0.016, y + 0.016], color="0.4",
            lw=1.2, transform=tA, clip_on=False)
    ax.plot([col_a - 0.005, col_a + 0.045], [y - 0.016, y - 0.016], color="0.4",
            lw=1.2, transform=tA, clip_on=False)
    label(y, r"$\pm$1 s.d. (4 replicas)")


def load_conductivity(path):
    """Aggregate the per-replica conductivity CSV to per-(system, conc) Onsager
    mean +/- s.d. at 298 K, with the experimental value, in parity-panel form."""
    c = pd.read_csv(path)
    c["system_id"] = c["system"].str.split("__").str[0]
    run = c["system"].str.split("__").str[1]
    conc = run.str.extract(r"npt_([0-9_]+M)_")[0]
    c["concentration_M"] = conc.str.replace("M", "", regex=False).str.replace("_", ".", regex=False).astype(float)
    c = c[(c["T_K"] == 298) & c["exp_conductivity_mS_cm"].notna()]
    agg = (c.groupby(["system_id", "concentration_M"])
             .agg(sig_mean=("sigma_onsager_mS_cm", "mean"),
                  sig_std=("sigma_onsager_mS_cm", "std"),
                  sig_exp=("exp_conductivity_mS_cm", "first"))
             .reset_index())
    return agg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--cond", type=Path, default=COND_CSV)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "figure2_results.png")
    args = ap.parse_args(argv)

    diff = pd.read_csv(args.root / "diffusivity_by_system.csv")
    cond = load_conductivity(args.cond)

    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 15, "axes.labelsize": 12.5,
        "axes.linewidth": 0.9, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 11.2))

    # Row 1: diffusivity (cation / anion / solvent)
    for ax, (mc, sc, ec, title) in zip(axes[0], SPECIES):
        draw_panel(ax, diff, mc, sc, ec, title, D_UNIT,
                   ylabel=D_UNIT if ax is axes[0, 0] else None, log=True)

    # Row 2: conductivity, viscosity-TODO, legend
    draw_panel(axes[1, 0], cond, "sig_mean", "sig_std", "sig_exp",
               r"Ionic conductivity  $\sigma$  (Onsager)", S_UNIT, ylabel=S_UNIT, log=True)
    todo_panel(axes[1, 1], r"Shear viscosity  $\eta$", r"mPa$\cdot$s", r"mPa$\cdot$s")
    legend_panel(axes[1, 2])

    for ax, lab in zip([axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]],
                       "abcde"):
        ax.annotate(lab, xy=(0, 1), xycoords="axes fraction", xytext=(-36, 10),
                    textcoords="offset points", fontsize=16, fontweight="bold",
                    va="bottom", ha="left")

    fig.suptitle("Transport properties: distilled PaiNN vs experiment, 298 K",
                 fontsize=17, fontweight="bold", y=0.975)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.91, bottom=0.07,
                        wspace=0.26, hspace=0.30)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor="white")
    fig.savefig(args.out.with_suffix(".pdf"), facecolor="white")
    print(f"saved -> {args.out}  and  {args.out.with_suffix('.pdf')}")
    plt.close(fig)


if __name__ == "__main__":
    main()
