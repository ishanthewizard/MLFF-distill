"""PaiNN conductivity code-check figure: byteff2 Onsager vs mdcraft collective,
both vs experiment, in a 2x1 parity stack (a = byteff2 Onsager, b = mdcraft
collective) + shared legend.  PaiNN only -- this is an internal cross-check of the
two conductivity estimators, so no OPLS overlay.  Reuses the styling helpers from
``plot_group_transport_parity.py`` so it matches the transport-parity figures.

    python plot_byteff2_vs_mdcraft.py --merged <.../merged> [--drop-01m] [--out <png>]
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_group_transport_parity import (
    load_painn_cond, draw_panel, legend_panel, S_UNIT,
)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--temperature", type=float, default=298.0)
    ap.add_argument("--drop-01m", action="store_true",
                    help="exclude 0.1 M systems (their small exp values dominate MAPD)")
    ap.add_argument("--linear", action="store_true",
                    help="use linear axes instead of the default log-log")
    args = ap.parse_args(argv)

    cond = load_painn_cond(args.merged)
    cond["exp_sigma"] = cond["pexp_sigma"]
    cond = cond[np.abs(cond["temperature_K"] - args.temperature) < 1.0].copy()
    if args.drop_01m:
        cond = cond[np.abs(cond["concentration_M"] - 0.1) > 0.01].copy()

    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 15, "axes.labelsize": 12.5,
        "axes.linewidth": 0.9, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig = plt.figure(figsize=(10.6, 9.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.05], left=0.11, right=0.995,
                          top=0.90, bottom=0.075, wspace=0.05, hspace=0.16)

    rows = [("painn_onsager_mean", "painn_onsager_std", "byteff2",
             r"Onsager  $\sigma$  (byteff2)", "byteff2 Onsager"),
            ("painn_mdcraft_mean", "painn_mdcraft_std", "mdcraft",
             "", "mdcraft collective")]
    letters = iter("ab")
    for i, (mean, std, plab, title, ylab) in enumerate(rows):
        ax = fig.add_subplot(gs[i, 0])
        draw_panel(ax, cond, mean, std, "exp_sigma", title, S_UNIT,
                   p_label=plab, show_xlabel=(i == 1), log=not args.linear)
        ax.set_ylabel(f"{ylab}\n({S_UNIT})", fontweight="bold")
        ax.annotate(next(letters), xy=(0, 1), xycoords="axes fraction",
                    xytext=(-46, 8), textcoords="offset points", fontsize=15,
                    fontweight="bold", va="bottom", ha="left")
    legend_panel(fig.add_subplot(gs[:, 1]), show_opls=False,
                 col_a=0.06, col_b=0.22, tx=0.34)

    cnote = "  (0.1 M excluded)" if args.drop_01m else ""
    fig.suptitle("PaiNN conductivity code check: byteff2 Onsager vs mdcraft vs exp, "
                 f"{args.temperature:.0f} K{cnote}",
                 fontsize=12, fontweight="bold", y=0.965)

    out = args.out or (args.merged / "figure_conductivity_paiNN_byteff2_vs_mdcraft.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    print(f"saved -> {out}  and  {out.with_suffix('.pdf')}")
    plt.close(fig)


if __name__ == "__main__":
    main()
