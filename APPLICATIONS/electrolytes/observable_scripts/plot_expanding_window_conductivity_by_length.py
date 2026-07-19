#!/usr/bin/env python3
"""Re-plot the NaOTf/DME 1 M 298 K PAINN FP32 expanding-window conductivity CSV
with the x-axis expressed as the AVERAGING-WINDOW LENGTH instead of window-end.

The original figure (expanding_window_conductivity.png) put the fixed 5 ns start
on the x-axis: window_end_ns 6 -> 20, labelled "[start fixed at 5 ns]".  Since
every window starts at 5 ns, window_end 6 == a 1 ns averaging window, 7 == 2 ns,
..., 20 == 15 ns.  Here we plot against that span (the `window_span_ns` column),
so the point formerly at x=6 sits at x=1 ns, and we drop the "start fixed at 5 ns"
framing from the title and axis label.

Pure re-plot: reads the existing CSV, computes nothing.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 24,
})

DEFAULT_DIR = Path(
    "/global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj"
    "/simulation_results/PAINN/electrolytes_data/analysis/fp32_simulation"
    "/In_distribution/multi_replicas/npt/naotf_dme_1M_298K_cond_expanding_5ns_50to200ps"
)

REPLICA_COLORS = {
    0: "#1f77b4",   # replica_0
    1: "#ff7f0e",   # replica_1
    2: "#2ca02c",   # replica_2
    3: "#d62728",   # replica_3
}
EXP_MS_CM = 1.233   # NaOTf/DME 1 M experimental conductivity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR),
                    help="directory holding expanding_window_conductivity.csv")
    ap.add_argument("--csv", default=None, help="override CSV path")
    ap.add_argument("--out", default=None, help="override output PNG path")
    args = ap.parse_args()

    base = Path(args.dir)
    csv = Path(args.csv) if args.csv else base / "expanding_window_conductivity.csv"
    out = Path(args.out) if args.out else base / "expanding_window_conductivity_by_length.png"

    df = pd.read_csv(csv)
    # x-axis = averaging-window length (ns); the point at window_end 6 -> 1 ns.
    xcol = "window_span_ns"
    x_all = np.sort(df[xcol].unique())

    panels = [
        ("sigma_onsager_mS_cm", "Onsager  (50-200 ps fit)"),
        ("sigma_NE_mS_cm",      "Nernst-Einstein  (50-200 ps fit)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(17.0, 7.5))
    for ax, (ycol, title) in zip(axes, panels):
        # per-replica curves
        for rep, sub in df.groupby("replica"):
            sub = sub.sort_values(xcol)
            ax.plot(sub[xcol], sub[ycol], "o-", ms=8, lw=2.5,
                    color=REPLICA_COLORS.get(int(rep), None),
                    label=f"replica_{int(rep)}")
        # cross-replica mean
        mean = df.groupby(xcol)[ycol].mean().reindex(x_all)
        ax.plot(x_all, mean.values, "k--", lw=2.8, label="mean")
        # experimental reference
        ax.axhline(EXP_MS_CM, color="0.5", ls=":", lw=2.2, label=f"exp {EXP_MS_CM:g}")

        ax.set_title(title)
        ax.set_xlabel("averaging window length (ns)")
        ax.set_ylabel("ionic conductivity (mS/cm)")
        ax.set_xticks(x_all)
        ax.grid(True, ls=":", alpha=0.5)

    axes[0].legend(ncol=2, loc="upper right")
    fig.suptitle(
        "NaOTf / DME  1 M  298 K  NVT-langevin (PAINN FP32) "
        "— expanding-window conductivity",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
