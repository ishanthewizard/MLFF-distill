#!/usr/bin/env python
"""Re-plot the OPLS 1 M conductivity parity figure in mS/cm (from the existing
conductivity_with_exp.csv; sim already in mS/cm, exp column is uS/cm -> /1000).
Log-log, equal x/y limits + equal aspect so the 1:1 line is a true diagonal.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="conductivity_with_exp.csv")
    ap.add_argument("--out", default=None, help="output png (default: next to csv)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "error" in df:
        df = df[df["error"].fillna("") == ""]
    df = df[df["exp_conductivity_uS_cm"].notna()].copy()

    # everything in mS/cm
    df["exp_mS_cm"] = df["exp_conductivity_uS_cm"] / 1000.0
    x   = df["exp_mS_cm"].values
    yon = df["sigma_onsager_mS_cm"].values
    yne = df["sigma_NE_mS_cm"].values

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(x, yon, s=80, c="#1f77b4", edgecolor="k",
               label="Onsager (collective)", zorder=3)
    ax.scatter(x, yne, s=80, marker="^", c="#ff7f0e", edgecolor="k",
               label="Nernst-Einstein", zorder=3)
    for _, rr in df.iterrows():
        ax.annotate(f"{rr['salt']}/{rr['solvent']}",
                    (rr["exp_mS_cm"], rr["sigma_onsager_mS_cm"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")

    allv = np.concatenate([x, yon, yne]); allv = allv[allv > 0]
    lo, hi = allv.min() * 0.5, allv.max() * 2.0
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("experimental ionic conductivity (mS/cm)")
    ax.set_ylabel("simulated ionic conductivity (mS/cm)")
    ax.set_title("OPLS 1 M NPT — conductivity parity (298 K)")
    ax.grid(alpha=0.25, which="both"); ax.legend()
    fig.tight_layout()

    out = Path(args.out) if args.out else Path(args.csv).parent / "conductivity_parity_with_exp_mS_cm.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")
    cols = ["salt", "solvent", "sigma_onsager_mS_cm", "sigma_NE_mS_cm", "exp_mS_cm"]
    print("\n(all mS/cm)\n" + df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
