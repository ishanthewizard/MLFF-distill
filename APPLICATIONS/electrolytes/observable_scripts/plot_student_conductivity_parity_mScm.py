#!/usr/bin/env python
"""Parity plot (mS/cm) of student-model ionic conductivity vs experiment, for a
per-replica conductivity CSV (columns already in mS/cm).

Aggregates replicas (the `model` column) per `system`; plots the three
estimators present in these CSVs — Onsager (collective), Nernst-Einstein, and
mdcraft collective — as mean ±1 s.d. over replicas vs the concentration/T-exact
experimental conductivity.  Log-log, equal x/y limits + equal aspect so the 1:1
line is a true diagonal.  Rows without an experimental value are dropped.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MDCRAFT_COL = "conductivity by md craft (mS/cm)"
ESTIMATORS = [
    ("sigma_onsager_mS_cm", "Onsager (collective)", "#1f77b4", "o"),
    ("sigma_NE_mS_cm",      "Nernst-Einstein",      "#ff7f0e", "^"),
    (MDCRAFT_COL,           "mdcraft (collective)", "#2ca02c", "s"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default="student conductivity parity (mS/cm)")
    ap.add_argument("--exp-col", default="exp_conductivity_mS_cm")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df[args.exp_col] = pd.to_numeric(df[args.exp_col], errors="coerce")
    df = df[df[args.exp_col].notna() & (df[args.exp_col] > 0)].copy()
    if df.empty:
        raise SystemExit(f"No rows with a valid {args.exp_col} in {args.csv}")

    # aggregate replicas per system identity
    keys = ["system", "cat_symbol", "anion_symbol", "solvent_symbol", "T_K"]
    keys = [k for k in keys if k in df.columns]
    agg = {}
    for col, *_ in ESTIMATORS:
        if col in df.columns:
            agg[col] = ["mean", "std"]
    agg[args.exp_col] = "mean"
    g = df.groupby(keys, dropna=False).agg(agg)
    g.columns = ["_".join(c).strip("_") for c in g.columns]
    g = g.reset_index()
    exp_mean_col = f"{args.exp_col}_mean"

    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    vals = [g[exp_mean_col].values]
    for col, label, color, marker in ESTIMATORS:
        mcol, scol = f"{col}_mean", f"{col}_std"
        if mcol not in g:
            continue
        y = g[mcol].values
        yerr = np.nan_to_num(g[scol].values) if scol in g else None
        ax.errorbar(g[exp_mean_col].values, y, yerr=yerr, fmt=marker, ms=8,
                    color=color, ecolor=color, elinewidth=1, capsize=3,
                    mec="k", mew=0.6, label=label, zorder=3, ls="none")
        vals.append(y)

    allv = np.concatenate([v[np.isfinite(v)] for v in vals])
    allv = allv[allv > 0]
    lo, hi = allv.min() * 0.5, allv.max() * 2.0
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")

    # annotate each point once (near its Onsager marker) with salt/solvent[/T]
    for _, r in g.iterrows():
        lab = f"{r.get('cat_symbol','')}{r.get('anion_symbol','')}/{r.get('solvent_symbol','')}"
        if "T_K" in g:
            lab += f" {int(r['T_K'])}K"
        ycol = "sigma_onsager_mS_cm_mean" if "sigma_onsager_mS_cm_mean" in g else exp_mean_col
        ax.annotate(lab, (r[exp_mean_col], r[ycol]), fontsize=6.5,
                    xytext=(4, 3), textcoords="offset points")

    ax.set_xlabel("experimental ionic conductivity (mS/cm)")
    ax.set_ylabel("student simulated ionic conductivity (mS/cm)")
    ax.set_title(args.title)
    ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=9)
    fig.tight_layout()

    out = Path(args.out) if args.out else Path(args.csv).with_name(
        Path(args.csv).stem + "_parity_mScm.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}  ({len(g)} systems, {len(df)} replica rows)")
    show = [exp_mean_col] + [f"{c}_mean" for c, *_ in ESTIMATORS if f"{c}_mean" in g]
    print("\n(mS/cm, replica means)\n" + g[keys + show].to_string(index=False))


if __name__ == "__main__":
    main()
