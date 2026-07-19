#!/usr/bin/env python3
"""Aggregate + plot expanding-window ("running") conductivity for one source
(OPLS or student/PAINN).

Reads the per-(replica,system) CSVs written by
run_expanding_conductivity_*.py under <dir>/intermediate/, then:
  * rebuilds the authoritative combined  conductivity_expanding_ALL.csv
  * writes  conductivity_expanding_by_window.csv  = per (system_id, window_ns)
    mean +/- s.d. over replicas of sigma_Onsager / sigma_NE / D's
  * draws convergence grids (one subplot per system_id, replicas overlaid + the
    replica-mean bold) for Onsager and for Nernst-Einstein.

Env: fairchemV2_new
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REP_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def load_all(d):
    files = sorted((d / "intermediate").glob("expanding_*.csv"))
    if not files:
        raise SystemExit(f"no intermediate CSVs under {d/'intermediate'}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df


def by_window(df):
    g = (df.groupby(["source", "system_id", "salt", "cation", "anion", "solvent",
                     "concentration_M", "temperature_K", "window_ns"])
           .agg(n_replicas=("replica", "nunique"),
                sigma_onsager_mScm_mean=("sigma_onsager_mS_cm", "mean"),
                sigma_onsager_mScm_std=("sigma_onsager_mS_cm", "std"),
                sigma_NE_mScm_mean=("sigma_NE_mS_cm", "mean"),
                sigma_NE_mScm_std=("sigma_NE_mS_cm", "std"),
                D_cat_mean=("D_cat_1e10_m2s", "mean"),
                D_anion_mean=("D_anion_1e10_m2s", "mean"),
                D_solvent_mean=("D_solvent_1e10_m2s", "mean"))
           .reset_index())
    return g


def grid_plot(df, bw, col_mean, col_std, raw_col, title, out_png):
    sysids = sorted(df["system_id"].unique())
    n = len(sysids)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.1 * nrow),
                             squeeze=False, sharex=True)
    for idx, sysid in enumerate(sysids):
        ax = axes[idx // ncol][idx % ncol]
        sub = df[df["system_id"] == sysid]
        reps = sorted(sub["replica"].unique())
        for ri, rep in enumerate(reps):
            s = sub[sub["replica"] == rep].sort_values("window_ns")
            ax.plot(s["window_ns"], s[raw_col], "-", lw=1.0, alpha=0.55,
                    color=REP_COLORS[ri % len(REP_COLORS)], marker="o", ms=3,
                    label=rep.replace("replicas_", "r").replace("npt_replica_", "r"))
        b = bw[bw["system_id"] == sysid].sort_values("window_ns")
        ax.plot(b["window_ns"], b[col_mean], "-", lw=2.4, color="k", zorder=5, label="mean")
        ax.fill_between(b["window_ns"], b[col_mean] - b[col_std].fillna(0),
                        b[col_mean] + b[col_std].fillna(0), color="k", alpha=0.12, zorder=1)
        ax.set_title(sysid, fontsize=9)
        ax.grid(True, ls=":", alpha=0.5)
        ax.set_xticks(range(0, 21, 5))
        if idx % ncol == 0:
            ax.set_ylabel("σ (mS/cm)")
        if idx // ncol == nrow - 1:
            ax.set_xlabel("window 0→W (ns)")
        ax.legend(fontsize=6, ncol=2, loc="best")
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(title, fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--label", required=True, help="e.g. 'OPLS NVT' or 'student PAINN NPT'")
    args = ap.parse_args()
    d = Path(args.dir)

    df = load_all(d)
    df.to_csv(d / "conductivity_expanding_ALL.csv", index=False)
    print(f"wrote {d/'conductivity_expanding_ALL.csv'} "
          f"({len(df)} rows, {df['system_id'].nunique()} systems, "
          f"{df['replica'].nunique()} replicas)")

    bw = by_window(df)
    bw.to_csv(d / "conductivity_expanding_by_window.csv", index=False)
    print(f"wrote {d/'conductivity_expanding_by_window.csv'} ({len(bw)} rows)")

    grid_plot(df, bw, "sigma_onsager_mScm_mean", "sigma_onsager_mScm_std",
              "sigma_onsager_mS_cm",
              f"{args.label} — expanding-window Onsager conductivity convergence\n"
              f"(thin = replicas, bold = mean±s.d.; fit 50-200 ps @ 1 ps/frame)",
              d / "convergence_grid_onsager.png")
    grid_plot(df, bw, "sigma_NE_mScm_mean", "sigma_NE_mScm_std",
              "sigma_NE_mS_cm",
              f"{args.label} — expanding-window Nernst-Einstein conductivity convergence\n"
              f"(thin = replicas, bold = mean±s.d.)",
              d / "convergence_grid_NE.png")

    # console summary at 0-20 ns (final window)
    wmax = df["window_ns"].max()
    fin = bw[bw["window_ns"] == wmax].sort_values("system_id")
    print(f"\n=== {args.label}: sigma at 0-{wmax:.0f} ns (mean over replicas) ===")
    print(fin[["system_id", "n_replicas", "sigma_onsager_mScm_mean",
               "sigma_onsager_mScm_std", "sigma_NE_mScm_mean", "sigma_NE_mScm_std"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
