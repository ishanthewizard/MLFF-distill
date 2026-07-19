#!/usr/bin/env python
"""Parity plot (mS/cm) for the OPLS 1 M mdcraft collective conductivity, with the
byteff2 Onsager + NE values overlaid (from the byteff2 conductivity_with_exp.csv)
and the experimental reference on x.  Log-log, equal x/y limits + equal aspect.
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
    ap.add_argument("--mdcraft-csv", required=True)
    ap.add_argument("--byteff2-csv", default="/global/homes/y/yuejian/project/MLFF-distill/m5024/"
                    "distillation_project/results/opls_baseline/analysis/"
                    "conductivity_20260708_232123/conductivity_with_exp.csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    md = pd.read_csv(args.mdcraft_csv)
    md = md[md.get("error", "").fillna("") == ""] if "error" in md else md
    bt = pd.read_csv(args.byteff2_csv)
    bt["exp_mS_cm"] = bt["exp_conductivity_uS_cm"] / 1000.0
    m = md.merge(bt[["name", "salt", "solvent", "sigma_onsager_mS_cm", "sigma_NE_mS_cm",
                     "exp_mS_cm"]], on="name", how="left", suffixes=("", "_bt"))
    m = m[m["exp_mS_cm"].notna()].copy()

    x = m["exp_mS_cm"].values
    series = [
        ("sigma_mdcraft_mS_cm",  "mdcraft (collective)",   "#2ca02c", "s"),
        ("sigma_onsager_mS_cm",  "byteff2 Onsager",         "#1f77b4", "o"),
        ("sigma_NE_mS_cm",       "byteff2 Nernst-Einstein", "#ff7f0e", "^"),
    ]

    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    allv = [x]
    for col, lab, c, mk in series:
        if col in m:
            ax.scatter(x, m[col], s=85, marker=mk, c=c, edgecolor="k", label=lab, zorder=3)
            allv.append(m[col].values)
    for _, r in m.iterrows():
        ax.annotate(f"{r['salt']}/{r['solvent']}", (r["exp_mS_cm"], r["sigma_mdcraft_mS_cm"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")

    allv = np.concatenate([np.asarray(v, float) for v in allv]); allv = allv[allv > 0]
    lo, hi = allv.min() * 0.5, allv.max() * 2.0
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("experimental ionic conductivity (mS/cm)")
    ax.set_ylabel("simulated ionic conductivity (mS/cm)")
    ax.set_title("OPLS 1 M NPT — conductivity parity (298 K)\nmdcraft vs byteff2, full length")
    ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=9)
    fig.tight_layout()

    out = Path(args.out) if args.out else Path(args.mdcraft_csv).with_name(
        "conductivity_mdcraft_parity_mScm.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}")
    print("\n(mS/cm)\n" + m[["salt", "solvent", "sigma_mdcraft_mS_cm",
          "sigma_onsager_mS_cm", "sigma_NE_mS_cm", "exp_mS_cm"]].to_string(index=False))


if __name__ == "__main__":
    main()
