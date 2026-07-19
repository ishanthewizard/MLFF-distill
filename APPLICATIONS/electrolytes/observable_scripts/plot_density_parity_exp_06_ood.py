#!/usr/bin/env python3
"""Density parity plot for the PAINN exp_06_ood systems: each model's density
vs experimental density, overlaid for comparison (PAINN, uma, orb, OPLS).

Usage:
  python plot_density_parity_exp_06_ood.py <parity_csv> [out_png]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj"
    "/simulation_results/PAINN/electrolytes_data/analysis/ood_results"
    "/density_parity_exp_06_ood.csv")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else CSV.parent / "density_parity_exp_06_ood.png"

# model column -> (label, color, marker)
MODELS = {
    "PAINN_density (g/cm3)": ("PAINN (this work)", "#1f77b4", "o"),
    "uma_density (g/cm3)":   ("uma",               "#ff7f0e", "s"),
    "orb_density (g/cm3)":   ("orb",               "#2ca02c", "^"),
    "OPLS_density (g/cm3)":  ("OPLS",              "#9467bd", "D"),
}

df = pd.read_csv(CSV)
exp = df["Exp_density (g/cm3)"].to_numpy(dtype=float)
std = df.get("PAINN_run_std (g/cm3)", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)

fig, ax = plt.subplots(figsize=(6.2, 6.2))

# 1:1 line range from all finite values
allv = [exp]
for col in MODELS:
    allv.append(df[col].to_numpy(dtype=float))
flat = np.concatenate([a[np.isfinite(a)] for a in allv])
lo, hi = flat.min(), flat.max()
pad = 0.05 * (hi - lo)
lims = (lo - pad, hi + pad)
ax.plot(lims, lims, "k--", lw=1, zorder=0, label="y = x")

summary = []
for col, (label, color, marker) in MODELS.items():
    y = df[col].to_numpy(dtype=float)
    m = np.isfinite(exp) & np.isfinite(y)
    if not m.any():
        continue
    if col.startswith("PAINN"):
        ax.errorbar(exp[m], y[m], yerr=std[m], fmt="none",
                    ecolor=color, elinewidth=1, capsize=3, zorder=2)
    ax.scatter(exp[m], y[m], c=color, marker=marker, s=70,
               edgecolors="k", linewidths=0.5, zorder=3, label="_nolegend_")
    mae = np.mean(np.abs(y[m] - exp[m]))
    mape = 100 * np.mean(np.abs(y[m] - exp[m]) / exp[m])
    summary.append((label, color, marker, mae, mape, int(m.sum())))

# legend with metrics
handles = []
for label, color, marker, mae, mape, n in summary:
    handles.append(plt.Line2D([], [], color=color, marker=marker, ls="",
                              markeredgecolor="k", markersize=9,
                              label=f"{label}: MAE={mae:.3f}, MAPE={mape:.1f}% (n={n})"))
handles.append(plt.Line2D([], [], color="k", ls="--", lw=1, label="y = x"))
ax.legend(handles=handles, fontsize=8.5, loc="upper left", framealpha=0.9)

# annotate points with system labels (one per system, near PAINN markers)
pcol = "PAINN_density (g/cm3)"
for _, r in df.iterrows():
    tag = f"{r['Salt']}/{r['Solvent']}"
    ax.annotate(tag, (r["Exp_density (g/cm3)"], r[pcol]),
                textcoords="offset points", xytext=(6, 4), fontsize=7, color="#333")

ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_aspect("equal")
ax.set_xlabel("Experimental density (g/cm³)")
ax.set_ylabel("Simulated density (g/cm³)")
ax.set_title("Density parity — PAINN exp_06_ood (1 M, 298 K)\nvs experiment, compared with uma / orb / OPLS")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT, dpi=200)
print(f"wrote: {OUT}")
print("\nper-model on these systems:")
for label, _, _, mae, mape, n in summary:
    print(f"  {label:18s} MAE={mae:.4f} g/cm³  MAPE={mape:5.2f}%  n={n}")
