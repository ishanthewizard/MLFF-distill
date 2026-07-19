#!/usr/bin/env python3
"""Parity plot: simulated Nernst-Einstein ionic conductivity at the 0-4 ns
window vs experimental conductivity, for the UMA 1 M / 298 K electrolytes.

Reads conductivity_final_window_vs_exp.csv (written by plot_uma_expanding_with_exp.py).
Log-log, identical x/y limits + equal aspect so the 1:1 line is a true diagonal.
Solvent -> colour, salt -> marker.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COND_OUT = Path("/global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj/"
                "simulation_results/UMA/uma/analysis/conductivity_expanding_window_first4ns")

# system -> (salt, solvent) for marker/colour encoding
META = {
    "napf6_diglyme": ("PF6", "Diglyme"),
    "napf6_pc":      ("PF6", "PC"),
    "naotf_diglyme": ("OTf", "Diglyme"),
    "naotf_dme":     ("OTf", "DME"),
    "napf6_dme":     ("PF6", "DME"),
}
SOLVENT_COLOR = {"DME": "#1f77b4", "Diglyme": "#2ca02c", "PC": "#ff7f0e"}
SALT_MARKER   = {"PF6": "o", "OTf": "s"}

df = pd.read_csv(COND_OUT / "conductivity_final_window_vs_exp.csv")
df = df.dropna(subset=["exp_sigma_mS_cm", "sigma_NE_mS_cm"])

x = df["exp_sigma_mS_cm"].to_numpy(float)     # experiment
y = df["sigma_NE_mS_cm"].to_numpy(float)      # simulated NE @ 4 ns

# equal square log limits
lo = 10 ** np.floor(np.log10(min(x.min(), y.min())) - 0.05)
hi = 10 ** np.ceil(np.log10(max(x.max(), y.max())) + 0.05)

fig, ax = plt.subplots(figsize=(6.4, 6.4))
ax.plot([lo, hi], [lo, hi], "k-", lw=1, zorder=0, label="1:1")
for _, r in df.iterrows():
    salt, solv = META[r["system"]]
    ax.scatter(r["exp_sigma_mS_cm"], r["sigma_NE_mS_cm"],
               marker=SALT_MARKER[salt], s=130, color=SOLVENT_COLOR[solv],
               edgecolor="k", linewidth=0.8, zorder=3)
    ax.annotate(r["system"], (r["exp_sigma_mS_cm"], r["sigma_NE_mS_cm"]),
                textcoords="offset points", xytext=(7, 4), fontsize=8)

# metrics (log space)
logmae = float(np.mean(np.abs(np.log10(y) - np.log10(x))))
rho = spearmanr(x, y).correlation

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_aspect("equal", "box")
ax.set_xlabel("Experimental ionic conductivity (mS/cm)")
ax.set_ylabel("Simulated Nernst-Einstein σ @ 0–4 ns (mS/cm)")
ax.set_title("UMA 1 M 298 K — Nernst-Einstein conductivity parity (0–4 ns)\n"
             f"log-MAE = {logmae:.2f} dex   Spearman ρ = {rho:.2f}")
ax.grid(True, which="both", ls=":", alpha=0.5)

# legends: solvent colour + salt marker
from matplotlib.lines import Line2D
solv_handles = [Line2D([0], [0], marker="o", ls="", mfc=c, mec="k", ms=9, label=s)
                for s, c in SOLVENT_COLOR.items()]
salt_handles = [Line2D([0], [0], marker=m, ls="", mfc="grey", mec="k", ms=9, label=s)
                for s, m in SALT_MARKER.items()]
leg1 = ax.legend(handles=solv_handles, title="solvent", loc="upper left", fontsize=8)
ax.add_artist(leg1)
ax.legend(handles=salt_handles + [Line2D([0], [0], color="k", lw=1, label="1:1")],
          title="salt", loc="lower right", fontsize=8)

fig.tight_layout()
p = COND_OUT / "parity_conductivity_NE_4ns.png"
fig.savefig(p, dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote", p)
print(df[["system", "exp_sigma_mS_cm", "sigma_NE_mS_cm"]].to_string(index=False))
print(f"log-MAE = {logmae:.3f} dex   Spearman rho = {rho:.3f}")
