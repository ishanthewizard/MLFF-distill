#!/usr/bin/env python3
"""Re-plot the UMA expanding-window conductivity sweep with an experimental
reference line per system (reads the CSVs written by
run_uma_expanding_window_conductivity.py; does NOT recompute conductivity).

Experimental ionic conductivity at 1 M / 298 K from
m5024/.../experiment_data/cleaned_version/conductivity.csv  (IC2, uS/cm).
Diglyme <-> DEGDME.  napf6_dme uses the VT temperature-series 298.05 K value
because the primary Na/PF6/DME 1 M row is blank.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COND_OUT = Path("/global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj/"
                "simulation_results/UMA/uma/analysis/conductivity_expanding_window_first4ns")

# experimental sigma (mS/cm) and the exp row used (for labels)
EXP = {
    "naotf_dme":     (1.233,  "Na/OTf/DME 1.0M 298.65K"),
    "naotf_diglyme": (2.778,  "Na/OTf/DEGDME 1.0M 298.55K"),
    "napf6_pc":      (6.505,  "Na/PF6/PC 1.0M 297.75K"),
    "napf6_diglyme": (6.686,  "Na/PF6/DEGDME 1.0M 297.95K"),
    "napf6_dme":     (12.96,  "Na/PF6/DME 1.0M 298.05K (VT)"),
}
COLORS = {
    "napf6_diglyme": "#1f77b4", "napf6_pc": "#ff7f0e", "naotf_diglyme": "#2ca02c",
    "naotf_dme": "#d62728", "napf6_dme": "#9467bd",
}

comb = pd.read_csv(COND_OUT / "conductivity_expanding_ALL.csv")

# ── per-system plots with exp reference ────────────────────────────────────
for lbl, g in comb.groupby("system"):
    g = g.sort_values("window_ns")
    col = COLORS.get(lbl, "#333333")
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot(g["window_ns"], g["sigma_onsager_mS_cm"], "o-", color=col, lw=2,
            label="Onsager (byteff2)")
    ax.plot(g["window_ns"], g["sigma_NE_mS_cm"], "s--", color=col, lw=2, alpha=0.55,
            label="Nernst-Einstein")
    if lbl in EXP:
        e, src = EXP[lbl]
        ax.axhline(e, color="k", ls=":", lw=1.8, label=f"exp {e:.3f} mS/cm\n({src})")
    ax.set_xlabel("Averaging window  0 → W  (ns)")
    ax.set_ylabel("Ionic conductivity (mS/cm)")
    ax.set_title(f"{lbl}  1 M 298 K — expanding-window conductivity\n"
                 f"(byteff2, fit 50–200 ps @ 1 ps/frame, first 4 ns)")
    ax.grid(True, ls=":", alpha=0.6)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = COND_OUT / f"conductivity_expanding_{lbl}_withexp.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)

# ── combined 2-panel (Onsager | NE), exp reference dots on the right edge ──
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharex=True)
for lbl, g in comb.groupby("system"):
    g = g.sort_values("window_ns")
    col = COLORS.get(lbl, "#333333")
    axes[0].plot(g["window_ns"], g["sigma_onsager_mS_cm"], "o-", color=col, lw=2, label=lbl)
    axes[1].plot(g["window_ns"], g["sigma_NE_mS_cm"], "s-", color=col, lw=2, label=lbl)
    if lbl in EXP:
        e, _ = EXP[lbl]
        for ax in axes:
            ax.axhline(e, color=col, ls=":", lw=1.2, alpha=0.7)
axes[0].set_title("Onsager (byteff2)  —  dotted = experiment")
axes[1].set_title("Nernst-Einstein  —  dotted = experiment")
for ax in axes:
    ax.set_xlabel("Averaging window  0 → W  (ns)")
    ax.set_ylabel("Ionic conductivity (mS/cm)")
    ax.grid(True, ls=":", alpha=0.6)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
fig.suptitle("UMA 1 M 298 K electrolytes — expanding-window conductivity vs experiment",
             fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
p = COND_OUT / "conductivity_expanding_ALL_withexp.png"
fig.savefig(p, dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote", p)

# ── final-window (0-4 ns) summary vs experiment ────────────────────────────
last = (comb.sort_values("window_ns").groupby("system").tail(1)
        .set_index("system"))
rows = []
for lbl in COLORS:
    if lbl not in last.index:
        continue
    r = last.loc[lbl]
    e = EXP.get(lbl, (np.nan, ""))[0]
    rows.append({
        "system": lbl, "window_ns": r["window_ns"],
        "sigma_onsager_mS_cm": round(float(r["sigma_onsager_mS_cm"]), 4),
        "sigma_NE_mS_cm": round(float(r["sigma_NE_mS_cm"]), 4),
        "exp_sigma_mS_cm": e,
        "onsager/exp": round(float(r["sigma_onsager_mS_cm"]) / e, 3) if e == e else np.nan,
    })
summ = pd.DataFrame(rows)
summ.to_csv(COND_OUT / "conductivity_final_window_vs_exp.csv", index=False)
print("wrote", COND_OUT / "conductivity_final_window_vs_exp.csv")
print(summ.to_string(index=False))
