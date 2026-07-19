"""
Combined (cation + anion + solvent) diffusivity parity plot for the
*uncorrected* simulated diffusivities, in linear and log versions.

The existing directory has one parity plot per species
(diffusivity_parity_{cation,anion,solvent}_uncorrected_npt_nvt_{linear,log}.png).
This script overlays the three species on a single set of axes:

  * colour  = species (cation / anion / solvent)
  * fill    = ensemble (NPT filled, NVT open)   [same convention as the
              per-species *_npt_nvt_* plots]
  * x = experimental D,  y = MD D (uncorrected, no Yeh-Hummer correction)

Uncorrected sim columns : D_cat_1e-10_m2s, D_ani_1e-10_m2s, D_sol_1e-10_m2s
Experimental columns    : exp_D_cation_1e-10_m2s, exp_D_anion_1e-10_m2s,
                          exp_D_solvent_1e-10_m2s

Selection rules follow the per-species plots: cation/anion drop systems with
concentration_M < 0.5; solvent keeps all. Rows with missing/non-positive
experimental D for a species are dropped for that species only.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

CSV = ("/global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj/"
       "simulation_results/PAINN/electrolytes_data/analysis/"
       "20260619_151043_msd_conductivity_pressure_energy_cell_size/"
       "group_results/diffusivity/diffusivity_with_exp_all.csv")
OUTDIR = ("/global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj/"
          "simulation_results/PAINN/electrolytes_data/analysis/"
          "20260619_151043_msd_conductivity_pressure_energy_cell_size/"
          "group_results/diffusivity")

# species -> (sim col, exp col, colour, min concentration)
SPECIES = {
    "cation":  ("D_cat_1e-10_m2s", "exp_D_cation_1e-10_m2s",  "tab:blue",  0.5),
    "anion":   ("D_ani_1e-10_m2s", "exp_D_anion_1e-10_m2s",   "tab:red",   0.5),
    "solvent": ("D_sol_1e-10_m2s", "exp_D_solvent_1e-10_m2s", "tab:green", 0.0),
}


def species_points(df, sim_col, exp_col, cmin):
    d = df[df["concentration_M"] >= cmin].copy()
    d = d[d[exp_col].notna() & (d[exp_col] > 0) & d[sim_col].notna()]
    return d[exp_col].to_numpy(float), d[sim_col].to_numpy(float), \
        d["model"].astype(str).to_numpy()          # 'model' col holds npt/nvt


def log_stats(exp, sim):
    m = np.isfinite(exp) & np.isfinite(sim) & (exp > 0) & (sim > 0)
    if m.sum() < 2:
        return np.nan, np.nan, int(m.sum())
    mae = float(np.mean(np.abs(np.log10(sim[m]) - np.log10(exp[m]))))
    rho, _ = spearmanr(exp[m], sim[m])
    return mae, float(rho), int(m.sum())


def make_plot(df, log_scale):
    scale = "log" if log_scale else "linear"
    fig, ax = plt.subplots(figsize=(8.2, 7))

    pts = {sp: species_points(df, *cfg[:2], cfg[3]) for sp, cfg in SPECIES.items()}
    allv = np.concatenate([np.concatenate([e, s]) for e, s, _ in pts.values()])

    if log_scale:
        pos = allv[allv > 0]
        lo, hi = pos.min() / 3.0, pos.max() * 3.0
        ax.set_xscale("log"); ax.set_yscale("log")
    else:
        lo, hi = 0.0, allv[np.isfinite(allv)].max() * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="1:1", zorder=1)

    # scatter: colour=species, fill=ensemble (npt filled, nvt open)
    for sp, cfg in SPECIES.items():
        color = cfg[2]
        exp, sim, ens = pts[sp]
        npt = ens == "npt"
        ax.scatter(exp[npt], sim[npt], s=70, marker="o", facecolor=color,
                   edgecolor="k", linewidths=0.6, alpha=0.85, zorder=3)
        ax.scatter(exp[~npt], sim[~npt], s=70, marker="o", facecolor="none",
                   edgecolor=color, linewidths=1.4, alpha=0.95, zorder=3)

    # red "fit all" line (in the plotted space) + slope in label
    e_all = np.concatenate([e for e, s, _ in pts.values()])
    s_all = np.concatenate([s for e, s, _ in pts.values()])
    if log_scale:
        m = (e_all > 0) & (s_all > 0)
        p = np.polyfit(np.log10(e_all[m]), np.log10(s_all[m]), 1)
        xx = np.logspace(np.log10(lo), np.log10(hi), 100)
        ax.plot(xx, 10 ** np.polyval(p, np.log10(xx)), "r-", lw=1.8,
                label=f"fit all (slope={p[0]:.2f})", zorder=2)
    else:
        m = np.isfinite(e_all) & np.isfinite(s_all)
        p = np.polyfit(e_all[m], s_all[m], 1)
        xx = np.linspace(lo, hi, 100)
        ax.plot(xx, np.polyval(p, xx), "r-", lw=1.8,
                label=f"fit all (slope={p[0]:.2f})", zorder=2)

    # per-species stats box
    lines = []
    for sp in SPECIES:
        e, s, _ = pts[sp]
        mae, rho, n = log_stats(e, s)
        lines.append(f"{sp:7s} n={n:2d}  logMAE={mae:.2f}  $\\rho$={rho:.2f}")
    mae, rho, n = log_stats(e_all, s_all)
    lines.append(f"{'all':7s} n={n:2d}  logMAE={mae:.2f}  $\\rho$={rho:.2f}")
    ax.text(0.97, 0.03, "\n".join(lines), transform=ax.transAxes, va="bottom",
            ha="right", fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray", alpha=0.9))

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel(r"Experimental D ($\times10^{-10}$ m$^2$/s)", fontsize=11)
    ax.set_ylabel(r"MD D, uncorrected ($\times10^{-10}$ m$^2$/s)", fontsize=11)
    ax.set_title(f"PAINN: diffusivity parity (uncorrected) — "
                 f"cation/anion/solvent, NPT vs NVT ({scale})", fontsize=12)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    # single in-axes legend (avoids clipping): 1:1, fit, species colours,
    # and the NPT/NVT fill convention
    handles = [
        Line2D([], [], ls="--", color="k", lw=1.0, label="1:1"),
        Line2D([], [], ls="-", color="r", lw=1.8,
               label=f"fit all (slope={p[0]:.2f})"),
        Line2D([], [], marker="o", ls="", mfc=SPECIES["cation"][2], mec="k", ms=9, label="Cation"),
        Line2D([], [], marker="o", ls="", mfc=SPECIES["anion"][2], mec="k", ms=9, label="Anion"),
        Line2D([], [], marker="o", ls="", mfc=SPECIES["solvent"][2], mec="k", ms=9, label="Solvent"),
        Line2D([], [], marker="o", ls="", mfc="gray", mec="k", ms=9, label="NPT (filled)"),
        Line2D([], [], marker="o", ls="", mfc="none", mec="gray", mew=1.4, ms=9, label="NVT (open)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    out = f"{OUTDIR}/diffusivity_parity_allspecies_uncorrected_npt_nvt_{scale}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
    for ln in lines:
        print("  " + ln.replace("$\\rho$", "rho"))
    return out


def main():
    df = pd.read_csv(CSV)
    for log_scale in (False, True):
        make_plot(df, log_scale)


if __name__ == "__main__":
    main()
