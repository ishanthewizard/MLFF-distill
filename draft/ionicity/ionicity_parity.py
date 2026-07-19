#!/usr/bin/env python3
"""Ionicity (inverse Haven ratio) parity: experiment vs simulation, PaiNN & OPLS.

    alpha = sigma / sigma_NE            (Haven ratio H_R = 1/alpha)

  - alpha_exp = sigma_exp / sigma_NE^exp,  sigma_NE^exp from EXPERIMENTAL D+/D-.
  - alpha_sim = sigma_Onsager / sigma_NE^sim,  sigma_NE^sim from the SIMULATED
    *Yeh-Hummer corrected* self-diffusivities (D_*_corrected).

Nernst-Einstein (1:1 salt, z=+1/-1), using molar salt concentration c:
    sigma_NE = e^2 N_A c / (k_B T) * (D+ + D-)          [S/m]   ->  x10 = mS/cm

Numerators:
  sigma_exp      = exp_conductivity_mS_cm       (measured)
  sigma_Onsager  = sigma_onsager_*_mS_cm        (byteff2 collective, per-system mean)

Inputs (per-system group CSVs):
  PaiNN : conductivity_by_system.csv + diffusivity_by_system.csv   (key: system_id,conc,T)
  OPLS  : conductivity_per_system.csv + diffusivity_per_system.csv (key: system)

Output: figure_ionicity_parity_painn_opls.png/.pdf  (1x2: PaiNN | OPLS), + printed table.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# physical constants (SI)
E    = 1.602176634e-19      # C
KB   = 1.380649e-23         # J/K
NA   = 6.02214076e23        # 1/mol

PA = Path("/global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj/"
          "simulation_results/PAINN/electrolytes_data/analysis/fp32_simulation/"
          "In_distribution/multi_replicas/nvt/merged")
OP = Path("/global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj/"
          "simulation_results/OPLS/analysis/nvt/group_results")
OUT = Path("/global/homes/y/yuejian/project/MLFF-distill/draft/ionicity")

# plot encoding (matches the transport-parity figures)
SOLV_COLOR = {"dme": "#E68310", "diglyme": "#11A579", "pc": "#E73F74", "tgdme": "#3969AC"}
SOLV_LABEL = {"dme": "DME", "diglyme": "DEGDME", "pc": "PC", "tgdme": "TEGDME"}
SALT_MARKER = {"napf6": "o", "naotf": "s", "lipf6": "^"}
SALT_LABEL = {"napf6": "NaPF$_6$", "naotf": "NaOTf", "lipf6": "LiPF$_6$"}


def sigma_NE_mScm(Dp, Dm, c_M, T):
    """NE conductivity [mS/cm] from D+/D- (in 1e-10 m^2/s), molarity c_M, T [K]."""
    c = c_M * 1000.0                       # mol/L -> mol/m^3
    s_si = E * E * NA * c / (KB * T) * ((Dp + Dm) * 1e-10)   # S/m
    return s_si * 10.0                     # -> mS/cm


def _norm_solv(s):
    return "tgdme" if s in ("tegdme", "tgdme") else s


def salt_solv(system_id):
    p = system_id.split("_")
    return p[0], _norm_solv(p[1])


def build_painn():
    c = pd.read_csv(PA / "conductivity_by_system.csv")
    d = pd.read_csv(PA / "diffusivity_by_system.csv")
    m = c.merge(d, on=["system_id", "concentration_M", "temperature_K"], how="inner")
    out = pd.DataFrame({
        "system_id":  m["system_id"],
        "conc":       m["concentration_M"],
        "T":          m["temperature_K"],
        "sigma_exp":  m["exp_conductivity_mS_cm"],
        "sigma_sim":  m["sigma_onsager_mS_cm_mean"],
        "Dp_exp":     m["exp_D_cation_ref"],
        "Dm_exp":     m["exp_D_anion_ref"],
        "Dp_sim":     m["D_cat_corrected_1e-10_m2s_mean"],
        "Dm_sim":     m["D_ani_corrected_1e-10_m2s_mean"],
    })
    return out


def build_opls():
    c = pd.read_csv(OP / "conductivity_per_system.csv")
    d = pd.read_csv(OP / "diffusivity_per_system.csv")
    m = c.merge(d, on="system", how="inner", suffixes=("_c", "_d"))
    out = pd.DataFrame({
        "system_id":  m["system"].map(lambda s: "_".join(s.split("_")[:2])),
        "conc":       m["concentration_M_c"],
        "T":          m["T_K"],
        "sigma_exp":  m["exp_conductivity_mS_cm"],
        "sigma_sim":  m["sigma_onsager_mean_mS_cm"],
        "Dp_exp":     m["exp_D_cation_1e-10_m2s"],
        "Dm_exp":     m["exp_D_anion_1e-10_m2s"],
        "Dp_sim":     m["D_cat_corrected_sim_mean"],
        "Dm_sim":     m["D_ani_corrected_sim_mean"],
    })
    return out


def add_ionicity(df):
    df = df.copy()
    df["sNE_exp"]   = sigma_NE_mScm(df["Dp_exp"], df["Dm_exp"], df["conc"], df["T"])
    df["sNE_sim"]   = sigma_NE_mScm(df["Dp_sim"], df["Dm_sim"], df["conc"], df["T"])
    df["alpha_exp"] = df["sigma_exp"] / df["sNE_exp"]
    df["alpha_sim"] = df["sigma_sim"] / df["sNE_sim"]
    return df


def loglog_r(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if ok.sum() < 3:
        return None
    lx, ly = np.log10(x[ok]), np.log10(y[ok])
    if lx.std() == 0 or ly.std() == 0:
        return None
    return float(np.corrcoef(lx, ly)[0, 1]), int(ok.sum())


def panel(ax, df, title):
    good = df[np.isfinite(df["alpha_exp"]) & np.isfinite(df["alpha_sim"])
              & (df["alpha_exp"] > 0) & (df["alpha_sim"] > 0)]
    vals = np.concatenate([good["alpha_exp"].values, good["alpha_sim"].values])
    lo, hi = vals.min() / 1.6, min(vals.max() * 1.6, 1.3)
    ax.fill_between([lo, hi], [lo / 2, hi / 2], [lo * 2, hi * 2],
                    color="0.8", alpha=0.30, lw=0, zorder=0)
    ax.plot([lo, hi], [lo, hi], color="0.45", ls="--", lw=1.3, zorder=1)
    for _, r in good.iterrows():
        salt, solv = salt_solv(r.system_id)
        if solv not in SOLV_COLOR or salt not in SALT_MARKER:
            continue
        filled = float(r.conc) != 0.1
        c = SOLV_COLOR[solv]
        ax.scatter(r.alpha_exp, r.alpha_sim, s=130, marker=SALT_MARKER[salt],
                   facecolor=c if filled else "white", edgecolor=c,
                   linewidths=1.9, alpha=0.95, zorder=3)
    rr = loglog_r(good["alpha_exp"], good["alpha_sim"])
    if rr:
        ax.text(0.04, 0.96, f"log-log Pearson\nr = {rr[0]:.2f}  (n={rr[1]})",
                transform=ax.transAxes, ha="left", va="top", fontsize=10.5,
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.4",
                facecolor="white", edgecolor="0.55", linewidth=1.3, alpha=0.9), zorder=5)
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi))
    ax.set_aspect("equal")
    ax.set_xlabel(r"Experiment  ionicity $\alpha=\sigma/\sigma_\mathrm{NE}$")
    ax.set_ylabel(r"Simulation  ionicity $\alpha=\sigma_\mathrm{Onsager}/\sigma_\mathrm{NE}$")
    ax.set_title(title, fontweight="bold", pad=8)
    ax.grid(True, which="major", ls=":", lw=0.7, alpha=0.55)
    ax.grid(True, which="minor", ls=":", lw=0.5, alpha=0.20)
    ax.tick_params(which="both", direction="in", top=True, right=True)


def legend(ax):
    ax.axis("off")
    h = []
    from matplotlib.lines import Line2D
    for salt in ("napf6", "naotf", "lipf6"):
        h.append(Line2D([0], [0], marker=SALT_MARKER[salt], color="0.3", ls="",
                        mfc="0.3", ms=10, label=SALT_LABEL[salt]))
    for solv in ("dme", "diglyme", "pc", "tgdme"):
        h.append(Line2D([0], [0], marker="s", color=SOLV_COLOR[solv], ls="",
                        mfc=SOLV_COLOR[solv], ms=10, label=SOLV_LABEL[solv]))
    h.append(Line2D([0], [0], marker="o", color="0.3", ls="", mfc="white",
                    mec="0.3", ms=10, label="0.1 M (hollow)"))
    h.append(Line2D([0], [0], ls="--", color="0.45", label="1:1"))
    ax.legend(handles=h, loc="center", fontsize=10, frameon=False, ncol=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-01m", action="store_true",
                    help="exclude 0.1 M systems")
    args = ap.parse_args()

    painn = add_ionicity(build_painn())
    opls  = add_ionicity(build_opls())
    tag = ""
    if args.drop_01m:
        painn = painn[np.abs(painn["conc"] - 0.1) > 0.01].copy()
        opls  = opls[np.abs(opls["conc"] - 0.1) > 0.01].copy()
        tag = "_no01M"

    for name, df in [("PaiNN", painn), ("OPLS", opls)]:
        print(f"\n===== {name} ionicity =====")
        cols = ["system_id", "conc", "T", "sigma_exp", "sNE_exp", "alpha_exp",
                "sigma_sim", "sNE_sim", "alpha_sim"]
        print(df[cols].round(3).to_string(index=False))
    painn.to_csv(OUT / f"ionicity_painn{tag}.csv", index=False)
    opls.to_csv(OUT / f"ionicity_opls{tag}.csv", index=False)

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig = plt.figure(figsize=(13.5, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.4], wspace=0.30,
                          left=0.07, right=0.99, top=0.90, bottom=0.13)
    panel(fig.add_subplot(gs[0, 0]), painn, "PaiNN (FP32, NVT)")
    panel(fig.add_subplot(gs[0, 1]), opls,  "OPLS-AA")
    legend(fig.add_subplot(gs[0, 2]))
    cnote = "  (0.1 M excluded)" if args.drop_01m else ""
    fig.suptitle(r"Ionicity (inverse Haven ratio) $\alpha=\sigma/\sigma_\mathrm{NE}$: "
                 r"simulation vs experiment   —   $\sigma_\mathrm{NE}$ from Yeh-Hummer-corrected $D$"
                 + cnote, fontsize=13, fontweight="bold", y=0.975)
    out = OUT / f"figure_ionicity_parity_painn_opls{tag}.png"
    fig.savefig(out, dpi=200, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
