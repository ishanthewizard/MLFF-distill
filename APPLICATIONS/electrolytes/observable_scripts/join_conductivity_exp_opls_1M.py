#!/usr/bin/env python
"""Attach concentration/solvent-exact experimental conductivity to the OPLS 1 M
sim results and make a parity plot. Sim sigma is mS/cm; exp IC2 is uS/cm.

Solvent label map (sim -> exp table): Diglyme -> DEGDME, DME -> DME, PC -> PC.
Match on (cation, anion, solvent, concentration=1.0, |T-298|<=T_tol), taking the
closest-T non-NaN IC2 row.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOL_SIM_TO_EXP = {"DME": "DME", "Diglyme": "DEGDME", "PC": "PC", "TGDME": "TEGDME"}


def match_exp(exp, cation, anion, solvent_exp, conc, T_K, T_tol=5.0):
    cand = exp[(exp["cation"] == cation) & (exp["anion"] == anion) &
               (exp["solvent"] == solvent_exp) &
               (np.isclose(exp["concentration (M)"].astype(float), conc, atol=1e-6)) &
               (exp["IC2 (uS/cm)"].notna())]
    if cand.empty:
        return None
    dT = (cand["temperature (K)"].astype(float) - T_K).abs()
    if dT.min() > T_tol:
        return None
    return cand.loc[dT.idxmin()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-csv", required=True)
    ap.add_argument("--exp-csv", default="/global/homes/y/yuejian/project/MLFF-distill/"
                    "m5024/distillation_project/experiment_data/cleaned_version/conductivity.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--T-tol", type=float, default=5.0)
    args = ap.parse_args()

    sim = pd.read_csv(args.sim_csv)
    sim = sim[sim.get("error", "").fillna("") == ""] if "error" in sim else sim
    exp = pd.read_csv(args.exp_csv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, r in sim.iterrows():
        sol_exp = SOL_SIM_TO_EXP.get(r["solvent"], r["solvent"])
        m = match_exp(exp, r["cation"], r["anion"], sol_exp, float(r["concentration_M"]),
                      float(r["T_K"]), args.T_tol)
        d = dict(r)
        d["solvent_exp"] = sol_exp
        d["sim_sigma_onsager_uS_cm"] = r["sigma_onsager_mS_cm"] * 1000.0
        d["sim_sigma_NE_uS_cm"] = r["sigma_NE_mS_cm"] * 1000.0
        if m is not None:
            d["exp_conductivity_uS_cm"] = float(m["IC2 (uS/cm)"])
            d["exp_temperature_K"] = float(m["temperature (K)"])
        else:
            d["exp_conductivity_uS_cm"] = np.nan
            d["exp_temperature_K"] = np.nan
        rows.append(d)

    df = pd.DataFrame(rows)
    csv_path = out / "conductivity_with_exp.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")
    cols = ["name", "salt", "solvent", "traj_ns", "sigma_onsager_mS_cm", "sigma_NE_mS_cm",
            "sim_sigma_onsager_uS_cm", "sim_sigma_NE_uS_cm", "exp_conductivity_uS_cm",
            "exp_temperature_K"]
    print("\n" + df[[c for c in cols if c in df]].to_string(index=False))

    # parity plot (log-log): sim Onsager & NE vs exp
    sub = df[df["exp_conductivity_uS_cm"].notna()].copy()
    if len(sub):
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        x = sub["exp_conductivity_uS_cm"].values
        ax.scatter(x, sub["sim_sigma_onsager_uS_cm"], s=70, c="#1f77b4",
                   edgecolor="k", label="Onsager (collective)", zorder=3)
        ax.scatter(x, sub["sim_sigma_NE_uS_cm"], s=70, marker="^", c="#ff7f0e",
                   edgecolor="k", label="Nernst-Einstein", zorder=3)
        for _, rr in sub.iterrows():
            ax.annotate(f"{rr['salt']}/{rr['solvent']}",
                        (rr["exp_conductivity_uS_cm"], rr["sim_sigma_onsager_uS_cm"]),
                        fontsize=7, xytext=(4, 3), textcoords="offset points")
        allv = np.concatenate([x, sub["sim_sigma_onsager_uS_cm"].values,
                               sub["sim_sigma_NE_uS_cm"].values])
        allv = allv[allv > 0]
        lo, hi = allv.min() * 0.5, allv.max() * 2
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel("experimental ionic conductivity (µS/cm)")
        ax.set_ylabel("simulated ionic conductivity (µS/cm)")
        ax.set_title("OPLS 1 M NPT — conductivity parity (298 K)")
        ax.grid(alpha=0.25, which="both"); ax.legend()
        fig.tight_layout()
        fig.savefig(out / "conductivity_parity_with_exp.png", dpi=140)
        plt.close(fig)
        print(f"wrote {out/'conductivity_parity_with_exp.png'}")


if __name__ == "__main__":
    main()
