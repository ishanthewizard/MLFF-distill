#!/usr/bin/env python3
"""Cross-displacement (collective ion-displacement correlation) vs lag tau, per
expanding window, for the UMA 1 M / 298 K electrolytes.

This visualises exactly what byteff2's Onsager fit acts on.  onsager_calc builds,
per species s, the COLLECTIVE coordinate

    R_s(t) = sum_over_molecules ( molecular-COM of species s )  -  N_s * system_COM

and forms the time-origin-averaged displacement cross-correlations

    M_ij(tau) = < ( R_i(t+tau) - R_i(t) ) . ( R_j(t+tau) - R_j(t) ) >_t

for the ion species i,j in {+,-}.  The charge-weighted combination

    M_q(tau) = z+^2 M_++  + 2 z+ z- M_+-  + z-^2 M_--   (= M_++ - 2 M_+- + M_-- )

is the collective charge-displacement MSD; its slope over the fixed fit window
[50, 200) ps (byteff2 hardcodes this, after dropping the first 200 frames) sets
sigma_Onsager via the Einstein-Helfand relation.

For every system we overlay M_q(tau) (and, in a 2nd panel, the +/- cross term
M_+-(tau)) for all expanding windows 0-1 ... 0-4 ns, shade the 50-200 ps fit
window, and draw the fitted slope line for each window.  The reproduced
sigma_Onsager is printed and matches the value in conductivity_expanding_*.csv.

Env: fairchemV2_new.  Reloads each 0-4 ns trajectory once (~1 min/system).
"""
import importlib.util
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

OBS = Path("/global/homes/y/yuejian/project/MLFF-distill/APPLICATIONS/"
           "electrolytes/observable_scripts")

# reuse the driver (sets up sys.path -> byteff2, loads compute.py)
spec = importlib.util.spec_from_file_location(
    "drv", OBS / "run_uma_expanding_window_conductivity.py")
drv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drv)

from byteff2.md_utils.onsager_conductivity import (
    correlate_xy, polyfit, remove_center_of_mass_error,
    Lambda_to_ionic_conductivity)

AseTraj = drv.AseTraj
COND_OUT = drv.COND_OUT
OUT = COND_OUT / "cross_displacement"
OUT.mkdir(parents=True, exist_ok=True)

DTYPE = torch.float64
NT_START, NT_END = 50, 200          # byteff2 fit window, frames == ps (1 ps/frame)
DROP = 200                          # byteff2 drops first 200 frames
TAU_PLOT_PS = 400                   # x-axis span for the plots


def collective_coords(pos_win, species_order, species_mass, species_number):
    """Reproduce byteff2 collective coords R_s(t) from an unwrapped window.

    pos_win: (T, N, 3) float64, atoms ordered [cat | anion | solvent].
    Returns Rxt, Ryt, Rzt (lists of (T,) torch tensors, one per species) and the
    per-atom mass vector / molecular masses needed for the COM-error correction.
    """
    gro_mass, ranges, mfrac, start = [], [], [], 0
    mol_masses = []
    for sp in species_order:
        m = species_mass[sp]; nmol = species_number[sp]
        gro_mass += list(m) * nmol
        ranges.append((start, start + len(m) * nmol)); start += len(m) * nmol
        molmass = float(sum(m)); mol_masses.append(molmass)
        mfrac.append(torch.tensor([a / molmass for a in m], dtype=DTYPE))
    AtomMasses = torch.tensor(gro_mass, dtype=DTYPE)
    Mtot = AtomMasses.sum()

    xu = torch.from_numpy(np.ascontiguousarray(pos_win[:, :, 0])).to(DTYPE)
    yu = torch.from_numpy(np.ascontiguousarray(pos_win[:, :, 1])).to(DTYPE)
    zu = torch.from_numpy(np.ascontiguousarray(pos_win[:, :, 2])).to(DTYPE)
    origx = torch.einsum("hi,i->h", xu, AtomMasses) / Mtot
    origy = torch.einsum("hi,i->h", yu, AtomMasses) / Mtot
    origz = torch.einsum("hi,i->h", zu, AtomMasses) / Mtot

    Rxt, Ryt, Rzt = [], [], []
    for i, sp in enumerate(species_order):
        a, b = ranges[i]; nmol = species_number[sp]; napm = len(species_mass[sp])
        x1 = xu[:, a:b].reshape(-1, nmol, napm)
        y1 = yu[:, a:b].reshape(-1, nmol, napm)
        z1 = zu[:, a:b].reshape(-1, nmol, napm)
        Rxt.append(torch.einsum("hij,j->h", x1, mfrac[i]) - nmol * origx)
        Ryt.append(torch.einsum("hij,j->h", y1, mfrac[i]) - nmol * origy)
        Rzt.append(torch.einsum("hij,j->h", z1, mfrac[i]) - nmol * origz)
    return Rxt, Ryt, Rzt, torch.tensor(mol_masses, dtype=DTYPE)


def pair_msd(Ra, Rb):
    """xyz-summed collective displacement cross-correlation M(tau) (Angstrom^2)."""
    return (correlate_xy(Ra[0], Rb[0]) + correlate_xy(Ra[1], Rb[1])
            + correlate_xy(Ra[2], Rb[2]))


def sigma_from_slopes(slope_mat, mol_masses, charges, n_total, T_K, V):
    """Reproduce onsager_calc sigma (mS/cm) from the collective-MSD slope matrix."""
    from byteff2.md_utils.onsager_conductivity import unit
    RawLambda = slope_mat * (unit.angstrom**2 / unit.ps) / (1e-10 * unit.m**2 / unit.s) \
        / (6 * n_total)
    Lambda = remove_center_of_mass_error(RawLambda, mol_masses)
    return float(Lambda_to_ionic_conductivity(Lambda, charges, n_total, T_K, V))


def run_system(dirname, cfg):
    label = cfg["label"]
    traj = drv.BASE / dirname / f"{dirname}.traj"
    print(f"\n=== {label} ({cfg['cat']}/{cfg['anion']}/{cfg['solvent']}) ===", flush=True)

    dt_ps = drv.DT_FS / 1000.0
    stride = max(1, round(drv.LOAD_DT_PS / dt_ps))
    with AseTraj(str(traj)) as t:
        f0 = t[0]; syms = f0.get_chemical_symbols(); mss = f0.get_masses()
    (reorder, species_order, species_mass, species_number,
     species_charge, ncat, nani, nsol) = drv.build_species(
        syms, mss, cfg["cat"], cfg["anion"], cfg["solvent"])
    n_load = int(round(drv.MAX_WINDOW_NS * 1000.0))            # 4000 frames @ 1 ps
    pos, vol = drv.load_unwrapped(traj, reorder, stride, n_load)
    print(f"    loaded {pos.shape[0]} frames; N+={ncat} N-={nani}", flush=True)

    charges = torch.tensor([species_charge[s] for s in species_order], dtype=DTYPE)
    n_total = ncat + nani + nsol

    # per-window collective cross-MSD curves
    windows = [w for w in drv.WINDOWS_NS if w <= drv.MAX_WINDOW_NS + 1e-9]
    curves = {}     # w -> dict(tau, M_q, M_pm, M_pp, M_mm, slope_q, sig)
    for w in windows:
        n_w = int(round(w * 1000.0))
        np_xyz = pos[:n_w][DROP:]                     # drop first 200, as byteff2 does
        V_w = float(vol[:n_w].mean())
        Rx, Ry, Rz, mol_masses = collective_coords(
            np_xyz, species_order, species_mass, species_number)
        R = [(Rx[i], Ry[i], Rz[i]) for i in range(len(species_order))]
        # species order is [cat(0), anion(1), solvent(2)]
        M_pp = pair_msd(R[0], R[0]).numpy()
        M_pm = pair_msd(R[0], R[1]).numpy()
        M_mm = pair_msd(R[1], R[1]).numpy()
        zc, za = species_charge[species_order[0]], species_charge[species_order[1]]
        M_q = zc*zc*M_pp + 2*zc*za*M_pm + za*za*M_mm         # charge MSD (ions only)

        # full slope matrix (incl solvent cross terms) to reproduce sigma exactly
        nsp = len(species_order)
        slope = torch.zeros((nsp, nsp), dtype=DTYPE)
        lags = torch.arange(NT_START, NT_END, dtype=DTYPE)
        for i in range(nsp):
            for j in range(i, nsp):
                m = pair_msd(R[i], R[j])
                _b, k = polyfit(lags, m[NT_START:NT_END], 1)
                slope[i, j] = k; slope[j, i] = k
        sig = sigma_from_slopes(slope, mol_masses, charges, n_total, drv.T_K, V_w)

        tau = np.arange(len(M_q))                    # ps
        # linear fit of the charge MSD over the fit window (for the drawn line)
        p = np.polyfit(tau[NT_START:NT_END], M_q[NT_START:NT_END], 1)  # [slope, intercept]
        curves[w] = dict(tau=tau, M_q=M_q, M_pm=M_pm, M_pp=M_pp, M_mm=M_mm,
                         fit=p, sig=sig)
        print(f"    window 0-{w:>4.1f} ns  sigma_Onsager(reproduced)={sig:7.4f} mS/cm", flush=True)

    # ── figure: 2 panels (charge MSD | +/- cross term), windows overlaid ──────
    colors = cm.viridis(np.linspace(0, 0.92, len(windows)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for c, w in zip(colors, windows):
        d = curves[w]; tau = d["tau"]
        m = tau <= TAU_PLOT_PS
        lab = f"0-{w:.1f} ns  (σ={d['sig']:.2f})"
        axes[0].plot(tau[m], d["M_q"][m], color=c, lw=1.8, label=lab)
        # fitted slope line over the fit window
        xf = np.array([NT_START, NT_END])
        axes[0].plot(xf, d["fit"][0]*xf + d["fit"][1], color=c, lw=2.4, ls="--",
                     alpha=0.9)
        axes[1].plot(tau[m], d["M_pm"][m], color=c, lw=1.8, label=f"0-{w:.1f} ns")
    for ax, ttl, yl in [
        (axes[0], f"{cfg['cat']}$^+$/{cfg['anion']}$^-$ charge displacement "
                  r"$M_q(\tau)=\langle|\sum_s z_s\,\Delta R_s|^2\rangle$",
         r"collective charge MSD  (Å$^2$)"),
        (axes[1], "cation–anion cross displacement "
                  r"$M_{+-}(\tau)=\langle\Delta R_+\cdot\Delta R_-\rangle$",
         r"cross displacement  (Å$^2$)")]:
        ax.axvspan(NT_START, NT_END, color="grey", alpha=0.18,
                   label="fit window 50–200 ps")
        ax.set_xlabel(r"lag $\tau$ (ps)")
        ax.set_ylabel(yl)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlim(0, TAU_PLOT_PS)
        ax.grid(True, ls=":", alpha=0.6)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{label}  1 M 298 K — collective displacement vs τ per expanding "
                 f"window (dashed = 50–200 ps Onsager fit)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = OUT / f"cross_displacement_{label}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {p}", flush=True)

    # save curves for reproducibility
    np.savez_compressed(
        OUT / f"cross_displacement_{label}.npz",
        windows=np.array(windows),
        **{f"tau_{w}": curves[w]["tau"] for w in windows},
        **{f"Mq_{w}": curves[w]["M_q"] for w in windows},
        **{f"Mpm_{w}": curves[w]["M_pm"] for w in windows},
        **{f"Mpp_{w}": curves[w]["M_pp"] for w in windows},
        **{f"Mmm_{w}": curves[w]["M_mm"] for w in windows},
        sig=np.array([curves[w]["sig"] for w in windows]),
    )
    return {label: {w: curves[w]["sig"] for w in windows}}


def main():
    for dirname, cfg in drv.SYSTEMS:
        try:
            run_system(dirname, cfg)
        except Exception:
            import traceback
            print(f"!! FAILED {cfg['label']}:\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
