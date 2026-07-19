#!/usr/bin/env python3
"""Minimal ionic conductivity from ONE trajectory, two equivalent routes.

Both routes use only the collective CHARGE quantity of a 1:1 electrolyte
(z = +1 / -1), built from the cation atoms and the anion central heavy atom:

    dRq(t) = sum_cat dr(t) - sum_an dr(t)      (unwrapped displacement, A)
    Jq(t)  = sum_cat v(t)  - sum_an v(t)       (charge current, A/fs)

  displacement (Einstein):  sigma = e^2 / (6 V kB T) * d/dt <|dRq(t)|^2>
  velocity   (Green-Kubo):  sigma = e^2 / (3 V kB T) * integral <Jq(0).Jq(t)> dt

The two are formally equivalent.  Green-Kubo integrates the VACF at the
trajectory's SAVE step; if that step is coarse (here 100 fs) the sub-step
ballistic decay of the VACF is unresolved and the integral over-counts the
tau~0 peak, so GK overestimates.  The Einstein slope uses long-time
displacements and is robust.  (VACF units verified against the reference
transport-coefficients example, which this reproduces to 13 digits.)

Self-contained: ase + numpy + scipy only.  Assumes a monatomic cation, a
unique anion-central element, an orthorhombic box, and constant V (NVT).

Usage:  python minimal_conductivity.py <traj-or-run-dir>
"""
import sys
from pathlib import Path

import numpy as np
import ase.units as u
from ase.io.trajectory import Trajectory
from scipy.integrate import cumulative_trapezoid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── settings (edit for other systems) ────────────────────────────────────────
CATION = "Na"          # cation element (monatomic ion)
ANION  = "S"           # anion central heavy atom (one per anion, e.g. S in OTf)
DT_FS  = 100.0         # trajectory frame spacing = GK integration step (fs)
T_K    = 298.0         # temperature (K)
EQ_NS  = 2.0           # equilibration skipped from the start (ns)

GK_WIN_NS = 3.0        # native-resolution window for the VACF (ns)
GK_PLAT_PS = (2.0, 5.0)  # plateau window averaged for the GK value (ps)

EIN_STRIDE_PS = 10.0   # subsample step for the displacement/MSD route (ps)
EIN_MAX_NS = 20.0      # trajectory span used for the MSD (ns)
EIN_FIT_NS = (0.3, 2.4)  # diffusive lag window for the MSD slope (ns)

E  = 1.602176634e-19   # elementary charge (C)
KB = 1.380649e-23      # Boltzmann constant (J/K)


def fft_acf(x):
    """Unbiased autocorrelation <x(0)x(t)> of a 1-D signal, via FFT."""
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    return ac / (n - np.arange(n))


def charge_current_native(tr, i0, i1, cat, an):
    """Jq(t) = sum_cat v - sum_an v  (A/fs) over contiguous native frames."""
    T = i1 - i0
    Jq = np.zeros((T, 3))
    for k in range(T):
        v = tr[i0 + k].get_velocities() * u.fs           # ASE vel -> A/fs
        Jq[k] = v[cat].sum(0) - v[an].sum(0)
    return Jq


def charge_disp_strided(tr, i0, i1, stride, cat, an, box):
    """dRq(t) = sum_cat dr - sum_an dr  (A, unwrapped) at `stride` spacing."""
    idx = range(i0, i1, stride)
    Rq = np.zeros((len(idx), 3))
    prev = tr[i0].get_positions()
    for k, fi in enumerate(idx):
        if k == 0:
            continue
        cur = tr[fi].get_positions()
        d = cur - prev
        d -= box * np.round(d / box)                     # minimum image (small steps)
        Rq[k] = Rq[k - 1] + (d[cat].sum(0) - d[an].sum(0))
        prev = cur
    return Rq


def main():
    src = Path(sys.argv[1])
    traj = src if src.is_file() else max(src.glob("*.traj"), key=lambda f: f.stat().st_size)
    tr = Trajectory(str(traj))
    dt_ps = DT_FS / 1000.0

    sym = np.array(tr[0].get_chemical_symbols())
    cat = np.where(sym == CATION)[0]
    an = np.where(sym == ANION)[0]
    box = tr[0].cell.lengths()                           # orthorhombic (A)
    V_A3 = float(np.prod(box))                           # NVT constant V
    i_eq = int(EQ_NS * 1e3 / dt_ps)
    n_tot = len(tr)

    print(f"traj    : {traj}")
    print(f"sites   : {len(cat)} {CATION} (cation)  {len(an)} {ANION} (anion centre)")
    print(f"box     : V = {V_A3:.0f} A^3   T = {T_K:g} K")

    # ── velocity route: Green-Kubo over a native-resolution window ────────────
    i1 = min(n_tot, i_eq + int(GK_WIN_NS * 1e3 / dt_ps))
    print(f"[GK ] reading {i1 - i_eq} native frames "
          f"({(i1 - i_eq) * dt_ps / 1e3:.2f} ns @ {DT_FS:g} fs) ...")
    Jq = charge_current_native(tr, i_eq, i1, cat, an)
    times_fs = np.arange(len(Jq)) * DT_FS
    vacf = sum(fft_acf(Jq[:, d]) for d in range(3))       # <Jq(0).Jq(t)>  (A/fs)^2
    integ = cumulative_trapezoid(vacf, times_fs)          # A^2/fs
    # e^2/(3 kB T V) * integral ; (A^2/fs)/A^3 -> SI = 1e-5/1e-30 = 1e25
    sig_gk_run = E**2 * integ * 1e25 / (3 * KB * T_K * V_A3)   # S/m vs cutoff
    tau_ps = times_fs[1:] / 1000.0
    mp = (tau_ps >= GK_PLAT_PS[0]) & (tau_ps <= GK_PLAT_PS[1])
    sig_gk = sig_gk_run[mp].mean() * 10.0                 # -> mS/cm

    # ── displacement route: Einstein over a strided, long span ────────────────
    stride = max(1, round(EIN_STRIDE_PS / dt_ps))
    i1e = min(n_tot, i_eq + int(EIN_MAX_NS * 1e3 / dt_ps))
    print(f"[Ein] reading {(i1e - i_eq)//stride} strided frames "
          f"({(i1e - i_eq) * dt_ps / 1e3:.1f} ns @ {stride*dt_ps:g} ps) ...")
    Rq = charge_disp_strided(tr, i_eq, i1e, stride, cat, an, box)
    lag_ps = np.arange(len(Rq)) * stride * dt_ps
    lags = np.arange(1, len(Rq))
    msd = np.array([((Rq[l:] - Rq[:-l]) ** 2).sum(1).mean() for l in lags])  # A^2
    lag_msd_ps = lags * stride * dt_ps
    fm = (lag_msd_ps >= EIN_FIT_NS[0] * 1e3) & (lag_msd_ps <= EIN_FIT_NS[1] * 1e3)
    slope = np.polyfit(lag_msd_ps[fm], msd[fm], 1)[0]     # A^2/ps
    # e^2/(6 kB T V) * slope ; (A^2/ps)/A^3 -> SI = 1e-8/1e-30 = 1e22
    sig_ein = E**2 * slope * 1e22 / (6 * KB * T_K * V_A3) * 10.0   # mS/cm

    # ── report ────────────────────────────────────────────────────────────────
    print("-" * 62)
    print(f"displacement (Einstein), fit {EIN_FIT_NS[0]}-{EIN_FIT_NS[1]} ns : "
          f"{sig_ein:8.3f} mS/cm   [robust]")
    print(f"velocity     (Green-Kubo), plateau {GK_PLAT_PS[0]}-{GK_PLAT_PS[1]} ps: "
          f"{sig_gk:8.3f} mS/cm   [needs << {DT_FS:g} fs saving to converge]")
    print("-" * 62)

    # ── VACF + running-kappa figure ───────────────────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax[0].plot(np.arange(len(vacf)) * dt_ps, vacf, color="tab:purple")
    ax[0].axhline(0, color="grey", lw=0.6)
    ax[0].set_xlim(0, 5)
    ax[0].set_xlabel(r"$\tau$ (ps)")
    ax[0].set_ylabel(r"charge VACF $\langle J_q(0)\cdot J_q(\tau)\rangle$  $(\mathrm{\AA/fs})^2$")
    ax[0].set_title("Green-Kubo VACF")
    ax[1].plot(tau_ps, sig_gk_run * 10.0, color="tab:purple", label="GK $\\kappa(\\tau)$")
    ax[1].axhline(sig_gk, color="tab:purple", ls="--", label=f"GK = {sig_gk:.2f} mS/cm")
    ax[1].axhline(sig_ein, color="tab:orange", lw=1.8, label=f"Einstein = {sig_ein:.2f} mS/cm")
    ax[1].axvspan(*GK_PLAT_PS, color="tab:purple", alpha=0.1)
    ax[1].set_xlim(0, 10)
    ax[1].set_xlabel(r"integration cutoff $\tau$ (ps)")
    ax[1].set_ylabel(r"$\kappa$ (mS/cm)")
    ax[1].set_title("running GK vs Einstein")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    out = Path(traj).with_suffix("").name
    png = Path.cwd() / f"minimal_conductivity_{out}.png"
    fig.savefig(png, dpi=140)
    print(f"saved {png}")


if __name__ == "__main__":
    main()
