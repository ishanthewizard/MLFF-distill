#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Na self-diffusion from VACF (Green–Kubo) for six NaPF6 solvents.
Outputs under: /projects/beye/iamin/observables/distilled_vacf

Per solvent:
  - VACF_<SOLVENT>_Na.csv                     (time_ps, VACF_A2fs2)
  - GK_integral_<SOLVENT>_Na.csv              (time_ps, GK_int_A2_per_fs, GK_int_SI_m2_s)

Combined:
  - D_summary.csv
  - VACF_and_GK_Na_all.png  (top: VACF overlays; bottom: GK integral overlays)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from ase.io.trajectory import Trajectory
from ase.geometry import find_mic
import matplotlib.pyplot as plt
from tqdm import tqdm

# ----------------------------- config -----------------------------
OUT_DIR = Path("/global/homes/y/yuejian/project/MLFF-distill/yuejian/MD_naotf/vacf")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOLVENTS = ["diglyme", "dme", "PC", "TGDME"]
TRAJS = ["/global/homes/y/yuejian/project/MLFF-distill/yuejian/MD_naotf/md_omol_naotf_diglyme_1m_s1p1/md_omol_naotf_diglyme_1m_s1p1.traj",
         "/global/homes/y/yuejian/project/MLFF-distill/yuejian/MD_naotf/md_omol_naotf_dme_s1p1_omol/md_omol_naotf_dme_s1p1_omol.traj",
         "/global/homes/y/yuejian/project/MLFF-distill/yuejian/MD_naotf/md_omol_naotf_pc_1m_s1p1/md_omol_naotf_pc_1m_s1p1.traj",
         "/global/homes/y/yuejian/project/MLFF-distill/yuejian/MD_naotf/md_omol_naotf_tgdme_1m_s1p1/md_omol_naotf_tgdme_1m_s1p1.traj"]
SPECIES = "Na"

# Trajectories sampled every 50 fs
DT_FS_FALLBACK = 10  # fs

# Optional cropping (None = use full run)
TMAX_PS = None

# Compute integral up to first zero crossing (standard practice)
TAIL_FIT = False         # keep False unless you want exponential tail continuation
TAIL_WINDOW_PS = 1.5

# ----------------------------- helpers -----------------------------

def unwrap_positions(frames):
    """
    Unwrap positions via cumulative MIC using per-frame cell and pbc.
    Returns array R [T, N, 3] in Å.
    """
    T = len(frames)
    R = np.empty((T, len(frames[0]), 3), dtype=float)
    R[0] = frames[0].get_positions(wrap=True)
    for i in tqdm(range(1, T), desc="Unwrapping", leave=False):
        r_prev = R[i-1]
        r_now  = frames[i].get_positions(wrap=True)
        cell   = frames[i].get_cell()
        pbc    = frames[i].get_pbc()
        disp, _ = find_mic(r_now - r_prev, cell, pbc=pbc)
        R[i] = r_prev + disp
    return R

def estimate_velocities_from_positions(R, dt_fs):
    """Central differences (Å/fs)."""
    V = np.zeros_like(R)
    V[1:-1] = (R[2:] - R[:-2]) / (2.0 * dt_fs)
    V[0]    = (R[1] - R[0])   / dt_fs
    V[-1]   = (R[-1] - R[-2]) / dt_fs
    return V

def vacf_fft_multi(V, stride=1, max_lag=None):
    """
    VACF via FFT for multi-particle velocities.
    V: [T, N, 3] in Å/fs
    Returns vacf[0..max_lag] in Å^2/fs^2, averaged over atoms and components.
    """
    V = V[::stride]
    T, N, D = V.shape
    if max_lag is None:
        max_lag = T - 1

    # remove per-particle drift
    Vm = V - V.mean(axis=0, keepdims=True)
    X  = Vm.reshape(T, N*D)

    nfft = 1 << (2*T - 1).bit_length()
    F    = np.fft.rfft(X, n=nfft, axis=0)
    S    = F * np.conjugate(F)
    ac   = np.fft.irfft(S, n=nfft, axis=0).real[:T]
    ac  /= (np.arange(T, 0, -1)[:, None])  # unbiased normalization
    vacf = ac.mean(axis=1)[:max_lag+1]
    return vacf

def first_zero_crossing(y):
    for i in range(1, len(y)):
        if y[i-1] > 0.0 and y[i] <= 0.0:
            return i
    return None

def cumulative_trapz(y, dx):
    out = np.zeros_like(y, dtype=float)
    if len(y) < 2:
        return out
    out[1:] = np.cumsum(0.5*(y[1:] + y[:-1]) * dx)
    return out

def fit_exponential_tail(t, y):
    """Fit y ≈ A * exp(-t/τ) on positive y; t in ps. Returns (A, tau_ps) or None."""
    mask = y > 0
    if mask.sum() < 5:
        return None
    t_pos = t[mask]
    y_pos = y[mask]
    logy  = np.log(y_pos)
    A = np.vstack([np.ones_like(t_pos), -t_pos]).T
    sol, *_ = np.linalg.lstsq(A, logy, rcond=None)
    logA, inv_tau = sol
    if inv_tau <= 0:
        return None
    return np.exp(logA), 1.0 / inv_tau

# ----------------------------- per-solvent pipeline -----------------------------
def process_one(solvent, traj_path):
    # Load frames
    with Trajectory(traj_path) as tr:
        frames = [at.copy() for at in tr]
    if len(frames) < 8:
        raise RuntimeError(f"{solvent}: not enough frames")

    # Species mask
    sym = np.array(frames[0].get_chemical_symbols())
    idx = np.where(sym == SPECIES)[0]
    if idx.size == 0:
        raise RuntimeError(f"{solvent}: no {SPECIES} atoms found")

    # Time base (fs)
    times_fs = None
    dt_fs    = DT_FS_FALLBACK

    # Optional crop by physical time
    if TMAX_PS is not None and times_fs is not None:
        T_keep = int(min(len(frames), np.floor(TMAX_PS * 1e3 / dt_fs)))
        frames = frames[:T_keep]
        times_fs = times_fs[:T_keep]

    # Unwrap, restrict to species, get velocities (from file if present, else estimate)
    R = unwrap_positions(frames)[:, idx, :]
    try:
        V_all = np.stack([at.get_velocities() for at in frames], axis=0)[:, idx, :]
        if np.isnan(V_all).any():
            raise ValueError
        V = V_all  # Å/fs if your trajectory stores Å/fs
    except Exception:
        V = estimate_velocities_from_positions(R, dt_fs)

    # VACF (Å^2/fs^2)
    vacf = vacf_fft_multi(V, stride=1)
    dt_eff_fs = dt_fs
    t_ps = np.arange(len(vacf)) * dt_eff_fs * 1e-3

    # First zero crossing
    iz = first_zero_crossing(vacf)
    if iz is None:
        iz = len(vacf) - 1
    t_zero_ps = float(t_ps[iz])

    # GK integral
    GK_A2_per_fs = (1.0/3.0) * cumulative_trapz(vacf, dx=dt_eff_fs)   # Å^2/fs
    GK_SI_m2_s   = GK_A2_per_fs * 1e-5                                # m^2/s (Å^2/fs → m^2/s)
    D_base_SI    = GK_SI_m2_s[iz]
    note = "integral to first zero crossing"
    use_tail = False

    # Optional exponential tail continuation
    if TAIL_FIT and iz < len(vacf) - 5:
        t0 = t_ps[iz]
        w  = float(TAIL_WINDOW_PS)
        mask = (t_ps >= t0) & (t_ps <= min(t0 + w, t_ps[-1]))
        fit = fit_exponential_tail(t_ps[mask] - t0, vacf[mask])
        if fit is not None:
            A, tau_ps = fit
            # VACF units Å^2/fs^2; tau in ps → fs
            D_tail_SI = (1.0/3.0) * A * (tau_ps * 1e3) * 1e-5
            D_base_SI += D_tail_SI
            note += f" + exp tail ({w:.2f} ps)"
            use_tail = True
        else:
            note += " (tail fit failed)"

    # Save per-solvent curves
    pd.DataFrame({"time_ps": t_ps, "VACF_A2fs2": vacf}).to_csv(
        OUT_DIR / f"VACF_{solvent}_{SPECIES}.csv", index=False)
    pd.DataFrame({
        "time_ps": t_ps,
        "GK_int_A2_per_fs": GK_A2_per_fs,
        "GK_int_SI_m2_s": GK_SI_m2_s
    }).to_csv(OUT_DIR / f"GK_integral_{solvent}_{SPECIES}.csv", index=False)

    return {
        "solvent": solvent,
        "dt_fs": dt_eff_fs,
        "n_frames": len(frames),
        f"n_{SPECIES}": int(idx.size),
        "t_max_ps": float(t_ps[-1]),
        "t_zero_ps": float(t_zero_ps),
        "use_tail": use_tail,
        "D_SI_m2_s": float(D_base_SI),
        "D_cm2_s": float(D_base_SI * 1e4),
        "note": note,
        "t_ps": t_ps,                 # for plotting
        "vacf": vacf,
        "GK_SI_m2_s": GK_SI_m2_s,
    }

# ----------------------------- run all & plot -----------------------------
if __name__ == "__main__":
    results = []
    for s, p in zip(SOLVENTS, TRAJS):
        try:
            print(f"Processing {s} ...")
            res = process_one(s, p)
            results.append(res)
            print(f"  D = {res['D_cm2_s']:.4e} cm^2/s  ({res['note']})")
        except Exception as e:
            print(f"[WARN] {s}: {e}")

    if results:
        # Summary CSV
        df = pd.DataFrame([{k: v for k, v in r.items()
                            if k not in ("t_ps", "vacf", "GK_SI_m2_s")} for r in results])
        df.to_csv(OUT_DIR / "D_summary.csv", index=False)
        print(f"Saved {OUT_DIR / 'D_summary.csv'}")

        # One combined figure: top = VACF overlays, bottom = GK integral overlays
        plt.figure(figsize=(8.5, 7.0))

        # Top: VACF (limit to first 2 ps)
        ax1 = plt.subplot(2, 1, 1)
        for r in results:
            mask = r["t_ps"] <= 2.0
            ax1.plot(r["t_ps"][mask], r["vacf"][mask], lw=1.6, label=r["solvent"])
        ax1.set_xlim(0, 2.0)
        ax1.set_xlabel("time (ps)")
        ax1.set_ylabel(r"VACF (Å$^2$/fs$^2$)")
        ax1.set_title(f"Na VACF — {len(results)} systems (first 2 ps)")
        ax1.grid(True, linestyle=":")
        ax1.legend(ncol=3, fontsize=8)

        # Bottom: GK integral (D(t)) overlays
# Bottom: GK integral on log scale (positive part only)
        ax2 = plt.subplot(2, 1, 2)
        for r in results:
            pos = r["GK_SI_m2_s"] > 0
            ax2.plot(r["t_ps"][pos], r["GK_SI_m2_s"][pos], lw=1.6, label=r["solvent"])
        ax2.set_yscale("log")
        ax2.set_xlabel("time (ps)")
        ax2.set_ylabel(r"$D(t)$ from GK integral (m$^2$/s)")
        ax2.set_title("Green–Kubo integral (positive part; log scale)")
        ax2.grid(True, linestyle=":")

        plt.tight_layout()
        out_png = OUT_DIR / "VACF_and_GK_Na_all.png"
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"Saved combined plot → {out_png}")
    else:
        print("No successful results.")
