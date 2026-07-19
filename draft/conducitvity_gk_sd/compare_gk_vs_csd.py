#!/usr/bin/env python3
"""Compare ionic conductivity from two methods on ONE ASE trajectory.

  1. Cross-square displacement (Einstein / Onsager, "CSD")
     -> compute.run_onsager_conductivity_mdcraft
     Fits the slope of the collective cross-displacement
     <dR_i . dR_j>(tau) in the diffusive regime (mdcraft backend).

  2. Green-Kubo ("GK")
     -> integrate the collective velocity ACF.  The ACF is built with the
     transport-coefficients submodule (lij_analysis.TransportCoefficients.
     compute_acf), then L_ij = 1/(3 kB T V) * int<sum v_i(t).sum v_j(0)> dt
     and  kappa = (L++ + L-- - 2 L+-) * 10 * e^2   [mS/cm]
     (exactly the notebook recipe in
      submodule/transport-coefficients/example_calculation/).

Both methods use the SAME ion-site definition -- cation atom and the anion's
central heavy atom -- and both remove the system centre-of-mass drift, so the
two conductivities are directly comparable (Green-Kubo and Einstein are
formally equivalent; they should agree up to statistics/convergence).

CAVEAT (read the printed interpretation): Green-Kubo integrates the velocity
ACF, whose fast (sub-100 fs) ballistic decay must be time-resolved.  If the
trajectory is saved coarsely (e.g. every 100 fs) the ACF near tau=0 is
under-sampled and trapezoidal integration over-counts the ballistic peak, so
GK OVERESTIMATES the conductivity.  The Einstein/CSD method uses long-time
displacements and is insensitive to frame spacing, so when the two disagree
the CSD value is the trustworthy one.  This script reports both plus their
ratio precisely so that this effect is visible rather than hidden.

Outputs
  - prints a side-by-side conductivity comparison
  - saves ONE figure: (left) collective VACF vs lag tau, (right) running
    Green-Kubo kappa(tau) with the CSD value overlaid.

This script imports the source modules; it does not modify any of them.
NOTE: scipy>=1.14 removed integrate.cumtrapz, which TransportCoefficients.
compute_lij relies on, so the (trivial) L_ij integral is re-done here with
scipy.integrate.cumulative_trapezoid -- identical math, working scipy call.
"""

import argparse
import importlib.util
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from tqdm import tqdm
import ase.units as ase_u

HERE = Path(__file__).resolve().parent


def _find_repo_root(start):
    """Ascend from ``start`` until the conductivity source dir is found.

    Lets this script live anywhere inside the MLFF-distill checkout (e.g. a
    draft/ scratch folder) while still importing the source modules by path.
    """
    marker = (Path("APPLICATIONS") / "electrolytes" / "observable_scripts"
              / "conductivity" / "compute.py")
    for p in [start, *start.parents]:
        if (p / marker).exists():
            return p
    raise RuntimeError(f"Cannot locate MLFF-distill repo root above {start}")


REPO_ROOT = _find_repo_root(HERE)
COND_DIR = (REPO_ROOT / "APPLICATIONS" / "electrolytes"
            / "observable_scripts" / "conductivity")


# ── import source modules by path (no cwd assumptions, no src edits) ─────────
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


compute = _load("cond_compute", COND_DIR / "compute.py")
lij = _load(
    "lij_analysis",
    REPO_ROOT / "submodule" / "transport-coefficients"
    / "example_calculation" / "lij_analysis.py",
)

# physical constants (mirror lij_analysis / the notebook)
_A2M, _FS2S, _KB = 1e-10, 1e-15, 1.3806504e-23
_E2C = 1.60217662e-19                              # elementary charge (C)
_CONVERT_LIJ = 1.0 / (_A2M * _FS2S)


# ══════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════
def find_traj(path):
    """Accept a .traj file or a run directory; return the .traj Path."""
    p = Path(path)
    if p.is_file():
        return p
    cands = sorted(p.glob("*.traj"), key=lambda f: f.stat().st_size, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No .traj file found in {p}")
    return cands[0]


def guess_species(traj_path):
    """Best-effort (cat, anion, solvent) keys from the path; None if unsure."""
    parts = [s.lower() for s in Path(traj_path).parts]
    tokens = [t for s in parts for t in s.split("_")]
    cat = next((k for k in compute.cation_dict if any(k.lower() in s for s in parts)), None)
    anion = next((k for k in compute.anion_dict if any(k.lower() in s for s in parts)), None)
    solv = next((k for k in compute.solvent_dict if k.lower() in tokens), None)
    return cat, anion, solv


def guess_dt_T(traj_path, dt_default=100.0, T_default=298.0):
    name = " ".join(Path(traj_path).parts[-3:])
    m_dt = re.search(r"(\d+)\s*fs", name)
    m_T = re.search(r"(\d+)\s*K", name)
    return (float(m_dt.group(1)) if m_dt else dt_default,
            float(m_T.group(1)) if m_T else T_default)


def ion_site_indices(symbols0, cat_symbol, anion_symbol):
    """Original-order atom index of each ion site.

    cation -> its (monoatomic) atom; anion -> its central heavy atom
    (compute.anion_central_dict), matching the mdcraft CSD backend.
    """
    symbols0 = np.asarray(symbols0)
    cat_groups = compute.direct_groups_from_species(symbols0, compute.cation_dict[cat_symbol])
    anion_groups = compute.direct_groups_from_species(symbols0, compute.anion_dict[anion_symbol])
    central = compute.anion_central_dict[anion_symbol]

    cat_idx = [int(g[0]) for g in cat_groups]
    an_idx = []
    for g in anion_groups:
        g = list(g)
        local = int(np.where(symbols0[g] == central)[0][0])
        an_idx.append(int(g[local]))
    return cat_idx, an_idx


# ══════════════════════════════════════════════════════════════════════════
# Green-Kubo conductivity
# ══════════════════════════════════════════════════════════════════════════
def green_kubo(traj_path, cat_symbol, anion_symbol, dt_fs, T_K,
               eq_cut_ns=2.0, window_ns=5.0):
    """Collective velocity ACF -> L_ij -> running kappa(tau).

    Reads a contiguous window of frames at the native frame spacing (needed to
    resolve the fast VACF decay), forms the per-species summed ion-site
    velocities in A/fs (system COM removed), and returns the diagnostics.

    Returns dict:
      tau_ps      : (T-1,) integration upper-limit / lag axis (ps)
      acf_pp/pm/mm: (T,3)  collective velocity ACF per Cartesian component
      kappa_tau   : (T-1,) running Green-Kubo conductivity (mS/cm)
      V_ang3, n_cat, n_anion, window_ns_used
    """
    traj_path = Path(traj_path)
    dt_ps = dt_fs / 1000.0
    i_start = int(eq_cut_ns * 1e3 / dt_ps)
    n_window = int(window_ns * 1e3 / dt_ps)

    with compute._AseTraj(str(traj_path)) as trj:
        n_total = len(trj)
        symbols0 = np.array(trj[0].get_chemical_symbols())
        masses0 = np.asarray(trj[0].get_masses(), dtype=np.float64)

    i_end = min(n_total, i_start + n_window)
    T = i_end - i_start
    if T < 100:
        raise RuntimeError(f"GK window too short: only {T} frames after eq cut.")

    cat_idx, an_idx = ion_site_indices(symbols0, cat_symbol, anion_symbol)
    Mtot = masses0.sum()
    vel_A_per_fs = ase_u.fs                        # ASE velocity unit -> A/fs

    v_cat = np.zeros((T, 3))
    v_an = np.zeros((T, 3))
    with compute._AseTraj(str(traj_path)) as trj:
        for k, fi in enumerate(tqdm(range(i_start, i_end), desc="GK velocities")):
            try:
                v = trj[fi].get_velocities() * vel_A_per_fs
            except Exception:
                v_cat[k], v_an[k] = v_cat[k - 1], v_an[k - 1]
                continue
            v = v - (masses0[:, None] * v).sum(0) / Mtot   # remove COM drift
            v_cat[k] = v[cat_idx].sum(0)                    # sum over cations
            v_an[k] = v[an_idx].sum(0)                      # sum over anions

    V_ang3 = float(np.prod(compute._average_cell(traj_path, i_start, i_end)))
    times_fs = np.arange(T) * dt_fs

    # ACF via the transport-coefficients submodule (pure-numpy FFT path)
    tc = lij.TransportCoefficients(None, None, V=V_ang3, times=times_fs, T=T_K)
    acf_pp, acf_pm, acf_mm = tc.compute_acf(v_cat, v_an)    # each (T,3)

    # L_ij = 1/(3 kB T V) int acf dt   (re-implemented: cumtrapz was removed)
    def _cum_L(acf):
        pref = _CONVERT_LIJ / (_KB * T_K * V_ang3)
        L = np.stack([cumulative_trapezoid(acf[:, i], times_fs) for i in range(3)], axis=1)
        return pref * L.mean(axis=1)                        # avg over xyz -> the 1/3

    L_pp, L_pm, L_mm = _cum_L(acf_pp), _cum_L(acf_pm), _cum_L(acf_mm)
    kappa_tau = (L_pp + L_mm - 2.0 * L_pm) * 10.0 * _E2C**2   # mS/cm

    return dict(
        tau_ps=times_fs[1:] / 1000.0,
        acf_pp=acf_pp, acf_pm=acf_pm, acf_mm=acf_mm,
        kappa_tau=kappa_tau,
        V_ang3=V_ang3, n_cat=len(cat_idx), n_anion=len(an_idx),
        window_ns_used=T * dt_ps / 1000.0,
    )


def plateau_value(tau_ps, kappa_tau, lo_ps, hi_ps):
    m = (tau_ps >= lo_ps) & (tau_ps <= hi_ps)
    if not m.any():
        return float("nan"), float("nan")
    return float(np.mean(kappa_tau[m])), float(np.std(kappa_tau[m]))


# ══════════════════════════════════════════════════════════════════════════
# figure
# ══════════════════════════════════════════════════════════════════════════
def make_figure(gk, sigma_gk, sigma_gk_std, sigma_csd, plateau, max_lag_ps,
                system, out_png):
    tau = gk["tau_ps"]
    # mean over xyz (the combination that enters conductivity)
    pp = gk["acf_pp"].mean(1)
    pm = gk["acf_pm"].mean(1)
    mm = gk["acf_mm"].mean(1)
    charge = pp + mm - 2.0 * pm                     # <J.J>-like collective ACF
    t_acf = np.arange(len(pp)) * (tau[0] if len(tau) else 0.0)  # ps (== frame*dt)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # ── VACF vs tau ──────────────────────────────────────────────────────────
    axL.plot(t_acf, pp, label="+ +", color="tab:blue")
    axL.plot(t_acf, pm, label="+ -", color="tab:green")
    axL.plot(t_acf, mm, label="- -", color="tab:red")
    axL.plot(t_acf, charge, label="+ + - - - 2 + -  (charge)", color="k", lw=1.6)
    axL.axhline(0, color="grey", lw=0.6)
    axL.set_xlim(0, max_lag_ps)
    axL.set_xlabel(r"lag  $\tau$  (ps)")
    axL.set_ylabel(r"collective velocity ACF  (xyz-mean)  [$\mathrm{\AA^2/fs^2}$]")
    axL.set_title(f"{system}: collective velocity ACF")
    axL.legend(fontsize=8)

    # ── running kappa(tau) ───────────────────────────────────────────────────
    axR.plot(tau, gk["kappa_tau"], color="tab:purple", lw=1.4,
             label="Green-Kubo  $\\kappa(\\tau)$")
    axR.axvspan(plateau[0], plateau[1], color="tab:purple", alpha=0.10,
                label=f"GK plateau {plateau[0]:g}-{plateau[1]:g} ps")
    axR.axhline(sigma_gk, color="tab:purple", ls="--", lw=1.2,
                label=f"GK = {sigma_gk:.3f} $\\pm$ {sigma_gk_std:.3f} mS/cm")
    if sigma_csd is not None and np.isfinite(sigma_csd):
        axR.axhline(sigma_csd, color="tab:orange", ls="-", lw=1.6,
                    label=f"CSD (Einstein) = {sigma_csd:.3f} mS/cm")
    axR.axhline(0, color="grey", lw=0.6)
    axR.set_xlim(0, max_lag_ps)
    axR.set_xlabel(r"integration cutoff  $\tau$  (ps)")
    axR.set_ylabel(r"conductivity  $\kappa$  (mS/cm)")
    axR.set_title(f"{system}: running Green-Kubo conductivity")
    axR.legend(fontsize=8, loc="best")

    if sigma_csd is not None and np.isfinite(sigma_csd) and sigma_csd > 0:
        ratio = sigma_gk / sigma_csd
        note = (f"GK/CSD = {ratio:.1f}\n"
                "GK over-counts the under-resolved\n"
                "ballistic ACF peak (100 fs frames);\n"
                "CSD is the robust value." if ratio > 1.5 else
                f"GK/CSD = {ratio:.2f} (consistent)")
        axR.text(0.97, 0.03, note, transform=axR.transAxes, va="bottom",
                 ha="right", fontsize=7.5,
                 bbox=dict(boxstyle="round,pad=0.3", fc="#fff6e6", ec="grey"))

    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ══════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traj", help=".traj file or run directory containing one")
    ap.add_argument("--cat", default=None, help="cation key (default: guess from path)")
    ap.add_argument("--anion", default=None, help="anion key (default: guess)")
    ap.add_argument("--solvent", default=None, help="solvent key (default: guess)")
    ap.add_argument("--dt_fs", type=float, default=None, help="frame spacing (fs)")
    ap.add_argument("--T", type=float, default=None, help="temperature (K)")
    ap.add_argument("--eq_cut_ns", type=float, default=2.0)
    # Green-Kubo knobs
    ap.add_argument("--gk_window_ns", type=float, default=3.0,
                    help="contiguous native-res window for the VACF (main GK cost)")
    ap.add_argument("--gk_max_lag_ps", type=float, default=10.0,
                    help="x-axis / plot range for VACF and kappa(tau)")
    ap.add_argument("--gk_plateau_ps", type=float, nargs=2, default=(2.0, 5.0),
                    help="tau window (ps) averaged for the reported GK kappa")
    # CSD (mdcraft) knobs
    ap.add_argument("--csd_load_dt_ps", type=float, default=10.0)
    ap.add_argument("--csd_max_traj_ns", type=float, default=20.0)
    ap.add_argument("--csd_fit_ns", type=float, nargs=2, default=(0.3, 2.4))
    ap.add_argument("--no_csd", action="store_true", help="skip the CSD method")
    ap.add_argument("--no_gk", action="store_true", help="skip the Green-Kubo method")
    ap.add_argument("--out_dir", default=None, help="output dir (default: ./gk_vs_csd_<system>)")
    args = ap.parse_args()

    traj_path = find_traj(args.traj)

    g_cat, g_an, g_solv = guess_species(traj_path)
    cat = args.cat or g_cat
    anion = args.anion or g_an
    solvent = args.solvent or g_solv
    if not (cat and anion and solvent):
        raise SystemExit(
            f"Could not determine species (cat={cat}, anion={anion}, solvent={solvent}). "
            "Pass --cat/--anion/--solvent explicitly.")

    g_dt, g_T = guess_dt_T(traj_path)
    dt_fs = args.dt_fs or g_dt
    T_K = args.T or g_T

    system = Path(traj_path).parent.parent.name or Path(traj_path).stem
    out_dir = Path(args.out_dir) if args.out_dir else Path.cwd() / f"gk_vs_csd_{system}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print(f"traj     : {traj_path}")
    print(f"system   : {system}")
    print(f"species  : cation={cat}  anion={anion}  solvent={solvent}")
    print(f"dt/T     : {dt_fs:g} fs   {T_K:g} K")
    print(f"out_dir  : {out_dir}")
    print("=" * 74)

    # ── 1. Cross-square displacement (Einstein / Onsager, mdcraft) ────────────
    sigma_csd = None
    if not args.no_csd:
        print("\n[CSD] cross-square-displacement conductivity (mdcraft) ...")
        csd = compute.run_onsager_conductivity_mdcraft(
            traj_path, cat, anion, solvent, dt_fs=dt_fs, T_K=T_K,
            eq_cut_ns=args.eq_cut_ns, load_dt_ps=args.csd_load_dt_ps,
            max_traj_ns=args.csd_max_traj_ns,
            fit_start_ns=args.csd_fit_ns[0], fit_stop_ns=args.csd_fit_ns[1],
            out_dir=out_dir, system_name=system, model_name="PAINN")
        sigma_csd = csd["sigma_mdcraft_mS_cm"]
        print(f"[CSD] kappa = {sigma_csd:.4f} mS/cm  "
              f"(t_cat={csd['t_cat']:.3f}, D_cat={csd['D_cat_cm2_s']:.2e} cm^2/s)")

    # ── 2. Green-Kubo ─────────────────────────────────────────────────────────
    sigma_gk = sigma_gk_std = None
    gk = None
    if not args.no_gk:
        print(f"\n[GK] Green-Kubo conductivity over {args.gk_window_ns:g} ns "
              f"@ {dt_fs:g} fs ...")
        gk = green_kubo(traj_path, cat, anion, dt_fs=dt_fs, T_K=T_K,
                        eq_cut_ns=args.eq_cut_ns, window_ns=args.gk_window_ns)
        sigma_gk, sigma_gk_std = plateau_value(
            gk["tau_ps"], gk["kappa_tau"], *args.gk_plateau_ps)
        print(f"[GK] kappa = {sigma_gk:.4f} +/- {sigma_gk_std:.4f} mS/cm "
              f"(plateau {args.gk_plateau_ps[0]:g}-{args.gk_plateau_ps[1]:g} ps, "
              f"window used {gk['window_ns_used']:.2f} ns)")

    # ── figure + summary ──────────────────────────────────────────────────────
    if gk is not None:
        png = make_figure(gk, sigma_gk, sigma_gk_std, sigma_csd,
                          args.gk_plateau_ps, args.gk_max_lag_ps, system,
                          out_dir / f"conductivity_gk_vs_csd_{system}.png")
        np.savez(out_dir / f"gk_data_{system}.npz",
                 tau_ps=gk["tau_ps"], kappa_tau=gk["kappa_tau"],
                 acf_pp=gk["acf_pp"], acf_pm=gk["acf_pm"], acf_mm=gk["acf_mm"])
        print(f"\n[plot] saved {png}")

    print("\n" + "=" * 74)
    print(f"{'method':<34}{'kappa (mS/cm)':>18}")
    print("-" * 74)
    if sigma_csd is not None:
        print(f"{'Cross-square displacement (CSD)':<34}{sigma_csd:>18.4f}")
    if sigma_gk is not None:
        print(f"{'Green-Kubo (GK)':<34}{sigma_gk:>18.4f}  +/- {sigma_gk_std:.4f}")
    if sigma_csd and sigma_gk and np.isfinite(sigma_csd) and np.isfinite(sigma_gk):
        ratio = sigma_gk / sigma_csd if sigma_csd else float("nan")
        print("-" * 74)
        print(f"{'GK / CSD ratio':<34}{ratio:>18.3f}")
        print("=" * 74)
        if ratio > 1.5 or ratio < 0.67:
            print(
                "\nInterpretation: the two methods disagree.  Green-Kubo integrates\n"
                f"the velocity ACF, which needs the sub-{dt_fs:g}fs ballistic decay to be\n"
                "time-resolved; with this frame spacing that peak is under-sampled and\n"
                "GK over-counts it (ratio > 1).  The Einstein/CSD value uses long-time\n"
                "displacements and is the robust estimate.  For a converged Green-Kubo\n"
                "number, re-run on a trajectory saved every ~1-5 fs.")
    else:
        print("=" * 74)


if __name__ == "__main__":
    main()
