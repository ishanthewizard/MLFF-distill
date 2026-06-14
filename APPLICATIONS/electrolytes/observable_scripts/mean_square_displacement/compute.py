#!/usr/bin/env python3
"""MSD and diffusivity computation.

Imports core trajectory processing from msd_with_com (unchanged).
Adds convergence analysis used for fitting diagnostics.

Client-facing parameters
------------------------
eq_cut_ns       : float  — skip this many ns from start of trajectory
fit_pct         : float  — fraction of max lag to use as tau_max (e.g. 0.8)
tau_min_fit_ns  : float  — minimum lag included in the linear fit
dt_fs           : float  — trajectory frame spacing in femtoseconds
n_sample        : int    — target number of frames after subsampling
n_conv_points   : int    — resolution of the convergence curves
slide_window_ns : float  — fixed window width for Panel 4 (sliding window D)
slide_step_ns   : float  — step size for sliding window
"""

import math
import sys
from pathlib import Path

import numpy as np
from ase.io.trajectory import Trajectory as _AseTraj

# ensure msd_with_com is importable from sibling location
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from msd_with_com import (
    stream_subsample_unwrap,
    msd_time_origin,
    fit_diffusion,
    a2ps_to_m2s,
)

_KB  = 1.380649e-23   # J/K
_XI  = 2.837297        # Yeh-Hummer dimensionless constant

KB_J_PER_K = _KB
YH_XI      = _XI


# ── Yeh-Hummer finite-size correction ────────────────────────────────────────

def yeh_hummer_correction(
    D_PBC_m2s: float,
    T_K: float,
    eta_Pa_s: float,
    L_m: float,
) -> float:
    """Return D0 = D_PBC + kB*T*xi / (6*pi*eta*L)  [m²/s].

    eta must be DYNAMIC viscosity in Pa·s (= kg·m⁻¹·s⁻¹), not kinematic.
    If you only have kinematic viscosity nu [m²/s], convert first:
        eta_Pa_s = nu_m2s * rho_kg_m3
    e.g. DME at 298 K: nu ≈ 5.5e-7 m²/s, rho ≈ 867 kg/m³ → eta ≈ 4.8e-4 Pa·s
    """
    return D_PBC_m2s + (_KB * T_K * _XI) / (6.0 * math.pi * eta_Pa_s * L_m)


def yeh_hummer_correction_a2ps(T_K: float, viscosity_cp: float, box_length_a: float) -> float:
    """Yeh-Hummer finite-size correction to a self-diffusivity, in A^2/ps.

    Periodic boundaries systematically *suppress* self-diffusion via the
    hydrodynamic self-interaction; the leading correction to the infinite-box
    value is

        dD = k_B T xi / (6 pi eta L),   xi = 2.837297 (cubic box).

    It is species-independent (depends only on T, the shear viscosity, and the
    box edge), so the same dD is added to every species' D_PBC to estimate
    D_inf. ``viscosity_cp`` in cP (= mPa.s), ``box_length_a`` in A (use V^(1/3)
    for a non-cubic box), ``T_K`` in K. (Yeh & Hummer, J. Phys. Chem. B 2004.)
    """
    eta_pas = viscosity_cp * 1e-3                 # cP -> Pa.s
    L_m = box_length_a * 1e-10                    # A -> m
    dD_m2s = KB_J_PER_K * T_K * YH_XI / (6.0 * math.pi * eta_pas * L_m)
    return dD_m2s / a2ps_to_m2s(1.0)              # m^2/s -> A^2/ps


def _extract_mean_box_L_ase(
    traj_path: Path,
    eq_cut_ps: float,
    dt_ps: float,
    n_sample: int,
    tau_max_ps: float,
) -> tuple[float, float, float, float, float]:
    """Return time-averaged (Lx, Ly, Lz, V, L=V^1/3) in Angstrom from ASE traj."""
    with _AseTraj(str(traj_path)) as trj:
        n_total = min(len(trj), max(1, int(tau_max_ps / dt_ps)))
        i_start = min(max(int(np.floor(eq_cut_ps / dt_ps)), 0), n_total - 1)
        avail   = n_total - i_start
        stride  = max(1, int(np.ceil(avail / n_sample)))
        idxs    = list(range(i_start, n_total, stride))

        Lx_vals, Ly_vals, Lz_vals = [], [], []
        for i in idxs:
            cell  = trj[i].get_cell().array         # 3×3 in Å
            # for orthorhombic: row norms give Lx, Ly, Lz
            norms = np.linalg.norm(cell, axis=1)
            Lx_vals.append(norms[0])
            Ly_vals.append(norms[1])
            Lz_vals.append(norms[2])

    Lx = float(np.mean(Lx_vals))
    Ly = float(np.mean(Ly_vals))
    Lz = float(np.mean(Lz_vals))
    V  = Lx * Ly * Lz
    L  = V ** (1.0 / 3.0)
    return Lx, Ly, Lz, V, L


def _extract_mean_box_L_gromacs(
    topology_path: Path,
    traj_path: Path,
    eq_cut_ps: float,
    dt_ps: float,
    n_sample: int,
    tau_max_ps: float,
) -> tuple[float, float, float, float, float]:
    """Return time-averaged (Lx, Ly, Lz, V, L=V^1/3) in Angstrom from a GROMACS traj."""
    import MDAnalysis as mda

    u = mda.Universe(str(topology_path), str(traj_path))
    n_total = min(len(u.trajectory), max(1, int(tau_max_ps / dt_ps)))
    i_start = min(max(int(np.floor(eq_cut_ps / dt_ps)), 0), n_total - 1)
    avail   = n_total - i_start
    stride  = max(1, int(np.ceil(avail / n_sample)))
    idxs    = list(range(i_start, n_total, stride))

    Lx_vals, Ly_vals, Lz_vals = [], [], []
    for i in idxs:
        dims = np.asarray(u.trajectory[i].dimensions[:3], dtype=float)
        Lx_vals.append(dims[0])
        Ly_vals.append(dims[1])
        Lz_vals.append(dims[2])

    Lx = float(np.mean(Lx_vals))
    Ly = float(np.mean(Ly_vals))
    Lz = float(np.mean(Lz_vals))
    V  = Lx * Ly * Lz
    L  = V ** (1.0 / 3.0)
    return Lx, Ly, Lz, V, L


# ── helpers ───────────────────────────────────────────────────────────────────

def _D_1e10(tau_ps, msd, tau_min_ps, tau_max_ps):
    """Fit D; return value in 1e-10 m²/s, or nan on failure."""
    if msd is None:
        return np.nan
    try:
        D_a2ps, _, _, _ = fit_diffusion(tau_ps, msd, tau_min_ps, tau_max_ps)
        return float(a2ps_to_m2s(D_a2ps) * 1e10)
    except Exception:
        return np.nan


# ── convergence analysis ──────────────────────────────────────────────────────

def compute_traj_length_convergence(
    tau_ps: np.ndarray,
    pos_cat: np.ndarray,
    pos_anion: np.ndarray | None,
    pos_solvent: np.ndarray | None,
    tau_min_fit_ps: float,
    fit_pct: float,
    n_points: int = 20,
) -> dict:
    """D vs total trajectory length used for MSD computation (t_end sweep).

    For each t_end, recomputes the MSD from scratch using only frames 0..t_end
    (giving more/fewer time origins), then fits D using [tau_min, fit_pct*t_end].
    This is distinct from Panel 2 which varies the fit range on a fixed MSD:
    here we ask "do I have enough trajectory data for stable statistics?"

    Returns arrays indexed by t_end_ns.
    """
    t_min = tau_min_fit_ps * 2  # need at least 2×tau_min lag range to fit
    t_max = float(tau_ps[-1])

    if t_max <= t_min:
        empty = np.array([])
        return {"t_end_ns": empty, "D_cat_tlen": empty,
                "D_ani_tlen": empty, "D_sol_tlen": empty}

    t_ends = np.linspace(t_min, t_max, n_points)
    D_cat_v, D_ani_v, D_sol_v = [], [], []

    for t_end in t_ends:
        n = min(int(np.searchsorted(tau_ps, t_end)) + 1, len(tau_ps))
        tau_sub = tau_ps[:n]
        tau_max_sub = fit_pct * float(tau_sub[-1])

        msd_c = msd_time_origin(pos_cat[:n])
        msd_a = msd_time_origin(pos_anion[:n])   if pos_anion   is not None else None
        msd_s = msd_time_origin(pos_solvent[:n]) if pos_solvent is not None else None

        D_cat_v.append(_D_1e10(tau_sub, msd_c, tau_min_fit_ps, tau_max_sub))
        D_ani_v.append(_D_1e10(tau_sub, msd_a, tau_min_fit_ps, tau_max_sub))
        D_sol_v.append(_D_1e10(tau_sub, msd_s, tau_min_fit_ps, tau_max_sub))

    return {
        "t_end_ns":    t_ends / 1000,
        "D_cat_tlen":  np.array(D_cat_v),
        "D_ani_tlen":  np.array(D_ani_v),
        "D_sol_tlen":  np.array(D_sol_v),
    }


def compute_convergence(
    tau_ps: np.ndarray,
    msd_cat: np.ndarray,
    msd_anion: np.ndarray | None,
    msd_solvent: np.ndarray | None,
    tau_min_fit_ps: float,
    tau_max_fit_ps: float,
    n_points: int = 200,
    slide_window_ps: float = 10_000.0,
    slide_step_ps: float = 100.0,
) -> dict:
    """Compute all convergence data needed for the 4-panel diagnostic figure.

    Panel 2 — cumulative: tau_max sweeps from 2*tau_min to tau_max_fit
    Panel 3 — delta D:    finite difference of Panel 2 curves
    Panel 4 — sliding:    fixed-width window slides from tau_min to tau_max_fit

    Returns a flat dict (all arrays in ns / 1e-10 m²/s units).
    """
    # ── Panel 2: D vs tau_max (grow fit window from left) ─────────────────────
    sweep = np.linspace(tau_min_fit_ps * 2, tau_max_fit_ps, n_points)
    D_cat_c  = np.array([_D_1e10(tau_ps, msd_cat,     tau_min_fit_ps, t) for t in sweep])
    D_ani_c  = np.array([_D_1e10(tau_ps, msd_anion,   tau_min_fit_ps, t) for t in sweep])
    D_sol_c  = np.array([_D_1e10(tau_ps, msd_solvent, tau_min_fit_ps, t) for t in sweep])

    # ── Panel 3: delta D ──────────────────────────────────────────────────────
    delta_cat = np.diff(D_cat_c, prepend=D_cat_c[0])
    delta_ani = np.diff(D_ani_c, prepend=D_ani_c[0])
    delta_sol = np.diff(D_sol_c, prepend=D_sol_c[0])

    # ── Panel 4: sliding window ───────────────────────────────────────────────
    ends, D_cat_s, D_ani_s, D_sol_s = [], [], [], []
    t = tau_min_fit_ps
    while t + slide_window_ps <= tau_max_fit_ps:
        w_end = t + slide_window_ps
        ends.append(w_end)
        D_cat_s.append(_D_1e10(tau_ps, msd_cat,     t, w_end))
        D_ani_s.append(_D_1e10(tau_ps, msd_anion,   t, w_end))
        D_sol_s.append(_D_1e10(tau_ps, msd_solvent, t, w_end))
        t += slide_step_ps

    return {
        # cumulative (Panel 2)
        "tau_max_sweep_ns": sweep / 1000,
        "D_cat_cumul":  D_cat_c,
        "D_ani_cumul":  D_ani_c,
        "D_sol_cumul":  D_sol_c,
        # delta D (Panel 3)
        "delta_D_cat": delta_cat,
        "delta_D_ani": delta_ani,
        "delta_D_sol": delta_sol,
        # sliding window (Panel 4)
        "slide_end_ns":  np.array(ends) / 1000,
        "D_cat_slide":   np.array(D_cat_s),
        "D_ani_slide":   np.array(D_ani_s),
        "D_sol_slide":   np.array(D_sol_s),
        "slide_window_ns": slide_window_ps / 1000,
        # chosen fit bounds
        "tau_min_fit_ns": tau_min_fit_ps / 1000,
        "tau_max_fit_ns": tau_max_fit_ps / 1000,
        # final D at chosen tau_max
        "D_cat_final":  _D_1e10(tau_ps, msd_cat,     tau_min_fit_ps, tau_max_fit_ps),
        "D_ani_final":  _D_1e10(tau_ps, msd_anion,   tau_min_fit_ps, tau_max_fit_ps),
        "D_sol_final":  _D_1e10(tau_ps, msd_solvent, tau_min_fit_ps, tau_max_fit_ps),
    }


# ── full pipeline ─────────────────────────────────────────────────────────────

def run_msd_analysis(
    traj_path: str | Path,
    cat_symbol: str,
    anion_symbol: str,
    solvent_symbol: str,
    dt_fs: float,
    topology_path: str | Path | None = None,
    eq_cut_ns: float = 0.0,
    fit_pct: float = 0.8,
    tau_min_fit_ns: float = 1.0,
    n_sample: int = 2000,
    n_conv_points: int = 200,
    slide_window_ns: float = 10.0,
    slide_step_ns: float = 0.1,
    max_traj_ns: float | None = None,
    yh_T: float | None = None,
    yh_eta: float | None = None,
) -> dict:
    """Run the full MSD + diffusivity pipeline for one trajectory.

    Parameters
    ----------
    traj_path       : path to .traj file
    cat_symbol      : key in cation_dict   (e.g. 'Na', 'Li')
    anion_symbol    : key in anion_dict    (e.g. 'PF6', 'OTf')
    solvent_symbol  : key in solvent_dict  (e.g. 'DME', 'PC')
    dt_fs           : frame spacing in femtoseconds
    eq_cut_ns       : equilibration cut from trajectory start (ns)
    fit_pct         : upper bound of fit = fit_pct * max_available_lag
    tau_min_fit_ns  : lower bound of linear fit (ns) — skips ballistic/cage regime
    n_sample        : target frames after subsampling
    n_conv_points   : resolution of convergence sweep curves
    slide_window_ns : width of fixed window in Panel 4
    slide_step_ns   : step of sliding window in Panel 4
    yh_T            : temperature in K for Yeh-Hummer correction (None = skip)
    yh_eta          : DYNAMIC viscosity in Pa·s for Yeh-Hummer correction
                      (not kinematic; if you have kinematic nu [m²/s]:
                       eta = nu * rho_kg_m3)

    Returns
    -------
    dict with keys:
      tau_ns, msd_cat, msd_anion, msd_solvent  — raw MSD curves
      fit_mask, fit_slope, fit_intercept        — linear fit on cat MSD
      convergence                               — dict from compute_convergence
      eq_cut_ns, fit_pct                        — echoed config
      yh_correction   — dict with corrected D values (only if yh_T and yh_eta provided)
    """
    traj_path = Path(traj_path)
    dt_ps = dt_fs / 1000.0
    eq_cut_ps = eq_cut_ns * 1000.0
    tau_min_fit_ps = tau_min_fit_ns * 1000.0
    is_gromacs = traj_path.suffix.lower() in (".xtc", ".trr")

    # determine total usable duration
    if is_gromacs:
        from gromacs_io import n_frames_gromacs
        topo = Path(topology_path) if topology_path else traj_path.with_suffix(".tpr")
        n_raw = n_frames_gromacs(topo, traj_path)
    else:
        with _AseTraj(str(traj_path)) as _t:
            n_raw = len(_t)
    actual_ns = n_raw * dt_fs * 1e-6
    analysis_ns = min(actual_ns, max_traj_ns) if max_traj_ns is not None else actual_ns
    # Pass the total analysis duration so stream_subsample_unwrap loads enough
    # frames from the start; eq_cut is applied internally via start_ps.
    tau_max_load_ps = max(analysis_ns * 1000.0, 1.0)

    # stream + unwrap
    if is_gromacs:
        sys.path.insert(0, str(_HERE))
        from msds_calculation_batch_gromacs import stream_subsample_unwrap_gromacs
        tau_ps, pos_cat, pos_anion, pos_solvent, stride, T, _ = stream_subsample_unwrap_gromacs(
            topo, traj_path, eq_cut_ps, dt_ps, n_sample,
            cat_symbol, anion_symbol, solvent_symbol,
            tau_max_load_ps,
        )
    else:
        tau_ps, pos_cat, pos_anion, pos_solvent, stride, T, _ = stream_subsample_unwrap(
            traj_path, eq_cut_ps, dt_ps, n_sample,
            cat_symbol, anion_symbol, solvent_symbol,
            tau_max_load_ps,
        )

    # MSD
    msd_cat     = msd_time_origin(pos_cat)
    msd_anion   = msd_time_origin(pos_anion)   if pos_anion   is not None else None
    msd_solvent = msd_time_origin(pos_solvent) if pos_solvent is not None else None

    # fit bounds
    tau_max_fit_ps = fit_pct * float(tau_ps[-1])

    # linear fit for all three species
    D_a2ps, slope_cat, intercept_cat, fit_mask = fit_diffusion(
        tau_ps, msd_cat, tau_min_fit_ps, tau_max_fit_ps
    )

    if msd_anion is not None:
        try:
            _, slope_ani, intercept_ani, _ = fit_diffusion(
                tau_ps, msd_anion, tau_min_fit_ps, tau_max_fit_ps)
        except Exception:
            slope_ani, intercept_ani = None, None
    else:
        slope_ani, intercept_ani = None, None

    if msd_solvent is not None:
        try:
            _, slope_sol, intercept_sol, _ = fit_diffusion(
                tau_ps, msd_solvent, tau_min_fit_ps, tau_max_fit_ps)
        except Exception:
            slope_sol, intercept_sol = None, None
    else:
        slope_sol, intercept_sol = None, None

    # convergence diagnostics
    conv = compute_convergence(
        tau_ps, msd_cat, msd_anion, msd_solvent,
        tau_min_fit_ps, tau_max_fit_ps,
        n_points=n_conv_points,
        slide_window_ps=slide_window_ns * 1000.0,
        slide_step_ps=slide_step_ns * 1000.0,
    )

    # trajectory-length convergence: recompute MSD at each t_end
    tlen_conv = compute_traj_length_convergence(
        tau_ps, pos_cat, pos_anion, pos_solvent,
        tau_min_fit_ps, fit_pct,
        n_points=max(10, n_conv_points // 10),
    )
    conv.update(tlen_conv)

    # ── mean box length (used for Yeh-Hummer correction) ─────────────────────
    L_A = None
    try:
        if is_gromacs:
            _, _, _, _, L_A = _extract_mean_box_L_gromacs(
                topo, traj_path, eq_cut_ps, dt_ps, n_sample, tau_max_load_ps
            )
        else:
            _, _, _, _, L_A = _extract_mean_box_L_ase(
                traj_path, eq_cut_ps, dt_ps, n_sample, tau_max_load_ps
            )
    except Exception:
        L_A = None

    # ── Yeh-Hummer finite-size correction ────────────────────────────────────
    yh_result = None
    if yh_T is not None and yh_eta is not None:
        if is_gromacs:
            raise NotImplementedError(
                "Yeh-Hummer box extraction is not yet implemented for GROMACS trajectories. "
                "Pass an ASE .traj file or implement _extract_mean_box_L for GROMACS."
            )
        Lx_A, Ly_A, Lz_A, V_A3, L_A = _extract_mean_box_L_ase(
            traj_path, eq_cut_ps, dt_ps, n_sample, tau_max_load_ps
        )
        L_m = L_A * 1e-10   # Å → m

        print(f"\n── Yeh-Hummer box dimensions ──────────────────────────────────")
        print(f"  Lx = {Lx_A:.4f} Å   Ly = {Ly_A:.4f} Å   Lz = {Lz_A:.4f} Å")
        print(f"  V  = {V_A3:.2f} Å³   L  = V^(1/3) = {L_A:.4f} Å = {L_m:.4e} m")
        print(f"  T  = {yh_T} K   eta = {yh_eta} Pa·s")

        correction_m2s = (_KB * yh_T * _XI) / (6.0 * math.pi * yh_eta * L_m)

        def _yh(D_1e10):
            if D_1e10 is None or np.isnan(D_1e10):
                return np.nan
            D_PBC = D_1e10 * 1e-10
            D0    = D_PBC + correction_m2s
            return D0

        D_cat_PBC = conv["D_cat_final"] * 1e-10
        D_ani_PBC = conv["D_ani_final"] * 1e-10 if not np.isnan(conv["D_ani_final"]) else np.nan
        D_sol_PBC = conv["D_sol_final"] * 1e-10 if not np.isnan(conv["D_sol_final"]) else np.nan

        D_cat_0 = _yh(conv["D_cat_final"])
        D_ani_0 = _yh(conv["D_ani_final"])
        D_sol_0 = _yh(conv["D_sol_final"])

        print(f"\n── Yeh-Hummer correction: {correction_m2s:.4e} m²/s  ({correction_m2s*1e4:.4e} cm²/s) ──")
        for label, D_PBC, D0 in [
            (cat_symbol, D_cat_PBC, D_cat_0),
            (anion_symbol, D_ani_PBC, D_ani_0),
            (solvent_symbol, D_sol_PBC, D_sol_0),
        ]:
            if np.isnan(D_PBC):
                continue
            print(f"  {label}:")
            print(f"    D_PBC = {D_PBC:.4e} m²/s  ({D_PBC*1e4:.4e} cm²/s)")
            print(f"    D0    = {D0:.4e} m²/s  ({D0*1e4:.4e} cm²/s)")
        print()

        yh_result = {
            "T_K":               yh_T,
            "eta_Pa_s":          yh_eta,
            "Lx_A":              Lx_A,
            "Ly_A":              Ly_A,
            "Lz_A":              Lz_A,
            "V_A3":              V_A3,
            "L_A":               L_A,
            "correction_m2s":    correction_m2s,
            "D_cat_PBC_m2s":     D_cat_PBC,
            "D_cat_0_m2s":       D_cat_0,
            "D_ani_PBC_m2s":     D_ani_PBC,
            "D_ani_0_m2s":       D_ani_0,
            "D_sol_PBC_m2s":     D_sol_PBC,
            "D_sol_0_m2s":       D_sol_0,
            # convenience: same values in 1e-10 m²/s and cm²/s
            "D_cat_PBC_1e10":    D_cat_PBC / 1e-10,
            "D_cat_0_1e10":      D_cat_0   / 1e-10,
            "D_ani_PBC_1e10":    D_ani_PBC / 1e-10,
            "D_ani_0_1e10":      D_ani_0   / 1e-10,
            "D_sol_PBC_1e10":    D_sol_PBC / 1e-10,
            "D_sol_0_1e10":      D_sol_0   / 1e-10,
            "D_cat_PBC_cm2s":    D_cat_PBC * 1e4,
            "D_cat_0_cm2s":      D_cat_0   * 1e4,
            "D_ani_PBC_cm2s":    D_ani_PBC * 1e4,
            "D_ani_0_cm2s":      D_ani_0   * 1e4,
            "D_sol_PBC_cm2s":    D_sol_PBC * 1e4,
            "D_sol_0_cm2s":      D_sol_0   * 1e4,
        }

    return {
        "tau_ns":         tau_ps / 1000,
        "msd_cat":        msd_cat,
        "msd_anion":      msd_anion,
        "msd_solvent":    msd_solvent,
        "fit_mask":        fit_mask,
        "fit_slope_cat":   slope_cat,
        "fit_intercept_cat": intercept_cat,
        "fit_slope_ani":   slope_ani,
        "fit_intercept_ani": intercept_ani,
        "fit_slope_sol":   slope_sol,
        "fit_intercept_sol": intercept_sol,
        "convergence":    conv,
        "eq_cut_ns":      eq_cut_ns,
        "fit_pct":        fit_pct,
        "cat_symbol":     cat_symbol,
        "anion_symbol":   anion_symbol,
        "solvent_symbol": solvent_symbol,
        "yh_correction":  yh_result,
        "L_A":            L_A,
    }


# ── I/O helpers ───────────────────────────────────────────────────────────────

def save_msd_pickle(result: dict, slug: str, output_dir: Path) -> Path:
    """Save raw MSD arrays to a pickle (mirrors original msd_with_com output).

    Pickle contains: tau_ns, msd_cat, msd_anion, msd_solvent, eq_cut_ns, fit_pct.
    Returns path to saved file.
    """
    import pickle
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "tau_ns":      result["tau_ns"],
        "msd_cat":     result["msd_cat"],
        "msd_anion":   result["msd_anion"],
        "msd_solvent": result["msd_solvent"],
        "eq_cut_ns":   result["eq_cut_ns"],
        "fit_pct":     result["fit_pct"],
    }
    out = output_dir / f"msd_{slug}.pkl"
    with open(out, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def save_diffusivity_csv(
    rows: list[dict],
    output_dir: Path,
    filename: str = "diffusivity.csv",
) -> Path:
    """Append D results to a CSV file.

    Each row dict should contain at minimum:
      system, model, D_cat, D_anion, D_solvent (all in 1e-10 m²/s),
      tau_min_fit_ns, tau_max_fit_ns, eq_cut_ns, fit_pct.
    Returns path to saved file.
    """
    import pandas as pd
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / filename
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return out


# default location of the curated experimental diffusivity table
DEFAULT_EXP_DIFFUSIVITY_CSV = (
    "/global/cfs/cdirs/m5024/distillation_project/experiment_data/"
    "cleaned_version/diffusivity.csv"
)

# trajectory-side solvent_dict keys -> experimental-table solvent names
# (the two tables use different abbreviations for the same molecule)
EXP_SOLVENT_NAME = {
    "Diglyme": "DEGDME",
    "TGDME":   "TEGDME",
}


def _match_exp_row(exp_df, cat_symbol, anion_symbol, solvent_symbol,
                    concentration_M=None, temperature_K=None):
    """Return the best-matching row (pandas Series) from the experimental
    diffusivity table, or None if no row matches the salt/solvent.

    Matches on cation/anion/solvent symbols exactly (treating empty cells as
    "no salt"), then picks the closest concentration / temperature among the
    remaining candidates.
    """
    import pandas as pd

    def _eq(col, val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return exp_df[col].isna()
        return exp_df[col] == val

    df = exp_df[
        _eq("cation", cat_symbol)
        & _eq("anion", anion_symbol)
        & _eq("solvent", solvent_symbol)
    ]
    if df.empty:
        return None

    sort_cols = []
    if concentration_M is not None and "concentration (M)" in df.columns:
        df = df.assign(_dconc=(df["concentration (M)"] - concentration_M).abs())
        sort_cols.append("_dconc")
    if temperature_K is not None and "temperature (K)" in df.columns:
        df = df.assign(_dT=(df["temperature (K)"] - temperature_K).abs())
        sort_cols.append("_dT")
    if sort_cols:
        df = df.sort_values(sort_cols)

    return df.iloc[0]


def save_diffusivity_with_exp_csv(
    rows: list[dict],
    output_dir: Path,
    exp_csv_path: str | Path = DEFAULT_EXP_DIFFUSIVITY_CSV,
    filename: str = "diffusivity_with_exp.csv",
) -> Path:
    """Combine simulated diffusivities with the experimental reference table.

    Each row dict should contain (in addition to the fields used by
    ``save_diffusivity_csv``):
      cat_symbol, anion_symbol, solvent_symbol,
      D_cat_1e-10_m2s, D_ani_1e-10_m2s, D_sol_1e-10_m2s,
      L_A (mean box edge in Angstrom, or None),
      concentration_M, temperature_K (optional, used to pick the matching
      experimental row and as fallback T for the Yeh-Hummer correction).

    For each row, the matching experimental entry (same cation/anion/solvent,
    closest concentration/temperature) is looked up and its D values and
    dynamic viscosity are added. The Yeh-Hummer finite-size correction
    dD = kB T xi / (6 pi eta L) is computed from the experimental viscosity
    and the simulated box size, and added to each species' D_PBC to give a
    finite-size-corrected simulated D.

    Returns path to the saved CSV.
    """
    import pandas as pd
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_df = pd.read_csv(exp_csv_path)

    out_rows = []
    for row in rows:
        row = dict(row)
        cat_symbol     = row.get("cat_symbol")
        anion_symbol   = row.get("anion_symbol")
        solvent_symbol = row.get("solvent_symbol")
        exp_solvent_symbol = row.get(
            "exp_solvent_symbol", EXP_SOLVENT_NAME.get(solvent_symbol, solvent_symbol)
        )
        concentration_M = row.get("concentration_M")
        temperature_K   = row.get("temperature_K")

        exp_row = _match_exp_row(
            exp_df, cat_symbol, anion_symbol, exp_solvent_symbol,
            concentration_M, temperature_K,
        )

        exp_eta_mPa_s = np.nan
        exp_T_K       = temperature_K
        exp_D_cat     = np.nan
        exp_D_anion   = np.nan
        exp_D_solvent = np.nan

        if exp_row is not None:
            exp_eta_mPa_s = exp_row.get("Dynamic Viscosity (mPa*s)", np.nan)
            exp_T_K       = exp_row.get("temperature (K)", temperature_K)
            exp_D_cat     = exp_row.get("D_cation (x 10^-10 m^2/s)", np.nan)
            exp_D_anion   = exp_row.get("D_anion (x 10^-10 m^2/s)", np.nan)
            exp_D_solvent = exp_row.get("D_solvent (x 10^-10 m^2/s)", np.nan)

            # the matched D row may not carry a viscosity value (e.g. it was
            # entered separately from a viscosity-only measurement) — fall
            # back to the closest row (same salt/solvent/conc) that has one
            if pd.isna(exp_eta_mPa_s):
                visc_df = exp_df[exp_df["Dynamic Viscosity (mPa*s)"].notna()]
                visc_row = _match_exp_row(
                    visc_df, cat_symbol, anion_symbol, exp_solvent_symbol,
                    concentration_M, temperature_K,
                )
                if visc_row is not None:
                    exp_eta_mPa_s = visc_row.get("Dynamic Viscosity (mPa*s)", np.nan)

        L_A = row.get("L_A")
        dD_1e10 = np.nan
        if L_A is not None and not np.isnan(exp_eta_mPa_s) and exp_T_K is not None:
            dD_a2ps = yeh_hummer_correction_a2ps(exp_T_K, exp_eta_mPa_s, L_A)
            dD_1e10 = a2ps_to_m2s(dD_a2ps) * 1e10

        D_cat_pbc = row.get("D_cat_1e-10_m2s", np.nan)
        D_ani_pbc = row.get("D_ani_1e-10_m2s", np.nan)
        D_sol_pbc = row.get("D_sol_1e-10_m2s", np.nan)

        row.update({
            "L_A":                  L_A,
            "exp_T_K":              exp_T_K,
            "exp_eta_mPa_s":        exp_eta_mPa_s,
            "dD_yh_1e-10_m2s":      dD_1e10,
            "D_cat_corrected_1e-10_m2s": D_cat_pbc + dD_1e10 if not np.isnan(D_cat_pbc) else np.nan,
            "D_ani_corrected_1e-10_m2s": D_ani_pbc + dD_1e10 if not np.isnan(D_ani_pbc) else np.nan,
            "D_sol_corrected_1e-10_m2s": D_sol_pbc + dD_1e10 if not np.isnan(D_sol_pbc) else np.nan,
            "exp_D_cation_1e-10_m2s":   exp_D_cat,
            "exp_D_anion_1e-10_m2s":    exp_D_anion,
            "exp_D_solvent_1e-10_m2s":  exp_D_solvent,
        })
        out_rows.append(row)

    out = output_dir / filename
    df = pd.DataFrame(out_rows)
    df.to_csv(out, index=False)
    return out


def save_yeh_hummer_csv(
    rows: list[dict],
    output_dir: Path,
    filename: str = "diffusivity_yeh_hummer.csv",
) -> Path:
    """Save Yeh-Hummer corrected diffusivity results to a dedicated CSV.

    Each row dict is built from the 'yh_correction' sub-dict returned by
    run_msd_analysis, merged with system/model/species metadata.

    Columns (per row):
      system, model, cat_symbol, anion_symbol, solvent_symbol,
      T_K, eta_Pa_s, Lx_A, Ly_A, Lz_A, V_A3, L_A, correction_m2s,
      D_cat_PBC_m2s, D_cat_0_m2s, D_cat_PBC_1e10, D_cat_0_1e10, D_cat_PBC_cm2s, D_cat_0_cm2s,
      D_ani_PBC_m2s, D_ani_0_m2s, D_ani_PBC_1e10, D_ani_0_1e10, D_ani_PBC_cm2s, D_ani_0_cm2s,
      D_sol_PBC_m2s, D_sol_0_m2s, D_sol_PBC_1e10, D_sol_0_1e10, D_sol_PBC_cm2s, D_sol_0_cm2s,

    Returns path to saved file.
    """
    import pandas as pd
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / filename
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return out
