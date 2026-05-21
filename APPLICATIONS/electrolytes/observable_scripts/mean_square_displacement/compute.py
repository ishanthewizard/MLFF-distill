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

    Returns
    -------
    dict with keys:
      tau_ns, msd_cat, msd_anion, msd_solvent  — raw MSD curves
      fit_mask, fit_slope, fit_intercept        — linear fit on cat MSD
      convergence                               — dict from compute_convergence
      eq_cut_ns, fit_pct                        — echoed config
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
    tau_max_load_ps = max((analysis_ns - eq_cut_ns) * 1000.0, 1.0)

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
