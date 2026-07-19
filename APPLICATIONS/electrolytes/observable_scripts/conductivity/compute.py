#!/usr/bin/env python3
"""Onsager / Nernst-Einstein ionic conductivity from MD trajectories.

Two trajectory backends are supported:

  - ASE ``.traj`` (e.g. PAINN/FENNIX simulations): ``run_onsager_conductivity``
    streams + unwraps a (possibly very long) trajectory, subsampling to an
    effective ``load_dt_ps`` and correcting for the actual frame spacing.

  - GROMACS ``.tpr``/``.xtc`` (e.g. OPLS production runs): species/masses/
    charges are parsed from the ``.top``/``.itp`` files and the full
    trajectory is loaded with MDAnalysis (unwrap + NoJump), mirroring
    ``conductivity/calc_all_conductivity.py``. Native GROMACS frame spacing
    here is 1 ps, matching the dt assumed internally by ``onsager_calc``, so
    no dt correction is needed.

Both backends call ``byteff2.md_utils.onsager_conductivity.onsager_calc``.

``run_conductivity_analysis`` is a thin wrapper around
``run_onsager_conductivity`` with the return-dict shape expected by
``observable_scripts/eval.py``.

For a batch driver that walks a directory of production systems, computes
conductivity + density per system, merges with experimental data, and
writes a ``conductivity_parity.csv``, see ``build_parity_csv.py``.
"""

import sys
import os
import glob
import logging
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io.trajectory import Trajectory as _AseTraj
from ase.geometry import find_mic
from tqdm import tqdm
# ── byteff2 ──────────────────────────────────────────────────────────────────
_BYTEFF2_DIR = Path(__file__).resolve().parents[4] / "submodule" / "byteff2"
if str(_BYTEFF2_DIR) not in sys.path:
    sys.path.insert(0, str(_BYTEFF2_DIR))

from byteff2.md_utils.onsager_conductivity import onsager_calc

# ── species dictionaries (mirror msd_with_com component_dictionary) ──────────
_MSD_DIR = Path(__file__).resolve().parent.parent / "mean_square_displacement"
if str(_MSD_DIR) not in sys.path:
    sys.path.insert(0, str(_MSD_DIR))

from utils.component_dictionary import cation_dict, anion_dict, solvent_dict
from msd_with_com import direct_groups_from_species

# ── density ────────────────────────────────────────────────────────────────
def _load_density_compute_density():
    import importlib.util
    density_compute_path = Path(__file__).resolve().parent.parent / "density" / "compute.py"
    spec = importlib.util.spec_from_file_location("_density_compute", density_compute_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_density


compute_density = _load_density_compute_density()

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# ASE-trajectory backend
# ══════════════════════════════════════════════════════════════════════════

def _load_unwrapped(traj_path, reorder_idx, stride, i_start, i_end):
    """Stream trajectory frames, unwrap PBC, return (n_frames, n_atoms, 3) in Angstrom."""
    frame_idxs = list(range(i_start, i_end, stride))
    T = len(frame_idxs)
    N = len(reorder_idx)
    positions = np.zeros((T, N, 3), dtype=np.float64)

    with _AseTraj(str(traj_path)) as trj:
        f_prev = trj[frame_idxs[0]]
        positions[0] = f_prev.get_positions()[reorder_idx]
        for k in tqdm(range(1, T)):
            try:
                f_curr = trj[frame_idxs[k]]
                f_prev = trj[frame_idxs[k - 1]]
            except Exception as exc:
                logger.warning(f"Frame read error at k={k}: {exc}")
                positions[k] = positions[k - 1]
                continue
            curr = f_curr.get_positions()[reorder_idx]
            prev = f_prev.get_positions()[reorder_idx]
            disp, _ = find_mic(curr - prev, f_curr.get_cell(), pbc=f_curr.get_pbc())
            positions[k] = positions[k - 1] + disp

    return positions


def _average_volume(traj_path, i_start, n_total, n_sample=200):
    """Return mean box volume (Angstrom^3) sampled over the usable trajectory."""
    stride = 1
    vols = []
    i_start = max(0, n_total - n_sample)
    with _AseTraj(str(traj_path)) as trj:
        for i in range(i_start, n_total, stride):
            try:
                vols.append(float(trj[i].get_volume()))
            except Exception:
                continue
    return float(np.mean(vols)) if vols else 0.0

def _average_cell(traj_path, i_start, n_total, n_sample=200):
    """Return mean cell length (Angstrom) sampled over the usable trajectory."""
    stride = 1
    cells = []
    i_start = max(0, n_total - n_sample)
    with _AseTraj(str(traj_path)) as trj:
        for i in range(i_start, n_total, stride):
            try:
                cells.append(trj[i].cell.array)
            except Exception:
                continue
    return np.diagonal(np.mean(cells, axis=0)) if cells else np.zeros(3)

def run_onsager_conductivity(
    traj_path,
    cat_symbol: str,
    anion_symbol: str,
    solvent_symbol: str,
    dt_fs: float,
    T_K: float,
    z_cat: float = 1.0,
    z_anion: float = -1.0,
    viscosity_cP: float = 1.0,
    eq_cut_ns: float = 2.0,
    load_dt_ps: float = 10.0,
    nt_start: int = 50,
    nt_end: int = 200,
    max_traj_ns: float = 20.0,
) -> dict:
    """Compute Onsager ionic conductivity for one ASE trajectory.

    Parameters
    ----------
    traj_path      : path to .traj file (read-only)
    cat_symbol     : cation key in cation_dict   (e.g. 'Na', 'Li')
    anion_symbol   : anion key in anion_dict     (e.g. 'PF6', 'OTf')
    solvent_symbol : solvent key in solvent_dict (e.g. 'DME', 'PC')
    dt_fs          : frame timestep in femtoseconds
    T_K            : simulation temperature in Kelvin
    z_cat          : formal cation charge (default +1)
    z_anion        : formal anion charge  (default -1)
    viscosity_cP   : solvent viscosity for YH finite-size correction (cP).
                     Only affects Dself_inf; does NOT affect sigma_onsager.
                     Use 1.0 as a placeholder if unknown.
    eq_cut_ns      : equilibration skip from start of trajectory (ns)
    load_dt_ps     : effective time step after subsampling (ps).
                     Determines which frames are loaded and how many.
                     The byteff2 fit range is [nt_start*load_dt_ps,
                     nt_end*load_dt_ps] ps.  Default 10 ps gives a
                     fit range of 500–2000 ps (0.5–2 ns).
    nt_start       : first frame lag used for MSD slope fit (in loaded frames)
    nt_end         : last  frame lag used for MSD slope fit (in loaded frames)
    max_traj_ns    : maximum trajectory length to use (ns)

    Returns
    -------
    dict with keys
      sigma_onsager_mS_cm   : Onsager (Green-Kubo) conductivity (mS/cm)
      sigma_NE_mS_cm        : Nernst-Einstein conductivity (mS/cm)
      Dself_1e10_m2s        : list[float] – self-diffusivity per species (10^-10 m²/s)
      species_order         : list[str]   – species names matching Dself
      N_cat, N_anion        : ion counts
      V_angstrom3           : mean box volume (Angstrom^3)
      T_K                   : temperature used (K)
      fit_lag_start_ns      : actual MSD fit start lag (ns)
      fit_lag_end_ns        : actual MSD fit end lag (ns)
      eq_cut_ns             : equilibration cut used (ns)
      load_dt_ps            : effective dt per loaded frame (ps)
    """
    traj_path = Path(traj_path)
    dt_ps     = dt_fs / 1000.0
    stride    = max(1, round(load_dt_ps / dt_ps))     # frames to skip between loads
    actual_load_dt_ps = stride * dt_ps                 # exact effective dt

    # ── identify usable frame range ───────────────────────────────────────────
    with _AseTraj(str(traj_path)) as trj:
        n_total_raw = len(trj)
        f0 = trj[0]
        symbols0 = f0.get_chemical_symbols()
        masses0  = f0.get_masses()

    n_max = int(max_traj_ns * 1e3 / dt_ps)
    n_total = min(n_total_raw, n_max)
    i_start = int(eq_cut_ns * 1e3 / dt_ps)
    if i_start >= n_total:
        raise RuntimeError("Equilibration cut longer than trajectory.")

    usable_ns = (n_total - i_start) * dt_ps / 1000.0
    n_loaded  = (n_total - i_start) // stride

    fit_lag_start_ns = nt_start * actual_load_dt_ps / 1000.0
    fit_lag_end_ns   = nt_end   * actual_load_dt_ps / 1000.0

    # need at least nt_end + 200 (onsager_calc drops the first 200 loaded
    # frames as additional equilibration before fitting lags nt_start..nt_end)
    if n_loaded <= nt_end + 200:
        raise RuntimeError(
            f"Not enough frames after subsampling: n_loaded={n_loaded}, "
            f"need > {nt_end + 200} (usable={usable_ns:.2f} ns @ "
            f"{actual_load_dt_ps:.2f} ps/frame). Reduce load_dt_ps or eq_cut_ns."
        )

    logger.info(
        f"{traj_path.name}: usable={usable_ns:.1f} ns, "
        f"loaded={n_loaded} frames @ {actual_load_dt_ps:.1f} ps/frame, "
        f"fit=[{fit_lag_start_ns*1000:.0f}, {fit_lag_end_ns*1000:.0f}] ps"
    )

    # ── build species groups and reorder index ────────────────────────────────
    cat_groups     = direct_groups_from_species(symbols0, cation_dict[cat_symbol])
    anion_groups   = direct_groups_from_species(symbols0, anion_dict[anion_symbol])
    solvent_groups = direct_groups_from_species(symbols0, solvent_dict[solvent_symbol])

    total_in_groups = sum(len(g) for g in cat_groups + anion_groups + solvent_groups)
    if total_in_groups != len(symbols0):
        raise RuntimeError(
            f"Atom count mismatch: {total_in_groups} in groups vs {len(symbols0)} total"
        )

    # Reorder: [all cation atoms | all anion atoms | all solvent atoms]
    reorder_idx = (
        [int(i) for g in cat_groups    for i in g] +
        [int(i) for g in anion_groups  for i in g] +
        [int(i) for g in solvent_groups for i in g]
    )

    # ── species metadata for onsager_calc ─────────────────────────────────────
    species_order  = [cat_symbol, anion_symbol, solvent_symbol]
    species_mass   = {
        cat_symbol:     [float(masses0[i]) for i in cat_groups[0]],
        anion_symbol:   [float(masses0[i]) for i in anion_groups[0]],
        solvent_symbol: [float(masses0[i]) for i in solvent_groups[0]],
    }
    species_number = {
        cat_symbol:     len(cat_groups),
        anion_symbol:   len(anion_groups),
        solvent_symbol: len(solvent_groups),
    }
    species_charge = {
        cat_symbol:     float(z_cat),
        anion_symbol:   float(z_anion),
        solvent_symbol: 0.0,
    }

    # ── load & unwrap positions ───────────────────────────────────────────────
    print(f"  Loading {n_loaded} frames @ {actual_load_dt_ps:.1f} ps/frame "
          f"from {traj_path.name} ...")
    i_end = i_start + n_loaded * stride
    positions = _load_unwrapped(traj_path, reorder_idx, stride, i_start, i_end)
    # shape: (n_loaded, n_atoms_reordered, 3)

    # ── average volume ────────────────────────────────────────────────────────
    V_ang3 = _average_volume(traj_path, i_start, n_total, n_sample=200)

    # ── call onsager_calc ─────────────────────────────────────────────────────
    # onsager_calc internally does positions[200:] and fits lags nt_start..nt_end
    # treating each frame as 1 ps.  We correct for actual_load_dt_ps afterwards.
    print(f"  Running onsager_calc for {traj_path.name} ...")
    result = onsager_calc(
        species_order=species_order,
        species_mass=species_mass,
        species_number=species_number,
        species_charge=species_charge,
        volume_angstrom3=V_ang3,
        viscosity_cP=viscosity_cP,
        T_K=T_K,
        positions=positions,
    )

    # ── correct for actual dt (byteff2 assumes 1 ps/frame) ───────────────────
    # The slope of MSD vs frame-index has units Å²/frame.
    # byteff2 converts assuming frame = 1 ps  →  Å²/ps.
    # Actual: frame = actual_load_dt_ps  →  scale factor = 1/actual_load_dt_ps.
    dt_correction = 1.0 / actual_load_dt_ps  # dimensionless (ps^-1 × ps = 1)

    sigma_onsager = result["conductivity_onsager"] * dt_correction
    sigma_NE      = result["conductivity_NE"]      * dt_correction
    Dself         = [d * dt_correction for d in result["Dself_inf"]]

    return {
        "sigma_onsager_mS_cm": sigma_onsager,
        "sigma_NE_mS_cm":      sigma_NE,
        "Dself_1e10_m2s":      Dself,
        "species_order":       species_order,
        "N_cat":               len(cat_groups),
        "N_anion":             len(anion_groups),
        "V_angstrom3":         V_ang3,
        "T_K":                 T_K,
        "fit_lag_start_ns":    fit_lag_start_ns,
        "fit_lag_end_ns":      fit_lag_end_ns,
        "eq_cut_ns":           eq_cut_ns,
        "load_dt_ps":          actual_load_dt_ps,
        "Lambda_onsager":      result["Lambda_onsager"],
        "Lambda_onsager_raw":  result["Lambda_onsager_raw"],
        "Lambda_onsager_unit": result["Lambda_onsager_unit"],
        "species_order":       species_order,
    }


def run_conductivity_analysis(
    traj_path,
    cat_symbol: str,
    anion_symbol: str,
    solvent_symbol: str,
    dt_fs: float,
    T_K: float,
    **kwargs,
) -> dict:
    """Wrapper around ``run_onsager_conductivity`` for ``observable_scripts/eval.py``.

    Returns a flattened dict with the keys eval.py's conductivity block expects:
    sigma_NE_mS_cm, sigma_onsager_mS_cm, D_cat_1e10_m2s, D_anion_1e10_m2s,
    D_solvent_1e10_m2s, N_cat, N_anion, V_angstrom3, tau_min_fit_ns,
    tau_max_fit_ns, eq_cut_ns.
    """
    # Translate / filter kwargs that eval.py sends but run_onsager_conductivity
    # does not accept.
    load_dt_ps = kwargs.pop("load_dt_ps", 10.0)
    tau_min_fit_ns = kwargs.pop("tau_min_fit_ns", None)
    nt_start = int(tau_min_fit_ns * 1000.0 / load_dt_ps) if tau_min_fit_ns is not None \
               else kwargs.pop("nt_start", 50)
    kwargs.pop("fit_pct",   None)   # not used by Onsager; MSD-only param
    kwargs.pop("n_sample",  None)   # not used by Onsager
    result = run_onsager_conductivity(
        traj_path, cat_symbol, anion_symbol, solvent_symbol, dt_fs, T_K,
        load_dt_ps=load_dt_ps, nt_start=nt_start, **kwargs
    )
    Dself = result["Dself_1e10_m2s"]
    return {
        "sigma_NE_mS_cm":      result["sigma_NE_mS_cm"],
        "sigma_onsager_mS_cm": result["sigma_onsager_mS_cm"],
        "D_cat_1e10_m2s":      Dself[0],
        "D_anion_1e10_m2s":    Dself[1],
        "D_solvent_1e10_m2s":  Dself[2],
        "N_cat":               result["N_cat"],
        "N_anion":             result["N_anion"],
        "V_angstrom3":         result["V_angstrom3"],
        "tau_min_fit_ns":      result["fit_lag_start_ns"],
        "tau_max_fit_ns":      result["fit_lag_end_ns"],
        "eq_cut_ns":           result["eq_cut_ns"],
        "Lambda_onsager":      result["Lambda_onsager"],
        "Lambda_onsager_raw":  result["Lambda_onsager_raw"],
        "Lambda_onsager_unit": result["Lambda_onsager_unit"],
        "species_order":       result["species_order"],
    }




# ── mdcraft (collective Onsager) backend ────────────────────────────────────
# Central heavy atom representing each anion's site (look-up by anion key).
anion_central_dict = {
    "PF6":  "P",
    "OTf":  "S",
    "TFSI": "N",
}

# Unit conversions for the mdcraft Onsager results.
_KAPPA_TO_SI = 1.0e19   # mdcraft conductivity unit -> S/m
_SI_TO_USCM  = 1.0e4    # S/m -> uS/cm
_D_TO_CM2_S  = 1.0e-4   # A^2/ps -> cm^2/s
_NDIM        = 3


def _load_conductivity_plot():
    """Import this directory's ``plot.py`` regardless of how compute.py was loaded."""
    plot_py = Path(__file__).resolve().parent / "plot.py"
    spec = importlib.util.spec_from_file_location("conductivity_plot", plot_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _collective_velocity_acf(traj_path, ion_orig_idx, n_cat, i_start, i_end,
                             stride, dt_ps, T_K):
    """Collective velocity ACF (++, +-, --) from stored frame velocities.

    Streams velocities (original atom order), subtracts the mass-weighted
    system-COM velocity (kills Langevin drift), sums per species, and feeds the
    cation / anion collective velocities to the transport-coefficients backend's
    ``compute_acf``.  Diagnostic only — does not enter the conductivity value.

    Returns ``(acf_pp, acf_pm, acf_mm, times_ps)``.
    """
    import ase.units as ase_u

    # transport-coefficients submodule (imported by path, mirroring compute.py)
    lij_py = (_BYTEFF2_DIR.parent / "transport-coefficients" /
              "example_calculation" / "lij_analysis.py")
    spec = importlib.util.spec_from_file_location("lij_analysis", lij_py)
    lij = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lij)
    TransportCoefficients = lij.TransportCoefficients

    frame_idxs = list(range(i_start, i_end, stride))
    T = len(frame_idxs)
    vel_a_per_ps = ase_u.fs * 1e3                    # ASE velocity unit -> A/ps (~98.2)
    v_cation = np.zeros((T, 3))
    v_anion  = np.zeros((T, 3))
    with _AseTraj(str(traj_path)) as trj:
        masses_all = trj[frame_idxs[0]].get_masses()
        Mtot = masses_all.sum()
        for k, fi in enumerate(tqdm(frame_idxs, desc="reading velocities")):
            v = trj[fi].get_velocities() * vel_a_per_ps      # (n_atoms, 3)
            v = v - (masses_all[:, None] * v).sum(0) / Mtot  # barycentric
            vi = v[ion_orig_idx]                             # (n_cat+n_an, 3)
            v_cation[k] = vi[:n_cat].sum(0)                  # sum over cations
            v_anion[k]  = vi[n_cat:].sum(0)                  # sum over anions

    tc = TransportCoefficients(v_cation_filename=None, v_anion_filename=None,
                               V=1.0, times=None, T=T_K)
    acf_pp, acf_pm, acf_mm = tc.compute_acf(v_cation, v_anion)
    times = np.arange(acf_pp.shape[0]) * dt_ps               # ps
    return acf_pp, acf_pm, acf_mm, times


def run_onsager_conductivity_mdcraft(
    traj_path,
    cat_symbol: str,
    anion_symbol: str,
    solvent_symbol: str,
    dt_fs: float,
    T_K: float,
    z_cat: float = 1.0,
    z_anion: float = -1.0,
    viscosity_cP: float = 1.0,
    eq_cut_ns: float = 2.0,
    load_dt_ps: float = 10.0,
    nt_start: int = 50,
    nt_end: int = 200,
    max_traj_ns: float = 20.0,
    fit_start_ns: float = 0.3,
    fit_stop_ns: float = 2.4,
    out_dir=None,
    system_name: str = "system",
    model_name: str = "model",
) -> dict:
    """Collective (Onsager) ionic conductivity for one ASE trajectory via mdcraft.

    Builds an in-memory MDAnalysis universe of the ion sites (every cation plus
    each anion's central heavy atom — looked up in ``anion_central_dict``),
    removes the mass-weighted system-COM drift, and runs
    ``mdcraft.analysis.transport.Onsager`` to obtain the collective conductivity
    ``kappa``.  Two diagnostic figures (collective velocity ACF and cross
    displacement vs lag) are saved through ``conductivity/plot.py``.

    Parameters mirror ``run_onsager_conductivity``; additionally ``fit_start_ns``
    / ``fit_stop_ns`` set the diffusive lag window for the L_ij fit, and
    ``out_dir`` / ``system_name`` / ``model_name`` control where plots are saved.

    Returns
    -------
    dict with keys: ``sigma_mdcraft_mS_cm`` (+ ``_uS_cm``), ``D_cat_cm2_s``,
    ``D_anion_cm2_s``, ``t_cat``, ``t_anion``, ``N_cat``, ``N_anion``,
    ``V_angstrom3``, ``fit_start_ns``, ``fit_stop_ns``, ``load_dt_ps``,
    ``vacf_plot``, ``cross_plot``.
    """
    from mdcraft.analysis.transport import Onsager
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader

    _plot = _load_conductivity_plot()
    out_dir = Path(out_dir) if out_dir is not None else Path.cwd()

    traj_path = Path(traj_path)
    dt_ps  = dt_fs / 1000.0
    stride = max(1, round(load_dt_ps / dt_ps))
    actual_load_dt_ps = stride * dt_ps

    # central heavy atom of each anion (look-up dictionary)
    if anion_symbol not in anion_central_dict:
        raise KeyError(
            f"anion_symbol '{anion_symbol}' not in anion_central_dict "
            f"({list(anion_central_dict)}); add its central heavy atom."
        )
    anion_central = anion_central_dict[anion_symbol]

    cat_symbol_list     = cation_dict[cat_symbol]
    anion_symbol_list   = anion_dict[anion_symbol]
    solvent_symbol_list = solvent_dict[solvent_symbol]

    # ── usable frame range (mirror run_onsager_conductivity) ──────────────────
    with _AseTraj(str(traj_path)) as trj:
        n_total_raw = len(trj)
        symbols0 = np.array(trj[0].get_chemical_symbols())

    n_total = min(n_total_raw, int(max_traj_ns * 1e3 / dt_ps))
    i_start = int(eq_cut_ns * 1e3 / dt_ps)
    if i_start >= n_total:
        raise RuntimeError("Equilibration cut longer than trajectory.")
    n_loaded = (n_total - i_start) // stride
    if n_loaded < 10:
        raise RuntimeError(f"Too few frames after subsampling: n_loaded={n_loaded}.")
    i_end = i_start + n_loaded * stride

    cat_groups     = direct_groups_from_species(symbols0, cat_symbol_list)
    anion_groups   = direct_groups_from_species(symbols0, anion_symbol_list)
    solvent_groups = direct_groups_from_species(symbols0, solvent_symbol_list)

    # Reorder: [all cation atoms | all anion atoms | all solvent atoms]
    reorder_idx = (
        [int(i) for g in cat_groups     for i in g] +
        [int(i) for g in anion_groups   for i in g] +
        [int(i) for g in solvent_groups for i in g]
    )
    reorder_arr = np.asarray(reorder_idx)

    dims = _average_cell(traj_path, i_start, n_total)   # (3,) box lengths (A)

    # ── ion-site columns in the reordered system (cations, then anion centers) ─
    na_cols, an_cols, col = [], [], 0
    for g in cat_groups:                        # monoatomic cation -> group == site
        na_cols.append(col); col += len(g)
    for g in anion_groups:                       # central heavy atom of each anion
        local = int(np.where(symbols0[g] == anion_central)[0][0])
        an_cols.append(col + local); col += len(g)
    n_cat, n_an  = len(na_cols), len(an_cols)
    ion_cols     = na_cols + an_cols
    ion_orig_idx = reorder_arr[ion_cols]         # original-order idx (for velocities)

    # ── load unwrapped positions, slice out the ion sites ─────────────────────
    print(f"  [mdcraft] loading {n_loaded} frames @ {actual_load_dt_ps:.1f} ps/frame "
          f"from {traj_path.name} ...")
    positions = _load_unwrapped(traj_path, reorder_idx, stride, i_start, i_end)
    ion_pos   = positions[:, ion_cols, :]        # (T, n_cat+n_an, 3)

    # ── collective velocity ACF (diagnostic plot) ─────────────────────────────
    vacf_plot = None
    try:
        acf_pp, acf_pm, acf_mm, times_acf = _collective_velocity_acf(
            traj_path, ion_orig_idx, n_cat, i_start, i_end, stride,
            actual_load_dt_ps, T_K)
        vacf_plot = _plot.plot_collective_vacf(
            times_acf, acf_pp, acf_pm, acf_mm, system_name, model_name,
            out_dir, dim=1)
    except Exception as exc:
        logger.warning(f"[mdcraft] VACF plot skipped: {exc}")

    # ── remove system COM drift (barycentric frame) ───────────────────────────
    with _AseTraj(str(traj_path)) as trj:
        masses0 = trj[0].get_masses()
    masses_re = masses0[reorder_idx]
    R = np.einsum("tnj,n->tj", positions, masses_re) / masses_re.sum()
    drift = np.linalg.norm(R - R[0], axis=1)
    print(f"  [mdcraft] system-COM drift: net {drift[-1]:.2f} A  max {drift.max():.2f} A")
    ion_pos_c = ion_pos - R[:, None, :]          # COM-removed ion sites

    # ── in-memory universe of ion sites + Onsager ─────────────────────────────
    n_at = ion_pos_c.shape[1]
    u = mda.Universe.empty(n_at, n_residues=n_at,
                           atom_resindex=np.arange(n_at), trajectory=True)
    u.add_TopologyAttr("name", ["NA"] * n_cat + ["AN"] * n_an)
    u.load_new(ion_pos_c.astype(np.float32), format=MemoryReader)

    ons = Onsager([u.atoms[:n_cat], u.atoms[n_cat:]], groupings="atoms",
                  temperature=T_K, charges=[z_cat, z_anion],
                  dimensions=dims, dt=actual_load_dt_ps, unwrap=False,
                  center=False, fft=True, verbose=False)
    ons.run()

    # ── fit MSDs -> L_ij -> kappa / t_i / mobility ────────────────────────────
    t = ons.results.times                        # lag time (ps)
    dt_lag_ns = (t[1] - t[0]) / 1000.0
    n_lag = len(t)
    s = max(1, int(round(fit_start_ns / dt_lag_ns)))
    e = min(n_lag, int(round(fit_stop_ns / dt_lag_ns)))
    if e <= s:                                    # degenerate window -> fallback
        s, e = max(1, n_lag // 5), n_lag

    ons.calculate_transport_coefficients(start=s, stop=e, scale="linear")
    ons.calculate_conductivity()
    ons.calculate_transference_numbers()
    ons.calculate_electrophoretic_mobilities()

    kappa_uScm = float(ons.results.conductivity[0] * _KAPPA_TO_SI * _SI_TO_USCM)
    kappa_mScm = kappa_uScm / 1.0e3
    D_i = ons.results.D_i[0] * _D_TO_CM2_S        # (2,) cm^2/s
    t_i = ons.results.transference_numbers[0]     # (2,)
    print(f"  [mdcraft] fit {fit_start_ns}-{fit_stop_ns} ns (lag {s}-{e}/{n_lag})  "
          f"kappa={kappa_uScm:.0f} uS/cm ({kappa_mScm:.3f} mS/cm)")

    # ── cross-displacement plot ───────────────────────────────────────────────
    # mdcraft stores msd_cross already divided by (2*ndim); undo to get raw CD.
    cross = ons.results.msd_cross[:, 0] * (2 * _NDIM)   # (3, nt): ++, +-, --
    cross_plot = _plot.plot_cross_displacement(
        t, cross, ons.results.pairs, s, e, kappa_uScm,
        fit_start_ns, fit_stop_ns, system_name, model_name, out_dir)

    return {
        "sigma_mdcraft_mS_cm": kappa_mScm,
        "sigma_mdcraft_uS_cm": kappa_uScm,
        "D_cat_cm2_s":   float(D_i[0]),
        "D_anion_cm2_s": float(D_i[1]),
        "t_cat":   float(t_i[0]),
        "t_anion": float(t_i[1]),
        "N_cat":   n_cat,
        "N_anion": n_an,
        "V_angstrom3":  float(np.prod(dims)),
        "fit_start_ns": fit_start_ns,
        "fit_stop_ns":  fit_stop_ns,
        "load_dt_ps":   actual_load_dt_ps,
        "vacf_plot":  str(vacf_plot) if vacf_plot is not None else None,
        "cross_plot": str(cross_plot),
    }



# ══════════════════════════════════════════════════════════════════════════
# GROMACS backend (mirrors conductivity/calc_all_conductivity.py)
# ══════════════════════════════════════════════════════════════════════════

def parse_top_molecules(top_path):
    """Return list of (molname, nmol) from the [ molecules ] section."""
    with open(top_path) as f:
        lines = f.readlines()
    molecules = []
    in_section = False
    for line in lines:
        line = line.split(';')[0].strip()
        if not line:
            continue
        if line.startswith('['):
            in_section = line.lower().replace(' ', '') == '[molecules]'
            continue
        if in_section:
            molname, nmol = line.split()
            molecules.append((molname, int(nmol)))
    return molecules


def parse_itp(itp_path):
    """Return (moleculetype_name, masses, charges) from an .itp file."""
    with open(itp_path) as f:
        lines = f.readlines()
    section = None
    name = None
    masses, charges = [], []
    for raw in lines:
        line = raw.split(';')[0].strip()
        if not line:
            continue
        if line.startswith('['):
            section = line.lower().replace(' ', '')
            continue
        if section == '[moleculetype]' and name is None:
            name = line.split()[0]
        elif section == '[atoms]':
            parts = line.split()
            charges.append(float(parts[6]))
            masses.append(float(parts[7]))
    return name, masses, charges


def build_species_dicts(sys_dir, molecules):
    """Map [molecules] entries to per-species mass/charge lists using the
    moleculetype names declared in the .itp files in sys_dir."""
    name_to_itp = {}
    for itp_path in glob.glob(os.path.join(sys_dir, '*.itp')):
        if 'forcefield' in os.path.basename(itp_path):
            continue
        molname, masses, charges = parse_itp(itp_path)
        if molname is not None:
            name_to_itp[molname] = (masses, charges)

    species_order, species_mass, species_number, species_charge = [], {}, {}, {}
    for molname, nmol in molecules:
        masses, charges = name_to_itp[molname]
        species_order.append(molname)
        species_mass[molname] = masses
        species_number[molname] = nmol
        species_charge[molname] = int(round(sum(charges)))
    return species_order, species_mass, species_number, species_charge


def parse_ref_temperature(mdp_path):
    with open(mdp_path) as f:
        for line in f:
            line = line.split(';')[0].strip()
            if line.lower().startswith('ref_t') or line.lower().startswith('ref-t'):
                return float(line.split('=')[1].split()[0])
    return None


def load_unwrapped_trajectory_gromacs(tpr_path, xtc_path):
    """Returns (positions [nframes, natoms, 3] in Angstrom, mean box volume in Angstrom^3)."""
    import MDAnalysis as mda
    from MDAnalysis.transformations import unwrap
    from MDAnalysis.transformations.nojump import NoJump

    u = mda.Universe(str(tpr_path), str(xtc_path))
    u.trajectory.add_transformations(unwrap(u.atoms), NoJump())

    nframes = len(u.trajectory)
    natoms = len(u.atoms)
    positions = np.empty((nframes, natoms, 3), dtype=np.float64)
    volumes = np.empty(nframes, dtype=np.float64)
    for i, ts in enumerate(u.trajectory):
        positions[i] = ts.positions
        dims = ts.dimensions
        volumes[i] = dims[0] * dims[1] * dims[2]
    return positions, volumes.mean()


def run_onsager_conductivity_gromacs(sys_dir, ensemble, T_K=None, viscosity_cP=1.0):
    """Compute Onsager/NE conductivity for one GROMACS production system.

    Native GROMACS frame spacing here is 1 ps (dt=1 fs, nstxout-compressed=1000),
    matching the dt onsager_calc assumes internally, so no dt correction is
    applied (mirrors calc_all_conductivity.py).

    Parameters
    ----------
    sys_dir      : directory containing <ensemble>.tpr / <ensemble>.xtc / *.top / *.itp / <ensemble>.mdp
    ensemble     : 'npt' or 'nvt'
    T_K          : reference temperature (K); if None, parsed from <ensemble>.mdp
    viscosity_cP : placeholder dynamic viscosity for the Yeh-Hummer Dself_inf
                   correction (does not affect conductivity outputs)

    Returns
    -------
    dict with keys: sigma_onsager_mS_cm, sigma_NE_mS_cm, Dself_1e10_m2s,
    species_order, V_angstrom3, T_K, n_frames
    """
    sys_dir = Path(sys_dir)
    top_path = glob.glob(str(sys_dir / '*.top'))[0]
    mdp_path = sys_dir / f'{ensemble}.mdp'
    tpr_path = sys_dir / f'{ensemble}.tpr'
    xtc_path = sys_dir / f'{ensemble}.xtc'

    molecules = parse_top_molecules(top_path)
    species_order, species_mass, species_number, species_charge = build_species_dicts(sys_dir, molecules)
    if T_K is None:
        T_K = parse_ref_temperature(mdp_path)

    print(f'[{ensemble}] {sys_dir.name}: loading trajectory ...')
    positions, volume_angstrom3 = load_unwrapped_trajectory_gromacs(tpr_path, xtc_path)
    print(f'[{ensemble}] {sys_dir.name}: {positions.shape[0]} frames, '
          f'<V> = {volume_angstrom3:.2f} A^3, T = {T_K} K')

    result = onsager_calc(
        species_order=species_order,
        species_mass=species_mass,
        species_number=species_number,
        species_charge=species_charge,
        volume_angstrom3=volume_angstrom3,
        viscosity_cP=viscosity_cP,
        T_K=T_K,
        positions=positions,
    )

    return {
        "sigma_onsager_mS_cm": result["conductivity_onsager"],
        "sigma_NE_mS_cm":      result["conductivity_NE"],
        "Dself_1e10_m2s":      result["Dself_inf"],
        "species_order":       species_order,
        "V_angstrom3":         volume_angstrom3,
        "T_K":                 T_K,
        "n_frames":            positions.shape[0],
        "Lambda_onsager":      result["Lambda_onsager"],
        "Lambda_onsager_raw":  result["Lambda_onsager_raw"],
        "Lambda_onsager_unit": result["Lambda_onsager_unit"],
        "species_order":       species_order,
    }
