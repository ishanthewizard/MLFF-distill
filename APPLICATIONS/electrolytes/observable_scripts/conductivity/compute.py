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
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io.trajectory import Trajectory as _AseTraj
from ase.geometry import find_mic

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
        for k in range(1, T):
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
    stride = max(1, (n_total - i_start) // n_sample)
    vols = []
    with _AseTraj(str(traj_path)) as trj:
        for i in range(i_start, n_total, stride):
            try:
                vols.append(float(trj[i].get_volume()))
            except Exception:
                continue
    return float(np.mean(vols)) if vols else 0.0


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
    result = run_onsager_conductivity(
        traj_path, cat_symbol, anion_symbol, solvent_symbol, dt_fs, T_K, **kwargs
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
    }
