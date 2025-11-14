#!/usr/bin/env python3
"""
Diffusion from MSD (10 ns trajs @ 10 fs).
- Streams frames (no full load), COM-drift removal, MIC unwrapping.
- Time-origin averaged MSD, linear fit in Einstein regime.
- Computes D for: cation (Na/Li), anion (molecular COM), solvent (molecular COM).
- Robust molecule grouping:
   * Prefer molecule IDs in atoms.arrays (molid/resid/etc.)
   * Else build connectivity from ASE neighbor list (once from first analyzed frame)
- Robust classifying:
   * Cation groups: contain Li/Na
   * Anion groups:
       PF6–: P + >=6 F
       OTf–: S + >=3 O + >=3 F
       TFSI–: N + >=2 S + >=6 F      (loose but effective)
   * Solvent groups: the rest
"""

from ase.io import Trajectory
from ase.geometry import find_mic
from ase.neighborlist import natural_cutoffs, NeighborList
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
import pickle
# ---------------- user config ----------------
OUT_DIR  = Path("/global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/temp/test_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# TO LLM: you need to pay attention to the format when doing auto completion: (traj_path, system_name, cat_symbol(need to be strictly matched with cation_dict), anion_symbol(need to be strictly matched with anion_dict), solvent_symbol(need to be strictly matched with solvent_dict))
TARGETS = [

    ("/global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/ablate_distillation/10ns_solvent_0_1M/md_omol_naotf_tgdme_1m_s1p1/md_omol_naotf_tgdme_1m_s1p1.traj",    "NaOTf — TGDME",     "Na", "OTf", "TGDME"),
    # ("/global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/ablate_distillation/10ns_solvent_0_1M/md_omol_naotf_diglyme_1m_s1p1/md_omol_naotf_diglyme_1m_s1p1.traj",    "NaOTf — Diglyme",     "Na", "OTf", "Diglyme"),
    ("/global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/ablate_distillation/10ns_solvent_0_1M/md_omol_naotf_dme_s1p1_omol/md_omol_naotf_dme_s1p1_omol.traj",    "NaOTf — DME",     "Na", "OTf", "DME"),
    ("/global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/ablate_distillation/10ns_solvent_0_1M/md_omol_naotf_pc_1m_s1p1/md_omol_naotf_pc_1m_s1p1.traj",    "NaOTf — PC",     "Na", "OTf", "PC"),
    # ("/global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/ablate_distillation/10ns_solvent_0_1M/md_omol_napf6_diglyme_pfactor_0.1_1fs/md_omol_napf6_diglyme_pfactor_0.1_1fs.traj", "NaPF6 — Diglyme", "Na", "PF6", "Diglyme"),
    ("/global/homes/y/yuejian/project/MLFF-distill/yuejian/OMOL/electrolyte_application/ablate_distillation/10ns_solvent_0_1M/md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1/md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1.traj", "NaPF6 — Propylene Carbonate", "Na", "PF6", "PC"),
]

cation_dict = {
    "Na": ["Na"],
    "Li": ["Li"],
}

anion_dict = {
    "PF6": ["P", "F", "F", "F", "F", "F", "F"],
    "OTf": ["C", "S", "F", "F", "F", "O", "O", "O"],
}

solvent_dict = {
    "Diglyme": ["H", "C", "O", "H", "H", "C", "C", "H", "H", "O", "H", "H", "C", "C", "H", "H", "O", "H", "H", "C", "H", "H", "H"],
    "DME":["H", "C", "O", "H", "H", "C", "C", "H", "H", "O", "H", "H", "C", "H", "H", "H"],
    "PC": ["C", "H", "O", "C", "C", "H", "H", "O", "O", "C", "H", "H", "H"],
    "TGDME": ["H", "C", "O", "H", "H", "C", "C", "H", "H", "O", "H", "H", "C", "C", "H", "H", "O", "H", "H", "C", "C",
              "H", "H", "O", "H", "H", "C", "C", "H", "H", "O", "H", "H", "C", "H", "H", "H"],
    "DEG": ["H", "O", "C", "C", "H", "H", "O", "H", "H", "C", "C", "H", "H", "O", "H", "H", "H"],
}

EQ_TIME_PS       = 100.0
KNOWN_DT_PS      = 0.01       # 10 fs
TARGET_FRAMES    = 10000 # this is about dt < 1ps
TAU_MIN_FIT_PS   = 1000.0 # 1000 ps  = 1 ns
TAU_MAX_FIT_PS   = 9000.0 # 9000 ps = 9 ns
N_WORKERS        = 8
PLOT_NCOLS       = 2
PARALLEL_MSD     = False # not sure if this is reliable
# ---------------- helpers ----------------
def a2ps_to_m2s(D_a2ps):
    return D_a2ps * 1e-20 / 1e-12

def species_indices(symbols, species):
    return [i for i, s in enumerate(symbols) if s == species]

def _first_available_mol_id_array(atoms):
    for key in ("molid","mol_id","molecule","resid","resids"):
        try:
            arr = atoms.get_array(key)
            return key, arr
        except Exception:
            continue
    return None, None

def direct_groups_from_species(symbols, species_list):
    """
    Identify atomic groups by matching a species pattern sequence in the atomic symbols.
    
    Uses a sliding window approach to find exact matches of the species pattern
    in the atomic symbol list. Each match represents a potential molecule/group.
    Overlapping matches are deduplicated, keeping only the first occurrence.
    
    Parameters
    ----------
    symbols : list[str]
        List of chemical symbols for all atoms in the system, in order.
    species_list : list[str]
        List of chemical symbols representing the pattern to match (e.g., 
        ["P", "F", "F", "F", "F", "F", "F"] for PF6).
        The pattern must match exactly in both element identity and order.
    
    Returns
    -------
    list[np.ndarray]
        List of numpy arrays, where each array contains integer indices of atoms
        that form a matching pattern. Each array represents one identified group.
        Returns empty list if no matches are found.
    
    Examples
    --------
    >>> symbols = ["Na", "P", "F", "F", "F", "F", "F", "F", "Na"]
    >>> species_list = ["P", "F", "F", "F", "F", "F", "F"]
    >>> direct_groups_from_species(symbols, species_list)
    [array([1, 2, 3, 4, 5, 6, 7])]
    """
    groups = []
    window_len = len(species_list)
    
    # Search for pattern matches using a sliding window
    for i in range(len(symbols) - window_len + 1):
        window = symbols[i:i+window_len]
        
        # Check for exact match in both element and order
        if list(window) == list(species_list):
            indices = np.array(range(i, i+window_len), dtype=int)
            groups.append(indices)
    
    # Early return if no matches found
    if len(groups) == 0:
        return groups
    
    # Remove overlapping groups, keeping the first occurrence of each pattern
    unique_groups = []
    seen_indices = set()
    
    for group in groups:
        group_set = set(group)
        # Only add group if it doesn't overlap with previously seen groups
        if not group_set.intersection(seen_indices):
            unique_groups.append(group)
            seen_indices.update(group_set)
    
    return unique_groups
    
def _groups_from_ids(ids):
    groups = []
    for uid in np.unique(ids):
        groups.append(np.where(ids == uid)[0].astype(int))
    return groups

def _groups_from_connectivity(atoms):
    # Build connectivity via covalent radii (this is run once on the first frame)
    cutoffs = natural_cutoffs(atoms, mult=1.1)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    n = len(atoms)
    adj = [[] for _ in range(n)]
    for i in range(n):
        ii, _ = nl.get_neighbors(i)
        adj[i].extend(ii.tolist())
    # Connected components
    seen = np.zeros(n, dtype=bool)
    groups = []
    for i in range(n):
        if seen[i]: continue
        stack = [i]
        seen[i] = True
        comp = []
        while stack:
            j = stack.pop()
            comp.append(j)
            for k in adj[j]:
                if not seen[k]:
                    seen[k] = True
                    stack.append(k)
        groups.append(np.array(comp, dtype=int))
    return groups

def _count_in_group(symbols, idxs, element):
    return sum(1 for i in idxs if symbols[i] == element)

def _classify_groups(groups, symbols):
    """Return dictionaries of index-lists for cation, anion, solvent molecules."""
    cats, ans, solv = [], [], []
    for g in groups:
        gset = {symbols[i] for i in g}
        # cation molecule = contains Li or Na (single atom group typically)
        if "Li" in gset or "Na" in gset:
            cats.append(g); continue
        # PF6–: P + >=6 F
        if ("P" in gset) and (_count_in_group(symbols, g, "F") >= 6):
            ans.append(g); continue
        # OTf–: S + >=3 O + >=3 F
        if ("S" in gset) and (_count_in_group(symbols, g, "O") >= 3) and (_count_in_group(symbols, g, "F") >= 3):
            ans.append(g); continue
        # TFSI– (loose): N + >=2 S + >=6 F
        if ("N" in gset) and (_count_in_group(symbols, g, "S") >= 2) and (_count_in_group(symbols, g, "F") >= 6):
            ans.append(g); continue
        # else solvent
        solv.append(g)
    return cats, ans, solv

def _mass_weighted_com(positions, masses):
    msum = masses.sum()
    return (positions * masses[:, None]).sum(axis=0) / msum if msum > 0 else positions.mean(axis=0)

def stream_subsample_unwrap(traj_path, start_ps, dt_ps, target_frames, cat_symbol, anion_symbol, solvent_symbol):
    """
    Streams frames, subsamples, removes system COM drift, and unwraps via MIC.
    Builds molecule groups once (first analyzed frame), classifies into cation/anion/solvent.
    Returns:
      tau_ps,
      pos_cat (T, M_cat, 3),
      pos_anion (T, M_an, 3) or None,
      pos_solvent (T, M_sol, 3) or None,
      stride, T, dt_ps
    """
    with Trajectory(str(traj_path)) as trj:
        n_total = len(trj)
        if n_total < 3: # at least 3 frames are needed for the analysis
            raise RuntimeError("Too few frames in trajectory.")

        # infer dt if needed
        if dt_ps is None:
            t0 = trj[0].info.get("time", None)
            t1 = trj[1].info.get("time", None)
            dt_ps = float(t1) - float(t0) if (t0 is not None and t1 is not None) else 0.01

        i_start = int(np.floor(start_ps / dt_ps))
        i_start = min(max(i_start, 0), n_total - 2)
        avail = n_total - i_start
        # subsample the trajectory uniformly over the available time window
        stride = max(1, int(np.ceil(avail / target_frames)))

        # First frame for setup
        f0 = trj[i_start]
        symbols0 = f0.get_chemical_symbols()
        masses0  = f0.get_masses()

        # # Molecule grouping (prefer IDs; else connectivity)
        # key, ids = _first_available_mol_id_array(f0)
        # if key is not None:
        #     print(f"Using molecule IDs for grouping")
        #     groups = _groups_from_ids(ids)
        # else:
        #     print(f"Using connectivity for grouping")
        #     groups = _groups_from_connectivity(f0)

        # # Classify groups
        # cat_groups, anion_groups, solvent_groups = _classify_groups(groups, symbols0)
        cat_symbol_list = cation_dict[cat_symbol]
        anion_symbol_list = anion_dict[anion_symbol]
        solvent_symbol_list = solvent_dict[solvent_symbol]
        
        cat_groups = direct_groups_from_species(symbols0, cat_symbol_list)
        anion_groups = direct_groups_from_species(symbols0, anion_symbol_list)
        solvent_groups = direct_groups_from_species(symbols0, solvent_symbol_list)
        # sanity check 1: flatting all groups and sum them all, then the length should be the same as the number of atoms in the system
        total_atoms = len(symbols0)
        total_atoms_in_groups = sum(len(g) for g in cat_groups + anion_groups + solvent_groups)
        if total_atoms_in_groups != total_atoms:
            raise RuntimeError(f"Total atoms in groups ({total_atoms_in_groups}) do not match total atoms in system ({total_atoms}) in {traj_path.name}.")
        else:
            print(f"Total atoms in groups ({total_atoms_in_groups}) match total atoms in system ({total_atoms}) in {traj_path.name}.")
        

        # Cation indices (single-atom groups usually)
        cat_idx = [i for i, s in enumerate(symbols0) if s == cat_symbol]
        if not cat_idx:
            raise RuntimeError(f"No {cat_symbol} atoms found in {traj_path.name}.")

        idxs = list(range(i_start, n_total, stride))
        T = len(idxs)

        # Allocate arrays
        pos_cat     = np.zeros((T, len(cat_idx), 3))
        pos_anion   = np.zeros((T, len(anion_groups), 3))   if anion_groups   else None
        pos_solvent = np.zeros((T, len(solvent_groups), 3)) if solvent_groups else None

        # First time point (t0)
        f_prev = trj[idxs[0]]
        com_prev = f_prev.get_center_of_mass()
        # cation positions
        pos_cat[0] = f_prev.get_positions()[cat_idx] - com_prev
        # anion COMs
        if pos_anion is not None:
            P0 = f_prev.get_positions() - com_prev
            M0 = f_prev.get_masses()
            for j, g in enumerate(anion_groups):
                pos_anion[0, j] = _mass_weighted_com(P0[g], M0[g])
        # solvent COMs
        if pos_solvent is not None:
            P0 = f_prev.get_positions() - com_prev
            M0 = f_prev.get_masses()
            for j, g in enumerate(solvent_groups):
                pos_solvent[0, j] = _mass_weighted_com(P0[g], M0[g])

        # Stream/unwrap
        for k in tqdm(range(1, T), desc=f"{traj_path.name}: unwrap", unit="frame", leave=False):
            f = trj[idxs[k]]
            f_p = trj[idxs[k - 1]]

            com_c = f.get_center_of_mass()
            com_p = f_p.get_center_of_mass()

            # --- cations ---
            curr_cat = f.get_positions()[cat_idx] - com_c
            prev_cat = f_p.get_positions()[cat_idx] - com_p
            disp_cat, _ = find_mic(curr_cat - prev_cat, f.get_cell(), pbc=f.get_pbc())
            pos_cat[k] = pos_cat[k - 1] + disp_cat

            # --- anions (COM per molecule) ---
            if pos_anion is not None:
                posC = f.get_positions() - com_c
                posP = f_p.get_positions() - com_p
                mC = f.get_masses()
                mP = f_p.get_masses()
                curr_com = np.empty_like(pos_anion[0])
                prev_com = np.empty_like(pos_anion[0])
                for j, g in enumerate(anion_groups):
                    curr_com[j] = _mass_weighted_com(posC[g], mC[g])
                    prev_com[j] = _mass_weighted_com(posP[g], mP[g])
                disp_an, _ = find_mic(curr_com - prev_com, f.get_cell(), pbc=f.get_pbc())
                pos_anion[k] = pos_anion[k - 1] + disp_an

            # --- solvent (COM per molecule) ---
            if pos_solvent is not None:
                posC = f.get_positions() - com_c
                posP = f_p.get_positions() - com_p
                mC = f.get_masses()
                mP = f_p.get_masses()
                curr_com = np.empty_like(pos_solvent[0])
                prev_com = np.empty_like(pos_solvent[0])
                for j, g in enumerate(solvent_groups):
                    curr_com[j] = _mass_weighted_com(posC[g], mC[g])
                    prev_com[j] = _mass_weighted_com(posP[g], mP[g])
                disp_sv, _ = find_mic(curr_com - prev_com, f.get_cell(), pbc=f.get_pbc())
                pos_solvent[k] = pos_solvent[k - 1] + disp_sv

        tau_ps = np.arange(T, dtype=float) * stride * dt_ps
        return tau_ps, pos_cat, pos_anion, pos_solvent, stride, T, dt_ps

def _msd_chunk(args):
    unwrapped, lag_start, lag_end = args
    T, M, _ = unwrapped.shape
    out = np.zeros(lag_end - lag_start)
    for k, lag in enumerate(range(lag_start, lag_end)):
        d = unwrapped[lag:] - unwrapped[:T - lag]
        out[k] = np.sum(d * d, axis=2).mean()
    return lag_start, out

def msd_time_origin_parallel(unwrapped, n_workers=4, chunk_size=200):
    T = unwrapped.shape[0]
    max_lag = T - 1
    chunks = [(unwrapped, s, min(s + chunk_size, max_lag + 1))
              for s in range(0, max_lag + 1, chunk_size)]
    msd = np.zeros(max_lag + 1)
    if n_workers <= 1:
        for lag_start, arr in map(_msd_chunk, chunks):
            msd[lag_start:lag_start + len(arr)] = arr
    else:
        with Pool(processes=n_workers) as pool:
            for lag_start, arr in pool.imap(_msd_chunk, chunks):
                msd[lag_start:lag_start + len(arr)] = arr
    return msd

def msd_time_origin(unwrapped):
    """
    Time-origin averaged MSD over atoms and time origins.
    Returns (msd, se) with shapes (T,).
    """
    T, M, _ = unwrapped.shape
    max_lag = T - 1
    msd = np.zeros(max_lag + 1, dtype=float)
    var = np.zeros_like(msd)
    print("Computing MSD...", flush=True)
    for lag in tqdm(range(max_lag + 1)):
        d = unwrapped[lag:] - unwrapped[:T-lag]      # (T-lag, M, 3)
        dr2 = np.sum(d**2, axis=2).reshape(-1)       # flatten
        msd[lag] = dr2.mean()
        var[lag] = dr2.var(ddof=1) / max(1, dr2.size)
    # se = np.sqrt(var)
    return msd

def fit_diffusion(tau_ps, msd, tau_min_ps=15.0, tau_max_ps=None):
    mask = tau_ps >= tau_min_ps
    if tau_max_ps is not None:
        mask &= (tau_ps <= tau_max_ps)
    x, y = tau_ps[mask], msd[mask]
    if x.size < 10:
        raise RuntimeError("Not enough points for linear fit.")
    slope, intercept = np.polyfit(x, y, 1)
    D_a2ps = slope / 6.0
    return D_a2ps, slope, intercept, mask

# ---------------- plotting grid ----------------
n_sys = len(TARGETS)
ncols = PLOT_NCOLS
nrows = (n_sys + ncols - 1) // ncols
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 6.5), squeeze=False)
rows_out = []

# ---------------- main loop ----------------
for k, (fname, title, cat, anion, solvent) in enumerate(tqdm(TARGETS, desc="Processing trajectories", unit="traj")):
    ax = axes[k // ncols, k % ncols]
    traj_path = Path(fname)
    if not traj_path.exists():
        print(f"⚠️ Missing {traj_path}")
        ax.set_visible(False)
        continue

    print(f"\n🔹 {title} ← {traj_path.name}")
    tau, pos_cat, pos_anion, pos_solvent, stride_used, T_used, dt_ps_used = stream_subsample_unwrap(
        traj_path, EQ_TIME_PS, KNOWN_DT_PS, TARGET_FRAMES, cat, anion, solvent
    )
    dt_ps_eff = stride_used * dt_ps_used
    print(f"   → stride={stride_used} (~{dt_ps_eff:.3f} ps), frames={T_used}, window≈{tau[-1]:.1f} ps")

    # MSD calculation
    if PARALLEL_MSD:
        print(f"   → MSD({cat}, {anion}, {solvent}) with {N_WORKERS} workers…")
        msd_cat = msd_time_origin_parallel(pos_cat, n_workers=N_WORKERS)
        msd_anion = msd_time_origin_parallel(pos_anion, n_workers=N_WORKERS) if pos_anion is not None else None
        msd_solvent = msd_time_origin_parallel(pos_solvent, n_workers=N_WORKERS) if pos_solvent is not None else None
    else:
        print(f"   → MSD({cat}, {anion}, {solvent})…")
        msd_cat = msd_time_origin(pos_cat)
        msd_anion = msd_time_origin(pos_anion) if pos_anion is not None else None
        msd_solvent = msd_time_origin(pos_solvent) if pos_solvent is not None else None

    # save msd to dictionary for anion cation and solvent respectively
    msd_dict = {
        # "frames_total": T_used,
        "msd_cat": msd_cat,
        "msd_anion": msd_anion,
        "msd_solvent": msd_solvent,
        "tau": tau,
        "dt_ps": dt_ps_eff,
        "EQ_TIME_PS": EQ_TIME_PS,
    }
    with open(OUT_DIR / f"msd_dict_{title}.pkl", "wb") as f:
        pickle.dump(msd_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Fits
    Dcat_a2ps, slope_c, b_c, mask_c = fit_diffusion(tau, msd_cat, TAU_MIN_FIT_PS, TAU_MAX_FIT_PS)
    Dcat_1e10 = a2ps_to_m2s(Dcat_a2ps) * 1e10
    print(f"   → D({cat}⁺) = {Dcat_1e10:.2f} ×10⁻¹⁰ m²/s ({Dcat_a2ps:.4f} Å²/ps)")

    Danion_a2ps = Danion_1e10 = None
    if msd_anion is not None:
        Danion_a2ps, slope_a, b_a, mask_a = fit_diffusion(tau, msd_anion, TAU_MIN_FIT_PS, TAU_MAX_FIT_PS)
        Danion_1e10 = a2ps_to_m2s(Danion_a2ps) * 1e10
        print(f"   → D(anion) = {Danion_1e10:.2f} ×10⁻¹⁰ m²/s ({Danion_a2ps:.4f} Å²/ps)")

    Dsolv_a2ps = Dsolv_1e10 = None
    if msd_solvent is not None:
        Dsolv_a2ps, slope_s, b_s, mask_s = fit_diffusion(tau, msd_solvent, TAU_MIN_FIT_PS, TAU_MAX_FIT_PS)
        Dsolv_1e10 = a2ps_to_m2s(Dsolv_a2ps) * 1e10
        print(f"   → D(solvent) = {Dsolv_1e10:.2f} ×10⁻¹⁰ m²/s ({Dsolv_a2ps:.4f} Å²/ps)")

    # Plot
    ax.plot(tau, msd_cat, lw=1.5, label=f"{cat} MSD")
    ax.plot(tau[mask_c], (b_c + slope_c * tau)[mask_c], "--", lw=1.0, label=f"{cat} fit")
    if msd_anion is not None:
        ax.plot(tau, msd_anion, lw=1.2, label="Anion MSD", alpha=0.9)
        ax.plot(tau[mask_a], (b_a + slope_a * tau)[mask_a], "--", lw=1.0, label="Anion fit")
    if msd_solvent is not None:
        ax.plot(tau, msd_solvent, lw=1.2, label="Solvent MSD", alpha=0.9)
        ax.plot(tau[mask_s], (b_s + slope_s * tau)[mask_s], "--", lw=1.0, label="Solvent fit")

    ax.set_title(title)
    ax.set_xlabel(r"$\tau$ since 100 ps (ps)")
    ax.set_ylabel(r"MSD ($\mathrm{\AA^2}$)")
    ax.grid(True, linestyle=":")

    note = (f"D({cat}⁺) = {Dcat_1e10:.2f}×10⁻¹⁰ m²/s\n= {Dcat_a2ps:.4f} Å²/ps")
    if Danion_1e10 is not None:
        note += f"\nD(anion) = {Danion_1e10:.2f}×10⁻¹⁰"
    if Dsolv_1e10 is not None:
        note += f"\nD(solvent) = {Dsolv_1e10:.2f}×10⁻¹⁰"
    ax.text(0.98, 0.02, note, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))
    ax.legend(frameon=False, fontsize=8)

    rows_out.append({
        "system": title,
        "cation": cat,
        "subsample_stride": stride_used,
        "frames_used": T_used,
        "effective_dt_ps": stride_used * dt_ps_used,
        "analysis_window_ps": float(tau[-1]),
        "tau_min_fit_ps": TAU_MIN_FIT_PS,
        "D_cation_(x1e-10_m2_s)": float(Dcat_1e10),
        "D_cation_A2_per_ps": float(Dcat_a2ps),
        "D_anion_(x1e-10_m2_s)": float(Danion_1e10) if Danion_1e10 is not None else None,
        "D_anion_A2_per_ps": float(Danion_a2ps) if Danion_a2ps is not None else None,
        "D_solvent_(x1e-10_m2_s)": float(Dsolv_1e10) if Dsolv_1e10 is not None else None,
        "D_solvent_A2_per_ps": float(Dsolv_a2ps) if Dsolv_a2ps is not None else None,
    })

# hide any empty axes
for j in range(len(TARGETS), nrows * ncols):
    axes[j // ncols, j % ncols].set_visible(False)

fig.suptitle(f"MSD & Diffusion — ≥{EQ_TIME_PS:.0f} ps, fit {TAU_MIN_FIT_PS:.0f}–{TAU_MAX_FIT_PS:.0f} ps", fontsize=12.5)
fig.tight_layout(rect=(0, 0.03, 1, 0.96))
png_path = OUT_DIR / "uma_diffusion_selected_fast.png"
fig.savefig(png_path, dpi=300)
plt.close(fig)
print(f"\n✅ Saved plot → {png_path}")

if rows_out:
    df = pd.DataFrame(rows_out)
    csv_path = OUT_DIR / "uma_diffusion_selected_fast.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Wrote CSV → {csv_path}\n")
    print(df.to_string(index=False))
else:
    print("⚠️ No results produced.")
