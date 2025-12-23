"""
MSD / diffusion calculator for GROMACS trajectories (XTC/TRR + topology).

Workflow (mirrors `msd_with_com.py` but uses MDAnalysis for I/O):
1) Stream frames from the trajectory with subsampling and COM-drift removal.
2) Build molecule groups once from the first analyzed frame using the same
   species patterns/dictionaries as `msd_with_com.py`.
3) Unwrap displacements via minimum image convention.
4) Compute time-origin averaged MSDs and Einstein-slope diffusivities.

Notes
-----
- Relies on functions imported from `msd_with_com.py` for MSD + fitting logic.
- Box handling: uses triclinic vectors if present, otherwise assumes
  orthorhombic (diagonal cell).
- Units: MDAnalysis exposes distances in Å and timestep in ps; ensure your
  topology/trajectory carries correct unit metadata.

Run example (uses fairchemV2 env and current TARGETS):
python /global/homes/y/yuejian/project/MLFF-distill/APPLICATIONS/electrolytes/observable_scripts/mean_square_displacement/msds_calculation_batch_gromacs.py -o /global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/OPLS_diffusivities/OPLS_init_try_3_systems -t 20000
"""

from pathlib import Path
from typing import Sequence, Tuple, Optional

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import MDAnalysis as mda
from ase.geometry import find_mic

from utils.component_dictionary import cation_dict, anion_dict, solvent_dict
from msd_with_com import (
    a2ps_to_m2s,
    direct_groups_from_species,
    _mass_weighted_com,
    msd_time_origin,
    msd_time_origin_parallel,
    fit_diffusion,
    fit_diffusion_wrapper,
)


def _cell_from_ts(ts) -> np.ndarray:
    """Return a 3x3 cell matrix (Å) from an MDAnalysis Timestep."""
    triclinic = getattr(ts, "triclinic_dimensions", None)
    if triclinic is not None:
        arr = np.asarray(triclinic, dtype=float)
        if arr.shape == (3, 3):
            return arr
    # Fallback: orthorhombic with lengths in the first three dims entries
    lengths = np.asarray(ts.dimensions[:3], dtype=float)
    return np.diag(lengths)


def stream_subsample_unwrap_gromacs(
    topology_path: Path,
    traj_path: Path,
    start_ps: float,
    dt_ps: Optional[float],
    target_frames: int,
    cat_symbol: str,
    anion_symbol: str,
    solvent_symbol: str,
    TAU_MAX_FIT_PS: float,
):
    """
    Streaming loader for GROMACS trajectories using MDAnalysis with COM drift
    removal and MIC unwrapping. Interface mirrors `stream_subsample_unwrap`
    in `msd_with_com.py`.
    """
    u = mda.Universe(str(topology_path), str(traj_path))
    n_total = len(u.trajectory)
    if n_total < 3:
        raise RuntimeError("Too few frames in trajectory.")

    # infer dt if not provided
    if dt_ps is None:
        dt_ps = float(u.trajectory.dt)
        if dt_ps <= 0:
            raise RuntimeError("Could not infer timestep (dt_ps). Provide --known-dt-ps.")

    # cap frames based on TAU_MAX_FIT_PS
    n_frames = int(TAU_MAX_FIT_PS / dt_ps)
    if n_total < n_frames:
        raise RuntimeError(
            f"Trajectory {traj_path.name} has fewer frames ({n_total}) than the maximum fitting time ({TAU_MAX_FIT_PS} ps)."
        )
    n_total = n_frames

    i_start = int(np.floor(start_ps / dt_ps))
    i_start = min(max(i_start, 0), n_total - 2)
    avail = n_total - i_start
    stride = max(1, int(np.ceil(avail / target_frames)))

    # Prime first frame
    ts0 = u.trajectory[i_start]
    atoms = u.atoms
    # Normalize types like "NA" -> "Na" for dictionary lookups
    symbols0 = [s.capitalize() for s in atoms.types]
    masses0 = atoms.masses

    cat_symbol_list = cation_dict[cat_symbol]
    anion_symbol_list = anion_dict[anion_symbol]
    solvent_symbol_list = solvent_dict[solvent_symbol]

    cat_groups = direct_groups_from_species(symbols0, cat_symbol_list)
    anion_groups = direct_groups_from_species(symbols0, anion_symbol_list)
    solvent_groups = direct_groups_from_species(symbols0, solvent_symbol_list)

    total_atoms = len(symbols0)
    assigned = set()
    for g in cat_groups + anion_groups + solvent_groups:
        assigned.update(g.tolist())
    missing = sorted(set(range(total_atoms)) - assigned)
    if missing:
        print(
            f"⚠️ {len(missing)} atoms were not matched by species patterns in {traj_path.name}; "
            "treating them as individual solvent groups."
        )
        for idx in missing:
            solvent_groups.append(np.array([idx], dtype=int))
    total_atoms_in_groups = sum(len(g) for g in cat_groups + anion_groups + solvent_groups)
    if total_atoms_in_groups != total_atoms:
        print(
            f"⚠️ After adding leftovers, grouped atoms {total_atoms_in_groups} still differ from total {total_atoms}; proceeding anyway."
        )
    else:
        print(f"Total atoms in groups ({total_atoms_in_groups}) match total atoms in system ({total_atoms}) in {traj_path.name}.")

    cat_idx = [i for i, s in enumerate(symbols0) if s == cat_symbol]
    if not cat_idx:
        raise RuntimeError(f"No {cat_symbol} atoms found in {traj_path.name}.")

    idxs = list(range(i_start, n_total, stride))
    T = len(idxs)

    pos_cat = np.zeros((T, len(cat_idx), 3))
    pos_anion = np.zeros((T, len(anion_groups), 3)) if anion_groups else None
    pos_solvent = np.zeros((T, len(solvent_groups), 3)) if solvent_groups else None

    com_prev = atoms.center_of_mass()
    prev_positions = atoms.positions.copy()
    pos_cat[0] = prev_positions[cat_idx] - com_prev
    if pos_anion is not None:
        for j, g in enumerate(anion_groups):
            pos_anion[0, j] = _mass_weighted_com(prev_positions[g] - com_prev, masses0[g])
    if pos_solvent is not None:
        for j, g in enumerate(solvent_groups):
            pos_solvent[0, j] = _mass_weighted_com(prev_positions[g] - com_prev, masses0[g])

    for k in tqdm(range(1, T), desc=f"{traj_path.name}: unwrap", unit="frame", leave=False):
        ts_curr = u.trajectory[idxs[k]]
        cell_curr = _cell_from_ts(ts_curr)
        curr_positions = atoms.positions.copy()
        com_curr = atoms.center_of_mass()

        curr_cat = curr_positions[cat_idx] - com_curr
        prev_cat = prev_positions[cat_idx] - com_prev
        disp_cat, _ = find_mic(curr_cat - prev_cat, cell_curr, pbc=[True, True, True])
        pos_cat[k] = pos_cat[k - 1] + disp_cat

        if pos_anion is not None:
            curr_com = np.empty_like(pos_anion[0])
            prev_com = np.empty_like(pos_anion[0])
            for j, g in enumerate(anion_groups):
                curr_com[j] = _mass_weighted_com(curr_positions[g] - com_curr, masses0[g])
                prev_com[j] = _mass_weighted_com(prev_positions[g] - com_prev, masses0[g])
            disp_an, _ = find_mic(curr_com - prev_com, cell_curr, pbc=[True, True, True])
            pos_anion[k] = pos_anion[k - 1] + disp_an

        if pos_solvent is not None:
            curr_com = np.empty_like(pos_solvent[0])
            prev_com = np.empty_like(pos_solvent[0])
            for j, g in enumerate(solvent_groups):
                curr_com[j] = _mass_weighted_com(curr_positions[g] - com_curr, masses0[g])
                prev_com[j] = _mass_weighted_com(prev_positions[g] - com_prev, masses0[g])
            disp_sv, _ = find_mic(curr_com - prev_com, cell_curr, pbc=[True, True, True])
            pos_solvent[k] = pos_solvent[k - 1] + disp_sv

        com_prev = com_curr
        prev_positions = curr_positions

    tau_ps = np.arange(T, dtype=float) * stride * dt_ps
    return tau_ps, pos_cat, pos_anion, pos_solvent, stride, T, dt_ps


def process_target(
    target,
    EQ_TIME_PS,
    KNOWN_DT_PS,
    TARGET_FRAMES,
    TAU_MIN_FIT_PS,
    TAU_MAX_FIT_PS,
    N_WORKERS,
    PARALLEL_MSD,
    OUT_DIR: Path,
    ax,
    idx: int,
):
    (
        topology_path,
        traj_path,
        title,
        cat,
        anion,
        solvent,
        concentration,
        temperature,
    ) = target

    traj_path = Path(traj_path)
    topology_path = Path(topology_path)
    if not traj_path.exists():
        print(f"⚠️ Missing trajectory {traj_path}")
        ax.set_visible(False)
        return None
    if not topology_path.exists():
        print(f"⚠️ Missing topology {topology_path}")
        ax.set_visible(False)
        return None

    title = f"{title}_{temperature}_{concentration}"
    print(f"\n🔹 {title} ← {traj_path.name}")
    tau, pos_cat, pos_anion, pos_solvent, stride_used, T_used, dt_ps_used = stream_subsample_unwrap_gromacs(
        topology_path,
        traj_path,
        EQ_TIME_PS,
        KNOWN_DT_PS,
        TARGET_FRAMES,
        cat,
        anion,
        solvent,
        TAU_MAX_FIT_PS,
    )
    dt_ps_eff = stride_used * dt_ps_used
    print(f"   → stride={stride_used} (~{dt_ps_eff:.3f} ps), frames={T_used}, window≈{tau[-1]:.1f} ps")

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

    msd_dict = {
        "msd_cat": msd_cat,
        "msd_anion": msd_anion,
        "msd_solvent": msd_solvent,
        "tau": tau,
        "dt_ps": dt_ps_eff,
        "EQ_TIME_PS": EQ_TIME_PS,
    }
    with open(OUT_DIR / f"msd_dict_{title}.pkl", "wb") as f:
        import pickle

        pickle.dump(msd_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

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

    ax.plot(tau, msd_cat, lw=1.5, label=f"{cat} MSD")
    ax.plot(tau[mask_c], (b_c + slope_c * tau)[mask_c], "--", lw=1.0, label=f"{cat} fit")
    if msd_anion is not None:
        ax.plot(tau, msd_anion, lw=1.2, label="Anion MSD", alpha=0.9)
        ax.plot(tau[mask_a], (b_a + slope_a * tau)[mask_a], "--", lw=1.0, label="Anion fit")
    if msd_solvent is not None:
        ax.plot(tau, msd_solvent, lw=1.2, label="Solvent MSD", alpha=0.9)
        ax.plot(tau[mask_s], (b_s + slope_s * tau)[mask_s], "--", lw=1.0, label="Solvent fit")

    ax.set_title(title)
    ax.set_xlabel(r"$\tau$ since equilibration (ps)")
    ax.set_ylabel(r"MSD ($\mathrm{\AA^2}$)")
    ax.grid(True, linestyle=":")

    note = (f"D({cat}⁺) = {Dcat_1e10:.2f}×10⁻¹⁰ m²/s\n= {Dcat_a2ps:.4f} Å²/ps")
    if Danion_1e10 is not None:
        note += f"\nD(anion) = {Danion_1e10:.2f}×10⁻¹⁰"
    if Dsolv_1e10 is not None:
        note += f"\nD(solvent) = {Dsolv_1e10:.2f}×10⁻¹⁰"
    ax.text(
        0.98,
        0.02,
        note,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9),
    )
    ax.legend(frameon=False, fontsize=8)

    # Diffusivity vs max fit time plot
    D_cations = []
    D_anions = []
    D_solvent = []
    tau_max_fit_ps_list = np.linspace(TAU_MIN_FIT_PS + 1000, TAU_MAX_FIT_PS, 10)
    for max_fit_ps in tau_max_fit_ps_list:
        Dcat_1e10_tmp, Danion_1e10_tmp, Dsolv_1e10_tmp, _, _, _ = fit_diffusion_wrapper(
            tau, msd_cat, msd_anion, msd_solvent, TAU_MIN_FIT_PS, max_fit_ps
        )
        D_cations.append(Dcat_1e10_tmp)
        D_anions.append(Danion_1e10_tmp)
        D_solvent.append(Dsolv_1e10_tmp)

    fig_diff, ax_diff = plt.subplots()
    ax_diff.plot(tau_max_fit_ps_list / 1000.0, D_cations, label="cation")
    ax_diff.plot(tau_max_fit_ps_list / 1000.0, D_anions, label="anion")
    ax_diff.plot(tau_max_fit_ps_list / 1000.0, D_solvent, label="solvent")
    ax_diff.legend()
    ax_diff.set_xlabel("simulation time (ns)")
    ax_diff.set_ylabel("Diffusion coefficient (×10⁻¹⁰m²/s)")
    ax_diff.set_title(f"{title} Diffusivity - simulation time")
    fig_diff.savefig(OUT_DIR / f"Diffusivity_vs_time_{title}.png")
    plt.close(fig_diff)

    return {
        "system": title,
        "cation": cat,
        "anion": anion,
        "solvent": solvent,
        "concentration_M": concentration,
        "temperature_K": temperature,
        "subsample_stride": stride_used,
        "frames_used": T_used,
        "effective_dt_ps": dt_ps_eff,
        "analysis_window_ps": float(tau[-1]),
        "tau_min_fit_ps": TAU_MIN_FIT_PS,
        "D_cation_(x1e-10_m2_s)": float(Dcat_1e10),
        "D_cation_A2_per_ps": float(Dcat_a2ps),
        "D_anion_(x1e-10_m2_s)": float(Danion_1e10) if Danion_1e10 is not None else None,
        "D_anion_A2_per_ps": float(Danion_a2ps) if Danion_a2ps is not None else None,
        "D_solvent_(x1e-10_m2_s)": float(Dsolv_1e10) if Dsolv_1e10 is not None else None,
        "D_solvent_A2_per_ps": float(Dsolv_a2ps) if Dsolv_a2ps is not None else None,
    }


def main(
    TARGETS: Sequence[Tuple[str, str, str, str, str, str, str, str]],
    EQ_TIME_PS: float,
    KNOWN_DT_PS: Optional[float],
    TARGET_FRAMES: int,
    TAU_MIN_FIT_PS: float,
    TAU_MAX_FIT_PS: float,
    N_WORKERS: int,
    PLOT_NCOLS: int,
    PARALLEL_MSD: bool,
    OUT_DIR: Path,
):
    if not TARGETS:
        print("⚠️ TARGETS is empty. Edit the TARGETS list with your topology/trajectory pairs.")
        return

    n_sys = len(TARGETS)
    ncols = PLOT_NCOLS
    nrows = (n_sys + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 15), squeeze=False)
    rows_out = []

    for k, target in enumerate(tqdm(TARGETS, desc="Processing trajectories", unit="traj")):
        ax = axes[k // ncols, k % ncols]
        row = process_target(
            target,
            EQ_TIME_PS,
            KNOWN_DT_PS,
            TARGET_FRAMES,
            TAU_MIN_FIT_PS,
            TAU_MAX_FIT_PS,
            N_WORKERS,
            PARALLEL_MSD,
            OUT_DIR,
            ax,
            k,
        )
        if row is not None:
            rows_out.append(row)

    for j in range(len(TARGETS), nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    fig.suptitle(
        f"MSD & Diffusion — ≥{EQ_TIME_PS:.0f} ps, fit {TAU_MIN_FIT_PS:.0f}–{TAU_MAX_FIT_PS:.0f} ps",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    png_path = OUT_DIR / "Diffusion_Coefficients.png"
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"\n✅ Saved plot → {png_path}")

    if rows_out:
        df = pd.DataFrame(rows_out)
        csv_path = OUT_DIR / "Diffusion_Coefficients.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Wrote CSV → {csv_path}\n")
        print(df.to_string(index=False))
    else:
        print("⚠️ No results produced.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute MSDs/diffusivities from GROMACS trajectories.")
    parser.add_argument("--out-dir", "-o", required=True, type=str, help="Output directory for MSDs/plots.")
    parser.add_argument("--tau-max-fit-ps", "-t", required=True, type=float, help="Maximum fitting time (ps).")
    parser.add_argument("--eq-time-ps", type=float, default=100.0, help="Equilibration time before analysis (ps).")
    parser.add_argument("--tau-min-fit-ps", type=float, default=1000.0, help="Minimum fit time (ps).")
    parser.add_argument(
        "--known-dt-ps",
        type=float,
        default=None,
        help="Base timestep (ps). If omitted, inferred from trajectory metadata.",
    )
    parser.add_argument("--n-workers", type=int, default=4, help="Workers for parallel MSD (if enabled).")
    parser.add_argument("--plot-ncols", type=int, default=2, help="Columns in subplot grid.")
    parser.add_argument("--parallel-msd", action="store_true", help="Enable parallel MSD calculation.")
    args = parser.parse_args()

    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    TAU_MIN_FIT_PS = args.tau_min_fit_ps
    TAU_MAX_FIT_PS = args.tau_max_fit_ps
    TARGET_FRAMES = int(TAU_MAX_FIT_PS - TAU_MIN_FIT_PS)

    # Format:
    # (topology_path, traj_path, system_name, cat_symbol, anion_symbol, solvent_symbol, concentration_M, temperature_K)
    # Keys must match cation_dict, anion_dict, solvent_dict.
    TARGETS = [
        (
            "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/OPLS_diffusivities/myfolder/t273_npt_run_napf6_dme.gro",
            "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/OPLS_diffusivities/myfolder/prod_t273_run_napf6_dme.xtc",
            "NaPF6 — DME",
            "Na",
            "PF6",
            "DME",
            "1M",
            "273K",
        ),
        (
            "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/OPLS_diffusivities/myfolder/t298_npt_run_napf6_dme.gro",
            "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/OPLS_diffusivities/myfolder/prod_t298_run_napf6_dme.xtc",
            "NaPF6 — DME",
            "Na",
            "PF6",
            "DME",
            "1M",
            "298K",
        ),
        (
            "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/OPLS_diffusivities/myfolder/t323_npt_run_napf6_dme.gro",
            "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/OPLS_diffusivities/myfolder/prod_t323_run_napf6_dme.xtc",
            "NaPF6 — DME",
            "Na",
            "PF6",
            "DME",
            "1M",
            "323K",
        ),
    ]

    main(
        TARGETS,
        args.eq_time_ps,
        args.known_dt_ps,
        TARGET_FRAMES,
        TAU_MIN_FIT_PS,
        TAU_MAX_FIT_PS,
        args.n_workers,
        args.plot_ncols,
        args.parallel_msd,
        OUT_DIR,
    )
