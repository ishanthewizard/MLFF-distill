#!/usr/bin/env python
"""Independent cross-check of run_conductivity_opls_1M.py's naotf_dme result,
using ONLY unmodified building blocks from conductivity/compute.py.

It computes sigma_onsager / sigma_NE for npt_1M_naotf_dme.xtc two ways that
differ ONLY in the PBC unwrap, feeding BOTH into the SAME established
conductivity.compute.onsager_calc:

  (A) established unwrap: MDAnalysis `find_mic` between consecutive subsampled
      frames — the exact minimum-image routine conductivity/compute.py::
      _load_unwrapped uses for the ASE path.
  (B) driver unwrap: the hand-rolled orthorhombic min-image from
      run_conductivity_opls_1M.py.

Everything else (subsample 100 fs->1 ps, species-from-TPR, eq_cut, dt_correction,
onsager_calc, fit window) is identical to the driver.  If (A) == (B) == driver,
the driver introduced no error and the numbers are physical.

No code in conductivity/ is modified; this only imports and composes it.

Env: /pscratch/sd/y/yuejian/envs/fairchemV2/bin/python  (MDAnalysis + byteff2)
"""
import argparse
import sys
import time
from pathlib import Path
import numpy as np

SCRIPT_ROOT = "/global/homes/y/yuejian/project/MLFF-distill/APPLICATIONS/electrolytes/observable_scripts"
BYTEFF2_DIR = "/global/homes/y/yuejian/project/MLFF-distill/submodule/byteff2"
for p in (SCRIPT_ROOT, BYTEFF2_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_species_from_tpr(u, cat_lab, ani_lab, sol_lab):
    rn_charge, rn_first = {}, {}
    for r in u.residues:
        rn = r.resname
        if rn not in rn_charge:
            rn_charge[rn] = int(round(float(sum(r.atoms.charges))))
            rn_first[rn] = r
    cat_rn = [rn for rn, c in rn_charge.items() if c > 0][0]
    ani_rn = [rn for rn, c in rn_charge.items() if c < 0][0]
    sol_rn = [rn for rn, c in rn_charge.items() if c == 0][0]
    ag_cat = u.select_atoms(f"resname {cat_rn}")
    ag_ani = u.select_atoms(f"resname {ani_rn}")
    ag_sol = u.select_atoms(f"resname {sol_rn}")
    reorder_idx = np.concatenate([ag_cat.indices, ag_ani.indices, ag_sol.indices]).astype(int)
    one = lambda rn: [float(m) for m in rn_first[rn].atoms.masses]
    species_order = [cat_lab, ani_lab, sol_lab]
    species_mass = {cat_lab: one(cat_rn), ani_lab: one(ani_rn), sol_lab: one(sol_rn)}
    species_number = {cat_lab: len(ag_cat.residues), ani_lab: len(ag_ani.residues),
                      sol_lab: len(ag_sol.residues)}
    species_charge = {cat_lab: float(rn_charge[cat_rn]), ani_lab: float(rn_charge[ani_rn]),
                      sol_lab: float(rn_charge[sol_rn])}
    return reorder_idx, species_order, species_mass, species_number, species_charge


def load_unwrapped(u, reorder_idx, frame_idxs, method):
    """method='findmic' (established MDAnalysis) or 'minimage' (driver's)."""
    from MDAnalysis.lib.distances import minimize_vectors  # backend of find_mic
    traj = u.trajectory
    T, N = len(frame_idxs), len(reorder_idx)
    pos = np.zeros((T, N, 3), dtype=np.float64)
    vols = np.zeros(T)
    traj[frame_idxs[0]]
    box = traj.ts.dimensions.copy()
    vols[0] = box[0] * box[1] * box[2]
    prev = u.atoms.positions[reorder_idx].astype(np.float64)
    pos[0] = prev
    for k in range(1, T):
        traj[frame_idxs[k]]
        box = traj.ts.dimensions.copy()
        vols[k] = box[0] * box[1] * box[2]
        curr = u.atoms.positions[reorder_idx].astype(np.float64)
        d = curr - prev
        if method == "findmic":
            d = minimize_vectors(d, box)                 # MDAnalysis MIC (established)
        else:
            b = box[:3].astype(np.float64)
            d = d - b * np.round(d / b)                  # driver's orthorhombic min-image
        pos[k] = pos[k - 1] + d
        prev = curr
    return pos, float(vols.mean())


def run(u, reorder_idx, so, sm, sn, sc, frame_idxs, method, load_dt_ps, T_K, visc):
    from conductivity.compute import onsager_calc      # UNMODIFIED established core
    pos, V = load_unwrapped(u, reorder_idx, frame_idxs, method)
    res = onsager_calc(species_order=so, species_mass=sm, species_number=sn,
                       species_charge=sc, volume_angstrom3=V, viscosity_cP=visc,
                       T_K=T_K, positions=pos)
    corr = 1.0 / load_dt_ps
    return (res["conductivity_onsager"] * corr, res["conductivity_NE"] * corr, V)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tpr", default="/global/homes/y/yuejian/project/MLFF-distill/m5024/"
                    "distillation_project/results/opls_baseline/tpr_files_1M/npt_1M_naotf_dme.tpr")
    ap.add_argument("--xtc", default="/global/homes/y/yuejian/project/MLFF-distill/m5024/"
                    "distillation_project/results/opls_baseline/tpr_files_1M/npt_1M_naotf_dme.xtc")
    ap.add_argument("--load-dt-ps", type=float, default=1.0)
    ap.add_argument("--eq-cut-ns", type=float, default=2.0)
    ap.add_argument("--max-traj-ns", type=float, default=None, help="cap length (default full)")
    ap.add_argument("--T-K", type=float, default=298.0)
    ap.add_argument("--viscosity-cP", type=float, default=1.0)
    args = ap.parse_args()

    import MDAnalysis as mda
    from MDAnalysis.coordinates.XDR import XDRBaseReader
    XDRBaseReader._load_offsets = lambda self: self._read_offsets(store=False)

    u = mda.Universe(args.tpr, args.xtc)
    dt_ps = float(u.trajectory.dt)
    stride = max(1, int(round(args.load_dt_ps / dt_ps)))
    actual = stride * dt_ps
    n_total = len(u.trajectory)
    if args.max_traj_ns is not None:
        n_total = min(n_total, int(round(args.max_traj_ns * 1000.0 / dt_ps)))
    i_start = int(round(args.eq_cut_ns * 1000.0 / dt_ps))
    frame_idxs = list(range(i_start, n_total, stride))

    reorder_idx, so, sm, sn, sc = load_species_from_tpr(u, "Na", "OTf", "DME")
    print(f"naotf_dme: dt={dt_ps:.3f} ps stride={stride}->{actual:.2f} ps/frame; "
          f"eq_cut={args.eq_cut_ns} ns -> {len(frame_idxs)} frames "
          f"({(n_total-i_start)*dt_ps/1000:.1f} ns); species {so} counts "
          f"{[sn[s] for s in so]}", flush=True)

    for method, label in [("findmic", "(A) established find_mic MIC"),
                          ("minimage", "(B) driver orthorhombic min-image")]:
        t0 = time.time()
        so_o, so_ne, V = run(u, reorder_idx, so, sm, sn, sc, frame_idxs, method,
                             actual, args.T_K, args.viscosity_cP)
        print(f"  {label:38s}  sigma_onsager={so_o:9.4f}  sigma_NE={so_ne:9.4f} mS/cm  "
              f"V={V:.0f} A^3  ({time.time()-t0:.0f}s)", flush=True)

    print("\nreference (run_conductivity_opls_1M.py, full 68 ns): "
          "sigma_onsager=0.9004  sigma_NE=11.8723 mS/cm")


if __name__ == "__main__":
    main()
