#!/usr/bin/env python
"""Apples-to-apples byteff2 Onsager conductivity for the m5250 OPLS naotf_dme 1M
boxes (old NPT + one NVT replica), using the SAME method as run_conductivity_opls_1M.py
(species from TPR, subsample to 1 ps, min-image unwrap, onsager_calc, 50-200 ps fit).
These are 1 ps-native (nstxout-compressed=1000) so stride=1, dt_correction=1.0.

Purpose: confirm whether the m5250 boxes give a higher sigma than the m5024 NPT box
(0.90 mS/cm) under identical analysis -> isolates the physical (force-field/density)
difference from any analysis difference.
"""
import sys, time
from pathlib import Path
import numpy as np

for p in ("/global/homes/y/yuejian/project/MLFF-distill/APPLICATIONS/electrolytes/observable_scripts",
          "/global/homes/y/yuejian/project/MLFF-distill/submodule/byteff2"):
    if p not in sys.path:
        sys.path.insert(0, p)

SYSTEMS = [
    ("m5250_old_npt", "/global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj/"
        "simulation_results/OPLS/old/npt/naotf_dme_1M_298K/npt.tpr",
        "/global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj/"
        "simulation_results/OPLS/old/npt/naotf_dme_1M_298K/npt.xtc"),
    ("m5250_nvt_rep1", "/global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj/"
        "simulation_results/OPLS/nvt/replicas_1/1M/naotf_dme_1M/nvt.tpr",
        "/global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj/"
        "simulation_results/OPLS/nvt/replicas_1/1M/naotf_dme_1M/nvt.xtc"),
]


def main():
    import MDAnalysis as mda
    from MDAnalysis.coordinates.XDR import XDRBaseReader
    XDRBaseReader._load_offsets = lambda self: self._read_offsets(store=False)
    from byteff2.md_utils.onsager_conductivity import onsager_calc

    load_dt_ps, eq_cut_ns, T_K = 1.0, 2.0, 298.0
    print(f"{'system':16s} {'traj_ns':>7s} {'dt_ps':>5s} {'nfr':>6s} {'box_nm':>7s} "
          f"{'sig_onsager':>11s} {'sig_NE':>8s}")
    for name, tpr, xtc in SYSTEMS:
        try:
            u = mda.Universe(tpr, xtc)
            dt_ps = float(u.trajectory.dt)
            stride = max(1, int(round(load_dt_ps / dt_ps)))
            actual = stride * dt_ps
            n_total = len(u.trajectory)
            i_start = int(round(eq_cut_ns * 1000.0 / dt_ps))
            # species from TPR charges
            rn_c, rn_f = {}, {}
            for r in u.residues:
                if r.resname not in rn_c:
                    rn_c[r.resname] = int(round(float(sum(r.atoms.charges)))); rn_f[r.resname] = r
            cat = [k for k, v in rn_c.items() if v > 0][0]
            ani = [k for k, v in rn_c.items() if v < 0][0]
            sol = [k for k, v in rn_c.items() if v == 0][0]
            ag = {s: u.select_atoms(f"resname {s}") for s in (cat, ani, sol)}
            reorder = np.concatenate([ag[cat].indices, ag[ani].indices, ag[sol].indices]).astype(int)
            one = lambda rn: [float(m) for m in rn_f[rn].atoms.masses]
            so = ["Na", "OTf", "DME"]
            sm = {"Na": one(cat), "OTf": one(ani), "DME": one(sol)}
            sn = {"Na": len(ag[cat].residues), "OTf": len(ag[ani].residues), "DME": len(ag[sol].residues)}
            sc = {"Na": float(rn_c[cat]), "OTf": float(rn_c[ani]), "DME": float(rn_c[sol])}
            fis = list(range(i_start, n_total, stride))
            # unwrap (min-image) + mean vol
            traj = u.trajectory
            T, N = len(fis), len(reorder)
            pos = np.zeros((T, N, 3)); vols = np.zeros(T)
            traj[fis[0]]; b = traj.ts.dimensions[:3].astype(float); vols[0] = np.prod(b)
            prev = u.atoms.positions[reorder].astype(float); pos[0] = prev
            for k in range(1, T):
                traj[fis[k]]; b = traj.ts.dimensions[:3].astype(float); vols[k] = np.prod(b)
                cur = u.atoms.positions[reorder].astype(float); d = cur - prev
                d -= b * np.round(d / b); pos[k] = pos[k-1] + d; prev = cur
            V = float(vols.mean())
            res = onsager_calc(species_order=so, species_mass=sm, species_number=sn,
                               species_charge=sc, volume_angstrom3=V, viscosity_cP=1.0,
                               T_K=T_K, positions=pos)
            corr = 1.0 / actual
            so_o, so_ne = res["conductivity_onsager"]*corr, res["conductivity_NE"]*corr
            print(f"{name:16s} {n_total*dt_ps/1000:7.1f} {dt_ps:5.2f} {T:6d} {V**(1/3)/10:7.3f} "
                  f"{so_o:11.4f} {so_ne:8.4f}   (resmap {cat}/{ani}/{sol}, N {sn['Na']}/{sn['OTf']}/{sn['DME']})",
                  flush=True)
        except Exception as e:
            import traceback; print(f"{name}: ERROR {e}\n{traceback.format_exc()}", flush=True)
    print("\nreference m5024 NPT (byteff2, full 66 ns): sigma_onsager=0.9004  sigma_NE=11.8723 mS/cm  exp=1.23")


if __name__ == "__main__":
    main()
