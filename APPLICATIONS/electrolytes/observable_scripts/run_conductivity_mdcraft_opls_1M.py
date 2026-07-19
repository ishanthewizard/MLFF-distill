#!/usr/bin/env python
"""mdcraft collective (Onsager) ionic conductivity for the OPLS 1 M NPT set
(GROMACS, 100 fs/frame) — the collective Einstein-Helfand estimator, as an
independent cross-check on the byteff2 Onsager values.

Ports the repo's established recipe (conductivity/compute.py::
run_onsager_conductivity_mdcraft, which is ASE-only) to GROMACS input:

  1. Universe(tpr, xtc) with the CFS flock bypass; species from TPR formal charge.
  2. Subsample 100 fs -> load_dt_ps (default 1 ps, stride 10); drop eq_cut_ns.
  3. Unwrap ALL atoms (per-frame orthorhombic minimum image), remove the
     mass-weighted system-COM drift (barycentric frame).
  4. Ion SITES = every cation atom (monoatomic Na) + each anion's central heavy
     atom (max-mass atom in the anion residue == P for PF6, S for OTf).
  5. Build an in-memory MDAnalysis universe of those sites and run
     mdcraft.analysis.transport.Onsager -> collective L_ij -> kappa.
     Fit window on the collective MSD matches byteff2 here: lags 50-200 ps
     (fit_start_ns=0.05, fit_stop_ns=0.20).  NO dt_correction (mdcraft uses the
     real dt), unlike the byteff2 path.
  6. Save the per-system collective MSD curves (++, +-, --) so the fit window can
     be changed later WITHOUT reloading the trajectory.

Env: /global/homes/y/yuejian/project/MLFF-distill/yuejian/envs/fairchemV2_new
     (this is the ONLY env here with mdcraft importable).
"""
import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

SCRIPT_ROOT = "/global/homes/y/yuejian/project/MLFF-distill/APPLICATIONS/electrolytes/observable_scripts"
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)

# (name, salt, cation_label, anion_label, solvent_label)
SYSTEMS = [
    ("npt_1M_naotf_diglyme", "NaOTf", "Na", "OTf", "Diglyme"),
    ("npt_1M_naotf_dme",     "NaOTf", "Na", "OTf", "DME"),
    ("npt_1M_napf6_diglyme", "NaPF6", "Na", "PF6", "Diglyme"),
    ("npt_1M_napf6_dme",     "NaPF6", "Na", "PF6", "DME"),
    ("npt_1M_napf6_pc",      "NaPF6", "Na", "PF6", "PC"),
]

# mdcraft unit conversions (from conductivity/compute.py)
_KAPPA_TO_SI = 1.0e19   # mdcraft conductivity unit -> S/m
_SI_TO_USCM  = 1.0e4    # S/m -> uS/cm
_D_TO_CM2_S  = 1.0e-4   # A^2/ps -> cm^2/s
_NDIM        = 3


def _load_unwrapped_gromacs(u, reorder_idx, frame_idxs):
    """Stream frames, unwrap via per-frame orthorhombic minimum image.
    Returns (positions [T,N,3] Angstrom, boxes [T,3] Angstrom)."""
    traj = u.trajectory
    T, N = len(frame_idxs), len(reorder_idx)
    positions = np.zeros((T, N, 3), dtype=np.float64)
    boxes = np.zeros((T, 3), dtype=np.float64)

    traj[frame_idxs[0]]
    boxes[0] = traj.ts.dimensions[:3].astype(np.float64)
    prev_raw = u.atoms.positions[reorder_idx].astype(np.float64)
    positions[0] = prev_raw
    for k in range(1, T):
        traj[frame_idxs[k]]
        box = traj.ts.dimensions[:3].astype(np.float64)
        boxes[k] = box
        curr_raw = u.atoms.positions[reorder_idx].astype(np.float64)
        disp = curr_raw - prev_raw
        disp -= box * np.round(disp / box)
        positions[k] = positions[k - 1] + disp
        prev_raw = curr_raw
        if k % 10000 == 0:
            print(f"    loaded {k}/{T} frames", flush=True)
    return positions, boxes


def compute_one(task):
    name = task["name"]
    t0 = time.time()
    try:
        import MDAnalysis as mda
        from MDAnalysis.coordinates.XDR import XDRBaseReader
        XDRBaseReader._load_offsets = lambda self: self._read_offsets(store=False)
        from MDAnalysis.coordinates.memory import MemoryReader
        from mdcraft.analysis.transport import Onsager

        tpr, xtc = task["tpr"], task["xtc"]
        load_dt_ps = task["load_dt_ps"]
        eq_cut_ns = task["eq_cut_ns"]
        T_K = task["T_K"]
        z_cat, z_anion = task["z_cat"], task["z_anion"]
        fit_start_ns, fit_stop_ns = task["fit_start_ns"], task["fit_stop_ns"]
        out_dir = Path(task["out_dir"])
        cat_lab, ani_lab, sol_lab = task["cat_label"], task["ani_label"], task["sol_label"]

        max_traj_ns = task.get("max_traj_ns", None)
        u = mda.Universe(str(tpr), str(xtc))
        dt_ps = float(u.trajectory.dt)
        stride = max(1, int(round(load_dt_ps / dt_ps)))
        actual_load_dt_ps = stride * dt_ps
        n_total = len(u.trajectory)
        if max_traj_ns is not None:
            n_total = min(n_total, int(round(max_traj_ns * 1000.0 / dt_ps)))
        i_start = int(round(eq_cut_ns * 1000.0 / dt_ps))
        if i_start >= n_total:
            raise RuntimeError(f"eq_cut {eq_cut_ns} ns >= traj {n_total*dt_ps/1000:.1f} ns")

        # ── species by residue formal charge ─────────────────────────────────
        rn_charge, rn_first = {}, {}
        for r in u.residues:
            rn = r.resname
            if rn not in rn_charge:
                rn_charge[rn] = int(round(float(sum(r.atoms.charges))))
                rn_first[rn] = r
        cat_rn = [rn for rn, c in rn_charge.items() if c > 0]
        ani_rn = [rn for rn, c in rn_charge.items() if c < 0]
        sol_rn = [rn for rn, c in rn_charge.items() if c == 0]
        if not (len(cat_rn) == 1 and len(ani_rn) == 1 and len(sol_rn) == 1):
            raise RuntimeError(f"ambiguous species from charges: {rn_charge}")
        cat_rn, ani_rn, sol_rn = cat_rn[0], ani_rn[0], sol_rn[0]

        ag_cat = u.select_atoms(f"resname {cat_rn}")
        ag_ani = u.select_atoms(f"resname {ani_rn}")
        ag_sol = u.select_atoms(f"resname {sol_rn}")
        reorder_idx = np.concatenate([ag_cat.indices, ag_ani.indices, ag_sol.indices]).astype(int)
        if len(reorder_idx) != len(u.atoms):
            raise RuntimeError(f"reorder covers {len(reorder_idx)} of {len(u.atoms)} atoms")

        n_cat = len(ag_cat.residues)
        n_ani = len(ag_ani.residues)
        cat_atoms_per_mol = len(rn_first[cat_rn].atoms)
        ani_atoms_per_mol = len(rn_first[ani_rn].atoms)
        if cat_atoms_per_mol != 1:
            raise RuntimeError(f"cation {cat_rn} not monoatomic ({cat_atoms_per_mol} atoms)")

        # central heavy atom of the anion = max-mass atom in one anion molecule
        ani_masses = np.array([float(m) for m in rn_first[ani_rn].atoms.masses])
        central_off = int(np.argmax(ani_masses))
        central_name = list(rn_first[ani_rn].atoms.names)[central_off]

        # ── ion-site columns in the REORDERED [cat | ani | sol] layout ────────
        na_cols = list(range(n_cat))                                  # monoatomic cations
        an_cols = [n_cat + j * ani_atoms_per_mol + central_off for j in range(n_ani)]
        ion_cols = np.array(na_cols + an_cols, dtype=int)
        n_at = n_cat + n_ani

        frame_idxs = list(range(i_start, n_total, stride))
        n_loaded = len(frame_idxs)
        print(f"[{name}] n_total={n_total} ({n_total*dt_ps/1000:.1f} ns) dt={dt_ps:.3f} "
              f"stride={stride}->{actual_load_dt_ps:.2f} ps/frame; eq_cut={eq_cut_ns} ns -> "
              f"{n_loaded} frames; {cat_rn}(+{rn_charge[cat_rn]},{n_cat}) "
              f"{ani_rn}({rn_charge[ani_rn]},{n_ani},central={central_name}@{central_off}) "
              f"{sol_rn}; ion sites={n_at}", flush=True)

        # ── load + unwrap all atoms; remove system-COM drift ─────────────────
        positions, boxes = _load_unwrapped_gromacs(u, reorder_idx, frame_idxs)
        masses_re = u.atoms.masses[reorder_idx].astype(np.float64)
        R = np.einsum("tnj,n->tj", positions, masses_re) / masses_re.sum()
        drift = np.linalg.norm(R - R[0], axis=1)
        pos_c = positions - R[:, None, :]
        ion_pos_c = pos_c[:, ion_cols, :]
        mean_dims = boxes.mean(axis=0)
        print(f"[{name}] loaded in {time.time()-t0:.0f}s; COM drift net={drift[-1]:.2f} "
              f"max={drift.max():.2f} A; box~{mean_dims.round(2)}", flush=True)

        # ── in-memory universe of ion sites -> mdcraft Onsager ───────────────
        u2 = mda.Universe.empty(n_at, n_residues=n_at,
                                atom_resindex=np.arange(n_at), trajectory=True)
        u2.add_TopologyAttr("name", ["NA"] * n_cat + ["AN"] * n_ani)
        u2.load_new(ion_pos_c.astype(np.float32), format=MemoryReader)

        ons = Onsager([u2.atoms[:n_cat], u2.atoms[n_cat:]], groupings="atoms",
                      temperature=T_K, charges=[z_cat, z_anion],
                      dimensions=mean_dims, dt=actual_load_dt_ps, unwrap=False,
                      center=False, fft=True, verbose=False)
        ons.run()

        t = np.asarray(ons.results.times)                 # ps
        dt_lag_ns = (t[1] - t[0]) / 1000.0
        n_lag = len(t)
        s = max(1, int(round(fit_start_ns / dt_lag_ns)))
        e = min(n_lag, int(round(fit_stop_ns / dt_lag_ns)))
        if e <= s:
            s, e = max(1, n_lag // 5), n_lag

        ons.calculate_transport_coefficients(start=s, stop=e, scale="linear")
        ons.calculate_conductivity()
        try:
            ons.calculate_transference_numbers()
            t_i = [float(x) for x in ons.results.transference_numbers[0]]
        except Exception:
            t_i = [np.nan, np.nan]

        kappa_uScm = float(ons.results.conductivity[0] * _KAPPA_TO_SI * _SI_TO_USCM)
        kappa_mScm = kappa_uScm / 1.0e3
        D_i = [float(x) * _D_TO_CM2_S for x in ons.results.D_i[0]]   # cm^2/s

        # save the collective MSD curves for later re-fitting (no reload needed)
        cross = np.asarray(ons.results.msd_cross)[:, 0] * (2 * _NDIM)  # (3, n_lag): ++,+-,--
        np.savez_compressed(out_dir / f"{name}_mdcraft_msd.npz",
                            times_ps=t, msd_cross=cross,
                            pairs=np.asarray(ons.results.pairs, dtype=object),
                            fit_start_ns=fit_start_ns, fit_stop_ns=fit_stop_ns,
                            fit_lag_idx=np.array([s, e]),
                            load_dt_ps=actual_load_dt_ps, mean_dims=mean_dims)

        print(f"[{name}] DONE {time.time()-t0:.0f}s  kappa_mdcraft={kappa_mScm:.4f} mS/cm "
              f"({kappa_uScm:.0f} uS/cm)  fit lags {s}-{e}/{n_lag} "
              f"({s*dt_lag_ns:.2f}-{e*dt_lag_ns:.2f} ns)  D_cat={D_i[0]:.2e} D_an={D_i[1]:.2e} cm2/s",
              flush=True)

        return {"name": name, "salt": task["salt"], "cation": cat_lab, "anion": ani_lab,
                "solvent": sol_lab, "concentration_M": 1.0, "T_K": T_K,
                "eq_cut_ns": eq_cut_ns, "load_dt_ps": actual_load_dt_ps,
                "fit_start_ns": fit_start_ns, "fit_stop_ns": fit_stop_ns,
                "N_cat": n_cat, "N_anion": n_ani, "anion_central": central_name,
                "V_angstrom3": float(np.prod(mean_dims)), "traj_ns": n_total * dt_ps / 1000.0,
                "n_frames_loaded": n_loaded, "com_drift_A": float(drift[-1]),
                "sigma_mdcraft_mS_cm": kappa_mScm, "sigma_mdcraft_uS_cm": kappa_uScm,
                "D_cat_cm2_s": D_i[0], "D_anion_cm2_s": D_i[1],
                "t_cat": t_i[0], "t_anion": t_i[1],
                "wall_s": round(time.time() - t0, 1), "error": ""}
    except Exception as e:
        print(f"[{name}] ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}", flush=True)
        return {"name": name, "salt": task["salt"], "solvent": task["sol_label"],
                "sigma_mdcraft_mS_cm": None, "error": f"{type(e).__name__}: {e}",
                "wall_s": round(time.time() - t0, 1)}


COLS = ["name", "salt", "cation", "anion", "solvent", "concentration_M", "T_K",
        "eq_cut_ns", "load_dt_ps", "fit_start_ns", "fit_stop_ns", "N_cat", "N_anion",
        "anion_central", "V_angstrom3", "traj_ns", "n_frames_loaded", "com_drift_A",
        "sigma_mdcraft_mS_cm", "sigma_mdcraft_uS_cm", "D_cat_cm2_s", "D_anion_cm2_s",
        "t_cat", "t_anion", "wall_s", "error"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tpr-dir", default="/global/homes/y/yuejian/project/MLFF-distill/"
                    "m5024/distillation_project/results/opls_baseline/tpr_files_1M")
    ap.add_argument("--out", required=True)
    ap.add_argument("--load-dt-ps", type=float, default=1.0)
    ap.add_argument("--eq-cut-ns", type=float, default=2.0)
    ap.add_argument("--fit-start-ns", type=float, default=0.05)   # match byteff2 lags 50-200 ps
    ap.add_argument("--fit-stop-ns", type=float, default=0.20)
    ap.add_argument("--T-K", type=float, default=298.0)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--max-traj-ns", type=float, default=None,
                    help="cap trajectory length (smoke test)")
    ap.add_argument("--only", default=None, help="substring filter on system name (smoke test)")
    args = ap.parse_args()

    tpr_dir = Path(args.tpr_dir)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    tasks = []
    for name, salt, cat, ani, sol in SYSTEMS:
        if args.only and args.only not in name:
            continue
        tpr, xtc = tpr_dir / f"{name}.tpr", tpr_dir / f"{name}.xtc"
        if not (tpr.exists() and xtc.exists()):
            print(f"SKIP {name}: missing tpr/xtc"); continue
        tasks.append({"name": name, "salt": salt, "cat_label": cat, "ani_label": ani,
                      "sol_label": sol, "concentration_M": 1.0, "T_K": args.T_K,
                      "eq_cut_ns": args.eq_cut_ns, "load_dt_ps": args.load_dt_ps,
                      "fit_start_ns": args.fit_start_ns, "fit_stop_ns": args.fit_stop_ns,
                      "z_cat": 1.0, "z_anion": -1.0, "out_dir": str(out),
                      "max_traj_ns": args.max_traj_ns,
                      "tpr": str(tpr), "xtc": str(xtc)})

    print(f"discovered {len(tasks)} systems; workers={args.workers}; load_dt={args.load_dt_ps} ps, "
          f"eq_cut={args.eq_cut_ns} ns, fit {args.fit_start_ns}-{args.fit_stop_ns} ns, T={args.T_K} K",
          flush=True)

    all_csv = out / "conductivity_mdcraft_all.csv"
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex, open(all_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        writer.writeheader(); fh.flush()
        futs = {ex.submit(compute_one, t): t for t in tasks}
        for fut in as_completed(futs):
            r = fut.result(); rows.append(r); writer.writerow(r); fh.flush()

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(all_csv, index=False)
    ok = df[df["error"] == ""] if "error" in df else df
    print(f"\nwrote {all_csv} ({len(df)} rows, {len(ok)} ok)")
    if len(ok):
        print("\n" + ok[["name", "salt", "solvent", "traj_ns", "sigma_mdcraft_mS_cm",
                         "anion_central", "com_drift_A"]].to_string(index=False))


if __name__ == "__main__":
    main()
