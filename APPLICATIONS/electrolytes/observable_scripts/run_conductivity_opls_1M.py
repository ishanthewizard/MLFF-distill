#!/usr/bin/env python
"""Onsager / Nernst-Einstein ionic conductivity for the OPLS 1 M NPT set
(GROMACS, saved every 100 fs).

Why a dedicated driver (not eval.py / run_onsager_conductivity_gromacs):
  * eval.py's `conductivity` analysis explicitly skips GROMACS.
  * conductivity.compute.run_onsager_conductivity_gromacs assumes the native
    frame spacing is 1 ps (nstxout-compressed=1000) and applies NO dt
    correction.  These runs save every nstxout-compressed=100 steps at dt=1 fs
    => 100 fs / frame.  Feeding them raw would (a) make onsager_calc's hardcoded
    `positions[200:]` drop only 20 ps and its lag fit window [50,200) land at
    5-20 ps, and (b) scale D and sigma by 10x (each frame mis-read as 1 ps).
  * There are no .top/.itp/.mdp next to these trajectories (only .tpr/.xtc/.edr),
    so species/masses/charges are taken from the TPR via MDAnalysis.

What this does, per system:
  1. Universe(tpr, xtc); confirm dt = 0.1 ps.
  2. Group atoms by residue -> cation (formal charge +1), anion (-1),
     solvent (0).  Per-molecule masses/charges/counts from the TPR.
  3. Subsample to load_dt_ps = 1.0 ps (stride = 10) so onsager_calc's internal
     1-ps assumption + [50,200)-lag fit window are correct; dt_correction = 1.0.
  4. Drop the first eq_cut_ns (default 2 ns) as equilibration; onsager_calc
     internally drops a further 200 loaded frames (=200 ps).
  5. Unwrap PBC via per-frame orthorhombic minimum image between consecutive
     subsampled frames (all boxes are cubic, angles = 90; 1-ps steps in a dense
     liquid move << L/2, so this matches the ASE `_load_unwrapped` recipe).
  6. onsager_calc -> sigma_onsager (full, with ion-ion cross terms) + sigma_NE
     (Nernst-Einstein) in mS/cm, plus self-diffusivities (1e-10 m^2/s).

Runs on a SLURM CPU node (premium); one worker per system.

Env: /pscratch/sd/y/yuejian/envs/fairchemV2/bin/python (MDAnalysis + torch;
byteff2 imported from the repo submodule).
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
BYTEFF2_DIR = "/global/homes/y/yuejian/project/MLFF-distill/submodule/byteff2"
for p in (SCRIPT_ROOT, BYTEFF2_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# (name, salt, cation_label, anion_label, solvent_label)
SYSTEMS = [
    ("npt_1M_naotf_diglyme", "NaOTf", "Na", "OTf", "Diglyme"),
    ("npt_1M_naotf_dme",     "NaOTf", "Na", "OTf", "DME"),
    ("npt_1M_napf6_diglyme", "NaPF6", "Na", "PF6", "Diglyme"),
    ("npt_1M_napf6_dme",     "NaPF6", "Na", "PF6", "DME"),
    ("npt_1M_napf6_pc",      "NaPF6", "Na", "PF6", "PC"),
]


def _load_unwrapped_gromacs(u, reorder_idx, frame_idxs):
    """Stream selected frames, unwrap via orthorhombic minimum image, return
    (T, N, 3) positions in Angstrom and the mean box volume over those frames."""
    traj = u.trajectory
    T = len(frame_idxs)
    N = len(reorder_idx)
    positions = np.zeros((T, N, 3), dtype=np.float64)
    vols = np.zeros(T, dtype=np.float64)

    traj[frame_idxs[0]]
    box = traj.ts.dimensions[:3].astype(np.float64)
    vols[0] = box[0] * box[1] * box[2]
    prev_raw = u.atoms.positions[reorder_idx].astype(np.float64)
    positions[0] = prev_raw
    for k in range(1, T):
        traj[frame_idxs[k]]
        box = traj.ts.dimensions[:3].astype(np.float64)
        vols[k] = box[0] * box[1] * box[2]
        curr_raw = u.atoms.positions[reorder_idx].astype(np.float64)
        disp = curr_raw - prev_raw
        disp -= box * np.round(disp / box)          # minimum image (orthorhombic, per-frame box)
        positions[k] = positions[k - 1] + disp
        prev_raw = curr_raw
    return positions, float(vols.mean())


def compute_one(task):
    name = task["name"]
    t0 = time.time()
    try:
        import MDAnalysis as mda
        # The trajectories live on CFS (m5024), whose flock() hangs indefinitely
        # for MDAnalysis's persistent XTC offset cache — and m5024 is over quota
        # so the cache can't be written anyway.  Bypass the FileLock entirely and
        # build frame offsets in memory (one ~2 min scan per 6 GB file, no lock).
        from MDAnalysis.coordinates.XDR import XDRBaseReader
        XDRBaseReader._load_offsets = lambda self: self._read_offsets(store=False)

        from byteff2.md_utils.onsager_conductivity import onsager_calc

        tpr, xtc = task["tpr"], task["xtc"]
        load_dt_ps = task["load_dt_ps"]
        eq_cut_ns = task["eq_cut_ns"]
        T_K = task["T_K"]
        viscosity_cP = task["viscosity_cP"]

        u = mda.Universe(str(tpr), str(xtc))
        dt_ps = float(u.trajectory.dt)
        stride = max(1, int(round(load_dt_ps / dt_ps)))
        actual_load_dt_ps = stride * dt_ps
        n_total = len(u.trajectory)
        i_start = int(round(eq_cut_ns * 1000.0 / dt_ps))
        if i_start >= n_total:
            raise RuntimeError(f"eq_cut {eq_cut_ns} ns >= trajectory {n_total*dt_ps/1000:.1f} ns")

        # ── species from residues (charge -> role) ────────────────────────────
        res = u.residues
        rn_charge, rn_first = {}, {}
        for r in res:
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

        def one_mol_masses(rn):
            return [float(m) for m in rn_first[rn].atoms.masses]

        cat_lab, ani_lab, sol_lab = task["cat_label"], task["ani_label"], task["sol_label"]
        species_order = [cat_lab, ani_lab, sol_lab]
        species_mass = {cat_lab: one_mol_masses(cat_rn),
                        ani_lab: one_mol_masses(ani_rn),
                        sol_lab: one_mol_masses(sol_rn)}
        species_number = {cat_lab: len(ag_cat.residues),
                          ani_lab: len(ag_ani.residues),
                          sol_lab: len(ag_sol.residues)}
        species_charge = {cat_lab: float(rn_charge[cat_rn]),
                          ani_lab: float(rn_charge[ani_rn]),
                          sol_lab: float(rn_charge[sol_rn])}

        frame_idxs = list(range(i_start, n_total, stride))
        n_loaded = len(frame_idxs)
        if n_loaded <= 200 + 200:
            raise RuntimeError(f"not enough loaded frames {n_loaded}")

        print(f"[{name}] n_total={n_total} ({n_total*dt_ps/1000:.1f} ns) dt={dt_ps:.3f} ps "
              f"stride={stride} -> {actual_load_dt_ps:.2f} ps/frame; "
              f"eq_cut={eq_cut_ns} ns -> loading {n_loaded} frames; "
              f"species {cat_rn}(+{species_charge[cat_lab]:.0f},{species_number[cat_lab]}) "
              f"{ani_rn}({species_charge[ani_lab]:.0f},{species_number[ani_lab]}) "
              f"{sol_rn}({species_number[sol_lab]}x{len(species_mass[sol_lab])}at)", flush=True)

        positions, V_ang3 = _load_unwrapped_gromacs(u, reorder_idx, frame_idxs)

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
        dt_corr = 1.0 / actual_load_dt_ps
        sigma_onsager = result["conductivity_onsager"] * dt_corr
        sigma_NE = result["conductivity_NE"] * dt_corr
        Dself = [d * dt_corr for d in result["Dself_inf"]]   # order = species_order

        row = dict(task)
        row.pop("tpr", None); row.pop("xtc", None)
        row.update({
            "cation": cat_lab, "anion": ani_lab, "solvent": sol_lab,
            "N_cat": species_number[cat_lab], "N_anion": species_number[ani_lab],
            "N_solvent": species_number[sol_lab],
            "V_angstrom3": V_ang3,
            "traj_ns": n_total * dt_ps / 1000.0,
            "load_dt_ps": actual_load_dt_ps,
            "n_frames_loaded": n_loaded,
            "sigma_onsager_mS_cm": sigma_onsager,
            "sigma_NE_mS_cm": sigma_NE,
            "D_cat_1e10_m2s": Dself[0],
            "D_anion_1e10_m2s": Dself[1],
            "D_solvent_1e10_m2s": Dself[2],
            "wall_s": round(time.time() - t0, 1),
            "error": "",
        })
        print(f"[{name}] DONE {row['wall_s']}s  sigma_onsager={sigma_onsager:.4f}  "
              f"sigma_NE={sigma_NE:.4f} mS/cm  D_cat={Dself[0]:.3f} D_ani={Dself[1]:.3f} "
              f"D_sol={Dself[2]:.3f} (1e-10 m2/s)  V={V_ang3:.0f} A^3", flush=True)
        return row
    except Exception as e:
        row = dict(task)
        row.pop("tpr", None); row.pop("xtc", None)
        row.update({"sigma_onsager_mS_cm": None, "sigma_NE_mS_cm": None,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(), "wall_s": round(time.time() - t0, 1)})
        print(f"[{name}] ERROR: {row['error']}\n{row['traceback']}", flush=True)
        return row


COLS = ["name", "salt", "cation", "anion", "solvent", "concentration_M", "T_K",
        "N_cat", "N_anion", "N_solvent", "V_angstrom3", "traj_ns",
        "load_dt_ps", "eq_cut_ns", "n_frames_loaded",
        "sigma_onsager_mS_cm", "sigma_NE_mS_cm",
        "D_cat_1e10_m2s", "D_anion_1e10_m2s", "D_solvent_1e10_m2s",
        "wall_s", "error"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tpr-dir", default="/global/homes/y/yuejian/project/MLFF-distill/"
                    "m5024/distillation_project/results/opls_baseline/tpr_files_1M")
    ap.add_argument("--out", required=True)
    ap.add_argument("--load-dt-ps", type=float, default=1.0)
    ap.add_argument("--eq-cut-ns", type=float, default=2.0)
    ap.add_argument("--T-K", type=float, default=298.0)
    ap.add_argument("--viscosity-cP", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    tpr_dir = Path(args.tpr_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tasks = []
    for name, salt, cat, ani, sol in SYSTEMS:
        tpr = tpr_dir / f"{name}.tpr"
        xtc = tpr_dir / f"{name}.xtc"
        if not (tpr.exists() and xtc.exists()):
            print(f"SKIP {name}: missing tpr/xtc")
            continue
        tasks.append({"name": name, "salt": salt, "cat_label": cat, "ani_label": ani,
                      "sol_label": sol, "concentration_M": 1.0, "T_K": args.T_K,
                      "eq_cut_ns": args.eq_cut_ns, "load_dt_ps": args.load_dt_ps,
                      "viscosity_cP": args.viscosity_cP,
                      "tpr": str(tpr), "xtc": str(xtc)})

    print(f"discovered {len(tasks)} systems; workers={args.workers}; "
          f"load_dt={args.load_dt_ps} ps, eq_cut={args.eq_cut_ns} ns, T={args.T_K} K", flush=True)

    all_csv = out / "conductivity_all.csv"
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex, open(all_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        writer.writeheader(); fh.flush()
        futures = {ex.submit(compute_one, t): t for t in tasks}
        for fut in as_completed(futures):
            r = fut.result()
            rows.append(r); writer.writerow(r); fh.flush()

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(all_csv, index=False)
    ok = df[df["error"] == ""] if "error" in df else df
    print(f"\nwrote {all_csv} ({len(df)} rows, {len(ok)} ok, {len(df)-len(ok)} failed)")

    if len(ok):
        # bar plot of sigma_onsager & sigma_NE per system
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            okp = ok.copy()
            okp["label"] = okp["salt"] + "/" + okp["solvent"]
            okp = okp.sort_values("label")
            x = np.arange(len(okp)); wbar = 0.38
            fig, ax = plt.subplots(figsize=(9, 5.5))
            ax.bar(x - wbar/2, okp["sigma_onsager_mS_cm"], wbar, label="Onsager (collective)", color="#1f77b4")
            ax.bar(x + wbar/2, okp["sigma_NE_mS_cm"], wbar, label="Nernst-Einstein", color="#ff7f0e")
            ax.set_xticks(x); ax.set_xticklabels(okp["label"], rotation=20, ha="right")
            ax.set_ylabel("ionic conductivity (mS/cm)")
            ax.set_title("OPLS 1 M NPT — ionic conductivity (byteff2), full length, eq_cut 2 ns")
            ax.grid(axis="y", alpha=0.25); ax.legend()
            for xi, (so, sne) in enumerate(zip(okp["sigma_onsager_mS_cm"], okp["sigma_NE_mS_cm"])):
                ax.text(xi - wbar/2, so, f"{so:.2f}", ha="center", va="bottom", fontsize=8)
                ax.text(xi + wbar/2, sne, f"{sne:.2f}", ha="center", va="bottom", fontsize=8)
            fig.tight_layout()
            fig.savefig(out / "conductivity_bar.png", dpi=140)
            plt.close(fig)
            print(f"wrote {out/'conductivity_bar.png'}")
        except Exception:
            traceback.print_exc()

        cols = ["name", "salt", "solvent", "traj_ns", "V_angstrom3",
                "sigma_onsager_mS_cm", "sigma_NE_mS_cm",
                "D_cat_1e10_m2s", "D_anion_1e10_m2s", "D_solvent_1e10_m2s"]
        print("\n" + ok[cols].to_string(index=False))


if __name__ == "__main__":
    main()
