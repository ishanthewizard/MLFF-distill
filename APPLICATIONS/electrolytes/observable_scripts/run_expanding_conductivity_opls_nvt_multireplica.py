#!/usr/bin/env python
"""Expanding-window (running) Onsager + Nernst-Einstein ionic conductivity for the
OPLS **NVT** production replicas (GROMACS, 1 ps/frame, 20 ns).

For every system found under the given replica roots we run byteff2 `onsager_calc`
over a set of *expanding* averaging windows that all start at t = 0:
    0-1, 0-2, 0-3, ..., 0-20 ns.
This is a convergence / "running conductivity" study: it shows how the fitted sigma
settles as the displacement-averaging window grows.

Method (matches run_expanding_window_onsager_naotf_1M.py / the established byteff2
convention):
  * Native 1 ps/frame -> displacement dt = 1 ps (stride 1, dt_correction = 1.0), so
    onsager_calc's hardcoded `positions[200:]` drop and [50,200)-lag fit land at
    200 ps / 50-200 ps.
  * "Slice once, sweep windows": each trajectory's 0-max window is read & unwrapped
    ONCE (orthorhombic per-frame minimum image); every window is a plain array slice.
  * Species/masses/charges from the TPR (residue formal charge -> cation/anion/solvent).
  * eq_cut = 0: windows are literally 0->W ns (onsager_calc still drops its internal
    first 200 ps).
  * Per-system CSV is written immediately (intermediate data persisted incrementally).
  * MDAnalysis XTC offset-cache flock() hangs on CFS compute nodes -> bypass it and
    build offsets in memory (memory mdanalysis-cfs-flock-offset-bypass).

Env: /global/homes/y/yuejian/project/MLFF-distill/yuejian/envs/fairchemV2_new
     (has MDAnalysis 2.10 + byteff2 + ase)
"""
import argparse
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

REPLICA_ROOTS = [
    "/global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj/simulation_results/OPLS/nvt/replicas_1",
    "/global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj/simulation_results/OPLS/nvt/replicas_2",
    "/global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj/simulation_results/OPLS/nvt/replicas_3",
    "/global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj/simulation_results/OPLS/nvt/replicas_4",
]
DEFAULT_OUT = ("/global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj/general_analysis/"
               "is_conductivity_trustable/convergence/OPLS_running_conductivity")

WINDOWS_NS = list(range(1, 21))          # 0-1, 0-2, ..., 0-20 ns
LOAD_DT_PS = 1.0                         # displacement dt (native 1 ps -> stride 1)

ION_MAP = {"lipf6": ("Li", "PF6"), "napf6": ("Na", "PF6"), "naotf": ("Na", "OTf")}
SOLVENT_MAP = {"dme": "DME", "diglyme": "Diglyme", "tegdme": "TGDME", "tgdme": "TGDME", "pc": "PC"}
TEMP_TOKENS = {"273": 273, "298": 298, "323": 323}


# ──────────────────────────────────────────────────────────────────────────
def discover_tasks(roots):
    tasks = []
    for root in roots:
        root = Path(root)
        replica = root.name                      # replicas_1 ...
        for conc_dir in sorted(root.glob("*M")):
            conc = float(conc_dir.name.replace("M", ""))
            for sysdir in sorted(p for p in conc_dir.iterdir() if p.is_dir()):
                xtc = sysdir / "nvt.xtc"
                tpr = sysdir / "nvt.tpr"
                if not (xtc.exists() and tpr.exists()):
                    continue
                leaf = sysdir.name
                parts = leaf.split("_")
                salt = parts[0].lower()
                solvent = parts[1].lower() if len(parts) > 1 else "?"
                temp = 298
                if len(parts) >= 3 and parts[-1] in TEMP_TOKENS:
                    temp = TEMP_TOKENS[parts[-1]]
                cat, ani = ION_MAP.get(salt, ("?", "?"))
                sol = SOLVENT_MAP.get(solvent, solvent)
                sysid = f"{salt}_{solvent}_{conc:g}M_{temp}K"
                tasks.append(dict(replica=replica, sysdir=str(sysdir), xtc=str(xtc), tpr=str(tpr),
                                  salt=salt, cation=cat, anion=ani, solvent=sol,
                                  concentration_M=conc, temperature_K=temp,
                                  system_id=sysid, leaf=leaf))
    return tasks


def _load_unwrapped_gromacs(u, reorder_idx, frame_idxs):
    traj = u.trajectory
    T = len(frame_idxs); N = len(reorder_idx)
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
        disp -= box * np.round(disp / box)          # minimum image (per-frame box)
        positions[k] = positions[k - 1] + disp
        prev_raw = curr_raw
    return positions, vols


def compute_one(task):
    name = f"{task['replica']}/{task['system_id']}"
    out_int = Path(task["out_int"])
    csv_path = out_int / f"expanding_{task['replica']}__{task['system_id']}.csv"
    t0 = time.time()
    try:
        import pandas as pd
        import MDAnalysis as mda
        from MDAnalysis.coordinates.XDR import XDRBaseReader
        XDRBaseReader._load_offsets = lambda self: self._read_offsets(store=False)
        from byteff2.md_utils.onsager_conductivity import onsager_calc

        T_K = float(task["temperature_K"])
        u = mda.Universe(task["tpr"], task["xtc"])
        dt_ps = float(u.trajectory.dt)
        stride = max(1, int(round(LOAD_DT_PS / dt_ps)))
        actual_load_dt_ps = stride * dt_ps
        dt_correction = 1.0 / actual_load_dt_ps
        n_total = len(u.trajectory)
        avail_ns = n_total * dt_ps / 1000.0

        # species from residues (formal charge -> role)
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

        cat_lab, ani_lab, sol_lab = task["cation"], task["anion"], task["solvent"]
        species_order = [cat_lab, ani_lab, sol_lab]
        species_mass = {cat_lab: [float(m) for m in rn_first[cat_rn].atoms.masses],
                        ani_lab: [float(m) for m in rn_first[ani_rn].atoms.masses],
                        sol_lab: [float(m) for m in rn_first[sol_rn].atoms.masses]}
        species_number = {cat_lab: len(ag_cat.residues), ani_lab: len(ag_ani.residues),
                          sol_lab: len(ag_sol.residues)}
        species_charge = {cat_lab: float(rn_charge[cat_rn]), ani_lab: float(rn_charge[ani_rn]),
                          sol_lab: float(rn_charge[sol_rn])}

        windows = [w for w in WINDOWS_NS if w <= avail_ns + 1e-9]
        n_load = int(round(max(windows) * 1000.0 / actual_load_dt_ps))
        frame_idxs = list(range(0, min(n_load * stride, n_total), stride))
        print(f"[{name}] n_total={n_total} ({avail_ns:.1f} ns) dt={dt_ps:.3f} stride={stride} "
              f"species {cat_rn}(+{species_charge[cat_lab]:.0f},{species_number[cat_lab]}) "
              f"{ani_rn}({species_charge[ani_lab]:.0f},{species_number[ani_lab]}) "
              f"{sol_rn}({species_number[sol_lab]}); loading {len(frame_idxs)} frames", flush=True)

        positions, vols = _load_unwrapped_gromacs(u, reorder_idx, frame_idxs)

        rows = []
        for w in windows:
            n_w = min(int(round(w * 1000.0 / actual_load_dt_ps)), positions.shape[0])
            V_w = float(vols[:n_w].mean())
            res = onsager_calc(species_order=species_order, species_mass=species_mass,
                               species_number=species_number, species_charge=species_charge,
                               volume_angstrom3=V_w, viscosity_cP=1.0, T_K=T_K,
                               positions=positions[:n_w])
            sig_o = res["conductivity_onsager"] * dt_correction
            sig_ne = res["conductivity_NE"] * dt_correction
            Dself = [d * dt_correction for d in res["Dself_inf"]]
            rows.append({"source": "OPLS", "replica": task["replica"], "system_id": task["system_id"],
                         "salt": task["salt"], "cation": cat_lab, "anion": ani_lab, "solvent": sol_lab,
                         "concentration_M": task["concentration_M"], "temperature_K": T_K,
                         "window_ns": w, "n_frames_used": n_w, "load_dt_ps": actual_load_dt_ps,
                         "fit_window_ps": "50-200", "V_mean_A3": V_w,
                         "sigma_onsager_mS_cm": sig_o, "sigma_NE_mS_cm": sig_ne,
                         "sigma_onsager_uS_cm": sig_o * 1000.0, "sigma_NE_uS_cm": sig_ne * 1000.0,
                         "D_cat_1e10_m2s": Dself[0], "D_anion_1e10_m2s": Dself[1],
                         "D_solvent_1e10_m2s": Dself[2], "traj_path": task["xtc"]})
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"[{name}] DONE {time.time()-t0:.0f}s -> {csv_path.name} "
              f"(sigma_O 0-{windows[-1]}ns={rows[-1]['sigma_onsager_mS_cm']:.3f} mS/cm)", flush=True)
        return {"name": name, "csv": str(csv_path), "rows": rows, "error": ""}
    except Exception as e:
        print(f"[{name}] ERROR {type(e).__name__}: {e}\n{traceback.format_exc()}", flush=True)
        return {"name": name, "csv": "", "rows": [], "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=18)
    args = ap.parse_args()

    out = Path(args.out); (out / "intermediate").mkdir(parents=True, exist_ok=True)
    tasks = discover_tasks(REPLICA_ROOTS)
    for t in tasks:
        t["out_int"] = str(out / "intermediate")
    print(f"discovered {len(tasks)} OPLS NVT systems; workers={args.workers}; "
          f"windows(ns)={WINDOWS_NS}; displacement dt={LOAD_DT_PS} ps", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(compute_one, t): t for t in tasks}
        for fut in as_completed(futs):
            results.append(fut.result())

    import pandas as pd
    all_rows = [r for res in results for r in res["rows"]]
    errs = [res for res in results if res["error"]]
    if all_rows:
        df = pd.DataFrame(all_rows)
        combined = out / "conductivity_expanding_ALL.csv"
        df.to_csv(combined, index=False)
        print(f"\nwrote {combined} ({len(df)} rows, {df['system_id'].nunique()} systems x replicas)")
    if errs:
        print(f"\n{len(errs)} systems errored:")
        for e in errs:
            print(f"  {e['name']}: {e['error']}")
    print("ALL DONE.", flush=True)


if __name__ == "__main__":
    main()
