#!/usr/bin/env python3
"""Expanding-window (running) Onsager + Nernst-Einstein ionic conductivity for the
student **PAINN FP32 In-distribution NPT_langevin** replicas (ASE .traj).

For every `.traj` found under the given `npt_replica_*` roots we compute the byteff2
Onsager + NE conductivity over expanding averaging windows all starting at t = 0:
    0-1, 0-2, ..., 0-20 ns.
Convergence / "running conductivity" study, sibling of the OPLS GROMACS driver.

Method (matches run_uma_expanding_window_conductivity.py / byteff2 convention):
  * PAINN electrolytes_data runs are saved every 100 fs (dt_fs=100; the dir-name
    "..ns_..fs" token is unreliable, tdump[ps]=0.1 in every .fennol.yaml). We
    subsample to a displacement dt of 1 ps (stride 10) so onsager_calc's internal
    first-200-frame drop and [50,200)-lag fit land at 200 ps / 50-200 ps.
  * "Slice once, sweep windows": each traj's 0-max window is read & PBC-unwrapped
    ONCE (ASE find_mic per frame); every window is a plain array slice.
  * NPT box (fluctuating V): volume fed to onsager_calc is the mean over each window.
  * eq_cut = 0: windows are literally 0->W ns.
  * Per-system CSV written immediately (intermediate data persisted incrementally).

Env: /global/homes/y/yuejian/project/MLFF-distill/yuejian/envs/fairchemV2_new
"""
import argparse
import re
import sys
import time
import traceback
import importlib.util
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

OBS = Path("/global/homes/y/yuejian/project/MLFF-distill/APPLICATIONS/electrolytes/observable_scripts")


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_cc = _load_module("cond_compute", OBS / "conductivity" / "compute.py")
onsager_calc               = _cc.onsager_calc
direct_groups_from_species = _cc.direct_groups_from_species
cation_dict, anion_dict, solvent_dict = _cc.cation_dict, _cc.anion_dict, _cc.solvent_dict
find_mic                   = _cc.find_mic
AseTraj                    = _cc._AseTraj

REPLICA_ROOTS = [
    "/global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj/simulation_results/PAINN/"
    "electrolytes_data/simulation/FP32_simulation/In_distribution_exp/NPT_langevin/npt_replica_0",
    "/global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj/simulation_results/PAINN/"
    "electrolytes_data/simulation/FP32_simulation/In_distribution_exp/NPT_langevin/npt_replica_1",
    "/global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj/simulation_results/PAINN/"
    "electrolytes_data/simulation/FP32_simulation/In_distribution_exp/NPT_langevin/npt_replica_2",
    "/global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj/simulation_results/PAINN/"
    "electrolytes_data/simulation/FP32_simulation/In_distribution_exp/NPT_langevin/npt_replica_3",
]
DEFAULT_OUT = ("/global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj/general_analysis/"
               "is_conductivity_trustable/convergence/student_running_conductivity")

WINDOWS_NS = list(range(1, 21))   # 0-1, ..., 0-20 ns
DT_FS      = 100.0                # native saved-frame spacing (tdump 0.1 ps)
LOAD_DT_PS = 1.0                  # displacement dt after subsampling
Z_CAT, Z_ANION = 1.0, -1.0

ION_MAP = {"lipf6": ("Li", "PF6"), "napf6": ("Na", "PF6"), "naotf": ("Na", "OTf")}
SOLVENT_MAP = {"dme": "DME", "diglyme": "Diglyme", "tegdme": "TGDME", "tgdme": "TGDME", "pc": "PC"}


def discover_tasks(roots):
    tasks = []
    for root in roots:
        root = Path(root)
        replica = root.name                              # npt_replica_0 ...
        for traj in sorted(root.glob("*/*/*.traj")):
            outer = traj.parent.parent.name              # naotf_dme, napf6_tgdme ...
            inner = traj.parent.name                     # npt_0_1M_298K_2ns_100fs ...
            salt = outer.split("_")[0].lower()
            solvent_tok = "_".join(outer.split("_")[1:]).lower()
            m_c = re.search(r"((?:\d+_)?\d+)M", inner)
            m_t = re.search(r"(\d+)K", inner)
            if not (m_c and salt in ION_MAP and solvent_tok in SOLVENT_MAP):
                print(f"SKIP (unparsed): {traj}")
                continue
            conc = float(m_c.group(1).replace("_", "."))
            temp = int(m_t.group(1)) if m_t else 298
            cat, ani = ION_MAP[salt]
            sol = SOLVENT_MAP[solvent_tok]
            sysid = f"{salt}_{solvent_tok}_{conc:g}M_{temp}K"
            tasks.append(dict(replica=replica, traj=str(traj), salt=salt, cation=cat, anion=ani,
                              solvent=sol, concentration_M=conc, temperature_K=temp, system_id=sysid))
    return tasks


def load_unwrapped(traj_path, reorder_idx, stride, n_loaded):
    frame_idxs = list(range(0, n_loaded * stride, stride))
    T = len(frame_idxs); N = len(reorder_idx)
    pos = np.zeros((T, N, 3), dtype=np.float64)
    vol = np.zeros(T, dtype=np.float64)
    reorder = np.asarray(reorder_idx)
    with AseTraj(str(traj_path)) as trj:
        f = trj[frame_idxs[0]]
        prev = f.get_positions()[reorder]; pos[0] = prev; vol[0] = f.get_volume()
        for k in range(1, T):
            f = trj[frame_idxs[k]]
            curr = f.get_positions()[reorder]
            disp, _ = find_mic(curr - prev, f.get_cell(), pbc=f.get_pbc())
            pos[k] = pos[k - 1] + disp
            prev = curr; vol[k] = f.get_volume()
    return pos, vol


def compute_one(task):
    name = f"{task['replica']}/{task['system_id']}"
    out_int = Path(task["out_int"])
    csv_path = out_int / f"expanding_{task['replica']}__{task['system_id']}.csv"
    t0 = time.time()
    try:
        import pandas as pd
        cat, ani, sol = task["cation"], task["anion"], task["solvent"]
        T_K = float(task["temperature_K"])

        dt_ps = DT_FS / 1000.0
        stride = max(1, round(LOAD_DT_PS / dt_ps))            # 10
        actual_load_dt_ps = stride * dt_ps                    # 1.0 ps
        dt_correction = 1.0 / actual_load_dt_ps               # 1.0

        with AseTraj(task["traj"]) as trj:
            n_total_raw = len(trj)
            f0 = trj[0]
            symbols0 = f0.get_chemical_symbols()
            masses0 = f0.get_masses()
        avail_ns = n_total_raw * dt_ps / 1000.0

        cat_g = direct_groups_from_species(symbols0, cation_dict[cat])
        ani_g = direct_groups_from_species(symbols0, anion_dict[ani])
        sol_g = direct_groups_from_species(symbols0, solvent_dict[sol])
        covered = sum(len(g) for g in cat_g + ani_g + sol_g)
        if covered != len(symbols0):
            raise RuntimeError(f"atom mismatch: {covered} grouped vs {len(symbols0)} total")
        reorder_idx = ([int(i) for g in cat_g for i in g] + [int(i) for g in ani_g for i in g] +
                       [int(i) for g in sol_g for i in g])
        species_order = [cat, ani, sol]
        species_mass = {cat: [float(masses0[i]) for i in cat_g[0]],
                        ani: [float(masses0[i]) for i in ani_g[0]],
                        sol: [float(masses0[i]) for i in sol_g[0]]}
        species_number = {cat: len(cat_g), ani: len(ani_g), sol: len(sol_g)}
        species_charge = {cat: float(Z_CAT), ani: float(Z_ANION), sol: 0.0}

        windows = [w for w in WINDOWS_NS if w <= avail_ns + 1e-9]
        n_load = int(round(max(windows) * 1000.0 / actual_load_dt_ps))
        if n_load * stride > n_total_raw:
            n_load = n_total_raw // stride
        print(f"[{name}] n_total={n_total_raw} ({avail_ns:.1f} ns) stride={stride} "
              f"N_cat={len(cat_g)} N_ani={len(ani_g)} N_sol={len(sol_g)}; loading {n_load} frames", flush=True)

        pos, vol = load_unwrapped(task["traj"], reorder_idx, stride, n_load)

        rows = []
        for w in windows:
            n_w = min(int(round(w * 1000.0 / actual_load_dt_ps)), pos.shape[0])
            V_w = float(vol[:n_w].mean())
            res = onsager_calc(species_order=species_order, species_mass=species_mass,
                               species_number=species_number, species_charge=species_charge,
                               volume_angstrom3=V_w, viscosity_cP=1.0, T_K=T_K, positions=pos[:n_w])
            sig_o = res["conductivity_onsager"] * dt_correction
            sig_ne = res["conductivity_NE"] * dt_correction
            Dself = [d * dt_correction for d in res["Dself_inf"]]
            rows.append({"source": "PAINN", "replica": task["replica"], "system_id": task["system_id"],
                         "salt": task["salt"], "cation": cat, "anion": ani, "solvent": sol,
                         "concentration_M": task["concentration_M"], "temperature_K": T_K,
                         "window_ns": w, "n_frames_used": n_w, "load_dt_ps": actual_load_dt_ps,
                         "fit_window_ps": "50-200", "V_mean_A3": V_w,
                         "sigma_onsager_mS_cm": sig_o, "sigma_NE_mS_cm": sig_ne,
                         "sigma_onsager_uS_cm": sig_o * 1000.0, "sigma_NE_uS_cm": sig_ne * 1000.0,
                         "D_cat_1e10_m2s": Dself[0], "D_anion_1e10_m2s": Dself[1],
                         "D_solvent_1e10_m2s": Dself[2], "traj_path": task["traj"]})
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
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out); (out / "intermediate").mkdir(parents=True, exist_ok=True)
    tasks = discover_tasks(REPLICA_ROOTS)
    for t in tasks:
        t["out_int"] = str(out / "intermediate")
    print(f"discovered {len(tasks)} PAINN NPT_langevin systems; workers={args.workers}; "
          f"windows(ns)={WINDOWS_NS}; dt_fs={DT_FS} -> displacement dt={LOAD_DT_PS} ps", flush=True)

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
