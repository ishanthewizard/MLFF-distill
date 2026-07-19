#!/usr/bin/env python
"""Multi-replica Onsager / Nernst-Einstein conductivity for the OPLS GROMACS NVT set.

eval.py's `conductivity` analysis only supports ASE .traj (it explicitly skips
GROMACS), so this standalone driver calls the GROMACS conductivity function
`conductivity.compute.run_onsager_conductivity_gromacs` directly, once per
(replica, system) trajectory, and aggregates replicas per system.

Layout walked (LOCAL pscratch copy staged with *.top / *.itp / nvt.mdp / nvt.tpr /
nvt.xtc, incl. the ion itp na/li/pf6 which are absent from the CFS dirs):
    <root>/replicas_<N>/<conc>/<system>/nvt.{xtc,tpr}

Each system dir must contain: nvt.tpr, nvt.xtc, one *.top, and the moleculetype
*.itp for every molecule listed in the .top's [ molecules ] block (solvent + ions).

Outputs (to --output):
  conductivity_all_replicas.csv  — one row per replica-trajectory (written live)
  conductivity_per_system.csv    — replica mean/std of sigma per system

Native GROMACS spacing here is 1 ps (dt=1fs, nstxout-compressed=1000) which
matches what onsager_calc assumes, so no dt correction is applied. The whole
unwrapped trajectory is used (byteff2 handles the fit window internally).

Run (fairchemV2_new has byteff2 + MDAnalysis):
  python run_conductivity_opls_nvt_multireplica.py \
      --root /pscratch/sd/y/yuejian/opls_nvt_msd_local \
      --output <.../OPLS/analysis/nvt/conductivity_<ts>> --workers 18
"""
import argparse
import csv
import os
import sys
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_ROOT = "/global/homes/y/yuejian/project/MLFF-distill/APPLICATIONS/electrolytes/observable_scripts"
sys.path.insert(0, SCRIPT_ROOT)
sys.path.insert(0, os.path.join(SCRIPT_ROOT, "conductivity"))

REPLICAS = ["replicas_1", "replicas_2", "replicas_3", "replicas_4"]
ION_MAP = {"lipf6": ("Li", "PF6"), "napf6": ("Na", "PF6"), "naotf": ("Na", "OTf")}
SOLVENT_MAP = {"dme": "DME", "diglyme": "Diglyme", "tegdme": "TGDME", "pc": "PC"}


def discover(root):
    tasks = []
    root = Path(root)
    for rlabel in REPLICAS:
        rroot = root / rlabel
        if not rroot.is_dir():
            continue
        for conc_dir in sorted(p for p in rroot.glob("*/") if p.is_dir()):
            conc_label = conc_dir.name
            for sd in sorted(p for p in conc_dir.glob("*/") if p.is_dir()):
                if not (sd / "nvt.xtc").exists() or not (sd / "nvt.tpr").exists():
                    continue
                parts = sd.name.split("_")
                if len(parts) < 2:
                    continue
                salt, solvent = parts[0], parts[1]
                third = parts[2] if len(parts) > 2 else ""
                if salt not in ION_MAP or solvent not in SOLVENT_MAP:
                    continue
                conc_M = float(conc_label[:-1]) if conc_label.endswith("M") else float(conc_label)
                T_K = float(third) if third.isdigit() else 298.0
                name = f"{salt}_{solvent}_{conc_label}_{int(T_K)}K"
                tasks.append({
                    "system": name,
                    "replica": rlabel.replace("replicas_", "replica_"),
                    "cation": ION_MAP[salt][0],
                    "anion": ION_MAP[salt][1],
                    "solvent": SOLVENT_MAP[solvent],
                    "concentration_M": conc_M,
                    "T_K": T_K,
                    "sys_dir": str(sd),
                })
    return tasks


def run_one(task):
    from conductivity import compute
    try:
        res = compute.run_onsager_conductivity_gromacs(task["sys_dir"], "nvt", T_K=task["T_K"])
        order = res["species_order"]           # [solvent, cation, anion] (‑> .top order)
        Dself = res["Dself_1e10_m2s"]
        row = dict(task)
        row.update({
            "sigma_onsager_mS_cm": res["sigma_onsager_mS_cm"],
            "sigma_NE_mS_cm": res["sigma_NE_mS_cm"],
            "D_solvent_1e10_m2s": Dself[0] if len(Dself) > 0 else None,
            "D_cat_1e10_m2s": Dself[1] if len(Dself) > 1 else None,
            "D_anion_1e10_m2s": Dself[2] if len(Dself) > 2 else None,
            "V_angstrom3": res["V_angstrom3"],
            "n_frames": res["n_frames"],
            "T_K_used": res["T_K"],
            "species_order": "|".join(order),
            "error": "",
        })
        return row
    except Exception as e:
        row = dict(task)
        row.update({
            "sigma_onsager_mS_cm": None, "sigma_NE_mS_cm": None,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        })
        return row


COLS = ["system", "replica", "cation", "anion", "solvent", "concentration_M",
        "T_K", "T_K_used", "sigma_onsager_mS_cm", "sigma_NE_mS_cm",
        "D_cat_1e10_m2s", "D_anion_1e10_m2s", "D_solvent_1e10_m2s",
        "V_angstrom3", "n_frames", "species_order", "sys_dir", "error"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=18)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tasks = discover(args.root)
    n_sys = len(set(t["system"] for t in tasks))
    print(f"discovered {len(tasks)} replica-trajectories across {n_sys} systems "
          f"(workers={args.workers})", flush=True)

    all_csv = out / "conductivity_all_replicas.csv"
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex, \
            open(all_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        writer.writeheader(); fh.flush()
        futures = {ex.submit(run_one, t): t for t in tasks}
        done = 0
        for fut in as_completed(futures):
            r = fut.result(); rows.append(r); writer.writerow(r); fh.flush()
            done += 1
            if r["error"]:
                print(f"[{done}/{len(tasks)}] {r['system']} {r['replica']}: ERROR {r['error']}", flush=True)
            else:
                print(f"[{done}/{len(tasks)}] {r['system']} {r['replica']}: "
                      f"sigma_onsager={r['sigma_onsager_mS_cm']:.4f} "
                      f"sigma_NE={r['sigma_NE_mS_cm']:.4f} mS/cm", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    ok = df[df["error"] == ""]
    if len(ok):
        agg = ok.groupby(["system", "cation", "anion", "solvent", "concentration_M", "T_K"]).agg(
            sigma_onsager_mean_mS_cm=("sigma_onsager_mS_cm", "mean"),
            sigma_onsager_std_mS_cm=("sigma_onsager_mS_cm", "std"),
            sigma_NE_mean_mS_cm=("sigma_NE_mS_cm", "mean"),
            sigma_NE_std_mS_cm=("sigma_NE_mS_cm", "std"),
            n_replicas=("sigma_onsager_mS_cm", "count"),
        ).reset_index()
        agg.to_csv(out / "conductivity_per_system.csv", index=False)
        print(f"\nwrote {out/'conductivity_per_system.csv'} ({len(agg)} systems)")
    print(f"wrote {all_csv} ({len(df)} rows)")
    print(f"failures: {int((df['error'] != '').sum())}/{len(df)}")


if __name__ == "__main__":
    main()
