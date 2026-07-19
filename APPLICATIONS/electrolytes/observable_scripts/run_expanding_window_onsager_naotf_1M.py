#!/usr/bin/env python
"""Expanding-window Onsager (+ Nernst-Einstein) ionic conductivity convergence
for the two OPLS 1 M NaOTf NPT systems (GROMACS, 100 fs/frame).

For each system we run byteff2 `onsager_calc` over a set of *expanding* averaging
windows that all start at t = 0:  0-5, 0-10, ..., 0-60 ns.  This is a convergence
study — it shows how the fitted sigma settles as the displacement-averaging
window grows (byteff2's 50-200 ps Onsager slope typically needs ~8-9 ns of
averaging to converge; NE converges faster).

Method (matches run_conductivity_opls_1M.py / the established byteff2 convention):
  * 100 fs/frame trajectory subsampled to 1 ps/frame (stride 10) so onsager_calc's
    hardcoded `positions[200:]` drop and [50,200)-lag fit land at 200 ps / 50-200 ps.
  * "Slice once, sweep windows": each trajectory's 0-60 ns positions are read &
    unwrapped ONCE; every window is a plain array slice of that single load
    (verified to match per-window reloads — memory byteff2-onsager-window-convergence).
  * Species/masses/charges from the TPR (residue formal charge -> cation/anion/solvent).
  * NPT box: each window uses the mean box volume over ITS OWN frames.
  * eq_cut = 0: windows are literally 0->W ns (onsager_calc still drops its internal
    first 200 ps, so each window effectively averages [0.2, W] ns).
  * dt_correction = 1.0 (1 ps/frame == byteff2's internal 1-ps assumption).
  * MDAnalysis XTC offset-cache flock() hangs on CFS compute nodes -> bypass it and
    build offsets in memory (memory mdanalysis-cfs-flock-offset-bypass).

Env: /pscratch/sd/y/yuejian/envs/fairchemV2/bin/python  (MDAnalysis + torch + byteff2)
"""
import argparse
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

# (name, salt, cation_label, anion_label, solvent_label, color)
SYSTEMS = [
    ("npt_1M_naotf_diglyme", "NaOTf", "Na", "OTf", "Diglyme", "#2ca02c"),
    ("npt_1M_naotf_dme",     "NaOTf", "Na", "OTf", "DME",     "#d62728"),
]

WINDOWS_NS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]


def _load_unwrapped_gromacs(u, reorder_idx, frame_idxs):
    """Stream selected frames, unwrap via orthorhombic minimum image, return
    (T, N, 3) positions (Angstrom) and a per-frame box-volume array (T,)."""
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
        if k % 5000 == 0:
            print(f"    [{len(frame_idxs)}] loaded {k} frames", flush=True)
    return positions, vols


def compute_one(task):
    name = task["name"]
    t0 = time.time()
    try:
        import MDAnalysis as mda
        # CFS flock() for MDAnalysis's persistent XTC offset cache hangs on compute
        # nodes -> bypass the FileLock and build frame offsets in memory.
        from MDAnalysis.coordinates.XDR import XDRBaseReader
        XDRBaseReader._load_offsets = lambda self: self._read_offsets(store=False)

        from byteff2.md_utils.onsager_conductivity import onsager_calc

        tpr, xtc = task["tpr"], task["xtc"]
        load_dt_ps = task["load_dt_ps"]
        T_K = task["T_K"]
        viscosity_cP = task["viscosity_cP"]
        windows_ns = task["windows_ns"]
        cat_lab, ani_lab, sol_lab = task["cat_label"], task["ani_label"], task["sol_label"]

        u = mda.Universe(str(tpr), str(xtc))
        dt_ps = float(u.trajectory.dt)
        stride = max(1, int(round(load_dt_ps / dt_ps)))
        actual_load_dt_ps = stride * dt_ps
        dt_correction = 1.0 / actual_load_dt_ps
        n_total = len(u.trajectory)
        avail_ns = n_total * dt_ps / 1000.0

        # ── species from residues (formal charge -> role) ─────────────────────
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

        def one_mol_masses(rn):
            return [float(m) for m in rn_first[rn].atoms.masses]

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

        # ── slice once: load 0 -> max feasible window, at 1 ps/frame ──────────
        windows = [w for w in windows_ns if w <= avail_ns + 1e-9]
        n_load = int(round(max(windows) * 1000.0 / actual_load_dt_ps))
        frame_idxs = list(range(0, min(n_load * stride, n_total), stride))
        n_loaded = len(frame_idxs)

        print(f"[{name}] n_total={n_total} ({avail_ns:.1f} ns) dt={dt_ps:.3f} ps "
              f"stride={stride} -> {actual_load_dt_ps:.2f} ps/frame; loading {n_loaded} frames "
              f"(0-{max(windows)} ns); species {cat_rn}(+{species_charge[cat_lab]:.0f},"
              f"{species_number[cat_lab]}) {ani_rn}({species_charge[ani_lab]:.0f},"
              f"{species_number[ani_lab]}) {sol_rn}({species_number[sol_lab]}mol)", flush=True)

        positions, vols = _load_unwrapped_gromacs(u, reorder_idx, frame_idxs)
        print(f"[{name}] load done in {time.time()-t0:.0f}s; sweeping {len(windows)} windows", flush=True)

        rows = []
        for w in windows:
            n_w = min(int(round(w * 1000.0 / actual_load_dt_ps)), positions.shape[0])
            V_w = float(vols[:n_w].mean())
            res = onsager_calc(
                species_order=species_order, species_mass=species_mass,
                species_number=species_number, species_charge=species_charge,
                volume_angstrom3=V_w, viscosity_cP=viscosity_cP, T_K=T_K,
                positions=positions[:n_w],
            )
            sig_o = res["conductivity_onsager"] * dt_correction
            sig_ne = res["conductivity_NE"] * dt_correction
            Dself = [d * dt_correction for d in res["Dself_inf"]]
            rows.append({
                "name": name, "salt": task["salt"], "cation": cat_lab,
                "anion": ani_lab, "solvent": sol_lab, "concentration_M": 1.0,
                "T_K": T_K, "window_ns": w, "n_frames_used": n_w,
                "load_dt_ps": actual_load_dt_ps, "fit_window_ps": "50-200",
                "V_mean_A3": V_w,
                "sigma_onsager_mS_cm": sig_o, "sigma_NE_mS_cm": sig_ne,
                "sigma_onsager_uS_cm": sig_o * 1000.0, "sigma_NE_uS_cm": sig_ne * 1000.0,
                "D_cat_1e10_m2s": Dself[0], "D_anion_1e10_m2s": Dself[1],
                "D_solvent_1e10_m2s": Dself[2],
            })
            print(f"[{name}] window 0-{w:>4.0f} ns (n={n_w:6d})  "
                  f"sigma_Onsager={sig_o:8.4f}  sigma_NE={sig_ne:8.4f} mS/cm", flush=True)

        print(f"[{name}] DONE {time.time()-t0:.0f}s", flush=True)
        return {"name": name, "salt": task["salt"], "solvent": sol_lab,
                "color": task["color"], "rows": rows, "error": ""}
    except Exception as e:
        print(f"[{name}] ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}", flush=True)
        return {"name": name, "salt": task["salt"], "solvent": task["sol_label"],
                "color": task["color"], "rows": [], "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tpr-dir", default="/global/homes/y/yuejian/project/MLFF-distill/"
                    "m5024/distillation_project/results/opls_baseline/tpr_files_1M")
    ap.add_argument("--out", required=True)
    ap.add_argument("--load-dt-ps", type=float, default=1.0)
    ap.add_argument("--T-K", type=float, default=298.0)
    ap.add_argument("--viscosity-cP", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    tpr_dir = Path(args.tpr_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tasks = []
    for name, salt, cat, ani, sol, color in SYSTEMS:
        tpr = tpr_dir / f"{name}.tpr"
        xtc = tpr_dir / f"{name}.xtc"
        if not (tpr.exists() and xtc.exists()):
            print(f"SKIP {name}: missing tpr/xtc")
            continue
        tasks.append({"name": name, "salt": salt, "cat_label": cat, "ani_label": ani,
                      "sol_label": sol, "color": color, "T_K": args.T_K,
                      "load_dt_ps": args.load_dt_ps, "viscosity_cP": args.viscosity_cP,
                      "windows_ns": WINDOWS_NS, "tpr": str(tpr), "xtc": str(xtc)})

    print(f"discovered {len(tasks)} NaOTf systems; workers={args.workers}; "
          f"windows(ns)={WINDOWS_NS}; load_dt={args.load_dt_ps} ps, T={args.T_K} K", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(compute_one, t): t for t in tasks}
        for fut in as_completed(futs):
            results.append(fut.result())

    import pandas as pd
    all_rows = [r for res in results for r in res["rows"]]
    if not all_rows:
        print("No results produced.")
        return
    df = pd.DataFrame(all_rows)
    csv = out / "conductivity_expanding_naotf_1M.csv"
    df.to_csv(csv, index=False)
    print(f"\nwrote {csv} ({len(df)} rows)")

    # ── plot: fitted Onsager sigma vs expanding window size ───────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # experimental reference conductivity (mS/cm) from the completed full-run join
    exp_ms_cm = {"npt_1M_naotf_diglyme": 2.778, "npt_1M_naotf_dme": 1.233}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    order = [res for res in results if res["rows"]]
    for res in order:
        sub = df[df["name"] == res["name"]].sort_values("window_ns")
        lbl = f"{res['salt']}/{res['solvent']}"
        col = res["color"]
        axes[0].plot(sub["window_ns"], sub["sigma_onsager_mS_cm"], "o-",
                     color=col, lw=2, ms=6, label=lbl)
        axes[1].plot(sub["window_ns"], sub["sigma_NE_mS_cm"], "s-",
                     color=col, lw=2, ms=6, label=lbl)
        ex = exp_ms_cm.get(res["name"])
        if ex is not None:
            axes[0].axhline(ex, color=col, ls=":", lw=1.5, alpha=0.8)

    axes[0].set_title("Onsager (collective, byteff2) — fitted σ vs window\n"
                      "dotted = experiment")
    axes[1].set_title("Nernst-Einstein — fitted σ vs window")
    for ax in axes:
        ax.set_xlabel("Expanding averaging window  0 → W  (ns)")
        ax.set_ylabel("Ionic conductivity (mS/cm)")
        ax.grid(True, ls=":", alpha=0.6)
        ax.legend(fontsize=10)
        ax.set_xticks(WINDOWS_NS)
    fig.suptitle("OPLS 1 M NaOTf NPT (298 K) — expanding-window conductivity convergence "
                 "(fit 50-200 ps @ 1 ps/frame)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = out / "conductivity_expanding_naotf_1M.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")

    # focused single-panel Onsager-only figure (what was asked for)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for res in order:
        sub = df[df["name"] == res["name"]].sort_values("window_ns")
        ax.plot(sub["window_ns"], sub["sigma_onsager_mS_cm"], "o-",
                color=res["color"], lw=2, ms=7, label=f"{res['salt']}/{res['solvent']}")
        ex = exp_ms_cm.get(res["name"])
        if ex is not None:
            ax.axhline(ex, color=res["color"], ls=":", lw=1.5, alpha=0.8,
                       label=f"{res['salt']}/{res['solvent']} exp")
    ax.set_xlabel("Expanding averaging window  0 → W  (ns)")
    ax.set_ylabel("Onsager ionic conductivity (mS/cm)")
    ax.set_title("OPLS 1 M NaOTf NPT 298 K\nfitted Onsager σ vs expanding window")
    ax.set_xticks(WINDOWS_NS)
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(fontsize=9)
    fig.tight_layout()
    png2 = out / "conductivity_expanding_naotf_1M_onsager_only.png"
    fig.savefig(png2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png2}")

    print("\n" + df[["name", "window_ns", "n_frames_used", "V_mean_A3",
                     "sigma_onsager_mS_cm", "sigma_NE_mS_cm"]].to_string(index=False))


if __name__ == "__main__":
    main()
