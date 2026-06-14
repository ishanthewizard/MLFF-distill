#!/usr/bin/env python3
"""CLI runner: compute MSD + diffusivity and dump plots using compute.py + plot.py.

Target JSON format (same as old msds_calculation_batch.py):
  [[traj_path, slug, cat_symbol, anion_symbol, solvent_symbol, ...], ...]

Extra columns beyond index 4 are ignored (conc, temp, etc.).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from compute import run_msd_analysis, save_msd_pickle, save_diffusivity_csv
from plot import plot_msd, plot_convergence


def main():
    parser = argparse.ArgumentParser(
        description="Compute MSD + diffusivity and dump plots (new compute/plot API)."
    )
    parser.add_argument("--out-dir", "-o", required=True, help="Output directory.")
    parser.add_argument("--known-dt-ps", "-d", type=float, default=0.01,
                        help="Frame spacing in ps (default 0.01 = 10 fs).")
    parser.add_argument("--eq-cut-ns", type=float, default=0.1,
                        help="Equilibration cut from trajectory start (ns).")
    parser.add_argument("--tau-min-fit-ns", type=float, default=1.0,
                        help="Lower bound of linear fit (ns).")
    parser.add_argument("--fit-pct", type=float, default=0.8,
                        help="Upper fit bound = fit_pct * max_lag.")
    parser.add_argument("--max-traj-ns", type=float, default=None,
                        help="Truncate trajectory to this duration (ns) before analysis.")
    parser.add_argument("--model-label", type=str, default="UMA",
                        help="Label used in plot titles and file names.")
    parser.add_argument("--targets-json", type=str, default=None,
                        help="JSON targets string. Falls back to MSD_TARGETS_JSON env var.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets_json_str = args.targets_json or os.environ.get("MSD_TARGETS_JSON")
    if not targets_json_str:
        print("Error: --targets-json or MSD_TARGETS_JSON env var required.", file=sys.stderr)
        sys.exit(1)

    targets = [tuple(t) for t in json.loads(targets_json_str)]
    dt_fs = args.known_dt_ps * 1000.0

    rows = []
    for entry in targets:
        traj_path, slug, cat, anion, solvent = entry[0], entry[1], entry[2], entry[3], entry[4]
        print(f"\n=== {slug} ===")
        print(f"  traj : {traj_path}")

        result = run_msd_analysis(
            traj_path=traj_path,
            cat_symbol=cat,
            anion_symbol=anion,
            solvent_symbol=solvent,
            dt_fs=dt_fs,
            eq_cut_ns=args.eq_cut_ns,
            fit_pct=args.fit_pct,
            tau_min_fit_ns=args.tau_min_fit_ns,
            max_traj_ns=args.max_traj_ns,
        )

        conv = result["convergence"]
        print(f"  fit window : [{conv['tau_min_fit_ns']:.2f}, {conv['tau_max_fit_ns']:.2f}] ns")
        print(f"  D_{cat:<6} = {conv['D_cat_final']:.4f} ×10⁻¹⁰ m²/s")
        if not np.isnan(conv["D_ani_final"]):
            print(f"  D_{anion:<6} = {conv['D_ani_final']:.4f} ×10⁻¹⁰ m²/s")
        if not np.isnan(conv["D_sol_final"]):
            print(f"  D_{solvent:<6} = {conv['D_sol_final']:.4f} ×10⁻¹⁰ m²/s")

        sys_dir = out_dir / slug
        save_msd_pickle(result, slug, sys_dir)
        p1 = plot_msd(result, slug, args.model_label, sys_dir)
        p2 = plot_convergence(result, slug, args.model_label, sys_dir)
        print(f"  Saved : {p1}")
        print(f"  Saved : {p2}")

        rows.append({
            "system":        slug,
            "model":         args.model_label,
            "D_cat":         conv["D_cat_final"],
            "D_anion":       conv["D_ani_final"],
            "D_solvent":     conv["D_sol_final"],
            "tau_min_fit_ns": conv["tau_min_fit_ns"],
            "tau_max_fit_ns": conv["tau_max_fit_ns"],
            "eq_cut_ns":     args.eq_cut_ns,
            "fit_pct":       args.fit_pct,
        })

    save_diffusivity_csv(rows, out_dir)
    print(f"\nDone. Summary CSV + per-system plots written to {out_dir}")


if __name__ == "__main__":
    main()
