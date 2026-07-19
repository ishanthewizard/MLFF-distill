#!/usr/bin/env python3
"""Aggregate per-system density.csv files from an eval.py density run into
parity-ready CSVs for the PAINN electrolytes_data exp_06_ood set.

Outputs (next to the run dir's parent, i.e. in ood_results/):
  density_parity_exp_06_ood_per_run.csv   one row per NPT trajectory (10 rows)
  density_parity_exp_06_ood.csv           one row per physical system (5 rows),
                                          PAINN density averaged over duplicate runs,
                                          matched to experimental density.

Usage:
  python build_density_parity_exp_06_ood.py <run_dir>
"""
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
EXP_CSV = Path(
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5024/distillation_project/experiment_data/cleaned_version/density_comparison.csv"
)

SALT_REF = {"naotf": "NaOTf", "napf6": "NaPF6", "lipf6": "LiPF6"}
# reference CSV solvent spellings
SOLV_REF = {"dme": "DME", "diglyme": "DEGDME", "tgdme": "TEGDME", "pc": "PC"}


def load_cfg():
    spec = importlib.util.spec_from_file_location(
        "cfg", _HERE / "config_density_painn_exp_06_ood.py")
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def main():
    run_dir = Path(sys.argv[1])
    cfg = load_cfg()
    # map system name -> metadata
    meta = {s["name"]: s for s in cfg.SYSTEMS}

    exp = pd.read_csv(EXP_CSV)

    def exp_lookup(salt_ref, solv_ref, conc, temp):
        m = exp[(exp["Salt"] == salt_ref) & (exp["Solvent"] == solv_ref) &
                (np.isclose(exp["Conc (M)"], conc)) &
                (np.isclose(exp["T (K)"], temp))]
        if m.empty:
            return {}
        r = m.iloc[0]
        return {
            "Exp_density (g/cm3)": r.get("Exp_density (g/cm3)"),
            "OPLS_density (g/cm3)": r.get("OPLS_density (g/cm3)"),
            "uma_density (g/cm3)": r.get("uma_density (g/cm3)"),
            "orb_density (g/cm3)": r.get("orb_density (g/cm3)"),
        }

    # ── per-run rows from each density.csv ───────────────────────────────────
    rows = []
    for csv in sorted(run_dir.glob("*/density.csv")):
        df = pd.read_csv(csv)
        df = df[df["Model"] == "npt"]
        for _, r in df.iterrows():
            name = r["System"]
            s = meta.get(name)
            if s is None:
                continue
            salt, solv = s["salt"], s["solvent_tok"]
            conc, temp = s["concentration_M"], s["temperature_K"]
            salt_ref, solv_ref = SALT_REF[salt], SOLV_REF[solv]
            e = exp_lookup(salt_ref, solv_ref, conc, temp)
            sim = float(r["Density (g/cm³)"])
            exp_d = e.get("Exp_density (g/cm3)")
            err = (100.0 * (sim - exp_d) / exp_d
                   if exp_d is not None and pd.notna(exp_d) else np.nan)
            rows.append({
                "System": name,
                "Salt": salt_ref, "Solvent": solv_ref,
                "Conc (M)": conc, "T (K)": temp,
                "PAINN_density (g/cm3)": sim,
                "PAINN_std (g/cm3)": float(r["Std (g/cm³)"]),
                "Exp_density (g/cm3)": exp_d,
                "Error (%)": err,
                "OPLS_density (g/cm3)": e.get("OPLS_density (g/cm3)"),
                "uma_density (g/cm3)": e.get("uma_density (g/cm3)"),
                "orb_density (g/cm3)": e.get("orb_density (g/cm3)"),
            })
    per_run = pd.DataFrame(rows).sort_values(["Salt", "Solvent"]).reset_index(drop=True)

    # ── aggregate duplicate physical systems (avg PAINN density across runs) ──
    keys = ["Salt", "Solvent", "Conc (M)", "T (K)"]
    agg_rows = []
    for kvals, g in per_run.groupby(keys):
        dvals = g["PAINN_density (g/cm3)"].to_numpy()
        first = g.iloc[0]
        exp_d = first["Exp_density (g/cm3)"]
        mean_d = float(np.mean(dvals))
        err = (100.0 * (mean_d - exp_d) / exp_d
               if pd.notna(exp_d) else np.nan)
        agg_rows.append({
            "Salt": kvals[0], "Solvent": kvals[1],
            "Conc (M)": kvals[2], "T (K)": kvals[3],
            "PAINN_density (g/cm3)": mean_d,
            "PAINN_run_std (g/cm3)": float(np.std(dvals, ddof=1)) if len(dvals) > 1 else 0.0,
            "n_runs": int(len(dvals)),
            "Exp_density (g/cm3)": exp_d,
            "Error (%)": err,
            "OPLS_density (g/cm3)": first["OPLS_density (g/cm3)"],
            "uma_density (g/cm3)": first["uma_density (g/cm3)"],
            "orb_density (g/cm3)": first["orb_density (g/cm3)"],
        })
    agg = pd.DataFrame(agg_rows).sort_values(["Salt", "Solvent"]).reset_index(drop=True)

    out_root = run_dir.parent
    per_run_path = out_root / "density_parity_exp_06_ood_per_run.csv"
    agg_path = out_root / "density_parity_exp_06_ood.csv"
    per_run.to_csv(per_run_path, index=False)
    agg.to_csv(agg_path, index=False)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("=== per-run (10 trajectories) ===")
    print(per_run.to_string(index=False))
    print("\n=== per-system (avg over duplicate runs) ===")
    print(agg.to_string(index=False))
    print(f"\nwrote: {per_run_path}")
    print(f"wrote: {agg_path}")


if __name__ == "__main__":
    main()
