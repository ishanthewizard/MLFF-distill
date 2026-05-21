#!/usr/bin/env python
"""Density comparison: UMA teacher, Micro student, Original student vs experiment."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

COMPUTED_CSV = Path("/global/homes/y/yuejian/project/MLFF-distill/yuejian/3d_turbulence/rdf/density_comparison.csv")
EXP_CSV      = Path("/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/csv/All_data - exp_with_density.csv")
OUT_DIR      = Path("/global/homes/y/yuejian/project/MLFF-distill/APPLICATIONS/electrolytes/plots")

# Map system name → (cation, anion, solvent, concentration) for lookup in exp CSV
SYSTEM_LOOKUP = {
    "NaPF6/DME 0.1M":  ("Na", "PF6", "DME", "0.1M"),
    "NaOTf/DME 0.1M":  ("Na", "OTf", "DME", "0.1M"),
    "LiPF6/DME 0.5M":  ("Li", "PF6", "DME", "0.5M"),
    "NaPF6/DME 0.5M":  ("Na", "PF6", "DME", "0.5M"),
    "NaOTf/DME 1M":    ("Na", "OTf", "DME", "1M"),
    "NaPF6/DME 1M":    ("Na", "PF6", "DME", "1M"),
}

SYSTEMS_ORDER = [
    "NaPF6/DME 0.1M",
    "NaOTf/DME 0.1M",
    "LiPF6/DME 0.5M",
    "NaPF6/DME 0.5M",
    "NaOTf/DME 1M",
    "NaPF6/DME 1M",
]

MODEL_ORDER = ["UMA (teacher)", "Micro student", "Original student"]
MODEL_COLORS = {
    "UMA (teacher)":    "#1f77b4",
    "Micro student":    "#ff7f0e",
    "Original student": "#2ca02c",
    "Experiment":       "#d62728",
}

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.linestyle": ":"})


def load_exp_densities(exp_csv):
    df = pd.read_csv(exp_csv)
    lookup = {}
    for sys_name, (cat, an, sol, conc) in SYSTEM_LOOKUP.items():
        mask = (
            (df["cation"] == cat) &
            (df["anion"] == an) &
            (df["solvent"] == sol) &
            (df["solute concentration"] == conc)
        )
        rows = df[mask]
        if len(rows) == 0:
            print(f"WARNING: no exp density found for {sys_name}")
            lookup[sys_name] = None
        elif len(rows) > 1:
            # prefer the 298K row if multiple temperatures
            rows_298 = rows[rows["temperature"].str.startswith("298")]
            lookup[sys_name] = float(rows_298["Density"].iloc[0]) if len(rows_298) else float(rows["Density"].iloc[0])
        else:
            lookup[sys_name] = float(rows["Density"].iloc[0])
    return lookup


def main():
    computed = pd.read_csv(COMPUTED_CSV)
    exp_densities = load_exp_densities(EXP_CSV)

    all_models = MODEL_ORDER + ["Experiment"]
    n_models = len(all_models)
    n_systems = len(SYSTEMS_ORDER)
    bar_width = 0.18
    x = np.arange(n_systems)

    fig, ax = plt.subplots(figsize=(13, 5))

    for i, model_name in enumerate(all_models):
        means, stds = [], []
        for sys_name in SYSTEMS_ORDER:
            if model_name == "Experiment":
                val = exp_densities.get(sys_name)
                means.append(val if val is not None else 0.0)
                stds.append(0.0)
            else:
                row = computed[(computed["System"] == sys_name) & (computed["Model"] == model_name)]
                if len(row) > 0:
                    means.append(row["Density (g/cm³)"].values[0])
                    stds.append(row["Std (g/cm³)"].values[0])
                else:
                    means.append(0.0)
                    stds.append(0.0)

        ax.bar(
            x + i * bar_width, means, bar_width,
            yerr=stds, label=model_name, color=MODEL_COLORS[model_name],
            capsize=3, edgecolor="black", linewidth=0.5,
            error_kw={"elinewidth": 1.0},
        )

    ax.set_xticks(x + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(SYSTEMS_ORDER, rotation=30, ha="right")
    ax.set_ylabel(r"Density (g/cm$^3$)")
    ax.set_title("Density Comparison: Student Models vs. UMA Teacher vs. Experiment", fontweight="bold")
    ax.legend(loc="upper left")
    fig.tight_layout()

    out_path = OUT_DIR / "density_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")

    # Print summary table
    rows = []
    for sys_name in SYSTEMS_ORDER:
        row = {"System": sys_name, "Experiment": exp_densities.get(sys_name)}
        for model_name in MODEL_ORDER:
            r = computed[(computed["System"] == sys_name) & (computed["Model"] == model_name)]
            row[model_name] = round(r["Density (g/cm³)"].values[0], 4) if len(r) else None
        rows.append(row)
    summary = pd.DataFrame(rows)
    print("\nDensity summary (g/cm³):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
