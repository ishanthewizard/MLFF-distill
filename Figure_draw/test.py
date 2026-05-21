#!/usr/bin/env python3
"""
Draw density vs temperature for 0.5M LiPF6 / NaPF6 in DME:
  - EXP (from wide-form CSV)
  - SIM (UMA) with error bars
  - SIM (Student) with error bars

Outputs a PNG figure (default: Figure_draw/density_vs_temp_0p5M_NaPF6_LiPF6_sim_vs_exp.png)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_exp_wide_csv(path: Path) -> pd.DataFrame:
    """
    Expected columns:
      - T_K
      - LiPF6_DME_0.5M
      - NaPF6_DME_0.5M
    Returns long-form with columns: system, temperature, density
    """
    df = pd.read_csv(path)
    if "T_K" not in df.columns:
        raise ValueError(f"EXP CSV missing 'T_K' column: {path}")

    mapping = {
        "LiPF6_DME_0.5M": "0.5M LiPF6",
        "NaPF6_DME_0.5M": "0.5M NaPF6",
    }
    missing = [c for c in mapping.keys() if c not in df.columns]
    if missing:
        raise ValueError(f"EXP CSV missing columns {missing}: {path}")

    long = (
        df.melt(
            id_vars=["T_K"],
            value_vars=list(mapping.keys()),
            var_name="system_raw",
            value_name="density",
        )
        .rename(columns={"T_K": "temperature"})
        .assign(system=lambda d: d["system_raw"].map(mapping))
        .drop(columns=["system_raw"])
        .dropna(subset=["temperature", "density", "system"])
        .sort_values(["system", "temperature"])
        .reset_index(drop=True)
    )
    return long[["system", "temperature", "density"]]


def load_sim_csv(path: Path) -> pd.DataFrame:
    """
    Expected columns:
      - system
      - temperature (K)
      - density (g/cm^3)
      - std_dev (g/cm^3)
    """
    df = pd.read_csv(path)
    required = {"system", "temperature", "density", "std_dev"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"SIM CSV missing columns {missing}: {path}")
    df = df.copy()
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["density"] = pd.to_numeric(df["density"], errors="coerce")
    df["std_dev"] = pd.to_numeric(df["std_dev"], errors="coerce")
    df = df.dropna(subset=["system", "temperature", "density", "std_dev"])
    return df.sort_values(["system", "temperature"]).reset_index(drop=True)


def make_plot(exp_df: pd.DataFrame, uma_df: pd.DataFrame, student_df: pd.DataFrame, outpath: Path) -> None:
    systems = ["0.5M LiPF6", "0.5M NaPF6"]
    system_color = {
        "0.5M LiPF6": "#2492f2",
        "0.5M NaPF6": "#d9534f",
    }
    # Same color within each salt group; different shapes for EXP/Student/UMA
    source_marker = {
        "EXP": "o",
        "Student": "s",
        "UMA": "^",
    }

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    # Student (dotted)
    for sys in systems:
        d = student_df[student_df["system"] == sys]
        if len(d) == 0:
            continue
        ax.errorbar(
            d["temperature"],
            d["density"],
            yerr=d["std_dev"],
            fmt=source_marker["Student"],
            color=system_color[sys],
            linestyle=":",
            linewidth=1.6,
            capsize=4,
            markersize=6,
            alpha=0.9,
            label=f"{sys} (Student)",
        )

    # UMA (dotted)
    for sys in systems:
        d = uma_df[uma_df["system"] == sys]
        if len(d) == 0:
            continue
        ax.errorbar(
            d["temperature"],
            d["density"],
            yerr=d["std_dev"],
            fmt=source_marker["UMA"],
            color=system_color[sys],
            linestyle=":",
            linewidth=1.6,
            capsize=4,
            markersize=6,
            alpha=0.9,
            label=f"{sys} (UMA)",
        )

    # EXP (dotted, open marker, no error bars)
    for sys in systems:
        d = exp_df[exp_df["system"] == sys]
        if len(d) == 0:
            continue
        ax.plot(
            d["temperature"],
            d["density"],
            marker=source_marker["EXP"],
            linestyle=":",
            linewidth=2.0,
            markersize=7,
            markerfacecolor="none",
            markeredgewidth=1.6,
            color=system_color[sys],
            alpha=1.0,
            label=f"{sys} (EXP)",
        )

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Density (g cm$^{-3}$)")
    ax.set_title(r"Density vs. Temperature @ 0.5M (LiPF$_6$ / NaPF$_6$ in DME)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, ncol=1, frameon=True)

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")


def main() -> int:
    repo_root = Path("/global/homes/y/yuejian/project/MLFF-distill")
    default_data_dir = repo_root / "yuejian/electrolyte_application/ablate_distillation/observables/data_for_plotting"

    parser = argparse.ArgumentParser(description="Plot density vs temperature (exp vs UMA vs student).")
    parser.add_argument(
        "--exp",
        type=Path,
        default=default_data_dir / "density_vs_temperature_0p5M_combined_with_errors_exp.csv",
        help="EXP CSV (wide-form).",
    )
    parser.add_argument(
        "--student",
        type=Path,
        default=default_data_dir / "density_vs_temperature_0p5M_combined_with_errors_student_model.csv",
        help="Student sim CSV (long-form with std_dev).",
    )
    parser.add_argument(
        "--uma",
        type=Path,
        default=default_data_dir / "density_vs_temperature_0p5M_combined_with_errors_uma.csv",
        help="UMA sim CSV (long-form with std_dev).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root / "Figure_draw/density_vs_temp_0p5M_NaPF6_LiPF6_sim_vs_exp.png",
        help="Output PNG path.",
    )
    args = parser.parse_args()

    exp_df = load_exp_wide_csv(args.exp)
    student_df = load_sim_csv(args.student)
    uma_df = load_sim_csv(args.uma)

    make_plot(exp_df=exp_df, uma_df=uma_df, student_df=student_df, outpath=args.out)
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
