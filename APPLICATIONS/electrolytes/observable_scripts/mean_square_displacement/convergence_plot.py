#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Quick run (from this directory or project root):
#   python convergence_plot.py --root-dir /global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/observables/msd/convergence_analysis
"""
Convergence plots for diffusion coefficients vs. analysis time.

Directory layout (example)
--------------------------
root_dir/
  1010ps/
    uma_diffusion_selected_fast.csv
  1600ps/
    uma_diffusion_selected_fast.csv
  2000ps/
    uma_diffusion_selected_fast.csv   <- template for which systems to plot
  ...

Each CSV is assumed to have at least the following columns:
    system,
    D_cation_(x1e-10_m2_s),
    D_anion_(x1e-10_m2_s),
    D_solvent_(x1e-10_m2_s),
plus any additional metadata columns.

What this script does
---------------------
- Read the template CSV at:  <root_dir>/<template_subdir>/uma_diffusion_selected_fast.csv
  (by default <template_subdir> = "2000ps").
- Use the "system" column in that file as the canonical list + order of systems.
- For every subdirectory of <root_dir> whose name ends with "ps" and which
  contains a CSV with the same filename, read out the three D values.
- Build three plots:
    1) D_anion vs time
    2) D_cation vs time
    3) D_solvent vs time
  with one line per system (only plotting times where that CSV exists).

Usage
-----
Example:
    python convergence_plot.py \\
        --root-dir /global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/observables/msd/convergence_analysis

To use a different template (e.g. "10000ps" under some other root):
    python convergence_plot.py --root-dir <other_root> --template-subdir 10000ps
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


CSV_NAME_DEFAULT = "uma_diffusion_selected_fast.csv"


def _parse_time_from_dirname(dirname: str) -> float:
    """
    Convert a directory name like '2000ps' to a float time in ps (2000.0).
    Returns None if it cannot be parsed.
    """
    if not dirname.endswith("ps"):
        return None
    base = dirname[:-2]
    try:
        return float(base)
    except ValueError:
        return None


def collect_diffusion_data(
    root_dir: Path,
    template_subdir: str = "2000ps",
    csv_name: str = CSV_NAME_DEFAULT,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Walk root_dir, read all subdirectories containing the requested CSV, and
    return:
      - a DataFrame with columns:
            ['system', 'time_ps', 'D_cation', 'D_anion', 'D_solvent']
      - an ordered list of systems taken from the template CSV.
    """
    template_csv = root_dir / template_subdir / csv_name
    if not template_csv.is_file():
        raise FileNotFoundError(
            f"Template CSV not found at {template_csv}. "
            f"Make sure --root-dir is correct and template-subdir ('{template_subdir}') exists."
        )

    print(f"Reading template CSV: {template_csv}")
    df_template = pd.read_csv(template_csv)
    if "system" not in df_template.columns:
        raise ValueError(f"'system' column not found in template CSV: {template_csv}")

    required_cols = [
        "D_cation_(x1e-10_m2_s)",
        "D_anion_(x1e-10_m2_s)",
        "D_solvent_(x1e-10_m2_s)",
    ]
    for col in required_cols:
        if col not in df_template.columns:
            raise ValueError(
                f"Required column '{col}' not found in template CSV: {template_csv}"
            )

    systems = df_template["system"].tolist()
    print(f"Found {len(systems)} systems in template: {systems}")

    records = []

    # Iterate over subdirectories under root_dir
    for sub in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        time_ps = _parse_time_from_dirname(sub.name)
        if time_ps is None:
            # Skip non-time directories (e.g. 'two_systems')
            continue

        csv_path = sub / csv_name
        if not csv_path.is_file():
            print(f"[skip] No '{csv_name}' in {sub}")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"[skip] Failed to read {csv_path}: {exc}")
            continue

        missing_cols = [c for c in ["system"] + required_cols if c not in df.columns]
        if missing_cols:
            print(f"[skip] {csv_path} missing columns {missing_cols}")
            continue

        print(f"[ok] Using data from {csv_path} at time {time_ps} ps")

        for _, row in df.iterrows():
            system = row["system"]
            # Only keep systems that appear in the template (and preserve that order later)
            if system not in systems:
                continue
            records.append(
                {
                    "system": system,
                    "time_ps": time_ps,
                    "D_cation": row["D_cation_(x1e-10_m2_s)"],
                    "D_anion": row["D_anion_(x1e-10_m2_s)"],
                    "D_solvent": row["D_solvent_(x1e-10_m2_s)"],
                }
            )

    if not records:
        raise RuntimeError(
            f"No usable diffusion data found under {root_dir}. "
            "Check that subdirectories like '1600ps', '2000ps', ... each contain "
            f"'{csv_name}' with the expected columns."
        )

    df_all = pd.DataFrame.from_records(records)
    return df_all, systems


def _plot_property_vs_time(
    df: pd.DataFrame,
    systems: List[str],
    prop: str,
    out_path: Path,
    ylabel: str,
    title: str,
) -> None:
    """
    Make a single plot of <prop> vs time for each system and save to out_path.
    """
    plt.figure(figsize=(6, 4))

    for system in systems:
        df_sys = df[df["system"] == system].sort_values("time_ps")
        if df_sys.empty:
            continue
        plt.plot(
            df_sys["time_ps"],
            df_sys[prop],
            marker="o",
            linewidth=1.5,
            label=system,
        )

    plt.xlabel("time (ps)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved plot → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot convergence of diffusivities (D_anion, D_cation, D_solvent) vs time."
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help=(
            "Root directory containing time subdirectories like '1600ps', '2000ps', ... "
            "each with 'uma_diffusion_selected_fast.csv'."
        ),
    )
    parser.add_argument(
        "--template-subdir",
        type=str,
        default="2000ps",
        help=(
            "Name of the subdirectory (relative to root-dir) whose CSV defines the "
            "set/order of systems to plot (default: 2000ps)."
        ),
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default=CSV_NAME_DEFAULT,
        help=f"CSV filename inside each time directory (default: {CSV_NAME_DEFAULT}).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="convergence",
        help="Prefix for output PNG filenames (default: 'convergence').",
    )

    args = parser.parse_args()
    root_dir = Path(args.root_dir).expanduser().resolve()

    if not root_dir.is_dir():
        raise NotADirectoryError(f"root-dir does not exist or is not a directory: {root_dir}")

    df_all, systems = collect_diffusion_data(
        root_dir=root_dir,
        template_subdir=args.template_subdir,
        csv_name=args.csv_name,
    )

    # Save combined CSV for convenience
    combined_csv = root_dir / f"{args.output_prefix}_diffusion_vs_time.csv"
    df_all_sorted = df_all.sort_values(["system", "time_ps"])
    df_all_sorted.to_csv(combined_csv, index=False)
    print(f"Wrote combined data CSV → {combined_csv}")

    # Generate three plots
    _plot_property_vs_time(
        df_all,
        systems,
        prop="D_anion",
        out_path=root_dir / f"{args.output_prefix}_D_anion_vs_time.png",
        ylabel=r"D$_{\mathrm{anion}}$ (1e-10 m$^2$/s)",
        title="Anion diffusivity vs time",
    )
    _plot_property_vs_time(
        df_all,
        systems,
        prop="D_cation",
        out_path=root_dir / f"{args.output_prefix}_D_cation_vs_time.png",
        ylabel=r"D$_{\mathrm{cation}}$ (1e-10 m$^2$/s)",
        title="Cation diffusivity vs time",
    )
    _plot_property_vs_time(
        df_all,
        systems,
        prop="D_solvent",
        out_path=root_dir / f"{args.output_prefix}_D_solvent_vs_time.png",
        ylabel=r"D$_{\mathrm{solvent}}$ (1e-10 m$^2$/s)",
        title="Solvent diffusivity vs time",
    )


if __name__ == "__main__":
    main()