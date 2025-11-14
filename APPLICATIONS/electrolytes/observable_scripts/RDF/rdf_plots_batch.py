#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch RDF Plotting Script

Given a root directory that contains multiple RDF base directories (each with
subdirectories like `O`, `F` holding `RDF_*.csv`), this script will iterate
through each base directory and generate all plots and a summary report using
the plotting utilities in `rdf_plots.py`.
"""

import argparse
import sys
from pathlib import Path

# Use a non-interactive backend so plt.show() calls do not block/pop up windows
import matplotlib
matplotlib.use("Agg")

# Ensure we can import the sibling module `rdf_plots.py` regardless of CWD
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import rdf_plots as rp


def is_valid_base_dir(base_dir: Path) -> bool:
    """Return True if the directory looks like an RDF base directory."""
    if not base_dir.is_dir():
        return False
    # Heuristic: must contain at least one of the expected atom pair folders
    # with at least one RDF_*.csv inside
    for pair in ("O", "F"):
        pair_dir = base_dir / pair
        if pair_dir.is_dir() and any(pair_dir.glob("RDF_*.csv")):
            return True
    return False


def discover_base_dirs(root_dir: Path) -> list[Path]:
    """Find all immediate child directories under root that look like base dirs."""
    if not root_dir.exists() or not root_dir.is_dir():
        raise FileNotFoundError(f"Root directory not found or not a directory: {root_dir}")
    candidates = [p for p in root_dir.iterdir() if p.is_dir()]
    base_dirs = [p for p in candidates if is_valid_base_dir(p)]
    return sorted(base_dirs)


def process_base_dir(base_dir: Path) -> None:
    """Generate plots and summary for a single base directory using rdf_plots."""
    print(f"\n===== Processing base dir: {base_dir} =====")

    # Re-point the plotting module to this base directory
    rp.BASE_DIR = base_dir

    plot_output_dir = base_dir / "plots"
    plot_output_dir.mkdir(exist_ok=True)

    # Discover available atom pairs in this base dir
    available_pairs = []
    for item in base_dir.iterdir():
        if item.is_dir() and item.name in ["O", "F"]:
            available_pairs.append(item.name)

    print(f"Available atom pairs: {available_pairs}")

    # Generate individual plots for each atom pair
    for atom_pair in available_pairs:
        rp.plot_all_metrics(atom_pair, save_dir=plot_output_dir)

    # Generate combined comparison plots if more than one pair exists
    if len(available_pairs) > 1:
        for metric in ["g_r", "n_r", "w_r_kJmol"]:
            rp.plot_combined_comparison(available_pairs, metric, save_dir=plot_output_dir)

    # Generate summary report
    rp.create_summary_report(available_pairs, save_dir=plot_output_dir)

    print(f"All plots saved to: {plot_output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-generate RDF plots for all base dirs under a root directory.")
    parser.add_argument("root_dir", type=Path, help="Root directory containing multiple RDF base directories")
    args = parser.parse_args()

    root_dir: Path = args.root_dir.resolve()
    print(f"Root directory: {root_dir}")

    base_dirs = discover_base_dirs(root_dir)
    if not base_dirs:
        print("No valid base directories found.")
        return

    print("\nFound base directories:")
    for bd in base_dirs:
        print(f" - {bd}")

    for bd in base_dirs:
        process_base_dir(bd)

    print("\nBatch processing complete.")


if __name__ == "__main__":
    main()


