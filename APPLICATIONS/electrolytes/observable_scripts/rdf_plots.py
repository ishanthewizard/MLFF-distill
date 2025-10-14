#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RDF Plotting Script: Create publication-quality plots from RDF CSV data.
Loads RDF data from CSV files and creates comparison plots for different trajectory types.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib import rcParams

# Set matplotlib style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 16
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 12
rcParams['figure.titlesize'] = 18

# ───────────────────────────── Configuration ─────────────────────────────
# Base directory containing RDF CSV files
BASE_DIR = Path("/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/ablate_distillation/rdf_output/ablate_wwo_hessian_500ps")

# Color palette for trajectory types (will cycle through if more types than colors)
COLOR_PALETTE = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf"   # Cyan
]

# Line styles for better distinction (will cycle through if needed)
LINE_STYLE_PALETTE = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 5))]

# RDF parameters (should match those used in computation)
r_max = 15.0  # Maximum distance for plotting

# ───────────────────────────── Helper Functions ─────────────────────────────
def discover_trajectory_types(atom_pair: str) -> list:
    """
    Automatically discover all available trajectory types from CSV files.
    
    Args:
        atom_pair: Atom pair directory name (e.g., "O", "F")
    
    Returns:
        List of trajectory type names sorted alphabetically
    """
    data_dir = BASE_DIR / atom_pair
    if not data_dir.exists():
        return []
    
    # Find all RDF CSV files
    trajectory_types = []
    for csv_file in data_dir.glob("RDF_*.csv"):
        # Extract trajectory type from filename (e.g., "RDF_pert_10.csv" -> "pert_10")
        traj_type = csv_file.stem.replace("RDF_", "")
        trajectory_types.append(traj_type)
    
    # Sort alphabetically for consistent ordering
    trajectory_types.sort()
    
    return trajectory_types

def get_trajectory_label(traj_type: str) -> str:
    """
    Generate a display label for a trajectory type.
    
    Args:
        traj_type: Trajectory type identifier
    
    Returns:
        Formatted display label
    """
    # Handle special cases
    if traj_type == "uma":
        return "UMA (Reference)"
    elif traj_type == "w_o hessian":
        return "w/o Hessian"
    elif traj_type.startswith("pert_"):
        # Extract number from "pert_10" -> "Perturbed (10)"
        num = traj_type.replace("pert_", "")
        return f"Perturbed ({num})"
    else:
        # Default: capitalize and replace underscores
        return traj_type.replace("_", " ").title()

def get_trajectory_color(traj_type: str, index: int) -> str:
    """
    Get color for a trajectory type based on its index.
    
    Args:
        traj_type: Trajectory type identifier
        index: Index in the sorted list of trajectory types
    
    Returns:
        Color hex code
    """
    return COLOR_PALETTE[index % len(COLOR_PALETTE)]

def get_trajectory_linestyle(traj_type: str, index: int) -> str:
    """
    Get line style for a trajectory type based on its index.
    
    Args:
        traj_type: Trajectory type identifier
        index: Index in the sorted list of trajectory types
    
    Returns:
        Line style
    """
    return LINE_STYLE_PALETTE[index % len(LINE_STYLE_PALETTE)]

def load_rdf_data(atom_pair: str) -> dict:
    """
    Load RDF data from CSV files for a specific atom pair.
    
    Args:
        atom_pair: Atom pair directory name (e.g., "O", "F")
    
    Returns:
        Dictionary with keys:
            - 'data': Dictionary mapping trajectory types to DataFrames
            - 'trajectory_types': List of trajectory type names
            - 'labels': Dictionary mapping trajectory types to display labels
            - 'colors': Dictionary mapping trajectory types to colors
            - 'linestyles': Dictionary mapping trajectory types to line styles
    """
    data_dir = BASE_DIR / atom_pair
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    
    # Discover all available trajectory types
    trajectory_types = discover_trajectory_types(atom_pair)
    
    if not trajectory_types:
        print(f"Warning: No RDF CSV files found in {data_dir}")
        return {
            'data': {},
            'trajectory_types': [],
            'labels': {},
            'colors': {},
            'linestyles': {}
        }
    
    # Load data and generate styling information
    data = {}
    labels = {}
    colors = {}
    linestyles = {}
    
    for idx, traj_type in enumerate(trajectory_types):
        csv_file = data_dir / f"RDF_{traj_type}.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            data[traj_type] = df
            labels[traj_type] = get_trajectory_label(traj_type)
            colors[traj_type] = get_trajectory_color(traj_type, idx)
            linestyles[traj_type] = get_trajectory_linestyle(traj_type, idx)
            print(f"Loaded {labels[traj_type]} data: {len(df)} points")
        else:
            print(f"Warning: File not found: {csv_file}")
    
    return {
        'data': data,
        'trajectory_types': trajectory_types,
        'labels': labels,
        'colors': colors,
        'linestyles': linestyles
    }

def plot_single_metric(rdf_data: dict, metric: str, ylabel: str, atom_pair: str, 
                      ylim: tuple = None, xlim: tuple = (0, r_max)):
    """
    Create a single plot for one RDF metric.
    
    Args:
        rdf_data: Dictionary containing 'data', 'labels', 'colors', 'linestyles'
        metric: Column name to plot (e.g., 'g_r', 'n_r', 'w_r_kJmol')
        ylabel: Y-axis label
        atom_pair: Atom pair name for title
        ylim: Y-axis limits (optional)
        xlim: X-axis limits
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = rdf_data['data']
    labels = rdf_data['labels']
    colors = rdf_data['colors']
    linestyles = rdf_data['linestyles']
    
    # Plot each trajectory type
    for traj_type, df in data.items():
        if metric in df.columns:
            r = df['r_A'].values
            y = df[metric].values
            
            # Handle NaN values in PMF data
            if metric == 'w_r_kJmol':
                mask = np.isfinite(y)
                r, y = r[mask], y[mask]
            
            ax.plot(r, y, 
                   color=colors[traj_type],
                   linestyle=linestyles[traj_type],
                   linewidth=2.5,
                   label=labels[traj_type],
                   alpha=0.8)
    
    # Formatting
    ax.set_xlabel("Distance r (Å)", fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    
    # Add title
    title = f"Na–{atom_pair} {metric.upper()} Comparison"
    ax.set_title(title, fontweight='bold', pad=20)
    
    # Tight layout and return figure
    plt.tight_layout()
    return fig

def plot_all_metrics(atom_pair: str, save_dir: Path = None):
    """
    Create all RDF metric plots for a given atom pair.
    
    Args:
        atom_pair: Atom pair to plot (e.g., "O", "F")
        save_dir: Directory to save plots (optional)
    """
    print(f"\n=== Creating plots for Na-{atom_pair} ===")
    
    # Load data
    rdf_data = load_rdf_data(atom_pair)
    if not rdf_data['data']:
        print(f"No data found for {atom_pair}")
        return
    
    # Define metrics to plot
    metrics = [
        ("g_r", r"$g(r)$", (0, 4)),  # RDF
        ("n_r", r"$n(r)$", (0, 8)),  # Coordination number  
        ("w_r_kJmol", r"$w(r)$ (kJ mol$^{-1}$)", None)  # PMF
    ]
    
    # Create individual plots
    for metric, ylabel, ylim in metrics:
        fig = plot_single_metric(rdf_data, metric, ylabel, atom_pair, ylim)
        
        if save_dir:
            save_path = save_dir / f"Na-{atom_pair}_{metric}_comparison.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()

def plot_combined_comparison(atom_pairs: list, metric: str, save_dir: Path = None):
    """
    Create combined comparison plot for multiple atom pairs.
    
    Args:
        atom_pairs: List of atom pairs to compare
        metric: Metric to plot ('g_r', 'n_r', 'w_r_kJmol')
        save_dir: Directory to save plots (optional)
    """
    print(f"\n=== Creating combined {metric} comparison ===")
    
    fig, axes = plt.subplots(1, len(atom_pairs), figsize=(5*len(atom_pairs), 6))
    if len(atom_pairs) == 1:
        axes = [axes]
    
    ylabel_map = {
        "g_r": r"$g(r)$",
        "n_r": r"$n(r)$", 
        "w_r_kJmol": r"$w(r)$ (kJ mol$^{-1}$)"
    }
    
    for i, atom_pair in enumerate(atom_pairs):
        ax = axes[i]
        rdf_data = load_rdf_data(atom_pair)
        
        data = rdf_data['data']
        labels = rdf_data['labels']
        colors = rdf_data['colors']
        linestyles = rdf_data['linestyles']
        
        # Plot each trajectory type
        for traj_type, df in data.items():
            if metric in df.columns:
                r = df['r_A'].values
                y = df[metric].values
                
                # Handle NaN values in PMF data
                if metric == 'w_r_kJmol':
                    mask = np.isfinite(y)
                    r, y = r[mask], y[mask]
                
                ax.plot(r, y,
                       color=colors[traj_type],
                       linestyle=linestyles[traj_type], 
                       linewidth=2.5,
                       label=labels[traj_type],
                       alpha=0.8)
        
        # Formatting
        ax.set_xlabel("Distance r (Å)", fontweight='bold')
        ax.set_ylabel(ylabel_map[metric], fontweight='bold')
        ax.set_xlim(0, r_max)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Na–{atom_pair}", fontweight='bold')
        
        # Add legend only to first subplot
        if i == 0:
            ax.legend(frameon=True, fancybox=True, shadow=True)
    
    # Overall title
    fig.suptitle(f"{metric.upper()} Comparison", fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        save_path = save_dir / f"combined_{metric}_comparison.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()

def create_summary_report(atom_pairs: list, save_dir: Path = None):
    """
    Create a summary report with key statistics.
    
    Args:
        atom_pairs: List of atom pairs to analyze
        save_dir: Directory to save report (optional)
    """
    print("\n=== RDF Summary Report ===")
    
    report_lines = ["RDF Analysis Summary Report", "=" * 50, ""]
    
    for atom_pair in atom_pairs:
        rdf_data = load_rdf_data(atom_pair)
        if not rdf_data['data']:
            continue
        
        data = rdf_data['data']
        labels = rdf_data['labels']
            
        report_lines.append(f"Na-{atom_pair} Analysis:")
        report_lines.append("-" * 30)
        
        for traj_type, df in data.items():
            if 'g_r' in df.columns:
                # Find first peak in RDF
                g_r = df['g_r'].values
                r = df['r_A'].values
                
                # Simple peak finding (first maximum after r > 1.5)
                mask = r > 1.5
                if np.any(mask):
                    peak_idx = np.argmax(g_r[mask]) + np.where(mask)[0][0]
                    peak_r = r[peak_idx]
                    peak_g = g_r[peak_idx]
                    
                    report_lines.append(f"  {labels[traj_type]}:")
                    report_lines.append(f"    First peak: r = {peak_r:.2f} Å, g(r) = {peak_g:.2f}")
        
        report_lines.append("")
    
    report_text = "\n".join(report_lines)
    print(report_text)
    
    if save_dir:
        report_path = save_dir / "rdf_summary_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        print(f"Report saved: {report_path}")

# ───────────────────────────── Main Execution ─────────────────────────────
if __name__ == "__main__":
    # Create output directory for plots
    plot_output_dir = BASE_DIR / "plots"
    plot_output_dir.mkdir(exist_ok=True)
    
    # Available atom pairs (based on directory structure)
    available_pairs = []
    for item in BASE_DIR.iterdir():
        if item.is_dir() and item.name in ["O", "F"]:
            available_pairs.append(item.name)
    
    print(f"Available atom pairs: {available_pairs}")
    
    # Generate individual plots for each atom pair
    for atom_pair in available_pairs:
        plot_all_metrics(atom_pair, save_dir=plot_output_dir)
    
    # Generate combined comparison plots
    if len(available_pairs) > 1:
        for metric in ["g_r", "n_r", "w_r_kJmol"]:
            plot_combined_comparison(available_pairs, metric, save_dir=plot_output_dir)
    
    # Generate summary report
    create_summary_report(available_pairs, save_dir=plot_output_dir)
    
    print(f"\nAll plots saved to: {plot_output_dir}")
    print("Analysis complete!")
