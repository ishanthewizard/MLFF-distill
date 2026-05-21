#!/usr/bin/env python3
"""Density plotting functions."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.linestyle": ":"})


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    w = max(1, int(w))
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()


def plot_density_timeseries(
    times_and_densities: dict[str, tuple[np.ndarray, np.ndarray]],
    sys_name: str,
    output_dir: Path,
    model_colors: dict[str, str] | None = None,
    roll_window_ns: float = 0.5,
    eq_cutoff_ns: float | None = None,
) -> Path:
    """Plot density vs simulation time for one system, overlaying multiple models.

    times_and_densities: {model_label: (times_ns, densities_g_cm3)}
    roll_window_ns: width of the rolling average window in ns. Controls how much
        short-time noise is smoothed out while preserving slower drift.
    eq_cutoff_ns: if given, draws a vertical dashed line marking the equilibration cutoff.
    Returns path to saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    defaults = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    if model_colors is None:
        model_colors = {}
    labels = list(times_and_densities.keys())
    for i, lbl in enumerate(labels):
        model_colors.setdefault(lbl, defaults[i % len(defaults)])

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle(f"{sys_name} — Density vs Time", fontsize=13, fontweight="bold")

    for lbl, (times, dens) in times_and_densities.items():
        color = model_colors[lbl]
        mean_d = dens.mean()
        std_d = dens.std(ddof=1)

        dt_ns = float(times[1] - times[0]) if len(times) > 1 else 1.0
        roll_w = max(3, int(round(roll_window_ns / dt_ns)))

        # instant density as faint background
        ax.plot(times, dens, color=color, lw=0.5, alpha=0.25, rasterized=True)
        # smoothed running average on top
        ax.plot(times, _rolling_mean(dens, roll_w), color=color, lw=2.0,
                label=f"{lbl}  {mean_d:.4f}±{std_d:.4f} g/cm³")

    if eq_cutoff_ns is not None:
        ax.axvline(eq_cutoff_ns, color="black", lw=1.2, ls=":",
                   label=f"eq cutoff {eq_cutoff_ns:.1f} ns")

    ax.set_ylabel(r"Density (g/cm$^3$)")
    ax.set_xlabel("Simulation time (ns)")
    ax.legend(fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    safe = sys_name.replace("/", "_").replace(" ", "_")
    out_path = output_dir / f"density_timeseries_{safe}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_density_bars(
    density_df: pd.DataFrame,
    systems_order: list[str],
    model_order: list[str],
    model_colors: dict[str, str],
    output_dir: Path,
    title: str = "Density Comparison",
) -> Path:
    """Bar chart of density per system and model.

    density_df must have columns: System, Model, Density (g/cm³), Std (g/cm³).
    Returns path to saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_sys = len(systems_order)
    bar_width = 0.8 / max(len(model_order), 1)
    x = np.arange(n_sys)

    fig, ax = plt.subplots(figsize=(max(8, 2 * n_sys), 5))
    for i, model_name in enumerate(model_order):
        means, stds = [], []
        for sys_name in systems_order:
            row = density_df[
                (density_df["System"] == sys_name) & (density_df["Model"] == model_name)
            ]
            if len(row) > 0:
                means.append(row["Density (g/cm³)"].values[0])
                stds.append(row["Std (g/cm³)"].values[0])
            else:
                means.append(0.0)
                stds.append(0.0)
        ax.bar(
            x + i * bar_width,
            means,
            bar_width,
            yerr=stds,
            label=model_name,
            color=model_colors.get(model_name, None),
            capsize=3,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x + bar_width * (len(model_order) - 1) / 2)
    ax.set_xticklabels(systems_order, rotation=30, ha="right")
    ax.set_ylabel(r"Density (g/cm$^3$)")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    fig.tight_layout()

    out_path = output_dir / "density_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
