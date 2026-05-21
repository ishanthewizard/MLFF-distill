#!/usr/bin/env python3
"""RMSD timeseries plotting functions."""

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


def plot_rmsd_timeseries(
    times_and_rmsd: dict[str, tuple[np.ndarray, np.ndarray]],
    sys_name: str,
    output_dir: Path,
    model_colors: dict[str, str] | None = None,
    roll_window_ns: float = 0.5,
    eq_cutoff_ns: float | None = None,
) -> Path:
    """Plot RMSD vs simulation time, overlaying multiple models.

    times_and_rmsd: {model_label: (times_ns, rmsd_ang)}
    Returns path to saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    defaults = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    if model_colors is None:
        model_colors = {}
    labels = list(times_and_rmsd.keys())
    for i, lbl in enumerate(labels):
        model_colors.setdefault(lbl, defaults[i % len(defaults)])

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle(f"{sys_name} — RMSD vs Time (Kabsch-aligned, ref=frame 0)",
                 fontsize=13, fontweight="bold")

    for lbl, (times, rmsd) in times_and_rmsd.items():
        color  = model_colors[lbl]
        mean_r = float(rmsd.mean())

        dt_ns  = float(times[1] - times[0]) if len(times) > 1 else 1.0
        roll_w = max(3, int(round(roll_window_ns / dt_ns)))

        ax.plot(times, rmsd, color=color, lw=0.5, alpha=0.25, rasterized=True)
        ax.plot(times, _rolling_mean(rmsd, roll_w), color=color, lw=2.0,
                label=f"{lbl}  mean={mean_r:.3f} Å")

    if eq_cutoff_ns is not None:
        ax.axvline(eq_cutoff_ns, color="black", lw=1.2, ls=":",
                   label=f"eq cutoff {eq_cutoff_ns:.1f} ns")

    ax.set_ylabel("RMSD (Å)")
    ax.set_xlabel("Simulation time (ns)")
    ax.legend(fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    safe = lambda s: s.replace("/", "_").replace(" ", "_")
    out_path = output_dir / f"rmsd_{safe(sys_name)}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
