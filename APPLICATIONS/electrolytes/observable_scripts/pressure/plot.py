#!/usr/bin/env python3
"""Pressure timeseries plotting functions."""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.linestyle": ":"})

_SAFE = lambda s: s.replace("/", "_").replace(" ", "_")


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    import pandas as pd
    w = max(1, int(w))
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()


def plot_pressure_timeseries(
    res: dict,
    sys_name: str,
    model_label: str,
    output_dir: Path,
    color: str = "#1f77b4",
    roll_frac: float = 0.05,
) -> Path:
    """Four-panel pressure timeseries: isotropic P, Pxx, Pyy, Pzz.

    Returns path to saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    times = res["times_ns"]
    roll_w = max(3, int(len(times) * roll_frac))

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    fig.suptitle(
        f"{model_label}  |  {sys_name}\nInstantaneous Pressure vs Time",
        fontsize=12, fontweight="bold",
    )

    datasets = [
        (res["pressure"], "P (GPa)",   "Isotropic"),
        (res["pxx"],      "Pxx (GPa)", "Pxx"),
        (res["pyy"],      "Pyy (GPa)", "Pyy"),
        (res["pzz"],      "Pzz (GPa)", "Pzz"),
    ]

    for ax, (vals, ylabel, lbl) in zip(axes, datasets):
        ax.scatter(times, vals, s=1.5, alpha=0.25, color=color, rasterized=True)
        ax.plot(times, _rolling_mean(vals, roll_w), color=color, lw=1.6,
                label=f"{lbl} (roll avg)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9, loc="upper right")
        mean_v, std_v = vals.mean(), vals.std()
        ax.axhline(mean_v, color="black", lw=0.8, ls="--", alpha=0.6)
        ax.set_ylim(mean_v - 5 * std_v, mean_v + 5 * std_v)
        ax.text(0.01, 0.05, f"mean={mean_v:.3f} GPa  std={std_v:.3f} GPa",
                transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    axes[-1].set_xlabel("Simulation time (ns)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = output_dir / f"pressure_{_SAFE(sys_name)}_{_SAFE(model_label)}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_pressure_comparison(
    pressure_data: dict,
    sys_name: str,
    model_order: list,
    colors: dict,
    output_dir: Path,
    roll_frac: float = 0.05,
) -> Path:
    """Single-panel isotropic pressure comparison across models."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_title(f"{sys_name} — Pressure comparison", fontsize=12, fontweight="bold")

    for model in model_order:
        if model not in pressure_data:
            continue
        res = pressure_data[model]
        times = res["times_ns"]
        vals  = res["pressure"]
        roll_w = max(3, int(len(times) * roll_frac))
        c = colors.get(model, "#1f77b4")
        ax.scatter(times, vals, s=1.0, alpha=0.15, color=c, rasterized=True)
        ax.plot(times, _rolling_mean(vals, roll_w), color=c, lw=1.6, label=model)

    ax.set_xlabel("Simulation time (ns)")
    ax.set_ylabel("P (GPa)")
    ax.legend(fontsize=9)
    fig.tight_layout()

    out_path = output_dir / f"pressure_comparison_{_SAFE(sys_name)}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
