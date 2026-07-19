#!/usr/bin/env python3
"""Energy timeseries plotting functions."""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.linestyle": ":"})


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    import pandas as pd
    w = max(1, int(w))
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()


def plot_energy_timeseries(
    times: np.ndarray,
    pe: np.ndarray,
    ke: np.ndarray,
    etot: np.ndarray,
    temp: np.ndarray,
    sys_name: str,
    model_label: str,
    output_dir: Path,
    color: str = "#1f77b4",
    roll_frac: float = 0.05,
) -> Path:
    """Four-panel energy/temperature timeseries plot.

    Returns path to saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    roll_w = max(3, int(len(times) * roll_frac))
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    fig.suptitle(
        f"{model_label}  |  {sys_name}\nEnergy & Temperature vs Time",
        fontsize=12, fontweight="bold",
    )

    datasets = [
        (pe,   r"$E_{pot}$ (eV)", "PE"),
        (ke,   r"$E_{kin}$ (eV)", "KE"),
        (etot, r"$E_{tot}$ (eV)", "Total E"),
        (temp, r"$T$ (K)",        "Temperature"),
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
        ax.text(0.01, 0.05, f"mean={mean_v:.4g}  std={std_v:.3g}",
                transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    axes[-1].set_xlabel("Simulation time (ns)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    safe = lambda s: s.replace("/", "_").replace(" ", "_")
    out_path = output_dir / f"energy_{safe(sys_name)}_{safe(model_label)}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_energy_comparison(
    energy_data: dict,
    sys_name: str,
    model_order: list,
    colors: dict,
    output_dir: Path,
    roll_frac: float = 0.05,
) -> Path:
    """Overlay potential energy, total energy and temperature across models.

    energy_data: {model_label: (times_ns, pe, ke, etot, temp)}.
    Intended for comparing replicas of the *same* system, where the absolute
    PE / E_tot / T are directly comparable. Returns path to saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panels = [
        (1, r"$E_{pot}$ (eV)", "Potential energy"),
        (3, r"$E_{tot}$ (eV)", "Total energy"),
        (4, r"$T$ (K)",        "Temperature"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    fig.suptitle(f"{sys_name} — Energy & Temperature comparison",
                 fontsize=13, fontweight="bold")

    for ax, (col, ylabel, title) in zip(axes, panels):
        for model in model_order:
            if model not in energy_data:
                continue
            tup = energy_data[model]
            times = tup[0]
            vals = tup[col]
            if len(times) == 0:
                continue
            roll_w = max(3, int(len(times) * roll_frac))
            c = colors.get(model, "#1f77b4")
            mean_v = float(vals.mean())
            std_v = float(vals.std())
            ax.scatter(times, vals, s=1.0, alpha=0.12, color=c, rasterized=True)
            ax.plot(times, _rolling_mean(vals, roll_w), color=c, lw=1.6,
                    label=f"{model}  {mean_v:.4g}±{std_v:.3g}")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10, loc="left")
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Simulation time (ns)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    safe = lambda s: s.replace("/", "_").replace(" ", "_")
    out_path = output_dir / f"energy_comparison_{safe(sys_name)}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
