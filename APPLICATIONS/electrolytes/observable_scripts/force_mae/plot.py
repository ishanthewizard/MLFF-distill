#!/usr/bin/env python3
"""Force MAE timeseries plotting functions."""

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


def plot_force_mae_timeseries(
    result: dict,
    sys_name: str,
    model_label: str,
    output_dir: Path,
    color: str = "#1f77b4",
    roll_frac: float = 0.03,
) -> Path:
    """Three-panel plot: Force MAE, Max Force Error, and RMSE vs simulation time.

    result: dict returned by compute_force_mae — keys times_ns, mae, max_err, rmse.
    Returns path to saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    times   = result["times_ns"]
    mae     = result["mae"]
    max_err = result["max_err"]
    rmse    = result["rmse"]

    roll_w = max(3, int(len(times) * roll_frac))

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    fig.suptitle(
        f"{model_label}  |  {sys_name}\nTeacher vs Student Per-Frame Force Errors",
        fontsize=12, fontweight="bold",
    )

    panels = [
        (mae,     "Force MAE (eV/Å)",     "mean |ΔF|"),
        (max_err, "Max Force Error (eV/Å)","max |ΔF|"),
        (rmse,    "Force RMSE (eV/Å)",     "RMSE"),
    ]

    for ax, (vals, ylabel, lbl) in zip(axes, panels):
        mean_v = float(vals.mean())
        ax.semilogy(times, vals, color=color, lw=0.6, alpha=0.35, rasterized=True)
        ax.semilogy(times, _rolling_mean(vals, roll_w), color=color, lw=2.0,
                    label=f"{lbl} (roll avg)")
        ax.axhline(mean_v, color="red", lw=1.2, ls="--",
                   label=f"mean = {mean_v:.4f} eV/Å")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9, loc="upper left")

    axes[-1].set_xlabel("Simulation time (ns)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    safe = lambda s: s.replace("/", "_").replace(" ", "_")
    out_path = output_dir / f"force_mae_{safe(sys_name)}_{safe(model_label)}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_force_mae_comparison(
    results: dict[str, dict],
    sys_name: str,
    model_order: list[str],
    model_colors: dict[str, str],
    output_dir: Path,
    roll_frac: float = 0.03,
) -> Path:
    """Overlay force MAE timeseries for multiple models in one figure.

    results: {model_label: result_dict}  where result_dict comes from compute_force_mae.
    Returns path to saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    fig.suptitle(f"{sys_name}\nTeacher vs Student Per-Frame Force Errors",
                 fontsize=12, fontweight="bold")

    metrics = [("mae", "Force MAE (eV/Å)"), ("max_err", "Max Force Error (eV/Å)"), ("rmse", "Force RMSE (eV/Å)")]

    for ax, (key, ylabel) in zip(axes, metrics):
        for model in model_order:
            if model not in results:
                continue
            res   = results[model]
            times = res["times_ns"]
            vals  = res[key]
            color = model_colors.get(model, "#1f77b4")
            roll_w = max(3, int(len(times) * roll_frac))
            mean_v = float(vals.mean())
            ax.semilogy(times, vals, color=color, lw=0.5, alpha=0.25, rasterized=True)
            ax.semilogy(times, _rolling_mean(vals, roll_w), color=color, lw=2.0,
                        label=f"{model}  mean={mean_v:.4f} eV/Å")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9, loc="upper left")

    axes[-1].set_xlabel("Simulation time (ns)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    safe = lambda s: s.replace("/", "_").replace(" ", "_")
    out_path = output_dir / f"force_mae_comparison_{safe(sys_name)}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
