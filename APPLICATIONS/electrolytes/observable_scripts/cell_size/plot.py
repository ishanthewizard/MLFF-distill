#!/usr/bin/env python3
"""Cell-size / cell-volume timeseries plotting functions."""

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


def plot_cell_timeseries(
    times_and_cells: dict[str, dict],
    sys_name: str,
    output_dir: Path,
    model_colors: dict[str, str] | None = None,
    roll_window_ns: float = 0.5,
    eq_cutoff_ns: float | None = None,
    rcut: float | None = None,
) -> Path:
    """4-panel plot of cell lengths (a, b, c) and volume vs simulation time.

    times_and_cells: {model_label: result_dict}
        where result_dict has keys times_ns, a, b, c, volume (all ndarray).
    roll_window_ns: smoothing window width in ns.
    eq_cutoff_ns: if given, draws a vertical dashed line marking equilibration cutoff.
    Returns path to saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    defaults = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    if model_colors is None:
        model_colors = {}
    for i, lbl in enumerate(times_and_cells):
        model_colors.setdefault(lbl, defaults[i % len(defaults)])

    panels = [
        ("a",      r"$a$ (Å)"),
        ("b",      r"$b$ (Å)"),
        ("c",      r"$c$ (Å)"),
        ("volume", r"Volume (Å$^3$)"),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    fig.suptitle(f"{sys_name} — Cell Parameters vs Time",
                 fontsize=13, fontweight="bold")

    # keys that get a half-initial-value reference line (not volume)
    _half_ref_keys = {"a", "b", "c"}

    for ax, (key, ylabel) in zip(axes, panels):
        for lbl, res in times_and_cells.items():
            times = res["times_ns"]
            vals  = res[key]
            color = model_colors[lbl]
            mean_v = float(vals.mean())
            std_v  = float(vals.std(ddof=1))

            dt_ns  = float(times[1] - times[0]) if len(times) > 1 else 1.0
            roll_w = max(3, int(round(roll_window_ns / dt_ns)))

            ax.plot(times, vals, color=color, lw=0.5, alpha=0.25, rasterized=True)
            ax.plot(times, _rolling_mean(vals, roll_w), color=color, lw=2.0,
                    label=f"{lbl}  {mean_v:.3f}±{std_v:.3f}")

            # half of initial frame value — frames below this risk finite-size artifacts
            if key in _half_ref_keys:
                half_init = float(vals[0]) / 2.0
                ax.axhline(half_init, color=color, lw=1.2, ls="--", alpha=0.7,
                           label=f"{lbl} {key}₀/2 = {half_init:.2f} Å")

        # 2*r_cut threshold: box below this causes self-loops in the neighbor graph
        if rcut is not None and key in _half_ref_keys:
            ax.axhline(2 * rcut, color="red", lw=1.5, ls="-.",
                       label=f"2·r_cut = {2*rcut:.1f} Å")

        if eq_cutoff_ns is not None:
            ax.axvline(eq_cutoff_ns, color="black", lw=1.2, ls=":",
                       label=f"eq cutoff {eq_cutoff_ns:.1f} ns")

        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9, loc="upper right")

    axes[-1].set_xlabel("Simulation time (ns)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    safe = sys_name.replace("/", "_").replace(" ", "_")
    out_path = output_dir / f"cell_size_{safe}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
