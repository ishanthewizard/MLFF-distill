#!/usr/bin/env python3
"""MSD and diffusivity plotting functions.

Two outputs per system/model:
  plot_msd          — MSD curves with linear fit overlay
  plot_convergence  — 2×2 diagnostic figure (4 panels)
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.linestyle": ":"})

_SPECIES_COLORS = {"cat": "#1f77b4", "anion": "#ff7f0e", "solvent": "#2ca02c"}


def _safe(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")


def plot_msd(
    result: dict,
    sys_name: str,
    model_label: str,
    output_dir: Path,
) -> Path:
    """MSD vs lag time with linear fit line overlaid on the cation curve.

    result: output dict from run_msd_analysis.
    Returns path to saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tau   = result["tau_ns"]
    cat   = result["cat_symbol"]
    ani   = result["anion_symbol"]
    sol   = result["solvent_symbol"]
    conv  = result["convergence"]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(f"{model_label}  |  {sys_name}\nMSD vs Lag Time", fontsize=12, fontweight="bold")

    tau_ps = tau * 1000  # ns → ps for fit line computation
    mask = result["fit_mask"]

    ax.plot(tau, result["msd_cat"], lw=1.5, color=_SPECIES_COLORS["cat"],
            label=f"{cat}  D={conv['D_cat_final']:.3f}×10⁻¹⁰ m²/s")
    ax.plot(tau[mask], result["fit_slope_cat"] * tau_ps[mask] + result["fit_intercept_cat"],
            "--", color=_SPECIES_COLORS["cat"], lw=1.4)

    if result["msd_anion"] is not None:
        ax.plot(tau, result["msd_anion"], lw=1.3, color=_SPECIES_COLORS["anion"], alpha=0.9,
                label=f"{ani}  D={conv['D_ani_final']:.3f}×10⁻¹⁰ m²/s")
        if result["fit_slope_ani"] is not None:
            ax.plot(tau[mask], result["fit_slope_ani"] * tau_ps[mask] + result["fit_intercept_ani"],
                    "--", color=_SPECIES_COLORS["anion"], lw=1.4, alpha=0.9)

    if result["msd_solvent"] is not None:
        ax.plot(tau, result["msd_solvent"], lw=1.3, color=_SPECIES_COLORS["solvent"], alpha=0.9,
                label=f"{sol}  D={conv['D_sol_final']:.3f}×10⁻¹⁰ m²/s")
        if result["fit_slope_sol"] is not None:
            ax.plot(tau[mask], result["fit_slope_sol"] * tau_ps[mask] + result["fit_intercept_sol"],
                    "--", color=_SPECIES_COLORS["solvent"], lw=1.4, alpha=0.9)

    # shade fit region
    ax.axvspan(conv["tau_min_fit_ns"], conv["tau_max_fit_ns"],
               alpha=0.07, color="grey", label="fit window")

    ax.set_xlabel(r"Lag time $\tau$ (ns)")
    ax.set_ylabel(r"MSD (Å²)")
    ax.legend(fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = output_dir / f"msd_{_safe(sys_name)}_{_safe(model_label)}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_convergence(
    result: dict,
    sys_name: str,
    model_label: str,
    output_dir: Path,
) -> Path:
    """2×2 convergence diagnostic figure.

    Panel 1 (top-left)  : MSD vs lag time
    Panel 2 (top-right) : D vs tau_max (fit grows rightward, fixed tau_min)
    Panel 3 (bot-left)  : ΔD per step (should → 0 in the plateau)
    Panel 4 (bot-right) : Sliding window D (fixed-width window slides across MSD)

    result: output dict from run_msd_analysis.
    Returns path to saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tau  = result["tau_ns"]
    conv = result["convergence"]
    cat  = result["cat_symbol"]
    ani  = result["anion_symbol"]
    sol  = result["solvent_symbol"]

    tau_min = conv["tau_min_fit_ns"]
    tau_max = conv["tau_max_fit_ns"]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    fig.suptitle(
        f"{model_label}  |  {sys_name}\n"
        f"fit: [{tau_min:.1f}–{tau_max:.1f} ns]  ({result['fit_pct']*100:.0f}% of max lag)",
        fontsize=12, fontweight="bold",
    )

    # ── Panel 1: MSD vs time ──────────────────────────────────────────────────
    tau_ps = tau * 1000
    mask = result["fit_mask"]
    axs[0].plot(tau, result["msd_cat"], lw=1.5, color=_SPECIES_COLORS["cat"],
                label=f"{cat}  D={conv['D_cat_final']:.3f}")
    axs[0].plot(tau[mask], result["fit_slope_cat"] * tau_ps[mask] + result["fit_intercept_cat"],
                "--", color=_SPECIES_COLORS["cat"], lw=1.2)
    if result["msd_anion"] is not None:
        axs[0].plot(tau, result["msd_anion"], lw=1.3, color=_SPECIES_COLORS["anion"], alpha=0.9,
                    label=f"{ani}  D={conv['D_ani_final']:.3f}")
        if result["fit_slope_ani"] is not None:
            axs[0].plot(tau[mask], result["fit_slope_ani"] * tau_ps[mask] + result["fit_intercept_ani"],
                        "--", color=_SPECIES_COLORS["anion"], lw=1.2, alpha=0.9)
    if result["msd_solvent"] is not None:
        axs[0].plot(tau, result["msd_solvent"], lw=1.3, color=_SPECIES_COLORS["solvent"], alpha=0.9,
                    label=f"{sol}  D={conv['D_sol_final']:.3f}")
        if result["fit_slope_sol"] is not None:
            axs[0].plot(tau[mask], result["fit_slope_sol"] * tau_ps[mask] + result["fit_intercept_sol"],
                        "--", color=_SPECIES_COLORS["solvent"], lw=1.2, alpha=0.9)
    axs[0].axvspan(tau_min, tau_max, alpha=0.07, color="grey")
    axs[0].set_xlabel(r"$\tau$ (ns)")
    axs[0].set_ylabel(r"MSD (Å²)")
    axs[0].set_title("MSD vs Lag Time")
    axs[0].legend(fontsize=9)

    # ── Panel 2: D vs tau_max (cumulative) ────────────────────────────────────
    x2 = conv["tau_max_sweep_ns"]
    axs[1].plot(x2, conv["D_cat_cumul"],  color=_SPECIES_COLORS["cat"],     lw=1.5,
                label=f"{cat}  {conv['D_cat_final']:.4f}")
    if not np.all(np.isnan(conv["D_ani_cumul"])):
        axs[1].plot(x2, conv["D_ani_cumul"],  color=_SPECIES_COLORS["anion"],   lw=1.3, alpha=0.9,
                    label=f"{ani}  {conv['D_ani_final']:.4f}")
    if not np.all(np.isnan(conv["D_sol_cumul"])):
        axs[1].plot(x2, conv["D_sol_cumul"],  color=_SPECIES_COLORS["solvent"], lw=1.3, alpha=0.9,
                    label=f"{sol}  {conv['D_sol_final']:.4f}")
    axs[1].axvline(tau_max, color="black", lw=1.0, ls=":", label=f"cut {tau_max:.1f} ns")
    axs[1].set_xlabel("tau_max (ns)")
    axs[1].set_ylabel(r"D (×10⁻¹⁰ m²/s)")
    axs[1].set_title(f"D vs Fit Window End\n(tau_min fixed at {tau_min:.1f} ns)")
    axs[1].legend(fontsize=9)

    # ── Panel 3: ΔD per step ──────────────────────────────────────────────────
    axs[2].plot(x2[1:], conv["delta_D_cat"][1:],  color=_SPECIES_COLORS["cat"],     lw=1.2, label=cat)
    if not np.all(np.isnan(conv["delta_D_ani"])):
        axs[2].plot(x2[1:], conv["delta_D_ani"][1:],  color=_SPECIES_COLORS["anion"],   lw=1.0, alpha=0.9, label=ani)
    if not np.all(np.isnan(conv["delta_D_sol"])):
        axs[2].plot(x2[1:], conv["delta_D_sol"][1:],  color=_SPECIES_COLORS["solvent"], lw=1.0, alpha=0.9, label=sol)
    axs[2].axhline(0, color="black", lw=0.8, ls="--")
    axs[2].axvline(tau_max, color="black", lw=1.0, ls=":")
    axs[2].set_xlabel("tau_max (ns)")
    axs[2].set_ylabel(r"$\Delta D$ (×10⁻¹⁰ m²/s)")
    axs[2].set_title("Change in D per Step\n(flat → 0 means plateau reached)")
    axs[2].legend(fontsize=9)

    # ── Panel 4: sliding window D ─────────────────────────────────────────────
    x4 = conv["slide_end_ns"]
    win = conv["slide_window_ns"]
    if len(x4) > 0:
        axs[3].plot(x4, conv["D_cat_slide"],  color=_SPECIES_COLORS["cat"],     lw=1.5, label=cat)
        if not np.all(np.isnan(conv["D_ani_slide"])):
            axs[3].plot(x4, conv["D_ani_slide"],  color=_SPECIES_COLORS["anion"],   lw=1.3, alpha=0.9, label=ani)
        if not np.all(np.isnan(conv["D_sol_slide"])):
            axs[3].plot(x4, conv["D_sol_slide"],  color=_SPECIES_COLORS["solvent"], lw=1.3, alpha=0.9, label=sol)
        axs[3].set_xlabel(f"Window end (ns)  [Δt = {win:.1f} ns]")
        axs[3].set_ylabel(r"D (×10⁻¹⁰ m²/s)")
        axs[3].set_title(f"Sliding Window D\n(flat → diffusive regime stable)")
        axs[3].legend(fontsize=9)
    else:
        axs[3].text(0.5, 0.5, f"slide_window_ns ({win:.1f}) > tau_max ({tau_max:.1f})\n— window too wide",
                    ha="center", va="center", transform=axs[3].transAxes)
        axs[3].set_visible(True)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = output_dir / f"convergence_{_safe(sys_name)}_{_safe(model_label)}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
