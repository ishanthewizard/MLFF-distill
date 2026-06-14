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


def plot_msd_loglog(
    result: dict,
    sys_name: str,
    model_label: str,
    output_dir: Path,
) -> Path:
    """Log-log MSD plot with slope-1 (diffusive) and slope-2 (ballistic) guides.

    The slope of log(MSD) vs log(τ) reveals the transport regime:
      slope ≈ 2  →  ballistic  (free-flight, short τ)
      slope ≈ 1  →  diffusive  (Einstein, long τ)
      0 < slope < 1  →  sub-diffusive / cage regime
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tau  = result["tau_ns"]
    cat  = result["cat_symbol"]
    ani  = result["anion_symbol"]
    sol  = result["solvent_symbol"]
    conv = result["convergence"]

    # only keep positive τ for log scale
    pos = tau > 0
    tau_p = tau[pos]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        f"{model_label}  |  {sys_name}\nMSD vs Lag Time (log-log)",
        fontsize=12, fontweight="bold",
    )

    def _plot_species(msd, color, label):
        if msd is None:
            return
        ax.plot(tau_p, msd[pos], lw=1.5, color=color, label=label)

    _plot_species(result["msd_cat"],    _SPECIES_COLORS["cat"],    cat)
    _plot_species(result["msd_anion"],  _SPECIES_COLORS["anion"],  ani)
    _plot_species(result["msd_solvent"],_SPECIES_COLORS["solvent"], sol)

    # ── reference slope guides anchored at fit-window start ──────────────────
    tau_ref = conv["tau_min_fit_ns"]
    if tau_ref <= 0:
        tau_ref = tau_p[len(tau_p) // 4]   # fallback: 25% of range

    # anchor MSD value: use cation MSD at tau_ref
    idx_ref = np.searchsorted(tau_p, tau_ref)
    idx_ref = min(idx_ref, len(tau_p) - 1)
    msd_ref = result["msd_cat"][pos][idx_ref]

    guide_tau = np.logspace(np.log10(tau_p[0]), np.log10(tau_p[-1]), 200)
    # slope-1: MSD ∝ τ¹
    ax.plot(guide_tau, msd_ref * (guide_tau / tau_ref) ** 1,
            "k--", lw=1.0, alpha=0.55, label="slope 1 (diffusive)")
    # slope-2: MSD ∝ τ²
    ax.plot(guide_tau, msd_ref * (guide_tau / tau_ref) ** 2,
            "k:",  lw=1.0, alpha=0.55, label="slope 2 (ballistic)")

    # shade fit window
    ax.axvspan(conv["tau_min_fit_ns"], conv["tau_max_fit_ns"],
               alpha=0.07, color="grey", label="fit window")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Lag time $\tau$ (ns)")
    ax.set_ylabel(r"MSD (Å²)")
    ax.legend(fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out = output_dir / f"msd_loglog_{_safe(sys_name)}_{_safe(model_label)}.png"
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
    yh   = result.get("yh_correction")  # None when not available

    tau_min = conv["tau_min_fit_ns"]
    tau_max = conv["tau_max_fit_ns"]

    fig = plt.figure(figsize=(14, 15))
    gs  = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])   # spans full bottom row
    axs = [ax1, ax2, ax3, ax4, ax5]

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
    if yh is not None:
        axs[1].axhline(yh["D_cat_0_1e10"], color=_SPECIES_COLORS["cat"],
                       lw=1.2, ls="--", alpha=0.7, label=f"{cat} D₀(YH)={yh['D_cat_0_1e10']:.4f}")
        if yh["D_ani_0_1e10"] is not None:
            axs[1].axhline(yh["D_ani_0_1e10"], color=_SPECIES_COLORS["anion"],
                           lw=1.0, ls="--", alpha=0.7, label=f"{ani} D₀(YH)={yh['D_ani_0_1e10']:.4f}")
        if yh["D_sol_0_1e10"] is not None:
            axs[1].axhline(yh["D_sol_0_1e10"], color=_SPECIES_COLORS["solvent"],
                           lw=1.0, ls="--", alpha=0.7, label=f"{sol} D₀(YH)={yh['D_sol_0_1e10']:.4f}")
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
        if yh is not None:
            axs[3].axhline(yh["D_cat_0_1e10"], color=_SPECIES_COLORS["cat"],
                           lw=1.2, ls="--", alpha=0.7, label=f"{cat} D₀(YH)={yh['D_cat_0_1e10']:.4f}")
            if yh["D_ani_0_1e10"] is not None:
                axs[3].axhline(yh["D_ani_0_1e10"], color=_SPECIES_COLORS["anion"],
                               lw=1.0, ls="--", alpha=0.7, label=f"{ani} D₀(YH)={yh['D_ani_0_1e10']:.4f}")
            if yh["D_sol_0_1e10"] is not None:
                axs[3].axhline(yh["D_sol_0_1e10"], color=_SPECIES_COLORS["solvent"],
                               lw=1.0, ls="--", alpha=0.7, label=f"{sol} D₀(YH)={yh['D_sol_0_1e10']:.4f}")
        axs[3].set_xlabel(f"Window end (ns)  [Δt = {win:.1f} ns]")
        axs[3].set_ylabel(r"D (×10⁻¹⁰ m²/s)")
        axs[3].set_title(f"Sliding Window D\n(flat → diffusive regime stable)")
        axs[3].legend(fontsize=9)
    else:
        axs[3].text(0.5, 0.5, f"slide_window_ns ({win:.1f}) > tau_max ({tau_max:.1f})\n— window too wide",
                    ha="center", va="center", transform=axs[3].transAxes)
        axs[3].set_visible(True)

    # ── Panel 5: D vs trajectory length (t_end) ───────────────────────────────
    t_end = conv.get("t_end_ns", np.array([]))
    if len(t_end) > 1:
        axs[4].plot(t_end, conv["D_cat_tlen"],  color=_SPECIES_COLORS["cat"],
                    lw=1.5, marker="o", ms=4, label=cat)
        if not np.all(np.isnan(conv["D_ani_tlen"])):
            axs[4].plot(t_end, conv["D_ani_tlen"],  color=_SPECIES_COLORS["anion"],
                        lw=1.3, marker="s", ms=4, alpha=0.9, label=ani)
        if not np.all(np.isnan(conv["D_sol_tlen"])):
            axs[4].plot(t_end, conv["D_sol_tlen"],  color=_SPECIES_COLORS["solvent"],
                        lw=1.3, marker="^", ms=4, alpha=0.9, label=sol)
        axs[4].axvline(tau_max, color="black", lw=1.0, ls=":", label=f"full traj {tau_max:.1f} ns")
        if yh is not None:
            axs[4].axhline(yh["D_cat_0_1e10"], color=_SPECIES_COLORS["cat"],
                           lw=1.2, ls="--", alpha=0.7, label=f"{cat} D₀(YH)={yh['D_cat_0_1e10']:.4f}")
            if yh["D_ani_0_1e10"] is not None:
                axs[4].axhline(yh["D_ani_0_1e10"], color=_SPECIES_COLORS["anion"],
                               lw=1.0, ls="--", alpha=0.7, label=f"{ani} D₀(YH)={yh['D_ani_0_1e10']:.4f}")
            if yh["D_sol_0_1e10"] is not None:
                axs[4].axhline(yh["D_sol_0_1e10"], color=_SPECIES_COLORS["solvent"],
                               lw=1.0, ls="--", alpha=0.7, label=f"{sol} D₀(YH)={yh['D_sol_0_1e10']:.4f}")
        axs[4].set_xlabel("Trajectory length used t_end (ns)")
        axs[4].set_ylabel(r"D (×10⁻¹⁰ m²/s)")
        axs[4].set_title(
            "D vs Trajectory Length\n"
            "(MSD recomputed at each t_end — flat → enough data for stable statistics)"
        )
        axs[4].legend(fontsize=9)
    else:
        axs[4].text(0.5, 0.5, "trajectory too short for t_end sweep",
                    ha="center", va="center", transform=axs[4].transAxes)

    fig.suptitle(
        f"{model_label}  |  {sys_name}\n"
        f"fit: [{tau_min:.1f}–{tau_max:.1f} ns]  ({result['fit_pct']*100:.0f}% of max lag)",
        fontsize=12, fontweight="bold", y=0.995,
    )
    out = output_dir / f"convergence_{_safe(sys_name)}_{_safe(model_label)}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
