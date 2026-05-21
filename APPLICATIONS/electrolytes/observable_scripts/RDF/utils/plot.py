#!/usr/bin/env python3
"""RDF plotting functions.

All functions save figures to disk and return the output path(s).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.linestyle": ":"})


def plot_rdf_comparison(
    rdf_results: dict,
    sys_name: str,
    model_order: list[str],
    model_colors: dict[str, str],
    output_dir: Path,
    r_max: float = 10.0,
) -> Path:
    """Plot g(r) and n(r) comparison across models for one system.

    rdf_results: {pair_label: {model_name: DataFrame(r, g_r, n_r)}}
    Returns path to saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = list(rdf_results.keys())
    n_pairs = len(pairs)
    fig, axes = plt.subplots(n_pairs, 2, figsize=(12, 4 * n_pairs), squeeze=False)
    fig.suptitle(f"{sys_name} — RDF Comparison", fontsize=14, fontweight="bold")

    for row, pair_label in enumerate(pairs):
        models_data = rdf_results[pair_label]
        for model_name in model_order:
            if model_name not in models_data:
                continue
            df = models_data[model_name]
            color = model_colors.get(model_name, None)
            axes[row, 0].plot(df["r"], df["g_r"], label=model_name, color=color, lw=1.5)
            axes[row, 1].plot(df["r"], df["n_r"], label=model_name, color=color, lw=1.5)
        axes[row, 0].set_ylabel(f"{pair_label}  $g(r)$")
        axes[row, 0].set_xlim(0, r_max)
        axes[row, 1].set_ylabel(f"{pair_label}  $n(r)$")
        axes[row, 1].set_xlim(0, r_max)
        axes[row, 0].legend(fontsize=9)

    for ax in axes[-1]:
        ax.set_xlabel(r"$r$ (Å)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    safe_name = sys_name.replace("/", "_").replace(" ", "_")
    out_path = output_dir / f"rdf_{safe_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_rdf_sliding(
    result: dict,
    sys_name: str,
    model_label: str,
    pair_label: str,
    output_dir: Path,
    r_max: float = 10.0,
) -> tuple[Path, Path]:
    """Plot sliding-window g(r) and n(r) for one system/model/pair.

    result: output dict from compute_rdf_sliding.
    Returns (path_gr_png, path_nr_png).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    r_mid = result["r_mid"]
    all_gr = result["all_gr"]
    all_nr = result["all_nr"]
    mean_gr = result["mean_gr"]
    std_gr = result["std_gr"]
    mean_nr = result["mean_nr"]
    std_nr = result["std_nr"]
    windows = result["windows"]
    n_win = all_gr.shape[0]
    t_starts = np.array([w[0] for w in windows])

    cmap = cm.get_cmap("coolwarm", n_win)
    safe = lambda s: s.replace("/", "_").replace(" ", "_")

    # g(r)
    fig, (ax_rdf, ax_std) = plt.subplots(
        2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [3, 1.2]}
    )
    fig.suptitle(
        f"{model_label}  |  {sys_name}  |  {pair_label}\nSliding window g(r)",
        fontsize=11, fontweight="bold",
    )
    for wi, g in enumerate(all_gr):
        ax_rdf.plot(r_mid, g, color=cmap(wi / max(n_win - 1, 1)), lw=0.8, alpha=0.85)
    ax_rdf.plot(r_mid, mean_gr, color="black", lw=1.8, ls="--", label="mean")
    ax_rdf.set_ylabel(r"$g(r)$")
    ax_rdf.set_xlim(0, r_max)
    ax_rdf.legend(fontsize=9)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(t_starts[0], t_starts[-1]))
    sm.set_array([])
    fig.colorbar(sm, ax=ax_rdf, pad=0.01).set_label("Window start (ns)")
    ax_std.fill_between(r_mid, 0, std_gr, alpha=0.4, color="steelblue")
    ax_std.plot(r_mid, std_gr, color="steelblue", lw=1.2, label="std")
    ax_std.set_ylabel(r"$\sigma[g(r)]$")
    ax_std.set_xlabel(r"$r$ (Å)")
    ax_std.set_xlim(0, r_max)
    ax_std.legend(fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_gr = output_dir / f"sliding_gr_{safe(sys_name)}_{safe(model_label)}_{pair_label}.png"
    fig.savefig(out_gr, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # n(r)
    fig2, (ax_nr, ax_nstd) = plt.subplots(
        2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [3, 1.2]}
    )
    fig2.suptitle(
        f"{model_label}  |  {sys_name}  |  {pair_label}\nSliding window n(r)",
        fontsize=11, fontweight="bold",
    )
    for wi, nw in enumerate(all_nr):
        ax_nr.plot(r_mid, nw, color=cmap(wi / max(n_win - 1, 1)), lw=0.8, alpha=0.85)
    ax_nr.plot(r_mid, mean_nr, color="black", lw=1.8, ls="--", label="mean")
    ax_nr.set_ylabel(r"$n(r)$")
    ax_nr.set_xlim(0, r_max)
    ax_nr.legend(fontsize=9)
    sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(t_starts[0], t_starts[-1]))
    sm2.set_array([])
    fig2.colorbar(sm2, ax=ax_nr, pad=0.01).set_label("Window start (ns)")
    ax_nstd.fill_between(r_mid, 0, std_nr, alpha=0.4, color="darkorange")
    ax_nstd.plot(r_mid, std_nr, color="darkorange", lw=1.2, label="std")
    ax_nstd.set_ylabel(r"$\sigma[n(r)]$")
    ax_nstd.set_xlabel(r"$r$ (Å)")
    ax_nstd.set_xlim(0, r_max)
    ax_nstd.legend(fontsize=9)
    fig2.tight_layout(rect=(0, 0, 1, 0.93))
    out_nr = output_dir / f"sliding_nr_{safe(sys_name)}_{safe(model_label)}_{pair_label}.png"
    fig2.savefig(out_nr, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    return out_gr, out_nr
