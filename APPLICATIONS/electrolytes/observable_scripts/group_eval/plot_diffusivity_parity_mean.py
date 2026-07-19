"""Replica-aggregated diffusivity parity plots from a diffusivity_with_exp_all.csv.

Same layout/filters/colors as ``plot_diffusivity_parity.py`` (3 species x
{corrected, uncorrected} x {log, linear} = 12 plots; cation/anion restricted to
concentration_M >= 0.5; color by cation-anion-solvent-temperature), but instead
of one point per replica each SYSTEM is collapsed to:

  - a dot at the MEAN of its replicas' MD diffusivity, and
  - a vertical bar spanning [min, max] across the replicas.

The experimental value (x) is shared by all replicas of a system. logMAE /
Spearman are computed on the per-system means.

Usage:
  python plot_diffusivity_parity_mean.py <diffusivity_with_exp_all.csv> [--output-dir <dir>]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from parity_plot import _stats, _log_lims
from properties import PROPERTIES

META = ("cat_symbol", "anion_symbol", "solvent_symbol", "concentration_M", "temperature_K")


def _aggregate(df, exp_col, sim_col):
    """Collapse replicas -> one row per system with mean/min/max of sim_col."""
    rows = []
    for system, sub in df.groupby("system", sort=False):
        sim = sub[sim_col].to_numpy(dtype=float)
        sim = sim[np.isfinite(sim)]
        exp = float(sub[exp_col].iloc[0])
        if sim.size == 0 or not np.isfinite(exp) or exp <= 0:
            continue
        f = sub.iloc[0]
        rows.append({
            "system": system, "exp": exp,
            "mean": float(np.mean(sim)), "lo": float(np.min(sim)),
            "hi": float(np.max(sim)), "n_rep": int(sim.size),
            "cat": f["cat_symbol"], "an": f["anion_symbol"], "sol": f["solvent_symbol"],
            "conc": float(f["concentration_M"]), "temp": float(f["temperature_K"]),
        })
    return pd.DataFrame(rows)


def plot_mean_minmax(csv_path, exp_col, sim_col, output, xlabel, ylabel, title, log_scale):
    df = pd.read_csv(csv_path)
    df = df[df[exp_col].notna() & (df[exp_col] > 0)].copy()
    agg = _aggregate(df, exp_col, sim_col)

    fig, ax = plt.subplots(figsize=(7, 7))
    vals = list(agg["exp"]) + list(agg["lo"]) + list(agg["hi"]) + list(agg["mean"])
    if log_scale:
        lo, hi = _log_lims(vals)
        ax.set_xscale("log"); ax.set_yscale("log")
    else:
        finite = [v for v in vals if np.isfinite(v)]
        lo, hi = (0, max(finite) * 1.05) if finite else (0, 1)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="1:1", zorder=1)

    # color by cation-anion-solvent-temperature (multi-temp salts get distinct colors)
    sys_keys = (agg["cat"] + "-" + agg["an"] + "-" + agg["sol"]
                + "-" + agg["temp"].astype(int).astype(str) + "K")
    uniq = sorted(sys_keys.unique())
    cmap = plt.get_cmap("tab10" if len(uniq) <= 10 else "tab20")
    color_of = {s: cmap(i % cmap.N) for i, s in enumerate(uniq)}

    seen = set()
    for _, r in agg.iterrows():
        key = f"{r['cat']}-{r['an']}-{r['sol']}-{int(r['temp'])}K"
        label = f"{r['cat']}-{r['an']}-{r['sol']}, {r['conc']:g}M, {r['temp']:.0f}K"
        yerr = [[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]]   # asymmetric min..max
        ax.errorbar(r["exp"], r["mean"], yerr=yerr, fmt="o", ms=7,
                    color=color_of[key], ecolor=color_of[key], elinewidth=1.3,
                    capsize=3, capthick=1.3, alpha=0.9,
                    markeredgecolor="gray", markeredgewidth=0.5,
                    label=(label if label not in seen else "_nolegend_"), zorder=3)
        seen.add(label)

    log_mae, rho, n = _stats(agg["exp"], agg["mean"])
    text = (f"PAINN replica-mean (n={n}): logMAE={log_mae:.2f}, Spearman={rho:.2f}\n"
            f"bars = replica min..max")

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel(xlabel, fontsize=11); ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0.0, ncol=1)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.text(0.97, 0.03, text, transform=ax.transAxes, va="bottom", ha="right",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
    fig.tight_layout()
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {output.name}  (n={n}, logMAE={log_mae:.2f}, Spearman={rho:.2f})")
    return output


def run(csv_path, output_dir=None):
    csv_path = Path(csv_path)
    output_dir = Path(output_dir) if output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df_all = pd.read_csv(csv_path)
    df_hico = df_all[df_all["concentration_M"] >= 0.5]
    tmp_all, tmp_hico = output_dir / "_tmp_all.csv", output_dir / "_tmp_hico.csv"
    df_all.to_csv(tmp_all, index=False); df_hico.to_csv(tmp_hico, index=False)

    JOBS = [
        ("diffusivity_cation", tmp_hico), ("diffusivity_anion", tmp_hico),
        ("diffusivity_solvent", tmp_all),
        ("diffusivity_cation_uncorrected", tmp_hico),
        ("diffusivity_anion_uncorrected", tmp_hico),
        ("diffusivity_solvent_uncorrected", tmp_all),
    ]
    for prop, csv in JOBS:
        cfg = PROPERTIES[prop]
        for log_scale, suffix in [(True, "log"), (False, "linear")]:
            plot_mean_minmax(
                csv, cfg["exp_col"], cfg["sim_col"],
                output=output_dir / f"diffusivity_parity_{prop}_{suffix}.png",
                xlabel=cfg["xlabel"], ylabel=cfg["ylabel"],
                title=f'{prop.replace("_", " ")} ({suffix}) - replica mean +/- min..max',
                log_scale=log_scale)
    tmp_all.unlink(missing_ok=True); tmp_hico.unlink(missing_ok=True)
    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--output-dir", default=None)
    a = p.parse_args()
    run(a.csv, a.output_dir)
