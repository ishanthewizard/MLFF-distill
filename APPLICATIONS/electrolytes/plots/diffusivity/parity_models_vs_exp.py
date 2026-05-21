#!/usr/bin/env python3
"""Parity plots: simulated vs experimental diffusivity for 3 models × 3 species.

Layout (3 rows × 3 cols):
  rows  — model: OPLS baseline | original student | micro student
  cols  — species: cation | anion | solvent

X-axis: experimental D from All_data - main.csv
Y-axis: simulated D from each model's diffusivity.csv
Cation & anion: 0.1M systems excluded. Solvent: all concentrations included.
Each point = one system; legend shows system name.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ── paths ─────────────────────────────────────────────────────────────────────
ANALYSIS_ROOT = Path(
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5024/distillation_project/results/analysis"
)
EXP_CSV = Path(
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/plotting/results/All_data - main.csv"
)
OUT_DIR = Path(
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/APPLICATIONS/electrolytes/plots/diffusivity"
)

MODEL_DIRS = {
    "OPLS":     ANALYSIS_ROOT / "20260509_225342_rdf_density_energy_msd_OPLS_baseline",
    "original": ANALYSIS_ROOT / "20260510_234155_rdf_density_energy_msd_origin",
    "micro":    ANALYSIS_ROOT / "20260510_234129_rdf_density_energy_msd_micro",
}
MODEL_ORDER  = ["OPLS", "original", "micro"]
MODEL_LABELS = {
    "OPLS":     "OPLS Baseline",
    "original": "Original Student",
    "micro":    "Micro Student",
}

SPECIES       = ["cation", "anion", "solvent"]
SIM_COLS      = {"cation": "D_cat_1e-10_m2s",
                 "anion":  "D_ani_1e-10_m2s",
                 "solvent": "D_sol_1e-10_m2s"}
EXP_COLS      = {"cation": "exp student cation diffusivity",
                 "anion":  "exp anion diffusivity",
                 "solvent": "exp solvent diffusivity"}

# simulation solvent name → experimental CSV solvent name
SOLVENT_NORM  = {"Diglyme": "DEGDME"}

# ── helpers ───────────────────────────────────────────────────────────────────

def _temp_float(t: str) -> float:
    """'298K' or '273.2K' → float."""
    return float(str(t).rstrip("K").strip())


def _parse_conc_temp(system: str):
    """'NaOTf/TGDME 0.1M 298K' → ('0.1M', '298K')."""
    parts = system.strip().split()
    return parts[-2], parts[-1]   # concentration, temperature


def load_sim(model_dir: Path) -> pd.DataFrame:
    dfs = [pd.read_csv(csv) for csv in model_dir.glob("*/diffusivity.csv")]
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df[["conc", "temp"]] = df["system"].apply(
        lambda s: pd.Series(_parse_conc_temp(s))
    )
    df["solvent_norm"] = df["solvent_symbol"].replace(SOLVENT_NORM)
    return df


def load_exp() -> pd.DataFrame:
    df = pd.read_csv(EXP_CSV)
    df["temp_float"] = df["temperature"].apply(_temp_float)
    df["conc_str"]   = df["solute concentration"].astype(str).str.strip()
    for col in EXP_COLS.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def merge_exp(sim_df: pd.DataFrame, exp_df: pd.DataFrame) -> pd.DataFrame:
    """Join simulated rows to experimental values on cation/anion/solvent/conc/temp."""
    rows = []
    for _, r in sim_df.iterrows():
        t_sim = _temp_float(r["temp"])
        mask  = (
            (exp_df["cation"]     == r["cat_symbol"])    &
            (exp_df["anion"]      == r["anion_symbol"])  &
            (exp_df["solvent"]    == r["solvent_norm"])  &
            (exp_df["conc_str"]   == r["conc"].strip())  &
            (exp_df["temp_float"].apply(lambda t: abs(t - t_sim) <= 1.0))
        )
        hits = exp_df[mask]
        rec  = r.to_dict()
        if len(hits):
            m = hits.iloc[0]
            rec["exp_cat"] = m[EXP_COLS["cation"]]
            rec["exp_ani"] = m[EXP_COLS["anion"]]
            rec["exp_sol"] = m[EXP_COLS["solvent"]]
        else:
            rec["exp_cat"] = rec["exp_ani"] = rec["exp_sol"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def _label(row) -> str:
    """'Na/OTf/DME 0.5M 273K' — human-readable system id."""
    return (f"{row['cat_symbol']}/{row['anion_symbol']}"
            f"/{row['solvent_symbol']} {row['conc']} {row['temp']}")


# ── plotting ──────────────────────────────────────────────────────────────────

def _draw_panel(ax, x: np.ndarray, y: np.ndarray, sys_labels: list[str],
                title: str, show_ylabel: bool, show_xlabel: bool,
                color_of: dict) -> None:
    # x = experimental, y = simulated
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    xv, yv, lv = x[valid], y[valid], [l for l, ok in zip(sys_labels, valid) if ok]

    for lbl, xi, yi in zip(lv, xv, yv):
        ax.scatter(xi, yi, s=55, alpha=0.85,
                   color=color_of[lbl], edgecolors="k", linewidths=0.4,
                   label=lbl, zorder=3)

    # parity line
    if len(xv):
        lo = min(xv.min(), yv.min()) * 0.85
        hi = max(xv.max(), yv.max()) * 1.20
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    # R² and RMSE
    if len(xv) >= 2:
        r, _   = pearsonr(xv, yv)
        rmse   = np.sqrt(np.mean((yv - xv) ** 2))
        mae    = np.mean(np.abs(yv - xv))
        ax.text(0.97, 0.04,
                f"R² = {r**2:.2f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}",
                transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7", pad=2))

    # deduplicated legend — placed outside to the right
    handles, labels = ax.get_legend_handles_labels()
    seen: dict = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(),
              fontsize=5.8, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              framealpha=0.85, handlelength=1.2, handletextpad=0.4,
              borderpad=0.5, labelspacing=0.25)

    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    if show_xlabel:
        ax.set_xlabel("Experimental  (×10⁻¹⁰ m²/s)", fontsize=8)
    if show_ylabel:
        ax.set_ylabel("Simulated  (×10⁻¹⁰ m²/s)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linestyle=":", alpha=0.45, zorder=0)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    exp_df = load_exp()

    # load & merge all models
    model_data: dict[str, pd.DataFrame] = {}
    for name, mdir in MODEL_DIRS.items():
        sim_df = load_sim(mdir)
        if sim_df.empty:
            print(f"[warn] no diffusivity CSVs found for {name}")
            continue
        merged = merge_exp(sim_df, exp_df)
        n_matched = merged[["exp_cat", "exp_ani", "exp_sol"]].notna().any(axis=1).sum()
        print(f"{name}: {len(merged)} systems loaded, {n_matched} matched to experiment")
        model_data[name] = merged

    exp_keys = {"cation": "exp_cat", "anion": "exp_ani", "solvent": "exp_sol"}

    fig, axes = plt.subplots(3, 3, figsize=(18, 14),
                             gridspec_kw={"hspace": 0.42, "wspace": 0.55})
    fig.suptitle(
        "Diffusivity Parity — Experimental vs Simulated  (×10⁻¹⁰ m²/s)\n"
        "Cation & anion: ≥0.5M only  |  Solvent: all concentrations",
        fontsize=12, fontweight="bold",
    )

    for row_i, model in enumerate(MODEL_ORDER):
        # build a shared color map for all systems in this model row
        if model in model_data:
            all_labels = model_data[model].apply(_label, axis=1).tolist()
            uniq_all   = list(dict.fromkeys(all_labels))
            cmap       = plt.get_cmap("tab20")
            color_of   = {lbl: cmap(i % 20 / 20) for i, lbl in enumerate(uniq_all)}
        else:
            color_of = {}

        for col_j, sp in enumerate(SPECIES):
            ax = axes[row_i, col_j]

            if model not in model_data:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10)
                ax.set_title(f"{MODEL_LABELS[model]} — {sp.capitalize()}",
                             fontsize=9, fontweight="bold")
                continue

            df = model_data[model].dropna(subset=[SIM_COLS[sp], exp_keys[sp]])
            # exclude 0.1M for cation and anion; keep all for solvent
            if sp in ("cation", "anion"):
                df = df[df["conc"].str.strip() != "0.1M"]
            x     = pd.to_numeric(df[exp_keys[sp]],  errors="coerce").to_numpy()  # experimental
            y     = pd.to_numeric(df[SIM_COLS[sp]],  errors="coerce").to_numpy()  # simulated
            lbls  = df.apply(_label, axis=1).tolist()
            title = f"{MODEL_LABELS.get(model, model)} — {sp.capitalize()}"

            _draw_panel(ax, x, y, lbls, title,
                        show_ylabel=(col_j == 0),
                        show_xlabel=(row_i == 2),
                        color_of=color_of)

    out = OUT_DIR / "diffusivity_parity_models.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
