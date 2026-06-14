#!/usr/bin/env python3
"""
Diffusivity parity plots — micro student vs experiment + outlier comparison.

Output:
  1. outlier_diffusivity_3methods.csv  — collected D for 4 outlier systems × 3 new methods
  2. parity_micro_student.png          — 3-panel parity: micro student vs experiment
  3. parity_outlier_nvt_uma.png        — all original + NVT-UMA-density outlier corrections
  4. parity_outlier_nvt_exp.png        — all original + NVT-exp-density outlier corrections
  5. parity_outlier_npt_uma.png        — all original + NPT-UMA-density outlier corrections

Outlier systems (NPT-anisotropic berendsen runs that diverged from experiment):
  li_pf6_dme_323K_0.5M, na_otf_dme_298K_0.1M,
  na_pf6_dme_298K_0.1M, na_pf6_dme_323K_0.5M
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Patch
from scipy.stats import pearsonr

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = Path("/global/homes/y/yuejian/project/MLFF-distill")

MICRO_DIR = (BASE / "yuejian/electrolyte_application/analysis"
             "/20ns_micro/20260510_234129_rdf_density_energy_msd")
EXP_CSV   = (BASE / "m5024/distillation_project/experiment_data"
             "/src_data/diffusivity/All_data - exp_with_density.csv")

NVT_UMA_DIR = (BASE / "yuejian/electrolyte_application/distort_temp_md_nvt"
               "/analysis/20260611_001221_msd_cell_size_pressure_energy")
NVT_EXP_DIR = (BASE / "yuejian/electrolyte_application/exp_density_nvt"
               "/analysis/20260611_001221_msd_cell_size_pressure_energy")
NPT_UMA_DIR = (BASE / "yuejian/electrolyte_application/uma_density_isotropic_mtk_npt"
               "/analysis/20260611_001221_msd_cell_size_pressure_energy")

PLOT_DIR = BASE / "m5024/distillation_project/results/analysis/diffusivity"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────────
SIM_COLS = {
    "cation":  "D_cat_1e-10_m2s",
    "anion":   "D_ani_1e-10_m2s",
    "solvent": "D_sol_1e-10_m2s",
}
EXP_COLS = {
    "cation":  "exp student cation diffusivity",
    "anion":   "exp anion diffusivity",
    "solvent": "exp solvent diffusivity",
}
SOLVENT_NORM = {"Diglyme": "DEGDME"}

# 4 outlier system directory names (identical across all 3 new-method dirs)
OUTLIER_DIRS = [
    "LiPF6_DME_323K_0.5M",
    "NaOTf_DME_298K_0.1M",
    "NaPF6_DME_298K_0.1M",
    "NaPF6_DME_323K_0.5M",
]
# canonical label used in the micro student data (matches _label() output)
OUTLIER_LABELS = {
    "LiPF6_DME_323K_0.5M": "Li/PF6/DME 0.5M 323K",
    "NaOTf_DME_298K_0.1M": "Na/OTf/DME 0.1M 298K",
    "NaPF6_DME_298K_0.1M": "Na/PF6/DME 0.1M 298K",
    "NaPF6_DME_323K_0.5M": "Na/PF6/DME 0.5M 323K",
}
# same mapping but keyed by label (for reverse look-up)
OUTLIER_LABEL_SET = set(OUTLIER_LABELS.values())
LABEL_TO_SYSDIR   = {v: k for k, v in OUTLIER_LABELS.items()}

# marker shape per outlier system (shared across original + all correction methods)
OUTLIER_MARKERS = {
    "LiPF6_DME_323K_0.5M": "o",
    "NaOTf_DME_298K_0.1M": "s",
    "NaPF6_DME_298K_0.1M": "^",
    "NaPF6_DME_323K_0.5M": "D",
}

METHOD_CONFIGS = [
    ("NVT UMA density",  NVT_UMA_DIR,  "teacher",    "#2196F3"),   # blue
    ("NVT exp density",  NVT_EXP_DIR,  "teacher",    "#4CAF50"),   # green
    ("NPT UMA density",  NPT_UMA_DIR,  "teacher",    "#FF9800"),   # orange
]

# ── helpers ───────────────────────────────────────────────────────────────────

def _temp_float(t: str) -> float:
    return float(str(t).rstrip("K").strip())


def _parse_conc_temp(system: str):
    parts = system.strip().split()
    return parts[-2], parts[-1]


def _label(row) -> str:
    return (f"{row['cat_symbol']}/{row['anion_symbol']}"
            f"/{row['solvent_symbol']} {row['conc']} {row['temp']}")


def load_sim(sim_dir: Path) -> pd.DataFrame:
    dfs = [pd.read_csv(p) for p in sim_dir.glob("*/diffusivity.csv")]
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
    rows = []
    for _, r in sim_df.iterrows():
        t_sim = _temp_float(r["temp"])
        mask  = (
            (exp_df["cation"]    == r["cat_symbol"])   &
            (exp_df["anion"]     == r["anion_symbol"]) &
            (exp_df["solvent"]   == r["solvent_norm"]) &
            (exp_df["conc_str"]  == r["conc"].strip()) &
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


# ── step 1: collect outlier data → CSV ────────────────────────────────────────

def collect_outlier_csv(exp_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method_name, base_dir, model_key, _ in METHOD_CONFIGS:
        for sys_dir in OUTLIER_DIRS:
            csv_path = base_dir / sys_dir / "diffusivity.csv"
            if not csv_path.exists():
                print(f"[warn] missing: {csv_path}")
                continue
            df = pd.read_csv(csv_path)
            row = df[df["model"] == model_key]
            if row.empty:
                print(f"[warn] no '{model_key}' row in {csv_path}")
                continue
            r = row.iloc[0]
            # system string may be "LiPF6/DME 323K 0.5M" (temp before conc)
            # or "LiPF6 0.5M 323K" (conc before temp) — find by suffix
            parts = r["system"].strip().split()
            temp_parts = [p for p in parts if p.upper().endswith("K") and p[:-1].replace(".", "").isdigit()]
            conc_parts = [p for p in parts if p.upper().endswith("M") and p[:-1].replace(".", "").isdigit()]
            t_sim = _temp_float(temp_parts[0]) if temp_parts else np.nan
            conc  = conc_parts[0] if conc_parts else ""
            mask  = (
                (exp_df["cation"]    == r["cat_symbol"])   &
                (exp_df["anion"]     == r["anion_symbol"]) &
                (exp_df["solvent"]   == r["solvent_symbol"].replace("Diglyme", "DEGDME")) &
                (exp_df["conc_str"]  == conc.strip())      &
                (exp_df["temp_float"].apply(lambda t: abs(t - t_sim) <= 1.0))
            )
            hits = exp_df[mask]
            exp_c = hits.iloc[0][EXP_COLS["cation"]]  if len(hits) else np.nan
            exp_a = hits.iloc[0][EXP_COLS["anion"]]   if len(hits) else np.nan
            exp_s = hits.iloc[0][EXP_COLS["solvent"]] if len(hits) else np.nan
            rows.append({
                "method":          method_name,
                "system_dir":      sys_dir,
                "system":          r["system"],
                "D_cat_sim":       r["D_cat_1e-10_m2s"],
                "D_ani_sim":       r["D_ani_1e-10_m2s"],
                "D_sol_sim":       r["D_sol_1e-10_m2s"],
                "D_cat_exp":       exp_c,
                "D_ani_exp":       exp_a,
                "D_sol_exp":       exp_s,
            })
    out_csv = PLOT_DIR / "outlier_diffusivity_3methods.csv"
    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"Saved CSV → {out_csv}")
    return result_df


# ── step 2: parity plot (micro student vs experiment) ─────────────────────────

def _draw_parity_panel(ax, df_sp: pd.DataFrame, x_col: str, y_col: str,
                       title: str, show_ylabel: bool, show_xlabel: bool,
                       color_of: dict) -> None:
    df_v = df_sp.dropna(subset=[x_col, y_col])
    df_v = df_v[(pd.to_numeric(df_v[x_col], errors="coerce") > 0) &
                (pd.to_numeric(df_v[y_col], errors="coerce") > 0)]
    x = pd.to_numeric(df_v[x_col], errors="coerce").to_numpy()
    y = pd.to_numeric(df_v[y_col], errors="coerce").to_numpy()
    lbls = df_v.apply(_label, axis=1).tolist()

    for lbl, xi, yi in zip(lbls, x, y):
        ax.scatter(xi, yi, s=55, alpha=0.85, color=color_of.get(lbl, "#888888"),
                   edgecolors="k", linewidths=0.4, label=lbl, zorder=3)

    if len(x):
        lo = min(x.min(), y.min()) * 0.80
        hi = max(x.max(), y.max()) * 1.25
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    if len(x) >= 2:
        r, _  = pearsonr(x, y)
        rmse  = np.sqrt(np.mean((y - x) ** 2))
        ax.text(0.97, 0.04,
                f"R² = {r**2:.2f}\nRMSE = {rmse:.2f}",
                transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7", pad=2))

    handles, labels_ = ax.get_legend_handles_labels()
    seen: dict = {}
    for h, l in zip(handles, labels_):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(),
              fontsize=5.5, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              framealpha=0.85, handlelength=1.2, handletextpad=0.4,
              borderpad=0.5, labelspacing=0.25)

    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    if show_xlabel:
        ax.set_xlabel("Experimental  (×10⁻¹⁰ m²/s)", fontsize=8)
    if show_ylabel:
        ax.set_ylabel("Simulated  (×10⁻¹⁰ m²/s)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linestyle=":", alpha=0.45, zorder=0)


def plot_micro_student(merged: pd.DataFrame) -> None:
    species_cfg = [
        ("cation",  "exp_cat", SIM_COLS["cation"],  "Micro Student — Cation"),
        ("anion",   "exp_ani", SIM_COLS["anion"],   "Micro Student — Anion"),
        ("solvent", "exp_sol", SIM_COLS["solvent"], "Micro Student — Solvent"),
    ]

    all_labels = merged.apply(_label, axis=1).tolist()
    uniq = list(dict.fromkeys(all_labels))
    cmap = plt.get_cmap("tab20")
    color_of = {lbl: cmap(i % 20 / 20) for i, lbl in enumerate(uniq)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                             gridspec_kw={"wspace": 0.55})
    fig.suptitle(
        "Diffusivity Parity — Micro Student vs Experimental  (×10⁻¹⁰ m²/s)",
        fontsize=12, fontweight="bold",
    )

    for j, (sp, exp_col, sim_col, title) in enumerate(species_cfg):
        _draw_parity_panel(
            axes[j], merged, exp_col, sim_col, title,
            show_ylabel=(j == 0), show_xlabel=True, color_of=color_of,
        )

    out = PLOT_DIR / "parity_micro_student.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── step 3: comparison plots (one per new method) ─────────────────────────────

def _draw_comparison_panel(ax, micro_df: pd.DataFrame,
                           outlier_df: pd.DataFrame,
                           x_col_micro: str, y_col_micro: str,
                           x_col_new: str, y_col_new: str,
                           title: str, method_label: str, method_color: str,
                           show_ylabel: bool, show_xlabel: bool) -> None:
    # split micro into outlier vs non-outlier
    micro_df = micro_df.copy()
    micro_df["_label"] = micro_df.apply(_label, axis=1)
    micro_df["_is_outlier"] = micro_df["_label"].isin(OUTLIER_LABEL_SET)

    normal = micro_df[~micro_df["_is_outlier"]].dropna(subset=[x_col_micro, y_col_micro])
    outlier_micro = micro_df[micro_df["_is_outlier"]].dropna(subset=[x_col_micro, y_col_micro])

    xn = pd.to_numeric(normal[x_col_micro], errors="coerce").to_numpy()
    yn = pd.to_numeric(normal[y_col_micro], errors="coerce").to_numpy()
    xo = pd.to_numeric(outlier_micro[x_col_micro], errors="coerce").to_numpy()
    yo = pd.to_numeric(outlier_micro[y_col_micro], errors="coerce").to_numpy()
    xnew = pd.to_numeric(outlier_df[x_col_new], errors="coerce").to_numpy()
    ynew = pd.to_numeric(outlier_df[y_col_new], errors="coerce").to_numpy()

    # filter positives
    mask_n   = (xn > 0) & (yn > 0) & np.isfinite(xn) & np.isfinite(yn)
    mask_o   = (xo > 0) & (yo > 0) & np.isfinite(xo) & np.isfinite(yo)
    mask_new = (xnew > 0) & (ynew > 0) & np.isfinite(xnew) & np.isfinite(ynew)

    xn, yn = xn[mask_n], yn[mask_n]
    xo, yo = xo[mask_o], yo[mask_o]
    xnew_v, ynew_v = xnew[mask_new], ynew[mask_new]
    new_lbls = outlier_df["system_dir"].tolist()
    new_lbls = [new_lbls[i] for i in range(len(new_lbls)) if mask_new[i]]

    # parity line range
    all_x = np.concatenate([xn, xo, xnew_v]) if len(xnew_v) else np.concatenate([xn, xo])
    all_y = np.concatenate([yn, yo, ynew_v]) if len(ynew_v) else np.concatenate([yn, yo])
    if len(all_x):
        lo = min(all_x.min(), all_y.min()) * 0.80
        hi = max(all_x.max(), all_y.max()) * 1.25
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    # non-outlier: grey circles
    if len(xn):
        ax.scatter(xn, yn, s=45, alpha=0.55, color="#AAAAAA",
                   edgecolors="k", linewidths=0.3, label="Non-outlier (micro student)",
                   zorder=2)

    # outlier micro: red circles
    for xi, yi in zip(xo, yo):
        ax.scatter(xi, yi, s=60, alpha=0.85, color="red",
                   edgecolors="darkred", linewidths=0.5,
                   label="Outlier (original micro student)", zorder=3)

    # new method: colored triangles + arrows from original
    already_labeled = False
    for sys_dir, xi_new, yi_new in zip(new_lbls, xnew_v, ynew_v):
        lbl_new = method_label if not already_labeled else "_nolegend_"
        already_labeled = True
        ax.scatter(xi_new, yi_new, s=75, alpha=0.90, color=method_color,
                   marker="^", edgecolors="k", linewidths=0.4,
                   label=lbl_new, zorder=4)
        # draw arrow from original outlier to new point if we can find it
        olbl = OUTLIER_LABELS.get(sys_dir, "")
        m = outlier_micro[outlier_micro["_label"] == olbl]
        if not m.empty:
            xi_orig = pd.to_numeric(m.iloc[0][x_col_micro], errors="coerce")
            yi_orig = pd.to_numeric(m.iloc[0][y_col_micro], errors="coerce")
            if np.isfinite(xi_orig) and np.isfinite(yi_orig):
                ax.annotate("", xy=(xi_new, yi_new), xytext=(xi_orig, yi_orig),
                            arrowprops=dict(arrowstyle="->", color=method_color,
                                           lw=1.0, alpha=0.7))

    # legend (deduplicated)
    handles, labels_ = ax.get_legend_handles_labels()
    seen: dict = {}
    for h, l in zip(handles, labels_):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(),
              fontsize=6.5, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              framealpha=0.85, handlelength=1.2, handletextpad=0.4,
              borderpad=0.5, labelspacing=0.3)

    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    if show_xlabel:
        ax.set_xlabel("Experimental  (×10⁻¹⁰ m²/s)", fontsize=8)
    if show_ylabel:
        ax.set_ylabel("Simulated  (×10⁻¹⁰ m²/s)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linestyle=":", alpha=0.45, zorder=0)


def plot_comparison(micro_merged: pd.DataFrame, outlier_csv: pd.DataFrame) -> None:
    exp_keys = {
        "cation":  ("exp_cat", "D_cat_exp"),
        "anion":   ("exp_ani", "D_ani_exp"),
        "solvent": ("exp_sol", "D_sol_exp"),
    }
    sim_keys = {
        "cation":  (SIM_COLS["cation"],  "D_cat_sim"),
        "anion":   (SIM_COLS["anion"],   "D_ani_sim"),
        "solvent": (SIM_COLS["solvent"], "D_sol_sim"),
    }
    species = ["cation", "anion", "solvent"]
    out_names = ["parity_outlier_nvt_uma.png",
                 "parity_outlier_nvt_exp.png",
                 "parity_outlier_npt_uma.png"]

    for (method_name, _, model_key, method_color), out_fname in zip(METHOD_CONFIGS, out_names):
        method_df = outlier_csv[outlier_csv["method"] == method_name].reset_index(drop=True)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                                 gridspec_kw={"wspace": 0.55})
        fig.suptitle(
            f"Diffusivity Parity — Micro Student (original) + {method_name} corrections\n"
            "Grey: non-outlier  |  Red: original outlier  |  "
            f"{method_color} triangle: {method_name}",
            fontsize=11, fontweight="bold",
        )

        for j, sp in enumerate(species):
            x_micro, y_micro = exp_keys[sp][0], sim_keys[sp][0]
            x_new,   y_new   = exp_keys[sp][1], sim_keys[sp][1]
            title = f"{sp.capitalize()}"
            _draw_comparison_panel(
                axes[j], micro_merged, method_df,
                x_micro, y_micro, x_new, y_new,
                title, method_name, method_color,
                show_ylabel=(j == 0), show_xlabel=True,
            )

        out = PLOT_DIR / out_fname
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out}")


# ── step 4: combined plot — all 3 methods together ────────────────────────────

def _draw_combined_panel(ax, micro_df: pd.DataFrame, outlier_csv: pd.DataFrame,
                         x_col_micro: str, y_col_micro: str,
                         x_col_new: str, y_col_new: str,
                         title: str, show_ylabel: bool, show_xlabel: bool,
                         species: str = "solvent") -> None:
    micro_df = micro_df.copy()
    micro_df["_label"] = micro_df.apply(_label, axis=1)
    micro_df["_is_outlier"] = micro_df["_label"].isin(OUTLIER_LABEL_SET)

    normal        = micro_df[~micro_df["_is_outlier"]].dropna(subset=[x_col_micro, y_col_micro])
    outlier_micro = micro_df[micro_df["_is_outlier"]].dropna(subset=[x_col_micro, y_col_micro])

    # 0.1M systems: only include in solvent panel
    if species in ("cation", "anion"):
        normal        = normal[normal["conc"] != "0.1M"]
        outlier_micro = outlier_micro[outlier_micro["conc"] != "0.1M"]

    xn = pd.to_numeric(normal[x_col_micro], errors="coerce").to_numpy()
    yn = pd.to_numeric(normal[y_col_micro], errors="coerce").to_numpy()
    mask_n = (xn > 0) & (yn > 0) & np.isfinite(xn) & np.isfinite(yn)
    xn, yn = xn[mask_n], yn[mask_n]

    # outlier original: collect (x, y, label, sys_dir) tuples
    o_rows = []
    for _, row in outlier_micro.iterrows():
        xi = pd.to_numeric(row[x_col_micro], errors="coerce")
        yi = pd.to_numeric(row[y_col_micro], errors="coerce")
        if xi > 0 and yi > 0 and np.isfinite(xi) and np.isfinite(yi):
            lbl     = row["_label"]
            sys_dir = LABEL_TO_SYSDIR.get(lbl, "")
            o_rows.append((xi, yi, lbl, sys_dir))

    # axis range — include all three method points
    all_xnew, all_ynew = [], []
    for method_name, _, _, _ in METHOD_CONFIGS:
        mdf = outlier_csv[outlier_csv["method"] == method_name]
        if species in ("cation", "anion"):
            mdf = mdf[~mdf["system_dir"].str.endswith("_0.1M")]
        xnew = pd.to_numeric(mdf[x_col_new], errors="coerce").to_numpy()
        ynew = pd.to_numeric(mdf[y_col_new], errors="coerce").to_numpy()
        mask = (xnew > 0) & (ynew > 0) & np.isfinite(xnew) & np.isfinite(ynew)
        all_xnew.extend(xnew[mask])
        all_ynew.extend(ynew[mask])

    xo_arr = np.array([r[0] for r in o_rows]) if o_rows else np.array([])
    yo_arr = np.array([r[1] for r in o_rows]) if o_rows else np.array([])
    all_x  = np.concatenate([xn, xo_arr] + ([np.array(all_xnew)] if all_xnew else []))
    all_y  = np.concatenate([yn, yo_arr] + ([np.array(all_ynew)] if all_ynew else []))
    if len(all_x):
        lo = min(all_x.min(), all_y.min()) * 0.80
        hi = max(all_x.max(), all_y.max()) * 1.25
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    ann_kw = dict(fontsize=7, ha="left", va="bottom",
                  textcoords="offset points", xytext=(3, 3),
                  bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, lw=0))

    # non-outlier: grey circles, no labels
    if len(xn):
        ax.scatter(xn, yn, s=45, alpha=0.55, color="#AAAAAA",
                   edgecolors="k", linewidths=0.3, zorder=2)

    # original outlier: red + system-specific shape, annotate with system name
    for xi, yi, lbl, sys_dir in o_rows:
        mkr = OUTLIER_MARKERS.get(sys_dir, "o")
        ax.scatter(xi, yi, s=70, alpha=0.85, color="red",
                   edgecolors="darkred", linewidths=0.5, marker=mkr, zorder=3)
        ax.annotate(lbl, (xi, yi), color="darkred", zorder=5, **ann_kw)

    # correction methods: method color + system-specific shape, arrows, no per-point label
    for method_name, _, _, method_color in METHOD_CONFIGS:
        mdf = outlier_csv[outlier_csv["method"] == method_name]
        if species in ("cation", "anion"):
            mdf = mdf[~mdf["system_dir"].str.endswith("_0.1M")]
        mdf = mdf.reset_index(drop=True)
        xnew    = pd.to_numeric(mdf[x_col_new], errors="coerce").to_numpy()
        ynew    = pd.to_numeric(mdf[y_col_new], errors="coerce").to_numpy()
        sys_dirs = mdf["system_dir"].tolist()
        mask    = (xnew > 0) & (ynew > 0) & np.isfinite(xnew) & np.isfinite(ynew)
        for i, (sys_dir, xi_new, yi_new) in enumerate(zip(sys_dirs, xnew, ynew)):
            if not mask[i]:
                continue
            mkr = OUTLIER_MARKERS.get(sys_dir, "o")
            ax.scatter(xi_new, yi_new, s=80, alpha=0.90, color=method_color,
                       marker=mkr, edgecolors="k", linewidths=0.4, zorder=4)
            # arrow from original outlier position
            olbl = OUTLIER_LABELS.get(sys_dir, "")
            for xi_o, yi_o, lbl_o, _ in o_rows:
                if lbl_o == olbl:
                    ax.annotate("", xy=(xi_new, yi_new), xytext=(xi_o, yi_o),
                                arrowprops=dict(arrowstyle="->", color=method_color,
                                               lw=1.0, alpha=0.7))
                    break

    # ── legend: two sections — colors (methods) and shapes (systems) ──
    color_handles = [
        mlines.Line2D([], [], color="#AAAAAA", marker="o", linestyle="None",
                      markersize=6, markeredgecolor="k", markeredgewidth=0.3,
                      label="Non-outlier\n(micro student,\nanisotropic NPT)"),
        mlines.Line2D([], [], color="red", marker="o", linestyle="None",
                      markersize=6, markeredgecolor="darkred", markeredgewidth=0.5,
                      label="Original outlier\n(micro student)"),
    ]
    for method_name, _, _, method_color in METHOD_CONFIGS:
        color_handles.append(
            mlines.Line2D([], [], color=method_color, marker="o", linestyle="None",
                          markersize=6, markeredgecolor="k", markeredgewidth=0.4,
                          label=method_name.replace(" ", "\n", 1))
        )

    shape_handles = [Patch(color="none", label="── Systems ──")]
    for sys_dir, sys_lbl in OUTLIER_LABELS.items():
        if species in ("cation", "anion") and sys_dir.endswith("_0.1M"):
            continue
        mkr = OUTLIER_MARKERS[sys_dir]
        # split "Li/PF6/DME 0.5M 323K" → "Li/PF6/DME\n0.5M 323K"
        parts = sys_lbl.split(" ", 1)
        shape_handles.append(
            mlines.Line2D([], [], color="k", marker=mkr, linestyle="None",
                          markersize=6, label="\n".join(parts))
        )

    ax.legend(handles=color_handles + shape_handles,
              fontsize=8.5, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              framealpha=0.85, handlelength=1.4, handletextpad=0.5,
              borderpad=0.6, labelspacing=0.35)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=4)
    if show_xlabel:
        ax.set_xlabel("Experimental  (×10⁻¹⁰ m²/s)", fontsize=11)
    if show_ylabel:
        ax.set_ylabel("Simulated  (×10⁻¹⁰ m²/s)", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(True, linestyle=":", alpha=0.45, zorder=0)


def plot_combined(micro_merged: pd.DataFrame, outlier_csv: pd.DataFrame) -> None:
    exp_keys = {"cation": "exp_cat", "anion": "exp_ani", "solvent": "exp_sol"}
    sim_keys = {"cation": SIM_COLS["cation"], "anion": SIM_COLS["anion"], "solvent": SIM_COLS["solvent"]}
    new_exp_keys = {"cation": "D_cat_exp", "anion": "D_ani_exp", "solvent": "D_sol_exp"}
    new_sim_keys = {"cation": "D_cat_sim", "anion": "D_ani_sim", "solvent": "D_sol_sim"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={"wspace": 0.55})
    method_legend = "  |  ".join(
        f"{color} ▲: {name}"
        for name, _, _, color in METHOD_CONFIGS
    )
    fig.suptitle(
        "Diffusivity Parity — Micro Student (original) + All 3 Correction Methods\n"
        "Grey: non-outlier  |  Red: original outlier  |  "
        "Blue ▲: NVT UMA  |  Green ▲: NVT exp  |  Orange ▲: NPT UMA",
        fontsize=11, fontweight="bold",
    )

    for j, sp in enumerate(["cation", "anion", "solvent"]):
        _draw_combined_panel(
            axes[j], micro_merged, outlier_csv,
            exp_keys[sp], sim_keys[sp],
            new_exp_keys[sp], new_sim_keys[sp],
            sp.capitalize(),
            show_ylabel=(j == 0), show_xlabel=True,
            species=sp,
        )

    out = PLOT_DIR / "parity_outlier_all_methods_combined.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    exp_df    = load_exp()
    micro_sim = load_sim(MICRO_DIR)
    if micro_sim.empty:
        raise RuntimeError(f"No diffusivity CSVs found under {MICRO_DIR}")

    micro_merged = merge_exp(micro_sim, exp_df)
    n_match = micro_merged[["exp_cat", "exp_ani", "exp_sol"]].notna().any(axis=1).sum()
    print(f"Micro student: {len(micro_merged)} systems, {n_match} matched to experiment")

    # step 1 — collect outlier CSV
    outlier_csv = collect_outlier_csv(exp_df)
    print(outlier_csv.to_string())

    # step 2 — micro student parity plot
    plot_micro_student(micro_merged)

    # step 3 — 3 comparison plots
    plot_comparison(micro_merged, outlier_csv)

    # step 4 — combined plot (all 3 methods together)
    plot_combined(micro_merged, outlier_csv)

    print("Done.")


if __name__ == "__main__":
    main()
