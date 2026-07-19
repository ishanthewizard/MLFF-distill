"""Group transport-property parity figure (distilled PaiNN, NVT) vs experiment
at 298 K -- a 2x3 panel styled after ``plots/plot_figure3_results.py`` /
``figure2_results.png`` but driven by the *merged, concentration-verified* CSVs
produced by ``merge_and_verify_exp.py``.

  Row 1 : self-diffusivity parity for cation / anion / solvent
          (sim mean +/-1 s.d. over the replicas vs exp).
  Row 2 : ionic conductivity parity -- byteff2 Onsager (d) and the mdcraft
          collective-Onsager estimate (e) -- plus the shared legend (f).

Only 298 K systems are shown (matching the reference figure); the LiPF6/DME
0.5 M 273/323 K runs are therefore excluded here.

Diffusivity x-axis uses the concentration-EXACT experimental match. Conductivity
uses ``exp_conductivity_mS_cm``.

OPTIONAL OPLS reference overlay (``--opls <OPLS group_results dir>``):
  overlays the classical OPLS-AA (GROMACS) result as a second, reference series
  (grey-edged diamonds + a faint connector to the PaiNN point at the shared
  experimental x). When --opls is given the experimental x-axis is unified to the
  OPLS group_results reference (the 298.2 K standard values), so both models are
  compared against ONE experimental number per system. Both models carry a
  Yeh-Hummer correction (PaiNN in diffusivity_by_system.csv, OPLS in
  diffusivity_per_system.csv's ``D_*_corrected_sim_*`` columns), so ``--diff-mode``
  applies to BOTH: default ``corrected`` = YH-corrected vs YH-corrected; pass
  ``--diff-mode raw`` for raw-PBC vs raw-PBC.

    python plot_group_transport_parity.py --merged <.../merged> [--out <png>]
    python plot_group_transport_parity.py --merged <PAINN merged> \
        --opls <OPLS group_results> --diff-mode raw
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# encoding: solvent -> colour, salt -> marker, concentration -> fill (0.1 M hollow)
SOLV_COLOR = {"dme": "#E68310", "diglyme": "#11A579", "pc": "#E73F74", "tgdme": "#3969AC"}
SOLV_LABEL = {"dme": "DME", "diglyme": "DEGDME", "pc": "PC", "tgdme": "TEGDME"}
SALT_MARKER = {"napf6": "o", "naotf": "s", "lipf6": "^"}
SALT_LABEL = {"napf6": "NaPF$_6$", "naotf": "NaOTf", "lipf6": "LiPF$_6$"}
OPLS_MARKER = "D"          # OPLS reference always uses a diamond (shape = model)
OPLS_EDGE = "0.25"         # dark-grey edge so it reads as the reference series

D_UNIT = r"$10^{-10}\,\mathrm{m^2\,s^{-1}}$"
S_UNIT = r"$\mathrm{mS\,cm^{-1}}$"

LEGEND_SYSTEMS = [
    ("napf6", "dme", True),     ("naotf", "dme", True),
    ("napf6", "diglyme", True), ("naotf", "diglyme", True),
    ("napf6", "pc", True),      ("naotf", "pc", True),
    ("napf6", "tgdme", False),  ("naotf", "tgdme", True),
]


def _fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _norm_solv(s):
    return "tgdme" if s in ("tegdme", "tgdme") else s


def _salt_solv(system_id):
    parts = system_id.split("_")
    return parts[0], _norm_solv(parts[1])


def _opls_system_id(name):
    """`naotf_tegdme_1M_298K` -> `naotf_tgdme` (matches PaiNN system_id)."""
    p = name.split("_")
    return f"{p[0]}_{_norm_solv(p[1])}"


# ── data loading / standardisation ───────────────────────────────────────────────
def load_painn_diff(merged, mode):
    d = pd.read_csv(merged / "diffusivity_by_system.csv")
    pre = "D_{}_corrected_1e-10_m2s" if mode == "corrected" else "D_{}_1e-10_m2s"
    out = pd.DataFrame({
        "system_id": d["system_id"], "concentration_M": d["concentration_M"],
        "temperature_K": d["temperature_K"],
    })
    for k, s in [("cat", "cat"), ("ani", "ani"), ("sol", "sol")]:
        out[f"painn_{k}_mean"] = d[f"{pre.format(s)}_mean"]
        out[f"painn_{k}_std"] = d[f"{pre.format(s)}_std"]
    out["pexp_cat"] = d["exp_D_cation_ref"]
    out["pexp_ani"] = d["exp_D_anion_ref"]
    out["pexp_sol"] = d["exp_D_solvent_ref"]
    return out


def load_opls_diff(gr, mode="corrected"):
    d = pd.read_csv(gr / "diffusivity_per_system.csv")
    out = pd.DataFrame({
        "system_id": d["system"].map(_opls_system_id),
        "concentration_M": d["concentration_M"], "temperature_K": d["temperature_K"],
    })
    pre = "D_{}_corrected_sim" if mode == "corrected" else "D_{}_sim"
    for k in ("cat", "ani", "sol"):
        mcol, scol = f"{pre.format(k)}_mean", f"{pre.format(k)}_std"
        if mcol not in d.columns:  # fallback: older per_system without corrected cols
            mcol, scol = f"D_{k}_sim_mean", f"D_{k}_sim_std"
        out[f"opls_{k}_mean"] = d[mcol]
        out[f"opls_{k}_std"] = d[scol]
    out["oexp_cat"] = d["exp_D_cation_1e-10_m2s"]
    out["oexp_ani"] = d["exp_D_anion_1e-10_m2s"]
    out["oexp_sol"] = d["exp_D_solvent_1e-10_m2s"]
    return out


def load_painn_cond(merged):
    d = pd.read_csv(merged / "conductivity_by_system.csv")
    return pd.DataFrame({
        "system_id": d["system_id"], "concentration_M": d["concentration_M"],
        "temperature_K": d["temperature_K"],
        "painn_onsager_mean": d["sigma_onsager_mS_cm_mean"],
        "painn_onsager_std": d["sigma_onsager_mS_cm_std"],
        "painn_NE_mean": d["sigma_NE_mS_cm_mean"],
        "painn_NE_std": d["sigma_NE_mS_cm_std"],
        "painn_mdcraft_mean": d["conductivity by md craft (mS/cm)_mean"],
        "painn_mdcraft_std": d["conductivity by md craft (mS/cm)_std"],
        "pexp_sigma": d["exp_conductivity_mS_cm"],
    })


def load_opls_cond(gr):
    d = pd.read_csv(gr / "conductivity_per_system.csv")
    return pd.DataFrame({
        "system_id": d["system"].map(_opls_system_id),
        "concentration_M": d["concentration_M"], "temperature_K": d["T_K"],
        "opls_onsager_mean": d["sigma_onsager_mean_mS_cm"],
        "opls_onsager_std": d["sigma_onsager_std_mS_cm"],
        "opls_NE_mean": d["sigma_NE_mean_mS_cm"],
        "opls_NE_std": d["sigma_NE_std_mS_cm"],
        "oexp_sigma": d["exp_conductivity_mS_cm"],
    })


def _merge(painn, opls, exp_keys):
    """Outer-merge on (system_id, conc); unify each exp to the OPLS value where
    available (298.2 K standard), else fall back to the PaiNN value."""
    if opls is None:
        for e in exp_keys:
            painn[f"exp_{e}"] = painn[f"pexp_{e}"]
        return painn
    # round T so the join keys match exactly, and include T to avoid a Cartesian
    # product for multi-temperature systems (lipf6_dme 0.5 M at 273/298/323 K)
    for df in (painn, opls):
        df["temperature_K"] = df["temperature_K"].round(1)
    m = painn.merge(opls, on=["system_id", "concentration_M", "temperature_K"],
                    how="outer", suffixes=("", "_o"))
    for e in exp_keys:
        m[f"exp_{e}"] = m[f"oexp_{e}"].where(m[f"oexp_{e}"].notna(), m[f"pexp_{e}"])
    return m


# ── plotting ─────────────────────────────────────────────────────────────────────
def _bounds(m, cols, exp_col):
    vals = []
    for _, r in m.iterrows():
        xe = _fnum(r[exp_col])
        cand = [xe]
        for mc, sc in cols:
            ys, sd = _fnum(r.get(mc)), _fnum(r.get(sc))
            cand += [ys]
            if np.isfinite(ys) and np.isfinite(sd):
                cand += [ys - sd, ys + sd]
        vals += [v for v in cand if np.isfinite(v) and v > 0]
    return (min(vals), max(vals)) if vals else (0.1, 10.0)


def _global_bounds(diff, species, models):
    """One (lo, hi) spanning every species column and every model, padded like
    ``_bounds``' callers -- used for ``--common-axis`` so all panels share a scale."""
    lo, hi = np.inf, -np.inf
    for k, _ in species:
        cols = [(f"{mm}_{k}_mean", f"{mm}_{k}_std") for mm in models]
        vmin, vmax = _bounds(diff, cols, f"exp_{k}")
        lo, hi = min(lo, vmin), max(hi, vmax)
    return lo / 1.4, hi * 1.4


def _loglog_pearson(pairs):
    """Pearson r of log10(exp) vs log10(sim) over (exp, sim) pairs; None if < 3
    finite positive points or a degenerate (zero-variance) axis."""
    pts = [(x, y) for x, y in pairs if np.isfinite(x) and np.isfinite(y) and x > 0 and y > 0]
    if len(pts) < 3:
        return None
    a = np.array(pts)
    lx, ly = np.log10(a[:, 0]), np.log10(a[:, 1])
    if np.std(lx) == 0 or np.std(ly) == 0:
        return None
    return float(np.corrcoef(lx, ly)[0, 1]), len(pts)


def draw_panel(ax, m, p_mean, p_std, exp_col, title, xlabel, ylabel=None,
               o_mean=None, o_std=None, p_label="PaiNN", o_label="OPLS",
               show_xlabel=True, xylim=None, show_mapd=True, show_r=False,
               log=True):
    has_opls = o_mean is not None and o_mean in m.columns
    cols = [(p_mean, p_std)] + ([(o_mean, o_std)] if has_opls else [])
    if xylim is not None:
        lo, hi = xylim
    else:
        vmin, vmax = _bounds(m, cols, exp_col)
        lo, hi = (vmin / 1.4, vmax * 1.4) if log else (0.0, vmax * 1.08)
    edge = max(lo, 1e-6) if log else 0.0
    ax.fill_between([edge, hi], [edge / 2, hi / 2], [edge * 2, hi * 2],
                    color="0.8", alpha=0.30, lw=0, zorder=0)
    ax.plot([lo, hi], [lo, hi], color="0.45", ls="--", lw=1.3, zorder=1)

    dev_p, dev_o = [], []
    xy_p, xy_o = [], []          # (exp, sim) pairs for the log-log Pearson r
    for _, r in m.iterrows():
        salt, solv = _salt_solv(r.system_id)
        if solv not in SOLV_COLOR or salt not in SALT_MARKER:
            continue
        c = SOLV_COLOR[solv]
        xe = _fnum(r[exp_col])
        yp, sp = _fnum(r[p_mean]), _fnum(r[p_std])
        yo = _fnum(r[o_mean]) if has_opls else np.nan
        so = _fnum(r[o_std]) if has_opls else np.nan
        if not np.isfinite(xe):
            continue
        filled = float(r.concentration_M) != 0.1

        # faint connector between the two models at the shared experimental x
        if np.isfinite(yp) and np.isfinite(yo):
            ax.plot([xe, xe], [yp, yo], color="0.6", lw=0.9, alpha=0.5, zorder=2.2)

        # OPLS reference (diamond, drawn under PaiNN)
        if np.isfinite(yo):
            dev_o.append(abs(yo - xe) / xe)
            xy_o.append((xe, yo))
            if np.isfinite(so) and so > 0:
                ax.errorbar(xe, yo, yerr=so, fmt="none", ecolor="0.55",
                            elinewidth=1.0, capsize=2.8, alpha=0.7, zorder=2.4)
            ax.scatter(xe, yo, s=78, marker=OPLS_MARKER,
                       facecolor=c if filled else "white",
                       edgecolor=OPLS_EDGE, linewidths=1.5, alpha=0.9, zorder=2.7)

        # PaiNN primary (salt-shaped, colour = solvent)
        if np.isfinite(yp):
            dev_p.append(abs(yp - xe) / xe)
            xy_p.append((xe, yp))
            if np.isfinite(sp) and sp > 0:
                ax.errorbar(xe, yp, yerr=sp, fmt="none", ecolor="0.35", elinewidth=1.2,
                            capsize=3.5, capthick=1.2, alpha=0.85, zorder=2.9)
            ax.scatter(xe, yp, s=120, marker=SALT_MARKER[salt],
                       facecolor=c if filled else "white",
                       edgecolor=c, linewidths=1.9, alpha=0.95, zorder=3)

    lines = [f"{p_label} {100*np.mean(dev_p):.0f}%  (n={len(dev_p)})"] if dev_p else []
    if dev_o:
        lines.append(f"{o_label} {100*np.mean(dev_o):.0f}%  (n={len(dev_o)})")
    if lines and show_mapd:
        # lower-right corner: these transport points ride above the 1:1 line
        # (sim >= exp), so the top-left where the box used to sit overlaps data.
        ax.text(0.96, 0.05, "MAPD\n" + "\n".join(lines), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=10.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="0.55", linewidth=1.3, alpha=0.9), zorder=5)
    if show_r:
        # top-left corner: these points sit below the 1:1 line (sim < exp), so the
        # upper-left triangle is empty.
        rp = _loglog_pearson(xy_p)
        rlines = [f"{p_label} r = {rp[0]:.2f}  (n={rp[1]})"] if rp else []
        if has_opls:
            ro = _loglog_pearson(xy_o)
            if ro:
                rlines.append(f"{o_label} r = {ro[0]:.2f}  (n={ro[1]})")
        if rlines:
            ax.text(0.04, 0.96, "log–log Pearson\n" + "\n".join(rlines),
                    transform=ax.transAxes, ha="left", va="top", fontsize=10.5,
                    fontweight="bold", bbox=dict(boxstyle="round,pad=0.4",
                    facecolor="white", edgecolor="0.55", linewidth=1.3, alpha=0.9),
                    zorder=5)
    _scale = "log" if log else "linear"
    ax.set(xscale=_scale, yscale=_scale, xlim=(lo, hi), ylim=(lo, hi))
    ax.set_aspect("equal")
    if show_xlabel:
        ax.set_xlabel(f"Experiment  ({xlabel})")
    if ylabel:
        ax.set_ylabel(f"Simulation  ({ylabel})")
    ax.set_title(title, fontweight="bold", pad=8)
    ax.grid(True, which="major", ls=":", lw=0.7, alpha=0.55)
    ax.grid(True, which="minor", ls=":", lw=0.5, alpha=0.20)
    ax.tick_params(which="both", direction="in", top=True, right=True)


def legend_panel(ax, show_opls, col_a=0.06, col_b=0.19, tx=0.30):
    ax.axis("off")
    tA = ax.transAxes
    ms, lw = 9.0, 1.7

    def marker(x, y, mk, color, filled, edge=None):
        ax.scatter(x, y, marker=mk, s=ms ** 2,
                   facecolors=color if filled else "white",
                   edgecolors=edge or color, linewidths=lw, transform=tA,
                   clip_on=False, zorder=5)

    def label(y, text):
        ax.text(tx, y, text, va="center", ha="left", fontsize=10, transform=tA)

    ax.text(col_a, 0.985, "0.1 M", ha="center", va="center", fontsize=10,
            fontweight="bold", transform=tA)
    ax.text(col_b, 0.985, "1.0 M", ha="center", va="center", fontsize=10,
            fontweight="bold", transform=tA)

    y, dy, gap = 0.92, 0.055, 0.015
    prev_solv = None
    for salt, solv, has01 in LEGEND_SYSTEMS:
        if prev_solv is not None and solv != prev_solv:
            y -= gap
        prev_solv = solv
        c, mk = SOLV_COLOR[solv], SALT_MARKER[salt]
        if has01:
            marker(col_a, y, mk, c, filled=False)
        marker(col_b, y, mk, c, filled=True)
        label(y, f"{SALT_LABEL[salt]} in {SOLV_LABEL[solv]}")
        y -= dy

    y -= dy * 0.4 + gap
    ax.text(col_b, y, "0.5 M", ha="center", va="center", fontsize=10,
            fontweight="bold", transform=tA)
    y -= dy * 0.9
    marker(col_b, y, SALT_MARKER["lipf6"], SOLV_COLOR["dme"], filled=True)
    label(y, f"{SALT_LABEL['lipf6']} in {SOLV_LABEL['dme']}")
    y -= dy

    # model / annotation key
    y -= gap * 1.4
    if show_opls:
        marker(col_a + 0.02, y, "s", "0.35", filled=True)
        label(y, "PaiNN (salt shape, colour)")
        y -= dy
        marker(col_a + 0.02, y, OPLS_MARKER, "0.75", filled=True, edge=OPLS_EDGE)
        label(y, "OPLS ref (diamond)")
        y -= dy
    y -= gap
    ax.plot([col_a - 0.03, col_a + 0.07], [y, y], ls="--", color="0.45", lw=1.4,
            transform=tA, clip_on=False)
    label(y, "1:1  ($y=x$)")
    y -= dy
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((col_a - 0.03, y - 0.014), 0.10, 0.028, facecolor="0.8",
                           alpha=0.55, lw=0, transform=tA, clip_on=False))
    label(y, r"within $\times 2$")
    y -= dy
    ax.plot([col_a + 0.02, col_a + 0.02], [y - 0.016, y + 0.016], color="0.4",
            lw=1.4, transform=tA, clip_on=False)
    label(y, r"$\pm$1 s.d. (replicas)")


def plot_diff_2row(diff, T, args):
    """Diffusivity-only comparison: row 1 = PaiNN, row 2 = OPLS, columns = species.
    Both rows share the same experimental x (298.2 K reference)."""
    diff = diff[np.abs(diff["temperature_K"] - T) < 1.0].copy()
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 15, "axes.labelsize": 12.5,
        "axes.linewidth": 0.9, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig = plt.figure(figsize=(15.5, 10.6))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.05],
                          left=0.085, right=0.995, top=0.90, bottom=0.075,
                          wspace=0.28, hspace=0.16)
    species = [("cat", r"Cation self-diffusivity  $D^{+}$"),
               ("ani", r"Anion self-diffusivity  $D^{-}$"),
               ("sol", r"Solvent self-diffusivity  $D^{0}$")]
    rows = [("painn", "PaiNN"), ("opls", "OPLS")]
    shared = {}
    if getattr(args, "common_axis", False):
        gl = _global_bounds(diff, species, ["painn", "opls"])
        shared = {k: gl for k, _ in species}
    elif args.shared_range:
        for k, _ in species:
            vmin, vmax = _bounds(diff, [(f"painn_{k}_mean", f"painn_{k}_std"),
                                        (f"opls_{k}_mean", f"opls_{k}_std")], f"exp_{k}")
            shared[k] = (vmin / 1.4, vmax * 1.4)
    letters = iter("abcdef")
    for i, (pref, plab) in enumerate(rows):
        for j, (k, title) in enumerate(species):
            ax = fig.add_subplot(gs[i, j])
            draw_panel(ax, diff, f"{pref}_{k}_mean", f"{pref}_{k}_std", f"exp_{k}",
                       title if i == 0 else "", D_UNIT, ylabel=None,
                       p_label=plab, show_xlabel=(i == 1), xylim=shared.get(k),
                       show_mapd=not args.no_mapd, show_r=args.show_r)
            if j == 0:
                ax.set_ylabel(f"{plab} simulation\n({D_UNIT})", fontweight="bold")
            ax.annotate(next(letters), xy=(0, 1), xycoords="axes fraction",
                        xytext=(-40, 8), textcoords="offset points", fontsize=15,
                        fontweight="bold", va="bottom", ha="left")
    legend_panel(fig.add_subplot(gs[:, 3]), show_opls=False,
                 col_a=0.05, col_b=0.24, tx=0.36)

    dmode = "raw PBC" if args.diff_mode == "raw" else "Yeh-Hummer corrected"
    cnote = ", 0.1 M excluded" if args.drop_01m else ""
    fig.suptitle(f"Self-diffusivity parity: PaiNN (FP32) vs OPLS-AA vs experiment, "
                 f"{T:.0f} K   —   {dmode}, exp @ 298.2 K{cnote}",
                 fontsize=14, fontweight="bold", y=0.965)

    out = args.out or (args.merged / "figure_diffusivity_parity_2row.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    print(f"saved -> {out}  and  {out.with_suffix('.pdf')}")
    plt.close(fig)


def plot_diff_1row(diff, T, args, model="opls"):
    """Single-model self-diffusivity parity: one row, columns = cation / anion /
    solvent, plus the shared legend. ``model`` selects which row of the 2-row
    figure to draw on its own ("opls" or "painn"); the experimental x is the same
    298.2 K reference used there, so this panel matches the corresponding row of
    ``figure_diffusivity_parity_2row.png`` exactly."""
    plab = {"opls": "OPLS", "painn": "PaiNN"}[model]
    diff = diff[np.abs(diff["temperature_K"] - T) < 1.0].copy()
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 15, "axes.labelsize": 12.5,
        "axes.linewidth": 0.9, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig = plt.figure(figsize=(15.5, 5.7))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.05],
                          left=0.085, right=0.995, top=0.83, bottom=0.135,
                          wspace=0.28)
    species = [("cat", r"Cation self-diffusivity  $D^{+}$"),
               ("ani", r"Anion self-diffusivity  $D^{-}$"),
               ("sol", r"Solvent self-diffusivity  $D^{0}$")]
    # per-column axis limits: `--axis-ref` selects which model's data defines the
    # range so this single-model panel can be made to overlay a row of the 2-row
    # figure exactly (self=this model, painn/opls=that row, shared=both rows).
    axis_ref = getattr(args, "axis_ref", "self") or "self"
    if args.shared_range:
        axis_ref = "shared"
    ref_models = {"self": [model], "painn": ["painn"], "opls": ["opls"],
                  "shared": ["painn", "opls"]}[axis_ref]
    if getattr(args, "common_axis", False):
        # one range across all species AND both models -> matches the identically
        # flagged 2-row figure so every panel in both figures shares a scale
        gl = _global_bounds(diff, species, ["painn", "opls"])
        lims = {k: gl for k, _ in species}
    else:
        lims = {}
        for k, _ in species:
            cols = [(f"{rm}_{k}_mean", f"{rm}_{k}_std") for rm in ref_models]
            vmin, vmax = _bounds(diff, cols, f"exp_{k}")
            lims[k] = (vmin / 1.4, vmax * 1.4)
    for j, (k, title) in enumerate(species):
        ax = fig.add_subplot(gs[0, j])
        draw_panel(ax, diff, f"{model}_{k}_mean", f"{model}_{k}_std", f"exp_{k}",
                   title, D_UNIT, ylabel=None, p_label=plab, show_xlabel=True,
                   xylim=lims[k], show_mapd=not args.no_mapd,
                   show_r=args.show_r)
        if j == 0:
            ax.set_ylabel(f"{plab} simulation\n({D_UNIT})", fontweight="bold")
        ax.annotate("abc"[j], xy=(0, 1), xycoords="axes fraction",
                    xytext=(-40, 8), textcoords="offset points", fontsize=15,
                    fontweight="bold", va="bottom", ha="left")
    legend_panel(fig.add_subplot(gs[0, 3]), show_opls=False,
                 col_a=0.05, col_b=0.24, tx=0.36)

    dmode = "raw PBC" if args.diff_mode == "raw" else "Yeh-Hummer corrected"
    cnote = ", 0.1 M excluded" if args.drop_01m else ""
    fig.suptitle(f"Self-diffusivity parity: {plab}-AA vs experiment, "
                 f"{T:.0f} K   —   {dmode}, exp @ 298.2 K{cnote}",
                 fontsize=14, fontweight="bold", y=0.95)

    out = args.out or (args.merged / f"figure_diffusivity_parity_{model}_1row.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    print(f"saved -> {out}  and  {out.with_suffix('.pdf')}")
    plt.close(fig)


def plot_cond_2row(cond, T, args):
    """Ionic-conductivity comparison: row 1 = PaiNN, row 2 = OPLS, columns = the
    two estimators both models share (byteff2 Onsager, Nernst-Einstein). Shared
    experimental x per system."""
    cond = cond[np.abs(cond["temperature_K"] - T) < 1.0].copy()
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 15, "axes.labelsize": 12.5,
        "axes.linewidth": 0.9, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    metrics = [("onsager", r"Onsager  $\sigma$  (byteff2)")]
    if not args.drop_ne:
        metrics.append(("NE", r"Nernst$-$Einstein  $\sigma$"))
    nc = len(metrics)
    fig = plt.figure(figsize=(4.55 * nc + 4.2, 10.6 if nc == 2 else 9.6))
    gs = fig.add_gridspec(2, nc + 1, width_ratios=[1] * nc + [1.25 if nc == 1 else 0.95],
                          left=0.10 if nc == 1 else 0.095, right=0.995,
                          top=0.90, bottom=0.075, wspace=0.28, hspace=0.16)
    rows = [("painn", "PaiNN"), ("opls", "OPLS")]
    # Align every panel to ONE shared x/y range so the 2x2 grid reads on a common
    # axis left-to-right (and top-to-bottom). The range spans both estimators and
    # both models; equal aspect + xlim==ylim keeps the 1:1 line a true diagonal.
    all_cols = []
    for k, _ in metrics:
        all_cols += [(f"painn_{k}_mean", f"painn_{k}_std"),
                     (f"opls_{k}_mean", f"opls_{k}_std")]
    gvmin, gvmax = _bounds(cond, all_cols, "exp_sigma")
    shared_xylim = (gvmin / 1.4, gvmax * 1.4)
    letters = iter("abcd")
    for i, (pref, plab) in enumerate(rows):
        for j, (k, title) in enumerate(metrics):
            ax = fig.add_subplot(gs[i, j])
            draw_panel(ax, cond, f"{pref}_{k}_mean", f"{pref}_{k}_std", "exp_sigma",
                       title if i == 0 else "", S_UNIT, ylabel=None,
                       p_label=plab, show_xlabel=(i == 1), xylim=shared_xylim)
            if j == 0:
                ax.set_ylabel(f"{plab} simulation\n({S_UNIT})", fontweight="bold")
            ax.annotate(next(letters), xy=(0, 1), xycoords="axes fraction",
                        xytext=(-40, 8), textcoords="offset points", fontsize=15,
                        fontweight="bold", va="bottom", ha="left")
    lcols = dict(col_a=0.06, col_b=0.22, tx=0.34) if nc == 1 else \
            dict(col_a=0.05, col_b=0.26, tx=0.40)
    legend_panel(fig.add_subplot(gs[:, nc]), show_opls=False, **lcols)

    cnote = ", 0.1 M excluded" if args.drop_01m else ""
    if nc == 1:
        fig.suptitle(r"Onsager $\sigma$: PaiNN (FP32) vs OPLS-AA vs exp, "
                     f"{T:.0f} K  (exp @ 298.2 K{cnote})",
                     fontsize=12.5, fontweight="bold", y=0.965)
    else:
        fig.suptitle(f"Ionic-conductivity parity (Onsager + NE): PaiNN (FP32) vs "
                     f"OPLS-AA vs experiment, {T:.0f} K   —   exp @ 298.2 K{cnote}",
                     fontsize=14, fontweight="bold", y=0.965)
    out = args.out or (args.merged / "figure_conductivity_parity_2row.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    print(f"saved -> {out}  and  {out.with_suffix('.pdf')}")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged", type=Path, required=True,
                    help="the merged/ dir written by merge_and_verify_exp.py (PaiNN)")
    ap.add_argument("--opls", type=Path, default=None,
                    help="OPLS group_results dir to overlay as a reference series")
    ap.add_argument("--diff-mode", choices=["raw", "corrected"], default="corrected",
                    help="which sim self-diffusivity to plot (default corrected; use "
                         "raw for a fair PaiNN-vs-OPLS comparison since OPLS has no YH)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--temperature", type=float, default=298.0)
    ap.add_argument("--layout",
                    choices=["overlay", "diff-2row", "diff-opls", "diff-painn",
                             "cond-2row"],
                    default="overlay",
                    help="overlay: 2x3 transport (PaiNN + OPLS overlaid). "
                         "diff-2row: diffusivity only, row1=PaiNN, row2=OPLS. "
                         "diff-opls / diff-painn: single-model diffusivity row "
                         "(cat/ani/sol + legend), matching that row of diff-2row. "
                         "cond-2row: conductivity only (Onsager, NE), row1=PaiNN, row2=OPLS.")
    ap.add_argument("--drop-01m", action="store_true",
                    help="exclude 0.1 M systems (their small exp values dominate MAPD)")
    ap.add_argument("--drop-ne", action="store_true",
                    help="cond-2row: drop the Nernst-Einstein column, keep Onsager only")
    ap.add_argument("--shared-range", action="store_true",
                    help="2-row layouts: give the PaiNN and OPLS rows the same x/y "
                         "limits per column (direct model comparison)")
    ap.add_argument("--axis-ref", choices=["self", "painn", "opls", "shared"],
                    default="self",
                    help="diff-opls/diff-painn: which model's data sets each column's "
                         "x/y limits (self=this model, painn/opls=overlay that row of "
                         "the 2-row figure, shared=union of both rows)")
    ap.add_argument("--common-axis", action="store_true",
                    help="diff-2row & diff-opls/diff-painn: force EVERY panel onto one "
                         "shared x/y range (union over all species and both models), so "
                         "all panels across both figures line up. Overrides --axis-ref/"
                         "--shared-range.")
    ap.add_argument("--no-mapd", action="store_true",
                    help="omit the MAPD (mean abs % deviation) box from each panel")
    ap.add_argument("--show-r", action="store_true",
                    help="add a log-log Pearson r box (r of log10 exp vs log10 sim) per panel")
    ap.add_argument("--drop-mdcraft", action="store_true",
                    help="overlay layout: drop the mdcraft panel and show only the "
                         "Onsager conductivity panel (0.1 M excluded there; title "
                         "without 'byteff2'). Diffusivity row keeps all systems.")
    args = ap.parse_args(argv)

    T = args.temperature
    opls_diff = load_opls_diff(args.opls, args.diff_mode) if args.opls else None
    opls_cond = load_opls_cond(args.opls) if args.opls else None
    def _drop01(df):
        return df[np.abs(df["concentration_M"] - 0.1) > 0.01].copy() if args.drop_01m else df

    diff = _drop01(_merge(load_painn_diff(args.merged, args.diff_mode), opls_diff,
                          ["cat", "ani", "sol"]))

    if args.layout == "diff-2row":
        plot_diff_2row(diff, T, args)
        return

    if args.layout in ("diff-opls", "diff-painn"):
        plot_diff_1row(diff, T, args, model=args.layout.split("-")[1])
        return

    cond = _drop01(_merge(load_painn_cond(args.merged), opls_cond, ["sigma"]))
    if args.layout == "cond-2row":
        plot_cond_2row(cond, T, args)
        return

    diff = diff[np.abs(diff["temperature_K"] - T) < 1.0].copy()
    cond = cond[np.abs(cond["temperature_K"] - T) < 1.0].copy()

    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 15, "axes.labelsize": 12.5,
        "axes.linewidth": 0.9, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 11.2))

    dmode = "raw PBC" if args.diff_mode == "raw" else "Yeh-Hummer corr."
    DIFF = [("cat", r"Cation self-diffusivity  $D^{+}$"),
            ("ani", r"Anion self-diffusivity  $D^{-}$"),
            ("sol", r"Solvent self-diffusivity  $D^{0}$")]
    for ax, (k, title) in zip(axes[0], DIFF):
        draw_panel(ax, diff, f"painn_{k}_mean", f"painn_{k}_std", f"exp_{k}",
                   title, D_UNIT, ylabel=(D_UNIT if k == "cat" else None),
                   o_mean=f"opls_{k}_mean" if args.opls else None,
                   o_std=f"opls_{k}_std" if args.opls else None)

    if args.drop_mdcraft:
        # single Onsager conductivity panel, 0.1 M excluded, no "byteff2" in title
        cond_ons = cond[np.abs(cond["concentration_M"] - 0.1) > 0.01].copy()
        draw_panel(axes[1, 0], cond_ons, "painn_onsager_mean", "painn_onsager_std",
                   "exp_sigma", r"Ionic conductivity  $\sigma$  (Onsager)",
                   S_UNIT, ylabel=S_UNIT,
                   o_mean="opls_onsager_mean" if args.opls else None,
                   o_std="opls_onsager_std" if args.opls else None)
        legend_panel(axes[1, 1], show_opls=bool(args.opls))
        axes[1, 2].axis("off")
        lettered = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]]
    else:
        draw_panel(axes[1, 0], cond, "painn_onsager_mean", "painn_onsager_std",
                   "exp_sigma", r"Ionic conductivity  $\sigma$  (byteff2 Onsager)",
                   S_UNIT, ylabel=S_UNIT,
                   o_mean="opls_onsager_mean" if args.opls else None,
                   o_std="opls_onsager_std" if args.opls else None)
        draw_panel(axes[1, 1], cond, "painn_mdcraft_mean", "painn_mdcraft_std",
                   "exp_sigma", r"Ionic conductivity  $\sigma$  (mdcraft collective)", S_UNIT)
        legend_panel(axes[1, 2], show_opls=bool(args.opls))
        lettered = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]

    for ax, lab in zip(lettered, "abcde"):
        ax.annotate(lab, xy=(0, 1), xycoords="axes fraction", xytext=(-36, 10),
                    textcoords="offset points", fontsize=16, fontweight="bold",
                    va="bottom", ha="left")

    models = "PaiNN (FP32) vs OPLS-AA" if args.opls else "distilled PaiNN (FP32, NVT)"
    fig.suptitle(f"Transport parity: {models} vs experiment, {T:.0f} K"
                 f"  —  diffusivity {dmode}, exp @ 298.2 K",
                 fontsize=13.5, fontweight="bold", y=0.975)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.91, bottom=0.07,
                        wspace=0.26, hspace=0.30)

    out = args.out or (args.merged / "figure_transport_parity_nvt.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    print(f"saved -> {out}  and  {out.with_suffix('.pdf')}")
    plt.close(fig)


if __name__ == "__main__":
    main()
