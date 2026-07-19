"""Merge per-system diffusivity + conductivity CSVs from an eval.py run into a
group results directory, attach the *concentration-aware* experimental
reference, and verify that the experimental values eval.py already wrote
(diffusivity) were matched to the correct concentration.

Why this exists
---------------
eval.py's MSD analysis writes ``diffusivity_with_exp.csv`` per system, but it
only passes ``concentration_M`` into the experiment matcher if the config set it
(our multi-replica configs do not). Without a concentration, the matcher sorts
the candidate experimental rows by |Delta T| **only** and, because pandas'
default sort is not stable, the concentration it lands on among equal-temperature
rows is effectively arbitrary. Conductivity gets no experiment column at all.

This tool independently re-matches every system to the experimental tables by
(cation, anion, solvent, concentration, temperature) -- concentration first,
then nearest temperature, requiring the needed value to be present -- and:
  * merges all per-system CSVs into group-level CSVs,
  * adds the concentration-aware exp reference (``*_ref`` columns),
  * for diffusivity, compares eval's exp value to the concentration-aware value
    and flags any disagreement,
  * writes a human-readable verification report.

Concentration is parsed from the run-dir token (``nvt_1M_...`` -> 1.0,
``nvt_0_1M_...`` -> 0.1, ``nvt_0_5M_...`` -> 0.5); species/temperature come from
the CSV columns. Solvent symbols are mapped to the experiment-table spelling
(Diglyme->DEGDME, TGDME->TEGDME) exactly as eval.py does.

Usage
-----
  python merge_and_verify_exp.py --base <analysis_dir>
  # or point at explicit run dirs:
  python merge_and_verify_exp.py --msd-dir <..._msd> --cond-dir <..._conductivity>

Defaults: auto-discovers the newest ``*_msd`` and ``*_conductivity`` run dirs
under --base, writes everything to ``<base>/merged/``.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ── experiment reference tables (repo m5024 -> CFS symlink) ────────────────────
DEFAULT_EXP_DIFFUSIVITY_CSV = (
    "/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project"
    "/experiment_data/cleaned_version/diffusivity.csv"
)
DEFAULT_EXP_CONDUCTIVITY_CSV = (
    "/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project"
    "/experiment_data/cleaned_version/conductivity.csv"
)

# trajectory-side solvent symbol -> experiment-table solvent name
EXP_SOLVENT_NAME = {"Diglyme": "DEGDME", "TGDME": "TEGDME"}

# experiment-table column names
EXP_D_COLS = {
    "cation":  "D_cation (x 10^-10 m^2/s)",
    "anion":   "D_anion (x 10^-10 m^2/s)",
    "solvent": "D_solvent (x 10^-10 m^2/s)",
}
EXP_COND_COL = "IC2 (uS/cm)"

_CONC_RE = re.compile(r"_(\d+(?:_\d+)?)M_")


def _parse_conc(system_name: str):
    """`naotf_dme__nvt_0_1M_298K_20ns_100fs` -> 0.1 (M), or None."""
    m = _CONC_RE.search(system_name)
    if not m:
        return None
    return float(m.group(1).replace("_", "."))


def _exp_solvent(sym):
    return EXP_SOLVENT_NAME.get(sym, sym)


def _match_exp(exp_df, cat, anion, solvent, conc, T, value_col, conc_tol=0.01):
    """Concentration-*exact* nearest-T match.

    Filter to the (cation, anion, solvent) salt and to rows where ``value_col``
    is present. If a sim concentration is given, keep ONLY rows at that
    concentration (|Delta conc| <= conc_tol) -- we never borrow a value from a
    different concentration, because that is a different physical system (e.g.
    NaPF6/TEGDME has no measured cation D at 1 M, so it must stay NaN rather than
    fall back to the 0.1 M number). Among the surviving rows, pick the nearest
    temperature. Returns (row_or_None, info_dict).
    """
    sol_exp = _exp_solvent(solvent)
    sub = exp_df[
        (exp_df["cation"].astype(object) == cat)
        & (exp_df["anion"].astype(object) == anion)
        & (exp_df["solvent"].astype(object) == sol_exp)
        & exp_df[value_col].notna()
    ].copy()
    if sub.empty:
        return None, {"status": "no_salt_or_value", "exp_solvent": sol_exp}
    if conc is not None:
        sub = sub[(sub["concentration (M)"] - conc).abs() <= conc_tol]
        if sub.empty:
            return None, {"status": "no_exp_at_conc", "exp_solvent": sol_exp}
    sub["_dconc"] = (sub["concentration (M)"] - conc).abs() if conc is not None else 0.0
    sub["_dT"] = (sub["temperature (K)"] - T).abs() if T is not None else 0.0
    sub = sub.sort_values(["_dT"], kind="mergesort")  # stable, nearest T
    row = sub.iloc[0]
    return row, {
        "status": "ok",
        "exp_solvent": sol_exp,
        "matched_conc": float(row["concentration (M)"]),
        "matched_T": float(row["temperature (K)"]),
        "dconc": float(row["_dconc"]),
        "dT": float(row["_dT"]),
    }


# ── discovery ──────────────────────────────────────────────────────────────────
def _newest(base: Path, suffix: str):
    cands = sorted(p for p in base.glob(f"*_{suffix}") if p.is_dir())
    return cands[-1] if cands else None


def _collect(run_dir: Path, filename: str):
    if run_dir is None:
        return pd.DataFrame()
    parts = [pd.read_csv(p) for p in sorted(run_dir.glob(f"*/{filename}"))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ── diffusivity ─────────────────────────────────────────────────────────────────
def process_diffusivity(df, exp_df, report):
    if df.empty:
        report.append("DIFFUSIVITY: no per-system diffusivity_with_exp.csv found.\n")
        return df
    df = df.copy()
    df["concentration_M_parsed"] = df["system"].map(_parse_conc)

    ref = {sp: [] for sp in EXP_D_COLS}
    info_conc, info_T, info_dconc, info_dT = [], [], [], []
    for _, r in df.iterrows():
        row_info = None
        for sp, col in EXP_D_COLS.items():
            m, info = _match_exp(
                exp_df, r["cat_symbol"], r["anion_symbol"], r["solvent_symbol"],
                r["concentration_M_parsed"], r.get("temperature_K"), col,
            )
            ref[sp].append(np.nan if m is None else float(m[col]))
            if row_info is None and info["status"] == "ok":
                row_info = info
        info_conc.append(row_info["matched_conc"] if row_info else np.nan)
        info_T.append(row_info["matched_T"] if row_info else np.nan)
        info_dconc.append(row_info["dconc"] if row_info else np.nan)
        info_dT.append(row_info["dT"] if row_info else np.nan)

    df["exp_D_cation_ref"]  = ref["cation"]
    df["exp_D_anion_ref"]   = ref["anion"]
    df["exp_D_solvent_ref"] = ref["solvent"]
    df["exp_matched_conc_M"] = info_conc
    df["exp_matched_T_K"]    = info_T
    df["exp_dconc_M"]        = info_dconc
    df["exp_dT_K"]           = info_dT

    # verify vs eval's own exp columns (which were concentration-blind)
    def _mismatch(a, b):
        a, b = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
        both = a.notna() & b.notna()
        return both & (np.abs(a - b) > 1e-3 * np.abs(b).clip(lower=1e-9))

    mism = pd.Series(False, index=df.index)
    for sp, evalcol in [("cation", "exp_D_cation_1e-10_m2s"),
                        ("anion", "exp_D_anion_1e-10_m2s"),
                        ("solvent", "exp_D_solvent_1e-10_m2s")]:
        if evalcol in df.columns:
            mism = mism | _mismatch(df[evalcol], df[f"exp_D_{sp}_ref"])
    df["exp_mismatch_vs_eval"] = mism

    # ── report ──
    report.append("=" * 78)
    report.append("DIFFUSIVITY experiment-reference verification")
    report.append("=" * 78)
    per_sys = df.drop_duplicates("system")
    report.append(f"systems: {len(per_sys)} | replicas(rows): {len(df)}")
    report.append(f"exp matched at EXACT sim concentration (species missing -> NaN, "
                  f"never borrowed from another concentration).")
    miss = per_sys[per_sys[["exp_D_cation_ref", "exp_D_anion_ref",
                            "exp_D_solvent_ref"]].isna().any(axis=1)]
    report.append(f"systems with >=1 species lacking an exp measurement at that "
                  f"concentration/T: {len(miss)}")
    for _, r in miss.iterrows():
        missing = [sp for sp, c in [("D+", "exp_D_cation_ref"),
                                    ("D-", "exp_D_anion_ref"),
                                    ("D0", "exp_D_solvent_ref")] if pd.isna(r[c])]
        report.append(f"    {r['system']} (sim {r['concentration_M_parsed']} M): "
                      f"no exp for {', '.join(missing)}")
    nmis = df["exp_mismatch_vs_eval"].sum()
    report.append(f"\nrows where eval's exp_D differs from concentration-aware ref: {nmis}")
    if nmis:
        cols = ["system", "model", "concentration_M_parsed",
                "exp_D_cation_1e-10_m2s", "exp_D_cation_ref",
                "exp_D_anion_1e-10_m2s", "exp_D_anion_ref",
                "exp_D_solvent_1e-10_m2s", "exp_D_solvent_ref"]
        cols = [c for c in cols if c in df.columns]
        for _, r in df[df["exp_mismatch_vs_eval"]].drop_duplicates("system")[cols].iterrows():
            report.append("    MISMATCH " + " ".join(f"{c}={r[c]}" for c in cols[:3]))
            report.append(f"        eval  Dcat/ani/sol = "
                          f"{r.get('exp_D_cation_1e-10_m2s')}/{r.get('exp_D_anion_1e-10_m2s')}/{r.get('exp_D_solvent_1e-10_m2s')}")
            report.append(f"        ref   Dcat/ani/sol = "
                          f"{r['exp_D_cation_ref']}/{r['exp_D_anion_ref']}/{r['exp_D_solvent_ref']}")
    else:
        report.append("    -> eval's diffusivity exp values agree with the "
                      "concentration-aware reference for every system. OK.")
    report.append("")
    return df


# ── conductivity ────────────────────────────────────────────────────────────────
def process_conductivity(df, exp_df, report):
    if df.empty:
        report.append("CONDUCTIVITY: no per-system conductivity.csv found.\n")
        return df
    df = df.copy()
    df["concentration_M_parsed"] = df["system"].map(_parse_conc)

    ic2, mc, mT, mdc, mdT = [], [], [], [], []
    for _, r in df.iterrows():
        m, info = _match_exp(
            exp_df, r["cat_symbol"], r["anion_symbol"], r["solvent_symbol"],
            r["concentration_M_parsed"], r.get("T_K"), EXP_COND_COL,
        )
        ic2.append(np.nan if m is None else float(m[EXP_COND_COL]))
        mc.append(info.get("matched_conc", np.nan))
        mT.append(info.get("matched_T", np.nan))
        mdc.append(info.get("dconc", np.nan))
        mdT.append(info.get("dT", np.nan))
    df["exp_conductivity_uS_cm"] = ic2
    df["exp_conductivity_mS_cm"] = np.array(ic2, float) / 1000.0
    df["exp_matched_conc_M"] = mc
    df["exp_matched_T_K"]    = mT
    df["exp_dconc_M"]        = mdc
    df["exp_dT_K"]           = mdT

    report.append("=" * 78)
    report.append("CONDUCTIVITY experiment-reference matching")
    report.append("=" * 78)
    per_sys = df.drop_duplicates("system")
    n_no = per_sys["exp_conductivity_uS_cm"].isna().sum()
    report.append(f"systems: {len(per_sys)} | replicas(rows): {len(df)}")
    report.append(f"systems with NO exp conductivity match: {n_no}")
    for _, r in per_sys.iterrows():
        tag = "OK "
        if np.isnan(r["exp_conductivity_uS_cm"]):
            tag = "MISS"
        elif r["exp_dconc_M"] and r["exp_dconc_M"] > 1e-6:
            tag = "CONC?"
        elif r["exp_dT_K"] and r["exp_dT_K"] > 10.0:
            tag = "T?"
        report.append(
            f"    [{tag:5s}] {r['system']}: sim {r['concentration_M_parsed']} M / "
            f"{r['T_K']:.0f} K -> exp {r['exp_conductivity_uS_cm']} uS/cm "
            f"@ conc={r['exp_matched_conc_M']} T={r['exp_matched_T_K']} "
            f"(dconc={r['exp_dconc_M']}, dT={r['exp_dT_K']})")
    report.append("")
    return df


# ── aggregation (per system/conc/T mean +/- s.d. over replicas) ─────────────────
def aggregate(df, value_cols, exp_cols):
    if df.empty:
        return df
    df = df.copy()
    df["system_id"] = df["system"].str.split("__").str[0]
    keys = ["system_id", "concentration_M_parsed"]
    Tcol = "temperature_K" if "temperature_K" in df.columns else "T_K"
    df["_T"] = df[Tcol]
    keys.append("_T")
    agg = {}
    for c in value_cols:
        if c in df.columns:
            agg[f"{c}_mean"] = (c, "mean")
            agg[f"{c}_std"] = (c, "std")
    for c in exp_cols:
        if c in df.columns:
            agg[c] = (c, "first")
    agg["n_replicas"] = ("model", "nunique")
    out = df.groupby(keys, dropna=False).agg(**agg).reset_index()
    out = out.rename(columns={"concentration_M_parsed": "concentration_M", "_T": "temperature_K"})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, help="analysis dir holding *_msd / *_conductivity run dirs")
    ap.add_argument("--msd-dir", type=Path, default=None)
    ap.add_argument("--cond-dir", type=Path, default=None)
    ap.add_argument("--exp-diff", type=Path, default=Path(DEFAULT_EXP_DIFFUSIVITY_CSV))
    ap.add_argument("--exp-cond", type=Path, default=Path(DEFAULT_EXP_CONDUCTIVITY_CSV))
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    msd_dir = args.msd_dir or (_newest(args.base, "msd") if args.base else None)
    cond_dir = args.cond_dir or (_newest(args.base, "conductivity") if args.base else None)
    base = args.base or (msd_dir or cond_dir).parent
    out_dir = args.out_dir or (base / "merged")
    out_dir.mkdir(parents=True, exist_ok=True)

    report = [f"merge_and_verify_exp report",
              f"  msd_dir  = {msd_dir}",
              f"  cond_dir = {cond_dir}",
              f"  exp_diff = {args.exp_diff}",
              f"  exp_cond = {args.exp_cond}",
              f"  out_dir  = {out_dir}", ""]

    exp_diff = pd.read_csv(args.exp_diff)
    exp_cond = pd.read_csv(args.exp_cond)

    diff = _collect(msd_dir, "diffusivity_with_exp.csv")
    cond = _collect(cond_dir, "conductivity.csv")

    diff = process_diffusivity(diff, exp_diff, report)
    cond = process_conductivity(cond, exp_cond, report)

    if not diff.empty:
        diff.to_csv(out_dir / "diffusivity_merged.csv", index=False)
        agg = aggregate(diff,
                        ["D_cat_corrected_1e-10_m2s", "D_ani_corrected_1e-10_m2s",
                         "D_sol_corrected_1e-10_m2s", "D_cat_1e-10_m2s",
                         "D_ani_1e-10_m2s", "D_sol_1e-10_m2s"],
                        ["exp_D_cation_ref", "exp_D_anion_ref", "exp_D_solvent_ref"])
        agg.to_csv(out_dir / "diffusivity_by_system.csv", index=False)
    if not cond.empty:
        cond.to_csv(out_dir / "conductivity_merged.csv", index=False)
        agg = aggregate(cond,
                        ["sigma_onsager_mS_cm", "sigma_NE_mS_cm",
                         "conductivity by md craft (mS/cm)"],
                        ["exp_conductivity_mS_cm", "exp_conductivity_uS_cm"])
        agg.to_csv(out_dir / "conductivity_by_system.csv", index=False)

    (out_dir / "verification_report.txt").write_text("\n".join(report))
    print("\n".join(report))
    print(f"\n[written] {out_dir}/diffusivity_merged.csv, conductivity_merged.csv, "
          f"*_by_system.csv, verification_report.txt")


if __name__ == "__main__":
    main()
