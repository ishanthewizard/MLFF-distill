#!/usr/bin/env python3
"""Observable evaluation dispatcher.

Runs RDF, density, energy, MSD, force_mae and/or RMSD analysis for a list
of systems, with parallelism at the system level.

Usage
-----
    python eval.py --config myconfig.py [--analyses rdf,density,energy]
                   [--output-dir ./results] [--workers 4]

Config file format
------------------
The config file is a plain Python module that defines SYSTEMS (required)
and optionally OUTPUT_DIR, ANALYSES, WORKERS.

    SYSTEMS = [
        {
            "name": "NaPF6/DME 0.1M",
            # ASE trajectories: value is a path string ending in .traj
            "traj_paths": {
                "UMA": "/path/to/uma.traj",
                "student": "/path/to/student.traj",
            },
            # GROMACS trajectories: value is either
            #   (a) path string ending in .xtc/.trr  →  TPR resolved via tpr_paths or same-stem sibling
            #   (b) dict {"xtc": "...", "tpr": "..."}
            # "traj_paths": {
            #     "OPLS": {"xtc": "/path/to.xtc", "tpr": "/path/to.tpr"},
            # },
            # "tpr_paths": {"OPLS": "/path/to.tpr"},  # alternative to dict form
            # ── shared ──
            "dt_fs": {"UMA": 10.0, "student": 100.0},  # fs; or single float
            "model_colors": {"UMA": "#1f77b4", "student": "#ff7f0e"},
            # ── RDF ──
            "rdf_pairs": [("Na", "O"), ("Na", "F")],
            "skip_ns": 0.1,          # equilibration skip for RDF/density windows
            "window_ns": 0.9,        # analysis window width
            "n_frames": 1000,        # frames sampled per window
            "sliding_window": False, # enable sliding-window RDF
            # ── density ──
            "density_roll_window_ns": 0.5,
            # ── MSD / diffusivity ──
            "cat_symbol":      "Na",   # key in cation_dict
            "anion_symbol":    "PF6",  # key in anion_dict
            "solvent_symbol":  "DME",  # key in solvent_dict
            "eq_cut_ns":       0.0,    # frames to skip from traj start
            "fit_pct":         0.8,    # fit up to this fraction of max lag
            "tau_min_fit_ns":  1.0,    # lower bound of linear fit
            "slide_window_ns": 10.0,   # sliding-window width (Panel 4)
            "slide_step_ns":   0.1,    # sliding-window step
            "n_conv_points":   200,    # resolution of convergence sweep
            # ── force_mae ──
            # force_mae computes per-frame MAE between teacher and student
            # on the student trajectory.  Only ASE .traj trajectories supported.
            # teacher_ckpt is required; student_ckpt is optional — when omitted
            # the forces already stored in the trajectory are used as the student
            # reference (avoids re-inference).
            "force_mae_teacher_ckpt":  "/path/to/teacher.pt",
            "force_mae_student_ckpt":  None,   # or "/path/to/student.pt"
            "force_mae_n_frames":      500,    # fallback frame count (ignored when analyze_dt_ps set)
            "force_mae_analyze_dt_ps": 100.0,  # evaluate every N ps of sim time (preferred)
            # ── energy_mae ──
            # energy_mae computes per-frame total and per-atom energy MAE between
            # teacher and student on the student trajectory.  Only ASE .traj
            # trajectories are supported.  teacher_ckpt is required;
            # student_ckpt is optional — when omitted the energy already stored
            # in the trajectory is used as the student reference.
            "energy_mae_teacher_ckpt":  "/path/to/teacher.pt",
            "energy_mae_student_ckpt":  None,    # or "/path/to/student.pt"
            "energy_mae_n_frames":      500,     # fallback frame count
            "energy_mae_analyze_dt_ps": 100.0,   # evaluate every N ps of sim time (preferred)
            # ── rmsd ──
            # RMSD of each frame relative to the reference frame using
            # Kabsch optimal superposition.
            "rmsd_ref_frame_idx":   0,      # reference frame index
            "rmsd_n_frames":        1000,   # fallback frame count (ignored when analyze_dt_ps set)
            "rmsd_analyze_dt_ps":   10.0,   # evaluate every N ps of sim time (preferred)
        },
    ]
    OUTPUT_DIR = "./results"
    ANALYSES   = ["rdf", "density", "energy", "msd", "force_mae", "energy_mae", "stress_mae", "rmsd"]  # subset as needed
    WORKERS    = 4   # or None → one per system

Model ordering in plots follows the key order of traj_paths (Python 3.7+).
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import importlib.util
import os
import sys
import traceback
import multiprocessing as _mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# ── import toolboxes ──────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from RDF.utils.compute import compute_rdf, compute_rdf_sliding
from RDF.utils.plot import plot_rdf_comparison, plot_rdf_sliding
from density.compute import compute_density
from density.plot import plot_density_bars
from energy.compute import extract_energies
from energy.plot import plot_energy_timeseries
from force_mae.compute import compute_force_mae
from force_mae.plot import plot_force_mae_timeseries, plot_force_mae_comparison
from energy_mae.compute import compute_energy_mae
from energy_mae.plot import plot_energy_mae_timeseries, plot_energy_mae_comparison
from stress_mae.compute import compute_stress_mae
from stress_mae.plot import plot_stress_mae_timeseries, plot_stress_mae_comparison
from rmsd.compute import compute_rmsd
from rmsd.plot import plot_rmsd_timeseries
from cell_size.compute import extract_cell_timeseries
from cell_size.plot import plot_cell_timeseries

import pandas as pd


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe(s: str) -> str:
    """Filesystem-safe version of a name: replace / and spaces with _."""
    return s.replace("/", "_").replace(" ", "_")


def _sys_out(output_dir: Path, sys_name: str) -> Path:
    """Return and create the per-system output subdirectory."""
    p = output_dir / _safe(sys_name)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _dt_fs(sys_cfg: dict, model: str) -> float:
    dt = sys_cfg.get("dt_fs", 100.0)
    if isinstance(dt, dict):
        return dt.get(model, 100.0)
    return float(dt)


def _traj_fmt(traj_val) -> str:
    """Return 'ase' or 'gromacs' based on a traj_paths entry value.

    traj_val may be:
      - str / Path  →  format inferred from extension (.xtc/.trr → gromacs, else ase)
      - dict {"xtc": ..., "tpr": ...}  →  gromacs
    """
    if isinstance(traj_val, dict):
        return "gromacs"
    if Path(str(traj_val)).suffix.lower() in (".xtc", ".trr"):
        return "gromacs"
    return "ase"


def _main_traj_path(traj_val) -> Path:
    """Return the primary trajectory path (XTC for GROMACS, .traj for ASE)."""
    if isinstance(traj_val, dict):
        return Path(traj_val["xtc"])
    return Path(str(traj_val))


def _xtc_tpr(traj_val, sys_cfg: dict, model: str):
    """Resolve (xtc_path, tpr_path) for a GROMACS traj_paths entry.

    Accepts traj_val as:
      - str .xtc/.trr  →  TPR from sys_cfg['tpr_paths'][model] or same-stem sibling
      - dict {"xtc": ..., "tpr": ...}
    """
    if isinstance(traj_val, dict):
        return Path(traj_val["xtc"]), Path(traj_val["tpr"])
    xtc = Path(str(traj_val))
    # print("here")
    tpr_map = sys_cfg.get("tpr_paths", {})
    # print("here2")
    tpr = Path(tpr_map[model]) if model in tpr_map else xtc.with_suffix(".tpr")
    # print("here3")
    return xtc, tpr


def _model_order(sys_cfg: dict) -> list[str]:
    return list(sys_cfg["traj_paths"].keys())


def _model_colors(sys_cfg: dict) -> dict[str, str]:
    defaults = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ]
    colors = sys_cfg.get("model_colors", {})
    for i, m in enumerate(_model_order(sys_cfg)):
        colors.setdefault(m, defaults[i % len(defaults)])
    return colors


# ── per-system worker (runs in subprocess) ────────────────────────────────────

def _run_system(sys_cfg: dict, analyses: list[str], output_dir: Path) -> str:
    """Process one system: run requested analyses and save outputs.

    This function is called in a separate process; it must only use
    picklable objects and import what it needs.
    """
    sys_path_str = str(Path(__file__).resolve().parent)
    if sys_path_str not in sys.path:
        sys.path.insert(0, sys_path_str)

    from RDF.utils.compute import compute_rdf, compute_rdf_sliding
    from RDF.utils.plot import plot_rdf_comparison, plot_rdf_sliding
    from density.compute import compute_density, extract_density_timeseries
    from density.plot import plot_density_bars, plot_density_timeseries
    from energy.compute import extract_energies
    from energy.plot import plot_energy_timeseries
    from mean_square_displacement.compute import run_msd_analysis, save_msd_pickle, save_diffusivity_csv
    from mean_square_displacement.plot import plot_msd, plot_convergence
    from force_mae.compute import compute_force_mae
    from force_mae.plot import plot_force_mae_timeseries, plot_force_mae_comparison
    from energy_mae.compute import compute_energy_mae
    from energy_mae.plot import plot_energy_mae_timeseries, plot_energy_mae_comparison
    from stress_mae.compute import compute_stress_mae
    from stress_mae.plot import plot_stress_mae_timeseries, plot_stress_mae_comparison
    from rmsd.compute import compute_rmsd
    from rmsd.plot import plot_rmsd_timeseries
    from cell_size.compute import extract_cell_timeseries
    from cell_size.plot import plot_cell_timeseries
    from gromacs_io import ensure_element_gro, n_frames_gromacs
    import pandas as pd

    from ase.io.trajectory import Trajectory as _AseTraj
    name = sys_cfg["name"]
    traj_paths = sys_cfg["traj_paths"]
    rdf_pairs = sys_cfg.get("rdf_pairs", [])
    skip_ns = sys_cfg.get("skip_ns", 0.1)
    window_ns = sys_cfg.get("window_ns", 0.9)
    n_frames = sys_cfg.get("n_frames", 1000)
    sliding = sys_cfg.get("sliding_window", False)
    max_traj_ns = sys_cfg.get("max_traj_ns", 20.0)
    model_order = _model_order(sys_cfg)
    colors = _model_colors(sys_cfg)

    sys_out = _sys_out(output_dir, name)
    # compute effective analysis length per model + resolve topologies for GROMACS
    analysis_ns: dict[str, float] = {}
    topology_for: dict[str, Path | None] = {}   # model → element-GRO (None for ASE)
    preloaded_for: dict[str, list | None] = {}  # model → pre-read frames (GROMACS only)
    # stable cache shared across all runs — GRO is reused instead of regenerated each time
    _topo_cache = Path("/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/results/opls_baseline/results/topologies_element_gro")

    for model, traj_val in traj_paths.items():
        fmt = _traj_fmt(traj_val)
        topology_for[model] = None
        preloaded_for[model] = None
        try:
            dt = _dt_fs(sys_cfg, model)
            if fmt == "gromacs":
                xtc, tpr = _xtc_tpr(traj_val, sys_cfg, model)
                # print(xtc, tpr)
                if not xtc.exists():
                    continue
                # print("here5")
                elem_gro = _topo_cache / f"{xtc.stem}.element.gro"
                if not elem_gro.exists():
                    elem_gro = ensure_element_gro(tpr, xtc, _topo_cache)
                topology_for[model] = elem_gro
                from gromacs_io import preload_gromacs_frames, n_frames_gromacs
                n = n_frames_gromacs(elem_gro, xtc)
                frames = preload_gromacs_frames(
                    elem_gro, xtc, dt,
                    max_ns=max_traj_ns,
                    n_sample=n_frames * 4,
                )
                preloaded_for[model] = frames
            else:
                p = _main_traj_path(traj_val)
                if not p.exists():
                    continue
                with _AseTraj(str(p)) as _t:
                    n = len(_t)
            analysis_ns[model] = min(max_traj_ns, n * dt * 1e-6)
        except Exception:
            analysis_ns[model] = max_traj_ns

    log_lines = [f"=== {name} (max_traj_ns={max_traj_ns}) ==="]
    for m, ans in analysis_ns.items():
        log_lines.append(f"  [setup] {m}: analysis_ns={ans:.2f}")
    # ── RDF ───────────────────────────────────────────────────────────────────
    if "rdf" in analyses and rdf_pairs:
        rdf_results = {}
        for pair in rdf_pairs:
            pair_label = f"{pair[0]}-{pair[1]}"
            rdf_results[pair_label] = {}
            for model, traj_path in traj_paths.items():
                p = _main_traj_path(traj_path)
                if not p.exists():
                    log_lines.append(f"  [RDF] SKIP {model} {pair_label}: missing {p}")
                    continue
                try:
                    dt = _dt_fs(sys_cfg, model)
                    eff_window = min(window_ns, analysis_ns.get(model, max_traj_ns) - skip_ns)
                    df = compute_rdf(p, pair[0], pair[1], dt, n_frames,
                                     skip_ns=skip_ns, window_ns=eff_window,
                                     topology=topology_for.get(model),
                                     preloaded_frames=preloaded_for.get(model))
                    rdf_results[pair_label][model] = df
                    log_lines.append(f"  [RDF] {model} {pair_label}: done")
                except Exception:
                    log_lines.append(f"  [RDF] ERROR {model} {pair_label}:\n{traceback.format_exc()}")

        if any(rdf_results[pl] for pl in rdf_results):
            out = plot_rdf_comparison(rdf_results, name, model_order, colors, sys_out)
            log_lines.append(f"  [RDF] saved comparison: {out}")

        if sliding:
            for pair in rdf_pairs:
                pair_label = f"{pair[0]}-{pair[1]}"
                for model, traj_path in traj_paths.items():
                    p = _main_traj_path(traj_path)
                    if not p.exists():
                        continue
                    try:
                        dt = _dt_fs(sys_cfg, model)
                        ans = analysis_ns.get(model, max_traj_ns)
                        eff_window = min(window_ns, ans - skip_ns)
                        slide_ns = sys_cfg.get("rdf_slide_ns", 1.0)
                        res = compute_rdf_sliding(
                            p, pair[0], pair[1], dt,
                            total_ns=ans,
                            skip_ns=skip_ns,
                            window_ns=eff_window,
                            slide_ns=slide_ns,
                            topology=topology_for.get(model),
                        )
                        if res:
                            og, on = plot_rdf_sliding(res, name, model, pair_label, sys_out)
                            log_lines.append(f"  [RDF sliding] {model} {pair_label}: {og.name}, {on.name}")
                            # save npz
                            import numpy as np
                            npz = sys_out / f"rdf_sliding_{model}_{pair_label}.npz".replace(" ", "_")
                            np.savez_compressed(npz, **{k: v for k, v in res.items()
                                                        if isinstance(v, np.ndarray)})
                    except Exception:
                        log_lines.append(f"  [RDF sliding] ERROR {model} {pair_label}:\n{traceback.format_exc()}")

    # ── Density ───────────────────────────────────────────────────────────────
    if "density" in analyses:
        density_rows = []
        ts_data = {}   # {model: (times_ns, densities)} for timeseries plot
        for model, traj_path in traj_paths.items():
            p = _main_traj_path(traj_path)
            if not p.exists():
                log_lines.append(f"  [Density] SKIP {model}: missing {p}")
                continue
            try:
                dt = _dt_fs(sys_cfg, model)
                ans = analysis_ns.get(model, max_traj_ns)
                eff_window = min(window_ns, ans - skip_ns)
                mean, std = compute_density(p, dt, n_frames,
                                            skip_ns=skip_ns, window_ns=eff_window,
                                            topology=topology_for.get(model),
                                            preloaded_frames=preloaded_for.get(model))
                density_rows.append({"System": name, "Model": model,
                                     "Density (g/cm³)": mean, "Std (g/cm³)": std})
                log_lines.append(f"  [Density] {model}: {mean:.4f} ± {std:.4f} g/cm³")
                times, dens = extract_density_timeseries(p, dt, n_sample=n_frames * 2,
                                                         max_ns=ans,
                                                         topology=topology_for.get(model))
                ts_data[model] = (times, dens)
            except Exception:
                log_lines.append(f"  [Density] ERROR {model}:\n{traceback.format_exc()}")

        if density_rows:
            df = pd.DataFrame(density_rows)
            csv_path = sys_out / "density.csv"
            df.to_csv(csv_path, index=False)
            log_lines.append(f"  [Density] saved: {csv_path}")

        if ts_data:
            out = plot_density_timeseries(ts_data, name, sys_out,
                                          model_colors=colors,
                                          roll_window_ns=sys_cfg.get("density_roll_window_ns", 0.5),
                                          eq_cutoff_ns=skip_ns)
            log_lines.append(f"  [Density] timeseries saved: {out}")

    # ── Energy ────────────────────────────────────────────────────────────────
    if "energy" in analyses:
        for model, traj_path in traj_paths.items():
            p = _main_traj_path(traj_path)
            if not p.exists():
                log_lines.append(f"  [Energy] SKIP {model}: missing {p}")
                continue
            try:
                if _traj_fmt(traj_path) == "gromacs":
                    log_lines.append(f"  [Energy] SKIP {model}: energy not available in XTC (use EDR)")
                    continue
                dt = _dt_fs(sys_cfg, model)
                times, pe, ke, etot, temp = extract_energies(p, n_frames, dt,
                                                              max_ns=analysis_ns.get(model, max_traj_ns))
                if len(times) == 0:
                    log_lines.append(f"  [Energy] {model}: no frames with energy data")
                    continue
                out = plot_energy_timeseries(times, pe, ke, etot, temp,
                                             name, model, sys_out,
                                             color=colors.get(model, "#1f77b4"))
                log_lines.append(f"  [Energy] {model}: saved {out.name}")
                import numpy as np
                npz = sys_out / f"energy_{model}.npz".replace(" ", "_")
                np.savez_compressed(npz, times=times, pe=pe, ke=ke, etot=etot, temp=temp)
            except Exception:
                log_lines.append(f"  [Energy] ERROR {model}:\n{traceback.format_exc()}")

    # ── MSD / Diffusivity ─────────────────────────────────────────────────────
    if "msd" in analyses:
        cat_sym  = sys_cfg.get("cat_symbol")
        ani_sym  = sys_cfg.get("anion_symbol")
        sol_sym  = sys_cfg.get("solvent_symbol")
        if not all([cat_sym, ani_sym, sol_sym]):
            log_lines.append("  [MSD] SKIP: cat_symbol / anion_symbol / solvent_symbol not set in config")
        else:
            eq_cut_ns       = sys_cfg.get("eq_cut_ns",       0.0)
            fit_pct         = sys_cfg.get("fit_pct",         0.8)
            tau_min_fit_ns  = sys_cfg.get("tau_min_fit_ns",  1.0)
            n_conv_points   = sys_cfg.get("n_conv_points",   200)
            slide_window_ns = sys_cfg.get("slide_window_ns", 10.0)
            slide_step_ns   = sys_cfg.get("slide_step_ns",   0.1)

            d_rows = []
            for model, traj_path in traj_paths.items():
                p = _main_traj_path(traj_path)
                if not p.exists():
                    log_lines.append(f"  [MSD] SKIP {model}: missing {p}")
                    continue
                try:
                    dt = _dt_fs(sys_cfg, model)
                    result = run_msd_analysis(
                        p, cat_sym, ani_sym, sol_sym, dt,
                        topology_path=topology_for.get(model),
                        eq_cut_ns=eq_cut_ns,
                        fit_pct=fit_pct,
                        tau_min_fit_ns=tau_min_fit_ns,
                        n_sample=n_frames,
                        n_conv_points=n_conv_points,
                        slide_window_ns=slide_window_ns,
                        slide_step_ns=slide_step_ns,
                        max_traj_ns=analysis_ns.get(model, max_traj_ns),
                    )
                    slug = f"{_safe(name)}_{_safe(model)}"

                    # pickle (raw MSD arrays)
                    pkl = save_msd_pickle(result, slug, sys_out)
                    log_lines.append(f"  [MSD] {model}: saved pkl {pkl.name}")

                    # plots
                    p_msd  = plot_msd(result, name, model, sys_out)
                    p_conv = plot_convergence(result, name, model, sys_out)
                    log_lines.append(f"  [MSD] {model}: {p_msd.name}, {p_conv.name}")

                    # collect D row for CSV
                    conv = result["convergence"]
                    d_rows.append({
                        "system":          name,
                        "model":           model,
                        "cat_symbol":      cat_sym,
                        "anion_symbol":    ani_sym,
                        "solvent_symbol":  sol_sym,
                        "eq_cut_ns":       eq_cut_ns,
                        "fit_pct":         fit_pct,
                        "tau_min_fit_ns":  conv["tau_min_fit_ns"],
                        "tau_max_fit_ns":  conv["tau_max_fit_ns"],
                        "D_cat_1e-10_m2s": conv["D_cat_final"],
                        "D_ani_1e-10_m2s": conv["D_ani_final"],
                        "D_sol_1e-10_m2s": conv["D_sol_final"],
                    })
                except Exception:
                    log_lines.append(f"  [MSD] ERROR {model}:\n{traceback.format_exc()}")

            if d_rows:
                csv = save_diffusivity_csv(d_rows, sys_out)
                log_lines.append(f"  [MSD] saved diffusivity CSV: {csv.name}")

    # ── Force MAE ─────────────────────────────────────────────────────────────
    if "force_mae" in analyses:
        teacher_ckpt = sys_cfg.get("force_mae_teacher_ckpt")
        if not teacher_ckpt:
            log_lines.append("  [ForceMae] SKIP: force_mae_teacher_ckpt not set in config")
        else:
            student_ckpt    = sys_cfg.get("force_mae_student_ckpt")
            fm_n_frames     = sys_cfg.get("force_mae_n_frames", 500)
            fm_analyze_dt   = sys_cfg.get("force_mae_analyze_dt_ps")   # ps; overrides n_frames
            fm_results      = {}
            for model, traj_path in traj_paths.items():
                p = _main_traj_path(traj_path)
                if not p.exists():
                    log_lines.append(f"  [ForceMae] SKIP {model}: missing {p}")
                    continue
                if _traj_fmt(traj_path) == "gromacs":
                    log_lines.append(f"  [ForceMae] SKIP {model}: only ASE .traj supported")
                    continue
                try:
                    dt   = _dt_fs(sys_cfg, model)
                    res  = compute_force_mae(
                        p, teacher_ckpt,
                        student_ckpt=student_ckpt,
                        n_frames=fm_n_frames,
                        dt_fs=dt,
                        max_ns=analysis_ns.get(model, max_traj_ns),
                        analyze_dt_ps=fm_analyze_dt,
                    )
                    log_lines.append(f"  [ForceMae] {model}: stride={res['stride']} "
                                     f"({res['stride']*dt*1e-3:.1f} ps/frame), "
                                     f"n_eval={len(res['times_ns'])}")
                    fm_results[model] = res
                    out = plot_force_mae_timeseries(res, name, model, sys_out,
                                                   color=colors.get(model, "#1f77b4"))
                    log_lines.append(f"  [ForceMae] {model}: saved {out.name}")
                    import numpy as np
                    npz = sys_out / f"force_mae_{model}.npz".replace(" ", "_")
                    np.savez_compressed(npz, **res)
                except Exception:
                    log_lines.append(f"  [ForceMae] ERROR {model}:\n{traceback.format_exc()}")

            if len(fm_results) > 1:
                out = plot_force_mae_comparison(fm_results, name, model_order, colors, sys_out)
                log_lines.append(f"  [ForceMae] comparison saved: {out.name}")

    # ── Energy MAE ────────────────────────────────────────────────────────────
    if "energy_mae" in analyses:
        teacher_ckpt = sys_cfg.get("energy_mae_teacher_ckpt")
        if not teacher_ckpt:
            log_lines.append("  [EnergyMae] SKIP: energy_mae_teacher_ckpt not set in config")
        else:
            student_ckpt  = sys_cfg.get("energy_mae_student_ckpt")
            em_n_frames   = sys_cfg.get("energy_mae_n_frames", 500)
            em_analyze_dt = sys_cfg.get("energy_mae_analyze_dt_ps")   # ps; overrides n_frames
            em_results    = {}
            for model, traj_path in traj_paths.items():
                p = _main_traj_path(traj_path)
                if not p.exists():
                    log_lines.append(f"  [EnergyMae] SKIP {model}: missing {p}")
                    continue
                if _traj_fmt(traj_path) == "gromacs":
                    log_lines.append(f"  [EnergyMae] SKIP {model}: only ASE .traj supported")
                    continue
                try:
                    dt  = _dt_fs(sys_cfg, model)
                    res = compute_energy_mae(
                        p, teacher_ckpt,
                        student_ckpt=student_ckpt,
                        n_frames=em_n_frames,
                        dt_fs=dt,
                        max_ns=analysis_ns.get(model, max_traj_ns),
                        analyze_dt_ps=em_analyze_dt,
                    )
                    mean_total    = float(res["mae_total"].mean())
                    mean_per_atom = float(res["mae_per_atom"].mean())
                    log_lines.append(
                        f"  [EnergyMae] {model}: stride={res['stride']} "
                        f"({res['stride']*dt*1e-3:.1f} ps/frame), "
                        f"n_eval={len(res['times_ns'])}, "
                        f"mean_total={mean_total:.4f} eV, "
                        f"mean_per_atom={mean_per_atom:.6f} eV/atom"
                    )
                    em_results[model] = res
                    out = plot_energy_mae_timeseries(res, name, model, sys_out,
                                                    color=colors.get(model, "#1f77b4"))
                    log_lines.append(f"  [EnergyMae] {model}: saved {out.name}")
                    import numpy as np
                    npz = sys_out / f"energy_mae_{model}.npz".replace(" ", "_")
                    np.savez_compressed(npz, **res)
                except Exception:
                    log_lines.append(f"  [EnergyMae] ERROR {model}:\n{traceback.format_exc()}")

            if len(em_results) > 1:
                out = plot_energy_mae_comparison(em_results, name, model_order, colors, sys_out)
                log_lines.append(f"  [EnergyMae] comparison saved: {out.name}")

    # ── Stress MAE ────────────────────────────────────────────────────────────
    if "stress_mae" in analyses:
        teacher_ckpt = sys_cfg.get("stress_mae_teacher_ckpt")
        if not teacher_ckpt:
            log_lines.append("  [StressMae] SKIP: stress_mae_teacher_ckpt not set in config")
        else:
            student_ckpt  = sys_cfg.get("stress_mae_student_ckpt")
            sm_n_frames   = sys_cfg.get("stress_mae_n_frames", 500)
            sm_analyze_dt = sys_cfg.get("stress_mae_analyze_dt_ps")   # ps; overrides n_frames
            sm_results    = {}
            for model, traj_path in traj_paths.items():
                p = _main_traj_path(traj_path)
                if not p.exists():
                    log_lines.append(f"  [StressMae] SKIP {model}: missing {p}")
                    continue
                if _traj_fmt(traj_path) == "gromacs":
                    log_lines.append(f"  [StressMae] SKIP {model}: only ASE .traj supported")
                    continue
                try:
                    dt  = _dt_fs(sys_cfg, model)
                    res = compute_stress_mae(
                        p, teacher_ckpt,
                        student_ckpt=student_ckpt,
                        n_frames=sm_n_frames,
                        dt_fs=dt,
                        max_ns=analysis_ns.get(model, max_traj_ns),
                        analyze_dt_ps=sm_analyze_dt,
                    )
                    log_lines.append(
                        f"  [StressMae] {model}: stride={res['stride']} "
                        f"({res['stride']*dt*1e-3:.1f} ps/frame), "
                        f"n_eval={len(res['times_ns'])}, "
                        f"mean={res['mae'].mean():.4e} eV/Å³"
                    )
                    sm_results[model] = res
                    out = plot_stress_mae_timeseries(res, name, model, sys_out,
                                                    color=colors.get(model, "#1f77b4"))
                    log_lines.append(f"  [StressMae] {model}: saved {out.name}")
                    import numpy as np
                    npz = sys_out / f"stress_mae_{model}.npz".replace(" ", "_")
                    np.savez_compressed(npz, **{k: v for k, v in res.items()
                                                if isinstance(v, np.ndarray)})
                except Exception:
                    log_lines.append(f"  [StressMae] ERROR {model}:\n{traceback.format_exc()}")

            if len(sm_results) > 1:
                out = plot_stress_mae_comparison(sm_results, name, model_order, colors, sys_out)
                log_lines.append(f"  [StressMae] comparison saved: {out.name}")

    # ── RMSD ──────────────────────────────────────────────────────────────────
    if "rmsd" in analyses:
        rmsd_data = {}
        rmsd_ref        = sys_cfg.get("rmsd_ref_frame_idx", 0)
        rmsd_n          = sys_cfg.get("rmsd_n_frames", 1000)
        rmsd_analyze_dt = sys_cfg.get("rmsd_analyze_dt_ps")   # ps; overrides n_frames
        for model, traj_path in traj_paths.items():
            p = _main_traj_path(traj_path)
            if not p.exists():
                log_lines.append(f"  [RMSD] SKIP {model}: missing {p}")
                continue
            if _traj_fmt(traj_path) == "gromacs":
                log_lines.append(f"  [RMSD] SKIP {model}: only ASE .traj supported")
                continue
            try:
                dt  = _dt_fs(sys_cfg, model)
                res = compute_rmsd(
                    p,
                    n_frames=rmsd_n,
                    dt_fs=dt,
                    max_ns=analysis_ns.get(model, max_traj_ns),
                    ref_frame_idx=rmsd_ref,
                    analyze_dt_ps=rmsd_analyze_dt,
                )
                rmsd_data[model] = (res["times_ns"], res["rmsd"])
                log_lines.append(f"  [RMSD] {model}: stride={res['stride']} "
                                 f"({res['stride']*dt*1e-3:.1f} ps/frame), "
                                 f"n_eval={len(res['times_ns'])}, "
                                 f"mean={res['rmsd'].mean():.3f} Å")
                import numpy as np
                npz = sys_out / f"rmsd_{model}.npz".replace(" ", "_")
                np.savez_compressed(npz, **res)
            except Exception:
                log_lines.append(f"  [RMSD] ERROR {model}:\n{traceback.format_exc()}")

        if rmsd_data:
            out = plot_rmsd_timeseries(
                rmsd_data, name, sys_out,
                model_colors=colors,
                roll_window_ns=sys_cfg.get("density_roll_window_ns", 0.5),
                eq_cutoff_ns=skip_ns,
            )
            log_lines.append(f"  [RMSD] saved: {out.name}")

    # ── Cell Size ─────────────────────────────────────────────────────────────
    if "cell_size" in analyses:
        cell_data = {}
        cell_n          = sys_cfg.get("cell_size_n_frames", n_frames * 2)
        cell_analyze_dt = sys_cfg.get("cell_size_analyze_dt_ps")   # ps; overrides n_frames
        for model, traj_path in traj_paths.items():
            p = _main_traj_path(traj_path)
            if not p.exists():
                log_lines.append(f"  [CellSize] SKIP {model}: missing {p}")
                continue
            try:
                dt  = _dt_fs(sys_cfg, model)
                res = extract_cell_timeseries(
                    p,
                    dt_fs=dt,
                    n_sample=cell_n,
                    max_ns=analysis_ns.get(model, max_traj_ns),
                    topology=topology_for.get(model),
                    preloaded_frames=preloaded_for.get(model),
                    analyze_dt_ps=cell_analyze_dt,
                )
                cell_data[model] = res
                log_lines.append(
                    f"  [CellSize] {model}: stride={res['stride']} "
                    f"({res['stride']*dt*1e-3:.1f} ps/frame), "
                    f"n_eval={len(res['times_ns'])}, "
                    f"a={res['a'].mean():.3f} Å  b={res['b'].mean():.3f} Å  "
                    f"c={res['c'].mean():.3f} Å  V={res['volume'].mean():.1f} Å³"
                )
                import numpy as np
                npz = sys_out / f"cell_size_{model}.npz".replace(" ", "_")
                np.savez_compressed(npz, **{k: v for k, v in res.items()
                                            if isinstance(v, np.ndarray)})
            except Exception:
                log_lines.append(f"  [CellSize] ERROR {model}:\n{traceback.format_exc()}")

        if cell_data:
            out = plot_cell_timeseries(
                cell_data, name, sys_out,
                model_colors=colors,
                roll_window_ns=sys_cfg.get("density_roll_window_ns", 0.5),
                eq_cutoff_ns=skip_ns,
            )
            log_lines.append(f"  [CellSize] saved: {out.name}")

    return "\n".join(log_lines)


# ── config loading ─────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    spec = importlib.util.spec_from_file_location("_eval_config", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run observable analysis (RDF/density/energy) for a list of systems.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True,
                        help="Path to Python config file defining SYSTEMS.")
    parser.add_argument("--analyses", default=None,
                        help="Comma-separated analyses: rdf,density,energy (default: from config or all).")
    parser.add_argument("--output-dir", default=None,
                        help="Root output directory (overrides config OUTPUT_DIR).")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: number of systems).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    systems = cfg.SYSTEMS

    analyses_default = getattr(cfg, "ANALYSES", ["rdf", "density", "energy", "msd"])
    analyses = [a.strip() for a in args.analyses.split(",")] if args.analyses else analyses_default

    root_dir = Path(args.output_dir or getattr(cfg, "OUTPUT_DIR", "./observable_results"))

    # create a timestamped run directory under the given root
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analyses_tag = "_".join(analyses)
    run_dir = root_dir / f"{timestamp}_{analyses_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    n_workers = args.workers or getattr(cfg, "WORKERS", None) or len(systems)
    n_workers = min(n_workers, len(systems))

    print(f"Systems   : {len(systems)}")
    print(f"Analyses  : {analyses}")
    print(f"Root      : {root_dir}")
    print(f"Run dir   : {run_dir}")
    print(f"Workers   : {n_workers}")
    print()

    # validate paths upfront
    for sys_cfg in systems:
        for model, tp in sys_cfg["traj_paths"].items():
            if not _main_traj_path(tp).exists():
                print(f"  WARNING missing: {sys_cfg['name']} / {model}: {_main_traj_path(tp)}")

    if n_workers == 1:
        for sys_cfg in systems:
            try:
                log = _run_system(sys_cfg, analyses, run_dir)
                print(log)
            except Exception:
                print(f"[FATAL] {sys_cfg['name']}:")
                traceback.print_exc()
    else:
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=_mp.get_context("spawn")) as exe:
            futures = {
                exe.submit(_run_system, sys_cfg, analyses, run_dir): sys_cfg["name"]
                for sys_cfg in systems
            }
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    log = fut.result()
                    print(log)
                except Exception:
                    print(f"[FATAL] {name}:")
                    traceback.print_exc()

    print("\nAll done.")


if __name__ == "__main__":
    main()
