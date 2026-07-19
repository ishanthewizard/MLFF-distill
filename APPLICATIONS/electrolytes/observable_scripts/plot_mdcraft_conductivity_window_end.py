"""Conductivity vs fit-window-END sweep for the mdcraft collective-Onsager backend.

The eval.py mdcraft backend (conductivity/compute.py::run_onsager_conductivity_mdcraft)
computes the collective cross-displacements <dR_i . dR_j>(t) with mdcraft's
`Onsager`, then fits a SINGLE lag window (default 0.3-2.4 ns) to get L_ij -> kappa.
That single number is what lands in the cross-displacement PNG / conductivity.csv.

This script asks a trustability question: how sensitive is the reported kappa to
where the fit window ENDS?  It reproduces the *exact* mdcraft pipeline for a system
(all its replicas), but runs `ons.run()` (the expensive collective-MSD compute) only
ONCE per replica, then re-fits L_ij -> kappa for a grid of fit-window END times with
the fit START held fixed (0.3 ns, matching the eval run).  Overlays kappa(fit_end)
for every replica + the cross-replica mean+/-1 s.d. band, and marks the nominal
2.4 ns window used by eval.py.

Because calculate_transport_coefficients()/calculate_conductivity() overwrite
results.L_ij / results.conductivity on each call, sweeping the window end is just
a sequence of cheap linear re-fits on the already-computed MSD arrays -- no reload.

Defaults target
  PAINN FP32 In-distribution NVT / naotf_dme__nvt_1M_298K_20ns_100fs
matching config_painn_fp32_indist_multireplica_nvt_conductivity.py
(dt_fs=100, T=298, eq_cut=2 ns, load_dt=5 ps, fit_start=0.3 ns).  Override any of
these on the CLI to point at another system.

Run on a SLURM CPU node (the CFS trajectories are slow random-access reads; the
`_load_unwrapped` pass dominates the wall-clock).  Replicas are loaded in parallel.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Reuse the EXACT helpers eval.py's mdcraft backend uses -- no reimplementation,
# so this sweep is guaranteed consistent with the reported cross-displacement kappa.
from conductivity.compute import (          # noqa: E402
    _load_unwrapped, _average_cell, _AseTraj,
    anion_central_dict, cation_dict, anion_dict, solvent_dict,
    direct_groups_from_species,
    _KAPPA_TO_SI, _SI_TO_USCM,
)

# ── defaults: PAINN FP32 In-dist NVT naotf_dme 1M 298K, 4 replicas ──────────────
_TRAJ_ROOT = Path(
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5250/distillation_proj/simulation_results/PAINN/electrolytes_data"
    "/simulation/FP32_simulation/In_distribution_exp/NVT_langevin"
)
_SYS_REL = "naotf_dme/nvt_1M_298K_20ns_100fs/nvt_1M_298K_20ns_100fs.traj"
_DEFAULT_TRAJS = {f"replica_{i}": str(_TRAJ_ROOT / f"nvt_replica_{i}" / _SYS_REL)
                  for i in range(4)}
_DEFAULT_OUT = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5250/distillation_proj/simulation_results/PAINN/electrolytes_data"
    "/analysis/fp32_simulation/In_distribution/multi_replicas/nvt"
    "/20260704_153405_conductivity/naotf_dme__nvt_1M_298K_20ns_100fs"
)
_REPLICA_COLORS = {
    "replica_0": "#1f77b4", "replica_1": "#ff7f0e",
    "replica_2": "#2ca02c", "replica_3": "#d62728",
    "replica_4": "#9467bd",
}


def _build_ons_and_run(traj_path, cat_symbol, anion_symbol, solvent_symbol,
                       dt_fs, T_K, z_cat, z_anion, eq_cut_ns, load_dt_ps,
                       max_traj_ns):
    """Mirror run_onsager_conductivity_mdcraft up to (and including) ons.run().

    Returns the *run* mdcraft Onsager object (collective MSDs computed) so the
    caller can re-fit any lag window cheaply. Skips the VACF diagnostic pass.
    """
    from mdcraft.analysis.transport import Onsager
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader

    traj_path = Path(traj_path)
    dt_ps  = dt_fs / 1000.0
    stride = max(1, round(load_dt_ps / dt_ps))
    actual_load_dt_ps = stride * dt_ps

    anion_central = anion_central_dict[anion_symbol]
    cat_list = cation_dict[cat_symbol]
    an_list  = anion_dict[anion_symbol]
    sol_list = solvent_dict[solvent_symbol]

    with _AseTraj(str(traj_path)) as trj:
        n_total_raw = len(trj)
        symbols0 = np.array(trj[0].get_chemical_symbols())

    n_total = min(n_total_raw, int(max_traj_ns * 1e3 / dt_ps))
    i_start = int(eq_cut_ns * 1e3 / dt_ps)
    if i_start >= n_total:
        raise RuntimeError("Equilibration cut longer than trajectory.")
    n_loaded = (n_total - i_start) // stride
    if n_loaded < 10:
        raise RuntimeError(f"Too few frames after subsampling: n_loaded={n_loaded}.")
    i_end = i_start + n_loaded * stride

    cat_groups     = direct_groups_from_species(symbols0, cat_list)
    anion_groups   = direct_groups_from_species(symbols0, an_list)
    solvent_groups = direct_groups_from_species(symbols0, sol_list)

    reorder_idx = (
        [int(i) for g in cat_groups     for i in g] +
        [int(i) for g in anion_groups   for i in g] +
        [int(i) for g in solvent_groups for i in g]
    )
    reorder_arr = np.asarray(reorder_idx)
    dims = _average_cell(traj_path, i_start, n_total)

    na_cols, an_cols, col = [], [], 0
    for g in cat_groups:
        na_cols.append(col); col += len(g)
    for g in anion_groups:
        local = int(np.where(symbols0[g] == anion_central)[0][0])
        an_cols.append(col + local); col += len(g)
    n_cat, n_an = len(na_cols), len(an_cols)
    ion_cols = na_cols + an_cols

    positions = _load_unwrapped(traj_path, reorder_idx, stride, i_start, i_end)
    ion_pos   = positions[:, ion_cols, :]

    with _AseTraj(str(traj_path)) as trj:
        masses0 = trj[0].get_masses()
    masses_re = masses0[reorder_idx]
    R = np.einsum("tnj,n->tj", positions, masses_re) / masses_re.sum()
    ion_pos_c = ion_pos - R[:, None, :]

    n_at = ion_pos_c.shape[1]
    u = mda.Universe.empty(n_at, n_residues=n_at,
                           atom_resindex=np.arange(n_at), trajectory=True)
    u.add_TopologyAttr("name", ["NA"] * n_cat + ["AN"] * n_an)
    u.load_new(ion_pos_c.astype(np.float32), format=MemoryReader)

    ons = Onsager([u.atoms[:n_cat], u.atoms[n_cat:]], groupings="atoms",
                  temperature=T_K, charges=[z_cat, z_anion],
                  dimensions=dims, dt=actual_load_dt_ps, unwrap=False,
                  center=False, fft=True, verbose=False)
    ons.run()
    return ons


def _sweep_one(job):
    """Worker: build+run mdcraft for one replica, sweep the fit-window end.

    Returns (label, fit_stops_ns, kappa_uScm array, max_lag_ns).
    """
    label, traj_path, params, fit_start_ns, fit_stops = job
    try:
        ons = _build_ons_and_run(traj_path, **params)
    except Exception as exc:  # noqa: BLE001
        print(f"  [{label}] FAILED: {exc}", flush=True)
        return label, np.asarray(fit_stops), np.full(len(fit_stops), np.nan), np.nan

    t = ons.results.times                     # lag time (ps)
    dt_lag_ns = (t[1] - t[0]) / 1000.0
    n_lag = len(t)
    s = max(1, int(round(fit_start_ns / dt_lag_ns)))
    kappas = np.full(len(fit_stops), np.nan)
    for k, fs in enumerate(fit_stops):
        e = min(n_lag, int(round(fs / dt_lag_ns)))
        if e <= s:
            continue
        ons.calculate_transport_coefficients(start=s, stop=e, scale="linear")
        ons.calculate_conductivity()
        kappas[k] = float(ons.results.conductivity[0] * _KAPPA_TO_SI * _SI_TO_USCM)
    max_lag_ns = float(t[-1] / 1000.0)
    print(f"  [{label}] swept {len(fit_stops)} window ends "
          f"(fit start {fit_start_ns} ns, max lag {max_lag_ns:.1f} ns)", flush=True)
    return label, np.asarray(fit_stops), kappas, max_lag_ns


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajs", nargs="+", default=None,
                    help="Replica .traj paths (default: naotf_dme 4 replicas).")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Labels for --trajs (default: replica_0..).")
    ap.add_argument("--out-dir", default=_DEFAULT_OUT)
    ap.add_argument("--system-name", default="naotf_dme__nvt_1M_298K_20ns_100fs")
    # physics / windowing (match config_painn_fp32_indist_multireplica_nvt_conductivity.py)
    ap.add_argument("--cat-symbol", default="Na")
    ap.add_argument("--anion-symbol", default="OTf")
    ap.add_argument("--solvent-symbol", default="DME")
    ap.add_argument("--dt-fs", type=float, default=100.0)
    ap.add_argument("--T-K", type=float, default=298.0)
    ap.add_argument("--z-cat", type=float, default=1.0)
    ap.add_argument("--z-anion", type=float, default=-1.0)
    ap.add_argument("--eq-cut-ns", type=float, default=2.0)
    ap.add_argument("--load-dt-ps", type=float, default=5.0)
    ap.add_argument("--max-traj-ns", type=float, default=20.0)
    # fit-window sweep
    ap.add_argument("--fit-start-ns", type=float, default=0.3,
                    help="Fixed fit-window START (matches eval run default).")
    ap.add_argument("--fit-end-min-ns", type=float, default=0.5)
    ap.add_argument("--fit-end-max-ns", type=float, default=10.0)
    ap.add_argument("--fit-end-step-ns", type=float, default=0.2)
    ap.add_argument("--nominal-fit-end-ns", type=float, default=2.4,
                    help="Window end used by the eval run (drawn as a marker line).")
    ap.add_argument("--nproc", type=int, default=4)
    args = ap.parse_args()

    if args.trajs:
        labels = args.labels or [f"replica_{i}" for i in range(len(args.trajs))]
        trajs = dict(zip(labels, args.trajs))
    else:
        trajs = dict(_DEFAULT_TRAJS)
    trajs = {lab: p for lab, p in trajs.items() if Path(p).exists()}
    if not trajs:
        raise SystemExit("No existing trajectories to analyze.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fit_stops = np.round(
        np.arange(args.fit_end_min_ns,
                  args.fit_end_max_ns + 1e-9,
                  args.fit_end_step_ns), 4)

    params = dict(
        cat_symbol=args.cat_symbol, anion_symbol=args.anion_symbol,
        solvent_symbol=args.solvent_symbol, dt_fs=args.dt_fs, T_K=args.T_K,
        z_cat=args.z_cat, z_anion=args.z_anion, eq_cut_ns=args.eq_cut_ns,
        load_dt_ps=args.load_dt_ps, max_traj_ns=args.max_traj_ns,
    )

    jobs = [(lab, p, params, args.fit_start_ns, fit_stops)
            for lab, p in trajs.items()]

    print(f"[window-end sweep] {args.system_name}: {len(jobs)} replicas, "
          f"{len(fit_stops)} window ends [{fit_stops[0]}..{fit_stops[-1]}] ns, "
          f"fit start {args.fit_start_ns} ns", flush=True)

    nproc = max(1, min(args.nproc, len(jobs)))
    if nproc > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(nproc, maxtasksperchild=1) as pool:
            results = pool.map(_sweep_one, jobs)
    else:
        results = [_sweep_one(j) for j in jobs]

    results = sorted(results, key=lambda r: r[0])

    # ── CSV: one row per (replica, fit_end) ───────────────────────────────────
    csv_path = out_dir / "conductivity_mdcraft_window_end_sweep.csv"
    with open(csv_path, "w") as fh:
        fh.write("system,replica,fit_start_ns,fit_end_ns,kappa_uS_cm,kappa_mS_cm\n")
        for label, fe, kap, _ in results:
            for e_ns, k in zip(fe, kap):
                fh.write(f"{args.system_name},{label},{args.fit_start_ns},"
                         f"{e_ns},{k:.6f},{k / 1e3:.6f}\n")
    print(f"[window-end sweep] wrote {csv_path}", flush=True)

    _plot(results, args, fit_stops, out_dir)


def _plot(results, args, fit_stops, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # stack replicas onto a common grid for the mean +/- s.d. band
    stack = np.vstack([kap for (_lab, _fe, kap, _ml) in results])   # (n_rep, n_end)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(stack, axis=0)
        sd   = np.nanstd(stack, axis=0)

    fig, ax = plt.subplots(figsize=(8.2, 5.4))

    for label, fe, kap, _ml in results:
        c = _REPLICA_COLORS.get(label, None)
        ax.plot(fe, kap, "-o", ms=3.2, lw=1.4, color=c, alpha=0.85, label=label)

    ax.plot(fit_stops, mean, "-", color="k", lw=2.4, zorder=5,
            label=f"mean (n={stack.shape[0]})")
    ax.fill_between(fit_stops, mean - sd, mean + sd, color="k", alpha=0.12,
                    zorder=0, label=r"$\pm$1 s.d.")

    # nominal window end used by the eval run + the kappa there
    ax.axvline(args.nominal_fit_end_ns, color="0.4", ls="--", lw=1.3, zorder=1)
    j = int(np.argmin(np.abs(fit_stops - args.nominal_fit_end_ns)))
    ax.annotate(f"eval fit end = {args.nominal_fit_end_ns:g} ns\n"
                fr"mean $\kappa$ = {mean[j]:.0f} $\mu$S/cm",
                xy=(args.nominal_fit_end_ns, mean[j]),
                xytext=(0.62, 0.9), textcoords="axes fraction",
                fontsize=9, color="0.25",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=1))

    ax.set_xlabel(f"fit-window END (ns)   [fit start fixed at {args.fit_start_ns:g} ns]")
    ax.set_ylabel(r"collective-Onsager conductivity $\kappa$ ($\mu$S/cm)")
    ax.set_title(f"{args.system_name}\n"
                 r"mdcraft $\kappa$ vs cross-displacement fit-window end")
    ax.grid(True, ls=":", alpha=0.5)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, ncol=2, loc="lower right", framealpha=0.9)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        p = out_dir / f"conductivity_mdcraft_window_end_overlay.{ext}"
        fig.savefig(p, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"[window-end sweep] wrote {p}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
