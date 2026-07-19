"""Config: multi-replica comparison for the PAINN FP32 In-distribution NPT set.

Compares the 4 NPT replicas (npt_replica_0..3) of every electrolyte system on
their full-trajectory timeseries of:
  - pressure           -> per-replica 4-panel + replica-overlay comparison
  - temperature + PE    -> via the `energy` analysis (4-panel per replica +
                           replica-overlay `energy_comparison_*` for E_pot/E_tot/T)
  - cell size           -> replica-overlay 4-panel (a/b/c/volume)

Each unique system = `<salt>_<solvent>/<run_dir>` (so 0.1 M vs 1 M and the three
lipf6 temperatures stay separate). Within a system, every replica that has the
trajectory becomes a "model" labelled `replica_<i>`, so the 4 replicas are drawn
on the same axes — appropriate here because they share composition/box, making
absolute PE / E_tot / T directly comparable.

dt_fs / max_traj_ns are hardcoded per the .fennol.yaml (dt[fs]=1.0, tdump[ps]=0.1
-> dt_fs=100, i.e. 0.1 ps/saved-frame), regardless of the dirname labels
(`1ns_1ps`, `2ns_100fs`). The runs target ~5 ns; max_traj_ns=20 means "use the
whole trajectory" since eval.py caps it to the actual length.

Run with:
  /global/homes/y/yuejian/project/MLFF-distill/yuejian/envs/fairchemV2_new/bin/python \
    eval.py --config config_painn_fp32_indist_multireplica.py
"""
import re
from pathlib import Path

_REPLICA_ROOT = Path(
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5250/distillation_proj/simulation_results/PAINN/electrolytes_data"
    "/simulation/FP32_simulation/In_distribution_exp/NPT_langevin"
)
REPLICA_DIRS = {
    "replica_0": _REPLICA_ROOT / "npt_replica_0",
    "replica_1": _REPLICA_ROOT / "npt_replica_1",
    "replica_2": _REPLICA_ROOT / "npt_replica_2",
    "replica_3": _REPLICA_ROOT / "npt_replica_3",
}

REPLICA_COLORS = {
    "replica_0": "#1f77b4",
    "replica_1": "#ff7f0e",
    "replica_2": "#2ca02c",
    "replica_3": "#d62728",
}

DT_FS = 100.0          # tdump=0.1 ps, dt=1 fs -> 100 fs/saved-frame (dirname unreliable)
MAX_TRAJ_NS = 20.0     # eval.py caps to actual traj length (~5 ns runs)

SYSTEM_DIR_RE = re.compile(r"^[a-z0-9]+_[a-z]+$")   # e.g. naotf_dme, lipf6_dme

# ── discover the union of <salt_solvent>/<run_dir> across all replicas ──────────
# {relative "system/run": {replica_label: traj_path}}
_systems: dict[str, dict[str, str]] = {}
for rlabel, rroot in REPLICA_DIRS.items():
    if not rroot.is_dir():
        continue
    for sys_dir in sorted(p for p in rroot.glob("*/") if p.is_dir()):
        if not SYSTEM_DIR_RE.match(sys_dir.name):
            continue
        for run_dir in sorted(p for p in sys_dir.glob("*/") if p.is_dir()):
            traj = run_dir / f"{run_dir.name}.traj"
            if not traj.exists():
                continue
            key = f"{sys_dir.name}/{run_dir.name}"
            _systems.setdefault(key, {})[rlabel] = str(traj)

SYSTEMS = []
for key in sorted(_systems):
    traj_paths = {r: _systems[key][r] for r in REPLICA_DIRS if r in _systems[key]}
    SYSTEMS.append({
        "name":                   key.replace("/", "__"),  # e.g. naotf_dme__npt_1M_298K_2ns_100fs
        "traj_paths":             traj_paths,
        "model_colors":           dict(REPLICA_COLORS),
        "dt_fs":                  DT_FS,
        "max_traj_ns":            MAX_TRAJ_NS,
        "skip_ns":                0.5,    # equilibration cutoff line / smoothing ref
        # These are full 20 ns / 200k-frame trajectories with ~0.4 s/frame random
        # reads, so sample every ~20 ps (~1000 points over 20 ns). Plenty of
        # resolution for these smooth NPT observables / overlay comparisons while
        # keeping the 72-trajectory run tractable.
        "n_frames":               1000,   # energy timeseries samples (~20 ps/frame)
        "pressure_analyze_dt_ps": 20.0,   # sample pressure every 20 ps of sim time
        "cell_size_analyze_dt_ps":20.0,   # sample cell every 20 ps of sim time
    })

OUTPUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5250/distillation_proj/simulation_results/PAINN/electrolytes_data"
    "/analysis/fp32_simulation/In_distribution/multi_replicas"
)
ANALYSES = ["pressure", "energy", "cell_size"]   # energy -> temperature + potential energy
WORKERS  = 6   # parallel across systems; no MAE here so no teacher re-inference
