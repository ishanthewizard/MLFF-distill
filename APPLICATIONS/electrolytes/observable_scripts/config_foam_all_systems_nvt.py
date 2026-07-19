"""Config: multi-replica timeseries (potential energy + cell size) for the
MLP `foam_all_systems` NVT set.

MLP analogue of `config_painn_fp32_indist_multireplica_nvt.py`.
Compares the 4 NVT replicas (nvt_replica_1..4) of every electrolyte system on
their full-trajectory timeseries of:
  - potential energy   -> via the `energy` analysis (4-panel per replica +
                          replica-overlay `energy_comparison_*` for E_pot/E_tot/T)
  - cell size           -> replica-overlay 4-panel (a/b/c/volume)

NOTE: these are NVT (fixed-box) runs, so cell size is constant by construction —
the cell_size panels are effectively a flat sanity check that the box was held
fixed. Included per user request.

Each unique system = `<salt>_<solvent>/<run_dir>` (so 0.1 M vs 1 M and the three
lipf6 temperatures stay separate). Within a system, every replica that has the
trajectory becomes a "model" labelled `replica_<i>`, so the replicas are drawn on
the same axes — appropriate here because they share composition/box/temperature,
making absolute PE / E_tot / T directly comparable. All 18 systems have all 4
replicas (72 trajectories total).

dt_fs / max_traj_ns are hardcoded per the .fennol.yaml (dt[fs]=1.0, tdump[ps]=0.1
-> dt_fs=100, i.e. 0.1 ps/saved-frame), regardless of the dirname labels
(`20ns_100fs`). max_traj_ns=20 means "use the whole trajectory" since eval.py
caps it to the actual length (all runs here target 20 ns).

Run with:
  /global/homes/y/yuejian/project/MLFF-distill/yuejian/envs/fairchemV2_new/bin/python \
    eval.py --config config_foam_all_systems_nvt.py
"""
import re
from pathlib import Path

_REPLICA_ROOT = Path(
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5250/distillation_proj/simulation_results/MLP/foam_all_systems"
)
REPLICA_DIRS = {
    "replica_1": _REPLICA_ROOT / "nvt_replica_1",
    "replica_2": _REPLICA_ROOT / "nvt_replica_2",
    "replica_3": _REPLICA_ROOT / "nvt_replica_3",
    "replica_4": _REPLICA_ROOT / "nvt_replica_4",
}

REPLICA_COLORS = {
    "replica_1": "#1f77b4",
    "replica_2": "#ff7f0e",
    "replica_3": "#2ca02c",
    "replica_4": "#d62728",
}

DT_FS = 100.0          # tdump=0.1 ps, dt=1 fs -> 100 fs/saved-frame (dirname unreliable)
MAX_TRAJ_NS = 20.0     # eval.py caps to actual traj length (20 ns runs)

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
        "name":                   key.replace("/", "__"),  # e.g. naotf_dme__nvt_1M_298K_20ns_100fs
        "traj_paths":             traj_paths,
        "model_colors":           dict(REPLICA_COLORS),
        "dt_fs":                  DT_FS,
        "max_traj_ns":            MAX_TRAJ_NS,
        "skip_ns":                0.5,    # equilibration cutoff line / smoothing ref
        # These are full 20 ns / 200k-frame trajectories with slow random reads,
        # so sample every ~20 ps (~1000 points over 20 ns). Plenty of resolution
        # for these smooth timeseries / overlay comparisons while keeping the
        # 72-trajectory run tractable.
        "n_frames":               1000,   # energy timeseries samples (~20 ps/frame)
        "cell_size_analyze_dt_ps":20.0,   # sample cell every 20 ps of sim time
    })

OUTPUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5250/distillation_proj/simulation_results/MLP/foam_all_systems/analysis"
)
ANALYSES = ["energy", "cell_size"]   # energy -> potential energy (+ E_tot / T)
WORKERS  = 6   # parallel across systems; no MAE here so no teacher re-inference
