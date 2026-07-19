"""Config: force MAE + energy MAE + stress MAE for the PAINN electrolytes_data
`exp_06_ood` simulation set.

Teacher : UMA-S-1p1 (re-inferred on each sampled frame).
Student : the PAINN/FENNIX values already stored in each `.traj`
          (SinglePointCalculator with energy / forces / stress) — so
          `*_student_ckpt = None` and no student re-inference is needed.

Directory layout under `_BASE`:

    exp_06_ood/<outer>/<salt>_<solvent>/<npt|nvt>_<conc>M_<temp>K_len20ns_dt100fs/<run>.traj

`<outer>` is a batch/pairing label that may repeat the same physical
`<salt>_<solvent>` across different outer dirs, so the system identity (and the
per-system output folder name) is `<outer>__<salt>_<solvent>` to stay unique.
Within one (outer, salt_solvent) the `npt` and `nvt` runs are grouped as two
*models* of one SYSTEMS entry, so their MAE timeseries land in one overlaid
comparison plot per system.

dt_fs / max_traj_ns are hardcoded per the analyze-md notes: every run's
`.fennol.yaml` has dt[fs]=1.0, tdump[ps]=0.1 (-> dt_fs=100) and nsteps=20e6
(20 ns target) regardless of the dirname. Each verified traj has 200000 frames
(= 20 ns @ 100 fs). analyze_dt_ps=100 -> stride 1000 -> ~200 eval frames per traj.

NOTE: force MAE and stress MAE are reference-independent and directly meaningful.
Energy MAE compares the *absolute total* PAINN energy against UMA-S-1p1; because
the two models use different energy references, mae_total is dominated by a
constant offset. Treat its *variation over time* (not its absolute value) as the
informative signal.

Run with (env has ase + fairchem/UMA + numpy/pandas/matplotlib):
  /global/homes/y/yuejian/project/MLFF-distill/yuejian/envs/fairchemV2_new/bin/python \
    eval.py --config config_mae_painn_electrolytes_exp_06_ood.py
"""
import re
from pathlib import Path

_BASE = Path(
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5250/distillation_proj/simulation_results/PAINN/electrolytes_data/exp_06_ood"
)

# UMA-S-1p1 teacher checkpoint (same one used by the prior MAE configs).
_TEACHER_CKPT = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m4558/distillation_project/models/uma-s-1p1.pt"
)

ION_MAP = {
    "lipf6": ("Li", "PF6"),
    "napf6": ("Na", "PF6"),
    "naotf": ("Na", "OTf"),
}
SOLVENT_SYMBOL = {
    "dme":     "DME",
    "diglyme": "Diglyme",
    "tgdme":   "TGDME",
    "pc":      "PC",
}

SYSTEM_DIR_RE = re.compile(r"^([a-z0-9]+)_([a-z]+)$")
RUN_DIR_RE = re.compile(r"^(npt|nvt)_(\d+(?:_\d+)?)M_(\d+)K_")

DT_FS = 100.0          # tdump = 0.1 ps -> 100 fs/frame
MAX_TRAJ_NS = 20.0     # eval.py caps to actual traj length
ANALYZE_DT_PS = 100.0  # 100 ps -> stride 1000 -> ~200 eval frames / 20 ns traj
ENSEMBLE_COLOR = {"npt": "#1f77b4", "nvt": "#d62728"}

SYSTEMS = []
for outer_dir in sorted(p for p in _BASE.glob("*/") if p.is_dir()):
    for sys_dir in sorted(p for p in outer_dir.glob("*/") if p.is_dir()):
        m = SYSTEM_DIR_RE.match(sys_dir.name)
        if not m:
            continue
        salt, solvent_tok = m.groups()
        if salt not in ION_MAP or solvent_tok not in SOLVENT_SYMBOL:
            continue

        traj_paths = {}
        for run_dir in sorted(p for p in sys_dir.glob("*/") if p.is_dir()):
            mr = RUN_DIR_RE.match(run_dir.name)
            if not mr:
                continue
            ensemble = mr.group(1)
            traj_path = run_dir / f"{run_dir.name}.traj"
            if not traj_path.exists():
                continue
            traj_paths[ensemble] = str(traj_path)

        if not traj_paths:
            continue

        # deterministic model order: npt first, then nvt
        ordered = {k: traj_paths[k] for k in ("npt", "nvt") if k in traj_paths}
        ordered.update({k: v for k, v in traj_paths.items() if k not in ordered})
        colors = {k: ENSEMBLE_COLOR.get(k, "#2ca02c") for k in ordered}

        SYSTEMS.append({
            "name":         f"{outer_dir.name}__{salt}_{solvent_tok}",
            "traj_paths":   ordered,
            "dt_fs":        DT_FS,
            "max_traj_ns":  MAX_TRAJ_NS,
            "model_colors": colors,
            # ── force MAE (teacher UMA-S-1p1 vs stored PAINN forces) ──
            "force_mae_teacher_ckpt":  _TEACHER_CKPT,
            "force_mae_student_ckpt":  None,          # use forces stored in .traj
            "force_mae_analyze_dt_ps": ANALYZE_DT_PS,
            "force_mae_n_frames":      200,           # fallback (ignored: analyze_dt_ps set)
            # ── energy MAE (teacher UMA-S-1p1 vs stored PAINN energy) ──
            "energy_mae_teacher_ckpt":  _TEACHER_CKPT,
            "energy_mae_student_ckpt":  None,         # use energy stored in .traj
            "energy_mae_analyze_dt_ps": ANALYZE_DT_PS,
            "energy_mae_n_frames":      200,
            # ── stress MAE (teacher UMA-S-1p1 vs stored PAINN stress) ──
            "stress_mae_teacher_ckpt":  _TEACHER_CKPT,
            "stress_mae_student_ckpt":  None,         # use stress stored in .traj
            "stress_mae_analyze_dt_ps": ANALYZE_DT_PS,
            "stress_mae_n_frames":      200,
        })

OUTPUT_DIR = (
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5250/distillation_proj/simulation_results/PAINN/electrolytes_data/analysis"
)
ANALYSES = ["force_mae", "energy_mae", "stress_mae"]
WORKERS  = 1   # GPU-bound (teacher re-inference); one system at a time
