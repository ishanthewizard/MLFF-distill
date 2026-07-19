"""Config: PAINN MLP-ablation comparison for naotf_dme (NPT, 298 K, 20 ns).

Directory layout under
  .../PAINN/mlp_ablation/<mlp_variant>/naotf_dme/<run_dir>/<run_dir>.traj
with three MLP variants {foam, eeacsf, gaussian_moments} and two
concentrations {0.1 M, 1 M}.

Grouping: one SYSTEM per concentration (run_dir), overlaying the three MLP
variants as "models" (same salt/solvent/box/temperature, only the MLP differs),
so foam vs eeacsf vs gaussian_moments are compared on the same axes.

Analyses (per user request):
  - energy    -> potential energy + temperature time series (pe/ke/etot/temp)
  - pressure  -> pressure time series
  - cell_size -> a/b/c/volume time series
  - msd       -> MSD / diffusivity / Nernst-Einstein conductivity

Windowing:
  - Full 20 ns simulation             -> max_traj_ns = 20.0
  - dt = 100 fs / saved frame         -> dt_fs = 100.0  (frames saved every 100 fs)
  - MSD/conductivity discard the first 2 ns  -> eq_cut_ns = 2.0
  - Diffusivity/conductivity fit restricted to lag tau in [2, 4] ns
        (tau_min_fit_ns=2.0, fit_pct=4/18 for msd; conductivity_tau_{min,max}_ns
        =2/4 -> byteff2/mdcraft Onsager nt_start=200, nt_end=400 at 10 ps/lag)
  - MSD effective lag spacing = 10 ps -> n_frames sized over the post-eq_cut window:
        n_frames = (max_traj_ns - eq_cut_ns) * 1000 / 10 = (20-2)*100 = 1800
  - cell_size / pressure use explicit analyze_dt_ps = 10 ps over the full 20 ns
  - energy uses n_frames (=1800) over the full 20 ns (~11 ps resolution)

Run with:
  /global/homes/y/yuejian/project/MLFF-distill/yuejian/envs/fairchemV2_new/bin/python \
    eval.py --config config_painn_mlp_ablation_naotf_dme.py
"""
from pathlib import Path

_ROOT = Path(
    "/global/homes/y/yuejian/project/MLFF-distill"
    "/m5250/distillation_proj/simulation_results/PAINN/mlp_ablation"
)

# MLP variants -> "model" labels (overlaid per system); plot order = insertion order
MLP_VARIANTS = ["foam", "eeacsf", "gaussian_moments"]
MODEL_COLORS = {
    "foam":             "#1f77b4",
    "eeacsf":           "#ff7f0e",
    "gaussian_moments": "#2ca02c",
}

# naotf_dme  ->  Na / OTf / DME (required by msd)
CAT_SYMBOL     = "Na"
ANION_SYMBOL   = "OTf"
SOLVENT_SYMBOL = "DME"
TEMPERATURE_K  = 298.0

DT_FS       = 100.0   # 100 fs / saved frame
MAX_TRAJ_NS = 20.0    # full 20 ns (eval.py caps to actual traj length)

# MSD / conductivity windowing
EQ_CUT_NS        = 2.0    # discard first 2 ns ("start from 2 ns to get displacement")
TARGET_MSD_DT_PS = 10.0   # effective MSD lag spacing
# Diffusivity / conductivity linear fit restricted to lag tau in [2, 4] ns.
TAU_MIN_FIT_NS   = 2.0    # fit lower bound (ns)
TAU_MAX_FIT_NS   = 4.0    # fit upper bound (ns)
# MSD fit upper bound = fit_pct * max_lag; max_lag ~= (MAX_TRAJ_NS - EQ_CUT_NS).
FIT_PCT          = TAU_MAX_FIT_NS / (MAX_TRAJ_NS - EQ_CUT_NS)   # ~0.222 -> tau_max ~4 ns
SLIDE_WINDOW_NS  = 1.0    # convergence-panel window (< fit width so it isn't empty)
SLIDE_STEP_NS    = 0.5
N_CONV_POINTS    = 20

# n_frames sized so the subsampled MSD lag step == TARGET_MSD_DT_PS (post eq_cut)
N_FRAMES = max(1, round((MAX_TRAJ_NS - EQ_CUT_NS) * 1000.0 / TARGET_MSD_DT_PS))  # = 1800

# Cell-size / pressure time-series resolution over the full 20 ns
ANALYZE_DT_PS = 10.0

# ── discover {run_dir -> {mlp_variant: traj_path}} across the three variants ────
_systems: dict[str, dict[str, str]] = {}
for variant in MLP_VARIANTS:
    sys_dir = _ROOT / variant / "naotf_dme"
    if not sys_dir.is_dir():
        continue
    for run_dir in sorted(p for p in sys_dir.glob("*/") if p.is_dir()):
        traj = run_dir / f"{run_dir.name}.traj"
        if not traj.exists():
            continue
        _systems.setdefault(run_dir.name, {})[variant] = str(traj)

SYSTEMS = []
for run_name in sorted(_systems):
    traj_paths = {v: _systems[run_name][v] for v in MLP_VARIANTS if v in _systems[run_name]}
    SYSTEMS.append({
        "name":            f"naotf_dme_{run_name}",
        "traj_paths":      traj_paths,
        "model_colors":    dict(MODEL_COLORS),
        "dt_fs":           DT_FS,
        "max_traj_ns":     MAX_TRAJ_NS,
        "n_frames":        N_FRAMES,
        "temperature_K":   TEMPERATURE_K,
        # ── cell_size / pressure (full 20 ns, explicit dt) ──
        "cell_size_analyze_dt_ps": ANALYZE_DT_PS,
        "pressure_analyze_dt_ps":  ANALYZE_DT_PS,
        # ── msd ──
        "cat_symbol":      CAT_SYMBOL,
        "anion_symbol":    ANION_SYMBOL,
        "solvent_symbol":  SOLVENT_SYMBOL,
        "eq_cut_ns":       EQ_CUT_NS,
        "fit_pct":         FIT_PCT,
        "tau_min_fit_ns":  TAU_MIN_FIT_NS,
        "slide_window_ns": SLIDE_WINDOW_NS,
        "slide_step_ns":   SLIDE_STEP_NS,
        "n_conv_points":   N_CONV_POINTS,
        # ── conductivity (same 2-4 ns fit window as msd) ──
        "conductivity_eq_cut_ns":  EQ_CUT_NS,
        "conductivity_load_dt_ps": TARGET_MSD_DT_PS,
        "conductivity_tau_min_ns": TAU_MIN_FIT_NS,
        "conductivity_tau_max_ns": TAU_MAX_FIT_NS,
    })

OUTPUT_DIR = str(_ROOT / "analysis")
ANALYSES   = ["energy", "pressure", "cell_size", "msd"]
WORKERS    = 2   # 2 systems; no teacher re-inference, so >1 is fine
