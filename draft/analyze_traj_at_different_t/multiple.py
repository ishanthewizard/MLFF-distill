"""
Batch wrapper to run the two-temperature analysis across multiple systems.

For each system, this sets the 293K/323K trajectory paths and per-system
output directory, then calls the main analysis in script.py. Results (plots
and stdout) are written per system under the configured plot root.

Running instructions (NERSC/HPC friendly)
----------------------------------------
Use the fairchemV2 environment Python directly:

  /global/homes/y/yuejian/project/MLFF-distill/yuejian/envs/fairchemV2/bin/python \
    /global/homes/y/yuejian/project/MLFF-distill/draft/analyze_traj_at_different_t/multiple.py

All systems, first 500 frames:

  /global/homes/y/yuejian/project/MLFF-distill/yuejian/envs/fairchemV2/bin/python \
    /global/homes/y/yuejian/project/MLFF-distill/draft/analyze_traj_at_different_t/multiple.py \
    --n-frames 500 --stride 10 --bins 40 --cluster-cutoff 2.0

Quick smoke test (1 system, fewer frames):

  /global/homes/y/yuejian/project/MLFF-distill/yuejian/envs/fairchemV2/bin/python \
    /global/homes/y/yuejian/project/MLFF-distill/draft/analyze_traj_at_different_t/multiple.py \
    --max-systems 1 --n-frames 200 --stride 20 --bins 40 --cluster-cutoff 2.0

Common options:
  --only naotf_diglyme naotf_dme        # run specific systems by name
  --plot-root /path/to/output_root      # write plots under per-system subdirs
"""

import sys
from pathlib import Path
import argparse
import os
from typing import List, Dict, Optional

# Ensure project root is on sys.path so `import draft...` works when running by path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from draft.analyze_traj_at_different_t import script as single


# Root output directory; each system gets its own subfolder
PLOT_ROOT = "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/plot"


# System mapping: name, 293K traj path, 323K traj path
SYSTEMS: List[Dict[str, str]] = [
    {
        "name": "naotf_diglyme",
        "traj_293": "/global/homes/y/yuejian/project/MLFF-distill/m5024/UMA_trajs_293K/naotf_diglyme.traj",
        "traj_323": "/global/homes/y/yuejian/project/MLFF-distill/m4558/323K_500ps_trajs/md_omol_naotf_diglyme_1m_s1p1/md_omol_naotf_diglyme_1m_s1p1.traj",
        "traj_353": "/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/data/raw_data_from_UMA_simulation/353K/all_systems_353K_UMAs1p1/naotf_diglyme/naotf_diglyme.traj",
    },
    {
        "name": "naotf_dme",
        "traj_293": "/global/homes/y/yuejian/project/MLFF-distill/m5024/UMA_trajs_293K/naotf_dme.traj",
        "traj_323": "/global/homes/y/yuejian/project/MLFF-distill/m4558/323K_500ps_trajs/md_omol_naotf_dme_s1p1_omol/md_omol_naotf_dme_s1p1_omol.traj",
        "traj_353": "/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/data/raw_data_from_UMA_simulation/353K/all_systems_353K_UMAs1p1/naotf_dme/naotf_dme.traj",
    },
    {
        "name": "naotf_pc",
        "traj_293": "/global/homes/y/yuejian/project/MLFF-distill/m5024/UMA_trajs_293K/naotf_pc.traj",
        "traj_323": "/global/homes/y/yuejian/project/MLFF-distill/m4558/323K_500ps_trajs/md_omol_naotf_pc_1m_s1p1/md_omol_naotf_pc_1m_s1p1.traj",
        "traj_353": "/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/data/raw_data_from_UMA_simulation/353K/all_systems_353K_UMAs1p1/naotf_pc/naotf_pc.traj",
    },
    {
        "name": "naotf_tgdme",
        "traj_293": "/global/homes/y/yuejian/project/MLFF-distill/m5024/UMA_trajs_293K/naotf_tgdme.traj",
        "traj_323": "/global/homes/y/yuejian/project/MLFF-distill/m4558/323K_500ps_trajs/md_omol_naotf_tgdme_1m_s1p1/md_omol_naotf_tgdme_1m_s1p1.traj",
        "traj_353": "/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/data/raw_data_from_UMA_simulation/353K/all_systems_353K_UMAs1p1/naotf_tgdme/naotf_tgdme.traj",
    },
    {
        "name": "natfsi_dme",
        "traj_293": "/global/homes/y/yuejian/project/MLFF-distill/m5024/UMA_trajs_293K/natfsi_dme.traj",
        "traj_323": "/global/homes/y/yuejian/project/MLFF-distill/m4558/323K_500ps_trajs/md_omol_natfsi_dme_s1p1/md_omol_natfsi_dme_s1p1.traj",
        "traj_353": "/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/data/raw_data_from_UMA_simulation/353K/all_systems_353K_UMAs1p1/natfsi_dme/natfsi_dme.traj",
    },
    {
        "name": "lipf6_dme",
        "traj_293": "/global/homes/y/yuejian/project/MLFF-distill/m5024/UMA_trajs_293K/lipf6_dme.traj",
        "traj_323": "/global/homes/y/yuejian/project/MLFF-distill/m4558/323K_500ps_trajs/md_omol_lipf6_pfactor_0.1_1fs_mask_t/md_omol_lipf6_pfactor_0.1_1fs_mask_t.traj",
        "traj_353": "/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/data/raw_data_from_UMA_simulation/353K/all_systems_353K_UMAs1p1/lipf6_dme/lipf6_dme.traj",
    },
    {
        "name": "cspf6_dme",
        "traj_293": "/global/homes/y/yuejian/project/MLFF-distill/m5024/UMA_trajs_293K/cspf6_dme.traj",
        "traj_323": "/global/homes/y/yuejian/project/MLFF-distill/m4558/323K_500ps_trajs/md_omol_cspf6_pfactor_0.1_1fs_mask_t/md_omol_cspf6_pfactor_0.1_1fs_mask_t.traj",
        "traj_353": "/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/data/raw_data_from_UMA_simulation/353K/all_systems_353K_UMAs1p1/cspf6_dme/cspf6_dme.traj",
    },
    {
        "name": "napf6_tgdme",
        "traj_293": "/global/homes/y/yuejian/project/MLFF-distill/m5024/UMA_trajs_293K/napf6_tgdme.traj",
        "traj_323": "/global/homes/y/yuejian/project/MLFF-distill/m4558/323K_500ps_trajs/md_omol_napf6_tgdme_1m_s1p1/md_omol_napf6_tgdme_1m_s1p1.traj",
        "traj_353": "/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/data/raw_data_from_UMA_simulation/353K/all_systems_353K_UMAs1p1/napf6_tgdme/napf6_tgdme.traj",
    },
    # {
    #     "name": "napf6_thf",
    #     "traj_293": "/global/homes/y/yuejian/project/MLFF-distill/m5024/UMA_trajs_293K/napf6_thf.traj",
    #     "traj_323": "/global/homes/y/yuejian/project/MLFF-distill/m4558/323K_500ps_trajs/md_omol_napf6_tetrahydrofuran_pfactor_0.1_1fs_mask_t_re2/md_omol_napf6_tetrahydrofuran_pfactor_0.1_1fs_mask_t_re2.traj",
    # },
    {
        "name": "napf6_pc",
        "traj_293": "/global/homes/y/yuejian/project/MLFF-distill/m5024/UMA_trajs_293K/napf6_pc.traj",
        "traj_323": "/global/homes/y/yuejian/project/MLFF-distill/m4558/323K_500ps_trajs/md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_re1_s1p1/md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_re1_s1p1.traj",
        "traj_353": "/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/data/raw_data_from_UMA_simulation/353K/all_systems_353K_UMAs1p1/napf6_pc/napf6_pc.traj",
    },
    {
        "name": "napf6_ec",
        "traj_293": "/global/homes/y/yuejian/project/MLFF-distill/m5024/UMA_trajs_293K/napf6_ec.traj",
        "traj_323": "/global/homes/y/yuejian/project/MLFF-distill/m4558/323K_500ps_trajs/md_omol_napf6_ethylene_carbonate_pfactor_0.1_1fs_mask_t_re2/md_omol_napf6_ethylene_carbonate_pfactor_0.1_1fs_mask_t_re2.traj",
        "traj_353": "/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/data/raw_data_from_UMA_simulation/353K/all_systems_353K_UMAs1p1/napf6_ec/napf6_ec.traj",
    },
    {
        "name": "napf6_dmc",
        "traj_293": "/global/homes/y/yuejian/project/MLFF-distill/m5024/UMA_trajs_293K/napf6_dmc.traj",
        "traj_323": "/global/homes/y/yuejian/project/MLFF-distill/m4558/323K_500ps_trajs/md_omol_napf6_dimethylcarbonate_pfactor_0.1_1fs_mask_t_re2/md_omol_napf6_dimethylcarbonate_pfactor_0.1_1fs_mask_t_re2.traj",
        "traj_353": "/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/data/raw_data_from_UMA_simulation/353K/all_systems_353K_UMAs1p1/napf6_dmc/napf6_dmc.traj",
    },
    {
        "name": "napf6_deg",
        "traj_293": "/global/homes/y/yuejian/project/MLFF-distill/m5024/UMA_trajs_293K/napf6_deg.traj",
        "traj_323": "/global/homes/y/yuejian/project/MLFF-distill/m4558/323K_500ps_trajs/md_omol_napf6_diethyleneglycol_pfactor_0.1_1fs_mask_t/md_omol_napf6_diethyleneglycol_pfactor_0.1_1fs_mask_t.traj",
        "traj_353": "/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/data/raw_data_from_UMA_simulation/353K/all_systems_353K_UMAs1p1/napf6_deg/napf6_deg.traj",
    },
    {
        "name": "napf6_dme",
        "traj_293": "/global/homes/y/yuejian/project/MLFF-distill/m5024/UMA_trajs_293K/napf6_dme.traj",
        "traj_323": "/global/homes/y/yuejian/project/MLFF-distill/m4558/323K_500ps_trajs/md_omol_napf6_dimethylcarbonate_pfactor_0.1_1fs_mask_t_re2/md_omol_napf6_dimethylcarbonate_pfactor_0.1_1fs_mask_t_re2.traj",
        "traj_353": "/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/data/raw_data_from_UMA_simulation/353K/all_systems_353K_UMAs1p1/napf6_dme/napf6_dme.traj",
    },
    # {
    #     "name": "napf6_diglyme",
    #     "traj_293": "/global/homes/y/yuejian/project/MLFF-distill/m5024/UMA_trajs_293K/napf6_diglyme.traj",
    #     "traj_323": "/global/homes/y/yuejian/project/MLFF-distill/m4558/323K_500ps_trajs/md_omol_napf6_diethyleneglycol_pfactor_0.1_1fs_mask_t/md_omol_napf6_diethyleneglycol_pfactor_0.1_1fs_mask_t.traj",
    # },
]


def run_system(system: Dict[str, str], plot_root: str) -> None:
    """Configure single-run globals and execute analysis."""
    single.TRAJ_293K = system["traj_293"]
    single.TRAJ_323K = system["traj_323"]
    # Optional third temperature (353K). If empty/missing, the single script will
    # fall back to the original 2-temperature behavior.
    if "traj_353" in system and system["traj_353"]:
        single.TRAJ_353K = system["traj_353"]
    else:
        single.TRAJ_353K = None
    single.PLOT_DIR = os.path.join(plot_root, system["name"])
    print(f"\n=== Running {system['name']} ===")
    print(f"293K: {single.TRAJ_293K}")
    print(f"323K: {single.TRAJ_323K}")
    print(f"353K: {getattr(single, 'TRAJ_353K', None)}")
    print(f"Output: {single.PLOT_DIR}")
    single.main()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Batch run RMSD/PCA/clustering analysis across multiple systems.")
    parser.add_argument("--max-systems", type=int, default=None, help="Run only the first N systems (for testing).")
    parser.add_argument("--only", nargs="*", default=None, help="Run only these system names.")
    parser.add_argument("--n-frames", type=int, default=None, help="Override N_FRAMES in the single-system script.")
    parser.add_argument("--stride", type=int, default=None, help="Override SUBSAMPLE_STRIDE in the single-system script.")
    parser.add_argument("--cluster-cutoff", type=float, default=None, help="Override RMSD_CLUSTER_CUTOFF (Å).")
    parser.add_argument("--bins", type=int, default=None, help="Override N_BINS in the single-system script.")
    parser.add_argument("--plot-root", type=str, default=PLOT_ROOT, help="Root output directory.")
    args = parser.parse_args(argv)
    plot_root = args.plot_root

    if args.n_frames is not None:
        single.N_FRAMES = args.n_frames
    if args.stride is not None:
        single.SUBSAMPLE_STRIDE = args.stride
    if args.cluster_cutoff is not None:
        single.RMSD_CLUSTER_CUTOFF = args.cluster_cutoff
    if args.bins is not None:
        single.N_BINS = args.bins

    systems = SYSTEMS
    if args.only:
        only_set = set(args.only)
        systems = [s for s in systems if s["name"] in only_set]
    if args.max_systems is not None:
        systems = systems[: args.max_systems]

    print(f"Will run {len(systems)} system(s). Output root: {plot_root}")
    for sysinfo in systems:
        try:
            run_system(sysinfo, plot_root=plot_root)
        except Exception as exc:
            print(f"[WARN] {sysinfo['name']} failed: {exc}")


if __name__ == "__main__":
    main()
