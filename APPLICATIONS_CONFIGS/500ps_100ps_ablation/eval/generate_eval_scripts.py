#!/usr/bin/env python3
"""Generate eval job scripts for all 500ps and 100ps trajectories."""

import os
import re
from pathlib import Path

BASE = Path("/u/yjian1/project/MLFF-distill")
EVAL_DIR = BASE / "APPLICATIONS_CONFIG/500ps_100ps_ablation/eval"

# (trajectory_dir, ablation, cation, anion, solvent, concentration, temperature)
# Built from parsing the MD scripts - traj_path = dir/basename(dir).traj
TARGETS = [
    # 500ps
    (
        "500ps_same_data/initial_box/20ns_solvent_0_1M/md_omol_naotf_dme_s1p1_omol",
        "500ps",
        "Na", "OTf", "DME", "0_1M", "298_2K",
    ),
    (
        "500ps_same_data/initial_box/20ns_solvent_0_1M/md_omol_napf6_dme_re1",
        "500ps",
        "Na", "PF6", "DME", "0_1M", "298_2K",
    ),
    (
        "500ps_same_data/initial_box/20ns_solvent_solute_0.5M/298_2K/md_omol_napf6_dme_re1",
        "500ps",
        "Na", "PF6", "DME", "0_5M", "298_2K",
    ),
    (
        "500ps_same_data/initial_box/20ns_solvent_solute_0.5M/323_2K/md_omol_lipf6_pfactor_0.1_1fs_mask_t",
        "500ps",
        "Li", "PF6", "DME", "0_5M", "323_2K",
    ),
    (
        "500ps_same_data/initial_box/20ns_solvent_solute_0.5M/323_2K/md_omol_napf6_dme_re1",
        "500ps",
        "Na", "PF6", "DME", "0_5M", "323_2K",
    ),
    # 100ps
    (
        "100ps_double_data/initial_box/20ns_solute_solvent_1M/md_omol_napf6_diglyme_pfactor_0.1_1fs",
        "100ps",
        "Na", "PF6", "Diglyme", "1M", "298_2K",
    ),
    (
        "100ps_double_data/initial_box/20ns_solute_solvent_1M/md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1",
        "100ps",
        "Na", "PF6", "PC", "1M", "298_2K",
    ),
    (
        "100ps_double_data/initial_box/20ns_solute_solvent_1M/naotf_diglyme",
        "100ps",
        "Na", "OTf", "Diglyme", "1M", "298_2K",
    ),
    (
        "100ps_double_data/initial_box/20ns_solute_solvent_1M/naotf_dme",
        "100ps",
        "Na", "OTf", "DME", "1M", "298_2K",
    ),
    (
        "100ps_double_data/initial_box/20ns_solute_solvent_1M/napf6_dme",
        "100ps",
        "Na", "PF6", "DME", "1M", "298_2K",
    ),
    (
        "100ps_double_data/initial_box/20ns_solvent_0_1M/md_omol_naotf_diglyme_1m_s1p1",
        "100ps",
        "Na", "OTf", "Diglyme", "0_1M", "298_2K",
    ),
    (
        "100ps_double_data/initial_box/20ns_solvent_0_1M/md_omol_naotf_dme_s1p1_omol",
        "100ps",
        "Na", "OTf", "DME", "0_1M", "298_2K",
    ),
    (
        "100ps_double_data/initial_box/20ns_solvent_0_1M/md_omol_naotf_pc_1m_s1p1",
        "100ps",
        "Na", "OTf", "PC", "0_1M", "298_2K",
    ),
    (
        "100ps_double_data/initial_box/20ns_solvent_0_1M/md_omol_naotf_tgdme_1m_s1p1",
        "100ps",
        "Na", "OTf", "TGDME", "0_1M", "298_2K",
    ),
    (
        "100ps_double_data/initial_box/20ns_solvent_0_1M/md_omol_napf6_diglyme_pfactor_0.1_1fs",
        "100ps",
        "Na", "PF6", "Diglyme", "0_1M", "298_2K",
    ),
    (
        "100ps_double_data/initial_box/20ns_solvent_0_1M/md_omol_napf6_dme_re1",
        "100ps",
        "Na", "PF6", "DME", "0_1M", "298_2K",
    ),
    (
        "100ps_double_data/initial_box/20ns_solvent_0_1M/md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1",
        "100ps",
        "Na", "PF6", "PC", "0_1M", "298_2K",
    ),
    (
        "100ps_double_data/initial_box/20ns_solvent_solute_0.5M/273_2K/md_omol_lipf6_pfactor_0.1_1fs_mask_t",
        "100ps",
        "Li", "PF6", "DME", "0_5M", "273_2K",
    ),
    (
        "100ps_double_data/initial_box/20ns_solvent_solute_0.5M/273_2K/md_omol_napf6_dme_re1",
        "100ps",
        "Na", "PF6", "DME", "0_5M", "273_2K",
    ),
    (
        "100ps_double_data/initial_box/20ns_solvent_solute_0.5M/298_2K/md_omol_lipf6_pfactor_0.1_1fs_mask_t",
        "100ps",
        "Li", "PF6", "DME", "0_5M", "298_2K",
    ),
    (
        "100ps_double_data/initial_box/20ns_solvent_solute_0.5M/298_2K/md_omol_napf6_dme_re1",
        "100ps",
        "Na", "PF6", "DME", "0_5M", "298_2K",
    ),
    (
        "100ps_double_data/initial_box/20ns_solvent_solute_0.5M/323_2K/md_omol_lipf6_pfactor_0.1_1fs_mask_t",
        "100ps",
        "Li", "PF6", "DME", "0_5M", "323_2K",
    ),
    (
        "100ps_double_data/initial_box/20ns_solvent_solute_0.5M/323_2K/md_omol_napf6_dme_re1",
        "100ps",
        "Na", "PF6", "DME", "0_5M", "323_2K",
    ),
]

TRAJ_BASE = BASE / "yjian1/electrolyte_application/ablate_temperature_whole_exp/outlier_and_all_sys_ablation_Mar_13_2026"

TEMPLATE = '''#!/bin/bash
# MSD Convergence Calculation Script
#
# Usage:
#   sbatch {fname}
#
# Note: Output directory is configured in the script (see OUT_DIR variable below)
#
# The TARGETS configuration is defined in the MSD_TARGETS_JSON variable below.
# Modify it to change which trajectories are analyzed.

# === SLURM Job Parameters (Delta) ===
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=ghx4
#SBATCH --time=1:00:00
#SBATCH --job-name=eval_msd
#SBATCH --account=bfoy-dtai-gh
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest

#SBATCH --output=/u/yjian1/project/MLFF-distill/yjian1/log/md_flex_%x_%j_%Y%m%d_%H%M%S.out
#SBATCH --error=/u/yjian1/project/MLFF-distill/yjian1/log/md_flex_%x_%j_%Y%m%d_%H%M%S.err

set -euo pipefail

# Make sure log dir exists (for Slurm --output/--error paths)
mkdir -p /u/yjian1/project/MLFF-distill/yjian1/log

# === Change to Working Directory ===
cd /u/yjian1/project/MLFF-distill

# Set matplotlib backend to non-interactive for batch jobs
export MPLBACKEND=Agg

# === Single target ===
MSD_TARGETS_JSON='[
    ["{traj_path}", "{label}", "{cation}", "{anion}", "{solvent}", "{conc}", "{temp}"]
]'

export MSD_TARGETS_JSON

# Output directory (isolated per system to allow parallel runs)
OUT_DIR="/u/yjian1/project/MLFF-distill/yjian1/electrolyte_application/ablate_temperature_whole_exp/outlier_and_all_sys_ablation_Mar_13_2026/eval"

# Set tau_max_fit_ps to 20000 ps (20 ns)
TAU_MAX_FIT_PS=8000
# Trajectory base timestep in picoseconds (100 fs)
KNOWN_DT_PS=0.1

# Avoid CPU oversubscription if SLURM sets cpus-per-task
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"

# Run the MSD calculation batch script
python /u/yjian1/project/MLFF-distill/APPLICATIONS/electrolytes/observable_scripts/mean_square_displacement/msds_calculation_batch.py \\
    --out-dir "$OUT_DIR" \\
    --tau-max-fit-ps "$TAU_MAX_FIT_PS" \\
    --known-dt-ps "$KNOWN_DT_PS"
'''


def sanitize_job_name(s: str) -> str:
    """Slurm job names must be <= 64 chars; replace problematic chars."""
    s = re.sub(r'[^\w\-]', '_', s)
    return s[:60]


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    for rel_path, ablation, cation, anion, solvent, conc, temp in TARGETS:
        traj_dir = TRAJ_BASE / rel_path
        sys_name = traj_dir.name
        traj_path = traj_dir / f"{sys_name}.traj"
        label = f"{sys_name}_{temp}"
        job_suffix = sanitize_job_name(f"{ablation}_{conc}_{sys_name}_{temp}")
        fname = f"msd_{job_suffix}.sh"

        content = TEMPLATE.format(
            fname=fname,
            label=label,
            traj_path=str(traj_path),
            cation=cation,
            anion=anion,
            solvent=solvent,
            conc=conc,
            temp=temp,
        )
        out_path = EVAL_DIR / fname
        out_path.write_text(content)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
