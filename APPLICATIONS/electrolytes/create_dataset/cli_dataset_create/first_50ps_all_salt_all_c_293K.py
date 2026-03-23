#!/usr/bin/env python3
"""Create first 500ps dataset for all 15 salt systems (293K).

USAGE
=====
1. Adjust the global variables below (output_root, RAW_DATA_ROOT, SUBSAMPLE_STEP,
   WINDOW_SIZE, TRAIN_RATIO, NUM_WORKERS) if needed.
2. Run the script:
       python first_500ps_all_salt_293K.py
3. The script will:
   - Subsample ASE .traj files from UMA_trajs_293K (first WINDOW_SIZE frames per traj)
   - Split frames into train/val and write combined XYZ files
   - Convert XYZ to ASE LMDB (data.*.aselmdb)
   - Remove intermediate XYZ files
   - Compute and save species_refs.yaml and force_rms.txt

OUTPUT
======
- output_root/train/  : combined_train_frames.xyz -> data.*.aselmdb
- output_root/val/    : combined_val_frames.xyz -> data.*.aselmdb
- output_root/species_refs.yaml, force_rms.txt
"""

import glob
import os
import sys
import yaml
from pathlib import Path

# Add create_dataset root so we can import from utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.convert_trajs_to_subsampled_xyz_core import main
from utils.create import launch_processing
from utils.compute_ref import compute_normalizer_and_linear_reference


# -----------------------------------------------------------------------------
# Output paths
# -----------------------------------------------------------------------------
output_root = '/u/yjian1/project/MLFF-distill/yjian1/MLFF-distill/lmdb'
# NOTE: Root directory for train/val LMDB shards and species_refs.yaml, force_rms.txt

output_train_folder = os.path.join(output_root, 'train')
# NOTE: Subfolder for training LMDB shards (combined_train_frames.xyz -> data.*.aselmdb)

output_val_folder = os.path.join(output_root, 'val')
# NOTE: Subfolder for validation LMDB shards (combined_val_frames.xyz -> data.*.aselmdb)

# -----------------------------------------------------------------------------
# Processing parameters
# -----------------------------------------------------------------------------
SUBSAMPLE_STEP = 5
# NOTE: Subsample every N frames (5 = every 5th frame) to reduce dataset size

WINDOW_SIZE = 50
# NOTE: Max frames to read per trajectory; 500 = 5 ps at 10 fs timestep (5000 = 50 ps)

TRAIN_RATIO = 0.9
# NOTE: Fraction of subsampled frames for training (0.9 = 90% train, 10% validation)

NUM_WORKERS = 8
# NOTE: Parallel workers for launch_processing and compute_normalizer_and_linear_reference

# -----------------------------------------------------------------------------
# Input trajectories
# -----------------------------------------------------------------------------
# NOTE: Base path for UMA 293K datasets
_BASE = '/u/yjian1/project/MLFF-distill/yjian1/MLFF-distill/dataset/UMA_293K/293K'
_ROOT_1M = f'{_BASE}/UMA_trajs_293K_1M_1ns'
_ROOT_0_5M = f'{_BASE}/UMA_trajs_293K_0_5M_100ps'
_ROOT_0_1M = f'{_BASE}/UMA_trajs_293K_0_1M_100ps'

# First sub trajs: direct .traj files under UMA_trajs_293K_1M_1ns
# Second sub trajs: .traj files in subdirs under UMA_trajs_293K_0_5M_100ps and UMA_trajs_293K_0_1M_100ps
input_trajs = [
    # --- First sub trajs (1M, 1ns) ---
    f'{_ROOT_1M}/cspf6_dme.traj',
    f'{_ROOT_1M}/lipf6_dme.traj',
    f'{_ROOT_1M}/naotf_dme.traj',
    f'{_ROOT_1M}/naotf_diglyme.traj',
    f'{_ROOT_1M}/naotf_pc.traj',
    f'{_ROOT_1M}/naotf_tgdme.traj',
    f'{_ROOT_1M}/napf6_deg.traj',
    f'{_ROOT_1M}/napf6_diglyme.traj',
    f'{_ROOT_1M}/napf6_dmc.traj',
    f'{_ROOT_1M}/napf6_dme.traj',
    f'{_ROOT_1M}/napf6_ec.traj',
    f'{_ROOT_1M}/napf6_pc.traj',
    f'{_ROOT_1M}/napf6_tgdme.traj',
    f'{_ROOT_1M}/napf6_thf.traj',
    f'{_ROOT_1M}/natfsi_dme.traj',
    # --- Second sub trajs (0.5M, 100ps) ---
    f'{_ROOT_0_5M}/md_omol_napf6_dme_re1/md_omol_napf6_dme_re1.traj',
    f'{_ROOT_0_5M}/md_omol_lipf6_pfactor_0.1_1fs_mask_t/md_omol_lipf6_pfactor_0.1_1fs_mask_t.traj',
    # --- Second sub trajs (0.1M, 100ps) ---
    f'{_ROOT_0_1M}/md_omol_napf6_dme_re1/md_omol_napf6_dme_re1.traj',
    f'{_ROOT_0_1M}/md_omol_naotf_dme_s1p1_omol/md_omol_naotf_dme_s1p1_omol.traj',
    f'{_ROOT_0_1M}/md_omol_naotf_pc_1m_s1p1/md_omol_naotf_pc_1m_s1p1.traj',
    f'{_ROOT_0_1M}/md_omol_napf6_diglyme_pfactor_0.1_1fs/md_omol_napf6_diglyme_pfactor_0.1_1fs.traj',
    f'{_ROOT_0_1M}/md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1/md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1.traj',
    f'{_ROOT_0_1M}/md_omol_naotf_tgdme_1m_s1p1/md_omol_naotf_tgdme_1m_s1p1.traj',
    f'{_ROOT_0_1M}/md_omol_naotf_diglyme_1m_s1p1/md_omol_naotf_diglyme_1m_s1p1.traj',
]

if __name__ == "__main__":
    main(
        input_trajs=input_trajs,
        output_train_folder=output_train_folder,
        output_val_folder=output_val_folder,
        subsample_step=SUBSAMPLE_STEP,
        window_size=WINDOW_SIZE,
        train_ratio=TRAIN_RATIO,
    )

    # XYZ to ASE LMDB (train and val)
    launch_processing(
        output_train_folder, Path(output_train_folder), NUM_WORKERS
    )
    launch_processing(output_val_folder, Path(output_val_folder), NUM_WORKERS)

    # Delete the xyz files in train and val folder
    for folder in (output_train_folder, output_val_folder):
        for xyz_file in glob.glob(os.path.join(folder, "*.xyz")):
            os.remove(xyz_file)
            print(f"Removed {xyz_file}")

    # Calculate normalizer and energy ref (species_refs.yaml, force_rms.txt)
    force_rms, linref_coeff = compute_normalizer_and_linear_reference(
        output_train_folder, NUM_WORKERS
    )
    parent_dir = Path(output_root)
    with open(parent_dir / "species_refs.yaml", "w") as f:
        yaml.dump({"omol_element_refs": linref_coeff}, f, default_flow_style=False)
    with open(parent_dir / "force_rms.txt", "w") as f:
        f.write(f"{force_rms}\n")
    print(f"Saved species_refs.yaml and force_rms.txt to {parent_dir}")

