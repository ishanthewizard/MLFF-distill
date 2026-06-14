#!/usr/bin/env python3
"""Create first 50ps dataset for napf6/dme systems only (413K).

USAGE
=====
1. Adjust the global variables below (output_root, SUBSAMPLE_STEP,
   WINDOW_SIZE, TRAIN_RATIO, NUM_WORKERS) if needed.
2. Run the script:
       python first_50ps_per_salt_napf6_413K.py
3. The script will:
   - Subsample ASE .traj files from 413K (first WINDOW_SIZE frames per traj)
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
output_root = '/pscratch/sd/y/yuejian/baseline/napf6_first_50ps_413K'
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

WINDOW_SIZE = 5000
# NOTE: Max frames to read per trajectory; 500 = 5 ps at 10 fs timestep (5000 = 50 ps)

TRAIN_RATIO = 0.9
# NOTE: Fraction of subsampled frames for training (0.9 = 90% train, 10% validation)

NUM_WORKERS = 8
# NOTE: Parallel workers for launch_processing and compute_normalizer_and_linear_reference

# -----------------------------------------------------------------------------
# Input trajectories
# -----------------------------------------------------------------------------
# napf6/dme at 3 concentrations from 413K simulations
input_trajs = [
    "/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/data/raw_data_from_UMA_simulation/413K/1M/napf6_dme/napf6_dme.traj",
    "/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/data/raw_data_from_UMA_simulation/413K/0.5M/md_omol_napf6_dme_re1/md_omol_napf6_dme_re1.traj",
    "/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/data/raw_data_from_UMA_simulation/413K/0.1M/md_omol_napf6_dme_re1/md_omol_napf6_dme_re1.traj",
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

