#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MULTI-TRAJECTORY PROCESSING AND COMBINATION SCRIPT
=================================================

This script processes multiple ASE trajectory files (.traj), subsamples them, creates
train/validation splits, and combines all data into single XYZ files for machine learning.

USAGE GUIDE:
-----------
1. CONFIGURE PATHS:
   - Edit the 'input_trajs' list to include all your trajectory file paths
   - Adjust 'output_train_folder' and 'output_val_folder' as needed
   
2. SET PARAMETERS:
   - SUBSAMPLE_STEP: How many frames to skip (e.g., 5 = every 5th frame)
   - TRAIN_RATIO: Fraction for training (0.9 = 90% train, 10% validation)
   
3. RUN THE SCRIPT:
   python convert_trajs_to_subsampled_xyz.py

4. OUTPUT:
   - combined_train_frames.xyz: All training frames from all trajectories
   - combined_val_frames.xyz: All validation frames from all trajectories

EXAMPLE:
--------
If you have trajectories from different solvents:
input_trajs = [
    '/path/to/napf6_dme_1ns.traj',
    '/path/to/napf6_dmc_1ns.traj', 
    '/path/to/napf6_pc_1ns.traj',
]

The script will process each, subsample, split into train/val, and combine all data.

REQUIREMENTS:
------------
- ASE (Atomic Simulation Environment)
- Python 3.6+
- Sufficient disk space for output files

Author: Generated script for MLFF data preparation
Date: 2024
"""

import os
import random
from ase.io import read, write
from pathlib import Path

# =============================================================================
# EDIT THESE PATHS AS NEEDED
# =============================================================================

# List of input trajectories to process
# input_trajs_1 = [
#     '/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolytes/first_100ps_traj_to_send/md_omol_naotf_diglyme_1m_s1p1_re1.traj',
#     # Add more trajectory paths here as needed
#     '/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolytes/first_100ps_traj_to_send/md_omol_naotf_dme_s1p1_omol.traj',
#     '/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolytes/first_100ps_traj_to_send/md_omol_naotf_pc_1m_s1p1_re1.traj',
#     '/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolytes/first_100ps_traj_to_send/md_omol_naotf_tgdme_1m_s1p1_re2.traj',
#     '/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolytes/first_100ps_traj_to_send/md_omol_naotf_tgdme_1m_s1p1.traj',
# ]

# input_trajs = [
#     '/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolytes/first_100ps_traj_to_send/md_omol_napf6_diethyleneglycol_pfactor_0.1_1fs_mask_t.traj',
#     '/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolytes/first_100ps_traj_to_send/md_omol_napf6_dimethylcarbonate_pfactor_0.1_1fs_mask_t_re2.traj',
#     '/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolytes/first_100ps_traj_to_send/md_omol_napf6_ethylene_carbonate_pfactor_0.1_1fs_mask_t_re2.traj',
#     '/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolytes/first_100ps_traj_to_send/md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_re1_s1p1.traj',
#     '/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolytes/first_100ps_traj_to_send/md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_re3_s1p1.traj',
#     '/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolytes/first_100ps_traj_to_send/md_omol_napf6_tetrahydrofuran_pfactor_0.1_1fs_mask_t_re2.traj',
#     '/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolytes/first_100ps_traj_to_send/md_omol_napf6_tgdme_1m_s1p1_re2.traj',
#     '/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolytes/first_100ps_traj_to_send/md_omol_napf6_tgdme_1m_s1p1.traj'
# ]

# input_trajs = [
#     '/global/homes/y/yuejian/project/MLFF-distill/m4558/distillation_project/UMA_simulate_data_checkpoints/Sep_18/md_omol_naotf_diglyme_1m_s1p1/md_omol_naotf_diglyme_1m_s1p1.traj',
#     '/global/homes/y/yuejian/project/MLFF-distill/m4558/distillation_project/UMA_simulate_data_checkpoints/Sep_18/md_omol_naotf_dme_s1p1_omol/md_omol_naotf_dme_s1p1_omol.traj',
#     '/global/homes/y/yuejian/project/MLFF-distill/m4558/distillation_project/UMA_simulate_data_checkpoints/Sep_18/md_omol_naotf_pc_1m_s1p1/md_omol_naotf_pc_1m_s1p1.traj',
#     '/global/homes/y/yuejian/project/MLFF-distill/m4558/distillation_project/UMA_simulate_data_checkpoints/Sep_18/md_omol_naotf_tgdme_1m_s1p1/md_omol_naotf_tgdme_1m_s1p1.traj'
# ]


input_trajs = [
    '/global/homes/y/yuejian/project/MLFF-distill/m5024/UMA_trajs_293K/naotf_diglyme.traj',
    '/global/homes/y/yuejian/project/MLFF-distill/m5024/UMA_trajs_293K/naotf_dme.traj',
    '/global/homes/y/yuejian/project/MLFF-distill/m5024/UMA_trajs_293K/naotf_pc.traj',
    '/global/homes/y/yuejian/project/MLFF-distill/m5024/UMA_trajs_293K/naotf_tgdme.traj'
]

# input_trajs = input_trajs_3 + input_trajs_2 + input_trajs_1

output_train_folder = '/global/homes/y/yuejian/project/MLFF-distill/m4558/distillation_project/new_toy_test/Sep_18_naotf_per_salt_data/train'
output_val_folder  = '/global/homes/y/yuejian/project/MLFF-distill/m4558/distillation_project/new_toy_test/Sep_18_naotf_per_salt_data/val'

# Subsample every N steps (adjust as needed)
SUBSAMPLE_STEP = 5

# Train/validation split ratio (0.9 = 90% train, 10% validation)
TRAIN_RATIO = 0.9

# =============================================================================

def process_single_trajectory(traj_path, subsample_step=5):
    """
    Process a single trajectory file.
    
    Args:
        traj_path (str): Path to the trajectory file
        subsample_step (int): Subsample every N steps
        
    Returns:
        tuple: (subsampled_frames, original_frame_count, subsampled_frame_count)
    """
    print(f"Reading trajectory from: {traj_path}")
    
    if not os.path.exists(traj_path):
        print(f"Warning: Trajectory file {traj_path} does not exist, skipping...")
        return None, 0, 0
    
    try:
        traj = read(traj_path, index=":")
        original_count = len(traj)
        print(f"  Total frames in trajectory: {original_count}")
        
        # Subsample
        subsampled_traj = traj[::subsample_step]
        subsampled_count = len(subsampled_traj)
        print(f"  Subsampled frames (every {subsample_step} steps): {subsampled_count}")
        
        return subsampled_traj, original_count, subsampled_count
        
    except Exception as e:
        print(f"Error reading trajectory {traj_path}: {e}")
        return None, 0, 0

def split_trajectory(frames, train_ratio=0.9):
    """
    Split frames into train and validation sets.
    
    Args:
        frames: List of ASE atoms objects
        train_ratio (float): Fraction of frames to use for training
        
    Returns:
        tuple: (train_frames, val_frames)
    """
    if not frames:
        return [], []
    
    # Create list of indices and shuffle for random split
    indices = list(range(len(frames)))
    random.shuffle(indices)
    
    # Split into train and validation
    split_point = int(train_ratio * len(indices))
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]
    
    # Extract frames
    train_frames = [frames[idx] for idx in train_indices]
    val_frames = [frames[idx] for idx in val_indices]
    
    return train_frames, val_frames

def main():
    # Check if output folders exist, create if they don't
    Path(output_train_folder).mkdir(parents=True, exist_ok=True)
    Path(output_val_folder).mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("MULTI-TRAJECTORY PROCESSING AND COMBINATION")
    print("=" * 60)
    
    all_train_frames = []
    all_val_frames = []
    total_original_frames = 0
    total_subsampled_frames = 0
    
    # Process each trajectory
    for i, traj_path in enumerate(input_trajs):
        print(f"\nProcessing trajectory {i+1}/{len(input_trajs)}:")
        print("-" * 40)
        
        # Process the trajectory
        subsampled_traj, orig_count, subsampled_count = process_single_trajectory(
            traj_path, SUBSAMPLE_STEP
        )
        
        if subsampled_traj is None:
            continue
            
        total_original_frames += orig_count
        total_subsampled_frames += subsampled_count
        
        # Split into train/val
        train_frames, val_frames = split_trajectory(subsampled_traj, TRAIN_RATIO)
        
        print(f"  Train frames: {len(train_frames)}")
        print(f"  Validation frames: {len(val_frames)}")
        
        # Add to our collection
        all_train_frames.extend(train_frames)
        all_val_frames.extend(val_frames)
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total trajectories processed: {len(input_trajs)}")
    print(f"Total original frames: {total_original_frames}")
    print(f"Total subsampled frames: {total_subsampled_frames}")
    print(f"Combined train frames: {len(all_train_frames)}")
    print(f"Combined validation frames: {len(all_val_frames)}")
    
    # Write combined datasets
    if all_train_frames:
        train_filepath = os.path.join(output_train_folder, "combined_train_frames.xyz")
        print(f"\nWriting combined train frames to: {train_filepath}")
        write(train_filepath, all_train_frames)
        print(f"Successfully wrote {len(all_train_frames)} train frames")
    
    if all_val_frames:
        val_filepath = os.path.join(output_val_folder, "combined_val_frames.xyz")
        print(f"Writing combined validation frames to: {val_filepath}")
        write(val_filepath, all_val_frames)
        print(f"Successfully wrote {len(all_val_frames)} validation frames")
    
    print("\nDone! All trajectories processed and combined.")

if __name__ == "__main__":
    main()