#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract first 2,000,000 frames from trajectories and save with 2ns suffix.
"""

from ase.io import read, write
from pathlib import Path
from tqdm import tqdm

# Input trajectories
input_trajs = [
    "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/ablate_distillation/10ns_intermediate/md_omol_naotf_dme_s1p1_omol_10/md_omol_naotf_dme_s1p1_omol_10.traj",
    "/home/yuejian/project/MLFF-distill/OMOL/electrolytes_application/ablate_distillation/10ns_intermediate/md_omol_naotf_dme_s1p1_omol_undistill/md_omol_naotf_dme_s1p1_omol_undistill.traj"
]

# Output directory
output_dir = Path("/home/yuejian/project/MLFF-distill/yuejian/distill_ablation_2nd")
output_dir.mkdir(parents=True, exist_ok=True)

# Number of frames to extract
n_frames = 2000000

# Process each trajectory
for input_traj in input_trajs:
    print(f"\n{'='*80}")
    print(f"Processing: {input_traj}")
    print(f"{'='*80}")
    
    # Read trajectory
    print("Reading trajectory...")
    frames = read(input_traj, index=f":{n_frames}")
    
    total_frames = len(frames)
    print(f"Extracted {total_frames:,} frames")
    
    # Generate output filename
    input_path = Path(input_traj)
    # Extract the base name (e.g., md_omol_naotf_dme_s1p1_omol_10)
    base_name = input_path.stem
    # Add _2ns suffix
    output_name = f"{base_name}_2ns.traj"
    output_path = output_dir / output_name
    
    print(f"Saving to: {output_path}")
    
    # Write trajectory
    write(str(output_path), frames)
    
    print(f"✓ Successfully saved {total_frames:,} frames to {output_path}")

print(f"\n{'='*80}")
print("All trajectories processed!")
print(f"{'='*80}")

