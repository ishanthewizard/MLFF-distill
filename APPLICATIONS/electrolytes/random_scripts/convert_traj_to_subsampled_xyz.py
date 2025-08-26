import os
import random
from ase.io import read, write

# =============================================================================
# EDIT THESE PATHS AS NEEDED
# =============================================================================

input_traj = '/data/ishan-amin/OMOL/electrolytes_application/npt_trajs_distillation/npt_trajs_napf6_dme_uma_omol/s1p1/md_omol_re5_small_1p1_wrapped.traj'
output_train_folder = '/data/ishan-amin/OMOL/electrolytes_application/xyz_data/NAPF6/s1p1/train'
output_val_folder  ='/data/ishan-amin/OMOL/electrolytes_application/xyz_data/NAPF6/s1p1/val'
# =============================================================================

def main():
    # Check if input file exists
    if not os.path.exists(input_traj):
        print(f"Error: Input trajectory file {input_traj} does not exist")
        return
    
    # Check if output folders exist
    if not os.path.exists(output_train_folder):
        print(f"Error: Output train folder {output_train_folder} does not exist")
        return
    
    if not os.path.exists(output_val_folder):
        print(f"Error: Output validation folder {output_val_folder} does not exist")
        return
    
    print(f"Reading trajectory from: {input_traj}")
    traj = read(input_traj, index=":")
    
    print(f"Total frames in trajectory: {len(traj)}")
    
    # Subsample every 5 steps
    subsampled_traj = traj[::5]
    print(f"Subsampled frames (every 5 steps): {len(subsampled_traj)}")
    
    # Create list of indices and shuffle for random split
    indices = list(range(len(subsampled_traj)))
    random.shuffle(indices)
    
    # Split into 90% train and 10% validation
    split_point = int(0.9 * len(indices))
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]
    
    print(f"Train frames: {len(train_indices)}")
    print(f"Validation frames: {len(val_indices)}")
    
    # Create train trajectory
    train_traj = [subsampled_traj[idx] for idx in train_indices]
    train_filepath = os.path.join(output_train_folder, "train_frames.xyz")
    print(f"Writing train frames to: {train_filepath}")
    write(train_filepath, train_traj)
    
    # Create validation trajectory
    val_traj = [subsampled_traj[idx] for idx in val_indices]
    val_filepath = os.path.join(output_val_folder, "val_frames.xyz")
    print(f"Writing validation frames to: {val_filepath}")
    write(val_filepath, val_traj)
    
    print("Done!")

if __name__ == "__main__":
    main()