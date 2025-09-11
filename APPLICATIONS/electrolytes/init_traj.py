#!/usr/bin/env python3
"""
Script to initialize trajectory directories for MD simulations.
Extracts only the index=0 frame from each trajectory file.

Usage: python init_traj.py -w <working_dir> -t <traj_file1> [<traj_file2> ...]
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from ase.io import read, write

def get_base_name(traj_path):
    """Extract base name from trajectory file path (without .traj extension)."""
    return Path(traj_path).stem

def create_traj_directory(working_dir, base_name, traj_path, use_symlink=False):
    """Create directory structure and extract index=0 frame for a trajectory."""
    # Create the subdirectory
    sub_dir = Path(working_dir) / base_name
    sub_dir.mkdir(parents=True, exist_ok=True)

    # Destination paths
    dest_traj = sub_dir / f"{base_name}.traj"
    dest_log = sub_dir / f"{base_name}.log"

    # Read the trajectory file and extract index=0 frame
    try:
        base_structure = read(traj_path, index=0)
        # Write only the first frame to the destination
        write(dest_traj, base_structure)
        print(f"Extracted frame 0 from {traj_path} -> {dest_traj}")
    except Exception as e:
        print(f"Error reading trajectory file {traj_path}: {e}")
        raise

    # Create empty log file
    dest_log.touch()
    print(f"Created empty log file: {dest_log}")

    return sub_dir

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Initialize trajectory directories for MD simulations by extracting index=0 frames.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python init_traj.py -w /path/to/working/dir -t traj1.traj traj2.traj
  python init_traj.py --working-dir /path/to/working/dir --trajectories *.traj
        """
    )
    
    parser.add_argument(
        '-w', '--working-dir',
        type=str,
        required=True,
        help='Working directory where subdirectories will be created'
    )
    
    parser.add_argument(
        '-t', '--trajectories',
        type=str,
        nargs='+',
        required=True,
        help='One or more trajectory files to process'
    )
    
    parser.add_argument(
        '--symlink',
        action='store_true',
        help='Create symbolic links instead of copying trajectory files'
    )
    
    return parser.parse_args()

def main():
    args = parse_arguments()

    working_dir = args.working_dir
    traj_files = args.trajectories
    use_symlink = args.symlink

    # Validate working directory
    working_path = Path(working_dir)
    if not working_path.exists():
        print(f"Warning: Working directory {working_dir} does not exist. Creating it.")
        working_path.mkdir(parents=True, exist_ok=True)

    # Process each trajectory file
    for traj_file in traj_files:
        traj_path = Path(traj_file)

        # Validate trajectory file exists
        if not traj_path.exists():
            print(f"Warning: Trajectory file {traj_file} does not exist. Skipping.")
            continue

        # Get base name and create directory structure
        base_name = get_base_name(traj_path)
        print(f"Processing trajectory: {base_name}")

        try:
            create_traj_directory(working_dir, base_name, traj_path, use_symlink)
        except Exception as e:
            print(f"Error processing {traj_file}: {e}")

    print(f"\nInitialization complete. Processed {len(traj_files)} trajectory files.")

if __name__ == "__main__":
    main()
