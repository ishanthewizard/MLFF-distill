#!/usr/bin/env python3
"""
Script to create a SLURM job array submission script for incomplete DFT calculations.
This ensures we only submit jobs for ligands that haven't been completed yet.
"""

import os
import sys
from pathlib import Path
import argparse


def find_incomplete_calculations(root_dir: Path) -> list:
    """
    Find all ligand directories that haven't been completed successfully.
    
    Args:
        root_dir: Root directory containing ligand subdirectories
        
    Returns:
        List of ligand names that need processing
    """
    incomplete_ligands = []
    
    # Get all subdirectories (ligands)
    ligand_dirs = [d for d in root_dir.iterdir() if d.is_dir()]
    
    for ligand_dir in ligand_dirs:
        ligand_name = ligand_dir.name
        dft_dir = ligand_dir / "DFT"
        success_file = dft_dir / "success.txt"
        
        # Check if calculation is incomplete
        if not success_file.exists():
            incomplete_ligands.append(ligand_name)
        else:
            print(f"Skipping {ligand_name}: already completed")
    
    return incomplete_ligands


def create_job_array_script(incomplete_ligands: list, output_file: str, 
                          root_dir: str, orca_path: str) -> None:
    """
    Create a SLURM job array submission script.
    
    Args:
        incomplete_ligands: List of ligand names to process
        output_file: Path to output script file
        root_dir: Root directory path
        orca_path: ORCA executable path
    """
    
    # Create the job array script content
    script_content = f"""#!/bin/bash

# SLURM Job Array for DFT calculations - Auto-generated
# Processing {len(incomplete_ligands)} incomplete calculations

#SBATCH -q regular
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 2:00:00
#SBATCH -C cpu
#SBATCH --account=m5024
#SBATCH --job-name=distillation_array
#SBATCH --output=/global/homes/y/yuejian/project/MLFF-distill/yuejian/DFT_output/%A_%a_log.out
#SBATCH --error=/global/homes/y/yuejian/project/MLFF-distill/yuejian/DFT_output/%A_%a_log.err
#SBATCH --array=0-{len(incomplete_ligands)-1}

# Set environment variables
export OMP_NUM_THREADS=1
export ORA_NUM_PROCS=1
export ORA_NUM_THREADS=1

# Create log directory
mkdir -p /global/homes/y/yuejian/project/MLFF-distill/yuejian/DFT_output

# Set paths
ROOT_DIR="{root_dir}"
ORCA_PATH="{orca_path}"

# Change to project directory
cd /global/homes/y/yuejian/project/MLFF-distill

# Array of incomplete ligands
ligands=(
{chr(10).join(f'    "{ligand}"' for ligand in incomplete_ligands)}
)

# Get the ligand for this array job
array_index=$SLURM_ARRAY_TASK_ID
if [ $array_index -lt ${{#ligands[@]}} ]; then
    ligand_name="${{ligands[$array_index]}}"
    ligand_dir="$ROOT_DIR/$ligand_name"
    dft_dir="$ligand_dir/DFT"
    
    echo "Processing ligand $array_index: $ligand_name"
    
    # Double-check if already completed (in case of race conditions)
    if [ -f "$dft_dir/success.txt" ]; then
        echo "Ligand $ligand_name already completed successfully - skipping"
        exit 0
    fi
    
    # Run single ORCA calculation
    python src_v2/DFT_calculation/calculate_single_point_energy.py "$ROOT_DIR" --orca-path "$ORCA_PATH" --ligand "$ligand_name"
else
    echo "Array index $array_index exceeds number of ligands (${{#ligands[@]}})"
fi

echo "Job $SLURM_ARRAY_TASK_ID completed at $(date)"
"""
    
    # Write the script
    with open(output_file, 'w') as f:
        f.write(script_content)
    
    print(f"Created job array script: {output_file}")
    print(f"Number of incomplete calculations: {len(incomplete_ligands)}")
    print(f"Array range: 0-{len(incomplete_ligands)-1}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Create SLURM job array script for incomplete DFT calculations"
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="Root directory containing ligand subdirectories"
    )
    parser.add_argument(
        "--orca-path",
        type=str,
        default="/global/homes/y/yuejian/project/MLFF-distill/yuejian/orca/orca-6.1.0-f.0_linux_x86-64/bin/orca",
        help="Path to ORCA executable"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submit_dft_array_dynamic.sh",
        help="Output script filename"
    )
    
    args = parser.parse_args()
    
    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        print(f"Error: Directory {root_dir} does not exist.")
        sys.exit(1)
    
    # Find incomplete calculations
    print(f"Scanning {root_dir} for incomplete calculations...")
    incomplete_ligands = find_incomplete_calculations(root_dir)
    
    if not incomplete_ligands:
        print("All calculations are already completed!")
        sys.exit(0)
    
    # Create job array script
    create_job_array_script(
        incomplete_ligands, 
        args.output, 
        str(root_dir.absolute()), 
        args.orca_path
    )
    
    print(f"\nTo submit the job array:")
    print(f"sbatch {args.output}")


if __name__ == "__main__":
    main() 