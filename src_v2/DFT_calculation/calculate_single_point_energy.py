#!/usr/bin/env python3
"""
Script to run DFT single-point energy calculations using ORCA.
Processes all ligands in a directory structure and manages calculation status.
"""

import os
import sys
import argparse
import subprocess
import signal
from pathlib import Path
from typing import List, Tuple, Optional
import time


def run_orca_calculation(orca_path: str, input_file: Path, output_dir: Path) -> Tuple[bool, str]:
    """
    Run ORCA calculation for a single ligand.
    
    Args:
        orca_path: Path to ORCA executable
        input_file: Path to ORCA input file
        output_dir: Directory to save output files
        
    Returns:
        Tuple of (success: bool, error_message: str)
    """
    try:
        # Prepare command with output redirection using absolute paths
        output_file = input_file.stem + ".out"
        input_abs_path = str(input_file.absolute())
        output_abs_path = str((input_file.parent.parent / output_file).absolute())
        cmd = f"{orca_path} {input_abs_path} > {output_abs_path}"
        # Run ORCA calculation with timeout (e.g., 24 hours)
        timeout_seconds = 24 * 3600  # 24 hours
        
        print(f"Running ORCA calculation: {cmd}")
        print(f"Output directory: {output_dir}")
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            text=True
        )
        
        # Check if calculation was successful
        if result.returncode == 0:
            # Check for ORCA-specific success indicators
            output_content = result.stdout
            if "ORCA TERMINATED NORMALLY" in output_content:
                return True, ""
            else:
                return False, "ORCA did not terminate normally"
        else:
            return False, f"ORCA failed with return code {result.returncode}"
            
    except subprocess.TimeoutExpired:
        return False, "Calculation timed out"
    except Exception as e:
        return False, f"Exception occurred: {str(e)}"


def check_calculation_status(dft_dir: Path) -> Optional[str]:
    """
    Check if a calculation has already been completed or failed.
    
    Args:
        dft_dir: DFT directory for the ligand
        
    Returns:
        'success', 'failure', or None (not run yet)
    """
    success_file = dft_dir / "success.txt"
    failure_file = dft_dir / "failure.txt"
    
    if success_file.exists():
        return "success"
    elif failure_file.exists():
        return "failure"
    else:
        return None


def mark_calculation_status(dft_dir: Path, status: str, message: str = ""):
    """
    Mark calculation status with success or failure file.
    
    Args:
        dft_dir: DFT directory for the ligand
        status: 'success' or 'failure'
        message: Optional error message for failure
    """
    if status == "success":
        success_file = dft_dir / "success.txt"
        with open(success_file, 'w') as f:
            f.write(f"Calculation completed successfully at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Remove failure file if it exists
        failure_file = dft_dir / "failure.txt"
        if failure_file.exists():
            failure_file.unlink()
            
    elif status == "failure":
        failure_file = dft_dir / "failure.txt"
        with open(failure_file, 'w') as f:
            f.write(f"Calculation failed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            if message:
                f.write(f"Error: {message}\n")


def process_ligand(orca_path: str, ligand_dir: Path) -> Tuple[bool, str]:
    """
    Process a single ligand directory.
    
    Args:
        orca_path: Path to ORCA executable
        ligand_dir: Directory containing the ligand
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    ligand_name = ligand_dir.name
    dft_dir = ligand_dir / "DFT"
    input_dir = dft_dir / "input"
    
    # Check if DFT directory structure exists
    if not dft_dir.exists():
        return False, f"DFT directory not found: {dft_dir}"
    
    if not input_dir.exists():
        return False, f"Input directory not found: {input_dir}"
    
    # Find input file
    input_files = list(input_dir.glob("*.inp"))
    if not input_files:
        return False, f"No .inp files found in {input_dir}"
    
    input_file = input_files[0]  # Take the first .inp file
    
    # Check current status
    status = check_calculation_status(dft_dir)
    if status == "success":
        print(f"  Skipping {ligand_name}: already completed successfully")
        return True, "Already completed"
    elif status == "failure":
        print(f"  Retrying {ligand_name}: previous attempt failed")
    
    # Run calculation
    print(f"  Processing {ligand_name}...")
    success, error_msg = run_orca_calculation(orca_path, input_file, dft_dir)
    
    # Mark status
    if success:
        mark_calculation_status(dft_dir, "success")
        print(f"  ✓ {ligand_name}: calculation completed successfully")
    else:
        mark_calculation_status(dft_dir, "failure", error_msg)
        print(f"  ✗ {ligand_name}: calculation failed - {error_msg}")
    
    return success, error_msg


def process_directory(root_dir: Path, orca_path: str) -> None:
    """
    Process all ligands in the root directory.
    
    Args:
        root_dir: Root directory containing ligand subdirectories
        orca_path: Path to ORCA executable
    """
    if not root_dir.exists():
        print(f"Error: Directory {root_dir} does not exist.")
        return
    
    if not Path(orca_path).exists():
        print(f"Error: ORCA executable not found at {orca_path}")
        return
    
    # Get all subdirectories (ligands)
    ligand_dirs = [d for d in root_dir.iterdir() if d.is_dir()]
    
    if not ligand_dirs:
        print(f"No ligand directories found in {root_dir}")
        return
    
    print(f"Found {len(ligand_dirs)} ligand directories")
    print(f"ORCA path: {orca_path}")
    print("-" * 50)
    
    successful_count = 0
    failed_count = 0
    skipped_count = 0
    
    for i, ligand_dir in enumerate(ligand_dirs, 1):
        print(f"[{i}/{len(ligand_dirs)}] Processing: {ligand_dir.name}")
        
        success, message = process_ligand(orca_path, ligand_dir)
        
        if success and message == "Already completed":
            skipped_count += 1
        elif success:
            successful_count += 1
        else:
            failed_count += 1
        
        print()  # Empty line for readability
    
    # Summary
    print("=" * 50)
    print("CALCULATION SUMMARY")
    print("=" * 50)
    print(f"Total ligands: {len(ligand_dirs)}")
    print(f"Successfully completed: {successful_count}")
    print(f"Failed: {failed_count}")
    print(f"Skipped (already completed): {skipped_count}")
    
    if failed_count > 0:
        print(f"\nNote: {failed_count} calculations failed. You can re-run this script")
        print("to retry the failed calculations.")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run DFT single-point energy calculations using ORCA"
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="Root directory containing ligand subdirectories"
    )
    parser.add_argument(
        "--orca-path",
        type=str,
        default="/home/yuejian/project/MLFF-distill/yuejian/orca/orca_6_0_0_shared_openmpi416/orca",
        help="Path to ORCA executable (default: /home/yuejian/project/MLFF-distill/yuejian/orca/orca_6_0_0_shared_openmpi416/orca)"
    )
    
    args = parser.parse_args()
    
    root_dir = Path(args.root_dir)
    orca_path = args.orca_path
    
    print("DFT Single-Point Energy Calculation Manager")
    print("=" * 50)
    print(f"Root directory: {root_dir}")
    print(f"ORCA executable: {orca_path}")
    print("=" * 50)
    
    # Confirm before starting
    response = input("Do you want to proceed with the calculations? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Calculation cancelled.")
        return
    
    process_directory(root_dir, orca_path)


if __name__ == "__main__":
    main()
