#!/usr/bin/env python3
"""
DFT Single-Point Energy Calculation Manager with Multiprocessing
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import Tuple, Optional, List
from multiprocessing import Pool, cpu_count
import multiprocessing as mp


def run_orca_calculation_mp(args_tuple: Tuple[str, Path, Path]) -> Tuple[str, bool, str]:
    """
    Run ORCA calculation for a single ligand (multiprocessing version).
    
    Args:
        args_tuple: (orca_path, input_file, output_dir)
        
    Returns:
        Tuple of (ligand_name, success: bool, error_message: str)
    """
    orca_path, input_file, output_dir = args_tuple
    ligand_name = input_file.parent.parent.name
    
    try:
        # Prepare command with output redirection using absolute paths
        output_file = input_file.stem + ".out"
        input_abs_path = str(input_file.absolute())
        output_abs_path = str((input_file.parent.parent / output_file).absolute())
        
        # Force ORCA to run in serial mode by setting environment variables
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = '1'
        env['ORA_NUM_PROCS'] = '1'
        env['ORA_NUM_THREADS'] = '1'
        
        cmd = f"{orca_path} {input_abs_path} > {output_abs_path}"
        
        # Run ORCA calculation with timeout (e.g., 24 hours)
        timeout_seconds = 24 * 3600  # 24 hours
        
        print(f"[{ligand_name}] Running ORCA calculation: {cmd}")
        
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            text=True,
            env=env
        )

        # Check if calculation was successful
        if result.returncode == 0:
            # Read the output file to check for ORCA-specific success indicators
            try:
                with open(output_abs_path, 'r') as f:
                    output_content = f.read()
                if "ORCA TERMINATED NORMALLY" in output_content:
                    return ligand_name, True, ""
                else:
                    return ligand_name, False, "ORCA did not terminate normally"
            except FileNotFoundError:
                return ligand_name, False, "Output file not found after calculation"
        else:
            return ligand_name, False, f"ORCA failed with return code {result.returncode}"
            
    except subprocess.TimeoutExpired:
        return ligand_name, False, "Calculation timed out"
    except Exception as e:
        return ligand_name, False, f"Exception occurred: {str(e)}"


def check_calculation_status(dft_dir: Path) -> Optional[str]:
    """Check if a calculation has already been completed or failed."""
    success_file = dft_dir / "success.txt"
    failure_file = dft_dir / "failure.txt"
    
    if success_file.exists():
        return "success"
    elif failure_file.exists():
        return "failure"
    else:
        return None


def mark_calculation_status(dft_dir: Path, status: str, message: str = ""):
    """Mark calculation status with success or failure file."""
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


def prepare_calculation_tasks(root_dir: Path, orca_path: str) -> List[Tuple[str, Path, Path]]:
    """
    Prepare list of calculation tasks for multiprocessing.
    
    Returns:
        List of (orca_path, input_file, output_dir) tuples
    """
    tasks = []
    
    # Get all subdirectories (ligands)
    ligand_dirs = [d for d in root_dir.iterdir() if d.is_dir()]
    
    for ligand_dir in ligand_dirs:
        ligand_name = ligand_dir.name
        dft_dir = ligand_dir / "DFT"
        input_dir = dft_dir / "input"
        
        # Check if DFT directory structure exists
        if not dft_dir.exists() or not input_dir.exists():
            print(f"Skipping {ligand_name}: missing DFT structure")
            continue
        
        # Find input file
        input_files = list(input_dir.glob("*.inp"))
        if not input_files:
            print(f"Skipping {ligand_name}: no .inp files found")
            continue
        
        input_file = input_files[0]  # Take the first .inp file
        
        # Check current status
        status = check_calculation_status(dft_dir)
        if status == "success":
            print(f"Skipping {ligand_name}: already completed successfully")
            continue
        elif status == "failure":
            print(f"Retrying {ligand_name}: previous attempt failed")
        
        # Add to task list
        tasks.append((orca_path, input_file, dft_dir))
    
    return tasks


def process_directory_mp(root_dir: Path, orca_path: str, n_processes: int = None) -> None:
    """
    Process all ligands using multiprocessing.
    
    Args:
        root_dir: Root directory containing ligand subdirectories
        orca_path: Path to ORCA executable
        n_processes: Number of processes to use (default: CPU count)
    """
    if not root_dir.exists():
        print(f"Error: Directory {root_dir} does not exist.")
        return
    
    if not Path(orca_path).exists():
        print(f"Error: ORCA executable not found at {orca_path}")
        return
    
    # Prepare tasks
    tasks = prepare_calculation_tasks(root_dir, orca_path)
    
    if not tasks:
        print(f"No calculations to run in {root_dir}")
        return
    
    print(f"Found {len(tasks)} calculations to run")
    print(f"ORCA path: {orca_path}")
    print(f"Using {n_processes or cpu_count()} processes")
    print("-" * 50)
    
    # Set number of processes
    if n_processes is None:
        n_processes = min(cpu_count(), len(tasks))
    
    # Run calculations using multiprocessing
    successful_count = 0
    failed_count = 0
    
    with Pool(processes=n_processes) as pool:
        # Submit all tasks
        results = pool.map(run_orca_calculation_mp, tasks)
        
        # Process results
        for ligand_name, success, error_msg in results:
            if success:
                print(f"  ✓ {ligand_name}: calculation completed successfully")
                successful_count += 1
                
                # Mark success
                for task in tasks:
                    if task[1].parent.parent.name == ligand_name:
                        mark_calculation_status(task[2], "success")
                        break
            else:
                print(f"  ✗ {ligand_name}: calculation failed - {error_msg}")
                failed_count += 1
                
                # Mark failure
                for task in tasks:
                    if task[1].parent.parent.name == ligand_name:
                        mark_calculation_status(task[2], "failure", error_msg)
                        break
    
    # Summary
    print("=" * 50)
    print("CALCULATION SUMMARY")
    print("=" * 50)
    print(f"Total calculations: {len(tasks)}")
    print(f"Successfully completed: {successful_count}")
    print(f"Failed: {failed_count}")
    
    if failed_count > 0:
        print(f"\nNote: {failed_count} calculations failed. You can re-run this script")
        print("to retry the failed calculations.")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run DFT single-point energy calculations using ORCA with multiprocessing"
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
        help="Path to ORCA executable"
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        default=None,
        help="Number of processes to use (default: CPU count)"
    )
    
    args = parser.parse_args()
    
    root_dir = Path(args.root_dir)
    orca_path = args.orca_path
    
    print("DFT Single-Point Energy Calculation Manager (Multiprocessing)")
    print("=" * 60)
    print(f"Root directory: {root_dir}")
    print(f"ORCA executable: {orca_path}")
    print(f"Processes: {args.n_processes or 'auto'}")
    print("=" * 60)
    
    # Confirm before starting (skip in batch mode)
    if not os.environ.get('SLURM_JOB_ID'):
        response = input("Do you want to proceed with the calculations? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Calculation cancelled.")
            return
    else:
        print("Running in batch mode - proceeding automatically...")
    
    process_directory_mp(root_dir, orca_path, args.n_processes)


if __name__ == "__main__":
    main() 