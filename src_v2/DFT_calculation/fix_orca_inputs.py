#!/usr/bin/env python3
"""
Script to fix ORCA input files by changing nprocs from 128 to 1
to avoid MPI conflicts when running in parallel.
"""

import os
import sys
from pathlib import Path
import re


def fix_orca_input_file(input_file: Path) -> bool:
    """
    Fix a single ORCA input file by changing nprocs from 128 to 1.
    
    Args:
        input_file: Path to ORCA input file
        
    Returns:
        True if file was modified, False if no changes needed
    """
    try:
        with open(input_file, 'r') as f:
            content = f.read()
        
        # Check if file contains nprocs 128
        if 'nprocs 128' in content:
            # Replace nprocs 128 with nprocs 2 (matching cpus-per-task)
            new_content = content.replace('nprocs 128', 'nprocs 2')
            
            # Write the modified content back
            with open(input_file, 'w') as f:
                f.write(new_content)
            
            print(f"Fixed: {input_file}")
            return True
        else:
            print(f"No change needed: {input_file}")
            return False
            
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return False


def find_and_fix_orca_inputs(root_dir: str):
    """
    Find all ORCA input files and fix them.
    
    Args:
        root_dir: Root directory to search for ORCA input files
    """
    root_path = Path(root_dir)
    
    # Find all .inp files
    input_files = list(root_path.rglob("*.inp"))
    
    if not input_files:
        print(f"No .inp files found in {root_dir}")
        return
    
    print(f"Found {len(input_files)} ORCA input files")
    
    fixed_count = 0
    for input_file in input_files:
        if fix_orca_input_file(input_file):
            fixed_count += 1
    
    print(f"\nSummary: Fixed {fixed_count} out of {len(input_files)} files")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fix_orca_inputs.py <root_directory>")
        sys.exit(1)
    
    root_dir = sys.argv[1]
    if not os.path.exists(root_dir):
        print(f"Directory not found: {root_dir}")
        sys.exit(1)
    
    find_and_fix_orca_inputs(root_dir) 