#!/usr/bin/env python3
"""
Script to prepare ORCA input files from SDF files.
Processes all SDF files in subdirectories and generates corresponding ORCA input files.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple


def parse_sdf_file(sdf_path: Path) -> Tuple[List[str], int]:
    """
    Parse SDF file and extract atomic coordinates and charge.
    
    Args:
        sdf_path: Path to the SDF file
        
    Returns:
        Tuple of (coordinates_list, total_charge)
    """
    coordinates = []
    total_charge = 0
    
    with open(sdf_path, 'r') as f:
        lines = f.readlines()
    
    # Find the atom block (starts with atom count line)
    atom_start = None
    for i, line in enumerate(lines):
        if line.strip() and line.strip().split()[0].isdigit():
            parts = line.strip().split()
            if len(parts) >= 3 and parts[1].isdigit():  # This should be the atom/bond count line
                atom_start = i + 1
                num_atoms = int(parts[0])
                break
    
    if atom_start is None:
        raise ValueError(f"Could not find atom block in SDF file: {sdf_path}")
    
    # Extract atom coordinates
    for i in range(num_atoms):
        if atom_start + i >= len(lines):
            break
        line = lines[atom_start + i].strip()
        if not line:
            continue
            
        parts = line.split()
        if len(parts) >= 4:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            atom_symbol = parts[3]
            coordinates.append(f"{atom_symbol}   {x:10.4f}    {y:10.4f}    {z:10.4f}")
    
    # Look for charge information in the file
    for line in lines:
        if "M  CHG" in line:
            parts = line.strip().split()
            if len(parts) >= 4:
                num_charges = int(parts[2])
                for i in range(num_charges):
                    if 3 + 2*i + 1 < len(parts):
                        charge = int(parts[3 + 2*i + 1])
                        total_charge += charge
            break
    
    return coordinates, total_charge


def generate_orca_input(coordinates: List[str], charge: int, nprocs: int = 8, max_iter: int = 100, guessmix: int = 50) -> str:
    """
    Generate ORCA input file content.
    
    Args:
        coordinates: List of atomic coordinates as strings
        charge: Total charge of the system
        nprocs: Number of processors to use
        max_iter: Maximum number of SCF iterations
        guessmix: Number for guessmix parameter
        
    Returns:
        ORCA input file content as string
    """
    orca_content = f"""! UKS wB97M-V DEF2-TZVPD TightSCF DEFGRID3 RIJCOSX
%scf
  MaxIter {max_iter}
  Thresh 1e-12        # Integral threshold
  TCut   1e-13        # Primitive batch threshold
  guessmix {guessmix}
end

%pal
  nprocs {nprocs}            # Adjust this to number of threads
end

* xyz {charge} 1
"""
    
    # Add coordinates
    for coord in coordinates:
        orca_content += coord + "\n"
    
    orca_content += "*\n"
    
    return orca_content


def process_directory(base_dir: Path, nprocs: int = 8, max_iter: int = 100, guessmix: int = 50) -> None:
    """
    Process all SDF files in subdirectories and generate ORCA input files.
    Saves ORCA input files in DFT/input/ subdirectory within each ligand directory.
    
    Args:
        base_dir: Base directory containing subdirectories with SDF files
        nprocs: Number of processors for ORCA calculations
        max_iter: Maximum number of SCF iterations
        guessmix: Number for guessmix parameter
    """
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist.")
        return
    
    processed_count = 0
    error_count = 0
    
    # Iterate through all subdirectories
    for subdir in base_dir.iterdir():
        if not subdir.is_dir():
            continue
            
        # Look for SDF files in the subdirectory
        sdf_files = list(subdir.glob("*.sdf"))
        
        for sdf_file in sdf_files:
            try:
                print(f"Processing: {sdf_file}")
                
                # Parse SDF file
                coordinates, charge = parse_sdf_file(sdf_file)
                
                # Generate ORCA input content
                orca_content = generate_orca_input(coordinates, charge, nprocs, max_iter, guessmix)
                
                # Create DFT/input directory within the ligand directory
                dft_input_dir = subdir / "DFT" / "input"
                dft_input_dir.mkdir(parents=True, exist_ok=True)
                
                # Create output filename
                output_filename = sdf_file.stem + ".inp"
                output_path = dft_input_dir / output_filename
                
                # Write ORCA input file
                with open(output_path, 'w') as f:
                    f.write(orca_content)
                
                print(f"  Generated: {output_path}")
                processed_count += 1
                
            except Exception as e:
                print(f"  Error processing {sdf_file}: {e}")
                error_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count} files")
    print(f"Errors: {error_count} files")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate ORCA input files from SDF files in subdirectories"
    )
    parser.add_argument(
        "base_dir",
        type=str,
        help="Base directory containing subdirectories with SDF files"
    )
    parser.add_argument(
        "--nprocs",
        type=int,
        default=8,
        help="Number of processors for ORCA calculations (default: 8)"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Maximum number of SCF iterations (default: 100)"
    )
    parser.add_argument(
        "--guessmix",
        type=int,
        default=50,
        help="Number for guessmix parameter (default: 50)"
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    print(f"Processing directory: {base_dir}")
    print(f"Number of processors: {args.nprocs}")
    print(f"Maximum SCF iterations: {args.max_iter}")
    print(f"Guessmix parameter: {args.guessmix}")
    print("ORCA input files will be saved in DFT/input/ subdirectories within each ligand directory")
    print("Existing .inp files will be overwritten")
    print("-" * 50)
    
    process_directory(base_dir, args.nprocs, args.max_iter, args.guessmix)


if __name__ == "__main__":
    main()
