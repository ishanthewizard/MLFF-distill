#!/usr/bin/env python3
"""
Script to parse ORCA output files and extract the final single-point energy.
"""

import re
import argparse
import subprocess
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


def parse_orca_energy(output_file: Path) -> Optional[float]:
    """
    Parse ORCA output file to extract the final single point energy.
    
    Args:
        output_file: Path to ORCA output file
        
    Returns:
        Final single point energy in Hartree, or None if not found
    """
    if not output_file.exists():
        print(f"Error: File {output_file} does not exist.")
        return None
    
    # Pattern to match the final single point energy line
    # Matches: "FINAL SINGLE POINT ENERGY      -669.309552145802"
    energy_pattern = r'FINAL SINGLE POINT ENERGY\s+([+-]?\d+\.\d+)'
    
    try:
        with open(output_file, 'r') as f:
            content = f.read()
            
        # Search for the energy pattern
        match = re.search(energy_pattern, content)
        
        if match:
            energy = float(match.group(1))
            return energy
        else:
            print(f"Warning: Could not find final single point energy in {output_file}")
            return None
            
    except Exception as e:
        print(f"Error reading file {output_file}: {e}")
        return None


def parse_total_scf_energy(output_file: Path) -> Optional[Tuple[float, float]]:
    """
    Parse ORCA output file to extract the total SCF energy in both Hartree and eV.
    
    Args:
        output_file: Path to ORCA output file
        
    Returns:
        Tuple of (energy_hartree, energy_eV), or None if not found
    """
    if not output_file.exists():
        print(f"Error: File {output_file} does not exist.")
        return None
    
    # Pattern to match the total SCF energy line
    # Matches: "Total Energy       :       -669.30955214580194 Eh          -18212.83884 eV"
    scf_energy_pattern = r'Total Energy\s+:\s+([+-]?\d+\.\d+)\s+Eh\s+([+-]?\d+\.\d+)\s+eV'
    
    try:
        with open(output_file, 'r') as f:
            content = f.read()
            
        # Search for the SCF energy pattern
        match = re.search(scf_energy_pattern, content)
        
        if match:
            energy_hartree = float(match.group(1))
            energy_eV = float(match.group(2))
            return energy_hartree, energy_eV
        else:
            print(f"Warning: Could not find total SCF energy in {output_file}")
            return None
            
    except Exception as e:
        print(f"Error reading file {output_file}: {e}")
        return None


def parse_orca_energy_with_units(output_file: Path) -> Optional[Tuple[float, str]]:
    """
    Parse ORCA output file to extract the final single point energy with units.
    
    Args:
        output_file: Path to ORCA output file
        
    Returns:
        Tuple of (energy_value, unit), or None if not found
    """
    if not output_file.exists():
        print(f"Error: File {output_file} does not exist.")
        return None
    
    # Pattern to match the final single point energy line with potential units
    # Matches: "FINAL SINGLE POINT ENERGY      -669.309552145802"
    energy_pattern = r'FINAL SINGLE POINT ENERGY\s+([+-]?\d+\.\d+)'
    
    try:
        with open(output_file, 'r') as f:
            content = f.read()
            
        # Search for the energy pattern
        match = re.search(energy_pattern, content)
        
        if match:
            energy = float(match.group(1))
            # ORCA typically reports energies in Hartree (atomic units)
            return energy, "Hartree"
        else:
            print(f"Warning: Could not find final single point energy in {output_file}")
            return None
            
    except Exception as e:
        print(f"Error reading file {output_file}: {e}")
        return None


def convert_energy_units(energy_hartree: float, target_unit: str) -> float:
    """
    Convert energy from Hartree to other common units.
    
    Args:
        energy_hartree: Energy in Hartree
        target_unit: Target unit ('eV', 'kcal/mol', 'kJ/mol')
        
    Returns:
        Energy in target unit
    """
    # Conversion factors from Hartree
    conversions = {
        'eV': 27.211386245988,  # 1 Hartree = 27.211386245988 eV
        'kcal/mol': 627.5094740631,  # 1 Hartree = 627.5094740631 kcal/mol
        'kJ/mol': 2625.4996394799,  # 1 Hartree = 2625.4996394799 kJ/mol
    }
    
    if target_unit not in conversions:
        raise ValueError(f"Unsupported unit: {target_unit}. Supported units: {list(conversions.keys())}")
    
    return energy_hartree * conversions[target_unit]


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Parse ORCA output files to extract final single point energy"
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to ORCA output file"
    )
    parser.add_argument(
        "--unit",
        type=str,
        choices=['hartree', 'eV', 'kcal/mol', 'kJ/mol'],
        default='hartree',
        help="Output unit for energy (default: hartree)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional information"
    )
    parser.add_argument(
        "--scf",
        action="store_true",
        help="Extract total SCF energy instead of final single point energy"
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Extract both final single point energy and total SCF energy"
    )
    
    args = parser.parse_args()
    
    output_file = Path(args.output_file)
    
    if args.both:
        # Extract both energies
        final_result = parse_orca_energy_with_units(output_file)
        scf_result = parse_total_scf_energy(output_file)
        
        if args.verbose:
            print(f"File: {output_file}")
            print("-" * 50)
            
            if final_result:
                energy_hartree, unit = final_result
                print(f"Final Single Point Energy:")
                print(f"  Hartree: {energy_hartree:.12f}")
                if args.unit != 'hartree':
                    final_energy = convert_energy_units(energy_hartree, args.unit)
                    print(f"  {args.unit}: {final_energy:.12f}")
            else:
                print("Final Single Point Energy: Not found")
                
            if scf_result:
                scf_hartree, scf_eV = scf_result
                print(f"Total SCF Energy:")
                print(f"  Hartree: {scf_hartree:.12f}")
                print(f"  eV: {scf_eV:.8f}")
            else:
                print("Total SCF Energy: Not found")
        else:
            # Simple output format
            if final_result:
                energy_hartree, unit = final_result
                if args.unit == 'hartree':
                    print(f"Final: {energy_hartree:.12f}")
                else:
                    final_energy = convert_energy_units(energy_hartree, args.unit)
                    print(f"Final: {final_energy:.12f}")
            
            if scf_result:
                scf_hartree, scf_eV = scf_result
                print(f"SCF: {scf_hartree:.12f} Hartree, {scf_eV:.8f} eV")
        
        return 0
    
    elif args.scf:
        # Extract only SCF energy
        scf_result = parse_total_scf_energy(output_file)
        
        if scf_result is None:
            return 1
        
        scf_hartree, scf_eV = scf_result
        
        if args.verbose:
            print(f"File: {output_file}")
            print(f"Total SCF Energy:")
            print(f"  Hartree: {scf_hartree:.12f}")
            print(f"  eV: {scf_eV:.8f}")
        else:
            if args.unit == 'hartree':
                print(f"{scf_hartree:.12f}")
            elif args.unit == 'eV':
                print(f"{scf_eV:.8f}")
            else:
                # Convert from Hartree to requested unit
                converted_energy = convert_energy_units(scf_hartree, args.unit)
                print(f"{converted_energy:.12f}")
        
        return 0
    
    else:
        # Original behavior - extract final single point energy
        result = parse_orca_energy_with_units(output_file)
        
        if result is None:
            return 1
        
        energy_hartree, unit = result
        
        # Convert to requested unit
        if args.unit == 'hartree':
            final_energy = energy_hartree
            final_unit = 'Hartree'
        else:
            final_energy = convert_energy_units(energy_hartree, args.unit)
            final_unit = args.unit
        
        # Print results
        if args.verbose:
            print(f"File: {output_file}")
            print(f"Energy in Hartree: {energy_hartree:.12f}")
            print(f"Energy in {final_unit}: {final_energy:.12f}")
        else:
            print(f"{final_energy:.12f}")
        
        return 0


if __name__ == "__main__":
    exit(main()) 