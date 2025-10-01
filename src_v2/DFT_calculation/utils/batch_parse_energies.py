#!/usr/bin/env python3
"""
Script to batch parse ORCA output files from multiple ligands and extract energy values.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import argparse

# Add the current directory to the path so we can import parse_orca_energy
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
from parse_orca_energy import parse_orca_energy_with_units, parse_total_scf_energy, convert_energy_units


def find_orca_output_files(root_dir: Path) -> Dict[str, Path]:
    """
    Find all ORCA output files in the ligand directory structure.
    
    Args:
        root_dir: Root directory containing ligand subdirectories
        
    Returns:
        Dictionary mapping ligand_id to output file path
    """
    output_files = {}
    
    if not root_dir.exists():
        print(f"Error: Directory {root_dir} does not exist.")
        return output_files
    
    # Look for ligand directories
    for ligand_dir in root_dir.iterdir():
        if not ligand_dir.is_dir():
            continue
            
        ligand_id = ligand_dir.name
        
        # Look for DFT directory
        dft_dir = ligand_dir / "DFT"
        if not dft_dir.exists():
            print(f"Warning: No DFT directory found for ligand {ligand_id}")
            continue
        
        # Look for .out file in DFT directory
        out_files = list(dft_dir.glob("*.out"))
        if not out_files:
            print(f"Warning: No .out files found for ligand {ligand_id}")
            continue
        
        # Use the first .out file found
        output_files[ligand_id] = out_files[0]
    
    return output_files


def parse_ligand_energies(output_file: Path) -> Optional[Dict[str, float]]:
    """
    Parse energy values from a single ORCA output file.
    
    Args:
        output_file: Path to ORCA output file
        
    Returns:
        Dictionary containing energy values, or None if parsing failed
    """
    if not output_file.exists():
        print(f"Error: File {output_file} does not exist.")
        return None
    
    energies = {}
    
    # Parse final single point energy
    final_result = parse_orca_energy_with_units(output_file)
    if final_result:
        energy_hartree, unit = final_result
        energies['final_hartree'] = energy_hartree
        energies['final_eV'] = convert_energy_units(energy_hartree, 'eV')
        energies['final_kcal_mol'] = convert_energy_units(energy_hartree, 'kcal/mol')
        energies['final_kJ_mol'] = convert_energy_units(energy_hartree, 'kJ/mol')
    
    # Parse total SCF energy
    scf_result = parse_total_scf_energy(output_file)
    if scf_result:
        scf_hartree, scf_eV = scf_result
        energies['scf_hartree'] = scf_hartree
        energies['scf_eV'] = scf_eV
        energies['scf_kcal_mol'] = convert_energy_units(scf_hartree, 'kcal/mol')
        energies['scf_kJ_mol'] = convert_energy_units(scf_hartree, 'kJ/mol')
    
    return energies if energies else None


def batch_parse_energies(root_dir: Path, verbose: bool = False) -> Dict[str, Dict[str, float]]:
    """
    Parse energy values from all ORCA output files in the ligand directory structure.
    
    Args:
        root_dir: Root directory containing ligand subdirectories
        verbose: Whether to print progress information
        
    Returns:
        Dictionary mapping ligand_id to energy values dictionary
    """
    if verbose:
        print(f"Scanning directory: {root_dir}")
    
    # Find all ORCA output files
    output_files = find_orca_output_files(root_dir)
    
    if verbose:
        print(f"Found {len(output_files)} ligands with ORCA output files")
    
    # Parse energies for each ligand
    results = {}
    successful_count = 0
    failed_count = 0
    
    for ligand_id, output_file in output_files.items():
        if verbose:
            print(f"Processing {ligand_id}...")
        
        energies = parse_ligand_energies(output_file)
        
        if energies:
            results[ligand_id] = energies
            successful_count += 1
            if verbose:
                print(f"  ✓ Successfully parsed energies for {ligand_id}")
        else:
            failed_count += 1
            if verbose:
                print(f"  ✗ Failed to parse energies for {ligand_id}")
    
    if verbose:
        print(f"\nSummary:")
        print(f"  Successfully parsed: {successful_count}")
        print(f"  Failed to parse: {failed_count}")
        print(f"  Total ligands: {len(output_files)}")
    
    return results


def print_energy_summary(results: Dict[str, Dict[str, float]], format_type: str = 'table'):
    """
    Print a summary of the parsed energy values.
    
    Args:
        results: Dictionary of ligand energies
        format_type: 'table' or 'csv'
    """
    if not results:
        print("No energy data to display.")
        return
    
    # Get all available energy types
    energy_types = set()
    for energies in results.values():
        energy_types.update(energies.keys())
    
    energy_types = sorted(list(energy_types))
    
    if format_type == 'csv':
        # Print CSV header
        header = ['ligand_id'] + energy_types
        print(','.join(header))
        
        # Print data rows
        for ligand_id, energies in sorted(results.items()):
            row = [ligand_id]
            for energy_type in energy_types:
                value = energies.get(energy_type, 'N/A')
                row.append(str(value))
            print(','.join(row))
    
    else:  # table format
        # Print table header
        print(f"{'Ligand ID':<20}", end="")
        for energy_type in energy_types:
            print(f"{energy_type:<15}", end="")
        print()
        
        print("-" * (20 + 15 * len(energy_types)))
        
        # Print data rows
        for ligand_id, energies in sorted(results.items()):
            print(f"{ligand_id:<20}", end="")
            for energy_type in energy_types:
                value = energies.get(energy_type, 'N/A')
                if isinstance(value, float):
                    print(f"{value:<15.6f}", end="")
                else:
                    print(f"{value:<15}", end="")
            print()


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Batch parse ORCA output files to extract energy values"
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="Root directory containing ligand subdirectories"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=['dict', 'table', 'csv'],
        default='dict',
        help="Output format (default: dict)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (for CSV format)"
    )
    
    args = parser.parse_args()
    
    root_dir = Path(args.root_dir)
    
    # Parse energies
    results = batch_parse_energies(root_dir, verbose=args.verbose)
    
    if not results:
        print("No energy data found.")
        return 1
    
    # Output results
    if args.format == 'dict':
        print("Energy Results Dictionary:")
        print("=" * 50)
        for ligand_id, energies in sorted(results.items()):
            print(f"\n{ligand_id}:")
            for energy_type, value in sorted(energies.items()):
                if isinstance(value, float):
                    print(f"  {energy_type}: {value:.12f}")
                else:
                    print(f"  {energy_type}: {value}")
    
    elif args.format == 'table':
        print_energy_summary(results, 'table')
    
    elif args.format == 'csv':
        if args.output:
            # Save to file
            with open(args.output, 'w') as f:
                # Redirect stdout to file
                import sys
                original_stdout = sys.stdout
                sys.stdout = f
                print_energy_summary(results, 'csv')
                sys.stdout = original_stdout
            print(f"Results saved to {args.output}")
        else:
            # Print to console
            print_energy_summary(results, 'csv')
    
    return 0


if __name__ == "__main__":
    exit(main()) 