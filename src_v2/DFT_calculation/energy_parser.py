#!/usr/bin/env python3
"""
Simple module to parse ORCA energies from a root directory.
Returns a dictionary with ligand IDs as keys and energy values as values.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any
from utils.batch_parse_energies import batch_parse_energies


def parse_ligand_energies_from_root(root_path: str) -> Dict[str, Dict[str, float]]:
    """
    Parse ORCA energy values from all ligands in a root directory.
    
    Args:
        root_path: Path to root directory containing ligand subdirectories
        
    Returns:
        Dictionary where:
        - Keys are ligand IDs (directory names)
        - Values are dictionaries containing energy values:
          - final_hartree: Final single point energy in Hartree
          - final_eV: Final single point energy in eV
          - final_kcal_mol: Final single point energy in kcal/mol
          - final_kJ_mol: Final single point energy in kJ/mol
          - scf_hartree: Total SCF energy in Hartree
          - scf_eV: Total SCF energy in eV
          - scf_kcal_mol: Total SCF energy in kcal/mol
          - scf_kJ_mol: Total SCF energy in kJ/mol
    """
    root_dir = Path(root_path)
    return batch_parse_energies(root_dir, verbose=False)


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse ORCA energies from a root directory")
    parser.add_argument("root_path", help="Path to root directory containing ligand subdirectories")
    args = parser.parse_args()
    
    print("Parsing energies from:", args.root_path)
    results = parse_ligand_energies_from_root(args.root_path)
    
    print(f"\nFound {len(results)} ligands with energy data:")
    for ligand_id, energies in results.items():
        print(f"\n{ligand_id}:")
        for energy_type, value in sorted(energies.items()):
            print(f"  {energy_type}: {value:.12f}") 