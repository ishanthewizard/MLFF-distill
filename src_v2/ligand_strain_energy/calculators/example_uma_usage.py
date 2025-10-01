#!/usr/bin/env python3
"""
Example usage of UMA calculator for strain relief.

This script demonstrates how to use the UMA calculator to minimize molecular conformers
using a UMA checkpoint.
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from rdkit import Chem
from ligand_strain_energy.calculators import UMA_min


def main():
    """Example of using UMA calculator for strain relief."""
    
    # Example SMILES for a molecule
    smiles = "CC(C)(C)OC(=O)N[C@@H](CC1=CC=CC=C1)C(=O)O"
    
    # Create a molecule with conformers
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    # Generate conformers (this is a simplified example)
    # In practice, you would use a proper conformer generation method
    from rdkit.Chem import AllChem
    AllChem.EmbedMultipleConfs(mol, numConfs=5, randomSeed=42)
    
    # Create a dictionary of molecules
    mols = {"example_mol": mol}
    
    # Path to your UMA checkpoint
    # Replace with your actual checkpoint path
    model_path = "/home/yuejian/project/MLFF-distill/OMol_Whole/ckpt/uma-s-1.pt"
    
    # Run UMA minimization
    # try:
    energies, minimized_mols = UMA_min(
        mols=mols,
        model_paths=model_path,
        maxIters=100,  # Maximum iterations
        fmax=0.05,     # Force convergence threshold
        fexit=250,     # Force exit threshold
        device="cuda", # Use GPU if available
        task_name="omol",  # Task name for UMA model
        uma_energy_units="eV",  # Energy units from model
    )
    
    print("Minimization completed!")
    print(f"Number of minimized molecules: {len(minimized_mols)}")
    
    for mol_id, mol in minimized_mols.items():
        print(f"\nMolecule {mol_id}:")
        print(f"  Number of conformers: {mol.GetNumConformers()}")
        
        if mol_id in energies:
            for conf_id, energy in energies[mol_id].items():
                print(f"  Conformer {conf_id} energy: {energy:.4f} kcal/mol")
    
    # except Exception as e:
    #     print(f"Error during minimization: {e}")
    #     print("Make sure you have:")
    #     print("1. A valid UMA checkpoint file")
    #     print("2. The required dependencies installed (fairchem, rdkit, etc.)")
    #     print("3. Sufficient GPU memory if using CUDA")


if __name__ == "__main__":
    main() 