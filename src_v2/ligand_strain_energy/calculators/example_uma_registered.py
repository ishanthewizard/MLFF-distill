#!/usr/bin/env python3
"""
Example usage of registered UMA methods for strain relief.

This script demonstrates how to use the UMA calculator through the registered
minimisation and energy evaluation systems.
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from rdkit import Chem
from ligand_strain_energy.utils._minimisation import minimise_conformers
from ligand_strain_energy.utils._energy_eval import predict_energy


def main():
    """Example of using registered UMA methods for strain relief."""
    
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
    model_path = "/path/to/your/uma-checkpoint.pt"
    
    print("=== UMA Energy Prediction ===")
    try:
        # Predict energies using UMA
        mols_with_energies = predict_energy(
            mols=mols,
            method="UMA",
            model_paths=model_path,
            device="cuda",
            task_name="omol",
            uma_energy_units="eV",
        )
        
        print("Energy prediction completed!")
        for mol_id, mol in mols_with_energies.items():
            print(f"\nMolecule {mol_id}:")
            print(f"  Number of conformers: {mol.GetNumConformers()}")
            
            for conf_id in range(mol.GetNumConformers()):
                energy = mol.GetConformer(conf_id).GetDoubleProp("energy")
                print(f"  Conformer {conf_id} energy: {energy:.4f} kcal/mol")
    
    except Exception as e:
        print(f"Error during energy prediction: {e}")
    
    print("\n=== UMA Minimisation ===")
    try:
        # Minimize conformers using UMA
        minimized_mols = minimise_conformers(
            mols=mols,
            method="UMA",
            model_paths=model_path,
            maxIters=100,
            fmax=0.05,
            fexit=250,
            device="cuda",
            task_name="omol",
            uma_energy_units="eV",
        )
        
        print("Minimization completed!")
        for mol_id, mol in minimized_mols.items():
            print(f"\nMolecule {mol_id}:")
            print(f"  Number of conformers: {mol.GetNumConformers()}")
            
            for conf_id in range(mol.GetNumConformers()):
                energy = mol.GetConformer(conf_id).GetDoubleProp("energy")
                print(f"  Conformer {conf_id} energy: {energy:.4f} kcal/mol")
    
    except Exception as e:
        print(f"Error during minimization: {e}")
        print("Make sure you have:")
        print("1. A valid UMA checkpoint file")
        print("2. The required dependencies installed (fairchem, rdkit, etc.)")
        print("3. Sufficient GPU memory if using CUDA")


if __name__ == "__main__":
    main() 