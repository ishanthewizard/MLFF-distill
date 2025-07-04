import os
import pytest
from rdkit import Chem
import numpy as np
from tqdm import tqdm

# Paths
input_sdf = '/home/yuejian/project/MLFF-distill/strain/ligboundconf_2_raw.sdf'
output_base = '/home/yuejian/project/MLFF-distill/data/ligandboundconf3.0/raw_mol'
def get_ligand_id(mol):
    if mol.HasProp('ligand_id'):
        ligand_id = mol.GetProp('ligand_id').strip()
        if ligand_id:
            return ligand_id
    return None

def get_atoms_info(mol):
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    return atoms, positions

def test_split_sdf_to_ligands():
    # Load original molecules
    supplier = Chem.SDMolSupplier(input_sdf, removeHs=False)
    orig_mols = {}
    for mol in tqdm(supplier, desc='Reading original SDF'):
        if mol is None:
            continue
        ligand_id = get_ligand_id(mol)
        if ligand_id:
            orig_mols[ligand_id] = mol
    print(f"Number of ligand_ids found: {len(orig_mols)}")
    print(f"Example ligand_ids: {list(orig_mols.keys())[:5]}")
    # For each ligand_id, load the split SDF and compare
    for ligand_id, orig_mol in tqdm(orig_mols.items(), desc='Comparing split SDFs', total=len(orig_mols)):
        split_path = os.path.join(output_base, ligand_id, f'{ligand_id}.sdf')
        assert os.path.exists(split_path), f"Split SDF not found for {ligand_id}"
        split_supplier = Chem.SDMolSupplier(split_path, removeHs=False)
        split_mols = [m for m in split_supplier if m is not None]
        assert len(split_mols) == 1, f"Expected 1 mol in split SDF for {ligand_id}, got {len(split_mols)}"
        split_mol = split_mols[0]
        # Compare atom types
        orig_atoms, orig_pos = get_atoms_info(orig_mol)
        split_atoms, split_pos = get_atoms_info(split_mol)
        assert orig_atoms == split_atoms, f"Atom types differ for {ligand_id}"
        # Compare positions (allowing for small numerical differences)
        assert np.allclose(orig_pos, split_pos, atol=1e-4), f"Atom positions differ for {ligand_id}"

if __name__ == "__main__":
    test_split_sdf_to_ligands()
    print("All split SDFs match the original molecules.") 