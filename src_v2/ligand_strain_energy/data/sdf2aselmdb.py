import os
from tqdm import tqdm
from rdkit import Chem
from ase.io import read
from ase.db import connect
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np
from typing import List, Tuple, Dict, Any

def compute_spin_multiplicity_and_charge(mol: Chem.Mol) -> Tuple[int, int]:
    """
    Compute the spin multiplicity and formal charge of an RDKit Mol object.

    Args:
        mol (rdkit.Chem.Mol): RDKit molecule object.
    Returns:
        multiplicity (int): Spin multiplicity (2S+1).
        formal_charge (int): Formal charge of the molecule.
    """
    unpaired_electrons = 0
    for atom in mol.GetAtoms():
        spin = atom.GetNumRadicalElectrons()
        unpaired_electrons += spin
    total_spin = 0.5 * unpaired_electrons
    multiplicity = int(2 * total_spin + 1)
    formal_charge = Chem.GetFormalCharge(mol)
    return multiplicity, formal_charge

def parse_dft_energy_forces(dft_out_path: str) -> Tuple[float, np.ndarray]:
    """
    Parse total energy (in eV) and forces (in eV/Å) from an ORCA DFT output file.

    Args:
        dft_out_path (str): Path to the ORCA DFT output file.
    Returns:
        energy (float): Total energy in eV.
        forces (np.ndarray): Forces in eV/Å, shape (natoms, 3).
    """
    import re
    # Read the DFT output file
    with open(dft_out_path, "r") as f:
        dft_lines = f.readlines()
    # Parse total energy in eV
    energy_eV = None
    energy_pattern = re.compile(r"Total Energy\s*:\s*([-\d\.Ee+]+)\s*Eh\s*([-\d\.Ee+]+)\s*eV")
    for line in dft_lines:
        match = energy_pattern.search(line)
        if match:
            energy_eV = float(match.group(2))
            break
    if energy_eV is None:
        raise ValueError("Could not find total energy in eV in the DFT output file.")
    # Parse forces from the "CARTESIAN GRADIENT" section
    forces = []
    cartesian_gradient_start = None
    for idx, line in enumerate(dft_lines):
        if line.strip() == "CARTESIAN GRADIENT":
            cartesian_gradient_start = idx
            break
    if cartesian_gradient_start is None:
        raise ValueError("Could not find 'CARTESIAN GRADIENT' section in the DFT output file.")
    # The actual data starts 3 lines after the header
    force_lines = []
    for line in dft_lines[cartesian_gradient_start+3:]:
        if not line.strip():
            break
        force_lines.append(line)
    hartree_to_eV = 27.2113961
    Bohr_to_angstrom = 0.529177249
    hartree_per_bohr_to_eV_per_angstrom = hartree_to_eV / Bohr_to_angstrom
    for line in force_lines:
        parts = line.split(":")
        if len(parts) != 2:
            continue
        force_str = parts[1].strip()
        force_vals = [float(x) * hartree_per_bohr_to_eV_per_angstrom for x in force_str.split()]
        if len(force_vals) == 3:
            forces.append(force_vals)
    forces = np.array(forces, dtype=np.float64)
    energy = float(energy_eV)
    return energy, forces

def process_single_sdf(sdf_path: str) -> List[Tuple[Any, Dict[str, Any]]]:
    """
    Process a single SDF file: read molecules, compute charge/spin, parse DFT energy/forces if present.

    Args:
        sdf_path (str): Path to the .sdf file.
    Returns:
        results (list of (ase.Atoms, dict)): List of tuples, each containing an ASE Atoms object and a data dict with charge, spin, ligand_id, energy, and forces.
    """
    ligand_id = os.path.basename(os.path.dirname(sdf_path))
    # Read all molecules in the SDF (could be more than one)
    mols = read(sdf_path, index=":")
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    rdkit_mols = [mol for mol in suppl if mol is not None]
    # DFT output file (optional)
    ligand_dir = os.path.dirname(sdf_path)
    dft_out_file = os.path.join(ligand_dir, "DFT", f"{ligand_id}.out")
    dft_present = os.path.isfile(dft_out_file)
    results = []
    for i, atoms in enumerate(mols):
        rdkit_mol = rdkit_mols[i] if i < len(rdkit_mols) else None
        if rdkit_mol is not None:
            spin, charge = compute_spin_multiplicity_and_charge(rdkit_mol)
        else:
            spin, charge = 1, 0
        data = {"charge": charge, "spin": spin, "ligand_id": ligand_id}
        # Try to get energy/forces from DFT if present
        if dft_present:
            try:
                energy, forces = parse_dft_energy_forces(dft_out_file)
            except Exception as e:
                print(f"Warning: Failed to parse DFT for {ligand_id}: {e}")
                energy = 0.0
                forces = np.zeros((len(atoms), 3), dtype=np.float64)
        else:
            energy = getattr(atoms, "energy", 0.0)
            forces = getattr(atoms, "forces", np.zeros((len(atoms), 3), dtype=np.float64))
        # Ensure correct types
        energy = float(energy)
        forces = np.asarray(forces, dtype=np.float64)
        calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        atoms.calc = calc
        data["energy"] = energy
        data["forces"] = forces
        results.append((atoms, data))
    return results

def find_all_sdf_files(root_dir: str) -> List[str]:
    """
    Recursively find all .sdf files in a root directory and its subdirectories.

    Args:
        root_dir (str): Root directory to search.
    Returns:
        sdf_paths (list of str): List of full paths to .sdf files found.
    """
    sdf_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith('.sdf'):
                sdf_paths.append(os.path.join(dirpath, f))
    return sdf_paths

def process_sdf_paths_to_aselmdb(sdf_paths: List[str], aselmdb_path: str, split_name: str) -> None:
    """
    Process a list of SDF file paths and write all molecules to a single aselmdb file.

    Args:
        sdf_paths (list of str): List of SDF file paths to process.
        aselmdb_path (str): Output path for the aselmdb file.
        split_name (str): Name of the split (for progress bar display).
    Returns:
        None. Writes to aselmdb file.
    """
    import os
    from ase.db import connect
    from tqdm import tqdm
    if os.path.exists(aselmdb_path):
        os.remove(aselmdb_path)
    written = 0
    with connect(aselmdb_path) as dst_db:
        for sdf_path in tqdm(sdf_paths, desc=f"Processing {split_name} SDFs"):
            try:
                for atoms, data in process_single_sdf(sdf_path):
                    dst_db.write(atoms, data=data)
                    written += 1
            except Exception as e:
                print(f"Failed to process {sdf_path}: {e}")
    print(f"Total structures written to {aselmdb_path}: {written}")


def main() -> None:
    """
    Example main function for processing a test set of SDF files into an aselmdb.
    Edit subdir_ligand_sdf_paths_test and test_aselmdb_path as needed.
    """
    # Example usage: process test set
    # Set these variables as needed
    subdir_ligand_sdf_paths_test: List[str] = [
        # list of SDF file paths for the test set
        # e.g. '/path/to/ligand1/ligand1.sdf', '/path/to/ligand2/ligand2.sdf', ...
    ]
    test_aselmdb_path: str = "test_data0000.aselmdb"  # Set your output path
    process_sdf_paths_to_aselmdb(subdir_ligand_sdf_paths_test, test_aselmdb_path, "test")


if __name__ == "__main__":
    main()
