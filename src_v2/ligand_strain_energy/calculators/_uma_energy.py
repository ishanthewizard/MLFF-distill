import tempfile
from typing import Literal

from loguru import logger as logging
from rdkit import Chem

from strain_relief.constants import EV_TO_KCAL_PER_MOL, HARTREE_TO_KCAL_PER_MOL
from strain_relief.io import rdkit_to_ase
from strain_relief.io.utils_s3 import copy_from_s3
from fairchem.core import pretrained_mlip, FAIRChemCalculator


def UMA_energy(
    mols: dict[str, Chem.Mol],
    model_paths: str,
    device: Literal["cpu", "cuda"] = "cpu",
    uma_energy_units: Literal["eV", "Hartrees", "kcal/mol"] = "eV",
    task_name: Literal["omol", "omat", "oc20", "odac", "omc"] = "omol",
) -> dict[str, dict[str, float]]:
    """Calculate the UMA energy for all conformers of all molecules.

    Parameters
    ----------
    mols : dict[str, Chem.Mol]
        A dictionary of molecules.
    model_paths : str
        Path to the UMA model checkpoint to use for energy calculation.
    device : Literal["cpu", "cuda"]
        The device to use for energy calculation.
    uma_energy_units : Literal["eV", "Hartrees", "kcal/mol"]
        The units output from the energy calculation.
    task_name: Literal["omol", "omat", "oc20", "odac", "omc"]
        The task name to use for the UMA model.

    Returns
    -------
    dict[str, dict[str, float]]
        A dictionary of dictionaries of conformer energies for each molecule.

        mol_energies = {
            "mol_id": {
                "conf_id": energy
            }
        }
    """
    if model_paths.startswith("s3://"):
        local_path = tempfile.mktemp(suffix=".pt")
        copy_from_s3(model_paths, local_path)
        model_paths = local_path

    if uma_energy_units == "eV":
        conversion_factor = EV_TO_KCAL_PER_MOL
        logging.info("UMA model outputs energies in eV. Converting to kcal/mol.")
    elif uma_energy_units == "Hartrees":
        conversion_factor = HARTREE_TO_KCAL_PER_MOL
        logging.info("UMA model outputs energies in Hartrees. Converting to kcal/mol.")
    elif uma_energy_units == "kcal/mol":
        conversion_factor = 1
        logging.info("UMA model outputs energies in kcal/mol. No conversion needed.")

    # Load UMA model and create calculator
    predictor = pretrained_mlip.get_predict_unit(model_paths, device=device)
    calculator = FAIRChemCalculator(predictor, task_name=task_name)

    energies = {}
    for id, mol in mols.items():
        energies[id] = _UMA_energy(mol, id, calculator, conversion_factor)
    return energies


def _UMA_energy(
    mol: Chem.Mol,
    id: str,
    calculator,
    conversion_factor: float,
) -> dict[int, float]:
    """Calculate the UMA energy for all conformers of a molecule.

    Parameters
    ----------
    mol : Chem.Mol
        A molecule.
    id : str
        ID of the molecule. Used for logging.
    calculator
        The ASE calculator to use for energy calculation.
    conversion_factor : float
        The conversion factor to use for energy calculation.

    Returns
    -------
    dict[int, float]
        A dictionary of conformer energies.

        conf_energies = {
            "conf_id": energy
        }
    """
    confs_and_ids = rdkit_to_ase(mol)
    for _, atoms in confs_and_ids:
        atoms.calc = calculator
    conf_energies = {
        conf_id: atoms.get_potential_energy() * conversion_factor
        for conf_id, atoms in confs_and_ids
    }
    for conf_id, energy in conf_energies.items():
        logging.debug(f"{id}: Conformer {conf_id} energy = {energy} kcal/mol")

    return conf_energies 