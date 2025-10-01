from timeit import default_timer as timer
from typing import Literal

from loguru import logger as logging
from rdkit import Chem

from strain_relief.constants import ENERGY_PROPERTY_NAME
from strain_relief.energy_eval import MACE_energy, MMFF94_energy
from ligand_strain_energy.calculators import UMA_energy
from .method_registry import register_method, get_method, get_all_methods

# Register built-in energy evaluation methods
register_method("MACE", MACE_energy, category="energy")
register_method("MMFF94", MMFF94_energy, category="energy")
register_method("MMFF94s", MMFF94_energy, category="energy")
register_method("UMA", UMA_energy, category="energy")


def predict_energy(
    mols: dict[str, Chem.Mol], method: str, **kwargs
):
    """Predict the energy of all conformers of molecules in mols using a specified method.

    Parameters
    ----------
    mols : dict[str, Chem.Mol]
        A dictionary of molecules.
    method : str
        The method to use for energy prediction.
    **kwargs
        Additional keyword arguments to pass to the energy prediction method.

    Returns
    -------
    dict[str, Chem.Mol]
        A dictionary of molecules with the predicted energies stored as a property on each
        conformer.
    """
    start = timer()

    if method not in get_all_methods(category="energy"):
        raise ValueError(f"method must be in {get_all_methods(category='energy')}")

    logging.info(f"Predicting energies using {method}")
    # Select method and run energy evaluation
    energy_method = get_method(method, category="energy")
    energies = energy_method(mols, **kwargs)

    # Store the predicted energies as a property on each conformer
    for id, mol in mols.items():
        [
            mol.GetConformer(conf_id).SetDoubleProp(ENERGY_PROPERTY_NAME, energy)
            for conf_id, energy in energies[id].items()
        ]
    logging.info(
        f"Predicted energies stored as '{ENERGY_PROPERTY_NAME}' property on each conformer"
    )

    end = timer()
    logging.info(f"Energy prediction took {end - start:.2f} seconds. \n")

    return mols