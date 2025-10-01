import tempfile
from typing import Literal

from loguru import logger as logging
from rdkit import Chem

from strain_relief.constants import EV_TO_KCAL_PER_MOL, HARTREE_TO_KCAL_PER_MOL
from strain_relief.io.utils_s3 import copy_from_s3
from strain_relief.minimisation.utils_minimisation import method_min
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit

def UMA_min(
    mols: dict[str, Chem.Mol],
    model_paths: str,
    maxIters: int,
    fmax: float,
    fexit: float,
    device: Literal["cpu", "cuda"] = "cpu",
    uma_energy_units: Literal["eV", "Hartrees", "kcal/mol"] = "eV",
    task_name: Literal["omol", "omat", "oc20", "odac", "omc"] = "omol",
) -> tuple[dict[str, dict[str, float]], dict[str, Chem.Mol]]:
    """Minimise all conformers of a Chem.Mol using UMA.

    Parameters
    ----------
    mols : dict[str, Chem.Mol]
        Dictionary of molecules to minimise.
    model_paths : str
        Path to the UMA model checkpoint to use for energy calculation.
    maxIters : int
        Maximum number of iterations for the minimisation.
    fmax : float
        Convergence criteria, converged when max(forces) < fmax.
    fexit : float
        Exit criteria, exit when max(forces) > fexit.
    device : Literal["cpu", "cuda"]
        The device to use for energy calculation.
    uma_energy_units: Literal["eV", "Hartrees", "kcal/mol"]
        The units output from the energy calculation.
    task_name: Literal["omol", "omat", "oc20", "odac", "omc"]
        The task name to use for the UMA model.

    Returns
    -------
    energies, mols : dict[str, dict[str, float]], dict[str, Chem.Mol]
        energies is a dict of final energy of each molecular conformer in kcal/mol (i.e. 0 = converged).
        mols contains the dictionary of molecules with the conformers minimised.

        energies = {
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
    # predictor = pretrained_mlip.get_predict_unit(model_paths, device=device)
    predictor = load_predict_unit(
        path=model_paths,
        inference_settings="default",
        overrides=None,
        device=device,
    )
    calculator = FAIRChemCalculator(predictor, task_name=task_name)
    
    energies, mols = method_min(mols, calculator, maxIters, fmax, fexit, conversion_factor)

    return energies, mols 