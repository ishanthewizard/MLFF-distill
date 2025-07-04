"""
Strain Energy Runner for fairchem CLI integration.

This module provides a custom runner that integrates strain energy calculations
with the fairchem CLI infrastructure, allowing the use of Slurm scheduling,
logging, and other fairchem features.
"""

import sys
from pathlib import Path
from typing import Any

# Add the src_v2 directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from copy import deepcopy
from timeit import default_timer as timer

import pandas as pd
from loguru import logger as logging
from omegaconf import DictConfig

from strain_relief.conformers import generate_conformers
from strain_relief.io import load_parquet, save_parquet, to_mols_dict
from ligand_strain_energy.utils._minimisation import minimise_conformers
from ligand_strain_energy.utils._energy_eval import predict_energy

# Import fairchem components
try:
    from fairchem.core.components.runner import Runner
    from fairchem.core.components.utils import ManagedAttribute
except ImportError:
    # Fallback for when fairchem is not available
    from abc import ABCMeta, abstractmethod
    
    class ManagedAttribute:
        def __init__(self, enforced_type=None):
            self.enforced_type = enforced_type
    
    class Runner(metaclass=ABCMeta):
        job_config = ManagedAttribute(enforced_type=DictConfig)
        
        @abstractmethod
        def run(self) -> Any:
            raise NotImplementedError
        
        @abstractmethod
        def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
            raise NotImplementedError
        
        @abstractmethod
        def load_state(self, checkpoint_location: str | None) -> None:
            raise NotImplementedError


class StrainEnergyRunner(Runner):
    """Runner for strain energy calculations using fairchem CLI infrastructure.
    
    This runner integrates strain energy calculations with the fairchem CLI,
    allowing the use of Slurm scheduling, logging, and other fairchem features.
    """
    
    job_config = ManagedAttribute(enforced_type=DictConfig)
    
    def __init__(self, config: DictConfig):
        """Initialize the strain energy runner.
        
        Parameters
        ----------
        config : DictConfig
            The configuration object containing all parameters for strain energy calculation.
        """
        self.config = config
        self.results = None
        
    def run(self) -> Any:
        """Run the strain energy calculation.
        
        Returns
        -------
        Any
            The results of the strain energy calculation.
        """
        start = timer()
        logging.info(f"STRAIN RELIEF RUN CONFIG: \n{self.config}")
        
        # Check if model path is required
        if (
            self.config.local_min.method == "MACE"
            or self.config.global_min.method == "MACE"
            or self.config.energy_eval.method == "MACE"
        ) and self.config.model.model_paths is None:
            raise ValueError("Model path must be provided if using MACE")
        
        # Load data
        df = load_parquet(**self.config.io.input)
        docked = to_mols_dict(df, self.config.io.input.mol_col_name, self.config.io.input.id_col_name)
        local_minima = {id: deepcopy(mol) for id, mol in docked.items()}
        global_minimia = {id: deepcopy(mol) for id, mol in docked.items()}
        
        # Find the local minimum using a looser convergence criteria
        logging.info("Minimising docked conformer...")
        local_minima = minimise_conformers(local_minima, **self.config.local_min)
        
        # Generate conformers from the docked conformer
        global_minimia = generate_conformers(global_minimia, **self.config.conformers)
        
        # Find approximate global minimum from generated conformers
        logging.info("Minimising generated conformers...")
        global_minimia = minimise_conformers(global_minimia, **self.config.global_min)
        
        # Predict single point energies (if using a different method from minimisation)
        if (
            self.config.local_min.method != self.config.energy_eval.method
            or self.config.global_min.method != self.config.energy_eval.method
        ):
            logging.info("Predicting energies of local minima poses...")
            local_minima = predict_energy(local_minima, **self.config.energy_eval)
            logging.info("Predicting energies of generated conformers...")
            global_minimia = predict_energy(global_minimia, **self.config.energy_eval)
        
        # Save torsional strains
        results = save_parquet(
            df, docked, local_minima, global_minimia, 
            self.config.threshold, **self.config.io.output
        )
        
        end = timer()
        logging.info(f"Ligand strain calculations took {end - start:.2f} seconds. \n")
        
        self.results = results
        return results
    
    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        """Save the current state of the runner.
        
        Parameters
        ----------
        checkpoint_location : str
            Path where to save the checkpoint.
        is_preemption : bool, optional
            Whether this is a preemption checkpoint, by default False.
            
        Returns
        -------
        bool
            True if save was successful.
        """
        # For strain energy calculations, we don't need complex checkpointing
        # since the calculation is typically fast and can be restarted
        logging.info(f"Saving checkpoint to {checkpoint_location}")
        return True
    
    def load_state(self, checkpoint_location: str | None) -> None:
        """Load the state of the runner from a checkpoint.
        
        Parameters
        ----------
        checkpoint_location : str | None
            Path to the checkpoint file.
        """
        # For strain energy calculations, we don't need complex state loading
        # since the calculation can be restarted from the beginning
        if checkpoint_location:
            logging.info(f"Loading checkpoint from {checkpoint_location}")
        else:
            logging.info("No checkpoint provided, starting fresh")


def create_strain_energy_runner(config: DictConfig) -> StrainEnergyRunner:
    """Factory function to create a strain energy runner.
    
    Parameters
    ----------
    config : DictConfig
        The configuration object.
        
    Returns
    -------
    StrainEnergyRunner
        The initialized strain energy runner.
    """
    return StrainEnergyRunner(config) 