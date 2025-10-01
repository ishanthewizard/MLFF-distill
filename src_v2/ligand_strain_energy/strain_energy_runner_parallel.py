"""
Strain Energy Runner with Data Parallelism for fairchem CLI integration.

This module provides a custom runner that integrates strain energy calculations
with the fairchem CLI infrastructure, supporting data parallelism for processing
large datasets across multiple processes/nodes.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List
import os

# Add the src_v2 directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from copy import deepcopy
from timeit import default_timer as timer
import pickle

import pandas as pd
import torch
import torch.distributed as dist
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
    from fairchem.core.common import distutils
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


class StrainEnergyRunnerParallel(Runner):
    """Runner for strain energy calculations with data parallelism support.
    
    This runner integrates strain energy calculations with the fairchem CLI,
    supporting data parallelism for processing large datasets across multiple
    processes and nodes.
    """
    
    job_config = ManagedAttribute(enforced_type=DictConfig)
    
    def __init__(self, config: DictConfig):
        """Initialize the strain energy runner with data parallelism.
        
        Parameters
        ----------
        config : DictConfig
            The configuration object containing all parameters for strain energy calculation.
        """
        self.config = config
        self.results = None
        self.is_distributed = dist.is_initialized() if dist.is_available() else False
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        
        # Setup distributed environment if needed
        if self.is_distributed:
            self._setup_distributed()
    
    def _setup_distributed(self):
        """Setup distributed computing environment."""
        if not dist.is_initialized():
            return
            
        # Set device based on local rank
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        
        logging.info(f"Distributed setup: rank={self.rank}, world_size={self.world_size}")
    
    def _split_data(self, data_dict: Dict) -> Dict:
        """Split data across processes for data parallelism.
        
        Parameters
        ----------
        data_dict : Dict
            Dictionary of molecules to split.
            
        Returns
        -------
        Dict
            Subset of data for current process.
        """
        if not self.is_distributed or self.world_size == 1:
            return data_dict
        
        # Convert to list for easier splitting
        items = list(data_dict.items())
        
        # Split data across processes
        chunk_size = len(items) // self.world_size
        start_idx = self.rank * chunk_size
        end_idx = start_idx + chunk_size if self.rank < self.world_size - 1 else len(items)
        
        # Handle remainder
        if self.rank == self.world_size - 1:
            end_idx = len(items)
        
        local_items = items[start_idx:end_idx]
        local_data = dict(local_items)
        
        logging.info(f"Rank {self.rank}: Processing {len(local_data)} molecules out of {len(data_dict)}")
        return local_data
    
    def _gather_results(self, local_results: pd.DataFrame) -> pd.DataFrame:
        """Gather results from all processes.
        
        Parameters
        ----------
        local_results : pd.DataFrame
            Results from current process.
            
        Returns
        -------
        pd.DataFrame
            Combined results from all processes.
        """
        if not self.is_distributed or self.world_size == 1:
            return local_results
        
        # Convert DataFrame to bytes for transmission
        local_bytes = pickle.dumps(local_results)
        
        # Gather all results to rank 0
        gathered_bytes = [None] * self.world_size
        dist.gather_object(local_bytes, gathered_bytes, dst=0)
        
        if self.rank == 0:
            # Combine all results
            all_results = []
            for bytes_data in gathered_bytes:
                if bytes_data is not None:
                    df = pickle.loads(bytes_data)
                    all_results.append(df)
            
            if all_results:
                combined_results = pd.concat(all_results, ignore_index=True)
                logging.info(f"Combined results from {len(all_results)} processes: {len(combined_results)} total rows")
                return combined_results
            else:
                return local_results
        else:
            return local_results
    
    def _save_results(self, results: pd.DataFrame, output_config: Dict):
        """Save results, handling distributed scenarios.
        
        Parameters
        ----------
        results : pd.DataFrame
            Results to save.
        output_config : Dict
            Output configuration.
        """
        if self.is_distributed and self.rank != 0:
            # Only rank 0 saves the final results
            return
        
        # Save results
        if output_config.get("output_file"):
            results.to_parquet(output_config["output_file"])
            logging.info(f"Results saved to {output_config['output_file']}")
    
    def run(self) -> Any:
        """Run the strain energy calculation with data parallelism.
        
        Returns
        -------
        Any
            The results of the strain energy calculation.
        """
        start = timer()
        logging.info(f"STRAIN RELIEF RUN CONFIG (Rank {self.rank}): \n{self.config}")
        
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
        
        # Split data across processes for data parallelism
        local_docked = self._split_data(docked)
        
        if len(local_docked) == 0:
            logging.warning(f"Rank {self.rank}: No data assigned, skipping processing")
            return None
        
        local_minima = {id: deepcopy(mol) for id, mol in local_docked.items()}
        global_minimia = {id: deepcopy(mol) for id, mol in local_docked.items()}
        
        # Find the local minimum using a looser convergence criteria
        logging.info(f"Rank {self.rank}: Minimising docked conformer...")
        local_minima = minimise_conformers(local_minima, **self.config.local_min)
        
        # Generate conformers from the docked conformer
        global_minimia = generate_conformers(global_minimia, **self.config.conformers)
        
        # Find approximate global minimum from generated conformers
        logging.info(f"Rank {self.rank}: Minimising generated conformers...")
        global_minimia = minimise_conformers(global_minimia, **self.config.global_min)
        
        # Predict single point energies (if using a different method from minimisation)
        if (
            self.config.local_min.method != self.config.energy_eval.method
            or self.config.global_min.method != self.config.energy_eval.method
        ):
            logging.info(f"Rank {self.rank}: Predicting energies of local minima poses...")
            local_minima = predict_energy(local_minima, **self.config.energy_eval)
            logging.info(f"Rank {self.rank}: Predicting energies of generated conformers...")
            global_minimia = predict_energy(global_minimia, **self.config.energy_eval)
        
        # Create local results DataFrame
        local_df = df[df[self.config.io.input.id_col_name].isin(local_docked.keys())].copy()
        
        # Save torsional strains for local data
        local_results = save_parquet(
            local_df, local_docked, local_minima, global_minimia, 
            self.config.threshold, **self.config.io.output
        )
        
        # Gather results from all processes
        combined_results = self._gather_results(local_results)
        
        # Save final results (only on rank 0)
        if combined_results is not None:
            self._save_results(combined_results, self.config.io.output)
        
        end = timer()
        logging.info(f"Rank {self.rank}: Ligand strain calculations took {end - start:.2f} seconds. \n")
        
        self.results = combined_results
        return combined_results
    
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
        logging.info(f"Rank {self.rank}: Saving checkpoint to {checkpoint_location}")
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
            logging.info(f"Rank {self.rank}: Loading checkpoint from {checkpoint_location}")
        else:
            logging.info(f"Rank {self.rank}: No checkpoint provided, starting fresh")


def create_strain_energy_runner_parallel(config: DictConfig) -> StrainEnergyRunnerParallel:
    """Factory function to create a strain energy runner with data parallelism.
    
    Parameters
    ----------
    config : DictConfig
        The configuration object.
        
    Returns
    -------
    StrainEnergyRunnerParallel
        The initialized strain energy runner with data parallelism.
    """
    return StrainEnergyRunnerParallel(config) 