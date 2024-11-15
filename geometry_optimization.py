import os
import torch
import logging
import numpy as np
from pathlib import Path
from fairchem.core import OCPCalculator
from fairchem.core.datasets import LmdbDataset
from ase.optimize import BFGS, BFGSLineSearch, LBFGS, LBFGSLineSearch, FIRE
from ase.md.logger import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase import Atoms, units
import ase.io
from fairchem.core.common.relaxation.ase_utils import data_to_atoms
from fairchem.core.common.utils import (
    build_config,
    create_grid,
    new_trainer_context,
    save_experiment_log,
    setup_logging,
    pyg2_data_transform,
)

from fairchem.core.common.flags import flags

from mace.calculators import mace_off, mace_mp


if __name__ == "__main__":

    setup_logging()

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)
    if args.timestamp_id is not None and len(args.identifier) == 0:
        args.identifier = args.timestamp_id

    data = LmdbDataset(config["dataset"]["test"])

    start_idx = config["start_idx"]
    end_idx = config["end_idx"]
    if end_idx == -1:
        end_idx = len(data)

    

    if "mace" in config.keys():
        calc = mace_mp(model="large", dispersion=False, default_dtype="float32", device='cuda')
        traj_path = "data/mace_off_geom_opt"
        os.makedirs(traj_path, exist_ok=True)
    else:
        # Set up the OCP calculator
        checkpoint_path = os.path.join(
            config["MODELPATH"], config["run_name"], "best_checkpoint.pt"
        )
        traj_path = os.path.dirname(checkpoint_path)
        calc = OCPCalculator(
            config_yml=args.config_yml.__str__(),
            checkpoint_path=checkpoint_path,
            cpu=False,
            seed=args.seed,
        )

        

    for idx in range(start_idx, end_idx):
        init_data = data.__getitem__(idx)
        if "cell" not in init_data:
            init_data["cell"] = 100 * torch.eye(3)
        atoms = data_to_atoms(init_data)

        atoms.calc = calc

        # Run geometry optimization
        
        dyn = FIRE(
            atoms, trajectory=os.path.join(traj_path, f"geom_opt_system{idx}.traj")
        )
        dyn.attach(
            MDLogger(
                dyn,
                atoms,
                os.path.join(traj_path, f"geom_opt_system{idx}.log"),
                header=True,
                stress=False,
                peratom=True,
                mode="a",
            ),
            interval=config["save_freq"],
        )

        dyn.run(steps=config["steps"], fmax=config["fmax"])

        print("Geometry optimization done.")

    
