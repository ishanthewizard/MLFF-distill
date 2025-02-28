import os
from fairchem.core import OCPCalculator
from fairchem.core.datasets import LmdbDataset
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.md.logger import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase import Atoms, units
from fairchem.core.common.utils import (
    build_config,
    create_grid,
    new_trainer_context,
    save_experiment_log,
    setup_logging,
)

from fairchem.core.common.flags import flags

from fairchem.core.common.relaxation.ase_utils import batch_to_atoms
from fairchem.core.common.data_parallel import OCPCollater
import torch
from tqdm import tqdm

if __name__ == "__main__":

    
    setup_logging()

    parser = flags.get_parser()
    parser.add_argument("--nersc", action="store_true", help="Run with NERSC")
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)
    if args.timestamp_id is not None and len(args.identifier) == 0:
        args.identifier = args.timestamp_id

    data = LmdbDataset(config['dataset']['test'])

    idx = config["init_data_idx"]
    init_data = data.__getitem__(idx)
    collater = OCPCollater()
    init_data.tags = torch.zeros(init_data.natoms, device=init_data.pos.device)
    atoms = batch_to_atoms(collater([init_data]))[0]

    # Set up the OCP calculator
    output_folder = config['output_folder']
    checkpoint_path = config['checkpoint']
    calc = OCPCalculator(
        config_yml=args.config_yml.__str__(),
        checkpoint_path=checkpoint_path,
        cpu=False,
        seed=args.seed,
    )
    # calc = mace_mp(model="large", dispersion=False, default_dtype="float32", device='cuda')

    atoms.calc = calc

    # Initialize atom velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=config["integrator_config"]["temperature"])
    Stationary(atoms)  # zero the center of mass velocity


    if config["nvt"]:
        dyn = Langevin(
            atoms,
            timestep=config["integrator_config"]["timestep"] * units.fs,
            temperature_K=config["integrator_config"]["temperature"],
            friction=config["integrator_config"]["friction"] / units.fs,
            # trajectory=os.path.join(os.path.dirname(output_folder), f"equilibration{idx}.traj"),
        )
    else:
        """Running Velocity VERLET"""
        dyn = VelocityVerlet(
            atoms,
            timestep=config["integrator_config"]["timestep"] * units.fs,
            trajectory=os.path.join(os.path.dirname(output_folder), f"md_system{idx}.traj"),
        )

        # Initialize the progress bar
    steps = config["steps"]
    pbar = tqdm(total=steps, desc="Molecular Dynamics Progress", unit="step")

    # Define a callback function to update the progress bar
    def update_pbar(atoms=atoms):
        pbar.update(1)

    dyn.attach(
            MDLogger(
                dyn, atoms, os.path.join(os.path.dirname(output_folder), f"md_system{idx}.log"), header=True, stress=False, peratom=True, mode="a"
            ),
            interval=config["save_freq"],
        )
    
    dyn.attach(update_pbar, interval=1)

    dyn.run(config["steps"])
    print("Done")
