import os
from fairchem.core import OCPCalculator
from fairchem.core.datasets import LmdbDataset
from ase.md.langevin import Langevin
from ase.md.logger import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import Atoms, units
from fairchem.core.common.relaxation.ase_utils import data_to_atoms
from fairchem.core.common.utils import (
    build_config,
    create_grid,
    new_trainer_context,
    save_experiment_log,
    setup_logging,
)

from fairchem.core.common.flags import flags



if __name__ == "__main__":

    
    setup_logging()

    parser = flags.get_parser()
    parser.add_argument("--nersc", action="store_true", help="Run with NERSC")
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)
    if args.timestamp_id is not None and len(args.identifier) == 0:
        args.identifier = args.timestamp_id

    data = LmdbDataset(config['dataset']['test'])

    init_data = data.__getitem__(config["init_data_idx"])
    atoms = data_to_atoms(init_data)

    # Set up the OCP calculator
    checkpoint_path = os.path.join("checkpoints", config["run_name"], "best_checkpoint.pt")
    calc = OCPCalculator(
        config_yml=args.config_yml.__str__(),
        checkpoint_path=checkpoint_path,
        cpu=False,
        seed=args.seed,
    )

    atoms.calc = calc

    # Set up Langevin dynamics
    MaxwellBoltzmannDistribution(atoms, temperature_K=config["integrator_config"]["temperature"])
    dyn = Langevin(
        atoms,
        timestep=config["integrator_config"]["timestep"] * units.fs,
        temperature_K=config["integrator_config"]["temperature"],
        friction=config["integrator_config"]["friction"] / units.fs,
        trajectory=os.path.join(os.path.dirname(checkpoint_path), "md.traj"),
    )
    dyn.attach(
        MDLogger(
            dyn, atoms, os.path.join(os.path.dirname(checkpoint_path), "md.log"), header=True, stress=False, peratom=True, mode="a"
        ),
        interval=config["save_freq"],
    )

    dyn.run(config["steps"])
    print("Done")
