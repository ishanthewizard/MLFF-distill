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

if TYPE_CHECKING:
    import argparse


if __name__ == "__main__":

    
    setup_logging()

    parser: argparse.ArgumentParser = flags.get_parser()
    parser.add_argument("--nersc", action="store_true", help="Run with NERSC")
    args: argparse.Namespace
    override_args: list[str]
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)
    if args.timestamp_id is not None and len(args.identifier) == 0:
        args.identifier = args.timestamp_id

    data = LmdbDataset(
        {"src": "/data/shared/ishan_stuff/spice_separated/Solvated_Amino_Acids/test"}
    )

    init_data = data.__getitem__(10)
    atoms = data_to_atoms(init_data)

    # Set up the OCP calculator
    checkpoint_path = "checkpoints/2024-09-18-20-05-36-solvated-gemSmall-DIST-n2n-correctemb/best_checkpoint.pt"
    calc = OCPCalculator(
        config_yml="configs/SPICE/solvated_amino_acids/gemnet-dT-small.yml",
        checkpoint_path=checkpoint_path,
        cpu=False,
        seed=0,
    )

    atoms.calc = calc

    # Set up Langevin dynamics
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)
    dyn = Langevin(
        atoms,
        timestep=1 * units.fs,
        temperature_K=300 * units.kB,
        friction=0.01 / units.fs,
        trajectory=os.path.join(os.path.dirname(checkpoint_path), "md.traj"),
    )
    dyn.attach(
        MDLogger(
            dyn, atoms, os.path.join(os.path.dirname(checkpoint_path), "md.log"), header=True, stress=False, peratom=True, mode="a"
        ),
        interval=1000,
    )

    dyn.run(100000)
    print("Done")
