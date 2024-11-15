import os
import torch
import logging
logging.basicConfig(level=logging.ERROR)
import numpy as np
from pathlib import Path
from fairchem.core import OCPCalculator
from fairchem.core.datasets import LmdbDataset
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.md.logger import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase import Atoms, units
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

# JMP stuff
from jmp.lightning.data.balanced_batch_sampler import BalancedBatchSampler
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch, Data
from jmp.configs.pretrain.jmp_l import jmp_l_pt_config_
from jmp.datasets.pretrain_lmdb import PretrainLmdbDataset, PretrainDatasetConfig
from jmp.tasks.pretrain import PretrainConfig, PretrainModel
from jmp.tasks.pretrain.module import (
    NormalizationConfig,
    PretrainDatasetConfig,
    TaskConfig,
)
from jmp.configs.finetune.jmp_l import jmp_l_ft_config_
from jmp.configs.finetune.jmp_s import jmp_s_ft_config_
from jmp.configs.finetune.md22 import jmp_l_md22_config_
from jmp.tasks.finetune.base import FinetuneConfigBase, FinetuneModelBase
from jmp.tasks.finetune.md22 import MD22Config, MD22Model


def atoms_to_batch(atoms, collate_fn):
    # convert list of atoms to a torch.geometric.data.Batch object
    data = []
    for atom in atoms:
        atomic_numbers = torch.Tensor(atom.get_atomic_numbers()).to(torch.long)
        positions = torch.Tensor(atom.get_positions())
        cell = torch.Tensor(np.array(atom.get_cell())).view(1, 3, 3)
        natoms = positions.shape[0]

        data.append(
            Data(
                cell=cell,
                pos=positions,
                atomic_numbers=atomic_numbers,
                natoms=natoms,
                tags = 2*torch.ones(natoms)
            )
        )
    return collate_fn(data)


if __name__ == "__main__":

    
    setup_logging()

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)
    if args.timestamp_id is not None and len(args.identifier) == 0:
        args.identifier = args.timestamp_id

    data = LmdbDataset(config['dataset']['val'])

    idx = config["init_data_idx"]
    init_data = data.__getitem__(idx)
    if 'cell' not in init_data:
        init_data['cell'] = 100 * torch.eye(3)
    atoms = data_to_atoms(init_data)

    # Set up the OCP calculator
    checkpoint_path = os.path.join(config["MODELPATH"], config["run_name"], "best_checkpoint.pt")
    calc = OCPCalculator(
        config_yml=args.config_yml.__str__(),
        checkpoint_path=checkpoint_path,
        cpu=False,
        seed=args.seed,
    )
    # calc = mace_mp(model="large", dispersion=False, default_dtype="float32", device='cuda')

    atoms.calc = calc

    if "jmp" in config.keys():
        jmp = True
        jmp_args = config["jmp"]
        if "exp_name" not in jmp_args.keys():
            jmp_args["exp_name"] = ""

        jmp_config = MD22Config.draft()
        
        if "-l" in jmp_args["checkpoint_path"]:
            jmp_l_ft_config_(jmp_config, jmp_args["checkpoint_path"], disable_force_output_heads = not jmp_args["direct_forces"])
        else:
            jmp_s_ft_config_(jmp_config, jmp_args["checkpoint_path"], disable_force_output_heads = not jmp_args["direct_forces"])
        
        # This loads the MD22-specific configuration
        jmp_l_md22_config_(jmp_config, jmp_args["molecule"], Path(jmp_args["data_path"]), direct_forces=jmp_args["direct_forces"])
        jmp_config = jmp_config.finalize()  # Actually construct the config object
        model = MD22Model(jmp_config)

        ckpt_path = Path(jmp_args["checkpoint_path"])
        
        # Load the checkpoint
        state_dict = torch.load(ckpt_path)
        sd = state_dict["state_dict"].copy()
        new_sd = {}
        # Map key to the right spot since only doing trans1x for now
        for k in sd:
            # TODO: make sure that this is the transition 1x head
            if "output.out_energy.3" in k:
                new_k = k.replace("output.out_energy.3", "output.out_energy.0")
                new_sd[new_k] = sd[k]
            elif "output.out_forces.3" in k:
                new_k = k.replace("output.out_forces.3", "output.out_forces.0")
                new_sd[new_k] = sd[k]
            elif "output.out_forces" in k or "output.out_energy" in k or "task_steps" in k:
                pass
            else:
                new_sd[k] = sd[k]
        
        model.load_state_dict(new_sd, strict=True)
        model.to("cuda")
        import pdb; pdb.set_trace()

        # get normalization constants
        mean_force = jmp_config.normalization['force'].mean
        std_force = jmp_config.normalization['force'].std

        mean_energy = jmp_config.normalization['y'].mean
        std_energy = jmp_config.normalization['y'].std

        
        def get_forces(atoms):
            batch = atoms_to_batch([atoms], collate_fn=model.collate_fn)
            batch = batch.to("cuda")
            with torch.no_grad():
                forces = model(batch)['force']
            # Un-normalize the forces
            forces = forces * std_force + mean_force
            return forces.detach().cpu().numpy()
        
        def get_potential_energy(atoms):
            batch = atoms_to_batch([atoms], collate_fn=model.collate_fn)
            batch = batch.to("cuda")
            with torch.no_grad():
                energy = model(batch)['y']
            
            # Un-normalize the energies
            energy = energy * std_energy + mean_energy
            return energy.detach().cpu().numpy()

        # override the calculator functions with the custom JMP ones
        calc.get_potential_energy = get_potential_energy
        calc.get_forces = get_forces

    # Initialize atom velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=config["integrator_config"]["temperature"])
    Stationary(atoms)  # zero the center of mass velocity

    if not jmp:
        traj_path = os.path.dirname(checkpoint_path) 
    else:
        if "-l" in jmp_args["checkpoint_path"]:
            traj_path = os.path.join(os.path.dirname(jmp_args["checkpoint_path"]), "jmp_l_sims", jmp_args["molecule"], jmp_args["exp_name"])
        else:
            traj_path = os.path.join(os.path.dirname(jmp_args["checkpoint_path"]), "jmp_s_sims", jmp_args["molecule"], jmp_args["exp_name"])
    
    os.makedirs(traj_path, exist_ok=True)
            
    if config["nvt"]:
        dyn = Langevin(
            atoms,
            timestep=config["integrator_config"]["timestep"] * units.fs,
            temperature_K=config["integrator_config"]["temperature"],
            friction=config["integrator_config"]["friction"] / units.fs,
            trajectory=os.path.join(traj_path, f"equilibration{idx}.traj"),
        )
    else:
        """Velocity Verlet is becoming unstable for Solvated Amino Acids for some reason"""
        dyn = VelocityVerlet(
            atoms,
            timestep=config["integrator_config"]["timestep"] * units.fs,
            trajectory=os.path.join(traj_path, f"md_system{idx}.traj"),
        )
    

    dyn.attach(
            MDLogger(
                dyn, atoms, os.path.join(traj_path, f"md_system{idx}.log"), header=True, stress=False, peratom=True, mode="a"
            ),
            interval=config["save_freq"],
        )

    dyn.run(config["steps"])
    print("Done")
