# Adapted from https://github.com/kyonofx/MDsim/blob/main/observable.ipynb

import os
import ase
import torch
import glob
import math
from copy import deepcopy
from tqdm import tqdm
import numpy as np
from ase.io import Trajectory
from ase.visualize import view
import nglview as nv
import ipywidgets
from ase.neighborlist import natural_cutoffs, NeighborList
from fairchem.core.datasets import LmdbDataset
from fairchem.core import OCPCalculator
from fairchem.core.common.flags import flags
from fairchem.core.common.utils import (
    build_config,
    create_grid,
    new_trainer_context,
    save_experiment_log,
    setup_logging,
    pyg2_data_transform,
)
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
import matplotlib.pylab as pylab
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scienceplots
from math import floor, log10
from matplotlib import rc
from mace.calculators import mace_off
plt.style.use(['science','no-latex'])

params = {'figure.dpi': 900,
            'axes.labelsize': 'small',
          'legend.fontsize': 'x-small',
         'axes.titlesize':'medium',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'font.family': 'DejaVu Sans'}

pylab.rcParams.update(params)

CYAN = "#00FFFF"

ORANGE = '#ffad33'
PINK = "#FF81C0"
MAGENTA = "#FF00FF"
YELLOW = "#FAC205"
SILVER = "#C5C9C7"
GRAY = "#929591"
DARK_GRAY = "#5b5b5b"
BLUE = "#069AF3"
DARK_BLUE = "#0000FF"
RBLUE = "#0088cc"
RPURPLE = "#5c12de"
RYELLOW = "#dead12"


def distance_pbc(x0, x1, lattices):
    delta = torch.abs(x0 - x1)
    lattices = lattices.view(-1,1,3)
    delta = torch.where(delta > 0.5 * lattices, delta - lattices, delta)
    return torch.sqrt((delta ** 2).sum(dim=-1))

def dir_to_legend_name_and_color(md_dir):
    name = ""
    color_list = [ORANGE, RBLUE, ORANGE, RBLUE, DARK_GRAY]
    if "gemT" in md_dir:
        name += "GemNet-T - Gradient-Based Forces"
        idx = 0
    elif "gem" in md_dir:
        name += "GemNet-dT - Direct Forces"
        idx = 2
    elif "jmp" in md_dir:
        name += "JMP-L - Direct Forces (Teacher)"
        idx = 4
    
    if "undistilled" in md_dir.lower():
        name += ", Undistilled"
    elif "distilled" in md_dir.lower():
        name += ", Distilled"
        idx +=1

    return name, color_list[idx]

def get_thermo(filename):
    """
    read thermo logs.
    """
    with open(filename, 'r') as f:
        thermo = f.read().splitlines()
        sim_time, Et, Ep, Ek, T = [], [], [], [], []
        for i in range(1, len(thermo)):
            try:
                t, Etot, Epot, Ekin, Temp = [float(x) for x in thermo[i].split(' ') if x]
                sim_time.append(t)
                Et.append(Etot)
                Ep.append(Epot)
                Ek.append(Ekin)
                T.append(Temp)
            except:
                sim_time, Et, Ep, Ek, T = [], [], [], [], []
    thermo = {
        'time': sim_time,
        'Et': Et,
        'Ep': Ep,
        'Ek': Ek,
        'T': T
    }
    return thermo
    
def plot_stability(data_split, checkpoint_dir, md_dirs, ref_calc, append_name = ""):
    stability_threshold = 0.5
    data_dir = "/data/shared/ishan_stuff/md22/lmdb"
    if append_name != "":
        append_name = "_" + append_name
    # First figure: Bond length deviation over time
    plt.figure()

    # Remove axis ticks from the top and right borders
    plt.tick_params(top = False)
    plt.tick_params(right = False)

    # Reduce font size of the axis number labels (tick labels)
    plt.tick_params(axis='both', which='major', labelsize=10)  # Adjust size as needed

    
    plt.xlabel("Simulation time (ps)", fontsize=10)
    plt.xlim(0, 105)
    plt.ylabel("Max bond length deviation (A)", fontsize=10)
    plt.yscale('log')
    plt.axhline(stability_threshold, linestyle="dashed", c='r', label="Stability Threshold")
    
    

    # teacher_calc = mace_off(model="large", dispersion=False, default_dtype="float32", device='cuda')

    # Prepare lists to hold ref energies for the second figure
    all_ref_energies = []
    all_energies = []
    labels = []
    colors=  []

    for md_dir in md_dirs:

        # load all system trajectories and logs for this model
        trajectories = glob.glob(os.path.join(checkpoint_dir, md_dir, "*.traj"))
        logs = glob.glob(os.path.join(checkpoint_dir, md_dir, "*.log"))

        for j, (traj, log) in enumerate(zip(trajectories, logs)):
            traj = Trajectory(traj)
            # get bonds from simulated trajectory
            atoms = traj[0]
            NL = NeighborList(natural_cutoffs(deepcopy(atoms)), self_interaction=False)
            NL.update(atoms)
            bonds = NL.get_connectivity_matrix().todense().nonzero()
            bonds = torch.tensor(bonds)

            # get bonds from GT trajectory
            data = LmdbDataset({"src": os.path.join(data_dir, data_split, "train")})
            gt_traj = []
            
                
            for idx in range(len(data)):
                dat = data.__getitem__(idx)
                if "md22" not in md_dir:
                    if dat.atomic_numbers.shape[0] == atoms.numbers.shape[0]:
                        if np.equal(np.array(dat.atomic_numbers), atoms.numbers).all():
                            gt_traj.append(dat.pos)
                else:
                    gt_traj.append(dat.pos) # all md22 data for a given molecule has the same number of atoms

            gt_traj = torch.stack(gt_traj)
            bond_lens = distance_pbc(
                gt_traj[:, bonds[0]], gt_traj[:, bonds[1]], torch.FloatTensor([100., 100., 100.])
            )
            mean_bond_lens = bond_lens.mean(dim=0)

            # remove bonds larger than 5 A
            bonds = bonds[:, mean_bond_lens < 5]
            mean_bond_lens = mean_bond_lens[mean_bond_lens < 5]

            run = {}
            # process trajectory
            x_traj = [x.positions for x in traj]
            run['traj'] = torch.from_numpy(np.stack(x_traj))
    
            # load thermo log
            run['thermo'] = get_thermo(log)
            T = np.array(run['thermo']['T'])
            collapse_pt = len(T)
            md_time = np.array(run['thermo']['time'])
    
            subsample = int(len(traj) / collapse_pt)
            run['traj'] = run['traj'][::subsample]
            
            # Track stability and energies
            devs = [0.01]
            collapsed = False
            ref_energies = []
            for i in tqdm(range(1, len(T))):
                atom = traj[subsample * i]
                atom.calc = ref_calc
                energy = atom.get_potential_energy() + atom.get_kinetic_energy()
                if i ==1:
                    initial_energy = energy
                ref_energies.append(abs(energy - initial_energy) + 1e-4)
    
                bond_lens = distance_pbc(
                    run['traj'][(i-1):i, bonds[0]], run['traj'][(i-1):i, bonds[1]], torch.FloatTensor([30., 30., 30.])
                )
                max_dev = (bond_lens[0] - mean_bond_lens).abs().max()
                max_dev = torch.nan_to_num(max_dev, 10000)
                devs.append(max_dev)
                if max_dev > stability_threshold and not collapsed:
                    collapse_pt = i+1
                    collapsed = True
                    
            run['collapse_pt'] = collapse_pt
            run['devs'] = devs[:collapse_pt]
            ref_energies = torch.tensor(np.array(ref_energies))
            all_ref_energies.append(ref_energies[:collapse_pt])
            energies = np.array(run['thermo']['Et'][:collapse_pt])
            all_energies.append(abs(energies - energies[0]) + 1e-4)

            label, color = dir_to_legend_name_and_color(md_dir)
            labels.append(label if j==0 else None)
            colors.append(color)
            
            # Plot bond length deviation
            plt.plot(run['devs'], label=label if j==0 else None, color=color)
            # Plot an 'x' at the collapse point
            plt.plot(len(run['devs']) - 1, run['devs'][-1], 'x', color=color)
            

    plt.legend(frameon=True, edgecolor='black')
    plt.savefig(f"stability{append_name}.png")
    plt.show()
    plt.close()

    # Second figure: energies over time
    plt.figure()

    # Remove axis ticks from the top and right borders
    plt.tick_params(top = False)
    plt.tick_params(right = False)

    # Reduce font size of the axis number labels (tick labels)
    plt.tick_params(axis='both', which='major', labelsize=10)  # Adjust size as needed


    plt.xlabel("Simulation time (ps)", fontsize=10)
    plt.xlim(0, 105)
    plt.ylabel("Change in Energy (eV)", fontsize=9)
    # plt.yscale("log")

    for energies, label, color in zip(all_ref_energies, labels, colors):
        plt.plot(energies, label=label, color = color)
        # Plot an 'x' at the collapse point
        plt.plot(len(energies) - 1, energies[-1], 'x', color=color)
    
    plt.legend(frameon=True, edgecolor='black')
    plt.savefig(f"energies{append_name}.png")
    plt.show()
    plt.close()


if __name__ == "__main__":

    
    setup_logging()

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)

    # Set up the OCP calculator
    calc_checkpoint_path = os.path.join(config["MODELPATH"], config["run_name"], "best_checkpoint.pt")
    calc = OCPCalculator(
        config_yml=args.config_yml.__str__(),
        checkpoint_path=calc_checkpoint_path,
        cpu=False,
        seed=args.seed,
    )

    
    checkpoint_path = "/data/shared/ishan_stuff"
    data_split = "buckyball-catcher"


    md_dirs = ["student_checkpoints/undistilled/md22/2024-11-07-11-56-48-gemSmall-buckyball", \
                "student_checkpoints/distilled/md22/2024-11-08-12-13-52-gemSmall-buckyball-dist-jmplarge", \
                "jmp_l_sims/buckyball-catcher/energyforce"
            ]

    plot_stability(data_split, checkpoint_path, md_dirs, calc, "gem_buckyball")

    md_dirs = ["student_checkpoints/undistilled/md22/2024-11-08-17-57-20-gemTSmall-buckyball-correctenergynorm", \
                "student_checkpoints/distilled/md22/2024-11-08-17-57-20-gemTSmall-buckyball-dist-jmplarge-correctenergynorm", \
                "jmp_l_sims/buckyball-catcher/energyforce"]

    plot_stability(data_split, checkpoint_path, md_dirs, calc, "gemT_buckyball")

    




