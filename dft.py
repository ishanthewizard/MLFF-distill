import os
import psi4
from tqdm import tqdm
import pickle
import argparse
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom
import ase, ase.io
from ase.units import Bohr, Hartree


def atom_to_geom(atom):
    symbols = atom.get_chemical_symbols()
    pos = atom.get_positions()
    lines = ['0 1']
    for i in range(len(pos)):
        lines.append(f'{symbols[i]} {pos[i][0]} {pos[i][1]} {pos[i][2]}')
    mol = psi4.geometry('\n'.join(lines))
    return mol

def smiles_to_geom(mol, positions):
    lines = ['0 1']
    for i, atom in enumerate(mol.GetAtoms()):
        pos = positions[i]
        lines.append(f'{atom.GetSymbol()} {pos[0]} {pos[1]} {pos[2]}')
    return psi4.geometry('\n'.join(lines))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_threads", type=int, default=32)
    parser.add_argument("--memory", type=int, default=1000)
    parser.add_argument("--mace", action='store_true')
    parser.add_argument("--calculate_every", type=int, default=1)
    parser.add_argument("--only_last_frame", action='store_true')
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=10)
    parser.add_argument("--modelpath", type=str, default='/data/shared/ishan_stuff/student_checkpoints/spice')
    parser.add_argument("--run_name", type=str, default='2024-09-05-09-42-40-amino-gemSmall')

    args = parser.parse_args()

    
    if args.mace:
        traj_path = 'data/mace_off_geom_opt'
    else:
        traj_path = os.path.join(args.modelpath, args.run_name)

    # Evaluate the final energy using PSI4
    num_threads = args.num_threads
    memory = args.memory
    calculate_every = args.calculate_every
    start_idx = args.start_idx
    end_idx = args.end_idx

    final_energies = []
    final_force_norms = []

    for idx in range(start_idx, end_idx):
        filename = os.path.join(traj_path, f"geom_opt_system{idx}.traj")
        print(f'Loading trajectory from {filename}...')
        molecules = ase.io.read(filename, ':')
        
        if args.only_last_frame:
            print('Only calculating the last frame.')
            molecules = [molecules[-1]]
        else:
            print(f'Calculating every {calculate_every} steps.')
            molecules = molecules[::calculate_every] + [molecules[-1]]


        engs = []
        forces = []

        psi4.set_options({'reference': 'uhf'})
        psi4.core.set_num_threads(num_threads)
        psi4.set_memory(f'{memory}MB')
        method = 'wB97M-D3BJ'
        basis = 'def2-TZVPPD'
        for i, mol in enumerate(tqdm(molecules)):
            
            geom = atom_to_geom(mol)
            grad, wf = psi4.driver.gradient('{}/{}'.format(method, basis), return_wfn=True, molecule=geom)
            energy = wf.energy()
            energy = energy * Hartree
            # convert to eV/A
            # also note that the gradient is -1 * forces
            force = -1 * np.array(grad) * Hartree / Bohr
            forces.append(force)
            engs.append(energy)

        # save to numpy array
        np.save(os.path.join(traj_path, f'dft_energies_system{idx}.npy'), np.array(engs))
        np.save(os.path.join(traj_path, f'dft_forces_system{idx}.npy'), np.array(forces))
            
        
        final_energies.append(energy)
        final_force_norms.append(np.mean(np.linalg.norm(force, axis = -1)))
        # print(f"Final energy={energy} eV")
        # print("Final force norm=", np.mean(np.linalg.norm(force, axis = -1)), " eV/A")
    

    print(f"Mean final energy (eV): {np.mean(final_energies)}")
    print(f"Mean final force norm (eV/A): {np.mean(final_force_norms)}")
  
    # save to numpy 
    np.save(os.path.join(traj_path, 'final_dft_energies.npy'), np.array(final_energies))
    np.save(os.path.join(traj_path, 'final_dft_force_norms.npy'), np.array(final_force_norms))
