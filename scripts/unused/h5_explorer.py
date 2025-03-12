import h5py
from ase import Atoms
import os
import numpy as np

# def read_h5_file(file_path):
#     atoms_list = []
#     with h5py.File(file_path, 'r') as f:
#         config_group = f['config_batch_0']
#         print(f.keys())
#         for config_name in config_group:
#             config = config_group[config_name]
#             positions = config['positions'][:]  # N x 3 array
#             atomic_numbers = config['atomic_numbers'][:]  # Atomic numbers
#             cell = config['cell'][:]  # 3 x 3 cell matrix
#             pbc = config['pbc'][:]  # Periodic boundary conditions (3 elements, e.g., [True, True, True])

#             # Create an ASE Atoms object
#             atoms = Atoms(numbers=atomic_numbers, positions=positions, cell=cell, pbc=pbc)
#             atoms.info['corrected_total_energy'] = config['energy'][()]
#             atoms.info['forces'] = config['forces'][:]
#             atoms_list.append(atoms)

#     return atoms_list


def read_h5_file(file_path):
    atoms_list = []

    with h5py.File(file_path, "r") as f:
        # Load the number of atoms and frames
        n_frames = f["coordinates"].shape[0]
        n_atoms = f["coordinates"].shape[1]

        # Load coordinates (positions) - shape: (n_frames, n_atoms, 3)
        coordinates = np.array(f["coordinates"])

        # Load atomic numbers - shape: (n_atoms,)
        atomic_numbers = np.array(f["topology/atomic_numbers"])

        # Load unit cell vectors (optional) - shape: (n_frames, 3, 3)
        if "unitcell_vectors" in f:
            cell = np.array(f["unitcell_vectors"])
        else:
            cell = None

        # Iterate over frames and create ASE atoms objects
        for frame_idx in range(n_frames):
            pos = coordinates[frame_idx]  # (n_atoms, 3) for the current frame
            if cell is not None:
                frame_cell = cell[frame_idx]  # (3, 3) for the current frame
                atoms = Atoms(
                    numbers=atomic_numbers, positions=pos, cell=frame_cell, pbc=True
                )
            else:
                atoms = Atoms(numbers=atomic_numbers, positions=pos)

            atoms_list.append(atoms)

    return atoms_list


def aggregate_atoms_from_folder(folder_path):
    atoms_list = []
    # Loop over all .h5 files in the directory
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".h5"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_path}")
            atoms_list.extend(
                read_h5_file(file_path)
            )  # Append atoms from the current file

    return atoms_list


# Example usage
folder_path = "/data/shared/MPtrj/mace_val/mptrj-gga-ggapu-val"
atoms_list = aggregate_atoms_from_folder(folder_path)

# Now you have an aggregated atoms_list with all the molecules
print(f"Total number of configurations: {len(atoms_list)}")
# Optional: Do something with the atoms objects
for atoms in atoms_list:
    print(atoms)
