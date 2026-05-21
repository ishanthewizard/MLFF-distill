import os
import MDAnalysis as mda

# Directory containing the trajectory files
traj_dir = "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/OPLS_diffusivities/myfolder"

# List all files in the trajectory directory
files = os.listdir(traj_dir)

# Find all .xtc files (trajectories)
xtc_files = [f for f in files if f.endswith('.xtc')]

# (Assume there is a single topology file, e.g., .gro, .top, .tpr)
top_files = [f for f in files if f.endswith('.gro') or f.endswith('.tpr') or f.endswith('.top')]
if len(top_files) == 0:
    raise FileNotFoundError("No topology (.gro, .tpr, .top) file found in the directory")
topology_file = os.path.join(traj_dir, top_files[0])

# The variable containing trajectory file names is 'xtc_files'
# Load the three trajectory Universes
universes = []
if len(xtc_files) < 3:
    raise ValueError("Fewer than 3 trajectory (.xtc) files found in the directory.")
xtc_files = sorted(xtc_files)[:3]  # Take the first three, sorted alphabetically
for xtc in xtc_files:
    xtc_path = os.path.join(traj_dir, xtc)
    u = mda.Universe(topology_file, xtc_path)
    universes.append(u)

# Example: Select all atoms from each universe (you can change the selection as needed)
all_atoms_list = [u.select_atoms('all') for u in universes]

# The variable 'xtc_files' holds the list of trajectory (.xtc) files.


# How to get a desired frame (e.g. frame index 5) from the trajectory
u = universes[0]  # select the first universe/trajectory

desired_frame = 5  # for example, get the 6th frame (0-based indexing)

u.trajectory[desired_frame]  # moves to the desired frame

positions = all_atoms_list[0].positions  # positions of all atoms at desired frame
print(positions)

# You can now use 'positions' as needed for the selected frame.