import numpy as np
from ase.io import read
from ase.geometry import find_mic
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# List of .traj files
# traj_paths = [
#     "NPT_sims/md_omol_re3.traj",
#     "NPT_sims/md_omol_re5_small_new_1p1.traj",
#     "NPT_sims/md_omol_re5_medium_1p1.traj"
# ]
traj_paths = [
    "/data/ishan-amin/OMOL/electrolytes_application/npt_trajs_distillation/npt_trajs_napf6_dme_uma_omol/s1p1/md_omol_re5_small_1p1_wrapped.traj", 
    "/home/ishan-amin/MLFF-distill/APPLICATIONS/electrolytes/md_trajs/napf6_xxsmall.traj"
]

# RDF parameters
r_max = 10.0
bin_width = 0.1
bins = np.arange(0.0, r_max + bin_width, bin_width)
shell_volumes = (4/3)*np.pi * (bins[1:]**3 - bins[:-1]**3)
r_centers = 0.5 * (bins[1:] + bins[:-1])

# Colors and labels for plotting
colors = ["tab:red", "tab:blue", "tab:green"]
labels = ["UMA", "OMol xxsmall", "OMol RE5 Medium"]

# Store RDFs for each trajectory
rdf_list = []

# Loop over all trajectories
for traj_path in traj_paths:
    atoms_list = read(traj_path, index=":")
    rdf_counts = np.zeros(len(bins) - 1)
    total_density = 0.0
    n_frames = 0

    for atoms in tqdm(atoms_list, desc=f"Processing trajectory"):
        pos = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        box = atoms.get_cell()
        pbc = atoms.get_pbc()

        na_idx = [i for i, s in enumerate(symbols) if s == "Na"]
        p_idx  = [i for i, s in enumerate(symbols) if s == "P"]

        pos_na = pos[na_idx]
        pos_p  = pos[p_idx]

        volume = atoms.get_volume()
        total_density += len(pos_p) / volume

        for r_na in pos_na:
            disp = pos_p - r_na
            disp_mic = find_mic(disp, box, pbc=pbc)[0]
            dists = np.linalg.norm(disp_mic, axis=1)
            hist, _ = np.histogram(dists[dists < r_max], bins=bins)
            rdf_counts += hist

        n_frames += 1

    rho_p = total_density / n_frames
    normalization = rho_p * shell_volumes * len(na_idx) * n_frames
    rdf = rdf_counts / normalization
    rdf_list.append(rdf)

# Plot all RDFs
plt.figure(figsize=(6, 4))
for i, rdf in enumerate(rdf_list):
    plt.plot(r_centers, rdf, label=labels[i], color=colors[i])
plt.xlabel("r (Å)")
plt.ylabel("g_Na–P(r)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Create the plots directory if it doesn't exist
plots_dir = "/home/ishan-amin/MLFF-distill/APPLICATIONS/electrolytes/plots"
os.makedirs(plots_dir, exist_ok=True)

# Save the plot
plot_path = os.path.join(plots_dir, "rdf_na_p.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"RDF plot saved to: {plot_path}")