import numpy as np
from ase.io import read
from ase.geometry import find_mic
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# List of .traj files (unchanged)
element = 'P'
steps = 62
# structure = "natfsi"
structure = "napf6"
fname = f"all_napf6_distill_DMC_md_{steps}k_50th_start"
xx_small_path = f"/u/czhang31/MLFF-distill/APPLICATIONS/electrolytes/md_results/{fname}.traj"
# xx_small_path = '/u/czhang31/MLFF-distill/APPLICATIONS/electrolytes/md_results/58_82_122k_models/natfsi_s1p1_distilled_82k_steps_md.traj'
uma_path = '/u/czhang31/data/1mnapf6_solvents_omol_trajs/md_omol_dimethylcarbonate_pfactor_0.1_1fs_mask_t_re3_s1p1.traj'
# uma_paths = [
#     "/u/czhang31/data/natfsi_s1p1/uma_traj/md_omol_1M_natfsi_s1p1.traj", # natfsi uma
#     "/u/czhang31/data/napf6_s1p1/uma_traj/md_omol_re5_small_1p1_wrapped.traj", # napf6 uma
# ]
# uma_path = uma_paths[0] if structure == "natfsi" else uma_paths[1]
traj_paths = [
    uma_path,
    xx_small_path
]

# RDF parameters (unchanged)
r_max = 15  # changed from 10 to 15
bin_width = 0.1
bins = np.arange(0.0, r_max + bin_width, bin_width)
shell_volumes = (4/3) * np.pi * (bins[1:]**3 - bins[:-1]**3)
r_centers = 0.5 * (bins[1:] + bins[:-1])

# Plot settings (unchanged)
colors = ["tab:red", "tab:blue"]
labels = ["UMA", f"Distilled {steps}k"]

def process_frame(atoms):
    """Return histogram counts and local density for a single frame."""
    pos = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    box = atoms.get_cell()
    pbc = atoms.get_pbc()

    na_idx = [i for i, s in enumerate(symbols) if s == "Na"]
    p_idx  = [i for i, s in enumerate(symbols) if s == element]

    pos_na = pos[na_idx]
    pos_p  = pos[p_idx]

    volume = atoms.get_volume()
    local_density = len(pos_p) / volume

    # per-frame histogram
    hist_acc = np.zeros(len(bins) - 1, dtype=np.int64)
    for r_na in pos_na:
        disp = pos_p - r_na
        disp_mic = find_mic(disp, box, pbc=pbc)[0]
        dists = np.linalg.norm(disp_mic, axis=1)
        hist, _ = np.histogram(dists[dists < r_max], bins=bins)
        hist_acc += hist

    return hist_acc, local_density

def compute_rdf(traj_path):
    atoms_list = read(traj_path, index="::10")
    n_frames = len(atoms_list)

    # get number of Na atoms per frame (assumed constant)
    first_symbols = atoms_list[0].get_chemical_symbols()
    N_na = sum(1 for s in first_symbols if s == "Na")

    rdf_counts = np.zeros(len(bins) - 1, dtype=np.int64)
    total_density = 0.0

    n_cpus =  12
    with ProcessPoolExecutor(max_workers=n_cpus) as exe:
        futures = [exe.submit(process_frame, atoms) for atoms in atoms_list]
        for hist_acc, local_density in tqdm(
            (f.result() for f in futures),
            total=n_frames,
            desc=f"Processing {os.path.basename(traj_path)}"
        ):
            rdf_counts    += hist_acc
            total_density += local_density

    # exact same normalization as before
    rho_p = total_density / n_frames
    normalization = rho_p * shell_volumes * N_na * n_frames
    rdf = rdf_counts / normalization
    return rdf

# run over each trajectory
rdf_list = [compute_rdf(p) for p in traj_paths]

# Plot & quantitative comparison (unchanged)
plt.figure(figsize=(6, 4))
if len(rdf_list) == 2:
    rdf1, rdf2 = rdf_list
    mae = np.mean(np.abs(rdf1 - rdf2))
    rmse = np.sqrt(np.mean((rdf1 - rdf2) ** 2))
    max_error = np.max(np.abs(rdf1 - rdf2))
    relative_error = mae / (
        np.max(np.concatenate([rdf1, rdf2])) -
        np.min(np.concatenate([rdf1, rdf2]))
    )
    labels[1] = f"{labels[1]} (MAE: {mae:.3f}, Max: {max_error:.3f})"
    print("Quantitative differences between RDFs:")
    print(f"  Mean Absolute Error (MAE): {mae:.6f}")
    print(f"  Root Mean Square Error (RMSE): {rmse:.6f}")
    print(f"  Maximum Absolute Error: {max_error:.6f}")
    print(f"  Relative Error: {relative_error:.6f}")

for i, rdf in enumerate(rdf_list):
    plt.plot(r_centers, rdf, label=labels[i], color=colors[i])

# **No change** to the y-axis label:
plt.xlabel("r (Å)")
plt.ylabel(f"g_Na-{element}(r)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plots_dir = f"/u/czhang31/MLFF-distill/APPLICATIONS/electrolytes/plots/{fname}"
os.makedirs(plots_dir, exist_ok=True)
plot_name = f"rdf_na_{element}_DIST.png"
plot_path = os.path.join(plots_dir, plot_name)
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"RDF plot saved to: {plot_path}")
