import numpy as np
import pandas as pd
from ase.io import read
from ase.geometry import find_mic
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# ========= User settings =========
element = 'O'
assert element in ["O", "F"]
distill = True
identifier = 'micro'
solvent = "DME"

# traj_paths = ["/projects/beyy/shared/md_results/all_napf6_train_diglyme_md_212k.traj"]
# traj_paths = ["/projects/beye/iamin/trajs/napf6_diglyme_1ns.traj"]
traj_paths = [f"/projects/beye/iamin/trajs/napf6_{solvent}_1ns.traj"]
csv_root = "/projects/beye/iamin/rdfs/Archive/rdf_csvs_na_o" if element == "O" else "/projects/beye/iamin/rdfs/Archive/rdf_csvs_na_F"

csv_path   = os.path.join(csv_root, f"RDF_{solvent}_Na-{element}.csv" )

plots_dir  = f"/projects/beye/iamin/rdfs/plots/"
os.makedirs(plots_dir, exist_ok=True)
plot_name = f"rdf_{solvent}_na_{element}_DIST.png" if distill else f"rdf_{solvent}na_{element}.png"
plot_path = os.path.join(plots_dir, plot_name)
plt_title = f"NAPF6_{solvent}_na_{element}_DIST_vs_UMA" if  distill else f"NAPF6_{solvent}_na_{element}_vsUMA.png"
# ========= RDF binning (unchanged) =========
r_max = 10.0
bin_width = 0.1
bins = np.arange(0.0, r_max + bin_width, bin_width)
shell_volumes = (4/3) * np.pi * (bins[1:]**3 - bins[:-1]**3)
r_centers = 0.5 * (bins[1:] + bins[:-1])

# ========= Plot look =========
colors = ["tab:red", "tab:blue", "tab:green", "tab:purple"]
labels = ["hessian_w80" if distill else "undist", "UMA_sm_consv", "Extra 1", "Extra 2"]

# ========= Helpers =========
def process_frame(atoms):
    """Return histogram counts and local density for a single frame."""
    pos = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    box = atoms.get_cell()
    pbc = atoms.get_pbc()

    na_idx = [i for i, s in enumerate(symbols) if s == "Na"]
    x_idx  = [i for i, s in enumerate(symbols) if s == element]

    pos_na = pos[na_idx]
    pos_x  = pos[x_idx]

    volume = atoms.get_volume()
    local_density = len(pos_x) / volume

    # per-frame histogram
    hist_acc = np.zeros(len(bins) - 1, dtype=np.int64)
    for r_na in pos_na:
        disp = pos_x - r_na
        disp_mic = find_mic(disp, box, pbc=pbc)[0]
        dists = np.linalg.norm(disp_mic, axis=1)
        hist, _ = np.histogram(dists[dists < r_max], bins=bins)
        hist_acc += hist

    return hist_acc, local_density

def compute_rdf_from_traj(traj_path, n_cpus=12):
    atoms_list = read(traj_path, index=":")
    n_frames = len(atoms_list)

    # assume constant Na count per frame
    first_symbols = atoms_list[0].get_chemical_symbols()
    N_na = sum(1 for s in first_symbols if s == "Na")

    rdf_counts = np.zeros(len(bins) - 1, dtype=np.int64)
    total_density = 0.0

    with ProcessPoolExecutor(max_workers=n_cpus) as exe:
        futures = [exe.submit(process_frame, atoms) for atoms in atoms_list]
        for hist_acc, local_density in tqdm(
            (f.result() for f in futures),
            total=n_frames,
            desc=f"Processing {os.path.basename(traj_path)}"
        ):
            rdf_counts    += hist_acc
            total_density += local_density

    rho_x = total_density / n_frames
    normalization = rho_x * shell_volumes * N_na * n_frames
    rdf = rdf_counts / normalization
    return rdf

def load_rdf_from_csv(csv_file, target_r=r_centers, r_col="r_A", g_col="g_r"):
    """
    Load CSV with columns like r_A,g_r,n_r,w_r_kJmol, and
    interpolate g(r) onto target_r (your bin centers).
    """
    df = pd.read_csv(csv_file)
    if r_col not in df.columns or g_col not in df.columns:
        raise ValueError(f"CSV must contain '{r_col}' and '{g_col}' columns.")
    r = df[r_col].to_numpy()
    g = df[g_col].to_numpy()

    # Keep only finite, sorted values for robust interpolation
    m = np.isfinite(r) & np.isfinite(g)
    r, g = r[m], g[m]
    order = np.argsort(r)
    r, g = r[order], g[order]

    # Interpolate onto your r_centers (out-of-range -> NaN so we can compare overlaps)
    g_on_target = np.interp(target_r, r, g, left=np.nan, right=np.nan)
    return g_on_target

def compare(rdf_a, rdf_b, name_a="A", name_b="B"):
    # Only compare where both are finite
    m = np.isfinite(rdf_a) & np.isfinite(rdf_b)
    if not np.any(m):
        print("No overlap between curves to compare.")
        return None
    diff = rdf_a[m] - rdf_b[m]
    mean_signed = np.mean(diff)
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    max_abs = np.max(np.abs(diff))
    print(f"\nDifferences ({name_a} - {name_b}) on {m.sum()} overlapping points:")
    print(f"  Mean (signed): {mean_signed:.6f}")
    print(f"  MAE:          {mae:.6f}")
    print(f"  RMSE:         {rmse:.6f}")
    print(f"  Max |diff|:   {max_abs:.6f}")
    return {"mean": mean_signed, "mae": mae, "rmse": rmse, "max_abs": max_abs}

# ========= Compute both RDFs =========
# 1) Trajectory-derived RDF (if multiple, you can loop; here we use the first)
traj_rdf = compute_rdf_from_traj(traj_paths[0])

# 2) CSV-derived RDF (interpolated onto r_centers)
csv_rdf  = load_rdf_from_csv(csv_path, target_r=r_centers)

# ========= Compare =========
stats = compare(traj_rdf, csv_rdf, name_a="Trajectory", name_b="CSV")

# ========= Plot =========
plt.figure(figsize=(6, 4))
plt.plot(r_centers, traj_rdf, label=f"{labels[0]}, mae= {stats['mae']:.3f}", color=colors[0])
plt.plot(r_centers, csv_rdf,  label=labels[1], color=colors[1], linestyle="--")

plt.xlabel("r (Å)")
plt.ylabel(f"g_Na–{element}(r)")
plt.legend()
plt.title(f"{plt_title}")
plt.grid(True)
plt.tight_layout()


plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"RDF plot (traj vs CSV) saved to: {plot_path}")
