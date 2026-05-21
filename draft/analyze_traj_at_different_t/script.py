import os
import matplotlib
from tqdm import tqdm

# Headless backend for batch saving
matplotlib.use("Agg")

from ase.io import Trajectory
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from scipy.stats import ks_2samp, wasserstein_distance, iqr, entropy
from scipy.spatial.distance import squareform


# ---- Config ----
TRAJ_323K = "/global/homes/y/yuejian/project/MLFF-distill/m4558/323K_500ps_trajs/md_omol_naotf_diglyme_1m_s1p1/md_omol_naotf_diglyme_1m_s1p1.traj"
TRAJ_293K = "/global/homes/y/yuejian/project/MLFF-distill/m5024/UMA_trajs_293K/naotf_diglyme.traj"
TRAJ_353K = None  # optional third temperature
N_FRAMES = 5000          # first 50 ps if dt=10 fs
DT_FS = 10               # timestep in femtoseconds
SUBSAMPLE_STRIDE = 10    # diversity metric stride (10 -> 500 frames)
N_BINS = 60              # histogram bins for RMSD-to-first and diversity
DIHEDRAL_BINS = 36       # for torsion entropy (10 deg bins)
RMSD_CLUSTER_CUTOFF = 2.0  # Å cutoff for hierarchical clustering
EXAMPLE_DIHEDRAL_COUNT = 3  # how many dihedrals to plot distributions for
PLOT_DIR = "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/plot"


def load_first_n_frames(traj_path: str, n_frames: int) -> list[Atoms]:
    """Load the first n_frames from a trajectory file."""
    with Trajectory(traj_path) as traj_file:
        if len(traj_file) < n_frames:
            raise ValueError(f"{traj_path} has only {len(traj_file)} frames; requested {n_frames}.")
        return [traj_file[i] for i in range(n_frames)]


def select_heavy_atoms(atoms: Atoms) -> np.ndarray:
    """Return indices of heavy atoms (atomic number > 1)."""
    numbers = atoms.get_atomic_numbers()
    idx = np.where(numbers > 1)[0]
    if len(idx) == 0:
        raise ValueError("No heavy atoms (Z>1) found; cannot compute RMSD.")
    return idx


def get_heavy_positions_list(traj: list[Atoms]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """For a trajectory, get heavy atom coordinates (n_frames, n_heavy, 3) and validate indexing/order."""
    idx = select_heavy_atoms(traj[0])
    symbols0 = traj[0][idx].get_chemical_symbols()
    arr = np.empty((len(traj), len(idx), 3))
    for f, at in enumerate(traj):
        idx_now = select_heavy_atoms(at)
        if not np.array_equal(idx, idx_now):
            raise ValueError(f"Heavy atom indices differ at frame {f}: {idx} vs {idx_now}")
        if at[idx].get_chemical_symbols() != symbols0:
            raise ValueError(f"Heavy atom order/symbols differ at frame {f}")
        arr[f] = at.get_positions()[idx]
    return arr, idx, symbols0


def kabsch_align(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Align P (Nx3) onto Q (Nx3) using the Kabsch algorithm.
    Returns rotated/centered P and centered Q.
    """
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    C = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    U = V @ np.diag([1, 1, d]) @ Wt
    P_rot = Pc @ U
    return P_rot, Qc


def rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    return np.sqrt(np.mean(np.sum((P - Q) ** 2, axis=1)))


def get_rmsd_vs_first(heavy_coords_arr: np.ndarray) -> np.ndarray:
    """Return RMSD array (aligned to first frame)."""
    ref = heavy_coords_arr[0]
    out = np.zeros(len(heavy_coords_arr))
    for i, xyz in enumerate(tqdm(heavy_coords_arr, desc="RMSD vs first", unit="frame")):
        xyz_aln, ref_aln = kabsch_align(xyz, ref)
        out[i] = rmsd(xyz_aln, ref_aln)
    return out


def subsample_frames(heavy_coords_arr: np.ndarray, stride: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Subsample frames as (n_sub, n_heavy, 3). Returns array and indices."""
    idxs = np.arange(0, heavy_coords_arr.shape[0], stride, dtype=int)
    return heavy_coords_arr[idxs], idxs


def pairwise_rmsd_matrix(coords: np.ndarray) -> np.ndarray:
    """Return condensed pairwise RMSD distances (upper triangle) after alignment."""
    n = coords.shape[0]
    pdmat = np.zeros((n, n))
    for i in tqdm(range(n), desc="Pairwise RMSD rows", unit="row"):
        for j in range(i + 1, n):
            a_aln, b_aln = kabsch_align(coords[i], coords[j])
            pdmat[i, j] = rmsd(a_aln, b_aln)
            pdmat[j, i] = pdmat[i, j]
    return squareform(pdmat)


def compute_summary_stats(data: np.ndarray) -> dict:
    data = np.asarray(data)
    return {
        "mean": np.mean(data),
        "median": np.median(data),
        "std": np.std(data),
        "iqr": iqr(data),
        "90th_perc": np.percentile(data, 90),
        "95th_perc": np.percentile(data, 95),
    }


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two histograms."""
    p = np.asarray(p, dtype=float) + 1e-12
    q = np.asarray(q, dtype=float) + 1e-12
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


# ---- Torsional analysis helpers ----
def build_heavy_neighbor_list(atoms: Atoms, heavy_idx: np.ndarray):
    """Build adjacency list for heavy atoms (local indexing) using ASE neighborlist with covalent cutoffs."""
    from ase.neighborlist import natural_cutoffs, NeighborList

    cutoffs = natural_cutoffs(atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    orig_to_local = {orig: i for i, orig in enumerate(heavy_idx)}
    adjacency = {i: [] for i in range(len(heavy_idx))}
    for orig_i in heavy_idx:
        local_i = orig_to_local[orig_i]
        indices, offsets = nl.get_neighbors(orig_i)
        for j, off in zip(indices, offsets):
            if j in orig_to_local:
                local_j = orig_to_local[j]
                if local_j not in adjacency[local_i]:
                    adjacency[local_i].append(local_j)
                if local_i not in adjacency[local_j]:
                    adjacency[local_j].append(local_i)
    return adjacency


def find_rotatable_dihedrals(adjacency: dict) -> list[tuple[int, int, int, int]]:
    """
    Identify rotatable dihedrals i-j-k-l where j and k are not terminal (degree >1).
    This is a heuristic and ignores ring detection; adjust if needed.
    """
    dihedrals = []
    for j in adjacency:
        for k in adjacency[j]:
            if j >= k:
                continue
            if len(adjacency[j]) <= 1 or len(adjacency[k]) <= 1:
                continue
            for i in adjacency[j]:
                if i == k:
                    continue
                for l in adjacency[k]:
                    if l == j or l == i:
                        continue
                    dihedral = (i, j, k, l)
                    dihedrals.append(dihedral)
    # deduplicate (sorted tuple as key)
    uniq = {}
    for d in dihedrals:
        key = tuple(d)
        uniq[key] = d
    return list(uniq.values())


def torsion_angle(p1, p2, p3, p4) -> float:
    """Return dihedral angle in radians in range (-pi, pi]."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    # normalize
    n1 /= np.linalg.norm(n1) + 1e-12
    n2 /= np.linalg.norm(n2) + 1e-12
    m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-12))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return np.arctan2(y, x)


def compute_dihedral_timeseries(coords: np.ndarray, dihedrals: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Compute dihedral angles (radians) for each frame and dihedral."""
    n_frames = coords.shape[0]
    angles = np.zeros((n_frames, len(dihedrals)))
    for f in tqdm(range(n_frames), desc="Dihedral frames", unit="frame"):
        frame = coords[f]
        for d_idx, (i, j, k, l) in enumerate(dihedrals):
            angles[f, d_idx] = torsion_angle(frame[i], frame[j], frame[k], frame[l])
    return angles


def circular_variance(angles: np.ndarray) -> float:
    """Circular variance in [0,1]."""
    C = np.mean(np.cos(angles))
    S = np.mean(np.sin(angles))
    R = np.sqrt(C**2 + S**2)
    return 1 - R


def entropy_from_hist(hist: np.ndarray) -> float:
    p = hist.astype(float)
    p /= p.sum() + 1e-12
    p = p + 1e-12
    return -np.sum(p * np.log(p))


def dihedral_entropy_stats(angles: np.ndarray, n_bins: int) -> tuple[np.ndarray, float, float]:
    """Return per-dihedral entropy, mean entropy, total entropy."""
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    entropies = np.zeros(angles.shape[1])
    for j in range(angles.shape[1]):
        hist, _ = np.histogram(angles[:, j], bins=bins)
        entropies[j] = entropy_from_hist(hist)
    return entropies, float(np.mean(entropies)), float(np.sum(entropies))


def count_multimodal(angles_a: np.ndarray, angles_b: np.ndarray, n_bins: int) -> int:
    """
    Count dihedrals that become more multimodal at higher T:
    multimodal if >=2 peaks above 10% of max bin.
    """
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    def peaks(x):
        hist, _ = np.histogram(x, bins=bins)
        if hist.max() == 0:
            return 0
        thresh = 0.1 * hist.max()
        return np.sum(hist > thresh)
    count = 0
    for j in range(angles_a.shape[1]):
        peaks_a = peaks(angles_a[:, j])
        peaks_b = peaks(angles_b[:, j])
        if peaks_a >= 2 and peaks_b < peaks_a:
            count += 1
    return count


# ---- PCA helpers ----
def pca_on_coords(aligned_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    aligned_coords: (n_frames, n_atoms, 3)
    Returns eigenvalues (desc) and projected coordinates (n_frames, n_components).
    """
    X = aligned_coords.reshape(aligned_coords.shape[0], -1)
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = np.cov(Xc, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    proj = Xc @ eigvecs
    return eigvals, proj


# ---- Clustering helpers ----
def cluster_via_rmsd(coords: np.ndarray, cutoff: float) -> dict:
    """
    Hierarchical clustering (average linkage) on aligned RMSD distances.
    Returns labels and stats.
    """
    from scipy.cluster.hierarchy import linkage, fcluster

    dists = pairwise_rmsd_matrix(coords)
    Z = linkage(dists, method="average")
    labels = fcluster(Z, t=cutoff, criterion="distance")
    n_clusters = labels.max()
    pops = np.array([(labels == k).sum() for k in range(1, n_clusters + 1)])
    probs = pops / pops.sum()
    H = entropy(probs)
    neff = float(np.exp(H))

    # mean cluster lifetime in frames (contiguous segments)
    lifetimes = []
    current = labels[0]
    length = 1
    for lab in labels[1:]:
        if lab == current:
            length += 1
        else:
            lifetimes.append(length)
            current = lab
            length = 1
    lifetimes.append(length)
    mean_lifetime = float(np.mean(lifetimes))

    return {
        "labels": labels,
        "n_clusters": n_clusters,
        "probs": probs,
        "entropy": float(H),
        "neff": neff,
        "mean_lifetime_frames": mean_lifetime,
        "dominant_prob": float(probs.max()),
    }


def save_fig(fig: matplotlib.figure.Figure, filename: str) -> None:
    """Save figure to PLOT_DIR and close it."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    out_path = os.path.join(PLOT_DIR, filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


def main() -> None:
    # Load trajectories (supports 293/323 and optional 353)
    print("Loading trajectories...")
    traj_paths = {
        293: TRAJ_293K,
        323: TRAJ_323K,
    }
    if TRAJ_353K:
        traj_paths[353] = TRAJ_353K

    frames_by_t: dict[int, list[Atoms]] = {}
    for T, p in traj_paths.items():
        if not p:
            continue
        frames_by_t[T] = load_first_n_frames(p, N_FRAMES)

    if len(frames_by_t) < 2:
        raise ValueError("Need at least two trajectories (e.g., 293K and 323K) to compare.")

    print("Trajectories loaded:")
    for T in sorted(frames_by_t):
        print(f"  {T}K: {traj_paths[T]}")

    # Heavy atom coordinates and validation
    print("Extracting heavy-atom coordinates and validating consistency...")
    coords_by_t: dict[int, np.ndarray] = {}
    idx_ref = None
    syms_ref = None
    T_ref = sorted(frames_by_t)[0]
    for T in sorted(frames_by_t):
        coords, idx, syms = get_heavy_positions_list(frames_by_t[T])
        coords_by_t[T] = coords
        if idx_ref is None:
            idx_ref = idx
            syms_ref = syms
        else:
            if len(idx) != len(idx_ref):
                raise ValueError(f"Number of heavy atoms differs: {T_ref}K={len(idx_ref)}, {T}K={len(idx)}")
            if syms != syms_ref:
                raise ValueError(f"Heavy atom order or elements differ between trajectories: {T_ref}K vs {T}K")
    # Note: trajectories assumed unwrapped / no PBC correction applied.
    print(f"Heavy atoms validated (n={len(idx_ref)}).")

    # RMSD vs first-frame
    print("Computing RMSD vs first frame...")
    times_ps = np.arange(N_FRAMES) * DT_FS * 1e-3  # ps
    rmsd_by_t: dict[int, np.ndarray] = {}
    for T in sorted(coords_by_t):
        rmsd_by_t[T] = get_rmsd_vs_first(coords_by_t[T])
    print("RMSD vs first frame done.")

    # Pairwise RMSD (diversity metric)
    print(f"Computing pairwise RMSD with stride={SUBSAMPLE_STRIDE}...")
    coords_sub_by_t: dict[int, np.ndarray] = {}
    pairw_by_t: dict[int, np.ndarray] = {}
    stats_by_t: dict[int, dict] = {}
    for T in sorted(coords_by_t):
        coords_sub, _idxs = subsample_frames(coords_by_t[T], stride=SUBSAMPLE_STRIDE)
        coords_sub_by_t[T] = coords_sub
        pairw = pairwise_rmsd_matrix(coords_sub)
        pairw_by_t[T] = pairw
        stats_by_t[T] = compute_summary_stats(pairw)
    print("Pairwise RMSD done.")

    # Distribution comparisons (use 293K as baseline when present)
    print("Distribution comparisons...")
    baseline_T = 293 if 293 in rmsd_by_t else sorted(rmsd_by_t)[0]
    bins = np.linspace(0, max(r.max() for r in rmsd_by_t.values()), N_BINS + 1)
    h_by_t = {T: np.histogram(rmsd_by_t[T], bins=bins)[0] for T in rmsd_by_t}

    comparisons = []
    for T in sorted(rmsd_by_t):
        if T == baseline_T:
            continue
        ks_stat, ks_p = ks_2samp(rmsd_by_t[T], rmsd_by_t[baseline_T])
        try:
            wass_dist = wasserstein_distance(rmsd_by_t[T], rmsd_by_t[baseline_T])
        except Exception:
            wass_dist = np.nan
        jsd = jensen_shannon_divergence(h_by_t[T], h_by_t[baseline_T])
        comparisons.append((T, ks_stat, ks_p, jsd, wass_dist))
    print("Distribution comparisons done.")

    # Plotting
    print("Saving plots...")
    fig = plt.figure(figsize=(7, 4))
    for T in sorted(rmsd_by_t):
        plt.plot(times_ps, rmsd_by_t[T], label=f"{T} K", alpha=0.85)
    plt.xlabel("Time (ps)")
    plt.ylabel("RMSD to first frame (Å)")
    plt.title("RMSD vs Time (Heavy Atoms, aligned)")
    plt.legend()
    plt.tight_layout()
    save_fig(fig, "rmsd_vs_time.png")

    fig = plt.figure(figsize=(6, 4))
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    for T in sorted(h_by_t):
        h = h_by_t[T]
        plt.step(bin_centers, h / np.sum(h), where="mid", label=f"{T} K")
    plt.xlabel("RMSD to first frame (Å)")
    plt.ylabel("Probability density")
    plt.title("RMSD distribution (heavy atoms)")
    plt.legend()
    plt.tight_layout()
    save_fig(fig, "rmsd_hist.png")

    pw_bins = np.linspace(0, max(max(p.max() for p in pairw_by_t.values()), 2.0), N_BINS)
    fig = plt.figure(figsize=(6, 4))
    for T in sorted(pairw_by_t):
        plt.hist(pairw_by_t[T], bins=pw_bins, alpha=0.4, label=f"{T} K", density=True, histtype="stepfilled")
    plt.xlabel("Pairwise RMSD (Å)")
    plt.ylabel("Probability density")
    plt.title(f"Pairwise RMSD diversity (stride={SUBSAMPLE_STRIDE})")
    plt.legend()
    plt.tight_layout()
    save_fig(fig, "pairwise_rmsd_hist.png")
    print("Plots saved.")

    # Text summary
    print("\n--- RMSD vs First Frame ---")
    for T in sorted(rmsd_by_t):
        r = rmsd_by_t[T]
        print(f"{T} K: mean={np.mean(r):.3f}, std={np.std(r):.3f}, median={np.median(r):.3f}, max={np.max(r):.3f}")
    for (T, ks_stat, ks_p, jsd, wass_dist) in comparisons:
        print(f"\n--- Distribution vs {baseline_T}K: {T}K ---")
        print(f"KS statistic={ks_stat:.4f}, p-value={ks_p:.3e}")
        print(f"Jensen-Shannon divergence={jsd:.4f}")
        if not np.isnan(wass_dist):
            print(f"Wasserstein distance={wass_dist:.4f}")

    print("\n--- Pairwise RMSD Diversity (heavy atoms, stride=%d) ---" % SUBSAMPLE_STRIDE)
    for T in sorted(stats_by_t):
        st = stats_by_t[T]
        print(f"  {T}K:", ", ".join(f"{k}={v:.3f}" for k, v in st.items()))

    print("\nNOTES:")
    print(f"  Used only heavy atoms (n={len(idx_ref)}), same order/chemistry checked across all trajectories.")
    print("  RMSD computed after optimal superposition; PBC unwrapping not applied (unwrap beforehand if needed).")
    print(f"  Frame stride for diversity: {SUBSAMPLE_STRIDE}. Adjust constants at top as desired.")

    # ---- Torsional diversity ----
    # Commented out per request; re-enable if torsional analysis is needed.
    # print("Finding rotatable dihedrals and computing torsional metrics...")
    # adjacency = build_heavy_neighbor_list(first_50ps_frames_323K[0], idx_323K)
    # dihedrals = find_rotatable_dihedrals(adjacency)
    # if len(dihedrals) == 0:
    #     print("\nNo rotatable dihedrals found among heavy atoms; skipping torsional analysis.")
    #     angles323 = angles293 = None
    # else:
    #     angles323 = compute_dihedral_timeseries(coords_323K, dihedrals)
    #     angles293 = compute_dihedral_timeseries(coords_293K, dihedrals)
    #     circ_var_323 = np.array([circular_variance(angles323[:, j]) for j in range(angles323.shape[1])])
    #     circ_var_293 = np.array([circular_variance(angles293[:, j]) for j in range(angles293.shape[1])])
    #     ent_323, mean_ent_323, sum_ent_323 = dihedral_entropy_stats(angles323, DIHEDRAL_BINS)
    #     ent_293, mean_ent_293, sum_ent_293 = dihedral_entropy_stats(angles293, DIHEDRAL_BINS)
    #     n_multimodal = count_multimodal(angles323, angles293, DIHEDRAL_BINS)
    #
    #     print("\n--- Torsional Diversity (heavy-atom dihedrals) ---")
    #     print(f"Dihedrals analyzed: {len(dihedrals)}")
    #     print(f"Mean circular variance: 323K={circ_var_323.mean():.3f}, 293K={circ_var_293.mean():.3f}")
    #     print(f"Mean torsional entropy: 323K={mean_ent_323:.3f}, 293K={mean_ent_293:.3f}")
    #     print(f"Total torsional entropy: 323K={sum_ent_323:.3f}, 293K={sum_ent_293:.3f}")
    #     print(f"Dihedrals with increased multimodality at 323K vs 293K: {n_multimodal}")
    #
    #     # Plot representative dihedrals
    #     bins = np.linspace(-np.pi, np.pi, DIHEDRAL_BINS + 1)
    #     for idx_plot in range(min(EXAMPLE_DIHEDRAL_COUNT, len(dihedrals))):
    #         fig = plt.figure(figsize=(6, 4))
    #         plt.hist(angles323[:, idx_plot], bins=bins, alpha=0.5, density=True, label="323 K")
    #         plt.hist(angles293[:, idx_plot], bins=bins, alpha=0.5, density=True, label="293 K")
    #         plt.xlabel("Dihedral angle (rad)")
    #         plt.ylabel("Density")
    #         plt.title(f"Dihedral {idx_plot} distribution")
    #         plt.legend()
    #         plt.tight_layout()
    #         save_fig(fig, f"dihedral_{idx_plot}_hist.png")
    #     print("Torsional analysis done.")

    # ---- PCA / essential dynamics ----
    print("Running PCA on aligned heavy-atom coordinates...")
    eigvals_by_t: dict[int, np.ndarray] = {}
    proj_by_t: dict[int, np.ndarray] = {}
    for T in sorted(coords_by_t):
        eigvals, proj = pca_on_coords(coords_by_t[T])
        eigvals_by_t[T] = eigvals
        proj_by_t[T] = proj

    fig = plt.figure(figsize=(6, 4))
    for T in sorted(eigvals_by_t):
        plt.plot(eigvals_by_t[T], label=f"{T} K")
    plt.xlabel("PC index")
    plt.ylabel("Variance")
    plt.title("PCA eigenvalue spectrum (heavy atoms)")
    plt.legend()
    plt.tight_layout()
    save_fig(fig, "pca_spectrum.png")

    # PC1-PC2 projections colored by time
    def scatter_pc12(proj, times, label, fname):
        fig = plt.figure(figsize=(5, 5))
        sc = plt.scatter(proj[:, 0], proj[:, 1], c=times, cmap="viridis", s=5)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(label)
        plt.colorbar(sc, label="time (ps)")
        plt.tight_layout()
        save_fig(fig, fname)

    for T in sorted(proj_by_t):
        scatter_pc12(proj_by_t[T], times_ps, f"{T} K PC1-PC2", f"pca_pc1_pc2_{T}K.png")

    print("\n--- PCA / Essential Dynamics ---")
    for T in sorted(eigvals_by_t):
        total_var = eigvals_by_t[T].sum()
        frac3 = eigvals_by_t[T][:3].sum() / total_var
        print(f"{T}K: total variance={total_var:.3f}, top-3 fraction={frac3:.3f}")
    print("PCA done.")

    # ---- Clustering-based diversity ----
    print("Clustering conformations (RMSD-based)...")
    cluster_by_t: dict[int, dict] = {}
    for T in sorted(coords_sub_by_t):
        cluster_by_t[T] = cluster_via_rmsd(coords_sub_by_t[T], RMSD_CLUSTER_CUTOFF)

    print("\n--- RMSD Clustering (stride=%d, cutoff=%.2f Å) ---" % (SUBSAMPLE_STRIDE, RMSD_CLUSTER_CUTOFF))
    for T in sorted(cluster_by_t):
        c = cluster_by_t[T]
        print(
            f"{T}K: clusters={c['n_clusters']}, entropy={c['entropy']:.3f}, "
            f"N_eff={c['neff']:.2f}, dominant_prob={c['dominant_prob']:.2f}, "
            f"mean_lifetime_frames={c['mean_lifetime_frames']:.2f}"
        )

    # Cluster population bar plots
    fig = plt.figure(figsize=(6, 4))
    # Plot only baseline vs the highest temperature (keeps the plot readable).
    Ts_sorted = sorted(cluster_by_t)
    T_low = Ts_sorted[0]
    T_high = Ts_sorted[-1]
    cl = cluster_by_t[T_low]
    ch = cluster_by_t[T_high]
    plt.bar(np.arange(1, ch["n_clusters"] + 1) - 0.2, ch["probs"], width=0.4, label=f"{T_high} K")
    plt.bar(np.arange(1, cl["n_clusters"] + 1) + 0.2, cl["probs"], width=0.4, label=f"{T_low} K")
    plt.xlabel("Cluster ID")
    plt.ylabel("Population probability")
    plt.title("Cluster populations")
    plt.legend()
    plt.tight_layout()
    save_fig(fig, "cluster_populations.png")
    print("Clustering done.")

    # Final summary
    print("\n=== SUMMARY ===")
    for T in sorted(stats_by_t):
        print(f"{T}K mean pairwise RMSD: {stats_by_t[T]['mean']:.3f} Å")
    if comparisons:
        for (T, ks_stat, _ks_p, jsd, wass_dist) in comparisons:
            wass_str = f"{wass_dist:.3f}" if not np.isnan(wass_dist) else "nan"
            print(f"Vs {baseline_T}K ({T}K): KS={ks_stat:.3f}, JSD={jsd:.3f}, Wasserstein={wass_str}")
    for T in sorted(cluster_by_t):
        c = cluster_by_t[T]
        print(f"{T}K cluster N_eff={c['neff']:.2f}, dominant_prob={c['dominant_prob']:.2f}")


if __name__ == "__main__":
    main()