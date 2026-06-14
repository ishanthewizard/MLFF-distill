"""
student equilibirated traj:
dt = 100fs/frame

/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/results/diffusivity_main_results_20ns_final/micro_acas_50ps/20ns_solvent_0_1M/298K/md_omol_naotf_dme_s1p1_omol/md_omol_naotf_dme_s1p1_omol.traj
/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/results/diffusivity_main_results_20ns_final/micro_acas_50ps/20ns_solvent_0_1M/298K/md_omol_napf6_dme_re1/md_omol_napf6_dme_re1.traj
/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/results/diffusivity_main_results_20ns_final/micro_acas_50ps/20ns_solvent_solute_0.5M/323_2K/md_omol_lipf6_pfactor_0.1_1fs_mask_t/md_omol_lipf6_pfactor_0.1_1fs_mask_t.traj
/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/results/diffusivity_main_results_20ns_final/micro_acas_50ps/20ns_solvent_solute_0.5M/323_2K/md_omol_napf6_dme_re1/md_omol_napf6_dme_re1.traj

teacher equilibirated traj:
dt = 10fs/frame

/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/data/raw_data_from_UMA_simulation/other_temperature_for_comparing_with_student_model/20ns_solvent_solute_0.5M/323_2K/md_omol_lipf6_pfactor_0.1_1fs_mask_t/md_omol_lipf6_pfactor_0.1_1fs_mask_t.traj
/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/data/raw_data_from_UMA_simulation/other_temperature_for_comparing_with_student_model/20ns_solvent_solute_0.5M/323_2K/md_omol_napf6_dme_re1/md_omol_napf6_dme_re1.traj
/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/data/raw_data_from_UMA_simulation/other_temperature_for_comparing_with_student_model/20ns_solvent_0_1M/298K/md_omol_napf6_dme_re1/md_omol_napf6_dme_re1.traj
/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/data/raw_data_from_UMA_simulation/other_temperature_for_comparing_with_student_model/20ns_solvent_0_1M/298K/md_omol_naotf_dme_s1p1_omol/md_omol_naotf_dme_s1p1_omol.traj

Analysis: density, cell lengths (a,b,c), potential energy over first 1 ns.
Selects init frame for NVT production: density-closest-to-mean in 0.5–1 ns,
with fallback to cell-size filter (all axes >= 20 Å).
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase.io import Trajectory, write as ase_write

OUTPUT_DIR = '/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/nvt_init_frame'
os.makedirs(OUTPUT_DIR, exist_ok=True)

NS_FS = 1_000_000.0
HALF_NS_FS = 500_000.0
MIN_CELL_ANG = 20.0

TRAJ_CONFIGS = [
    # --- student (dt_fs=100 per frame, interval=100 steps at 1 fs/step) ---
    {
        'label': 'student_NaOTF_298K',
        'init_frame_name': 'na_otf_dme_298K_0.1M_student',
        'traj': '/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/results/diffusivity_main_results_20ns_final/micro_acas_50ps/20ns_solvent_0_1M/298K/md_omol_naotf_dme_s1p1_omol/md_omol_naotf_dme_s1p1_omol.traj',
        'dt_fs': 100.0,
    },
    {
        'label': 'student_NaPF6_298K',
        'init_frame_name': 'na_pf6_dme_298K_0.1M_student',
        'traj': '/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/results/diffusivity_main_results_20ns_final/micro_acas_50ps/20ns_solvent_0_1M/298K/md_omol_napf6_dme_re1/md_omol_napf6_dme_re1.traj',
        'dt_fs': 100.0,
    },
    {
        'label': 'student_LiPF6_323K',
        'init_frame_name': 'li_pf6_dme_323K_0.5M_student',
        'traj': '/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/results/diffusivity_main_results_20ns_final/micro_acas_50ps/20ns_solvent_solute_0.5M/323_2K/md_omol_lipf6_pfactor_0.1_1fs_mask_t/md_omol_lipf6_pfactor_0.1_1fs_mask_t.traj',
        'dt_fs': 100.0,
    },
    {
        'label': 'student_NaPF6_323K',
        'init_frame_name': 'na_pf6_dme_323K_0.5M_student',
        'traj': '/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/results/diffusivity_main_results_20ns_final/micro_acas_50ps/20ns_solvent_solute_0.5M/323_2K/md_omol_napf6_dme_re1/md_omol_napf6_dme_re1.traj',
        'dt_fs': 100.0,
    },
    # --- teacher (dt_fs=10 per frame, interval=10 steps at 1 fs/step) ---
    {
        'label': 'teacher_LiPF6_323K',
        'init_frame_name': 'li_pf6_dme_323K_0.5M_teacher',
        'traj': '/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/data/raw_data_from_UMA_simulation/other_temperature_for_comparing_with_student_model/20ns_solvent_solute_0.5M/323_2K/md_omol_lipf6_pfactor_0.1_1fs_mask_t/md_omol_lipf6_pfactor_0.1_1fs_mask_t.traj',
        'dt_fs': 10.0,
    },
    {
        'label': 'teacher_NaPF6_323K',
        'init_frame_name': 'na_pf6_dme_323K_0.5M_teacher',
        'traj': '/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/data/raw_data_from_UMA_simulation/other_temperature_for_comparing_with_student_model/20ns_solvent_solute_0.5M/323_2K/md_omol_napf6_dme_re1/md_omol_napf6_dme_re1.traj',
        'dt_fs': 10.0,
    },
    {
        'label': 'teacher_NaPF6_298K',
        'init_frame_name': 'na_pf6_dme_298K_0.1M_teacher',
        'traj': '/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/data/raw_data_from_UMA_simulation/other_temperature_for_comparing_with_student_model/20ns_solvent_0_1M/298K/md_omol_napf6_dme_re1/md_omol_napf6_dme_re1.traj',
        'dt_fs': 10.0,
    },
    {
        'label': 'teacher_NaOTF_298K',
        'init_frame_name': 'na_otf_dme_298K_0.1M_teacher',
        'traj': '/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/data/raw_data_from_UMA_simulation/other_temperature_for_comparing_with_student_model/20ns_solvent_0_1M/298K/md_omol_naotf_dme_s1p1_omol/md_omol_naotf_dme_s1p1_omol.traj',
        'dt_fs': 10.0,
    },
]


def read_traj_first_ns(traj_path, dt_fs, max_time_fs=NS_FS, target_points=1000):
    """
    Read traj frames up to max_time_fs by index — no full load into memory.
    Frame 0 has no calculator so Epot is NaN for that point.
    Returns: time_fs, density (g/cm3), a, b, c (Å), epot (eV), frame_indices.
    """
    n_frames_1ns = int(max_time_fs / dt_fs) + 1
    stride = max(1, n_frames_1ns // target_points)

    times, densities, as_, bs_, cs_, epots, frame_idxs = [], [], [], [], [], [], []

    traj = Trajectory(traj_path, 'r')
    try:
        total_mass_amu = traj[1].get_masses().sum()

        for i in range(0, n_frames_1ns, stride):
            try:
                atoms = traj[i]
            except (IndexError, Exception):
                break
            time_fs = i * dt_fs
            vol_ang3 = atoms.get_volume()
            density = total_mass_amu * 1.66054 / vol_ang3  # g/cm^3
            abc = atoms.cell.cellpar()[:3]
            epot = atoms.get_potential_energy() if atoms.calc is not None else float('nan')
            times.append(time_fs)
            densities.append(density)
            as_.append(abc[0])
            bs_.append(abc[1])
            cs_.append(abc[2])
            epots.append(epot)
            frame_idxs.append(i)
    finally:
        traj.close()

    return (np.array(times), np.array(densities),
            np.array(as_), np.array(bs_), np.array(cs_),
            np.array(epots), np.array(frame_idxs))


def closest_to_mean_in_window(values, times, frame_idxs, lo_fs=HALF_NS_FS, hi_fs=NS_FS):
    """Return info dict for the point in [lo_fs, hi_fs] closest to the full-array mean."""
    valid = np.isfinite(values)
    mean_val = values[valid].mean() if valid.any() else float('nan')
    mask = (times >= lo_fs) & (times <= hi_fs) & valid
    if not mask.any():
        return None
    sub_vals = values[mask]
    sub_times = times[mask]
    sub_frames = frame_idxs[mask]
    idx = np.argmin(np.abs(sub_vals - mean_val))
    return {
        'time_fs': float(sub_times[idx]),
        'time_ps': float(sub_times[idx] / 1000.0),
        'frame_idx': int(sub_frames[idx]),
        'value': float(sub_vals[idx]),
        'mean': float(mean_val),
    }


def select_init_frame(traj_times, densities, as_, bs_, cs_, frame_idxs):
    """
    Select the NVT init frame from the 0.5–1 ns window:

    1. Find the frame whose density is closest to the full-array mean density.
    2. Accept it if all cell lengths (a, b, c) >= MIN_CELL_ANG.
    3. Otherwise fall back: restrict to frames where a,b,c >= MIN_CELL_ANG,
       then pick the one closest to mean density among those.
    4. If no frame passes the cell criterion, use the density-closest frame anyway.

    Returns a dict with selection metadata and a 'method' field.
    """
    window_mask = (traj_times >= HALF_NS_FS) & (traj_times <= NS_FS) & np.isfinite(densities)
    if not window_mask.any():
        raise ValueError("No valid frames found in 0.5–1 ns window")

    mean_density = densities[np.isfinite(densities)].mean()

    w_times    = traj_times[window_mask]
    w_density  = densities[window_mask]
    w_a        = as_[window_mask]
    w_b        = bs_[window_mask]
    w_c        = cs_[window_mask]
    w_frames   = frame_idxs[window_mask]

    # primary: density-closest
    primary_idx = int(np.argmin(np.abs(w_density - mean_density)))
    pa, pb, pc = w_a[primary_idx], w_b[primary_idx], w_c[primary_idx]

    if pa >= MIN_CELL_ANG and pb >= MIN_CELL_ANG and pc >= MIN_CELL_ANG:
        method = 'primary_density'
        chosen = primary_idx
        sub_arrays = (w_times, w_density, w_a, w_b, w_c, w_frames)
    else:
        print(f"    Primary frame fails cell check (a={pa:.2f}, b={pb:.2f}, c={pc:.2f} Å) — falling back")
        cell_ok = (w_a >= MIN_CELL_ANG) & (w_b >= MIN_CELL_ANG) & (w_c >= MIN_CELL_ANG)
        if cell_ok.any():
            method = 'fallback_cell_then_density'
            fb_times   = w_times[cell_ok]
            fb_density = w_density[cell_ok]
            fb_a       = w_a[cell_ok]
            fb_b       = w_b[cell_ok]
            fb_c       = w_c[cell_ok]
            fb_frames  = w_frames[cell_ok]
            chosen = int(np.argmin(np.abs(fb_density - mean_density)))
            sub_arrays = (fb_times, fb_density, fb_a, fb_b, fb_c, fb_frames)
        else:
            print(f"    No frame in 0.5–1 ns has all axes >= {MIN_CELL_ANG} Å — using density-closest anyway")
            method = 'primary_density_no_cell_valid'
            chosen = primary_idx
            sub_arrays = (w_times, w_density, w_a, w_b, w_c, w_frames)

    t, d, a, b, c, fi = sub_arrays
    return {
        'frame_idx':   int(fi[chosen]),
        'time_fs':     float(t[chosen]),
        'time_ps':     float(t[chosen] / 1000.0),
        'density':     float(d[chosen]),
        'mean_density': float(mean_density),
        'a_ang':       float(a[chosen]),
        'b_ang':       float(b[chosen]),
        'c_ang':       float(c[chosen]),
        'method':      method,
    }


def dump_init_frame(src_traj_path, frame_idx, out_dir, name):
    """
    Extract frame_idx from src_traj_path and write:
      {out_dir}/{name}/{name}.traj   (single-frame ASE trajectory)
      {out_dir}/{name}/{name}.log    (plain-text metadata)
    Returns the path to the output directory.
    """
    frame_dir = os.path.join(out_dir, name)
    os.makedirs(frame_dir, exist_ok=True)

    traj = Trajectory(src_traj_path, 'r')
    try:
        atoms = traj[frame_idx]
    finally:
        traj.close()

    # Drop momenta so the resume script treats this as a fresh start
    # and resamples velocities from Maxwell-Boltzmann distribution.
    for key in ('momenta', 'velocities'):
        if key in atoms.arrays:
            del atoms.arrays[key]

    out_traj = os.path.join(frame_dir, f'{name}.traj')
    out_log  = os.path.join(frame_dir, f'{name}.log')

    writer = Trajectory(out_traj, 'w')
    writer.write(atoms)
    writer.close()

    with open(out_log, 'w') as fh:
        abc = atoms.cell.cellpar()
        vol = atoms.get_volume()
        mass = atoms.get_masses().sum()
        density = mass * 1.66054 / vol
        epot = atoms.get_potential_energy() if atoms.calc is not None else float('nan')
        fh.write(f"source_traj:  {src_traj_path}\n")
        fh.write(f"frame_index:  {frame_idx}\n")
        fh.write(f"time_ps:      {frame_idx * 0.001:.3f} ps  (approx, depends on dt)\n")
        fh.write(f"n_atoms:      {len(atoms)}\n")
        fh.write(f"a (Å):        {abc[0]:.4f}\n")
        fh.write(f"b (Å):        {abc[1]:.4f}\n")
        fh.write(f"c (Å):        {abc[2]:.4f}\n")
        fh.write(f"volume (Å³):  {vol:.4f}\n")
        fh.write(f"density g/cm³:{density:.6f}\n")
        fh.write(f"Epot (eV):    {epot}\n")

    return frame_dir


def analyze_and_plot(cfg, per_source_dir):
    label = cfg['label']
    dt_fs = cfg['dt_fs']
    print(f"\n=== {label} ===")

    print("  Reading traj...")
    traj_times, densities, as_, bs_, cs_, epots, frame_idxs = read_traj_first_ns(
        cfg['traj'], dt_fs
    )

    time_ps = traj_times / 1000.0

    metrics = {
        'density_g_cm3': densities,
        'a_ang':          as_,
        'b_ang':          bs_,
        'c_ang':          cs_,
        'epot_eV':        epots,
    }
    ylabels = {
        'density_g_cm3': 'Density (g/cm³)',
        'a_ang':          'a (Å)',
        'b_ang':          'b (Å)',
        'c_ang':          'c (Å)',
        'epot_eV':        'E_pot (eV)',
    }

    fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
    fig.suptitle(label, fontsize=13)

    per_metric_selections = {}
    for ax, (key, vals) in zip(axes, metrics.items()):
        mean_val = np.nanmean(vals)
        ax.plot(time_ps, vals, lw=0.8, alpha=0.85, label=key)
        ax.axhline(mean_val, color='red', lw=1.2, ls='--', label=f'mean={mean_val:.4g}')

        sel = closest_to_mean_in_window(vals, traj_times, frame_idxs)
        if sel is not None:
            ax.axvline(sel['time_ps'], color='orange', lw=1.0, ls=':',
                       label=f"best t={sel['time_ps']:.1f} ps (frame {sel['frame_idx']})")
            per_metric_selections[key] = sel

        ax.set_ylabel(ylabels[key], fontsize=9)
        ax.legend(fontsize=7, loc='upper right')

    axes[-1].set_xlabel('Time (ps)')
    plt.tight_layout()
    plot_path = os.path.join(per_source_dir, 'analysis.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved plot: {plot_path}")

    for key, sel in per_metric_selections.items():
        print(f"  {key}: best @ {sel['time_ps']:.1f} ps "
              f"(frame {sel['frame_idx']}), value={sel['value']:.4g}, mean={sel['mean']:.4g}")

    return traj_times, densities, as_, bs_, cs_, epots, frame_idxs, per_metric_selections


def main():
    all_results = []

    for cfg in TRAJ_CONFIGS:
        label = cfg['label']
        name  = cfg['init_frame_name']

        # per-source output dir
        per_source_dir = os.path.join(OUTPUT_DIR, label)
        os.makedirs(per_source_dir, exist_ok=True)

        # analysis + plot
        (traj_times, densities, as_, bs_, cs_,
         epots, frame_idxs, per_metric_sel) = analyze_and_plot(cfg, per_source_dir)

        # init frame selection
        print(f"  Selecting init frame...")
        sel = select_init_frame(traj_times, densities, as_, bs_, cs_, frame_idxs)
        print(f"  Selected frame {sel['frame_idx']} @ {sel['time_ps']:.1f} ps | "
              f"density={sel['density']:.4f} g/cm³ (mean={sel['mean_density']:.4f}) | "
              f"a={sel['a_ang']:.2f} b={sel['b_ang']:.2f} c={sel['c_ang']:.2f} Å | "
              f"method={sel['method']}")

        # dump init frame
        frame_dir = dump_init_frame(cfg['traj'], sel['frame_idx'], per_source_dir, name)
        print(f"  Dumped init frame: {frame_dir}")

        # save per-source selection JSON
        result = {
            'label': label,
            'init_frame_name': name,
            'traj_path': cfg['traj'],
            'dt_fs_per_frame': cfg['dt_fs'],
            'init_frame_selection': sel,
            'per_metric_closest_to_mean': per_metric_sel,
        }
        with open(os.path.join(per_source_dir, 'selection.json'), 'w') as f:
            json.dump(result, f, indent=2)

        all_results.append(result)

    out_json = os.path.join(OUTPUT_DIR, 'nvt_init_frame_selection.json')
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved summary JSON: {out_json}")


if __name__ == '__main__':
    main()
