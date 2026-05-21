#!/usr/bin/env python3
"""Build the outlier test-set LMDB from 20 ns trajectories.

Samples frames from the anomalous / convergence-failure windows identified
from diffusivity analysis, for both 'original' and 'micro' student models.
Each system's window produces N_SAMPLES frames at fixed stride; all are
combined into a single LMDB at OUTPUT_ROOT.

USAGE
=====
python outlier_testset_20ns.py

OUTPUT
======
OUTPUT_ROOT/data.*.aselmdb  — combined test-set LMDB
OUTPUT_ROOT/metadata.npz
"""

from __future__ import annotations

import glob
import os
import sys
import tempfile
from pathlib import Path

from ase.io import write
print("here")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.subsample_window import subsample_traj_window
from utils.new_create import launch_processing

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DT_FS = 100.0       # femtoseconds per saved frame
N_SAMPLES = 4000     # frames to sample per window
NUM_WORKERS = 4     # parallel workers for LMDB conversion

OUTPUT_ROOT = Path(
    "/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project"
    "/data/lmdb_for_distillation/outlier_testset_20ns_40000frames_large_redo"
)

_ORIG = (
    "/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project"
    "/results/diffusivity_main_results_20ns_final/original_100ps"
)
_MICRO = (
    "/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project"
    "/results/diffusivity_main_results_20ns_final/micro_acas_50ps"
)

# Each entry: (label, traj_path, window_start_ns, window_end_ns)
SYSTEMS = [
    # ---------- original student ----------
    (
        "orig_napf6_dme_0.1M_298K",
        f"{_ORIG}/20ns_solvent_0_1M/298K/md_omol_napf6_dme_re1/md_omol_napf6_dme_re1.traj",
        4.0, 20.0,
    ),
    (
        "orig_naotf_dme_0.1M_298K",
        f"{_ORIG}/20ns_solvent_0_1M/298K/md_omol_naotf_dme_s1p1_omol/md_omol_naotf_dme_s1p1_omol.traj",
        6.5, 20.0,
    ),
    (
        "orig_napf6_dme_0.5M_323K",
        f"{_ORIG}/20ns_solvent_solute_0.5M/323_2K/md_omol_napf6_dme_re1/md_omol_napf6_dme_re1.traj",
        0.1, 20.0,
    ),
    (
        "orig_lipf6_dme_0.5M_323K",
        f"{_ORIG}/20ns_solvent_solute_0.5M/323_2K/md_omol_lipf6_pfactor_0.1_1fs_mask_t/md_omol_lipf6_pfactor_0.1_1fs_mask_t.traj",
        1.5, 20.0,
    ),
    (
        "orig_napf6_diglyme_0.1M_298K",
        f"{_ORIG}/20ns_solvent_0_1M/298K/md_omol_napf6_diglyme_pfactor_0.1_1fs/md_omol_napf6_diglyme_pfactor_0.1_1fs.traj",
        1.5, 20.0,
    ),
    # ---------- micro student ----------
    (
        "micro_napf6_dme_0.1M_298K",
        f"{_MICRO}/20ns_solvent_0_1M/298K/md_omol_napf6_dme_re1/md_omol_napf6_dme_re1.traj",
        4.0, 20.0,
    ),
    (
        "micro_naotf_dme_0.1M_298K",
        f"{_MICRO}/20ns_solvent_0_1M/298K/md_omol_naotf_dme_s1p1_omol/md_omol_naotf_dme_s1p1_omol.traj",
        9.0, 20.0,
    ),
    (
        "micro_napf6_dme_0.5M_323K",
        f"{_MICRO}/20ns_solvent_solute_0.5M/323_2K/md_omol_napf6_dme_re1/md_omol_napf6_dme_re1.traj",
        0.1, 20.0,
    ),
    (
        "micro_lipf6_dme_0.5M_323K",
        f"{_MICRO}/20ns_solvent_solute_0.5M/323_2K/md_omol_lipf6_pfactor_0.1_1fs_mask_t/md_omol_lipf6_pfactor_0.1_1fs_mask_t.traj",
        11.0, 20.0,
    ),
    (
        "micro_napf6_diglyme_0.1M_298K",
        f"{_MICRO}/20ns_solvent_0_1M/298K/md_omol_napf6_diglyme_pfactor_0.1_1fs/md_omol_napf6_diglyme_pfactor_0.1_1fs.traj",
        6.5, 20.0,
    ),
]

# ---------------------------------------------------------------------------


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    tmp_xyz_dir = OUTPUT_ROOT / "_tmp_xyz"
    tmp_xyz_dir.mkdir(exist_ok=True)

    n_systems = len(SYSTEMS)
    all_frames = []
    for idx, (label, traj_path, t_start, t_end) in enumerate(SYSTEMS, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{n_systems}] {label}")
        print(f"  traj   : {traj_path}")
        print(f"  window : {t_start} – {t_end} ns | n_samples: {N_SAMPLES}")

        # --- stage 1: subsample traj ---
        frames = subsample_traj_window(
            traj_path=traj_path,
            dt_per_frame_fs=DT_FS,
            window_start_ns=t_start,
            window_end_ns=t_end,
            n_samples=N_SAMPLES,
        )
        all_frames.extend(frames)
        print(f"  [1/3 traj]  collected {len(frames)} frames  (running total: {len(all_frames)})")

        # --- stage 2: dump XYZ ---
        xyz_path = tmp_xyz_dir / f"{label}.xyz"
        print(f"  [2/3 xyz]   writing -> {xyz_path.name} ...", end=" ", flush=True)
        write(str(xyz_path), frames)
        xyz_mb = xyz_path.stat().st_size / 1e6
        print(f"done  ({xyz_mb:.1f} MB)")

    print(f"\n{'='*60}")
    print(f"All trajs done. Total frames: {len(all_frames)}")
    print(f"XYZ files in: {tmp_xyz_dir}")

    # --- stage 3: XYZ -> LMDB ---
    print(f"\n[3/3 lmdb]  Converting XYZ -> LMDB at {OUTPUT_ROOT}  (workers={NUM_WORKERS}) ...")
    launch_processing(
        data_dir=str(tmp_xyz_dir),
        output_dir=OUTPUT_ROOT,
        num_workers=NUM_WORKERS,
    )
    lmdb_files = sorted(OUTPUT_ROOT.glob("data.*.aselmdb"))
    print(f"  Written {len(lmdb_files)} LMDB shard(s):")
    for lf in lmdb_files:
        print(f"    {lf.name}  ({lf.stat().st_size / 1e6:.1f} MB)")

    # Clean up intermediate XYZ files
    print(f"\n[cleanup]  Removing temporary XYZ files ...")
    for f in glob.glob(str(tmp_xyz_dir / "*.xyz")):
        os.remove(f)
        print(f"  removed {os.path.basename(f)}")
    tmp_xyz_dir.rmdir()

    print(f"\nDone. LMDB written to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
