#!/usr/bin/env python3
"""Create a test-set LMDB from a time window of an ASE .traj file.

USAGE
=====
python create_testset.py \\
    --traj /path/to/sim.traj \\
    --dt 100 \\
    --window-start 4.0 \\
    --window-end 20.0 \\
    --n-samples 200 \\
    --output-dir /path/to/output \\
    [--num-workers 1]

WORKFLOW
========
1. Subsample n_samples frames with fixed stride from the given time window.
2. Write frames as a temporary XYZ file inside output_dir.
3. Convert XYZ to ASE LMDB (data.*.aselmdb) via launch_processing.
4. Remove the intermediate XYZ file.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
from pathlib import Path

from ase.io import write

# Add create_dataset root so we can import from utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.subsample_window import subsample_traj_window
from utils.new_create import launch_processing


def main(
    traj_path: str,
    dt_per_frame_fs: float,
    window_start_ns: float,
    window_end_ns: float,
    n_samples: int,
    output_dir: str,
    num_workers: int = 1,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    traj_name = Path(traj_path).stem
    print(f"{'='*60}")
    print(f"create_testset: {traj_name}")
    print(f"  traj   : {traj_path}")
    print(f"  window : {window_start_ns} – {window_end_ns} ns | n_samples: {n_samples} | dt: {dt_per_frame_fs} fs")
    print(f"  output : {output_dir}")

    # Step 1: subsample traj
    print(f"\n[1/3 traj]  Subsampling frames ...")
    frames = subsample_traj_window(
        traj_path=traj_path,
        dt_per_frame_fs=dt_per_frame_fs,
        window_start_ns=window_start_ns,
        window_end_ns=window_end_ns,
        n_samples=n_samples,
    )
    print(f"[1/3 traj]  Done — {len(frames)} frames ready")

    # Step 2: write to a temporary XYZ directory
    tmp_xyz_dir = output_dir / "_tmp_xyz"
    tmp_xyz_dir.mkdir(exist_ok=True)
    xyz_path = tmp_xyz_dir / f"{traj_name}.xyz"

    print(f"\n[2/3 xyz]   Writing {len(frames)} frames -> {xyz_path} ...", end=" ", flush=True)
    write(str(xyz_path), frames)
    xyz_mb = xyz_path.stat().st_size / 1e6
    print(f"done  ({xyz_mb:.1f} MB)")

    # Step 3: XYZ -> LMDB
    print(f"\n[3/3 lmdb]  Converting XYZ -> LMDB at {output_dir}  (workers={num_workers}) ...")
    launch_processing(
        data_dir=str(tmp_xyz_dir),
        output_dir=output_dir,
        num_workers=num_workers,
    )
    lmdb_files = sorted(output_dir.glob("data.*.aselmdb"))
    print(f"  Written {len(lmdb_files)} LMDB shard(s):")
    for lf in lmdb_files:
        print(f"    {lf.name}  ({lf.stat().st_size / 1e6:.1f} MB)")

    # Step 4: remove intermediate XYZ files
    print(f"\n[cleanup]  Removing temporary XYZ files ...")
    for f in glob.glob(str(tmp_xyz_dir / "*.xyz")):
        os.remove(f)
        print(f"  removed {os.path.basename(f)}")
    tmp_xyz_dir.rmdir()

    print(f"\nDone. LMDB written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Subsample a time window from an ASE .traj and write to LMDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--traj", required=True, help="Path to ASE .traj file.")
    parser.add_argument(
        "--dt",
        type=float,
        default=100.0,
        metavar="FS",
        help="Simulation time per saved frame in femtoseconds.",
    )
    parser.add_argument(
        "--window-start",
        type=float,
        required=True,
        metavar="NS",
        help="Start of sampling window in nanoseconds.",
    )
    parser.add_argument(
        "--window-end",
        type=float,
        required=True,
        metavar="NS",
        help="End of sampling window in nanoseconds.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        required=True,
        help="Number of frames to extract from the window (fixed stride).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for LMDB shards.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers for LMDB conversion.",
    )
    args = parser.parse_args()

    main(
        traj_path=args.traj,
        dt_per_frame_fs=args.dt,
        window_start_ns=args.window_start,
        window_end_ns=args.window_end,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
    )
