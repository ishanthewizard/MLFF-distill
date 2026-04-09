#!/usr/bin/env python3

"""
Trajectory cutter (ASE .traj) for MD resume workflows.

Goal
  Given a simulation `root_path` directory that contains `<basename>.traj`,
  keep only the first N frames and replace the original trajectory with the cut one,
  while preserving per-frame arrays needed for resuming MD (e.g. `momenta`).

What it does
  - Reads:  `<root_path>/<basename>.traj`
  - Writes: `<root_path>/<basename>_cut.traj`  (temporary)
  - If (and only if) cutting succeeds:
      1) deletes the original `<basename>.traj`
      2) renames `<basename>_cut.traj` -> `<basename>.traj`

IMPORTANT
  - This is a destructive operation once it succeeds (the original long trajectory is removed).
  - The cut trajectory is written by copying `Atoms` objects frame-by-frame, which preserves
    arrays like `momenta`/`velocities` stored in the `.traj`, so the last retained frame remains
    resume-ready for scripts like `solv_uma_npt_flex_ablation_resume.py`.

Usage (recommended env)
  source "/u/yjian1/project/MLFF-distill/yjian1/env/miniconda3/etc/profile.d/conda.sh"
  conda activate "/u/yjian1/project/MLFF-distill/yjian1/env/miniconda3/envs/fairchemV2"

  python APPLICATIONS/electrolytes/cutting_traj/cutting_traj.py \
    /path/to/sim_root_dir \
    --keep-frames 100000

Example
  root_path:
    .../md_omol_napf6_dme_re1
  expects:
    .../md_omol_napf6_dme_re1/md_omol_napf6_dme_re1.traj
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def cut_traj(src: str, dst: str, keep_frames: int) -> int:
    """Copy only the first `keep_frames` frames from `src` into `dst`.

    All per-frame arrays (e.g. ``momenta``, ``velocities``, ``initial_charges``)
    are preserved by ASE when we write each ``Atoms`` object, so the last
    retained frame stays fully resume-ready.
    """
    try:
        from ase.io.trajectory import Trajectory  # type: ignore
    except Exception as e:
        print(
            "Failed to import ASE. Make sure you're running inside the conda env "
            "`/u/yjian1/project/MLFF-distill/yjian1/env/miniconda3/envs/fairchemV2`.\n"
            f"Import error: {e}",
            file=sys.stderr,
        )
        return 2

    if keep_frames <= 0:
        print(f"keep_frames must be > 0, got {keep_frames}", file=sys.stderr)
        return 2

    if not os.path.exists(src):
        print(f"Source traj not found: {src}", file=sys.stderr)
        return 2

    # Open source trajectory, get total length.
    try:
        with Trajectory(src, mode="r") as src_traj:
            try:
                total = len(src_traj)
            except Exception as e:
                print(f"Could not determine number of frames: {e}", file=sys.stderr)
                return 2

            if total == 0:
                print(f"Source trajectory {src} has no frames", file=sys.stderr)
                return 2

            if keep_frames > total:
                print(
                    f"Requested keep_frames={keep_frames} but trajectory only has {total} frames.",
                    file=sys.stderr,
                )
                return 2

            print(f"Source traj: {src}")
            print(f"Original total frames: {total}")
            print(f"Cutting to first {keep_frames} frames")

            # Create destination trajectory and copy frames 0..keep_frames-1.
            # We read a frame and write it out as-is; ASE preserves arrays.
            frame_iter = range(keep_frames)
            if tqdm is not None:
                frame_iter = tqdm(frame_iter, total=keep_frames, desc="Cutting frames")
            with Trajectory(dst, mode="w") as dst_traj:
                for i in frame_iter:
                    atoms = src_traj[i]
                    dst_traj.write(atoms)
    except Exception as e:
        print(f"Error while cutting trajectory: {e}", file=sys.stderr)
        return 2

    print(f"Wrote cut trajectory to: {dst}")
    print(f"Final frame count after cut: {keep_frames}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Given a root_path directory, cut its <basename>.traj to first N frames.\n"
            "Writes <basename>_cut.traj in the same directory, then replaces the\n"
            "original .traj with the cut version after a successful cut."
        )
    )
    p.add_argument(
        "root_path",
        help=(
            "Root directory containing <basename>.traj, e.g. "
            "…/md_omol_napf6_dme_re1"
        ),
    )
    p.add_argument(
        "--keep-frames",
        type=int,
        required=True,
        help="Number of initial frames to keep (must be <= total frames).",
    )
    args = p.parse_args(argv)

    root = os.path.abspath(args.root_path)
    base = os.path.basename(root)
    src_traj = os.path.join(root, f"{base}.traj")
    cut_traj_path = os.path.join(root, f"{base}_cut.traj")

    print(f"Root path: {root}")
    print(f"Original traj: {src_traj}")
    print(f"Temporary cut traj: {cut_traj_path}")

    rc = cut_traj(src_traj, cut_traj_path, args.keep_frames)
    if rc != 0:
        print("Cut failed; original trajectory left untouched.", file=sys.stderr)
        return rc

    # Only after a successful cut, replace original with cut version.
    try:
        os.remove(src_traj)
        os.replace(cut_traj_path, src_traj)
        print(f"Replaced original traj with cut traj at: {src_traj}")
    except Exception as e:
        print(
            f"Error while replacing original traj with cut version: {e}",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
