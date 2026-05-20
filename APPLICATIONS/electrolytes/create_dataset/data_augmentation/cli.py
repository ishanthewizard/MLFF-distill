"""
Data augmentation pipeline for MLFF training data.

USAGE TEMPLATE:

    python /home/yuejian/project/MLFF-distill/APPLICATIONS/electrolytes/create_dataset/data_augmentation/cli.py \\
        --input-dir  /data/yuejian/electrolyte/test \\
        --output-dir /data/yuejian/electrolyte_augmented \\
        --calculator-path /home/yuejian/project/MLFF-distill/OMol_Whole/ckpt/uma-s-1p1.pt \\
        --augmentations volume_preserving_distortion \\
        --augment-probability 1.0 \\
        --num-workers 8

SANITY CHECK (run UMA on original frames before augmenting, verify labels match):

    python /home/yuejian/project/MLFF-distill/APPLICATIONS/electrolytes/create_dataset/data_augmentation/cli.py \\
        --input-dir  /data/yuejian/electrolyte \\
        --output-dir /data/yuejian/electrolyte_augmented \\
        --calculator-path /home/yuejian/project/MLFF-distill/OMol_Whole/ckpt/uma-s-1p1.pt \\
        --sanity-check --sanity-check-n 3 --stress-tol 0.5

ARGUMENTS:
    --input-dir            Directory with train/ and (optionally) val/ subdirs of .aselmdb files
    --output-dir           Where to write augmented .aselmdb files + species_refs.yaml + force_rms.txt
    --calculator-path      Path to UMA checkpoint (.pt or .ckpt)
    --augmentations        One or more of: volume_preserving_distortion, rattle  (default: volume_preserving_distortion)
    --augment-probability  Fraction of frames to augment, in [0, 1]  (default: 1.0)
    --max-stretch-min      Lower bound of max_stretch sampled per frame  (default: 0.05)
    --max-stretch-max      Upper bound of max_stretch sampled per frame  (default: 0.15)
    --num-workers          Parallel workers for saving  (default: 8)
    --sanity-check         Before augmenting, compare UMA predictions to stored labels
    --sanity-check-n       Number of frames for sanity check  (default: 3)
    --stress-tol           Max stress MAE in GPa for sanity check  (default: 0.5)
"""
from __future__ import annotations

import argparse
import random
import sys
import yaml
from pathlib import Path

# data_augmentation/ must be first so `utils` resolves to utils.py here, not create_dataset/utils/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))  # create_dataset/utils/
sys.path.insert(0, str(Path(__file__).resolve().parent))                   # data_augmentation/
from utils import load_frames, relabel, save_frames, sanity_check_labels, AUGMENTATIONS
from compute_ref import compute_normalizer_and_linear_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--calculator-path", type=str, required=True)
    parser.add_argument("--augmentations", nargs="+",
                        default=["volume_preserving_distortion"],
                        choices=list(AUGMENTATIONS.keys()))
    # augment with probability
    parser.add_argument("--augment-probability", type=float, default=1.0)
    parser.add_argument("--max-stretch-min", type=float, default=0.05,
                        help="Lower bound of the max_stretch range sampled per frame (default: 0.05).")
    parser.add_argument("--max-stretch-max", type=float, default=0.15,
                        help="Upper bound of the max_stretch range sampled per frame (default: 0.15).")
    parser.add_argument("--sanity-check", action="store_true",
                        help="Before augmenting, verify UMA predictions match stored labels on a few frames.")
    parser.add_argument("--sanity-check-n", type=int, default=3,
                        help="Number of frames to use for sanity check (default: 3).")
    parser.add_argument("--stress-tol", type=float, default=0.5,
                        help="Max allowed stress MAE in GPa for sanity check (default: 0.5).")
    args = parser.parse_args()

    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)

    for split in ("train", "val"):
        src = input_dir / split
        if not src.exists():
            continue
        frames = load_frames(src)
        new_frames = []
        if args.sanity_check and split == "train":
            sanity_check_labels(frames, args.calculator_path, n_frames=args.sanity_check_n, stress_tol=args.stress_tol)
        if split == "train":
            for aug in args.augmentations:
                augmented = []
                for f in frames:
                    if random.random() < args.augment_probability:
                        try:
                            max_stretch = random.uniform(args.max_stretch_min, args.max_stretch_max)
                            a = AUGMENTATIONS[aug](f, max_stretch=max_stretch)
                            if a is not None:
                                augmented.append(a)
                        except Exception as e:
                            print(f"[{aug}] augmentation failed for frame {f}: {e}")
                    else:
                        # just add original frame
                        augmented.append(f)
                new_frames += augmented
        relabel(new_frames, args.calculator_path)
        save_frames(new_frames, output_dir / split, args.num_workers)

    force_rms, linref_coeff = compute_normalizer_and_linear_reference(
        str(output_dir / "train"), args.num_workers
    )
    with open(output_dir / "species_refs.yaml", "w") as f:
        yaml.dump({"omol_element_refs": linref_coeff}, f, default_flow_style=False)
    with open(output_dir / "force_rms.txt", "w") as f:
        f.write(f"{force_rms}\n")


if __name__ == "__main__":
    main()
