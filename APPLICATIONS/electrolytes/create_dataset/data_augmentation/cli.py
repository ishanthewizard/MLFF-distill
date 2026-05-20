from __future__ import annotations

import argparse
import sys
import yaml
from pathlib import Path

# data_augmentation/ must be first so `utils` resolves to utils.py here, not create_dataset/utils/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))  # create_dataset/utils/
sys.path.insert(0, str(Path(__file__).resolve().parent))                   # data_augmentation/
from utils import load_frames, relabel, save_frames, AUGMENTATIONS
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
    args = parser.parse_args()

    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)

    for split in ("train", "val"):
        src = input_dir / split
        if not src.exists():
            continue
        frames = load_frames(src)
        for aug in args.augmentations:
            frames += [a for f in frames if (a := AUGMENTATIONS[aug](f)) is not None]
        relabel(frames, args.calculator_path)
        save_frames(frames, output_dir / split, args.num_workers)

    force_rms, linref_coeff = compute_normalizer_and_linear_reference(
        str(output_dir / "train"), args.num_workers
    )
    with open(output_dir / "species_refs.yaml", "w") as f:
        yaml.dump({"omol_element_refs": linref_coeff}, f, default_flow_style=False)
    with open(output_dir / "force_rms.txt", "w") as f:
        f.write(f"{force_rms}\n")


if __name__ == "__main__":
    main()
