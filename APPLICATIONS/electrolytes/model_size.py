"""
Report the number of parameters (in millions) for a UMA checkpoint.

Usage:
    python model_size.py /path/to/inference_ckpt.pt
"""

import argparse
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fairchem.core.units.mlip_unit import load_predict_unit


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    parser = argparse.ArgumentParser(description="Print model size in M parameters")
    parser.add_argument("ckpt", help="Path to inference checkpoint (.pt)")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"ERROR: checkpoint not found: {args.ckpt}")
        sys.exit(1)

    print(f"Loading checkpoint: {args.ckpt}")
    predictor = load_predict_unit(args.ckpt, device="cpu")

    total, trainable = count_parameters(predictor.model)
    print(f"Total parameters:     {total / 1e6:.3f} M")
    print(f"Trainable parameters: {trainable / 1e6:.3f} M")


if __name__ == "__main__":
    main()
