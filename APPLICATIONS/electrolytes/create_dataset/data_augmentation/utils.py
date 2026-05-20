from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from fairchem.core.datasets import AseDBDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from get_calc import get_uma_calc


def load_frames(src_dir):
    db = AseDBDataset({"src": str(src_dir)})
    return [db.get_atoms(i) for i in tqdm(range(len(db)), desc="Loading")]


def relabel(frames, calculator_path):
    calc = get_uma_calc(calculator_path, small_model=True)
    for frame in tqdm(frames, desc="Relabeling"):
        frame.calc = calc
        energy = frame.get_potential_energy()
        forces = frame.get_forces().copy()
        results = {"energy": energy, "forces": forces}
        if "stress" in calc.results:
            results["stress"] = calc.results["stress"].copy()
        frame.calc = SinglePointCalculator(frame, **results)


def save_frames(frames, output_dir, num_workers=8):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    natoms = []
    for i, chunk in enumerate(np.array_split(frames, min(num_workers, len(frames)))):
        if len(chunk) == 0:
            continue
        with connect(str(output_dir / f"data.{i:04d}.aselmdb")) as db:
            for frame in chunk:
                db.write(frame, data=frame.info)
                natoms.append(len(frame))
    np.savez_compressed(output_dir / "metadata.npz", natoms=natoms)


def volume_preserving_distortion(frame):
    pass


def rattle(frame):
    pass


AUGMENTATIONS = {
    "volume_preserving_distortion": volume_preserving_distortion,
    "rattle": rattle,
}
