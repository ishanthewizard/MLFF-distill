"""
Relabel energy and forces in .aselmdb shards using a UMA model checkpoint.
Reads each shard, runs inference on every frame, writes new shards with
updated energy/forces (and stress if available). All other atoms data
(positions, cell, info, etc.) is preserved unchanged.

Usage:
    python relabel_with_UMA.py \
        --ckpt /path/to/uma-s-1p1.pt \
        --input-dirs /path/to/lmdb_dir1 /path/to/lmdb_dir2 \
        --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lmdb
import numpy as np
from ase_db_backends.aselmdb import LMDBDatabase
from fairchem.core.units.mlip_unit import InferenceSettings, load_predict_unit
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from get_calc import UMACalculatorWrapper


class NoLockLMDBDatabase(LMDBDatabase):
    """LMDBDatabase with file locking disabled — required on $HOME filesystems."""

    def _open_lmdb_env(self):
        if self.readonly:
            self._env = lmdb.open(
                str(self.filename),
                subdir=False, meminit=False, map_async=True,
                readonly=True, lock=False, readahead=self.readahead,
            )
        else:
            self._env = lmdb.open(
                str(self.filename),
                map_size=2**41,
                subdir=False, meminit=False, map_async=True,
                readahead=self.readahead, lock=False,
            )
        import os as _os
        self._env_pid = _os.getpid()


def open_aselmdb(path: Path, readonly: bool):
    return NoLockLMDBDatabase(str(path), readonly=readonly, serial=True)


def get_uma_calc(ckpt_path: str) -> UMACalculatorWrapper:
    """Load UMA for relabeling heterogeneous LMDBs (multiple stoichiometries per shard)."""
    inference_settings = InferenceSettings(
        tf32=False,
        activation_checkpointing=False,
        merge_mole=False,  # merge_mole=True only works when composition is fixed (e.g. MD)
        compile=False,
        wigner_cuda=False,
        external_graph_gen=False,
        internal_graph_gen_version=2,
    )
    predictor = load_predict_unit(
        ckpt_path,
        device="cuda",
        inference_settings=inference_settings,
        overrides={"_target_": "fairchem.core.models.base.HydraModel"},
    )
    return UMACalculatorWrapper(predictor, task_name="omol")


def relabel_shard(src_path: Path, dst_path: Path, calc) -> list[int]:
    """Returns list of natoms for each successfully written frame."""
    natoms_list = []
    skipped = 0
    with open_aselmdb(src_path, readonly=True) as src_db:
        total = src_db.count()
        with open_aselmdb(dst_path, readonly=False) as dst_db:
            for row in tqdm(src_db.select(), total=total, desc=src_path.name, leave=False):

                # breakpoint()
                atoms = row.toatoms()
                atoms.set_pbc([True, True, True])
                atoms.wrap()

                # carry over spin/charge so omol head gets the right inputs
                if "spin" not in atoms.info:
                    atoms.info["spin"] = 1
                if "charge" not in atoms.info:
                    atoms.info["charge"] = 0

                try:
                    calc.calculate(atoms, properties=["energy", "forces"], system_changes=[])
                except Exception as e:
                    print(f"  Skipping frame (row id={row.id}): {e}")
                    skipped += 1
                    continue

                atoms.calc = None  # detach calc before writing

                # rebuild SinglePointCalculator with new labels
                from ase.calculators.singlepoint import SinglePointCalculator

                results = {"energy": calc.results["energy"], "forces": calc.results["forces"]}
                if "stress" in calc.results:
                    results["stress"] = calc.results["stress"]

                atoms.calc = SinglePointCalculator(atoms, **results)

                # preserve any extra data stored in the row
                data = dict(row.data) if row.data else {}
                # keep original info keys (spin, charge, etc.)
                data.update({k: v for k, v in atoms.info.items() if k not in ("energy", "forces", "stress")})
                dst_db.write(atoms, data=data)
                natoms_list.append(len(atoms))
    if skipped:
        print(f"  Skipped {skipped} frames in {src_path.name}")
    return natoms_list


def relabel_dir(input_dir: Path, output_dir: Path, calc) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(input_dir.glob("*.aselmdb"))
    if not shards:
        print(f"  No .aselmdb shards found in {input_dir}, skipping.")
        return

    print(f"  {len(shards)} shards in {input_dir}")
    all_natoms = []
    for shard in shards:
        dst = output_dir / shard.name
        all_natoms.extend(relabel_shard(shard, dst, calc))
    print(f"  Done: {len(all_natoms)} frames written to {output_dir}")

    # use original metadata.npz if present, otherwise recalculate from written frames
    meta = input_dir / "metadata.npz"
    if meta.exists():
        import shutil
        shutil.copy(meta, output_dir / "metadata.npz")
        print(f"  metadata.npz copied from source.")
    else:
        natoms_arr = np.array(all_natoms, dtype=np.int32)
        np.savez(output_dir / "metadata.npz", natoms=natoms_arr)
        print(f"  metadata.npz recalculated: {len(natoms_arr)} frames, natoms range [{natoms_arr.min()}, {natoms_arr.max()}]")


def main():
    parser = argparse.ArgumentParser(description="Relabel LMDB datasets with a UMA model.")
    parser.add_argument("--ckpt", required=True, help="Path to UMA inference checkpoint (.pt)")
    parser.add_argument("--input-dirs", nargs="+", required=True, help="Input .aselmdb dataset directories")
    parser.add_argument("--output-dir", required=True, help="Root output directory")
    args = parser.parse_args()

    print(f"Loading UMA calculator from {args.ckpt}")
    calc = get_uma_calc(args.ckpt)
    print("Calculator loaded.")

    output_root = Path(args.output_dir)
    for input_dir in args.input_dirs:
        input_path = Path(input_dir)
        output_path = output_root / input_path.name
        print(f"\nRelabeling {input_path} -> {output_path}")
        relabel_dir(input_path, output_path, calc)

    print("\nAll done.")


if __name__ == "__main__":
    main()
