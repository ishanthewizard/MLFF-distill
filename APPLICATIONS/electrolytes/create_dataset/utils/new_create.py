from __future__ import annotations

import argparse
import glob
import logging
import multiprocessing as mp
import os
from pathlib import Path

import lmdb
import numpy as np
from ase.db import connect
from ase.io import read
from tqdm import tqdm

# try:
#     # When imported as part of the utils package
#     from .compute_ref import compute_normalizer_and_linear_reference
# except ImportError:
#     # When executed as a standalone script:
#     #   python APPLICATIONS/electrolytes/create_dataset/utils/new_create.py ...
#     import sys

#     sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
#     from utils.compute_ref import compute_normalizer_and_linear_reference


logging.basicConfig(level=logging.INFO)


def write_ase_db(mp_arg):
    """
    Write ASE atoms objects to an ASE database file (ASE LMDB backend shard).

    Notes
    -----
    Workers write to a local /tmp shard first, then the result is moved to the
    final destination by the caller. This avoids LMDB file-locking failures on
    HPC network filesystems (Lustre/CFS) that do not support fcntl/flock in
    forked child processes.
    """
    db_file, file_list, worker_id, tmp_db_file = mp_arg

    import os as _os
    import shutil
    import lmdb as _lmdb
    from ase_db_backends.aselmdb import LMDBDatabase
    from ase.io import read as _read
    from pathlib import Path as _Path

    successful = []
    failed = []
    natoms = []

    # Always write to /tmp first; move to final destination on success.
    tmp_path = _Path(tmp_db_file)
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    class _NoLockDB(LMDBDatabase):
        def _open_lmdb_env(self):
            kwargs = dict(subdir=False, meminit=False, map_async=True,
                          readahead=self.readahead, lock=False)
            if self.readonly:
                kwargs["readonly"] = True
            else:
                kwargs["map_size"] = 2 ** 41
            self._env = _lmdb.open(str(self.filename), **kwargs)
            self._env_pid = _os.getpid()

    with _NoLockDB(str(tmp_path), serial=True, readonly=False) as db:
        for file in tqdm(list(file_list), position=worker_id):
            atoms_list = _read(str(file), ":")
            for i, atoms in enumerate(atoms_list):
                try:
                    assert atoms.calc is not None, "No calculator attached to atoms object."
                    assert "energy" in atoms.calc.results, "Missing energy result"
                    assert "forces" in atoms.calc.results, "Missing forces result"
                    db.write(atoms, data=atoms.info)
                    natoms.append(len(atoms))
                    successful.append(f"{file},{i}")
                except AssertionError as err:
                    failed.append(f"{file},{i}: {err!s}")

    # Move from /tmp to the final network-filesystem destination.
    shutil.move(str(tmp_path), str(db_file))

    return db_file, natoms, successful, failed


def launch_processing(data_dir: str | os.PathLike, output_dir: Path, num_workers: int):
    """
    Convert a directory of ASE-readable files into sharded `.aselmdb` files.

    Workers write LMDB shards to /tmp first (avoids LMDB locking failures on
    HPC network filesystems), then move the finished shards to output_dir.
    """
    import tempfile
    os.makedirs(output_dir, exist_ok=True)
    input_files = [
        f
        for f in glob.glob(os.path.join(str(data_dir), "**/*"), recursive=True)
        if os.path.isfile(f)
    ]
    # np.array_split yields numpy arrays whose elements are np.str_ objects;
    # convert to plain Python str lists so ase.io.read accepts them.
    chunked_files = [list(map(str, chunk)) for chunk in np.array_split(input_files, num_workers)]
    db_files = [output_dir / f"data.{i:04d}.aselmdb" for i in range(num_workers)]

    # Unique tmp directory so concurrent runs don't collide.
    tmp_dir = Path(tempfile.mkdtemp(prefix="aselmdb_workers_"))
    tmp_db_files = [tmp_dir / f"data.{i:04d}.aselmdb" for i in range(num_workers)]

    mp_args = [(db_files[i], chunked_files[i], i, tmp_db_files[i]) for i in range(num_workers)]

    try:
        with mp.Pool(num_workers) as pool:
            outputs = pool.map(write_ase_db, mp_args)
    finally:
        # Clean up tmp dir even if workers fail.
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Log results
    natoms = []
    for output in outputs:
        db_file, _natoms, successful, failed = output
        natoms.extend(_natoms)
        log_file = db_file.with_suffix(".log")
        failed_file = db_file.with_suffix(".failed")

        with open(log_file, "w") as log:
            log.write("\n".join(successful))
        with open(failed_file, "w") as failed_log:
            failed_log.write("\n".join(failed))

    np.savez_compressed(output_dir / "metadata.npz", natoms=natoms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-dir",
        type=str,
        required=True,
        help="Directory of ASE atoms objects to convert for training.",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        required=True,
        help="Directory of ASE atoms objects to convert for validation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory to save required finetuning artifacts.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers for processing files.",
    )
    args = parser.parse_args()

    # Launch processing for training data
    train_path = args.output_dir / "train"
    launch_processing(args.train_dir, train_path, args.num_workers)
    _force_rms, _linref_coeff = compute_normalizer_and_linear_reference(
        train_path, args.num_workers
    )

    # Launch processing for validation data
    val_path = args.output_dir / "val"
    launch_processing(args.val_dir, val_path, args.num_workers)

    logging.info(f"Generated dataset at {args.output_dir}")

