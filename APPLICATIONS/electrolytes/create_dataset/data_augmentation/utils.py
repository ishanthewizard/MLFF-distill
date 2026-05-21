from __future__ import annotations

from cell_augmentation import diagonal_stretch_only

# AseDBDataset pipeline: ASE Atoms -> AtomicData (fairchem internal graph)
#
# AtomicData fields (example: C576H1440F102Na17O432P17, 2584 atoms):
#   atomic_numbers  [N]          – atomic numbers
#   pos             [N, 3]       – Cartesian positions (Angstrom)
#   cell            [B, 3, 3]    – unit cell matrix
#   pbc             [B, 3]       – periodic boundary flags per direction
#   edge_index      [2, E]       – neighbor list (src/dst); E=0 means no neighbors (cutoff issue)
#   cell_offsets    [E, 3]       – periodic image offsets for each edge
#   fixed           [N]          – frozen-atom mask
#   tags            [N]          – per-atom tags (surface/bulk/adsorbate in OC20 convention)
#   charge          [B]          – total system charge
#   spin            [B]          – total spin multiplicity
#   natoms          [B]          – number of atoms per batch element
#   nedges          [B]          – number of edges per batch element
#   batch           [N]          – batch index per atom
#   sid             [B]          – system ID
#   energy          [B]          – DFT energy label (eV)
#   forces          [N, 3]       – DFT force labels (eV/Ang)
#   stress          [B, 3, 3]    – DFT stress label (eV/Ang^3)
#
# ASE Atoms -> AtomicData field mapping:
#   atoms.get_positions()         -> pos
#   atoms.get_atomic_numbers()    -> atomic_numbers
#   atoms.get_cell()              -> cell
#   atoms.get_pbc()               -> pbc  (True/True/True = periodic in all 3 directions)
#   atoms.get_potential_energy()  -> energy  (stored label via SinglePointCalculator)
#   atoms.get_forces()            -> forces
#
# NOTE: edge_index=0 (no neighbors) indicates neighbor list construction failed —
#       likely caused by wrong pbc/cell at graph-build time or cutoff too small.

from pathlib import Path

import numpy as np
import torch
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from fairchem.core import FAIRChemCalculator
from fairchem.core.datasets import AseDBDataset
from fairchem.core.units.mlip_unit import InferenceSettings, load_predict_unit
from tqdm import tqdm


class _UMACalc(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, predictor):
        Calculator.__init__(self)
        self.fairchem_calc = FAIRChemCalculator(predictor, task_name="omol")

    def calculate(self, atoms, properties=["energy", "forces"], system_changes=None):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.fairchem_calc.calculate(atoms, properties, system_changes)
        self.results = self.fairchem_calc.results.copy()


def _build_calc(uma_path):
    inference_settings = InferenceSettings(
        tf32=True,
        activation_checkpointing=False,
        merge_mole=False,
        compile=False,
        wigner_cuda=False,
        external_graph_gen=False,
        internal_graph_gen_version=2,
    )
    predictor = load_predict_unit(
        uma_path,
        device="cuda",
        inference_settings=inference_settings,
        overrides={"_target_": "fairchem.core.models.base.HydraModel"},
    )
    return _UMACalc(predictor)


def load_frames(src_dir):
    db = AseDBDataset({"src": str(src_dir)})
    # breakpoint()
    return [db.get_atoms(i) for i in tqdm(range(len(db)), desc="Loading")]


def _relabel_worker(rank, chunk, calculator_path, result_queue):
    """Worker that relabels a chunk of frames on a single GPU and puts results in the queue."""
    try:
        calc = _build_calc_on_device(calculator_path, device=f"cuda:{rank}")
        labeled = []
        with torch.no_grad():
            for frame in tqdm(chunk, desc=f"GPU {rank}", position=rank):
                frame.set_pbc(True)
                frame.wrap()

                scaled = frame.get_scaled_positions()
                if (scaled < -1e-6).any() or (scaled > 1 + 1e-6).any():
                    raise ValueError("Atom coordinates are outside the cell")
                frame.calc = calc
                energy = frame.get_potential_energy()
                forces = frame.get_forces().copy()
                results = {"energy": energy, "forces": forces}
                if "stress" in calc.results:
                    results["stress"] = calc.results["stress"].copy()
                frame.calc = SinglePointCalculator(frame, **results)
                labeled.append(frame)
                torch.cuda.empty_cache()
        result_queue.put((rank, labeled))
    except Exception as e:
        result_queue.put((rank, e))


def _build_calc_on_device(uma_path, device="cuda:0"):
    inference_settings = InferenceSettings(
        tf32=True,
        activation_checkpointing=True,
        merge_mole=True,
        compile=False,
        wigner_cuda=False,
        external_graph_gen=False,
        internal_graph_gen_version=2,
    )
    predictor = load_predict_unit(
        uma_path,
        device=device,
        inference_settings=inference_settings,
        overrides={"_target_": "fairchem.core.models.base.HydraModel"},
    )
    return _UMACalc(predictor)


def relabel(frames, calculator_path):
    n_gpus = torch.cuda.device_count()
    if n_gpus <= 1:
        # single GPU path
        calc = _build_calc(calculator_path)
        with torch.no_grad():
            for frame in tqdm(frames, desc="Relabeling"):
                frame.set_pbc(True)
                frame.wrap()

                scaled = frame.get_scaled_positions()
                if (scaled < -1e-6).any() or (scaled > 1 + 1e-6).any():
                    raise ValueError("Atom coordinates are outside the cell")
                frame.calc = calc
                energy = frame.get_potential_energy()
                forces = frame.get_forces().copy()
                results = {"energy": energy, "forces": forces}
                if "stress" in calc.results:
                    results["stress"] = calc.results["stress"].copy()
                frame.calc = SinglePointCalculator(frame, **results)
                torch.cuda.empty_cache()
        return

    # multi-GPU path: split frames evenly across GPUs
    print(f"Relabeling {len(frames)} frames across {n_gpus} GPUs")
    chunks = [frames[i::n_gpus] for i in range(n_gpus)]

    ctx = torch.multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []
    for rank, chunk in enumerate(chunks):
        p = ctx.Process(target=_relabel_worker, args=(rank, chunk, calculator_path, result_queue))
        p.start()
        processes.append(p)

    # collect results in rank order
    results_by_rank = {}
    for _ in range(n_gpus):
        rank, result = result_queue.get()
        if isinstance(result, Exception):
            for p in processes:
                p.terminate()
            raise RuntimeError(f"GPU {rank} worker failed: {result}")
        results_by_rank[rank] = result

    for p in processes:
        p.join()

    # reassemble in original order: chunk[i::n_gpus] -> interleave back
    for rank in range(n_gpus):
        for i, frame in enumerate(results_by_rank[rank]):
            frames[rank + i * n_gpus] = frame


def save_frames(frames, output_dir, num_workers=8):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    natoms = []
    n = min(num_workers, len(frames))
    chunks = [frames[j::n] for j in range(n)]
    for i, chunk in enumerate(chunks):
        if len(chunk) == 0:
            continue
        with connect(str(output_dir / f"data.{i:04d}.aselmdb")) as db:
            for frame in chunk:
                db.write(frame, data=frame.info)
                natoms.append(len(frame))
    np.savez_compressed(output_dir / "metadata.npz", natoms=natoms)


def volume_preserving_distortion(frame, max_stretch=0.10):
    """Apply a random volume-preserving diagonal stretch to an ASE Atoms frame.

    Args:
        frame: ASE Atoms with orthorhombic cell and pbc=True
        max_stretch: max strain magnitude per axis (default 0.10)

    Returns:
        new ASE Atoms with deformed cell and wrapped Cartesian positions
    """
    cell_diag = frame.get_cell().diagonal()  # (3,)
    positions = frame.get_positions()        # (N,3)

    new_cell, new_positions = diagonal_stretch_only(cell_diag, positions, max_stretch)

    new_frame = frame.copy()
    new_frame.set_cell(new_cell)
    new_frame.set_positions(new_positions)
    new_frame.set_pbc(True)
    return new_frame


def sanity_check_labels(frames, calculator_path, n_frames=3, energy_tol=0.1, force_tol=0.5, stress_tol=0.5):
    """Check that UMA predictions on original (unaugmented) frames match stored DFT labels.

    Args:
        frames:          list of ASE Atoms with SinglePointCalculator labels
        calculator_path: path to UMA checkpoint
        n_frames:        number of frames to check (default 3, to keep it fast)
        energy_tol:      max allowed energy difference per atom (eV/atom)
        force_tol:       max allowed force RMSE (eV/Å)
        stress_tol:      max allowed stress MAE (GPa)
    """
    calc = _build_calc(calculator_path)
    frames_to_check = frames[:n_frames]
    print(f"\nSanity check: running UMA on {len(frames_to_check)} original frames")
    all_pass = True
    with torch.no_grad():
        for i, frame in enumerate(frames_to_check):
            # stored DFT labels
            ref_energy = frame.get_potential_energy()
            ref_forces = frame.get_forces().copy()
            ref_stress = frame.get_stress(voigt=True).copy() if frame.calc.results.get("stress") is not None else None

            # UMA prediction
            frame.set_pbc(True)
            frame.wrap()
            frame.calc = calc
            pred_energy = frame.get_potential_energy()
            pred_forces = frame.get_forces().copy()
            pred_stress = calc.results.get("stress")  # (6,) Voigt in eV/Å³

            natoms = len(frame)
            de_per_atom = abs(pred_energy - ref_energy) / natoms
            force_rmse = float(np.sqrt(np.mean((pred_forces - ref_forces) ** 2)))

            # stress: convert eV/Å³ -> GPa (1 eV/Å³ = 160.218 GPa)
            stress_line = ""
            stress_pass = True
            if ref_stress is not None and pred_stress is not None:
                eVA3_to_GPa = 160.218
                stress_mae = float(np.mean(np.abs(pred_stress - ref_stress))) * eVA3_to_GPa
                stress_pass = stress_mae < stress_tol
                stress_line = f"  stress_mae={stress_mae:.4f} GPa"

            passed = de_per_atom < energy_tol and force_rmse < force_tol and stress_pass
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
            print(f"  Frame {i}: dE/atom={de_per_atom:.4f} eV/atom  force_rmse={force_rmse:.4f} eV/Å{stress_line}  [{status}]")

            # restore original labels
            restore = {"energy": ref_energy, "forces": ref_forces}
            if ref_stress is not None:
                restore["stress"] = ref_stress
            frame.calc = SinglePointCalculator(frame, **restore)
            torch.cuda.empty_cache()

    if all_pass:
        print("Sanity check PASSED: UMA predictions are consistent with stored labels.")
    else:
        print("WARNING: Sanity check FAILED — large discrepancy between UMA and stored labels.")
    return all_pass


def rattle(frame):
    return frame


AUGMENTATIONS = {
    "volume_preserving_distortion": volume_preserving_distortion,
    "rattle": rattle,
}
