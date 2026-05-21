#!/usr/bin/env python3
"""Benchmark script: force, energy, and stress MAE against an ASE-LMDB dataset.

Usage:
    python force_energy_stress.py --ckpt /path/to/ckpt.pt --dataset /path/to/dataset_dir

The dataset directory must contain one or more data.*.aselmdb files (ASE LMDB
shards) with ground-truth energy/forces/stress stored in each row's calc results.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm
from fairchem.core.units.mlip_unit import InferenceSettings, load_predict_unit

_ELEC = Path(__file__).resolve().parents[1]   # electrolytes/
if str(_ELEC) not in sys.path:
    sys.path.insert(0, str(_ELEC))

from get_calc import UMACalculatorWrapper


def get_customized_uma_calc(uma_path: str) -> UMACalculatorWrapper:
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
    return UMACalculatorWrapper(predictor, task_name="omol")


def _iter_dataset(dataset_dir: Path):
    """Yield ASE Atoms objects from all data.*.aselmdb shards, with GT results."""
    from ase.db import connect

    shards = sorted(dataset_dir.glob("data.*.aselmdb"))
    if not shards:
        raise FileNotFoundError(f"No data.*.aselmdb files found in {dataset_dir}")

    for shard in shards:
        db = connect(str(shard))
        for row in db.select():
            at = row.toatoms()
            # charge/spin needed by FAIRChem omol task
            charge = row.data.get("charge", 0)
            spin = row.data.get("spin", 1)
            at.info["charge"] = charge
            at.info["spin"] = spin
            yield at


def _voigt_to_flat(stress) -> np.ndarray:
    """Ensure stress is a flat 6-element Voigt array (eV/Å³)."""
    s = np.array(stress)
    if s.shape == (6,):
        return s
    if s.shape == (3, 3):
        # xx yy zz yz xz xy  (standard Voigt)
        return np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])
    raise ValueError(f"Unexpected stress shape: {s.shape}")


def run_benchmark(ckpt_path: Path, dataset_dir: Path) -> dict:
    calc = get_customized_uma_calc(str(ckpt_path))

    energy_errs = []      # |E_pred - E_ref| / N_atoms  (eV/atom)
    total_energy_errs = []  # |E_pred - E_ref|  (eV)
    force_maes = []            # mean |F_pred - F_ref| per frame (eV/Å)
    force_rmses = []
    force_max_errs = []
    force_mag_errs = []        # mean ||F_pred_i| - |F_ref_i|| per frame (eV/Å)
    force_cos_sims = []        # mean cosine similarity per frame
    force_dir_maes = []        # per-frame mean error per direction, shape (3,): [x, y, z]
    force_dir_spike_errs = []  # per-frame max error per direction (spike atom), shape (3,)
    stress_component_errs = [] # per-frame mean absolute errors, shape (6,): [xx, yy, zz, yz, xz, xy]
    stress_component_max_errs = []  # per-frame max absolute errors, shape (6,)

    for at in tqdm(_iter_dataset(dataset_dir), desc="evaluating frames"):
        gt_energy = float(at.calc.results["energy"])
        gt_forces = np.array(at.calc.results["forces"])           # (N, 3)
        gt_stress = _voigt_to_flat(at.calc.results["stress"])     # (6,)

        n_atoms = len(at)
        at.set_pbc([True, True, True])
        at.wrap()

        # detach ground-truth calc so FAIRChem sees a bare atoms object
        at_pred = at.copy()
        at_pred.calc = calc
        pred_energy = at_pred.get_potential_energy()
        pred_forces = at_pred.get_forces()
        pred_stress_raw = at_pred.calc.results.get("stress")

        # energy errors
        energy_errs.append(abs(pred_energy - gt_energy) / n_atoms)
        total_energy_errs.append(abs(pred_energy - gt_energy))

        # force errors
        f_err = np.abs(pred_forces - gt_forces)          # (N, 3)
        force_maes.append(float(f_err.mean()))
        force_rmses.append(float(np.sqrt((f_err ** 2).mean())))
        force_max_errs.append(float(f_err.max()))
        force_dir_maes.append(f_err.mean(axis=0))        # (3,): mean over atoms per direction
        force_dir_spike_errs.append(f_err.max(axis=0))   # (3,): max atom error per direction

        # force magnitude error: mean | |F_pred_i| - |F_ref_i| |
        pred_mag = np.linalg.norm(pred_forces, axis=1)
        gt_mag = np.linalg.norm(gt_forces, axis=1)
        force_mag_errs.append(float(np.mean(np.abs(pred_mag - gt_mag))))

        # force cosine similarity: mean cos(F_pred_i, F_ref_i)
        denom = pred_mag * gt_mag
        safe = denom > 0
        cos_sim = np.where(safe,
                           np.einsum("ij,ij->i", pred_forces, gt_forces) / np.where(safe, denom, 1.0),
                           0.0)
        force_cos_sims.append(float(cos_sim.mean()))

        # stress errors
        if pred_stress_raw is not None:
            pred_stress = _voigt_to_flat(pred_stress_raw)
            s_err = np.abs(pred_stress - gt_stress)   # (6,)
            stress_component_errs.append(s_err)
            stress_component_max_errs.append(s_err)
              
    _FORCE_DIR_LABELS = ["x", "y", "z"]
    _STRESS_LABELS    = ["xx", "yy", "zz", "yz", "xz", "xy"]

    force_dir_arr   = np.stack(force_dir_maes)        # (N_frames, 3)
    force_spike_arr = np.stack(force_dir_spike_errs)  # (N_frames, 3)

    results = {
        "n_frames": len(energy_errs),
        "energy_mae_eV_per_atom":         float(np.mean(energy_errs)),
        "total_energy_mae_eV":            float(np.mean(total_energy_errs)),
        "force_mae_eV_per_A":             float(np.mean(force_maes)),
        "force_rmse_eV_per_A":            float(np.mean(force_rmses)),
        "force_max_err_eV_per_A":         float(np.mean(force_max_errs)),
        "force_magnitude_error_eV_per_A": float(np.mean(force_mag_errs)),
        "force_cosine_similarity":        float(np.mean(force_cos_sims)),
    }
    for i, lbl in enumerate(_FORCE_DIR_LABELS):
        results[f"force_mae_{lbl}_eV_per_A"]       = float(force_dir_arr[:, i].mean())
        results[f"force_spike_max_{lbl}_eV_per_A"] = float(force_spike_arr[:, i].mean())

    if stress_component_errs:
        mean_arr = np.stack(stress_component_errs)      # (N_frames, 6)
        max_arr  = np.stack(stress_component_max_errs)  # (N_frames, 6)
        per_dir_mean = mean_arr.mean(axis=0)
        per_dir_max  = max_arr.max(axis=0)
        for i, lbl in enumerate(_STRESS_LABELS):
            results[f"stress_mae_{lbl}_eV_per_A3"]     = float(per_dir_mean[i])
            results[f"stress_max_{lbl}_eV_per_A3"]     = float(per_dir_max[i])
        results["stress_mae_eV_per_A3"] = float(per_dir_mean.mean())
    else:
        for lbl in _STRESS_LABELS:
            results[f"stress_mae_{lbl}_eV_per_A3"] = None
            results[f"stress_max_{lbl}_eV_per_A3"] = None
        results["stress_mae_eV_per_A3"] = None

    return results


def _format_results(ckpt: Path, dataset: Path, results: dict) -> str:
    lines = [
        "=" * 60,
        f"Checkpoint            : {ckpt}",
        f"Dataset               : {dataset}",
        "=" * 60,
        f"Frames evaluated      : {results['n_frames']}",
        f"Energy MAE            : {results['energy_mae_eV_per_atom']:.4f}  eV/atom",
        f"Total Energy MAE      : {results['total_energy_mae_eV']:.4f}  eV",
        f"Force  MAE            : {results['force_mae_eV_per_A']:.4f}  eV/Å",
        f"  Force MAE  (x)      : {results['force_mae_x_eV_per_A']:.4f}  eV/Å",
        f"  Force MAE  (y)      : {results['force_mae_y_eV_per_A']:.4f}  eV/Å",
        f"  Force MAE  (z)      : {results['force_mae_z_eV_per_A']:.4f}  eV/Å",
        f"Force  spike-max (x)  : {results['force_spike_max_x_eV_per_A']:.4f}  eV/Å",
        f"Force  spike-max (y)  : {results['force_spike_max_y_eV_per_A']:.4f}  eV/Å",
        f"Force  spike-max (z)  : {results['force_spike_max_z_eV_per_A']:.4f}  eV/Å",
        f"Force  RMSE           : {results['force_rmse_eV_per_A']:.4f}  eV/Å",
        f"Force  max-err (mean) : {results['force_max_err_eV_per_A']:.4f}  eV/Å",
        f"Force  magnitude err  : {results['force_magnitude_error_eV_per_A']:.4f}  eV/Å",
        f"Force  cosine sim     : {results['force_cosine_similarity']:.4f}",
    ]
    if results["stress_mae_eV_per_A3"] is not None:
        lines.append(f"Stress MAE            : {results['stress_mae_eV_per_A3']:.6f}  eV/Å³")
        for lbl in ["xx", "yy", "zz", "yz", "xz", "xy"]:
            lines.append(f"  Stress MAE ({lbl:>2s})    : {results[f'stress_mae_{lbl}_eV_per_A3']:.6f}  eV/Å³"
                         f"  max: {results[f'stress_max_{lbl}_eV_per_A3']:.6f}  eV/Å³")
    else:
        lines.append("Stress MAE            : not available")
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, nargs="+", type=Path,
                        help="One or more checkpoint paths to benchmark")
    parser.add_argument("--dataset", type=Path,
                        default=Path("/global/homes/y/yuejian/project/MLFF-distill/yuejian/relabeled_datasets/outlier_testset_20ns_400frames"),
                        help="Dataset directory with data.*.aselmdb shards")
    parser.add_argument("--dump", type=Path,
                        default=Path("/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/model_checkpoints/ckpt_eval"),
                        help="Directory to dump per-checkpoint .txt files")
    args = parser.parse_args()

    for ckpt in args.ckpt:
        if not ckpt.exists():
            parser.error(f"Checkpoint not found: {ckpt}")
    if not args.dataset.is_dir():
        parser.error(f"Dataset directory not found: {args.dataset}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.dump is not None:
        args.dump.mkdir(parents=True, exist_ok=True)

    all_outputs = []
    for ckpt in args.ckpt:
        print(f"\nEvaluating: {ckpt}")
        results = run_benchmark(ckpt, args.dataset)
        output = _format_results(ckpt, args.dataset, results)
        print(output)
        all_outputs.append(output)

    if args.dump is not None:
        out_file = args.dump / f"results_{ts}.txt"
        out_file.write_text("\n\n".join(all_outputs) + "\n")
        print(f"\nAll results saved to: {out_file}")


if __name__ == "__main__":
    main()
