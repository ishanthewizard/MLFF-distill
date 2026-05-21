"""
Analyze force errors between two UMA models (student vs teacher) on the same trajectory.

This script:
  - Loads an ASE trajectory (.traj)
  - Evaluates forces from a *teacher* UMA checkpoint and a *student* UMA checkpoint
  - Computes per-frame force error statistics: MAE, RMSE, mean |ΔF|, max |ΔF|
  - Saves the results to a CSV file for later plotting/inspection

USAGE (example with the paths you provided):

    python analyze_error.py \
        --traj /global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablate_distillation/debug_run/1/md_omol_naotf_pc_1m_s1p1_10/md_omol_naotf_pc_1m_s1p1_10.traj \
        --student_ckpt /global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablation_ckpt/Ishan_all_salt_wo_hessian/final/inference_ckpt.pt \
        --teacher_ckpt /global/homes/y/yuejian/project/MLFF-distill/m4558/distillation_project/models/uma-s-1p1.pt

You can also pass multiple trajectories:

    python analyze_error.py \
        --traj traj1.traj traj2.traj \
        --student_ckpt path/to/student.pt \
        --teacher_ckpt path/to/teacher.pt

By default, results are written next to each trajectory as
    <traj_basename>_force_error.csv
"""

import argparse
import csv
import os
import sys
from typing import Optional, List

import numpy as np
from ase.io import Trajectory
import matplotlib.pyplot as plt
from tqdm import tqdm

# Make sure we can import get_calc from APPLICATIONS/electrolytes
CURRENT_DIR = os.path.dirname(__file__)
ELECTROLYTES_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ELECTROLYTES_DIR not in sys.path:
    sys.path.insert(0, ELECTROLYTES_DIR)

from get_calc import get_customized_eval_uma_calc as get_customized_uma_calc  # noqa: E402


def build_calculators(student_ckpt: str, teacher_ckpt: str):
    """Create ASE calculators for student and teacher UMA models (on GPU)."""
    print(f"Loading teacher UMA model from: {teacher_ckpt}", flush=True)
    teacher_calc = get_customized_uma_calc(uma_path=teacher_ckpt)
    print("Teacher UMA model loaded.", flush=True)

    print(f"Loading student UMA model from: {student_ckpt}", flush=True)
    student_calc = get_customized_uma_calc(uma_path=student_ckpt)
    print("Student UMA model loaded.", flush=True)

    return student_calc, teacher_calc


def resolve_traj_path(path: str) -> str:
    """
    Resolve a trajectory path.

    - If `path` is a file and ends with '.traj', return it.
    - If `path` is a directory, follow the convention used in MD scripts:
        <dir>/<basename(dir)>.traj
    """
    if os.path.isfile(path):
        if path.endswith(".traj"):
            return path
        raise ValueError(f"Provided file is not a .traj file: {path}")

    if os.path.isdir(path):
        candidate = os.path.join(path, f"{os.path.basename(path)}.traj")
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(
            f"Directory provided but expected trajectory not found: {candidate}"
        )

    raise FileNotFoundError(f"Trajectory path does not exist: {path}")


def analyze_trajectory(
    traj_path: str,
    student_calc,
    teacher_calc,
    max_frames: Optional[int] = None,
    stride: int = 1,
    output_csv: Optional[str] = None,
    plot_path: Optional[str] = None,
) -> List[dict]:
    """
    Analyze one trajectory and return per-frame error statistics.

    For each frame, we compute:
      - natoms
      - MAE over all force components
      - RMSE over all force components
      - mean |ΔF| per atom (vector norm)
      - max |ΔF| per atom (vector norm)
    """
    traj_path = os.path.abspath(traj_path)
    print(f"\n=== Analyzing trajectory: {traj_path} ===", flush=True)

    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    results: List[dict] = []

    with Trajectory(traj_path, "r") as traj:
        n_frames_total = len(traj)
        print(f"Total frames in trajectory: {n_frames_total}", flush=True)

        n_frames_to_process = n_frames_total
        if max_frames is not None:
            n_frames_to_process = min(n_frames_total, max_frames * stride)
        print(f"Processing {n_frames_to_process} frames", flush=True)
        print(f"Stride: {stride}", flush=True)
        frame_counter = 0
        for idx in tqdm(range(0, n_frames_to_process, stride)):
            if max_frames is not None and frame_counter >= max_frames:
                break

            atoms = traj[idx]
            natoms = len(atoms)

            # Create separate Atoms copies so each has its own calculator
            atoms_teacher = atoms.copy()
            atoms_student = atoms.copy()

            atoms_teacher.calc = teacher_calc
            atoms_student.calc = student_calc

            # Compute forces (N_atoms x 3)
            forces_teacher = atoms_teacher.get_forces()
            forces_student = atoms_student.get_forces()

            if forces_teacher.shape != forces_student.shape:
                raise RuntimeError(
                    f"Force shape mismatch at frame {idx}: "
                    f"teacher {forces_teacher.shape}, student {forces_student.shape}"
                )

            delta = forces_student - forces_teacher  # student - teacher

            mae = float(np.mean(np.abs(delta)))
            rmse = float(np.sqrt(np.mean(delta ** 2)))
            # Per-atom vector norms of ΔF
            norms = np.linalg.norm(delta, axis=1)
            mean_norm = float(np.mean(norms))
            max_norm = float(np.max(norms))

            frame_result = {
                "frame_index": idx,
                "natoms": natoms,
                "mae_force_components": mae,
                "rmse_force_components": rmse,
                "mean_dF_norm_per_atom": mean_norm,
                "max_dF_norm_per_atom": max_norm,
            }
            results.append(frame_result)
            frame_counter += 1

            if frame_counter % 10 == 0 or frame_counter == 1:
                print(
                    f"Processed frame {idx}/{n_frames_total - 1} "
                    f"(MAE={mae:.4f}, RMSE={rmse:.4f}, "
                    f"mean|ΔF|={mean_norm:.4f}, max|ΔF|={max_norm:.4f})",
                    flush=True,
                )

    # Optionally write CSV
    if output_csv is not None:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        fieldnames = [
            "frame_index",
            "natoms",
            "mae_force_components",
            "rmse_force_components",
            "mean_dF_norm_per_atom",
            "max_dF_norm_per_atom",
        ]
        print(f"Writing per-frame error metrics to: {output_csv}", flush=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

    # Print simple overall summary
    if results:
        maes = np.array([r["mae_force_components"] for r in results])
        rmses = np.array([r["rmse_force_components"] for r in results])
        mean_norms = np.array([r["mean_dF_norm_per_atom"] for r in results])
        max_norms = np.array([r["max_dF_norm_per_atom"] for r in results])

        print(
            "Overall trajectory statistics "
            f"(based on {len(results)} frames, stride={stride}):",
            flush=True,
        )
        print(f"  MAE over components: mean={maes.mean():.4f}, max={maes.max():.4f}")
        print(f"  RMSE over components: mean={rmses.mean():.4f}, max={rmses.max():.4f}")
        print(
            f"  mean|ΔF| per atom: mean={mean_norms.mean():.4f}, "
            f"max={mean_norms.max():.4f}"
        )
        print(
            f"  max|ΔF| per atom: mean={max_norms.mean():.4f}, "
            f"max={max_norms.max():.4f}"
        )

        # Optionally make a simple MAE vs frame plot
        if plot_path is not None:
            print(f"Saving MAE vs frame plot to: {plot_path}", flush=True)
            frames = [r["frame_index"] for r in results]
            mae_vals = [r["mae_force_components"] for r in results]
            mean_mae = float(maes.mean())

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(frames, mae_vals, marker="o", linestyle="-", linewidth=1, label="MAE per frame")
            # Horizontal line at mean MAE across frames
            ax.axhline(mean_mae, color="red", linestyle="--", linewidth=1.2, label=f"Mean MAE = {mean_mae:.4f}")
            ax.set_xlabel("Frame index")
            ax.set_ylabel("Force MAE (|F_student - F_teacher|, components)")
            ax.set_title(f"Force MAE vs Frame (mean forces mae={mean_mae:.4f}eV)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            fig.savefig(plot_path, dpi=200)
            plt.close(fig)

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze force errors between student and teacher UMA models "
        "on ASE trajectories."
    )
    parser.add_argument(
        "--traj",
        nargs="+",
        required=True,
        help="Path(s) to trajectory files (.traj) or trajectory directories.",
    )
    parser.add_argument(
        "--student_ckpt",
        type=str,
        required=False,
        default="/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/ablation_ckpt/Ishan_all_salt_wo_hessian/final/inference_ckpt.pt",
        help="Path to student UMA checkpoint (.pt/.ckpt).",
    )
    parser.add_argument(
        "--teacher_ckpt",
        type=str,
        required=False,
        default="/global/homes/y/yuejian/project/MLFF-distill/m4558/distillation_project/models/uma-s-1p1.pt",
        help="Path to teacher UMA checkpoint (.pt/.ckpt).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=100,# 1ns for 10fs per frame
        help="Maximum number of frames to process per trajectory (after stride). "
        "If None, process all frames.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1000,
        help="Stride when iterating over frames (e.g., 10 => every 10th frame).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/global/homes/y/yuejian/project/MLFF-distill/APPLICATIONS/electrolytes/observable_scripts/error_analysis",
        help="Directory to store CSV results. Default: same directory as each trajectory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Build calculators once and reuse across trajectories
    student_calc, teacher_calc = build_calculators(
        student_ckpt=args.student_ckpt,
        teacher_ckpt=args.teacher_ckpt,
    )

    for traj_entry in args.traj:
        resolved_traj = resolve_traj_path(traj_entry)
        traj_dir = os.path.dirname(resolved_traj)
        traj_base = os.path.splitext(os.path.basename(resolved_traj))[0]

        # Build identifier using base name plus two parent directory levels
        # Format: "grandparent-parent-base" with dashes
        parent_dir = os.path.basename(traj_dir)
        grandparent_dir = os.path.basename(os.path.dirname(traj_dir))
        # Combine with dashes, replacing any slashes with dashes for safety
        traj_id = f"{grandparent_dir}-{parent_dir}-{traj_base}".replace("/", "-")

        if args.output_dir is not None:
            out_dir = os.path.abspath(args.output_dir)
        else:
            out_dir = traj_dir

        output_csv = os.path.join(out_dir, f"{traj_id}_force_error.csv")
        plot_path = os.path.join(out_dir, f"{traj_id}_force_mae_vs_frame.png")

        analyze_trajectory(
            traj_path=resolved_traj,
            student_calc=student_calc,
            teacher_calc=teacher_calc,
            max_frames=args.max_frames,
            stride=args.stride,
            output_csv=output_csv,
            plot_path=plot_path,
        )


if __name__ == "__main__":
    main()

