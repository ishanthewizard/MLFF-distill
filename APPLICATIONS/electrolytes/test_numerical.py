"""
TF32 pressure diagnostic for UMA single-point inference.
"""

from __future__ import annotations

import argparse
import numpy as np
import torch

from ase import units
from ase.io import read
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import InferenceSettings, load_predict_unit


def build_calc(uma_path: str, tf32: bool) -> FAIRChemCalculator:
    """Build an UMA calculator with fixed inference settings except tf32."""
    inference_settings = InferenceSettings(
        tf32=tf32,
        activation_checkpointing=False,
        merge_mole=True,
        compile=True,
        wigner_cuda=False,
        external_graph_gen=False,
        internal_graph_gen_version=2,
    )
    predictor = load_predict_unit(
        uma_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        inference_settings=inference_settings,
        overrides={"_target_": "fairchem.core.models.base.HydraModel"},
    )
    return FAIRChemCalculator(predictor, task_name="omol")


def pressure_gpa_from_atoms(atoms) -> float:
    """Run single-point stress and return instantaneous pressure in GPa."""
    stress_voigt = atoms.get_stress(voigt=True)
    return -stress_voigt[:3].mean() / units.GPa


def pressure_gpa_from_stress_voigt(stress_voigt) -> float:
    """Return instantaneous pressure in GPa from a 6-component Voigt stress."""
    return -stress_voigt[:3].mean() / units.GPa


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare pressure from UMA inference with tf32 on/off."
    )
    parser.add_argument("traj_path", type=str, help="Path to trajectory file.")
    parser.add_argument("uma_path", type=str, help="Path to UMA checkpoint.")
    args = parser.parse_args()

    print(f"traj_path: {args.traj_path}")
    print(f"uma_path:  {args.uma_path}")
    print("1 bar = 0.000101 GPa")

    frames = read(args.traj_path, index=":")
    calc_tf32 = build_calc(args.uma_path, tf32=True)
    calc_fp32 = build_calc(args.uma_path, tf32=False)

    for frame_idx, atoms in enumerate(frames):
        atoms_tf32 = atoms.copy()
        atoms_fp32 = atoms.copy()

        atoms_tf32.calc = calc_tf32
        atoms_fp32.calc = calc_fp32

        forces_tf32 = atoms_tf32.get_forces()
        forces_fp32 = atoms_fp32.get_forces()
        stress_tf32 = atoms_tf32.get_stress(voigt=True)
        stress_fp32 = atoms_fp32.get_stress(voigt=True)

        pressure_tf32 = pressure_gpa_from_stress_voigt(stress_tf32)
        pressure_fp32 = pressure_gpa_from_stress_voigt(stress_fp32)
        pressure_diff = abs(pressure_tf32 - pressure_fp32)
        force_diff = forces_tf32 - forces_fp32
        force_diff_mean = force_diff.mean(axis=0)
        force_diff_std = force_diff.std(axis=0)
        stress_diff_gpa = (stress_tf32 - stress_fp32) / units.GPa

        print(f"\nframe index: {frame_idx}")
        print(f"pressure (tf32=True):  {pressure_tf32:.10f} GPa")
        print(f"pressure (tf32=False): {pressure_fp32:.10f} GPa")
        print(f"|delta pressure|:      {pressure_diff:.10f} GPa")

        verdict = "FAIL" if pressure_diff > 0.01 else "PASS"
        print(f"verdict: {verdict} (threshold: 0.01 GPa)")
        print("\n=== Force directionality analysis ===")
        print(f"force diff mean (x): {force_diff_mean[0]:.10f} eV/Å")
        print(f"force diff mean (y): {force_diff_mean[1]:.10f} eV/Å")
        print(f"force diff mean (z): {force_diff_mean[2]:.10f} eV/Å")
        print(f"force diff std  (x): {force_diff_std[0]:.10f} eV/Å")
        print(f"force diff std  (y): {force_diff_std[1]:.10f} eV/Å")
        print(f"force diff std  (z): {force_diff_std[2]:.10f} eV/Å")

        print("\n=== Stress directionality analysis (GPa) ===")
        print(
            "stress diff [xx, yy, zz, yz, xz, xy]: "
            f"[{stress_diff_gpa[0]:.10f}, {stress_diff_gpa[1]:.10f}, "
            f"{stress_diff_gpa[2]:.10f}, {stress_diff_gpa[3]:.10f}, "
            f"{stress_diff_gpa[4]:.10f}, {stress_diff_gpa[5]:.10f}]"
        )

        systematic_force_bias = bool(np.any(np.abs(force_diff_mean) > 1e-4))
        systematic_stress_bias = bool(np.any(np.abs(stress_diff_gpa) > 1e-4))
        print("\n=== Verdict ===")
        print(f"Systematic force bias: {'YES' if systematic_force_bias else 'NO'}")
        print(f"Systematic stress bias: {'YES' if systematic_stress_bias else 'NO'}")


if __name__ == "__main__":
    main()