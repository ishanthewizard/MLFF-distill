"""
TF32 pressure diagnostic for UMA single-point inference.
"""

from __future__ import annotations

import argparse

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
        device="cuda",
        inference_settings=inference_settings,
        overrides={"_target_": "fairchem.core.models.base.HydraModel"},
    )
    return FAIRChemCalculator(predictor, task_name="omol")


def pressure_gpa_from_atoms(atoms) -> float:
    """Run single-point stress and return instantaneous pressure in GPa."""
    stress_voigt = atoms.get_stress(voigt=True)
    return -stress_voigt[:3].mean() / units.GPa


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare pressure from UMA inference with tf32 on/off."
    )
    parser.add_argument("traj_path", type=str, help="Path to trajectory file.")
    parser.add_argument("uma_path", type=str, help="Path to UMA checkpoint.")
    args = parser.parse_args()

    atoms_tf32 = read(args.traj_path, index=10).copy()
    atoms_fp32 = read(args.traj_path, index=10).copy()

    atoms_tf32.calc = build_calc(args.uma_path, tf32=True)
    atoms_fp32.calc = build_calc(args.uma_path, tf32=False)

    pressure_tf32 = pressure_gpa_from_atoms(atoms_tf32)
    pressure_fp32 = pressure_gpa_from_atoms(atoms_fp32)
    pressure_diff = abs(pressure_tf32 - pressure_fp32)

    print(f"traj_path: {args.traj_path}")
    print(f"uma_path:  {args.uma_path}")
    print(f"pressure (tf32=True):  {pressure_tf32:.6f} GPa")
    print(f"pressure (tf32=False): {pressure_fp32:.6f} GPa")
    print(f"|delta pressure|:      {pressure_diff:.6f} GPa")
    print("1 bar = 0.000101 GPa")

    verdict = "FAIL" if pressure_diff > 0.01 else "PASS"
    print(f"verdict: {verdict} (threshold: 0.01 GPa)")


if __name__ == "__main__":
    main()