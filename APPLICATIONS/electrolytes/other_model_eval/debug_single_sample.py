#!/usr/bin/env python3
"""
Diagnostic: compare raw model predictions + GT labels between
  (A) custom script path  (load_predict_unit + UMACalculatorWrapper)
  (B) fairchem runner path (fairchem dataset + MLIPEvalUnit / model direct)

Run with the same checkpoint for both paths.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from ase.db import connect

from fairchem.core.units.mlip_unit import InferenceSettings, load_predict_unit

_ELEC = Path(__file__).resolve().parents[1]
if str(_ELEC) not in sys.path:
    sys.path.insert(0, str(_ELEC))
from get_calc import UMACalculatorWrapper


# ── helpers ──────────────────────────────────────────────────────────────────

def _voigt_to_flat(stress):
    s = np.array(stress)
    if s.shape == (6,):
        return s
    if s.shape == (3, 3):
        return np.array([s[0,0], s[1,1], s[2,2], s[1,2], s[0,2], s[0,1]])
    raise ValueError(f"Unexpected stress shape: {s.shape}")


# ── Path A: UMACalculatorWrapper (same as force_energy_stress.py) ─────────

def path_a_predict(ckpt: str, at_orig):
    inference_settings = InferenceSettings(
        tf32=True, activation_checkpointing=False, merge_mole=False,
        compile=False, wigner_cuda=False, external_graph_gen=False,
        internal_graph_gen_version=2,
    )
    predictor = load_predict_unit(
        ckpt, device="cuda", inference_settings=inference_settings,
        overrides={"_target_": "fairchem.core.models.base.HydraModel"},
    )
    calc = UMACalculatorWrapper(predictor, task_name="omol")

    at = at_orig.copy()
    at.set_pbc([True, True, True])
    at.wrap()
    at.calc = calc

    energy = at.get_potential_energy()
    forces = at.get_forces()
    stress_raw = at.calc.results.get("stress")
    stress = _voigt_to_flat(stress_raw) if stress_raw is not None else None
    return energy, forces, stress


# ── Path B: fairchem dataset + model direct call ──────────────────────────

def path_b_predict(ckpt: str, at_orig):
    from fairchem.core.datasets import AseDBDataset
    from fairchem.core.preprocessing import AtomsToGraphs
    from fairchem.core.units.mlip_unit.mlip_unit import load_inference_model

    # Build graph the same way the runner does
    a2g = AtomsToGraphs(
        max_neigh=50, radius=12.0,
        r_energy=True, r_forces=True,
        r_distances=False, r_pbc=True,
    )

    at = at_orig.copy()
    data = a2g.convert(at)
    # add spin/charge from at.info
    data.spin   = torch.tensor([at.info.get("spin", 1)],   dtype=torch.long)
    data.charge = torch.tensor([at.info.get("charge", 0)], dtype=torch.long)

    model = load_inference_model(
        checkpoint_location=ckpt,
        return_checkpoint=False,
        use_ema=True,
        overrides={"backbone": {"otf_graph": True}},
    )
    model = model.cuda().eval()

    batch = data.to("cuda")
    # add batch index
    batch.batch = torch.zeros(batch.num_nodes, dtype=torch.long, device="cuda")
    batch.natoms = torch.tensor([batch.num_nodes], dtype=torch.long, device="cuda")

    with torch.no_grad():
        out = model(batch)

    energy = out["energy"].item() if "energy" in out else None
    forces = out["forces"].cpu().numpy() if "forces" in out else None
    stress = out["stress"].cpu().numpy().flatten() if "stress" in out else None
    return energy, forces, stress


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--idx", type=int, default=0, help="Row index within first shard")
    parser.add_argument("--skip_path_b", action="store_true",
                        help="Skip path B (direct model call) if it fails due to missing a2g setup")
    args = parser.parse_args()

    # Load sample from LMDB
    shard = sorted(args.dataset.glob("data.*.aselmdb"))[0]
    db = connect(str(shard))
    rows = list(db.select())
    row = rows[args.idx]
    at = row.toatoms()
    at.info["charge"] = row.data.get("charge", 0)
    at.info["spin"]   = row.data.get("spin", 1)

    gt_energy = float(at.calc.results["energy"])
    gt_forces = np.array(at.calc.results["forces"])
    gt_stress = _voigt_to_flat(at.calc.results["stress"]) if "stress" in at.calc.results else None
    n_atoms = len(at)

    print(f"\n{'='*60}")
    print(f"Sample idx={args.idx}  |  n_atoms={n_atoms}")
    print(f"GT energy        : {gt_energy:.6f} eV  ({gt_energy/n_atoms:.6f} eV/atom)")
    print(f"GT forces shape  : {gt_forces.shape}  mean|F|={np.abs(gt_forces).mean():.4f} eV/Å")
    if gt_stress is not None:
        print(f"GT stress (Voigt): {gt_stress}")

    # ── Path A ──
    print(f"\n{'─'*60}")
    print("PATH A: load_predict_unit + UMACalculatorWrapper")
    e_a, f_a, s_a = path_a_predict(args.ckpt, at)
    print(f"  Pred energy      : {e_a:.6f} eV  ({e_a/n_atoms:.6f} eV/atom)")
    print(f"  Energy err/atom  : {abs(e_a - gt_energy)/n_atoms:.6f} eV/atom")
    print(f"  Force MAE        : {np.abs(f_a - gt_forces).mean():.6f} eV/Å")
    if s_a is not None and gt_stress is not None:
        print(f"  Stress MAE       : {np.abs(s_a - gt_stress).mean():.6f} eV/Å³")

    # ── Path B ──
    if not args.skip_path_b:
        print(f"\n{'─'*60}")
        print("PATH B: load_inference_model direct call (runner-like)")
        try:
            e_b, f_b, s_b = path_b_predict(args.ckpt, at)
            print(f"  Pred energy (raw): {e_b:.6f} eV  ({e_b/n_atoms:.6f} eV/atom)")

            # Check if path A and B raw outputs match
            print(f"\n  --- Raw output comparison (A vs B) ---")
            print(f"  Energy diff      : {abs(e_a - e_b):.6e} eV")
            if f_b is not None:
                print(f"  Force max diff   : {np.abs(f_a - f_b).max():.6e} eV/Å")
                print(f"  Force mean diff  : {np.abs(f_a - f_b).mean():.6e} eV/Å")

            # Element references from the runner config
            # These are what the runner subtracts before computing energy MAE
            print(f"\n  --- If element refs were applied to GT energy ---")
            print(f"  (See runner config for element_references vector)")
            print(f"  GT energy        : {gt_energy:.6f} eV")
            print(f"  Pred energy (A)  : {e_a:.6f} eV")
            print(f"  Pred energy (B)  : {e_b:.6f} eV")
        except Exception as exc:
            print(f"  Path B failed: {exc}")
            print("  Rerun with --skip_path_b to skip")

    # ── Element reference analysis ──
    print(f"\n{'─'*60}")
    print("ELEMENT REFERENCE ANALYSIS")
    # From the runner config, element refs for the elements in this system
    element_refs = np.array([
        -8.36322661811501e-14, -15.83347671606407, 1.1368683772161603e-13,
        564.8626210931106, -2.7284841053187847e-12, -9.094947017729282e-13,
        -1037.4989166577907, -716.4529145501006, -2047.7160753118965,
        -4728.774560921986, 0.0, -3646.525049008545, 0.0, 0.0, 0.0,
        1998.6885030547571, -5573.63519308328, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 223.1686524368015,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ])
    numbers = at.get_atomic_numbers()
    ref_sum = sum(element_refs[z - 1] for z in numbers if z <= len(element_refs))
    print(f"  Atom composition : {dict(zip(*np.unique(numbers, return_counts=True)))}")
    print(f"  Sum of elem refs : {ref_sum:.6f} eV")
    print(f"  GT energy - refs : {gt_energy - ref_sum:.6f} eV  ({(gt_energy - ref_sum)/n_atoms:.6f} eV/atom)")
    print(f"  Pred(A) - refs   : {e_a - ref_sum:.6f} eV  ({(e_a - ref_sum)/n_atoms:.6f} eV/atom)")
    print(f"  |err| with refs  : {abs(e_a - gt_energy)/n_atoms:.6f} eV/atom  (script way)")
    print(f"  |err| no refs    : {abs((e_a - ref_sum) - (gt_energy - ref_sum))/n_atoms:.6f} eV/atom  (same - refs cancel)")
    print(f"\n  => Energy MAE difference is NOT from element refs (they cancel in subtraction).")
    print(f"     Check: does load_predict_unit already subtract refs internally?")
    print(f"     If pred_energy from path A already has refs removed, then:")
    print(f"       script err = |pred_no_ref - gt_with_ref| / N  (wrong baseline)")
    print(f"       runner err = |pred_no_ref - gt_no_ref| / N    (correct)")
    print(f"  Runner energy err would be: {abs(e_a - (gt_energy - ref_sum))/n_atoms:.6f} eV/atom")


if __name__ == "__main__":
    main()
