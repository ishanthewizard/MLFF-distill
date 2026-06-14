"""
Scale cell and atomic positions of NVT init frames to match experimental density.
Reads target densities from density_comparison.csv, writes rescaled single-frame
trajectories into scale_up_boxes/match_exp_density/ mirroring the original layout.
"""

import json
import os
import shutil
import pandas as pd
from ase.io import read
from ase.io.trajectory import Trajectory

BASE = "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/nvt_init_frame"
CSV  = os.path.join(BASE, "density_analysis", "density_comparison.csv")
OUT  = os.path.join(BASE, "scale_up_boxes", "match_exp_density")

df = pd.read_csv(CSV)

summary = {}

for _, row in df.iterrows():
    system      = row["system"]          # e.g. student_NaPF6_298K_0.1M
    rho_sim     = row["traj_density_g_cm3"]
    rho_exp     = row["exp_density_g_cm3"]

    src_dir     = os.path.join(BASE, system)
    json_path   = os.path.join(src_dir, "selection.json")

    with open(json_path) as f:
        sel = json.load(f)
    init_frame_name = sel["init_frame_name"]

    traj_path = os.path.join(src_dir, init_frame_name, f"{init_frame_name}.traj")

    # Linear scale factor: density ∝ 1/V, V ∝ L³  →  L_new = L_old * (rho_sim/rho_exp)^(1/3)
    scale = (rho_sim / rho_exp) ** (1.0 / 3.0)

    atoms = read(traj_path, index=0)
    atoms.set_cell(atoms.get_cell() * scale, scale_atoms=False)

    # Verify
    rho_new = atoms.get_masses().sum() * 1.66054e-24 / (atoms.get_volume() * 1e-24)
    print(f"{system}: rho_sim={rho_sim:.4f}  rho_exp={rho_exp:.4f}  "
          f"scale={scale:.5f}  rho_scaled={rho_new:.4f}")

    # Output paths (mirror original layout)
    out_sys_dir = os.path.join(OUT, system)
    out_sub_dir = os.path.join(out_sys_dir, init_frame_name)
    os.makedirs(out_sub_dir, exist_ok=True)

    out_traj = os.path.join(out_sub_dir, f"{init_frame_name}.traj")
    traj_writer = Trajectory(out_traj, "w")
    traj_writer.write(atoms)
    traj_writer.close()

    # Write updated selection.json
    new_sel = json.loads(json.dumps(sel))
    new_sel["init_frame_selection"]["density"] = float(rho_new)
    new_sel["init_frame_selection"]["a_ang"]   = float(atoms.cell.lengths()[0])
    new_sel["init_frame_selection"]["b_ang"]   = float(atoms.cell.lengths()[1])
    new_sel["init_frame_selection"]["c_ang"]   = float(atoms.cell.lengths()[2])
    new_sel["scaling"] = {
        "original_density_g_cm3":    float(rho_sim),
        "target_exp_density_g_cm3":  float(rho_exp),
        "linear_scale_factor":       float(scale),
        "scaled_density_g_cm3":      float(rho_new),
    }

    with open(os.path.join(out_sys_dir, "selection.json"), "w") as f:
        json.dump(new_sel, f, indent=2)

    # Copy log file if present
    src_log = os.path.join(src_dir, init_frame_name, f"{init_frame_name}.log")
    if os.path.exists(src_log):
        shutil.copy2(src_log, os.path.join(out_sub_dir, f"{init_frame_name}.log"))

    summary[system] = {
        "init_frame_name":            init_frame_name,
        "original_density_g_cm3":     float(rho_sim),
        "target_exp_density_g_cm3":   float(rho_exp),
        "scaled_density_g_cm3":       float(rho_new),
        "linear_scale_factor":        float(scale),
        "scaled_cell_a_ang":          float(atoms.cell.lengths()[0]),
        "scaled_cell_b_ang":          float(atoms.cell.lengths()[1]),
        "scaled_cell_c_ang":          float(atoms.cell.lengths()[2]),
        "traj_path":                  out_traj,
    }

summary_path = os.path.join(OUT, "scaled_density_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nDone. Output: {OUT}")
print(f"Summary JSON: {summary_path}")

# --- Sanity check ---
print("\n--- Sanity check ---")
print(f"{'System':<35} {'orig':>6}  {'exp':>6}  {'actual':>8}  {'diff':>10}  cell (Ang)")
for system in sorted(os.listdir(OUT)):
    sys_dir = os.path.join(OUT, system)
    if not os.path.isdir(sys_dir):
        continue
    with open(os.path.join(sys_dir, "selection.json")) as f:
        sel = json.load(f)
    name = sel["init_frame_name"]
    atoms = read(os.path.join(sys_dir, name, f"{name}.traj"), index=0)
    rho_actual = atoms.get_masses().sum() * 1.66054e-24 / (atoms.get_volume() * 1e-24)
    rho_exp    = sel["scaling"]["target_exp_density_g_cm3"]
    rho_orig   = sel["scaling"]["original_density_g_cm3"]
    diff       = rho_actual - rho_exp
    cell       = atoms.cell.lengths()
    print(f"{system:<35} {rho_orig:.4f}  {rho_exp:.4f}  {rho_actual:.6f}  {diff:+.2e}  "
          f"{cell[0]:.3f} x {cell[1]:.3f} x {cell[2]:.3f}")
