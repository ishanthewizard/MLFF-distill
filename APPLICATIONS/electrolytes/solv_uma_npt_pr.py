import sys
from ase.io import read, Trajectory
from ase.md.npt import NPT
from ase import units
from fairchem.core import pretrained_mlip, FAIRChemCalculator
import torch, os
from copy import deepcopy
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from get_calc import get_uma_calc

# === Get ion type from command line ===
identifier = 'test_traj_naotf'
working_dir = "yuejian/electrolytes/trajs"
uma_path = "logs/202509-0220-1938-d9be/checkpoints/final/inference_ckpt.pt"
# === Input traj and output files ===

# input_traj ="/projects/beyy/shared/data/1mnapf6_solvents_omol_trajs/md_omol_diglyme_pfactor_0.1_1fs_mask_t_re1_s1p1.traj" # DG
# input_traj = "/projects/beyy/shared/data/1mnapf6_solvents_omol_trajs/md_omol_dimethylcarbonate_pfactor_0.1_1fs_mask_t_re3_s1p1.traj" # DMC
# input_traj = "/projects/beyy/shared/data/1mnapf6_solvents_omol_trajs/md_omol_napf6_tgdme_1m_s1p1.traj" # TGDME
# input_traj = "/projects/beyy/shared/data/1mnapf6_solvents_omol_trajs/md_omol_propylene_carbonate_pfactor_0.1_1fs_mask_t_re1_s1p1.traj" # PC
# input_traj = "/projects/beyy/shared/data/1mnapf6_solvents_omol_trajs/md_omol_tetrahydrofuran_pfactor_0.1_1fs_mask_t_re3_m1p1.traj"
# DONT USE!!!!! input_traj = "/projects/beyy/shared/data/1mnapf6_solvents_omol_trajs/md_omol_diethyleneglycol_pfactor_0.1_1fs_mask_t_re3_s1p1.traj"
input_traj = "m4558/distillation_project/all_trajs_min50ps/md_omol_naotf_pc_1m_s1p1.traj"

# diglyn, DMC, 
output_traj = f"{working_dir}/{identifier}.traj"
output_log = f"{working_dir}/md_logs/{identifier}_test.log"

if not os.path.exists(input_traj):
    raise FileNotFoundError(f"Trajectory not found: {input_traj}")

if not os.path.exists(uma_path):
    raise FileNotFoundError(f"UMA model not found: {uma_path}")

# === Load last frame ===
print(f"Loading trajectory from: {input_traj}")
base_structure = read(input_traj, index=0)
base_structure.set_pbc([True, True, True])
#base_structure.wrap()
structure = deepcopy(base_structure)
print(f"Loaded structure with {len(structure)} atoms")

# === Initial velocity ===
MaxwellBoltzmannDistribution(structure, temperature_K=300)

# === Set up model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.set_num_threads(28)
print(f"Loading UMA model from: {uma_path}")
structure.calc = get_uma_calc(uma_path= uma_path, small_model=True)
print("UMA model loaded successfully")


# === Set up NPT dynamics ===
dyn = NPT(
    atoms=structure,
    timestep = 1 * units.fs,
    temperature_K=323,
    externalstress=1.0 * units.bar,
    ttime=100 * units.fs,
    pfactor=0.1, ### larger value mean it will relax slower, typical 10^-3 
    mask=([[1,0,0],[0,1,0],[0,0,1]]),
)

# === Output files ===
# Create output directories if they don't exist
os.makedirs(os.path.dirname(output_traj), exist_ok=True)
os.makedirs(os.path.dirname(output_log), exist_ok=True)

print(f"Output trajectory: {output_traj}")
print(f"Output log: {output_log}")

traj = Trajectory(output_traj, "w", structure)
dyn.attach(traj.write, interval=50)

log_fh = open(output_log, "w", buffering=1)

import time

# Variables to track timing for iterations per second
_last_print_step = [None]
_last_print_time = [None]

def print_status(a=structure, fh=log_fh):
    # Use nonlocal to update the outer variables
    epot = a.get_potential_energy()
    ekin = a.get_kinetic_energy()
    temp = a.get_temperature()
    vol = a.get_volume()
    step = dyn.nsteps

    # Compute iterations per second over the last 20 steps
    its_per_sec_str = ""
    if _last_print_step[0] is not None and _last_print_time[0] is not None:
        steps_since = step - _last_print_step[0]
        time_since = time.time() - _last_print_time[0]
        if steps_since > 0 and time_since > 0:
            its_per_sec = steps_since / time_since
            its_per_sec_str = f" | {its_per_sec:6.2f} it/s"
    # Update last print step/time
    _last_print_step[0] = step
    _last_print_time[0] = time.time()

    line = (f"Step {step:>8} | T={temp:6.1f} K | Epot={epot:10.3f} eV | "
            f"Ekin={ekin:10.3f} eV | Vol={vol:10.3f} Å³{its_per_sec_str}")
    print(line, file=fh)

dyn.attach(print_status, interval=20)

print("Starting NPT molecular dynamics simulation...")
print(f"Target: 1,000,000 steps at 1 fs timestep")
print(f"Temperature: 323 K, Pressure: 1.0 bar")
start_time = time.time()
dyn.run(steps=1000000)
end_time = time.time()
elapsed = end_time - start_time
print(f"NPT run completed in {elapsed:.2f} seconds", file=log_fh)
log_fh.close()
