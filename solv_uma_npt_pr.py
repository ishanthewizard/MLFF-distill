import os
from ase.io import read, Trajectory
from ase.md.npt import NPT
from ase import units
import torch
from copy import deepcopy
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from APPLICATIONS.electrolytes.get_calc import get_distilled_calc
import time

identifier = 'all_napf6_distill_DMC_md_62k_50th_start'
input_traj = '/u/czhang31/data/1mnapf6_solvents_omol_trajs/md_omol_dimethylcarbonate_pfactor_0.1_1fs_mask_t_re3_s1p1.traj'
# input_traj = "/u/czhang31/data/natfsi_s1p1/uma_traj/md_omol_1M_natfsi_s1p1.traj"
# distilled_path = '/u/czhang31/MLFF-distill/logs/202508-0101-1411-6197/checkpoints/step_150000/inference_ckpt.pt' # napf6
# distilled_path = '/u/czhang31/MLFF-distill/logs/202507-3122-0139-17a9/checkpoints/step_150000/inference_ckpt.pt' # natfsi
distilled_path = '/u/czhang31/data/all_NAPF6/distill_62000.pt'
num_steps = 150000 # 150000 steps = 150 ps, 1000000 steps = 1 ns
save_interval = 100
start_index = 50

output_dir = "/u/czhang31/MLFF-distill/APPLICATIONS/electrolytes/md_results"
output_traj = os.path.join(output_dir, f"{identifier}.traj")
output_log = os.path.join(output_dir, f"{identifier}.log")

if not os.path.exists(input_traj):
    raise FileNotFoundError(f"Trajectory not found: {input_traj}")

# === Load first frame ===
base_structure = read(input_traj, index=start_index)
base_structure.set_pbc([True, True, True])
#base_structure.wrap()
structure = deepcopy(base_structure)

# === Initial velocity ===
MaxwellBoltzmannDistribution(structure, temperature_K=300)

# === Set up model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(28)
structure.calc = get_distilled_calc(distilled_path)

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
traj = Trajectory(output_traj, "w", structure)
dyn.attach(traj.write, interval=save_interval)

log_fh = open(output_log, "w", buffering=1)

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

dyn.attach(print_status, interval=save_interval)
dyn.run(steps=num_steps)
log_fh.close()
