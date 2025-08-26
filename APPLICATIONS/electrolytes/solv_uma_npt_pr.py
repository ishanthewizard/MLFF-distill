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
identifier = 'napf6_DME_1ns'
working_dir = "/projects/beye/iamin/trajs"
uma_path = "/projects/beyy/shared/data/all_NAPF6/distill_131000_hessiancoef80.pt"
# === Input traj and output files ===

# input_traj ="/projects/beyy/shared/data/1mnapf6_solvents_omol_trajs/md_omol_diglyme_pfactor_0.1_1fs_mask_t_re1_s1p1.traj" # DG
# input_traj = "/projects/beyy/shared/data/1mnapf6_solvents_omol_trajs/md_omol_dimethylcarbonate_pfactor_0.1_1fs_mask_t_re3_s1p1.traj" # DMC
# input_traj = "/projects/beyy/shared/data/1mnapf6_solvents_omol_trajs/md_omol_napf6_tgdme_1m_s1p1.traj" # TGDME
# input_traj = "/projects/beyy/shared/data/1mnapf6_solvents_omol_trajs/md_omol_propylene_carbonate_pfactor_0.1_1fs_mask_t_re1_s1p1.traj" # PC
# input_traj = "/projects/beyy/shared/data/1mnapf6_solvents_omol_trajs/md_omol_tetrahydrofuran_pfactor_0.1_1fs_mask_t_re3_m1p1.traj"
# DONT USE!!!!! input_traj = "/projects/beyy/shared/data/1mnapf6_solvents_omol_trajs/md_omol_diethyleneglycol_pfactor_0.1_1fs_mask_t_re3_s1p1.traj"
input_traj = "/projects/beyy/shared/data/napf6_s1p1/uma_traj/md_omol_re5_small_1p1_wrapped.traj"

# diglyn, DMC, 
output_traj = f"{working_dir}/{identifier}.traj"
output_log = f"{working_dir}/md_logs/{identifier}_test.log"

if not os.path.exists(input_traj):
    raise FileNotFoundError(f"Trajectory not found: {input_traj}")

# === Load last frame ===
base_structure = read(input_traj, index=0)
base_structure.set_pbc([True, True, True])
#base_structure.wrap()
structure = deepcopy(base_structure)

# === Initial velocity ===
MaxwellBoltzmannDistribution(structure, temperature_K=300)

# === Set up model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(28)
structure.calc = get_uma_calc(uma_path= uma_path, small_model=True)


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
dyn.attach(traj.write, interval=50)

# Create log directory if it doesn't exist
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

start_time = time.time()
dyn.run(steps=1000000)
end_time = time.time()
elapsed = end_time - start_time
print(f"NPT run completed in {elapsed:.2f} seconds", file=log_fh)
log_fh.close()
