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
path = '/data/ishan-amin/OMOL/electrolytes_application/datasets'
identifier = 'napf6_s1p1_val'

# === Input traj and output files ===

input_traj =f"{path}/{identifier}.traj"
output_traj = f"{identifier}_test.traj"
output_log = f"{identifier}_test.log"

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
structure.calc = get_uma_calc()


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
dyn.attach(traj.write, interval=10)

log_fh = open(output_log, "w", buffering=1)

def print_status(a=structure, fh=log_fh):
    epot = a.get_potential_energy()
    ekin = a.get_kinetic_energy()
    temp = a.get_temperature()
    vol = a.get_volume()
    line = (f"Step {dyn.nsteps:>8} | T={temp:6.1f} K | Epot={epot:10.3f} eV | "
            f"Ekin={ekin:10.3f} eV | Vol={vol:10.3f} Å³")
    print(line); print(line, file=fh)

dyn.attach(print_status, interval=10)
dyn.run(steps=100)
log_fh.close()

