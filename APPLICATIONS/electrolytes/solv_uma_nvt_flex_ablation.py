"""
Molecular Dynamics Simulation Script with UMA Potential

This script runs NVT (constant temperature and volume) molecular dynamics simulations
using UMA (Universal Machine-learned Atomic) potentials. It supports both single GPU
and multi-GPU parallel execution.

USAGE:
    # Single trajectory with single model (single GPU)
    python solv_uma_nvt_flex_ablation.py /path/to/trajectory --models /path/to/model.ckpt

    # Multiple trajectories with multiple models (multi-GPU)
    python solv_uma_nvt_flex_ablation.py /path/to/traj1 /path/to/traj2 /path/to/traj3 /path/to/traj4 \
        --models /path/to/model1.ckpt /path/to/model2.ckpt /path/to/model3.ckpt /path/to/model4.ckpt

    # With custom parameters
    python solv_uma_nvt_flex_ablation.py /path/to/trajectory --models /path/to/model.ckpt \
        --steps 2000000 --interval 50 --temperature 350 --initial_temperature 300

REQUIRED ARGUMENTS:
    trajectories: Path(s) to trajectory directories containing .traj files
    --models: Path(s) to UMA model checkpoint files (.ckpt)

OPTIONAL ARGUMENTS:
    --steps: Total target steps for simulation (default: 1000000)
    --interval: Interval for trajectory writing and status printing (default: 10)
    --temperature: Simulation temperature in Kelvin (default: 323)
    --initial_temperature: Initial temperature in Kelvin (default: 300)

REQUIREMENTS:
    - CUDA-capable GPU(s)
    - Trajectory files (.traj) in specified directories
    - UMA model checkpoint files (.ckpt)
    - Number of trajectory-model pairs cannot exceed 4
    - Number of trajectory-model pairs cannot exceed number of available GPUs

OUTPUT:
    - Appends to existing .traj files in trajectory directories
    - Creates/updates .log files with simulation status
    - Supports resuming interrupted simulations

EXAMPLES:
    # Basic usage with single trajectory
    python solv_uma_nvt_flex_ablation.py ./my_simulation --models ./uma_model.ckpt

    # Multi-GPU simulation with 4 trajectories
    python solv_uma_nvt_flex_ablation.py ./sim1 ./sim2 ./sim3 ./sim4 \
        --models ./model1.ckpt ./model2.ckpt ./model3.ckpt ./model4.ckpt \
        --steps 500000 --interval 20 --temperature 300
"""

import sys
import os
import time
import signal
import importlib.util
import torch.distributed as dist
import torch.multiprocessing as mp

# Add project root to Python path to make src_v2 module accessible
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from ase.io import read, Trajectory
from ase.md.langevin import Langevin
from ase.md.nvtberendsen import NVTBerendsen
from ase import units
from fairchem.core import pretrained_mlip, FAIRChemCalculator
import torch
import numpy as np
from copy import deepcopy
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from get_calc import get_uma_calc, get_customized_uma_calc

# Reuse common distributed helpers from NPT script to avoid divergence.
_npt_path = os.path.join(os.path.dirname(__file__), "solv_uma_npt_flex_ablation.py")
_spec = importlib.util.spec_from_file_location("solv_uma_npt_flex_ablation", _npt_path)
_npt_module = importlib.util.module_from_spec(_spec)
# Ensure the module is registered so fork/spawn pickling finds the same object
sys.modules[_spec.name] = _npt_module
_spec.loader.exec_module(_npt_module)
setup_distributed = _npt_module.setup_distributed
cleanup_distributed = _npt_module.cleanup_distributed




def worker(rank, trajectory_model_pairs, world_size, interval, total_target_steps, temperatures, initial_temperatures):
    """Thin wrapper to reuse NPT worker with NVT simulate"""
    _npt_module.simulate = simulate
    return _npt_module.worker(rank, trajectory_model_pairs, world_size, interval, total_target_steps, temperatures, initial_temperatures)


def run_parallel_simulations(trajectory_model_pairs, world_size, interval=50, total_target_steps=1000000, temperatures=[323], initial_temperatures=[300]):
    """Thin wrapper to reuse NPT parallel runner with NVT simulate"""
    _npt_module.simulate = simulate
    return _npt_module.run_parallel_simulations(trajectory_model_pairs, world_size, interval=interval, total_target_steps=total_target_steps, temperatures=temperatures, initial_temperatures=initial_temperatures)



def simulate(root_path, rank=None, world_size=None, interval=50, total_target_steps=1000000, model_checkpoint=None, temperature=323, initial_temperature=300):
    """
    Run MD simulation on a specific GPU rank.

    Args:
        rank: GPU rank (0, 1, 2, 3 for 4-GPU node)
        world_size: Total number of GPUs
    """
    # Skip distributed setup if already initialized (when called from worker)
    print(f"Rank {rank}/{world_size}: dist.is_initialized(): {dist.is_initialized()}")
    # print the combination of traj and ckpt path and temperature and initial temperature
    print(f"Trajectory: {root_path}, Model: {model_checkpoint}, Temperature: {temperature}, Initial temperature: {initial_temperature}",flush=True)
    if rank is not None and world_size is not None and not dist.is_initialized():
        setup_distributed(rank, world_size)
        print(f"Rank {rank}/{world_size}: Initialized distributed setup")
    else:
        print(f"Not using distributed setup, single GPU mode")
    # print ckpt and traj path
    print(f"UMA model checkpoint: {model_checkpoint}"+f"Trajectory: {root_path}")
    # === Get ion type from command line ===
    uma_path = model_checkpoint
    # === Input traj and output files ===
    # diglyn, DMC,
    output_traj = f"{root_path}/{os.path.basename(root_path)}.traj"
    output_log = f"{root_path}/{os.path.basename(root_path)}.log"

    if not os.path.exists(output_traj):
        raise FileNotFoundError(f"Trajectory not found: {output_traj}")

    if not os.path.exists(uma_path):
        raise FileNotFoundError(f"UMA model not found: {uma_path}")

    # === Load last frame ===
    print(f"Loading trajectory from: {output_traj}")
    # TODO: This is cpu memory intensive, we should load the last frame only, right now on nersc, when using 80G cpu memory, it's still fine, but we need to modify the code to make it take less memory
    try:
        # Check if trajectory has any frames and get frame count efficiently
        with Trajectory(output_traj, 'r') as traj_reader:
            frame_count = len(traj_reader)
            if frame_count == 0:
                raise ValueError(f"Trajectory file {output_traj} is empty")

            print(f"Trajectory contains {frame_count} frames")

            # Load only the last frame
            base_structure = traj_reader[-1]
            existing_simulated_steps = (frame_count - 1) * interval
    except Exception as e:
        print(f"Error loading trajectory {output_traj}: {e}")
        raise

    base_structure.set_pbc([True, True, True])
    #base_structure.wrap()
    structure = deepcopy(base_structure)
    print(f"Loaded structure with {len(structure)} atoms")

    # Additional validation
    if len(structure) == 0:
        raise ValueError(f"Structure from {output_traj} has no atoms")

    # Check for NaN or infinite positions
    positions = structure.get_positions()
    if not np.isfinite(positions).all():
        raise ValueError(f"Structure from {output_traj} contains invalid positions (NaN or infinite)")

    print(f"Structure validation passed: {len(structure)} atoms, positions are finite")

    # === Initial velocity ===
    # note this is the initial temperature, not the target temperature, as long as it's close to the target temperature, it's fine for system around 300K
    MaxwellBoltzmannDistribution(structure, temperature_K=initial_temperature)

    # === Set up model ===
    if rank is not None:
        device = torch.device(f"cuda:{rank}")
        print(f"Rank {rank}: Using device: {device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
    
    # === Configure CPU threading ===
    # For parallel execution, divide cores among processes to avoid oversubscription
    # For single process, use all available cores
    total_cpu_cores = os.cpu_count()
    if total_cpu_cores is None:
        total_cpu_cores = 1
        print("Warning: Could not detect CPU core count, defaulting to 1 thread")
    
    if world_size is not None and world_size > 1:
        # Multi-GPU mode: divide cores among processes
        # Ensure each process gets at least 1 core
        cores_per_process = max(1, total_cpu_cores // world_size)
        print(f"Multi-GPU mode: Dividing {total_cpu_cores} CPU cores among {world_size} processes",flush=True)
        print(f"Rank {rank}: Allocated {cores_per_process} CPU cores per process",flush=True)
    else:
        # Single GPU mode: use all available cores
        cores_per_process = total_cpu_cores
        print(f"Single GPU mode: Using all {total_cpu_cores} CPU cores",flush=True)
    
    # Set PyTorch to use allocated CPU cores
    torch.set_num_threads(cores_per_process)
    
    # Set environment variables for OpenMP and MKL if not already set
    # These affect NumPy, SciPy, and other libraries that use these backends
    # if 'OMP_NUM_THREADS' not in os.environ:
    #     os.environ['OMP_NUM_THREADS'] = str(cores_per_process)
    # if 'MKL_NUM_THREADS' not in os.environ:
    #     os.environ['MKL_NUM_THREADS'] = str(cores_per_process)
    # if 'NUMEXPR_NUM_THREADS' not in os.environ:
    #     os.environ['NUMEXPR_NUM_THREADS'] = str(cores_per_process)
    
    # print(f"Rank {rank if rank is not None else 'N/A'}: Configured threading - PyTorch={cores_per_process}, OMP={os.environ.get('OMP_NUM_THREADS')}, MKL={os.environ.get('MKL_NUM_THREADS')}")
    print(f"Loading UMA model from: {uma_path}")
    structure.calc = get_customized_uma_calc(uma_path= uma_path)
    # structure.calc = get_uma_calc(uma_path= uma_path, small_model=False)
    print("UMA model loaded successfully")


    # === Set up NVT dynamics (Berendsen thermostat) ===
    # Previous Langevin block kept for quick switches if needed:
    # dyn = Langevin(
    #     atoms=structure,
    #     timestep=1. * units.fs,
    #     temperature=temperature,  # target temperature
    #     friction=0.01 / units.fs,  # 1/fs, mild damping
    # )
    dyn = NVTBerendsen(
    atoms=structure,
    timestep=1.0 * units.fs,
    temperature_K=temperature,
    taut=100 * units.fs
    )

    # === Output files ===
    traj = Trajectory(output_traj, "a", structure)
    dyn.attach(traj.write, interval=interval)

    log_fh = open(output_log, "a", buffering=1)
    # print(f"Already simulated steps: {existing_simulated_steps},total target steps: {total_target_steps}",file=log_fh)
    # Variables to track timing for iterations per second
    _last_print_step = [None]
    _last_print_time = [None]
    
    def print_status(a=structure, fh=log_fh):
        # Use nonlocal to update the outer variables
        epot = a.get_potential_energy()
        ekin = a.get_kinetic_energy()
        temp = a.get_temperature()
        vol = a.get_volume()
        step = existing_simulated_steps + dyn.nsteps

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
        remaining_steps = total_target_steps - step
        remaining_steps_str = f"{remaining_steps}"
        # Extract remaining time into days, hours, minutes, seconds
        if 'its_per_sec' in locals() and its_per_sec > 0:
            remaining_seconds = remaining_steps / its_per_sec
            days = int(remaining_seconds // 86400)
            hours = int((remaining_seconds % 86400) // 3600)
            minutes = int((remaining_seconds % 3600) // 60)
            seconds = int(remaining_seconds % 60)
            remaining_time_str = f"{days}d {hours}h {minutes}m {seconds}s"
        else:
            remaining_time_str = "N/A"
        line = (f"Step {step:>8} | T={temp:6.1f} K | Epot={epot:10.3f} eV | "
                f"Ekin={ekin:10.3f} eV | Vol={vol:10.3f} Å³{its_per_sec_str} | remaining steps: {remaining_steps_str} | remaining time: {remaining_time_str}")
        print(line, file=fh)
    
    dyn.attach(print_status, interval=interval)

    print("Starting NVT (Berendsen) molecular dynamics simulation...")
    print(f"Target: {total_target_steps} steps at 1 fs timestep")
    print(f"Simulation temperature: {temperature} K, Initial temperature: {initial_temperature} K")
    start_time = time.time()
    dyn.run(steps=total_target_steps - existing_simulated_steps)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"NVT run completed in {elapsed:.2f} seconds", file=log_fh)
    log_fh.close()

    # Clean up distributed setup only if we initialized it (single GPU mode)
    if rank is not None and world_size is not None and world_size == 1:
        cleanup_distributed()
        print(f"Rank {rank}: Cleaned up distributed setup")



def main():
    """Delegate CLI handling to NPT script while using NVT simulate"""
    _npt_module.simulate = simulate
    return _npt_module.main()


if __name__ == "__main__":
    main()