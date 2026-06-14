"""
Molecular Dynamics Simulation Script with UMA Potential

This script runs NPT (constant pressure and temperature) molecular dynamics simulations
using UMA (Universal Machine-learned Atomic) potentials. It supports three execution modes:
1. Single-node single-GPU
2. Single-node multi-GPU  
3. Multi-node multi-GPU

EXECUTION MODES:

1. SINGLE-NODE SINGLE-GPU:
    python solv_uma_npt_flex_ablation_resume.py /path/to/trajectory --models /path/to/model.ckpt

2. SINGLE-NODE MULTI-GPU:
    python solv_uma_npt_flex_ablation_resume.py /path/to/traj1 /path/to/traj2 /path/to/traj3 /path/to/traj4 \
        --models /path/to/model1.ckpt /path/to/model2.ckpt /path/to/model3.ckpt /path/to/model4.ckpt

3. MULTI-NODE MULTI-GPU (via torchrun or SLURM):
    # Using torchrun:
    torchrun --nnodes=2 --nproc_per_node=4 --rdzv_id=12345 --rdzv_backend=c10d \
        --rdzv_endpoint=node1:29500 solv_uma_npt_flex_ablation_resume.py \
        /path/to/traj1 /path/to/traj2 ... --models /path/to/model1.ckpt /path/to/model2.ckpt ...
    
    # Using SLURM with srun:
    srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=4 \
        --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        solv_uma_npt_flex_ablation_resume.py ...

REQUIRED ARGUMENTS:
    trajectories: Path(s) to trajectory directories containing .traj files
    --models: Path(s) to UMA model checkpoint files (.ckpt)

OPTIONAL ARGUMENTS:
    --steps: Total target steps for simulation (default: 1000000)
    --interval: Interval for trajectory writing and status printing (default: 10)
    --temperature: Simulation temperature in Kelvin (default: 323)
    --initial_temperature: Initial temperature in Kelvin (default: 300)
    --timestep: MD timestep in femtoseconds (default: 1.0)

REQUIREMENTS:
    - CUDA-capable GPU(s)
    - Trajectory files (.traj) in specified directories
    - UMA model checkpoint files (.ckpt)
    - For multi-node: proper distributed environment setup

OUTPUT:
    - Appends to existing .traj files in trajectory directories
    - Creates/updates .log files with simulation status
    - Supports resuming interrupted simulations

SLURM SCRIPT EXAMPLE for Multi-Node:
    #!/bin/bash
    #SBATCH --job-name=multinode_md
    #SBATCH --nodes=2
    #SBATCH --ntasks-per-node=4
    #SBATCH --gpus-per-task=1
    #SBATCH --cpus-per-task=8
    
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    export MASTER_PORT=29500
    
    srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=$SLURM_NTASKS_PER_NODE \
        --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        solv_uma_npt_flex_ablation_resume.py /path/to/trajectories --models /path/to/models
"""

import sys
import os
import time
import signal
import socket
import torch.distributed as dist
import torch.multiprocessing as mp

# Add project root to Python path to make src_v2 module accessible
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from ase.io import read, Trajectory
from ase.md.npt import NPT
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.md.nvtberendsen import NVTBerendsen
from ase import units
from fairchem.core import pretrained_mlip, FAIRChemCalculator
import torch
import numpy as np
from copy import deepcopy
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

# Add the electrolytes directory to path to find get_calc
import sys
import os
electrolytes_dir = os.path.dirname(os.path.abspath(__file__))
if electrolytes_dir not in sys.path:
    sys.path.insert(0, electrolytes_dir)

from get_calc import get_uma_calc, get_customized_uma_calc, get_customized_uma_calc_frozen


def get_execution_mode():
    """Determine execution mode based on environment variables"""
    # Check if running with torchrun or distributed launcher
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        if world_size > torch.cuda.device_count():  # Multi-node
            return 'multi_node'
        else:  # Single node multi-GPU
            return 'single_node_multi_gpu'
    elif 'SLURM_PROCID' in os.environ and 'SLURM_NTASKS' in os.environ:
        # SLURM environment without torchrun
        return 'multi_node'
    elif torch.cuda.device_count() > 1:
        return 'single_node_multi_gpu'
    else:
        return 'single_node_single_gpu'

def setup_distributed_multinode():
    """Initialize torch.distributed for multi-node multi-GPU setup"""
    # Handle both torchrun and SLURM environments
    if 'RANK' in os.environ and 'LOCAL_RANK' in os.environ:
        # torchrun environment
        rank = int(os.environ.get('RANK', '0'))  # Global rank across all nodes
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))  # Local GPU rank on this node  
        world_size = int(os.environ.get('WORLD_SIZE', '1'))  # Total processes across all nodes
    else:
        # SLURM environment
        rank = int(os.environ.get('SLURM_PROCID', '0'))  # Global process ID
        world_size = int(os.environ.get('SLURM_NTASKS', '1'))  # Total tasks
        # Calculate local rank from task ID and tasks per node
        ntasks_per_node = int(os.environ.get('SLURM_NTASKS_PER_NODE', '1'))
        local_rank = rank % ntasks_per_node
    
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '29500')
    
    # Set NCCL environment variables for better multi-node communication
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_SOCKET_IFNAME'] = '^docker0,lo'
    
    print(f"Multi-node setup: global_rank={rank}, local_rank={local_rank}, world_size={world_size}")
    print(f"Master endpoint: {master_addr}:{master_port}")
    
    # Check available GPUs and handle SLURM GPU allocation
    num_gpus = torch.cuda.device_count()
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    
    print(f"Available GPUs on this node: {num_gpus}")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    print(f"Global rank {rank}: Calculated local rank: {local_rank}")
    
    # If SLURM assigns only 1 GPU per task, use GPU 0 regardless of local_rank
    if num_gpus == 1:
        actual_device = 0
        print(f"Global rank {rank}: SLURM assigned single GPU, using device 0")
    else:
        actual_device = local_rank
        if local_rank >= num_gpus:
            raise RuntimeError(f"Local rank {local_rank} exceeds available GPUs {num_gpus}")
        print(f"Global rank {rank}: Using calculated local rank {local_rank}")
    
    # Set the GPU device BEFORE initializing process group
    torch.cuda.set_device(actual_device)
    
    # Verify device assignment
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    print(f"Global rank {rank}: Successfully set CUDA device {current_device} ({device_name})")
    print(f"Global rank {rank}: Device {current_device} memory: {torch.cuda.get_device_properties(current_device).total_memory / 1e9:.1f} GB")
    
    # Initialize the process group - remove device_id parameter to avoid conflicts
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size
    )
    
    print(f"Global rank {rank}: Process group initialized successfully")
    return rank, local_rank, world_size

def setup_distributed_single_node(rank, world_size):
    """Initialize torch.distributed for single-node multi-GPU setup"""
    if world_size <= 1:
        print(f"Rank {rank}/{world_size}: No need for distributed setup with single GPU")
        return rank, rank, world_size  # Return consistent format

    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'

    print(f"Single-node multi-GPU setup: rank={rank}, world_size={world_size}")
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    return rank, rank, world_size  # global_rank = local_rank for single node


def cleanup_distributed():
    """Clean up distributed processes"""
    if dist.is_initialized():
        dist.destroy_process_group()




def worker_single_node(
    rank,
    trajectory_model_pairs,
    world_size,
    interval,
    total_target_steps,
    temperatures,
    initial_temperatures,
    timestep_fs_list,
    freeze_graph=False,
    pfactor_list=[0.1],
    dynamics_type_list=["npt"],
):
    """Worker function for single-node execution (spawned by mp.spawn)"""
    try:
        # Initialize distributed setup if running on multiple GPUs
        global_rank, local_rank, world_size = setup_distributed_single_node(rank, world_size)
        if world_size > 1:
            print(f"Rank {rank}/{world_size}: Initialized single-node distributed setup")

        # Get trajectory-model pair assigned to this rank
        if rank < len(trajectory_model_pairs):
            traj_path, model_checkpoint = trajectory_model_pairs[rank]
            # Get the temperature for this specific pair
            temperature = temperatures[rank] if rank < len(temperatures) else temperatures[0]
            initial_temperature = initial_temperatures[rank] if rank < len(initial_temperatures) else initial_temperatures[0]
            timestep_fs = timestep_fs_list[rank] if rank < len(timestep_fs_list) else timestep_fs_list[0]
            pfactor = pfactor_list[rank] if rank < len(pfactor_list) else pfactor_list[0]
            dynamics_type = dynamics_type_list[rank] if rank < len(dynamics_type_list) else dynamics_type_list[0]
            print(f"Rank {rank}: Assigned trajectory: {traj_path}, model: {model_checkpoint}, temperature: {temperature} K, initial_temperature: {initial_temperature} K, timestep: {timestep_fs} fs, dynamics: {dynamics_type}")
        else:
            print(f"Rank {rank}: No trajectory-model pair assigned")
            return

        # Synchronize all ranks before starting
        if world_size > 1:
            dist.barrier()

        # Run simulation for the assigned trajectory-model pair
        print(f"Rank {rank}: Starting simulation for {traj_path} with model {model_checkpoint}")
        try:
            simulate(
                root_path=traj_path,
                rank=local_rank,
                world_size=world_size,
                interval=interval,
                total_target_steps=total_target_steps,
                model_checkpoint=model_checkpoint,
                temperature=temperature,
                initial_temperature=initial_temperature,
                timestep_fs=timestep_fs,
                freeze_graph=freeze_graph,
                pfactor=pfactor,
                dynamics_type=dynamics_type,
            )
            print(f"Rank {rank}: Completed simulation for {traj_path}")
        except Exception as sim_error:
            print(f"Rank {rank}: Simulation failed for {traj_path}: {sim_error}")
            raise

    except Exception as e:
        print(f"Rank {rank}: Error in worker: {e}")
        raise
    finally:
        # Synchronize all ranks before cleanup
        if world_size > 1 and dist.is_initialized():
            print(f"Rank {rank}: Waiting for all workers to complete before cleanup...")
            dist.barrier()
            print(f"Rank {rank}: All workers completed, proceeding with cleanup")
        
        # Clean up distributed setup
        if world_size > 1:
            if dist.is_initialized():
                cleanup_distributed()
                print(f"Rank {rank}: Cleaned up distributed setup")


def worker_multinode(
    trajectory_model_pairs,
    interval,
    total_target_steps,
    temperatures,
    initial_temperatures,
    timestep_fs_list,
    freeze_graph=False,
    pfactor_list=[0.1],
    dynamics_type_list=["npt"],
):
    """Worker function for multi-node execution (called directly, no mp.spawn)"""
    global_rank, local_rank, world_size = setup_distributed_multinode()
    
    try:
        print(f"Global rank {global_rank} (local rank {local_rank}): Initialized multi-node distributed setup")

        # Get trajectory-model pair assigned to this global rank
        if global_rank < len(trajectory_model_pairs):
            traj_path, model_checkpoint = trajectory_model_pairs[global_rank]
            temperature = temperatures[global_rank] if global_rank < len(temperatures) else temperatures[0]
            initial_temperature = initial_temperatures[global_rank] if global_rank < len(initial_temperatures) else initial_temperatures[0]
            timestep_fs = timestep_fs_list[global_rank] if global_rank < len(timestep_fs_list) else timestep_fs_list[0]
            pfactor = pfactor_list[global_rank] if global_rank < len(pfactor_list) else pfactor_list[0]
            dynamics_type = dynamics_type_list[global_rank] if global_rank < len(dynamics_type_list) else dynamics_type_list[0]

            print(f"Global rank {global_rank} (local rank {local_rank}): Assigned trajectory: {traj_path}")
            print(f"Global rank {global_rank}: Model: {model_checkpoint}, Temperature: {temperature} K, Initial: {initial_temperature} K, Timestep: {timestep_fs} fs, Dynamics: {dynamics_type}")
        else:
            print(f"Global rank {global_rank}: No trajectory-model pair assigned")
            return

        # Run simulation independently (no distributed synchronization needed)
        print(f"Global rank {global_rank} (local rank {local_rank}): Starting simulation")
        # With --gpus-per-task=1, each task only sees GPU 0 in its isolated context
        # Run as single GPU mode since each task is independent
        simulate(
            root_path=traj_path,
            rank=None,  # Single GPU mode - no distributed coordination needed
            world_size=None,
            interval=interval,
            total_target_steps=total_target_steps,
            model_checkpoint=model_checkpoint,
            temperature=temperature,
            initial_temperature=initial_temperature,
            timestep_fs=timestep_fs,
            freeze_graph=freeze_graph,
            pfactor=pfactor,
            dynamics_type=dynamics_type,
        )
        print(f"Global rank {global_rank}: Completed simulation")
        
    except Exception as e:
        print(f"Global rank {global_rank}: Error in multi-node worker: {e}")
        raise
    finally:
        # Since we're running independent simulations, clean up distributed only if it was used
        if dist.is_initialized():
            print(f"Global rank {global_rank}: Cleaning up distributed setup")
            cleanup_distributed()
            print(f"Global rank {global_rank}: Cleaned up distributed setup")


def run_parallel_simulations_single_node(
    trajectory_model_pairs,
    world_size,
    interval=50,
    total_target_steps=1000000,
    temperatures=[323],
    initial_temperatures=[300],
    timestep_fs_list=[1.0],
    freeze_graph=False,
    pfactor_list=[0.1],
    dynamics_type_list=["npt"],
):
    """Main function to run parallel simulations across multiple GPUs"""
    # Pick a per-job free local port to avoid collisions with other jobs on the same node.
    # Child processes spawned by torch.multiprocessing inherit this env var.
    if 'MASTER_PORT' not in os.environ:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            os.environ['MASTER_PORT'] = str(s.getsockname()[1])
    os.environ.setdefault('MASTER_ADDR', 'localhost')

    # Spawn worker processes
    print(f"Starting parallel simulations on {world_size} GPUs")
    print(f"Distributed master endpoint: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
    print(f"Total trajectory-model pairs: {len(trajectory_model_pairs)}")
    print(f"Temperatures: {temperatures}")
    print(f"Initial temperatures: {initial_temperatures}")
    print(f"Timesteps: {timestep_fs_list} fs")

    mp.spawn(
        worker_single_node,
        args=(
            trajectory_model_pairs,
            world_size,
            interval,
            total_target_steps,
            temperatures,
            initial_temperatures,
            timestep_fs_list,
            freeze_graph,
            pfactor_list,
            dynamics_type_list,
        ),
        nprocs=world_size,
        join=True
    )
    print("All single-node parallel simulations completed successfully!")


def run_parallel_simulations_multinode(
    trajectory_model_pairs,
    interval=50,
    total_target_steps=1000000,
    temperatures=[323],
    initial_temperatures=[300],
    timestep_fs_list=[1.0],
    freeze_graph=False,
    pfactor_list=[0.1],
    dynamics_type_list=["npt"],
):
    """Main function to run multi-node parallel simulations (no mp.spawn needed)"""
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    
    print(f"Starting multi-node parallel simulations across {world_size} processes")
    print(f"Total trajectory-model pairs: {len(trajectory_model_pairs)}")
    print(f"Temperatures: {temperatures}")
    print(f"Initial temperatures: {initial_temperatures}")
    print(f"Timesteps: {timestep_fs_list} fs")

    # Call worker directly (no mp.spawn for multi-node)
    worker_multinode(
        trajectory_model_pairs=trajectory_model_pairs,
        interval=interval,
        total_target_steps=total_target_steps,
        temperatures=temperatures,
        initial_temperatures=initial_temperatures,
        timestep_fs_list=timestep_fs_list,
        freeze_graph=freeze_graph,
        pfactor_list=pfactor_list,
        dynamics_type_list=dynamics_type_list,
    )
    print("Multi-node parallel simulations completed successfully!")



def simulate(
    root_path,
    rank=None,
    world_size=None,
    interval=50,
    total_target_steps=1000000,
    model_checkpoint=None,
    temperature=323,
    initial_temperature=300,
    timestep_fs=1.0,
    freeze_graph=False,
    pfactor=0.1,
    dynamics_type="npt",
):
    """
    Run MD simulation on a specific GPU rank.

    Args:
        rank: GPU rank (0, 1, 2, 3 for 4-GPU node)
        world_size: Total number of GPUs
    """
    # Print simulation info
    print(f"Trajectory: {root_path}, Model: {model_checkpoint}, Temperature: {temperature}, Initial temperature: {initial_temperature}",flush=True)
    
    # Distributed setup is handled by the caller (worker functions)
    if rank is not None and world_size is not None:
        print(f"Rank {rank}/{world_size}: dist.is_initialized(): {dist.is_initialized()}")
        if world_size > 1 and not dist.is_initialized():
            print(f"Warning: Rank {rank}: Expected distributed to be initialized but it's not")
    else:
        print(f"Single GPU mode: rank={rank}, world_size={world_size}")
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
    # For resume runs: keep velocities/momenta from the last frame when available.
    # Fall back to Maxwell-Boltzmann initialization only if velocity info is absent.
    has_saved_velocity = (
        ("momenta" in structure.arrays) or ("velocities" in structure.arrays)
    )
    if has_saved_velocity:
        print("Resume mode: using velocity/momenta from the last trajectory frame")
    else:
        print(
            "No velocity/momenta found in last frame; "
            f"initializing Maxwell-Boltzmann at {initial_temperature} K"
        )
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
    structure.calc = get_customized_uma_calc(uma_path=uma_path)
    print("UMA model loaded successfully")

    if dynamics_type == "npt":
        # === Set up NPT dynamics ===
        dyn = NPT(
            atoms=structure,
            timestep=timestep_fs * units.fs,
            temperature_K= temperature, # 298.2,
            externalstress=1.0 * units.bar,
            ttime=100 * units.fs,
            pfactor=pfactor,
            mask=([[1,0,0],[0,1,0],[0,0,1]]),
        )
    elif dynamics_type == "nvt_berendsen":
        # === Set up NVT dynamics (Berendsen thermostat) ===
        dyn = NVTBerendsen(
        atoms=structure,
        timestep=1.0 * units.fs,
        temperature_K=temperature,
        taut=100 * units.fs
        )
    elif dynamics_type == "nvt_noose_hoover":
        dyn = NPT(
            atoms=structure,
            timestep=timestep_fs * units.fs,
            temperature_K= temperature, # 298.2,
            externalstress=1.0 * units.bar,
            ttime=100 * units.fs,
            pfactor=None,
            mask=([[1,0,0],[0,1,0],[0,0,1]]),
        )
    elif dynamics_type == "isotropic_MTK_nose_hoover_npt":
        dyn = IsotropicMTKNPT(
            atoms = structure,
            timestep      = timestep_fs * units.fs,
            temperature_K = temperature,
            pressure_au   = 1.0 * units.bar,
            tdamp  = 100  * timestep_fs * units.fs,   # 100 fs — 100x timestep
            pdamp  = 1000 * timestep_fs * units.fs,   # 1000 fs — 1000x timestep
            tchain = 3,                 # Nose-Hoover chain length
            pchain = 3,                 # barostat chain length
            tloop  = 1,                 # thermostat sub-steps
            ploop  = 1                 # barostat sub-steps
        )

    else:
        raise ValueError(f"Invalid dynamics type: {dynamics_type}")

    # # === Output files ===
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

    print("Starting NPT molecular dynamics simulation...")
    print(f"Target: {total_target_steps} steps at {timestep_fs} fs timestep")
    print(f"Simulation temperature: {temperature} K, Initial temperature: {initial_temperature} K, Pressure: 1.0 bar")
    start_time = time.time()
    dyn.run(steps=total_target_steps - existing_simulated_steps)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"NPT run completed in {elapsed:.2f} seconds", file=log_fh)
    log_fh.close()

    # Clean up distributed setup only if we initialized it (single GPU mode)
    if rank is not None and world_size is not None and world_size == 1:
        cleanup_distributed()
        print(f"Rank {rank}: Cleaned up distributed setup")



def parse_arguments():
    """Parse command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run MD simulations with UMA potential - supports single-node single-GPU, single-node multi-GPU, and multi-node multi-GPU')
    parser.add_argument('trajectories', nargs='+', 
                       help='Path(s) to trajectory directories (space-separated)')
    parser.add_argument('--models', nargs='+', required=True,
                       help='Path(s) to UMA model checkpoint files (space-separated)')
    parser.add_argument('--steps', type=int, default=1000000,
                       help='Total target steps for simulation (default: 1000000)')
    parser.add_argument('--interval', type=int, default=10,
                       help='Interval for trajectory writing and status printing (default: 10)')
    parser.add_argument('--temperature', type=float, nargs='+', default=[323],
                       help='Temperature(s) for simulation (default: 323 K). Can specify multiple temperatures, one per trajectory.')
    parser.add_argument('--initial_temperature', type=float, nargs='+', default=[300],
                       help='Initial temperature(s) for simulation (default: 300 K). Can specify multiple temperatures, one per trajectory.')
    parser.add_argument(
        '--timestep',
        type=float,
        nargs='+',
        default=[1.0],
        dest='timestep_fs_list',
        help='MD timestep(s) in femtoseconds (default: 1.0). Can specify multiple values, one per trajectory.',
    )
    # --- FREEZE_GRAPH CHANGE: add --freeze_graph flag; remove this block to revert ---
    parser.add_argument(
        '--freeze_graph',
        action='store_true',
        default=False,
        help='Freeze graph after step 0 (skip OTF graph regen). Speed benchmark only — physics will be wrong.',
    )
    # --- END FREEZE_GRAPH CHANGE ---
    parser.add_argument(
        '--pfactor',
        type=float,
        nargs='+',
        default=[0.1],
        help='Pressure relaxation factor(s) for NPT barostat (default: 0.1). Can specify multiple values, one per trajectory. Larger value = slower pressure relaxation, typical 10^-3.',
    )
    parser.add_argument(
        '--dynamics_type',
        type=str,
        nargs='+',
        default=["npt"],
        help='Dynamics type per trajectory: npt | nvt_berendsen | nvt_noose_hoover | isotropic_MTK_nose_hoover_npt (default: npt). Can specify multiple values, one per trajectory.',
    )
    return parser.parse_args()


def validate_inputs(trajectory_paths, model_checkpoints, temperatures, initial_temperatures, timestep_fs_list, execution_mode, dynamics_type_list=None):
    """Validate input arguments"""
    # Basic validation
    if len(timestep_fs_list) != len(trajectory_paths):
        raise ValueError(f"Number of timesteps ({len(timestep_fs_list)}) must equal number of trajectories ({len(trajectory_paths)})")

    for timestep_fs in timestep_fs_list:
        if timestep_fs <= 0:
            raise ValueError(f"timestep must be > 0 fs, got {timestep_fs}")
    
    # === Validation ===
    # Check that trajectory and model lists have the same length
    if len(trajectory_paths) != len(model_checkpoints):
        raise ValueError(f"Number of trajectories ({len(trajectory_paths)}) must equal number of models ({len(model_checkpoints)})")
    
    # Check that temperature arrays have the same length as trajectories
    if len(temperatures) != len(trajectory_paths):
        raise ValueError(f"Number of temperatures ({len(temperatures)}) must equal number of trajectories ({len(trajectory_paths)})")
    
    if len(initial_temperatures) != len(trajectory_paths):
        raise ValueError(f"Number of initial temperatures ({len(initial_temperatures)}) must equal number of trajectories ({len(trajectory_paths)})")

    valid_dynamics = {"npt", "nvt_berendsen", "nvt_noose_hoover", "isotropic_MTK_nose_hoover_npt"}
    if dynamics_type_list is not None:
        if len(dynamics_type_list) != len(trajectory_paths):
            raise ValueError(f"Number of dynamics types ({len(dynamics_type_list)}) must equal number of trajectories ({len(trajectory_paths)})")
        for dt in dynamics_type_list:
            if dt not in valid_dynamics:
                raise ValueError(f"Invalid dynamics type '{dt}'. Must be one of {valid_dynamics}")

    # Mode-specific validation
    if execution_mode == 'single_node_single_gpu' or execution_mode == 'single_node_multi_gpu':
        world_size = torch.cuda.device_count()
        if world_size == 0:
            raise RuntimeError("No CUDA devices available")
            
        # Check GPU limits for single node
        if len(trajectory_paths) > world_size:
            raise ValueError(f"Number of trajectory-model pairs ({len(trajectory_paths)}) is greater than the number of GPUs ({world_size})")
    
    # Verify all model checkpoints exist
    for model_checkpoint in model_checkpoints:
        if not os.path.exists(model_checkpoint):
            raise FileNotFoundError(f"Model checkpoint not found: {model_checkpoint}")

    # Verify all trajectory directories exist
    for traj_path in trajectory_paths:
        if not os.path.exists(traj_path):
            raise FileNotFoundError(f"Trajectory directory not found: {traj_path}")


def main():
    """Main function that parses command line arguments and runs simulations"""
    # Determine execution mode
    execution_mode = get_execution_mode()
    print(f"Detected execution mode: {execution_mode}")
    
    # Parse arguments
    args = parse_arguments()
    
    # Extract configuration
    total_target_steps = args.steps
    interval = args.interval
    model_checkpoints = args.models
    trajectory_paths = args.trajectories
    temperatures = args.temperature
    initial_temperatures = args.initial_temperature
    timestep_fs_list = args.timestep_fs_list
    freeze_graph = args.freeze_graph  # --- FREEZE_GRAPH CHANGE ---
    pfactor_list = args.pfactor
    dynamics_type_list = args.dynamics_type

    print(f"Target steps: {total_target_steps}")
    print(f"Interval: {interval}")
    print(f"Model checkpoints: {model_checkpoints}")
    print(f"Trajectory paths: {trajectory_paths}")
    print(f"Temperatures: {temperatures}")
    print(f"Initial temperatures: {initial_temperatures}")
    print(f"Timesteps: {timestep_fs_list} fs")
    print(f"Pfactor list: {pfactor_list}")
    print(f"Dynamics types: {dynamics_type_list}")

    # Validate inputs
    validate_inputs(trajectory_paths, model_checkpoints, temperatures, initial_temperatures, timestep_fs_list, execution_mode, dynamics_type_list)
    
    # Create trajectory-model pairs
    trajectory_model_pairs = list(zip(trajectory_paths, model_checkpoints))
    print(f"Created {len(trajectory_model_pairs)} trajectory-model pairs")
    
    # Run simulations based on execution mode
    if execution_mode == 'multi_node':
        print("Running in multi-node multi-GPU mode")
        run_parallel_simulations_multinode(
            trajectory_model_pairs=trajectory_model_pairs,
            interval=interval,
            total_target_steps=total_target_steps,
            temperatures=temperatures,
            initial_temperatures=initial_temperatures,
            timestep_fs_list=timestep_fs_list,
            freeze_graph=freeze_graph,
            pfactor_list=pfactor_list,
            dynamics_type_list=dynamics_type_list,
        )

    elif execution_mode == 'single_node_multi_gpu':
        world_size = torch.cuda.device_count()
        print(f"Running in single-node multi-GPU mode with {world_size} GPUs")
        run_parallel_simulations_single_node(
            trajectory_model_pairs=trajectory_model_pairs,
            world_size=min(world_size, len(trajectory_model_pairs)),  # Don't use more GPUs than pairs
            interval=interval,
            total_target_steps=total_target_steps,
            temperatures=temperatures,
            initial_temperatures=initial_temperatures,
            timestep_fs_list=timestep_fs_list,
            freeze_graph=freeze_graph,
            pfactor_list=pfactor_list,
            dynamics_type_list=dynamics_type_list,
        )

    elif execution_mode == 'single_node_single_gpu':
        # Single trajectory-model pair - run on single GPU
        if len(trajectory_model_pairs) > 1:
            raise ValueError(f"Single GPU mode can only handle 1 trajectory-model pair, got {len(trajectory_model_pairs)}")

        traj_path, model_checkpoint = trajectory_model_pairs[0]
        temperature = temperatures[0]
        initial_temperature = initial_temperatures[0]
        timestep_fs = timestep_fs_list[0]
        pfactor = pfactor_list[0]
        dynamics_type = dynamics_type_list[0]
        print(f"Running single trajectory-model pair on single GPU: {traj_path} + {model_checkpoint}")
        print(f"Temperature: {temperature} K, Initial temperature: {initial_temperature} K, Timestep: {timestep_fs} fs, Dynamics: {dynamics_type}")
        simulate(
            root_path=traj_path,
            rank=None,
            world_size=None,
            interval=interval,
            total_target_steps=total_target_steps,
            model_checkpoint=model_checkpoint,
            temperature=temperature,
            initial_temperature=initial_temperature,
            timestep_fs=timestep_fs,
            freeze_graph=freeze_graph,
            pfactor=pfactor,
            dynamics_type=dynamics_type,
        )
    
    else:
        # Multiple trajectory-model pairs - distribute across GPUs
        print(f"Running {len(trajectory_model_pairs)} trajectory-model pairs on {world_size} GPUs")
        run_parallel_simulations(
            trajectory_model_pairs=trajectory_model_pairs,
            world_size=min(world_size, len(trajectory_model_pairs)),  # Don't use more GPUs than pairs
            interval=interval,
            total_target_steps=total_target_steps,
            temperatures=temperatures,
            initial_temperatures=initial_temperatures,
            timestep_fs_list=timestep_fs_list,
        )


if __name__ == "__main__":
    main()