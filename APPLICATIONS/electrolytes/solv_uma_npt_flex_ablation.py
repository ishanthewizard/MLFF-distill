import sys
import os
import time
import signal
import torch.distributed as dist
import torch.multiprocessing as mp

# Add project root to Python path to make src_v2 module accessible
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from ase.io import read, Trajectory
from ase.md.npt import NPT
from ase import units
from fairchem.core import pretrained_mlip, FAIRChemCalculator
import torch
import numpy as np
from copy import deepcopy
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from get_calc import get_uma_calc


def setup_distributed(rank, world_size):
    """Initialize torch.distributed for multi-GPU setup"""
    if world_size <= 1:
        print(f"Rank {rank}/{world_size}: No need for distributed setup with single GPU")
        return  # No need for distributed setup with single GPU

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed processes"""
    if dist.is_initialized():
        dist.destroy_process_group()




def worker(rank, trajectory_model_pairs, world_size, interval, total_target_steps, temperature, initial_temperature):
    """Worker function that runs on each GPU"""
    try:
        # Initialize distributed setup if running on multiple GPUs
        if world_size > 1:
            setup_distributed(rank, world_size)
            print(f"Rank {rank}/{world_size}: Initialized distributed setup")

        # Get trajectory-model pair assigned to this rank
        if rank < len(trajectory_model_pairs):
            traj_path, model_checkpoint = trajectory_model_pairs[rank]
            print(f"Rank {rank}: Assigned trajectory: {traj_path}, model: {model_checkpoint}")
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
                rank=rank,
                world_size=world_size,
                interval=interval,
                total_target_steps=total_target_steps,
                model_checkpoint=model_checkpoint,
                temperature=temperature,
                initial_temperature=initial_temperature
            )
            print(f"Rank {rank}: Completed simulation for {traj_path}")
        except Exception as sim_error:
            print(f"Rank {rank}: Simulation failed for {traj_path}: {sim_error}")

        # Synchronize all ranks after completion
        if world_size > 1:
            dist.barrier()

    except Exception as e:
        print(f"Rank {rank}: Error in worker: {e}")
        # Ensure cleanup happens even on error
        try:
            if dist.is_initialized():
                cleanup_distributed()
        except:
            pass
        raise
    finally:
        # Clean up distributed setup
        if world_size > 1:
            cleanup_distributed()
            print(f"Rank {rank}: Cleaned up distributed setup")


def run_parallel_simulations(trajectory_model_pairs, world_size, interval=50, total_target_steps=1000000, temperature=323, initial_temperature=300):
    """Main function to run parallel simulations across multiple GPUs"""

    # Spawn worker processes
    print(f"Starting parallel simulations on {world_size} GPUs")
    print(f"Total trajectory-model pairs: {len(trajectory_model_pairs)}")

    mp.spawn(
        worker,
        args=(trajectory_model_pairs, world_size, interval, total_target_steps, temperature, initial_temperature),
        nprocs=world_size,
        join=True
    )
    print("All parallel simulations completed successfully!")



def simulate(root_path, rank=None, world_size=None, interval=50, total_target_steps=1000000, model_checkpoint=None, temperature=323, initial_temperature=300):
    """
    Run MD simulation on a specific GPU rank.

    Args:
        rank: GPU rank (0, 1, 2, 3 for 4-GPU node)
        world_size: Total number of GPUs
    """
    # Skip distributed setup if already initialized (when called from worker)
    print(f"Rank {rank}/{world_size}: dist.is_initialized(): {dist.is_initialized()}")
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
    torch.set_num_threads(28)
    print(f"Loading UMA model from: {uma_path}")
    structure.calc = get_uma_calc(uma_path= uma_path, small_model=False)
    print("UMA model loaded successfully")


    # === Set up NPT dynamics ===
    dyn = NPT(
        atoms=structure,
        timestep = 1 * units.fs,
        temperature_K= temperature, # 298.2,
        externalstress=1.0 * units.bar,
        ttime=100 * units.fs,
        pfactor=0.1, ### larger value mean it will relax slower, typical 10^-3 
        mask=([[1,0,0],[0,1,0],[0,0,1]]),
    )

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
    print(f"Target: {total_target_steps} steps at 1 fs timestep")
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



def main():
    """Main function that parses command line arguments and runs simulations"""
    import argparse
    
    # === Set up signal handler for graceful shutdown ===
    # signal.signal(signal.SIGTERM, signal_handler)
    
    # === Command Line Argument Parser ===
    parser = argparse.ArgumentParser(description='Run MD simulations with UMA potential')
    parser.add_argument('trajectories', nargs='+', 
                       help='Path(s) to trajectory directories (space-separated)')
    parser.add_argument('--models', nargs='+', required=True,
                       help='Path(s) to UMA model checkpoint files (space-separated)')
    parser.add_argument('--steps', type=int, default=1000000,
                       help='Total target steps for simulation (default: 1000000)')
    parser.add_argument('--interval', type=int, default=10,
                       help='Interval for trajectory writing and status printing (default: 10)')
    parser.add_argument('--temperature', type=float, default=323,
                       help='Temperature for simulation (default: 323 K)')
    parser.add_argument('--initial_temperature', type=float, default=300,
                       help='Initial temperature for simulation (default: 300 K)')
    args = parser.parse_args()
    
    # === Configuration ===
    total_target_steps = args.steps
    interval = args.interval
    model_checkpoints = args.models
    trajectory_paths = args.trajectories
    temperature = args.temperature
    initial_temperature = args.initial_temperature
    world_size = torch.cuda.device_count()  # Number of available GPUs

    if world_size == 0:
        raise RuntimeError("No CUDA devices available")

    print(f"Found {world_size} CUDA devices")
    print(f"Model checkpoints: {model_checkpoints}")
    print(f"Target steps: {total_target_steps}")
    print(f"Interval: {interval}")
    print(f"Trajectory paths: {trajectory_paths}")
    print(f"Temperature: {temperature}")
    print(f"Initial temperature: {initial_temperature}")
    
    # === Validation ===
    # Check that trajectory and model lists have the same length
    if len(trajectory_paths) != len(model_checkpoints):
        raise ValueError(f"Number of trajectories ({len(trajectory_paths)}) must equal number of models ({len(model_checkpoints)})")
    
    # Check that both lists don't exceed 4 items
    if len(trajectory_paths) > 4:
        raise ValueError(f"Number of trajectory-model pairs ({len(trajectory_paths)}) cannot exceed 4")
    
    # Check that we don't have more pairs than GPUs
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

    # === Create trajectory-model pairs ===
    trajectory_model_pairs = list(zip(trajectory_paths, model_checkpoints))
    print(f"Created {len(trajectory_model_pairs)} trajectory-model pairs")

    # === Run parallel simulations ===
    if len(trajectory_model_pairs) == 1:
        # Single trajectory-model pair - run on single GPU
        traj_path, model_checkpoint = trajectory_model_pairs[0]
        print(f"Running single trajectory-model pair on single GPU: {traj_path} + {model_checkpoint}")
        simulate(
            root_path=traj_path,
            rank=None,
            world_size=None,
            interval=interval,
            total_target_steps=total_target_steps,
            model_checkpoint=model_checkpoint,
            temperature=temperature,
            initial_temperature=initial_temperature
        )
    else:
        # Multiple trajectory-model pairs - distribute across GPUs
        print(f"Running {len(trajectory_model_pairs)} trajectory-model pairs on {world_size} GPUs")
        run_parallel_simulations(
            trajectory_model_pairs=trajectory_model_pairs,
            world_size=min(world_size, len(trajectory_model_pairs)),  # Don't use more GPUs than pairs
            interval=interval,
            total_target_steps=total_target_steps,
            temperature=temperature,
            initial_temperature=initial_temperature
        )


if __name__ == "__main__":
    main()