#!/usr/bin/env python3
import argparse, os, time, logging
from pathlib import Path
from copy import deepcopy
import torch
from ase.io import read, Trajectory
from ase.md.npt import NPT
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

# from fairchem.core import FAIRChemCalculator  # if you need the original
from get_calc import get_uma_calc

def setup_md_logger(path: Path, cont: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("md_logger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    mode = "a" if cont and path.exists() else "w"
    fh = logging.FileHandler(path, mode=mode)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(fh)
    return logger

def broadcast_list(lst, N):
    """
    If lst has length 0 -> return [None]*N
    If lst has length 1 -> broadcast to length N
    If lst has length N -> return as-is
    Else -> raise
    """
    if lst is None:
        return [None] * N
    if len(lst) == 0:
        return [None] * N
    if len(lst) == 1:
        return lst * N
    if len(lst) == N:
        return lst
    raise ValueError(f"List length {len(lst)} incompatible with N={N}")

def select_rank_index(default_idx: int, idx_list, rank: int):
    """Prefer per-rank list if provided, otherwise use default_idx."""
    if idx_list is not None and idx_list[rank] is not None:
        return int(idx_list[rank])
    return int(default_idx)

def select_rank_model(default_model: str, model_list, rank: int):
    """Prefer per-rank model if provided, otherwise use default_model."""
    if model_list is not None and model_list[rank] is not None:
        return model_list[rank]
    if default_model is None:
        raise ValueError("No model specified: provide --model or --model-list")
    return default_model

def main():
    ap = argparse.ArgumentParser()
    # Core per-rank lists
    ap.add_argument("--input-list", nargs="+", required=True, help="List of input trajs (one per rank)")
    ap.add_argument("--identifier-list", nargs="+", required=True, help="List of identifiers (one per rank)")

    # New degrees of freedom (lists)
    ap.add_argument("--start-index-list", nargs="*", type=int, default=None,
                    help="Optional list of start indices (1 per rank) or a single value to broadcast")
    ap.add_argument("--model-list", nargs="*", default=None,
                    help="Optional list of model paths (1 per rank) or a single path to broadcast")

    # Legacy singletons kept for convenience/backward compat
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--model", default=None)

    # MD/IO
    ap.add_argument("--workdir", default="/projects/beye/iamin/distillation_project/test_trajs")
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--print-interval", dest="print_interval", type=int, default=20)
    ap.add_argument("--traj-interval",  dest="traj_interval",  type=int, default=10)
    ap.add_argument("--temp-k", type=float, default=293.0)
    ap.add_argument("--timestep-fs", type=float, default=1.0)
    ap.add_argument("--ttime-fs", type=float, default=100.0)
    ap.add_argument("--pfactor", type=float, default=0.1)
    ap.add_argument("--ext-press-bar", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--continue", dest="cont", action="store_true", help="Continue from existing traj/log if present")
    args = ap.parse_args()

    # Determine rank
    rank = int(os.getenv("SLURM_PROCID", os.getenv("SLURM_LOCALID", "0")))
    world = len(args.input_list)

    # Basic list sanity
    if len(args.identifier_list) != world:
        raise ValueError(f"--identifier-list length {len(args.identifier_list)} != --input-list length {world}")

    # Broadcast new lists if needed
    starts_b = broadcast_list(args.start_index_list, world) if args.start_index_list is not None else None
    models_b = broadcast_list(args.model_list, world) if args.model_list is not None else None

    # Per-rank selections
    in_path = Path(args.input_list[rank])
    ident   = args.identifier_list[rank]
    start_idx = select_rank_index(args.start_index, starts_b, rank)
    model_path = select_rank_model(args.model, models_b, rank)

    workdir = Path(args.workdir)
    (workdir / "md_logs").mkdir(parents=True, exist_ok=True)
    out_traj = workdir / f"{ident}.traj"
    out_log  = workdir / "md_logs" / f"{ident}.log"
    logger   = setup_md_logger(out_log, args.cont)

    print(f"[INFO] Host={os.uname().nodename} Rank={rank} Ident={ident} "
          f"Input={in_path} StartIdx={start_idx} Model={model_path} "
          f"CVD={os.getenv('CUDA_VISIBLE_DEVICES','')}")
    torch.set_num_threads(args.threads)

    # ==== Decide where to start from ====
    current_step = 0
    if args.cont and out_traj.exists():
        # Continue overrides start_idx
        from ase.io import Trajectory as TrajReader
        current_step = len(TrajReader(out_traj))
        print(f"[INFO] Resuming from last frame of {out_traj}")
        base = read(str(out_traj), index=-1)
    else:
        if not in_path.exists():
            print(f"[ERROR] Missing input: {in_path}")
            return
        base = read(str(in_path), index=start_idx)

    base.set_pbc([True, True, True])
    structure = deepcopy(base)

    remaining_steps = max(0, args.steps - current_step)

    # Seeds per rank
    try:
        import numpy as np
        np.random.seed(1234 + rank)
    except Exception:
        pass
    torch.manual_seed(1234 + rank)
    MaxwellBoltzmannDistribution(structure, temperature_K=args.temp_k)

    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # 0 inside each rank's CVD
    else:
        print("[WARN] CUDA not available; running on CPU.")

    # Choose calculator (UMA small model example)
    structure.calc = get_uma_calc(uma_path=model_path, small_model=True)
    # Example for FAIRChem if needed:
    # structure.calc = FAIRChemCalculator.from_model_checkpoint(model_path, task_name="omol")

    # ==== Trajectory I/O ====
    mode = "a" if args.cont and out_traj.exists() else "w"
    traj = Trajectory(str(out_traj), mode, structure)

    dyn = NPT(
        atoms=structure,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temp_k,
        externalstress=args.ext_press_bar * units.bar,
        ttime=args.ttime_fs * units.fs,
        pfactor=args.pfactor,
        mask=[[1,0,0],[0,1,0],[0,0,1]],
    )
    dyn.attach(traj.write, interval=args.traj_interval)

    last_step = [None]; last_time = [None]
    def print_status(a=structure):
        epot=a.get_potential_energy(); ekin=a.get_kinetic_energy()
        temp=a.get_temperature(); vol=a.get_volume(); step=dyn.nsteps
        its=""; tnow=time.time()
        if last_step[0] is not None:
            ds=step-last_step[0]; dt=tnow-last_time[0]
            if ds>0 and dt>0: its=f" | {ds/dt:6.2f} it/s"
        last_step[0]=step; last_time[0]=tnow
        mem=""
        if torch.cuda.is_available():
            try:
                alloc=torch.cuda.memory_allocated()/1024**2
                reserv=torch.cuda.memory_reserved()/1024**2
                mem=f" | CUDA alloc={alloc:8.0f}MB reserv={reserv:8.0f}MB"
            except Exception:
                pass
        logger.info(
            f"Step {(current_step + step):>8} | T={temp:6.1f} K | "
            f"Epot={epot:10.3f} eV | Ekin={ekin:10.3f} eV | "
            f"Vol={vol:10.3f} Å³{its}{mem}"
        )

    dyn.attach(print_status, interval=args.print_interval)

    t0=time.time()
    try:
        dyn.run(steps=remaining_steps)
    finally:
        logger.info(f"Completed in {(time.time()-t0)/60:.2f} min; output={out_traj}")
        try:
            traj.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
