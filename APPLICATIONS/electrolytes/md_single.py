#!/usr/bin/env python3
import argparse, os, time, logging
from pathlib import Path
from copy import deepcopy
from get_calc import get_uma_calc
import torch
from ase.io import read, Trajectory
from ase.md.npt import NPT
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from fairchem.core import FAIRChemCalculator

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-list", nargs="+", required=True, help="List of input trajs")
    ap.add_argument("--identifier-list", nargs="+", required=True, help="List of identifiers (same order as inputs)")
    ap.add_argument("--workdir", default="/projects/beye/iamin/distillation_project/test_trajs")
    ap.add_argument("--model", required=True,
                    default="/projects/beye/iamin/distillation_project/models/omol_all_sm_NeAnNoSi_ft_40E20F_fixed.pt")
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--print-interval", dest="print_interval", type=int, default=20)
    ap.add_argument("--traj-interval",  dest="traj_interval",  type=int, default=10)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--temp-k", type=float, default=293.0)
    ap.add_argument("--timestep-fs", type=float, default=1.0)
    ap.add_argument("--ttime-fs", type=float, default=100.0)
    ap.add_argument("--pfactor", type=float, default=0.1)
    ap.add_argument("--ext-press-bar", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--continue", dest="cont", action="store_true",
                    help="Continue from existing traj/log if present")
    args = ap.parse_args()

    # rank = int(os.getenv("SLURM_LOCALID", os.getenv("SLURM_PROCID", "0")))
    # global rank first; fallback to local if needed
    rank = int(os.getenv("SLURM_PROCID", os.getenv("SLURM_LOCALID", "0")))

    in_path = Path(args.input_list[rank])
    ident   = args.identifier_list[rank]

    workdir = Path(args.workdir)
    (workdir / "md_logs").mkdir(parents=True, exist_ok=True)
    out_traj = workdir / f"{ident}.traj"
    out_log  = workdir / "md_logs" / f"{ident}.log"
    logger   = setup_md_logger(out_log, args.cont)

    print(f"[INFO] Host={os.uname().nodename} Rank={rank} Ident={ident} "
          f"CVD={os.getenv('CUDA_VISIBLE_DEVICES','')}")
    torch.set_num_threads(args.threads)

    # ==== Decide where to start from ====
    current_step = 0
    if args.cont and out_traj.exists():
        current_step = len(Trajectory(out_traj))
        print(f"[INFO] Resuming from last frame of {out_traj}")
        base = read(str(out_traj), index=-1)  # last frame of output traj
    else:
        if not in_path.exists():
            print(f"[ERROR] Missing input: {in_path}")
            return
        base = read(str(in_path), index=args.start_index)

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
        torch.cuda.set_device(0)  # always 0, since sbatch sets CVD per-rank
    else:
        print("[WARN] CUDA not available; running on CPU.")

    # structure.calc = FAIRChemCalculator.from_model_checkpoint(args.model, task_name="omol")
    structure.calc = get_uma_calc(uma_path= args.model, small_model=True)
    # ==== Trajectory writing mode ====
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
            except Exception: pass
        logger.info(f"Step {(current_step + step):>8} | T={temp:6.1f} K | "
                    f"Epot={epot:10.3f} eV | Ekin={ekin:10.3f} eV | "
                    f"Vol={vol:10.3f} Å³{its}{mem}")

    dyn.attach(print_status, interval=args.print_interval)

    t0=time.time()
    try:
        dyn.run(steps=remaining_steps)
    finally:
        logger.info(f"Completed in {(time.time()-t0)/60:.2f} min; output={out_traj}")
        try: traj.close()
        except: pass

if __name__ == "__main__":
    main()
