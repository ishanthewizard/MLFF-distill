from msd_with_com import main

import argparse
import json
import os
from pathlib import Path

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compute MSDs and diffusion coefficients with optional convergence time list."
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        metavar="DIR",
        type=str,
        required=True,
        help="Output directory for MSDs.",
    )
    parser.add_argument(
        "--tau-max-fit-ps",
        "-t",
        metavar="PS",
        type=int,
        required=True,
        help="Maximum fitting time (in ps).",
    )
    parser.add_argument(
        "--known-dt-ps",
        "-d",
        metavar="PS",
        type=float,
        default=0.01,
        help="Base timestep of the trajectory in picoseconds (default: 0.01 ps = 10 fs).",
    )
    parser.add_argument(
        "--targets-json",
        metavar="JSON",
        type=str,
        help="JSON string containing TARGETS list. If not provided, reads from MSD_TARGETS_JSON environment variable.",
    )
    args = parser.parse_args()

    # ---------------- user config ----------------
    OUT_DIR  = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"tau_max_fit_ps: {args.tau_max_fit_ps}")
    print(f"known_dt_ps: {args.known_dt_ps}")
    
    # Get TARGETS from command-line argument or environment variable
    targets_json_str = args.targets_json
    if targets_json_str is None:
        targets_json_str = os.environ.get('MSD_TARGETS_JSON')
        if targets_json_str is None:
            print("Error: --targets-json not provided and MSD_TARGETS_JSON environment variable not set.")
            exit(1)
    
    try:
        TARGETS = json.loads(targets_json_str)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON string: {targets_json_str}")
        print(f"JSON decode error: {e}")
        exit(1)
    # Convert TARGETS from list of lists to list of tuples
    TARGETS = [tuple(t) for t in TARGETS]
    print(f"TARGETS: {TARGETS}")


    EQ_TIME_PS       = 100.0
    KNOWN_DT_PS      = args.known_dt_ps
    # TARGET_FRAMES    = 20000 # this is about dt < 1ps
    TAU_MIN_FIT_PS   = 1000.0 # 1000 ps  = 1 ns
    TAU_MAX_FIT_PS   = args.tau_max_fit_ps
    TARGET_FRAMES    = int(TAU_MAX_FIT_PS - TAU_MIN_FIT_PS) # dt = 1ps
    N_WORKERS        = 8
    PLOT_NCOLS       = 2
    PARALLEL_MSD     = False # not sure if this is reliable
    
    main(
        TARGETS,
        EQ_TIME_PS,
        KNOWN_DT_PS,
        TARGET_FRAMES,
        TAU_MIN_FIT_PS,
        TAU_MAX_FIT_PS,
        N_WORKERS,
        PLOT_NCOLS,
        PARALLEL_MSD,
        OUT_DIR
    )