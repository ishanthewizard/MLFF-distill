#!/usr/bin/env python3
import os
import time
from ase.io import Trajectory

# Directories and labels
dirs = {
    "293K": "/projects/beye/iamin/distillation_project/UMA_trajs_293K",
    "323K": "/projects/beye/iamin/distillation_project/UMA_trajs_323K",
}

# Parameters
target_frames = 100_000   # 1 ns, since 10 fs per frame
bar_length = 40           # characters in progress bar
check_interval = 10       # seconds between length checks

def progress_bar(current, total, length=40):
    frac = min(current / total, 1.0)
    filled = int(frac * length)
    bar = "█" * filled + "-" * (length - filled)
    return f"[{bar}] {frac*100:5.1f}%"

def get_traj_len(path):
    try:
        traj = Trajectory(path)
        return len(traj)
    except Exception:
        return None

# Collect all files across temps
files_by_temp = {}
all_files = []
for temp, d in dirs.items():
    if not os.path.exists(d):
        continue
    files = [f for f in os.listdir(d) if f.endswith(".traj")]
    files_by_temp[temp] = [os.path.join(d, f) for f in sorted(files)]
    all_files.extend((temp, os.path.join(d, f)) for f in files)

# First measurement
lengths_before = {}
for temp, path in all_files:
    lengths_before[path] = get_traj_len(path)

print(f"\nChecking run status for {len(all_files)} trajectories... waiting {check_interval}s...\n")
time.sleep(check_interval)

# Second measurement and printing
for temp in sorted(files_by_temp.keys()):
    print(f"=== {temp} ===")
    for path in files_by_temp[temp]:
        f = os.path.basename(path)
        length_before = lengths_before[path]
        length_after = get_traj_len(path)

        if length_before is None or length_after is None:
            print(f"{f:40s}  Error reading trajectory.")
            continue

        bar = progress_bar(length_after, target_frames, bar_length)
        time_fs = length_after * 10
        time_ps = time_fs / 1000
        status = "RUNNING" if length_after > length_before else "STOPPED"

        print(f"{f:40s}  {bar}  ({length_after:6d} frames, {time_ps:8.1f} ps)  [{status}]")
    print()
