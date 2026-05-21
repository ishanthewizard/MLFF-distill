from pathlib import Path
import shutil
from typing import List

from ase.io import Trajectory


# Source and destination roots
SRC_DIR = Path("/global/homes/y/yuejian/project/MLFF-distill/m5024/UMA_trajs_293K")
DEST_DIR = Path("/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/initial_boxes_for_UMA")
LOGS_DIR = SRC_DIR / "md_logs"

# Expected number of top-level trajectory files
EXPECTED_TRAJ_COUNT = 15


def find_top_level_trajs(src_dir: Path) -> List[Path]:
    """Return sorted list of .traj files directly under src_dir (not in subfolders)."""
    return sorted(p for p in src_dir.iterdir() if p.is_file() and p.suffix == ".traj")


def extract_first_frame(src_traj: Path, dest_traj: Path) -> None:
    """Write only the first frame of src_traj into dest_traj."""
    with Trajectory(src_traj) as traj:
        if len(traj) == 0:
            raise ValueError(f"{src_traj} is empty; cannot extract first frame.")
        first = traj[0]

    dest_traj.parent.mkdir(parents=True, exist_ok=True)
    with Trajectory(dest_traj, "w") as out:
        out.write(first)


def create_empty_log(dest_log: Path) -> None:
    """Create an empty log file at dest_log, overwriting if exists."""
    dest_log.parent.mkdir(parents=True, exist_ok=True)
    dest_log.write_text("")


def main() -> None:
    traj_paths = find_top_level_trajs(SRC_DIR)
    if len(traj_paths) != EXPECTED_TRAJ_COUNT:
        print(f"[warn] Found {len(traj_paths)} trajs (expected {EXPECTED_TRAJ_COUNT}). Proceeding with found files.")

    for src_traj in traj_paths:
        name = src_traj.stem
        dest_dir = DEST_DIR / name
        dest_traj = dest_dir / f"{name}.traj"
        dest_log = dest_dir / f"{name}.log"

        print(f"Processing {name}...")
        extract_first_frame(src_traj, dest_traj)
        create_empty_log(dest_log)
        print(f"  -> wrote {dest_traj}")
        print(f"  -> created empty {dest_log}")

    print("Done. First-frame trajs are in", DEST_DIR)


if __name__ == "__main__":
    main()
