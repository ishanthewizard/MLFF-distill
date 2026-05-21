





from pathlib import Path
import argparse
import sys
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
# Ensure repo root is importable when running as a script:
#   python3 draft/prepare_box/get_box.py
sys.path.insert(0, str(REPO_ROOT))

from draft.prepare_box.uma_boxes import (  # noqa: E402
    create_empty_log,
    extract_first_frame,
    find_top_level_trajs,
)
ABLATION_ROOT = (
    REPO_ROOT
    / "yuejian/electrolyte_application/ablate_distillation/ablation_diffusivity"
)
TARGET_ROOT = ABLATION_ROOT / "initial_box"


DEFAULT_SOURCE_ROOTS = [
    ABLATION_ROOT / "20ns_solute_solvent_1M",
    ABLATION_ROOT / "20ns_solvent_0_1M",
    ABLATION_ROOT / "20ns_solvent_solute_0.5M/273_2K",
    ABLATION_ROOT / "20ns_solvent_solute_0.5M/298_2K",
    ABLATION_ROOT / "20ns_solvent_solute_0.5M/323_2K",
]


def _pick_single_traj(system_dir: Path) -> Optional[Path]:
    """Pick the trajectory file for a system directory.

    Returns None if no traj is found.
    Raises if multiple trajs exist and none matches the folder name.
    """
    trajs = find_top_level_trajs(system_dir)
    if not trajs:
        return None
    if len(trajs) == 1:
        return trajs[0]

    # Prefer `<folder_name>.traj` when multiple exist.
    preferred = system_dir / f"{system_dir.name}.traj"
    if preferred in trajs:
        return preferred

    raise ValueError(
        f"Multiple .traj files in {system_dir} and none matches folder name: "
        f"{[p.name for p in trajs]}"
    )


def create_initial_boxes(
    *,
    source_roots: List[Path],
    target_root: Path,
    overwrite: bool = False,
) -> None:
    """Create one-frame traj 'boxes' under target_root mirroring source_roots."""
    target_root.mkdir(parents=True, exist_ok=True)

    for src_root in source_roots:
        if not src_root.exists():
            print(f"[skip] missing source root: {src_root}")
            continue

        rel_root = src_root.relative_to(ABLATION_ROOT)
        sub_target_root = target_root / rel_root
        sub_target_root.mkdir(parents=True, exist_ok=True)

        for system_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
            src_traj = _pick_single_traj(system_dir)
            if src_traj is None:
                continue

            name = system_dir.name
            dest_dir = sub_target_root / name
            dest_traj = dest_dir / f"{name}.traj"
            dest_log = dest_dir / f"{name}.log"

            if dest_traj.exists() and not overwrite:
                print(f"[skip] exists: {dest_traj}")
                continue
            print("--------------------------------")
            print(f"src_traj: {src_traj.parts[-5:]}")
            print(f"dest_traj: {dest_traj.parts[-5:]}")
            # assert the last 3 level of path of src_traj and dest_traj are the same
            assert src_traj.parts[-3:] == dest_traj.parts[-3:], "The last 3 levels of path of src_traj and dest_traj are not the same"
            print(f"[box] {rel_root}/{name}")
            print("--------------------------------")
            extract_first_frame(src_traj, dest_traj)
            create_empty_log(dest_log)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create initial one-frame trajectory boxes for ablation_diffusivity runs."
        )
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing boxes if present.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    create_initial_boxes(
        source_roots=DEFAULT_SOURCE_ROOTS,
        target_root=TARGET_ROOT,
        overwrite=bool(args.overwrite),
    )
    print(f"Done. Wrote initial boxes under: {TARGET_ROOT}")


if __name__ == "__main__":
    main()
