from pathlib import Path
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

from ase.io import Trajectory


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from draft.prepare_box.uma_boxes import find_top_level_trajs  # noqa: E402


ABLATION_ROOT = (
    REPO_ROOT
    / "yuejian/electrolyte_application/ablate_distillation/ablation_diffusivity"
)

SOURCE_ROOTS = [
    ABLATION_ROOT / "20ns_solute_solvent_1M",
    ABLATION_ROOT / "20ns_solvent_0_1M",
    ABLATION_ROOT / "20ns_solvent_solute_0.5M",
]

DEST_ROOTS = [
    ABLATION_ROOT / "initial_box/20ns_solute_solvent_1M",
    ABLATION_ROOT / "initial_box/20ns_solvent_0_1M",
    ABLATION_ROOT / "initial_box/20ns_solvent_solute_0.5M",
]


def _pick_single_traj(system_dir: Path) -> Optional[Path]:
    trajs = find_top_level_trajs(system_dir)
    if not trajs:
        return None
    if len(trajs) == 1:
        return trajs[0]
    preferred = system_dir / ("%s.traj" % system_dir.name)
    if preferred in trajs:
        return preferred
    raise ValueError(
        "Multiple .traj files in %s and none matches folder name: %s"
        % (system_dir, [p.name for p in trajs])
    )


def len_and_first_stats(traj_path: Path) -> Tuple[int, int, Dict[str, int]]:
    """Return (num_frames, atom_count_first_frame, species_counter) for a trajectory."""
    with Trajectory(traj_path) as traj:
        n_frames = len(traj)
        if n_frames == 0:
            raise ValueError(f"{traj_path} is empty.")
        atoms = traj[0]
        return n_frames, len(atoms), dict(Counter(atoms.get_chemical_symbols()))


def _rel_to_one_of(path: Path, roots: List[Path]) -> Tuple[Optional[Path], Optional[Path]]:
    """Return (matching_root, relative_path) if path is under one of roots."""
    for r in roots:
        try:
            return r, path.relative_to(r)
        except ValueError:
            continue
    return None, None


def main() -> int:
    dest_trajs: List[Path] = []
    for dr in DEST_ROOTS:
        dest_trajs.extend(sorted(dr.glob("**/*.traj")))
    if not dest_trajs:
        print("No trajs found under: %s" % DEST_ROOTS)
        return 1

    missing_sources: List[str] = []
    atom_mismatches: List[str] = []
    species_mismatches: List[str] = []
    errors: List[str] = []
    bad_frame_counts: List[str] = []

    for dest_traj in dest_trajs:
        # Map:
        #   .../initial_box/<root>/<maybe temp>/<system>/<system>.traj
        # -> .../<root>/<maybe temp>/<system>/<system>.traj
        dest_root, rel_file = _rel_to_one_of(dest_traj, DEST_ROOTS)
        if dest_root is None or rel_file is None:
            continue

        # determine which SOURCE_ROOT matches this DEST_ROOT
        try:
            dest_suffix = dest_root.relative_to(ABLATION_ROOT / "initial_box")
        except ValueError:
            dest_suffix = None
        if dest_suffix is None:
            errors.append("Could not map dest root: %s" % dest_root)
            continue
        src_root = ABLATION_ROOT / str(dest_suffix)

        system_dir_rel = rel_file.parent  # includes temp directories, ends with system folder
        src_system_dir = src_root / system_dir_rel
        src_traj = _pick_single_traj(src_system_dir)
        if src_traj is None or (not src_traj.exists()):
            missing_sources.append("%s (expected under %s)" % (dest_traj, src_system_dir))
            continue

        name = dest_traj.stem
        assert dest_traj.parts[-3:] == src_traj.parts[-3:], "The last 5 levels of path of dest_traj and src_traj are not the same"
        print(f"dest_traj: {dest_traj.parts[-5:]}")
        print(f"src_traj: {src_traj.parts[-5:]}")
        print("--------------------------------")
        try:
            dest_frames, dest_atoms, dest_species = len_and_first_stats(dest_traj)
            src_frames, src_atoms, src_species = len_and_first_stats(src_traj)
        except Exception as exc:  # pragma: no cover
            errors.append("%s: %s" % (name, exc))
            continue
        if dest_atoms != src_atoms:
            atom_mismatches.append("%s: atoms dest=%d, src=%d" % (name, dest_atoms, src_atoms))

        if dest_species != src_species:
            species_mismatches.append(
                "%s: species dest=%s, src=%s" % (name, dest_species, src_species)
            )
        if dest_frames != 1:
            bad_frame_counts.append("%s: dest frames=%d (expected 1)" % (name, dest_frames))
        # We do not enforce source frame count.
        if not (
            name in [m.split(":")[0] for m in atom_mismatches]
            or name in [m.split(":")[0] for m in species_mismatches]
            or name in [b.split(":")[0] for b in bad_frame_counts]
        ):
            print("[ok] %s: atoms=%d, dest_frames=%d" % (name, dest_atoms, dest_frames))

    if missing_sources:
        print("\nMissing source trajs:")
        for n in missing_sources:
            print(f"  - {n}")
    if atom_mismatches:
        print("\nAtom-count mismatches:")
        for msg in atom_mismatches:
            print(f"  - {msg}")
    if species_mismatches:
        print("\nSpecies mismatches (per-element counts differ):")
        for msg in species_mismatches:
            print(f"  - {msg}")
    if bad_frame_counts:
        print("\nDestination frame-count issues (expected 1):")
        for msg in bad_frame_counts:
            print(f"  - {msg}")
    if errors:
        print("\nErrors encountered:")
        for msg in errors:
            print(f"  - {msg}")

    if missing_sources or atom_mismatches or species_mismatches or errors or bad_frame_counts:
        print("\nSanity check failed.")
        return 1

    print("\nSanity check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
