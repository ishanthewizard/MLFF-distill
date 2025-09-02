#!/usr/bin/env python3
import re
import shutil
from pathlib import Path
from collections import defaultdict

from ase.io import iread
from ase.io.trajectory import TrajectoryWriter

SRC = Path("/global/cfs/cdirs/m4558/first_100ps_traj_to_send")
DST = Path("/global/cfs/cdirs/m4558/distillation_project/all_trajs_min50ps")
DST.mkdir(parents=True, exist_ok=True)

# Matches suffixes like _re1, _re2, _re10, optionally followed by more tokens (e.g., _s1p1)
# We only remove the _re\d+ part from the "base key"
RE_SUFFIX = re.compile(r"_re(?P<rep>\d+)(?=$|[_\.])")

def base_key_and_rep(fname: str):
    """Return (base_key, replicate_number or None)."""
    stem = Path(fname).stem  # no extension
    m = RE_SUFFIX.search(stem)
    if m:
        rep = int(m.group("rep"))
        base_key = RE_SUFFIX.sub("", stem)
        return (base_key, rep)
    else:
        # No explicit _re\d+; treat as base (rep=None)
        return (stem, None)

def stitch_group(files, out_path):
    """Concatenate frames from a list of files (ordered) into a single .traj at out_path."""
    # Stream frames with iread; write sequentially
    with TrajectoryWriter(str(out_path), mode="w") as w:
        total = 0
        for f in files:
            for atoms in iread(str(f), index=":"):
                w.write(atoms)
                total += 1
    return total

def main():
    trajs = sorted(SRC.glob("*.traj"))
    if not trajs:
        print(f"No .traj files found in {SRC}")
        return

    # Group by base_key; collect (rep, path)
    groups = defaultdict(list)
    for f in trajs:
        base_key, rep = base_key_and_rep(f.name)
        groups[base_key].append((rep, f))

    stitched = 0
    copied = 0

    for base_key, items in groups.items():
        # Determine output filename: use the first item's name but with any _re\d+ stripped
        # Keep the rest of the tokens (e.g., _s1p1) intact by deriving from base_key.
        out_name = f"{base_key}.traj"
        out_path = DST / out_name

        # If there are 2+ files (i.e., duplicates/replicates), stitch them
        # Order: base (rep=None) first, then ascending rep numbers.
        if len(items) >= 2:
            # Sort: base (None) -> reps ascending
            items_sorted = sorted(items, key=lambda x: (x[0] is not None, x[0] if x[0] is not None else -1))
            files_in_order = [p for _, p in items_sorted]

            print(f"[STITCH] {base_key} -> {out_path.name}")
            for rep, p in items_sorted:
                tag = f"re{rep}" if rep is not None else "base"
                print(f"         - {tag}: {p.name}")

            total = stitch_group(files_in_order, out_path)
            print(f"         Total frames written: {total}")
            stitched += 1

        else:
            # Single file: just copy to destination
            src_file = items[0][1]
            print(f"[COPY]   {src_file.name} -> {out_path.name}")
            shutil.copy2(src_file, out_path)
            copied += 1

    print("\nDone.")
    print(f"Stitched systems: {stitched}")
    print(f"Copied single trajectories: {copied}")
    print(f"Output directory: {DST}")

if __name__ == "__main__":
    main()
