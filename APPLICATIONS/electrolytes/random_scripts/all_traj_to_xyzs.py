#!/usr/bin/env python3
import os
import re
import random
from collections import defaultdict
from ase.io import iread, write
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator as SPC

# ----------------------------- CONFIG ---------------------------------
INPUT_DIR = "/global/cfs/cdirs/m4558/distillation_project/all_trajs_min50ps"
OUTPUT_ROOT = "/global/cfs/cdirs/m4558/distillation_project/all_xyzs"
PER_SYSTEM_DIR = os.path.join(OUTPUT_ROOT, "per_systems")
NAPF6_DIR = os.path.join(OUTPUT_ROOT, "napf6")
NAOTF_DIR = os.path.join(OUTPUT_ROOT, "naotf")
COMBINED_DIR = os.path.join(OUTPUT_ROOT, "combined")

# Sampling settings: first 50 ps at 10 fs base step, keep every 5th → 50 fs → 1000 frames
BASE_DT_FS = 10          # underlying traj stride in fs (your note implies 10 fs)
FIRST_WINDOW_PS = 50     # only the first 50 ps
KEEP_EVERY = 5           # take every 5th frame -> 50 fs sampling
FRAMES_TO_CONSIDER = int((FIRST_WINDOW_PS * 1000) / BASE_DT_FS)  # 50 ps / 10 fs = 5000 frames
EXPECTED_SUBSAMPLED = FRAMES_TO_CONSIDER // KEEP_EVERY           # 5000 / 5 = 1000

# Train/Val split
TRAIN_FRAC = 0.9
RNG_SEED = 1337  # for reproducibility
# ----------------------------------------------------------------------


STOP_TOKENS = set([
    "pfactor", "1fs", "mask", "t", "s1p1", "s2p2", "s3p3", "re1", "re2", "re3",
    "1m", "traj", "wrapped", "small", "1p1", "2p2", "omol", "0.1"
])

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def clone_with_spc(src):
    """Copy Atoms and re-attach a SinglePointCalculator with any available results."""
    a = src.copy()
    e = f = s = None
    # Grab results if present; these **won’t** trigger a new calc if SinglePointCalculator is attached
    try: e = src.get_potential_energy()
    except Exception: pass
    try: f = src.get_forces()
    except Exception: pass
    try: s = src.get_stress()
    except Exception: pass

    if (e is not None) or (f is not None) or (s is not None):
        # Attach a fresh SinglePointCalculator bound to the *new* Atoms copy
        a.calc = SPC(a,
                     energy=float(e) if e is not None else None,
                     forces=np.array(f) if f is not None else None,
                     stress=np.array(s) if s is not None else None)
    return a

def parse_system_name(filepath: str) -> str:
    """
    Extract a system key like 'napf6_diglyme' or 'naotf_dme' from the filename.
    Assumes names like:
      md_omol_napf6_diglyme_pfactor_0.1_1fs_mask_t.traj
      md_omol_naotf_pc_1m_s1p1.traj
      md_omol_napf6_dme_re1.traj
    Logic: find 'omol' token, take subsequent tokens until a STOP token.
    """
    base = os.path.basename(filepath)
    if base.endswith(".traj"):
        base = base[:-5]
    parts = base.split("_")
    start = 0
    if "omol" in parts:
        start = parts.index("omol") + 1

    sys_tokens = []
    for tok in parts[start:]:
        # normalize token
        tok_norm = tok.lower()
        if tok_norm in STOP_TOKENS:
            break
        # handle generic "re<number>" like re4
        if re.fullmatch(r"re\d+", tok_norm):
            break
        sys_tokens.append(tok_norm)

    # Safety: if somehow empty, fall back to whatever follows 'omol' or whole base
    if not sys_tokens and start < len(parts):
        sys_tokens = [parts[start].lower()]
    if not sys_tokens:
        sys_tokens = [base.lower()]

    return "_".join(sys_tokens)

def load_first_window_and_subsample(filepath: str):
    """
    From the first 5000 frames (50 ps at 10 fs), take every 5th → 1000 frames.
    Returns list[Atoms], and also the list of original frame indices kept.
    """
    selected = []
    kept_indices = []
    for i, atoms in enumerate(iread(filepath, index=":")):
        if i >= FRAMES_TO_CONSIDER:
            break
        if i % KEEP_EVERY == 0:
            # a = atoms.copy()  # detach from generator state
            a = clone_with_spc(atoms)
            kept_indices.append(i)
            selected.append(a)
    return selected, kept_indices

def main():
    random.seed(RNG_SEED)

    # Collect per-system data
    per_system_train = defaultdict(list)  # system -> list[Atoms]
    per_system_val   = defaultdict(list)  # system -> list[Atoms]

    # Aggregation buckets
    agg_napf6_train, agg_napf6_val = [], []
    agg_naotf_train, agg_naotf_val = [], []
    agg_all_train,  agg_all_val    = [], []

    # Global unique ID counter across ALL trajectories and frames
    global_uid = 0

    # Discover trajectories
    traj_files = [os.path.join(INPUT_DIR, f)
                  for f in os.listdir(INPUT_DIR)
                  if f.endswith(".traj")]
    traj_files.sort()

    print(f"Found {len(traj_files)} .traj files in {INPUT_DIR}")

    for tpath in traj_files:
        system = parse_system_name(tpath)
        print(f"\nProcessing: {os.path.basename(tpath)}  → system='{system}'")


        subsampled, kept_src_idx = load_first_window_and_subsample(tpath)
        
        n_sub = len(subsampled)
        if n_sub != EXPECTED_SUBSAMPLED:
            print(f"  WARNING: expected {EXPECTED_SUBSAMPLED} frames, got {n_sub}. Proceeding anyway.")

        # Assign globally unique IDs + annotate info
        for j, atoms in enumerate(subsampled):
            atoms.info["id"] = global_uid           # globally unique integer
            atoms.info["uid"] = f"f{global_uid:08d}"# optional nice string format
            atoms.info["system"] = system
            atoms.info["source_file"] = os.path.basename(tpath)
            atoms.info["source_index"] = kept_src_idx[j] if j < len(kept_src_idx) else None
            global_uid += 1

        # Per-trajectory random split (90/10)
        idxs = list(range(n_sub))
        random.shuffle(idxs)
        n_train = int(round(TRAIN_FRAC * n_sub))
        train_idx = set(idxs[:n_train])
        val_idx   = set(idxs[n_train:])

        train_atoms = [subsampled[i] for i in range(n_sub) if i in train_idx]
        val_atoms   = [subsampled[i] for i in range(n_sub) if i in val_idx]

        print(f"  Frames: {n_sub}  → train={len(train_atoms)}  val={len(val_atoms)}")

        # Stash for per-system writes
        per_system_train[system].extend(train_atoms)
        per_system_val[system].extend(val_atoms)

        # Add to salt-level and global aggregations WITHOUT re-splitting
        salt = system.split("_")[0]
        if salt == "napf6":
            agg_napf6_train.extend(train_atoms)
            agg_napf6_val.extend(val_atoms)
        if salt == "naotf":
            agg_naotf_train.extend(train_atoms)
            agg_naotf_val.extend(val_atoms)

        agg_all_train.extend(train_atoms)
        agg_all_val.extend(val_atoms)

    # ----------------- Write per-system XYZs -----------------
    print("\nWriting per-system train/val XYZs ...")
    for system in sorted(per_system_train.keys() | per_system_val.keys()):
        sys_root = os.path.join(PER_SYSTEM_DIR, system)
        sys_train_dir = os.path.join(sys_root, "train")
        sys_val_dir   = os.path.join(sys_root, "val")
        ensure_dir(sys_train_dir)
        ensure_dir(sys_val_dir)

        train_xyz = os.path.join(sys_train_dir, "train.xyz")
        val_xyz   = os.path.join(sys_val_dir, "val.xyz")

        tlist = per_system_train.get(system, [])
        vlist = per_system_val.get(system, [])

        if tlist:
            write(train_xyz, tlist)
        if vlist:
            write(val_xyz, vlist)

        print(f"  {system:30s}  train={len(tlist):4d}  → {train_xyz}")
        print(f"  {system:30s}  val  ={len(vlist):4d}  → {val_xyz}")

    # --------------- Write napf6 / naotf aggregates ----------
    print("\nWriting salt-level aggregates ...")

    # napf6
    napf6_train_dir = os.path.join(NAPF6_DIR, "train")
    napf6_val_dir   = os.path.join(NAPF6_DIR, "val")
    ensure_dir(napf6_train_dir); ensure_dir(napf6_val_dir)
    write(os.path.join(napf6_train_dir, "train.xyz"), agg_napf6_train) if agg_napf6_train else None
    write(os.path.join(napf6_val_dir,   "val.xyz"),   agg_napf6_val)   if agg_napf6_val   else None
    print(f"  napf6: train={len(agg_napf6_train)}  val={len(agg_napf6_val)}")

    # naotf
    naotf_train_dir = os.path.join(NAOTF_DIR, "train")
    naotf_val_dir   = os.path.join(NAOTF_DIR, "val")
    ensure_dir(naotf_train_dir); ensure_dir(naotf_val_dir)
    write(os.path.join(naotf_train_dir, "train.xyz"), agg_naotf_train) if agg_naotf_train else None
    write(os.path.join(naotf_val_dir,   "val.xyz"),   agg_naotf_val)   if agg_naotf_val   else None
    print(f"  naotf: train={len(agg_naotf_train)}  val={len(agg_naotf_val)}")

    # ---------------- Write combined aggregates ---------------
    print("\nWriting combined (all-systems) aggregates ...")
    comb_train_dir = os.path.join(COMBINED_DIR, "train")
    comb_val_dir   = os.path.join(COMBINED_DIR, "val")
    ensure_dir(comb_train_dir); ensure_dir(comb_val_dir)
    write(os.path.join(comb_train_dir, "train.xyz"), agg_all_train) if agg_all_train else None
    write(os.path.join(comb_val_dir,   "val.xyz"),   agg_all_val)   if agg_all_val   else None
    print(f"  combined: train={len(agg_all_train)}  val={len(agg_all_val)}")

    print("\nDone.")

if __name__ == "__main__":
    main()
