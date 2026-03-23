#!/usr/bin/env python3
"""
CLI wrapper for `msds_calculation_batch_gromacs.py` that:
- does NOT require editing the batch script
- auto-discovers GROMACS trajectories in a directory (e.g. OPLS baseline runs)
- builds TARGETS and calls the existing batch `main()` to compute diffusivities

Run: 1 ps analysis spacing (traj dt 0.1 ps → stride 10), first 20 ns
--------------------------------------------------------------------
Use `--analysis-dt-ps 1` so analysis frames are 1 ps apart (every 10th frame).

/global/homes/y/yuejian/project/MLFF-distill/yuejian/envs/fairchemV2/bin/python \
  /global/homes/y/yuejian/project/MLFF-distill/APPLICATIONS/electrolytes/observable_scripts/mean_square_displacement/msds_calculation_gromacs_CLI.py \
  --input-dir /global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/opls_baseline/tpr_files_1M \
  --out-dir /global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/opls_baseline/new_results \
  --eq-time-ps 0 \
  --tau-max-fit-ps 20000 \
  --tau-min-fit-ps 1000 \
  --analysis-dt-ps 1

Typical use (OPLS baseline directory with `npt_1M_*.xtc` + matching `.tpr`):

python msds_calculation_gromacs_CLI.py \
  --input-dir /global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/opls_baseline/tpr_files_1M \
  --out-dir /path/to/output \
  --tau-max-fit-ps 20000 \
  --eq-time-ps 100 \
  --tau-min-fit-ps 1000 \
  --temperature-k 298
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


# Ensure sibling imports work when running by path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

Target = Tuple[str, str, str, str, str, str, str, str]


def discover_targets(
    input_dir: Path,
    temperature_k: int,
    concentration: str = "1M",
    only: Optional[Sequence[str]] = None,
) -> List[Target]:
    """
    Discover `*.xtc` files under input_dir and pair each with a matching `.tpr`
    in the same directory. Builds TARGETS tuples expected by the batch script.
    """
    input_dir = input_dir.resolve()
    xtcs = sorted(p for p in input_dir.glob("*.xtc") if p.is_file())
    if not xtcs:
        raise FileNotFoundError(f"No .xtc files found in {input_dir}")

    only_set = set(only) if only else None
    targets: List[Target] = []

    for xtc in xtcs:
        stem = xtc.stem
        if only_set is not None and stem not in only_set:
            continue

        tpr = input_dir / f"{stem}.tpr"
        if not tpr.exists():
            # allow alternative topology extensions if you later have them
            raise FileNotFoundError(f"Missing topology for {xtc.name}: expected {tpr}")
        # NOTE: For safety, this CLI does NOT infer components from filenames.
        # The caller must use --targets-csv or the hard-coded mapping in main().
        raise RuntimeError(
            "Auto-discovery without explicit components is disabled for safety. "
            "Use --targets-csv or run without calling discover_targets()."
        )

    if not targets:
        msg = "No targets selected."
        if only_set is not None:
            msg += f" `--only` filtered everything (requested: {sorted(only_set)})"
        raise RuntimeError(msg)

    return targets


def read_targets_csv(path: Path, only: Optional[Sequence[str]] = None) -> List[Target]:
    """
    Read explicit TARGETS from a CSV file (no inference).

    Required columns (header names):
      topology_path,traj_path,title,cat_symbol,anion_symbol,solvent_symbol,concentration_M,temperature_K

    `--only` filters by trajectory stem (Path(traj_path).stem).
    """
    only_set = set(only) if only else None
    out: List[Target] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = [
            "topology_path",
            "traj_path",
            "title",
            "cat_symbol",
            "anion_symbol",
            "solvent_symbol",
            "concentration_M",
            "temperature_K",
        ]
        missing_cols = [c for c in required if c not in (reader.fieldnames or [])]
        if missing_cols:
            raise ValueError(f"{path} missing columns: {missing_cols}. Found: {reader.fieldnames}")

        for row in reader:
            traj_path = row["traj_path"].strip()
            stem = Path(traj_path).stem
            if only_set is not None and stem not in only_set:
                continue
            out.append(
                (
                    row["topology_path"].strip(),
                    traj_path,
                    row["title"].strip(),
                    row["cat_symbol"].strip(),
                    row["anion_symbol"].strip(),
                    row["solvent_symbol"].strip(),
                    row["concentration_M"].strip(),
                    row["temperature_K"].strip(),
                )
            )
    if not out:
        msg = f"No targets loaded from {path}."
        if only_set is not None:
            msg += f" `--only` filtered everything (requested: {sorted(only_set)})"
        raise RuntimeError(msg)
    return out


def _infer_element_from_atomname(name: str) -> str:
    """
    Best-effort conversion from common GROMACS atom names (e.g. C01, H0A, O02, NA)
    to element symbols (C, H, O, Na, ...).
    """
    s = name.strip()
    if not s:
        return ""
    s_up = s.upper()
    # common 2-letter elements
    if s_up.startswith("NA"):
        return "Na"
    if s_up.startswith("LI"):
        return "Li"
    if s_up.startswith("CL"):
        return "Cl"
    if s_up.startswith("BR"):
        return "Br"
    if s_up.startswith("SI"):
        return "Si"
    # otherwise first character as element
    c = s_up[0]
    return c


def write_element_topology_gro(
    tpr_path: Path,
    xtc_path: Path,
    out_gro_path: Path,
) -> Path:
    """
    Create a minimal GRO topology file where atomnames are element-like symbols
    inferred from the original atom names. This is a workaround for TPR+XTC where
    MDAnalysis `atoms.types` are force-field types (e.g. OPLS_800) rather than elements.

    The MSD code in `msds_calculation_batch_gromacs.py` matches species by comparing
    `atoms.types` strings; using a GRO with element-ish atomnames makes that logic work
    without editing the batch script.
    """
    # Import locally to keep --dry-run lightweight
    import MDAnalysis as mda

    u = mda.Universe(str(tpr_path), str(xtc_path))
    ts = u.trajectory[0]
    atoms = u.atoms

    # positions in Å -> nm
    pos_nm = atoms.positions / 10.0

    # box lengths in Å -> nm (triclinic angles are ignored in GRO writer here)
    dims = getattr(ts, "dimensions", None)
    if dims is None or len(dims) < 3:
        raise RuntimeError("Could not read unit cell dimensions from trajectory.")
    box_nm = [float(d) / 10.0 for d in dims[:3]]

    # Write GRO manually (simple & dependency-free)
    out_gro_path.parent.mkdir(parents=True, exist_ok=True)
    with out_gro_path.open("w") as f:
        f.write("Generated element topology from TPR+XTC\n")
        f.write(f"{len(atoms):5d}\n")
        resnr = 1
        resname = "SYS"
        for i, (atom, xyz) in enumerate(zip(atoms, pos_nm), start=1):
            elem = _infer_element_from_atomname(atom.name)
            atomname = elem if elem else atom.name
            # GRO: resnr(5) resname(5) atomname(5) atomnr(5) x y z (8.3f, nm)
            f.write(f"{resnr:5d}{resname:<5}{atomname:>5}{i:5d}{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}\n")
        f.write(f"{box_nm[0]:10.5f}{box_nm[1]:10.5f}{box_nm[2]:10.5f}\n")
    return out_gro_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute MSD/diffusivities from GROMACS XTC/TRR with auto TARGETS discovery.")
    p.add_argument(
        "--input-dir",
        type=str,
        default="/global/homes/y/yuejian/project/MLFF-distill/m5024/distillation_project/opls_baseline/tpr_files_1M",
        help="Directory containing npt_1M_*.xtc and matching .tpr files.",
    )
    p.add_argument(
        "--targets-csv",
        type=str,
        default=None,
        help=(
            "Optional explicit TARGETS CSV (disables inference). "
            "Columns: topology_path,traj_path,title,cat_symbol,anion_symbol,solvent_symbol,concentration_M,temperature_K"
        ),
    )
    p.add_argument("--out-dir", "-o", required=True, type=str, help="Output directory for MSDs/plots/CSV.")
    p.add_argument("--tau-max-fit-ps", "-t", required=True, type=float, help="Maximum fitting time (ps).")
    p.add_argument("--eq-time-ps", type=float, default=100.0, help="Equilibration time before analysis (ps).")
    p.add_argument("--tau-min-fit-ps", type=float, default=1000.0, help="Minimum fit time (ps).")
    p.add_argument(
        "--known-dt-ps",
        type=float,
        default=None,
        help="Base timestep (ps). If omitted, inferred from trajectory metadata.",
    )
    p.add_argument("--n-workers", type=int, default=4, help="Workers for parallel MSD (if enabled).")
    p.add_argument("--plot-ncols", type=int, default=2, help="Columns in subplot grid.")
    p.add_argument("--parallel-msd", action="store_true", help="Enable parallel MSD calculation.")
    p.add_argument(
        "--temperature-k",
        type=int,
        default=298,
        help="Temperature label (K) to write into titles/CSV (filenames here do not encode temperature).",
    )
    p.add_argument(
        "--concentration",
        type=str,
        default="1M",
        help="Concentration label to write into titles/CSV.",
    )
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional list of trajectory stems to run (e.g. npt_1M_naotf_dme). If omitted, runs all discovered.",
    )
    p.add_argument(
        "--target-frames",
        type=int,
        default=None,
        help=(
            "Number of analysis frames to sample (controls subsampling stride). "
            "If omitted, uses int(tau_max_fit_ps - tau_min_fit_ps) to mirror the batch script."
        ),
    )
    p.add_argument(
        "--analysis-dt-ps",
        type=float,
        default=None,
        help=(
            "Desired time between analysis frames in ps. Sets target_frames so effective dt = this value. "
            "E.g. with traj dt 0.1 ps, use --analysis-dt-ps 1 to analyze every 10th frame (1 ps spacing). "
            "Ignored if --target-frames is set."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the resolved TARGETS and exit (does not import heavy deps / run MSD).",
    )
    p.add_argument(
        "--topology-mode",
        choices=["tpr", "element-gro"],
        default="element-gro",
        help=(
            "Which topology file to pass to MDAnalysis. "
            "`element-gro` generates a GRO with element-like atomnames so species matching works. "
            "`tpr` uses the raw .tpr (may fail if atom types are forcefield types)."
        ),
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tau_min = float(args.tau_min_fit_ps)
    tau_max = float(args.tau_max_fit_ps)

    if tau_max <= tau_min:
        raise ValueError(f"--tau-max-fit-ps ({tau_max}) must be > --tau-min-fit-ps ({tau_min})")

    eq_ps = float(args.eq_time_ps)
    if args.target_frames is not None:
        target_frames = int(args.target_frames)
    elif args.analysis_dt_ps is not None:
        target_frames = int(round((tau_max - eq_ps) / args.analysis_dt_ps))
    else:
        target_frames = int(tau_max - tau_min)
    if target_frames < 10:
        raise ValueError(f"target_frames too small ({target_frames}). Increase --target-frames or tau window.")

    if args.targets_csv:
        targets = read_targets_csv(Path(args.targets_csv), only=args.only)
    else:
        # Hard-coded components mapping (no inference).
        # Mirrors the explicit TARGETS style in `msds_calculation_batch_gromacs.py`.
        components = {
            # stem: (title, cat_symbol, anion_symbol, solvent_symbol)
            "npt_1M_naotf_diglyme": ("NaOTf — Diglyme", "Na", "OTf", "Diglyme"),
            "npt_1M_naotf_dme": ("NaOTf — DME", "Na", "OTf", "DME"),
            "npt_1M_napf6_diglyme": ("NaPF6 — Diglyme", "Na", "PF6", "Diglyme"),
            "npt_1M_napf6_dme": ("NaPF6 — DME", "Na", "PF6", "DME"),
            "npt_1M_napf6_pc": ("NaPF6 — PC", "Na", "PF6", "PC"),
        }

        only_set = set(args.only) if args.only else None
        xtcs = sorted(p for p in input_dir.resolve().glob("*.xtc") if p.is_file())
        if not xtcs:
            raise FileNotFoundError("No .xtc files found in %s" % str(input_dir))

        targets = []
        missing = []
        for xtc in xtcs:
            stem = xtc.stem
            if only_set is not None and stem not in only_set:
                continue

            tpr = input_dir / ("%s.tpr" % stem)
            if not tpr.exists():
                raise FileNotFoundError("Missing topology for %s: expected %s" % (xtc.name, str(tpr)))

            if stem not in components:
                missing.append(stem)
                continue

            title, cat, anion, solv = components[stem]
            if args.topology_mode == "tpr":
                topo_path = tpr
            else:
                topo_path = out_dir / "topologies_element_gro" / ("%s.element.gro" % stem)
                if not topo_path.exists():
                    write_element_topology_gro(tpr, xtc, topo_path)
            targets.append(
                (
                    str(topo_path),
                    str(xtc),
                    title,
                    cat,
                    anion,
                    solv,
                    str(args.concentration),
                    "%dK" % int(args.temperature_k),
                )
            )

        if missing:
            raise RuntimeError(
                "Found .xtc files without hard-coded component entries: %s. "
                "Add them to the `components` dict in main() or provide --targets-csv."
                % ", ".join(sorted(missing))
            )
        if not targets:
            raise RuntimeError("No targets selected (check --only filter).")

    print(f"Discovered {len(targets)} trajectory(ies) in {input_dir}")
    for tpr, xtc, title, cat, anion, solv, conc, temp in targets:
        print(f" - {Path(xtc).name} (top={Path(tpr).name}) :: {title} [{cat}/{anion}/{solv}] {temp} {conc}")

    if args.dry_run:
        return

    # Import here so `--dry-run` works even if the runtime environment
    # is missing heavy deps (pandas/MDAnalysis/matplotlib).
    from msds_calculation_batch_gromacs import main as batch_main  # noqa: E402

    batch_main(
        targets,
        float(args.eq_time_ps),
        args.known_dt_ps,
        target_frames,
        tau_min,
        tau_max,
        int(args.n_workers),
        int(args.plot_ncols),
        bool(args.parallel_msd),
        out_dir,
    )


if __name__ == "__main__":
    main()

