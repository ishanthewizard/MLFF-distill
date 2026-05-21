#!/usr/bin/env python3
"""Shared GROMACS I/O utilities for the observable toolbox.

Provides a thin adapter that makes MDAnalysis frames look like ASE Atoms
so that RDF, density, and energy compute modules can handle both .traj and
.xtc trajectories transparently.
"""

from pathlib import Path
from typing import Iterator

import numpy as np


# ── cell helper (mirrors msds_calculation_batch_gromacs.py) ──────────────────

def _cell_from_ts(ts) -> np.ndarray:
    """Return a 3×3 cell matrix (Å) from an MDAnalysis Timestep."""
    triclinic = getattr(ts, "triclinic_dimensions", None)
    if triclinic is not None:
        arr = np.asarray(triclinic, dtype=float)
        if arr.shape == (3, 3):
            return arr
    lengths = np.asarray(ts.dimensions[:3], dtype=float)
    return np.diag(lengths)


# ── element-GRO topology generator ───────────────────────────────────────────

def _infer_element(name: str) -> str:
    """Convert GROMACS atom type / name to element symbol (best-effort)."""
    s = name.strip().upper()
    if s.startswith("NA"):  return "Na"
    if s.startswith("LI"):  return "Li"
    if s.startswith("CL"):  return "Cl"
    if s.startswith("BR"):  return "Br"
    if s.startswith("SI"):  return "Si"
    if s.startswith("MG"):  return "Mg"
    return s[0].upper() + s[1:2].lower() if len(s) > 1 and s[1:2].islower() else s[0]


def ensure_element_gro(tpr: Path, xtc: Path, cache_dir: Path) -> Path:
    """Return path to element-GRO topology, creating it if absent.

    The GRO has element-like atom names so species matching in the
    component_dictionary works (same approach as msds_calculation_gromacs_CLI.py).

    cache_dir should be a STABLE shared path (not a timestamped run dir)
    so the GRO is reused across runs.
    """
    import MDAnalysis as mda

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{xtc.stem}.element.gro"
    if out.exists():
        return out

    # TPR already contains starting coordinates — no need to open the XTC
    u = mda.Universe(str(tpr))
    ts = u.trajectory[0]
    atoms = u.atoms
    pos_nm = atoms.positions / 10.0
    dims = np.asarray(ts.dimensions[:3], dtype=float) / 10.0

    with out.open("w") as f:
        f.write("Element topology generated from TPR+XTC\n")
        f.write(f"{len(atoms):5d}\n")
        for i, (atom, xyz) in enumerate(zip(atoms, pos_nm), start=1):
            elem = _infer_element(atom.name)
            f.write(f"{1:5d}{'SYS':<5}{elem:>5}{i:5d}"
                    f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}\n")
        f.write(f"{dims[0]:10.5f}{dims[1]:10.5f}{dims[2]:10.5f}\n")

    return out


# ── frame iterator ────────────────────────────────────────────────────────────

class _MDAFrame:
    """Minimal ASE-Atoms-like wrapper around an MDAnalysis timestep."""

    def __init__(self, symbols, positions, cell, masses):
        self._symbols  = symbols
        self._pos      = positions.copy()
        self._cell     = cell
        self._masses   = masses

    def get_chemical_symbols(self):
        return self._symbols

    def get_positions(self):
        return self._pos

    def get_cell(self):
        return self._cell

    def get_pbc(self):
        return [True, True, True]

    def get_masses(self):
        return self._masses

    def get_volume(self):
        return float(abs(np.linalg.det(self._cell)))

    def get_kinetic_energy(self):
        return None

    def get_temperature(self):
        return None


def preload_gromacs_frames(
    topology: Path,
    xtc: Path,
    dt_fs: float,
    max_ns: float | None = None,
    n_sample: int = 2000,
    start_frame: int = 0,
) -> list:
    """Read all needed GROMACS frames into memory in a single sequential pass.

    Returns a list of _MDAFrame objects subsampled to n_sample frames,
    covering [start_frame, max_ns] of the trajectory.

    This avoids reopening the Universe multiple times for RDF, density, and MSD.
    """
    import MDAnalysis as mda

    u = mda.Universe(str(topology), str(xtc))
    n_total = len(u.trajectory)
    atoms = u.atoms
    symbols = [s.capitalize() for s in atoms.types]
    masses = atoms.masses

    end_frame = n_total
    if max_ns is not None:
        end_frame = min(n_total, start_frame + int(max_ns * 1e6 / dt_fs))

    avail = end_frame - start_frame
    stride = max(1, avail // n_sample)

    frames = []
    for ts in u.trajectory[start_frame:end_frame:stride]:
        cell = _cell_from_ts(ts)
        frames.append(_MDAFrame(symbols, atoms.positions, cell, masses))

    return frames


def mda_frame_iter(
    topology: Path,
    xtc: Path,
    start_frame: int = 0,
    end_frame: int | None = None,
    stride: int = 1,
    max_ns: float | None = None,
) -> Iterator[_MDAFrame]:
    """Iterate over MDAnalysis frames, yielding ASE-like _MDAFrame objects.

    For GROMACS, topology should be the element-GRO (from ensure_element_gro).
    For self-contained formats (.traj), pass the same path as both arguments.
    max_ns caps the end frame using dt read from the trajectory metadata.
    """
    import MDAnalysis as mda
    if str(topology) == str(xtc):
        u = mda.Universe(str(xtc))
    else:
        u = mda.Universe(str(topology), str(xtc))
    atoms = u.atoms
    symbols = [s.capitalize() for s in atoms.types]
    masses  = atoms.masses

    n_total = len(u.trajectory)
    end = n_total if end_frame is None else min(end_frame, n_total)
    if max_ns is not None:
        dt_ps = float(u.trajectory.dt)  # ps, from trajectory metadata
        end = min(end, start_frame + int(max_ns * 1000.0 / dt_ps))
    for ts in u.trajectory[start_frame:end:stride]:
        cell = _cell_from_ts(ts)
        yield _MDAFrame(symbols, atoms.positions, cell, masses)


def n_frames_gromacs(topology: Path, xtc: Path) -> int:
    """Return total number of frames in a GROMACS trajectory.

    topology should be the element-GRO (fast) not the raw TPR (slow).
    Falls back to reading the offset NPZ directly if available.
    """
    # fastest: read the pre-built offset index directly (MDAnalysis naming: .stem.xtc_offsets.npz)
    offset_npz = xtc.parent / f".{xtc.stem}.xtc_offsets.npz"
    if offset_npz.exists():
        import numpy as np
        try:
            return int(np.load(str(offset_npz))["offsets"].shape[0])
        except Exception:
            pass
    import MDAnalysis as mda
    u = mda.Universe(str(topology), str(xtc))
    return len(u.trajectory)
