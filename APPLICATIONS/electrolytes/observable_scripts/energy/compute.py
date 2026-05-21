#!/usr/bin/env python3
"""Core energy extraction functions.

All functions are pure: accept paths/arrays, return data structures.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from ase.io.trajectory import Trajectory


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Centered rolling average with min_periods=1."""
    w = max(1, int(w))
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()


def extract_energies(
    traj_path: str | Path,
    n_sample: int = 3000,
    dt_fs: float = 100.0,
    max_ns: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract PE, KE, total energy, and temperature from a trajectory.

    Samples n_sample frames uniformly up to max_ns (or full traj if shorter).
    Returns (times_ns, pe_eV, ke_eV, etot_eV, temp_K).
    """
    traj = Trajectory(str(traj_path), mode="r")
    n_tot = len(traj)
    if max_ns is not None:
        n_tot = min(n_tot, max(1, int(max_ns * 1e6 / dt_fs)))
    stride = max(1, n_tot // n_sample)
    indices = range(0, n_tot, stride)


    times, pe, ke, etot, temp = [], [], [], [], []
    try:
        for idx in indices:
            at = traj[idx]
            if at.calc is None or "energy" not in at.calc.results:
                continue
            times.append(idx * dt_fs * 1e-6)
            pe.append(at.calc.results["energy"])
            ke.append(at.get_kinetic_energy())
            etot.append(at.calc.results["energy"] + at.get_kinetic_energy())
            temp.append(at.get_temperature())
    finally:
        traj.close()

    return (
        np.array(times),
        np.array(pe),
        np.array(ke),
        np.array(etot),
        np.array(temp),
    )
