#!/usr/bin/env python3
"""Energy MAE computation between teacher and student models on a trajectory.

All functions are pure: accept paths/arrays, return data structures.
Energy is a scalar per frame (eV); per-atom energy divides by N_atoms.
"""

import sys
from pathlib import Path

import numpy as np
from ase.io.trajectory import Trajectory
from tqdm import tqdm

_OBS  = Path(__file__).resolve().parents[1]   # observable_scripts/
_ELEC = Path(__file__).resolve().parents[2]   # electrolytes/
for _p in (_OBS, _ELEC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load_calc(ckpt_path: str | Path):
    from get_calc import get_customized_uma_calc
    return get_customized_uma_calc(str(ckpt_path))


def _resolve_stride(n_tot: int, dt_fs: float,
                    analyze_dt_ps: float | None, n_frames: int) -> int:
    """Return frame stride.

    Priority: analyze_dt_ps (time-based) > n_frames (count-based).
    """
    if analyze_dt_ps is not None:
        return max(1, round(analyze_dt_ps * 1000.0 / dt_fs))
    return max(1, n_tot // n_frames)


def compute_energy_mae(
    traj_path: str | Path,
    teacher_ckpt: str | Path,
    student_ckpt: str | Path | None = None,
    n_frames: int = 500,
    dt_fs: float = 100.0,
    max_ns: float | None = None,
    analyze_dt_ps: float | None = None,
) -> dict:
    """Compute per-frame energy MAE between teacher and student on a trajectory.

    If student_ckpt is None, the energy stored in the trajectory's calc results
    is used as the student reference (avoids redundant re-inference).

    Sampling resolution is controlled by analyze_dt_ps (preferred) or n_frames:
      analyze_dt_ps : evaluate one frame every this many picoseconds of sim time.
      n_frames      : fallback — sample this many frames uniformly (used when
                      analyze_dt_ps is None).

    Returns dict with keys:
        times_ns     : ndarray (n,)  — frame timestamps in ns
        mae_total    : ndarray (n,)  — |E_teacher − E_student| per frame (eV)
        mae_per_atom : ndarray (n,)  — |E_teacher − E_student| / N_atoms per frame (eV/atom)
        n_atoms      : ndarray (n,)  — number of atoms per frame
        stride       : int           — frame stride used
        dt_fs        : float         — trajectory frame interval (fs)
    """
    traj_path = Path(traj_path)

    with Trajectory(str(traj_path), mode="r") as _t:
        n_tot = len(_t)
    if max_ns is not None:
        n_tot = min(n_tot, max(1, int(max_ns * 1e6 / dt_fs)))
    stride  = _resolve_stride(n_tot, dt_fs, analyze_dt_ps, n_frames)
    indices = list(range(0, n_tot, stride))

    teacher_calc = _load_calc(teacher_ckpt)
    student_calc = _load_calc(student_ckpt) if student_ckpt is not None else None

    times, mae_total_list, mae_per_atom_list, n_atoms_list = [], [], [], []
    with Trajectory(str(traj_path), mode="r") as traj:
        for idx in tqdm(indices, desc=f"energy_mae {traj_path.name}", unit="frame"):
            at = traj[idx]
            at.set_pbc([True, True, True])
            at.wrap()
            at.info.setdefault("charge", 0)
            at.info.setdefault("spin", 1)

            at_t = at.copy()
            at_t.calc = teacher_calc
            e_teacher = at_t.get_potential_energy()  # scalar eV

            if student_calc is not None:
                at_s = at.copy()
                at_s.calc = student_calc
                e_student = at_s.get_potential_energy()
            elif at.calc is not None and "energy" in at.calc.results:
                e_student = float(at.calc.results["energy"])
            else:
                continue

            n = len(at)
            err = abs(e_teacher - e_student)
            mae_total_list.append(err)
            mae_per_atom_list.append(err / n)
            n_atoms_list.append(n)
            times.append(idx * dt_fs * 1e-6)

    return {
        "times_ns":     np.array(times),
        "mae_total":    np.array(mae_total_list),
        "mae_per_atom": np.array(mae_per_atom_list),
        "n_atoms":      np.array(n_atoms_list),
        "stride":       stride,
        "dt_fs":        dt_fs,
    }
