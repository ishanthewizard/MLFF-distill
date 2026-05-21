#!/usr/bin/env python3
"""Core RDF computation functions.

All functions are pure: they accept paths/arrays and return data structures,
no file I/O or plotting side effects.
"""

import sys
from math import pi
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io.trajectory import Trajectory
from ase.geometry import find_mic

# make gromacs_io importable when this module is imported from any cwd
_OBS = Path(__file__).resolve().parents[2]
if str(_OBS) not in sys.path:
    sys.path.insert(0, str(_OBS))


def _is_gromacs(path: Path) -> bool:
    return path.suffix.lower() in (".xtc", ".trr")


def _open_frames(traj_path: Path, start: int, end: int, stride: int,
                 topology: Path | None = None):
    """Return an iterable of ASE-Atoms-like frame objects."""
    from gromacs_io import mda_frame_iter
    topo = topology or traj_path
    return list(mda_frame_iter(topo, traj_path, start, end, stride))


def get_rdf_pairs(sys_dir: str) -> list[tuple[str, str]]:
    """Infer cation-partner pairs from a system directory name."""
    n = sys_dir.lower()
    if "lipf6" in n:
        return [("Li", "O"), ("Li", "F")]
    if "naotf" in n:
        return [("Na", "S"), ("Na", "O")]
    if "napf6" in n:
        return [("Na", "O"), ("Na", "F")]
    return [("Na", "O")]


def get_windows(
    dt_fs: float,
    total_ns: float,
    skip_ns: float = 0.1,
    window_ns: float = 0.9,
    slide_ns: float = 1.0,
) -> list[tuple[float, float, int, int]]:
    """Return list of (t_start_ns, t_end_ns, f_start, f_end) sliding windows."""
    windows = []
    t = skip_ns
    while t + window_ns <= total_ns + 1e-9:
        t_end = min(t + window_ns, total_ns)
        f_start = int(round(t * 1e6 / dt_fs))
        f_end = int(round(t_end * 1e6 / dt_fs))
        windows.append((t, t_end, f_start, f_end))
        t += slide_ns
    return windows


def _rdf_hist(traj_frames, cation: str, partner: str, bins: np.ndarray):
    """Accumulate pair-distance histogram over a sequence of ASE Atoms frames."""
    r_mid = 0.5 * (bins[:-1] + bins[1:])
    hist = np.zeros(len(r_mid), dtype=np.float64)
    n_cat = n_part = vol_sum = frame_count = 0.0

    for at in traj_frames:
        syms = at.get_chemical_symbols()
        ic = [i for i, s in enumerate(syms) if s == cation]
        ip = [i for i, s in enumerate(syms) if s == partner]
        if not ic or not ip:
            continue
        pos = at.get_positions()
        cell, pbc = at.get_cell(), at.get_pbc()
        for rc in pos[ic]:
            disp, _ = find_mic(pos[ip] - rc, cell, pbc)
            hist += np.histogram(np.linalg.norm(disp, axis=1), bins=bins)[0]
        n_cat += len(ic)
        n_part += len(ip)
        vol_sum += at.get_volume()
        frame_count += 1

    return hist, n_cat, n_part, vol_sum, frame_count


def _gr_from_hist(hist, n_cat, n_part, vol_sum, frame_count, bins):
    """Normalise accumulated histogram into g(r) and n(r)."""
    shell_vol = 4.0 / 3.0 * pi * (bins[1:] ** 3 - bins[:-1] ** 3)
    r_mid = 0.5 * (bins[:-1] + bins[1:])
    rho = (n_part / frame_count) / (vol_sum / frame_count)
    counts_per_cat = hist / n_cat
    g_r = counts_per_cat / (rho * shell_vol)
    n_r = np.cumsum(counts_per_cat)
    return r_mid, g_r, n_r


def compute_rdf(
    traj_path: str | Path,
    cation: str,
    partner: str,
    dt_fs: float,
    n_frames: int = 1000,
    r_max: float = 10.0,
    dr: float = 0.05,
    skip_ns: float = 0.1,
    window_ns: float = 0.9,
    topology: Path | None = None,
    preloaded_frames: list | None = None,
) -> pd.DataFrame:
    """Compute g(r) and n(r) over a fixed time window.

    If preloaded_frames is provided (list of ASE-like frame objects), it is
    used directly — no file I/O. This is the fast path for GROMACS.
    Returns a DataFrame with columns: r, g_r, n_r.
    """
    bins = np.arange(0.0, r_max + dr, dr)

    if preloaded_frames is not None:
        frames = preloaded_frames
    else:
        traj_path = Path(traj_path)
        start = int(skip_ns * 1e6 / dt_fs)
        if _is_gromacs(traj_path):
            from gromacs_io import n_frames_gromacs
            tpr = topology or traj_path.with_suffix(".tpr")
            n_tot = n_frames_gromacs(tpr, traj_path)
            end = min(start + int(window_ns * 1e6 / dt_fs), n_tot)
            stride = max(1, (end - start) // n_frames)
            frames = _open_frames(traj_path, start, end, stride, topology)
        else:
            traj = Trajectory(str(traj_path), mode="r")
            end = min(start + int(window_ns * 1e6 / dt_fs), len(traj))
            stride = max(1, (end - start) // n_frames)
            frames = list(traj[start:end:stride])
            traj.close()

    hist, n_cat, n_part, vol_sum, fc = _rdf_hist(frames, cation, partner, bins)
    if fc == 0:
        raise ValueError("No valid frames found")
    r_mid, g_r, n_r = _gr_from_hist(hist, n_cat, n_part, vol_sum, fc, bins)
    return pd.DataFrame({"r": r_mid, "g_r": g_r, "n_r": n_r})


def compute_rdf_window(
    traj: Trajectory,
    f_start: int,
    f_end: int,
    cation: str,
    partner: str,
    n_frames: int = 900,
    r_max: float = 10.0,
    dr: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute g(r) and n(r) for a specific frame range within an open trajectory.

    Returns (r_mid, g_r, n_r).
    """
    bins = np.arange(0.0, r_max + dr, dr)
    stride = max(1, (f_end - f_start) // n_frames)
    frames = [traj[i] for i in range(f_start, f_end, stride)]
    hist, n_cat, n_part, vol_sum, fc = _rdf_hist(frames, cation, partner, bins)
    if fc == 0:
        r_mid = 0.5 * (bins[:-1] + bins[1:])
        return r_mid, np.zeros(len(r_mid)), np.zeros(len(r_mid))
    r_mid, g_r, n_r = _gr_from_hist(hist, n_cat, n_part, vol_sum, fc, bins)
    return r_mid, g_r, n_r


def compute_rdf_sliding(
    traj_path: str | Path,
    cation: str,
    partner: str,
    dt_fs: float,
    total_ns: float = 20.0,
    skip_ns: float = 0.1,
    window_ns: float = 0.9,
    slide_ns: float = 1.0,
    n_frames_per_window: int = 900,
    r_max: float = 10.0,
    dr: float = 0.05,
    std_thresh: float = 0.05,
    topology: Path | None = None,
) -> dict:
    """Sliding-window RDF analysis over a trajectory.

    Returns a dict with keys:
      r_mid, all_gr, all_nr, mean_gr, std_gr, mean_nr, std_nr,
      windows, equilibration_cutoff_ns, peak_summary (list of dicts)
    """
    traj_path = Path(traj_path)
    windows = get_windows(dt_fs, total_ns, skip_ns, window_ns, slide_ns)

    if _is_gromacs(traj_path):
        from gromacs_io import n_frames_gromacs
        tpr = topology or traj_path.with_suffix(".tpr")
        n_total = n_frames_gromacs(tpr, traj_path)
        _traj_handle = None   # will use _open_frames per window
    else:
        _traj_handle = Trajectory(str(traj_path), mode="r")
        n_total = len(_traj_handle)

    all_gr, all_nr = [], []
    r_mid = None
    summary = []

    try:
        for w_idx, (t_s, t_e, f_s, f_e) in enumerate(windows):
            f_e = min(f_e, n_total)
            if f_e <= f_s:
                continue
            if _traj_handle is not None:
                r, g, n = compute_rdf_window(_traj_handle, f_s, f_e, cation, partner,
                                             n_frames_per_window, r_max, dr)
            else:
                frames = _open_frames(traj_path, f_s, f_e,
                                      max(1, (f_e - f_s) // n_frames_per_window),
                                      topology)
                bins = np.arange(0.0, r_max + dr, dr)
                hist, nc, np_, vs, fc = _rdf_hist(frames, cation, partner, bins)
                if fc == 0:
                    continue
                r, g, n = _gr_from_hist(hist, nc, np_, vs, fc, bins)
            if r_mid is None:
                r_mid = r
            all_gr.append(g)
            all_nr.append(n)

            # first peak
            mask = r >= 1.5
            pk_idx_local = int(np.argmax(g[mask]))
            pk_r = float(r[mask][pk_idx_local])
            pk_h = float(g[mask][pk_idx_local])
            abs_pk = np.searchsorted(r, 1.5) + pk_idx_local
            min_idx = int(np.argmin(g[abs_pk:])) + abs_pk if len(g[abs_pk:]) else abs_pk
            summary.append({
                "window": w_idx + 1,
                "t_start_ns": t_s,
                "t_end_ns": t_e,
                "peak_r_A": round(pk_r, 3),
                "peak_gr": round(pk_h, 3),
                "shell_cut_A": round(float(r[min_idx]), 3),
                "CN": round(float(n[min_idx]), 3),
            })
    finally:
        if _traj_handle is not None:
            _traj_handle.close()

    if not all_gr:
        return {}

    all_gr = np.array(all_gr)
    all_nr = np.array(all_nr)
    n_win = all_gr.shape[0]

    # equilibration cutoff: first window from which std/mean < threshold
    cutoff_ns = None
    for w in range(n_win):
        seg = all_gr[w:]
        if seg.shape[0] < 2:
            break
        seg_std = seg.std(axis=0, ddof=1)
        seg_mean = seg.mean(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            rr = np.nanmean(np.where(seg_mean > 0.1, seg_std / seg_mean, np.nan))
        if rr < std_thresh:
            cutoff_ns = windows[w][0]
            break

    return {
        "r_mid": r_mid,
        "all_gr": all_gr,
        "all_nr": all_nr,
        "mean_gr": all_gr.mean(axis=0),
        "std_gr": all_gr.std(axis=0, ddof=1),
        "mean_nr": all_nr.mean(axis=0),
        "std_nr": all_nr.std(axis=0, ddof=1),
        "windows": windows[:n_win],
        "equilibration_cutoff_ns": cutoff_ns,
        "peak_summary": summary,
    }
