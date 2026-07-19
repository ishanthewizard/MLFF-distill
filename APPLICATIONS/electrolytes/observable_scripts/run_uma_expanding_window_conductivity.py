#!/usr/bin/env python3
"""Expanding-window Onsager/Nernst-Einstein conductivity + cell-size analysis
for the UMA (uma) 1 M / 298 K electrolyte production trajectories.

WHAT THIS DOES
--------------
For each system we compute the byteff2 Onsager and Nernst-Einstein ionic
conductivity over a set of *expanding* averaging windows that all start at
t = 0:  0-1 ns, 0-1.5 ns, ..., 0-4 ns.  This is a convergence study: it shows
how sigma settles as the displacement-averaging window grows.

METHOD (matches the established byteff2 convention, see analyze-md memory)
-------------------------------------------------------------------------
  * Trajectories are saved at dt = 10 fs/frame.  We subsample to an effective
    lag dt of 1 ps/frame (stride = 100).
  * byteff2.onsager_calc() drops the first 200 loaded frames as extra
    equilibration and hardcodes its MSD-slope fit to lags [50, 200) frames.
    At 1 ps/frame that is a fixed 50-200 ps diffusive fit window, applied to
    the MSD averaged over each 0-W window.
  * "Slice once, sweep windows": each trajectory's 0-4 ns positions are read &
    unwrapped ONE time; every window is a plain array slice of that load, so
    the whole sweep costs one read per system.  (Verified to match a full
    per-window reload -- see memory `byteff2-onsager-window-convergence`.)
  * Box is NPT (anisotropic, fluctuating V); volume fed to onsager_calc is the
    mean over each individual window.

Also runs the framework cell-size analysis (a, b, c, volume vs time) over the
full available trajectory for every system.

Env: /global/homes/y/yuejian/project/MLFF-distill/yuejian/envs/fairchemV2_new
"""

import sys
import json
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────
# paths / config
# ──────────────────────────────────────────────────────────────────────────
OBS = Path("/global/homes/y/yuejian/project/MLFF-distill/APPLICATIONS/"
           "electrolytes/observable_scripts")
BASE = Path("/global/homes/y/yuejian/project/MLFF-distill/m5250/distillation_proj/"
            "simulation_results/UMA/uma/20ns/20ns_solute_solvent_1M/298K")
ANALYSIS_ROOT = Path("/global/homes/y/yuejian/project/MLFF-distill/m5250/"
                     "distillation_proj/simulation_results/UMA/uma/analysis")
COND_OUT = ANALYSIS_ROOT / "conductivity_expanding_window_first4ns"
CELL_OUT = ANALYSIS_ROOT / "cell_size"
COND_OUT.mkdir(parents=True, exist_ok=True)
CELL_OUT.mkdir(parents=True, exist_ok=True)

DT_FS        = 10.0        # saved frame spacing (fs)
LOAD_DT_PS   = 1.0         # effective lag dt after subsampling (ps) -> fit 50-200 ps
MAX_WINDOW_NS = 4.0        # sweep the first 4 ns
WINDOWS_NS   = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]   # expanding windows (all start at 0)
T_K          = 298.0
Z_CAT, Z_ANION = 1.0, -1.0
VISC_CP      = 1.0         # placeholder; only affects Dself_inf, NOT sigma

# system dir-name -> (label, cat_symbol, anion_symbol, solvent_symbol, color)
SYSTEMS = [
    ("md_omol_napf6_diglyme_pfactor_0.1_1fs",
        dict(label="napf6_diglyme", cat="Na", anion="PF6", solvent="Diglyme", color="#1f77b4")),
    ("md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1",
        dict(label="napf6_pc",      cat="Na", anion="PF6", solvent="PC",      color="#ff7f0e")),
    ("naotf_diglyme",
        dict(label="naotf_diglyme", cat="Na", anion="OTf", solvent="Diglyme", color="#2ca02c")),
    ("naotf_dme",
        dict(label="naotf_dme",     cat="Na", anion="OTf", solvent="DME",     color="#d62728")),
    ("napf6_dme",
        dict(label="napf6_dme",     cat="Na", anion="PF6", solvent="DME",     color="#9467bd")),
]

# ──────────────────────────────────────────────────────────────────────────
# imports from the framework (reuse compute.py's carefully-set import paths)
# ──────────────────────────────────────────────────────────────────────────
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_cc = _load_module("cond_compute", OBS / "conductivity" / "compute.py")
onsager_calc              = _cc.onsager_calc
direct_groups_from_species = _cc.direct_groups_from_species
cation_dict, anion_dict, solvent_dict = _cc.cation_dict, _cc.anion_dict, _cc.solvent_dict
find_mic                  = _cc.find_mic
AseTraj                   = _cc._AseTraj

if str(OBS) not in sys.path:
    sys.path.insert(0, str(OBS))
from cell_size.compute import extract_cell_timeseries
from cell_size.plot import plot_cell_timeseries


# ──────────────────────────────────────────────────────────────────────────
# trajectory loading (single sequential read + PBC unwrap)
# ──────────────────────────────────────────────────────────────────────────
def build_species(symbols0, masses0, cat, anion, solvent):
    cat_g = direct_groups_from_species(symbols0, cation_dict[cat])
    ani_g = direct_groups_from_species(symbols0, anion_dict[anion])
    sol_g = direct_groups_from_species(symbols0, solvent_dict[solvent])
    tot = sum(len(g) for g in cat_g + ani_g + sol_g)
    if tot != len(symbols0):
        raise RuntimeError(f"atom mismatch: {tot} grouped vs {len(symbols0)} total")
    reorder_idx = ([int(i) for g in cat_g for i in g] +
                   [int(i) for g in ani_g for i in g] +
                   [int(i) for g in sol_g for i in g])
    species_order = [cat, anion, solvent]
    species_mass = {
        cat:     [float(masses0[i]) for i in cat_g[0]],
        anion:   [float(masses0[i]) for i in ani_g[0]],
        solvent: [float(masses0[i]) for i in sol_g[0]],
    }
    species_number = {cat: len(cat_g), anion: len(ani_g), solvent: len(sol_g)}
    species_charge = {cat: float(Z_CAT), anion: float(Z_ANION), solvent: 0.0}
    return (reorder_idx, species_order, species_mass, species_number,
            species_charge, len(cat_g), len(ani_g), len(sol_g))


def load_unwrapped(traj_path, reorder_idx, stride, n_loaded):
    """Sequential single-read unwrap of `n_loaded` frames at `stride`.

    Returns (positions [n_loaded, N, 3] float64 Angstrom, volumes [n_loaded])."""
    frame_idxs = list(range(0, n_loaded * stride, stride))
    T = len(frame_idxs)
    N = len(reorder_idx)
    pos = np.zeros((T, N, 3), dtype=np.float64)
    vol = np.zeros(T, dtype=np.float64)
    reorder = np.asarray(reorder_idx)
    with AseTraj(str(traj_path)) as trj:
        f = trj[frame_idxs[0]]
        prev_wrapped = f.get_positions()[reorder]
        pos[0] = prev_wrapped
        vol[0] = f.get_volume()
        for k in range(1, T):
            f = trj[frame_idxs[k]]
            curr_wrapped = f.get_positions()[reorder]
            disp, _ = find_mic(curr_wrapped - prev_wrapped,
                               f.get_cell(), pbc=f.get_pbc())
            pos[k] = pos[k - 1] + disp
            prev_wrapped = curr_wrapped
            vol[k] = f.get_volume()
            if k % 500 == 0:
                print(f"      loaded {k}/{T} frames", flush=True)
    return pos, vol


# ──────────────────────────────────────────────────────────────────────────
# main per-system driver
# ──────────────────────────────────────────────────────────────────────────
def run_system(dirname, cfg):
    label = cfg["label"]
    traj_path = BASE / dirname / f"{dirname}.traj"
    print(f"\n=== {label}  ({cfg['cat']}/{cfg['anion']}/{cfg['solvent']}) ===", flush=True)
    print(f"    {traj_path}", flush=True)

    dt_ps  = DT_FS / 1000.0
    stride = max(1, round(LOAD_DT_PS / dt_ps))          # 100
    actual_load_dt_ps = stride * dt_ps                  # 1.0 ps
    dt_correction = 1.0 / actual_load_dt_ps             # 1.0 (byteff2 assumes 1 ps/frame)

    with AseTraj(str(traj_path)) as trj:
        n_total_raw = len(trj)
        f0 = trj[0]
        symbols0 = f0.get_chemical_symbols()
        masses0  = f0.get_masses()

    avail_ns = n_total_raw * dt_ps / 1000.0
    (reorder_idx, species_order, species_mass, species_number,
     species_charge, n_cat, n_ani, n_sol) = build_species(
        symbols0, masses0, cfg["cat"], cfg["anion"], cfg["solvent"])
    print(f"    n_total={n_total_raw} ({avail_ns:.2f} ns)  "
          f"N_cat={n_cat} N_anion={n_ani} N_solvent={n_sol}", flush=True)

    # windows that fit in the available trajectory
    windows = [w for w in WINDOWS_NS if w <= avail_ns + 1e-9]
    n_load = int(round(max(windows) * 1000.0 / actual_load_dt_ps))   # loaded frames for the largest window
    if n_load * stride > n_total_raw:
        n_load = n_total_raw // stride
    print(f"    loading {n_load} frames @ {actual_load_dt_ps:.2f} ps/frame "
          f"(stride={stride}) ...", flush=True)

    pos, vol = load_unwrapped(traj_path, reorder_idx, stride, n_load)
    print(f"    load done. mean V (0-{max(windows):.0f} ns) = {vol.mean():.1f} A^3", flush=True)

    rows = []
    for w in windows:
        n_w = int(round(w * 1000.0 / actual_load_dt_ps))
        n_w = min(n_w, pos.shape[0])
        V_w = float(vol[:n_w].mean())
        res = onsager_calc(
            species_order=species_order,
            species_mass=species_mass,
            species_number=species_number,
            species_charge=species_charge,
            volume_angstrom3=V_w,
            viscosity_cP=VISC_CP,
            T_K=T_K,
            positions=pos[:n_w],
        )
        sig_o = res["conductivity_onsager"] * dt_correction
        sig_ne = res["conductivity_NE"]     * dt_correction
        Dself = [d * dt_correction for d in res["Dself_inf"]]   # cat, anion, solvent
        rows.append({
            "system": label,
            "cation": cfg["cat"], "anion": cfg["anion"], "solvent": cfg["solvent"],
            "concentration_M": 1.0, "temperature_K": T_K,
            "window_ns": w,
            "n_frames_used": n_w,
            "load_dt_ps": actual_load_dt_ps,
            "fit_window_ps": "50-200",
            "V_mean_A3": V_w,
            "sigma_onsager_mS_cm": sig_o,
            "sigma_NE_mS_cm": sig_ne,
            "sigma_onsager_uS_cm": sig_o * 1000.0,
            "sigma_NE_uS_cm": sig_ne * 1000.0,
            "D_cat_1e10_m2s": Dself[0],
            "D_anion_1e10_m2s": Dself[1],
            "D_solvent_1e10_m2s": Dself[2],
        })
        print(f"    window 0-{w:>4.1f} ns  (n={n_w:5d})  "
              f"sigma_Onsager={sig_o:8.4f} mS/cm   sigma_NE={sig_ne:8.4f} mS/cm",
              flush=True)

    df = pd.DataFrame(rows)
    csv = COND_OUT / f"conductivity_expanding_{label}.csv"
    df.to_csv(csv, index=False)
    print(f"    wrote {csv}", flush=True)

    # per-system convergence plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["window_ns"], df["sigma_onsager_mS_cm"], "o-",
            color=cfg["color"], lw=2, label="Onsager (byteff2)")
    ax.plot(df["window_ns"], df["sigma_NE_mS_cm"], "s--",
            color=cfg["color"], lw=2, alpha=0.6, label="Nernst-Einstein")
    ax.set_xlabel("Averaging window  0 → W  (ns)")
    ax.set_ylabel("Ionic conductivity (mS/cm)")
    ax.set_title(f"{label}  ({cfg['cat']}/{cfg['anion']}/{cfg['solvent']}) "
                 f"1 M 298 K\nexpanding-window conductivity (fit 50-200 ps @ 1 ps/frame)")
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    p = COND_OUT / f"conductivity_expanding_{label}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {p}", flush=True)

    return df


def run_cell_size(dirname, cfg):
    label = cfg["label"]
    traj_path = BASE / dirname / f"{dirname}.traj"
    print(f"\n--- cell_size: {label} ---", flush=True)
    res = extract_cell_timeseries(traj_path, dt_fs=DT_FS, analyze_dt_ps=10.0)  # full traj @ 10 ps
    print(f"    stride={res['stride']} ({res['stride']*DT_FS*1e-3:.1f} ps/frame)  "
          f"n={len(res['times_ns'])}  span={res['times_ns'][-1]:.2f} ns  "
          f"a={res['a'].mean():.3f} b={res['b'].mean():.3f} c={res['c'].mean():.3f} "
          f"V={res['volume'].mean():.1f}", flush=True)
    npz = CELL_OUT / f"cell_size_{label}.npz"
    np.savez_compressed(npz, **{k: v for k, v in res.items()
                                if isinstance(v, np.ndarray)})
    out = plot_cell_timeseries({label: res}, label, CELL_OUT,
                               model_colors={label: cfg["color"]},
                               roll_window_ns=0.2)
    print(f"    wrote {out}", flush=True)
    return {"system": label, "cation": cfg["cat"], "anion": cfg["anion"],
            "solvent": cfg["solvent"], "concentration_M": 1.0, "temperature_K": T_K,
            "n_samples": len(res["times_ns"]), "span_ns": float(res["times_ns"][-1]),
            "a_mean_A": float(res["a"].mean()), "a_std_A": float(res["a"].std(ddof=1)),
            "b_mean_A": float(res["b"].mean()), "b_std_A": float(res["b"].std(ddof=1)),
            "c_mean_A": float(res["c"].mean()), "c_std_A": float(res["c"].std(ddof=1)),
            "V_mean_A3": float(res["volume"].mean()), "V_std_A3": float(res["volume"].std(ddof=1))}


def main():
    all_cond = []
    cell_rows = []
    for dirname, cfg in SYSTEMS:
        try:
            all_cond.append(run_system(dirname, cfg))
        except Exception:
            import traceback
            print(f"!! conductivity FAILED for {cfg['label']}:\n{traceback.format_exc()}",
                  flush=True)
        try:
            cell_rows.append(run_cell_size(dirname, cfg))
        except Exception:
            import traceback
            print(f"!! cell_size FAILED for {cfg['label']}:\n{traceback.format_exc()}",
                  flush=True)

    # combined conductivity csv
    if all_cond:
        comb = pd.concat(all_cond, ignore_index=True)
        comb.to_csv(COND_OUT / "conductivity_expanding_ALL.csv", index=False)
        print(f"\nwrote {COND_OUT/'conductivity_expanding_ALL.csv'}", flush=True)

        # combined 2-panel figure: Onsager (left), NE (right)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)
        lbl2color = {c["label"]: c["color"] for _, c in SYSTEMS}
        for df in all_cond:
            lbl = df["system"].iloc[0]
            col = lbl2color[lbl]
            axes[0].plot(df["window_ns"], df["sigma_onsager_mS_cm"], "o-",
                         color=col, lw=2, label=lbl)
            axes[1].plot(df["window_ns"], df["sigma_NE_mS_cm"], "s-",
                         color=col, lw=2, label=lbl)
        axes[0].set_title("Onsager (byteff2)  — expanding window")
        axes[1].set_title("Nernst-Einstein  — expanding window")
        for ax in axes:
            ax.set_xlabel("Averaging window  0 → W  (ns)")
            ax.set_ylabel("Ionic conductivity (mS/cm)")
            ax.grid(True, ls=":", alpha=0.6)
            ax.legend(fontsize=9)
        fig.suptitle("UMA 1 M 298 K electrolytes — expanding-window conductivity "
                     "(fit 50-200 ps @ 1 ps/frame, first 4 ns)", fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        p = COND_OUT / "conductivity_expanding_ALL.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {p}", flush=True)

    if cell_rows:
        pd.DataFrame(cell_rows).to_csv(CELL_OUT / "cell_size_summary.csv", index=False)
        print(f"wrote {CELL_OUT/'cell_size_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
