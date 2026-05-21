"""Production config: all 5 systems, original student model, full 20 ns analysis."""
from pathlib import Path

_ROOT = Path("/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application"
             "/other_fix_run/original_100ps/20ns_solute_solvent_1M/298K")

_SHARED = {
    # ── trajectory ────────────────────────────────────────────────────────────
    "dt_fs":           100.0,   # frame spacing in femtoseconds (100 fs = 0.1 ps for student trajs)
    "max_traj_ns":     20.0,    # cap analysis at this length; falls back to actual traj length if shorter

    # ── RDF (mean + sliding window) ───────────────────────────────────────────
    "skip_ns":         0.1,     # skip this many ns from traj start before any RDF/density analysis
    "window_ns":       10.0,    # width of each RDF window (ns); also the sliding window width
    "n_frames":        2000,    # frames sampled per window for RDF and density mean
    "sliding_window":  True,    # enable sliding-window g(r) and n(r) plots
    "rdf_slide_ns":    1.0,     # step between consecutive sliding windows (ns)
                                # → with window_ns=10, slide=1: windows [0.1-10.1], [1.1-11.1], ...

    # ── density ───────────────────────────────────────────────────────────────
    "density_roll_window_ns": 0.5,  # rolling average width for density timeseries plot (ns)

    # ── MSD / diffusivity ─────────────────────────────────────────────────────
    "eq_cut_ns":       0.0,     # skip from traj start before computing MSD time origins
    "fit_pct":         0.8,     # fit linear regime up to this fraction of max lag (e.g. 0.8 → 16 ns)
    "tau_min_fit_ns":  1.0,     # lower bound of linear fit (ns); skips ballistic + cage regime
    "slide_window_ns": 10.0,    # Panel 4: fixed window width for sliding-window D diagnostic (ns)
    "slide_step_ns":   0.5,     # Panel 4: step size of sliding window (ns)
    "n_conv_points":   200,     # resolution of D vs tau_max convergence sweep (Panel 2/3)
}

SYSTEMS = [
    {
        **_SHARED,
        "name":           "NaOTf/DME 1M",
        "traj_paths":     {"original": str(_ROOT / "naotf_dme/naotf_dme.traj")},
        "rdf_pairs":      [("Na", "S"), ("Na", "O")],
        "cat_symbol":     "Na",
        "anion_symbol":   "OTf",
        "solvent_symbol": "DME",
    },
    {
        **_SHARED,
        "name":           "NaPF6/DME 1M",
        "traj_paths":     {"original": str(_ROOT / "napf6_dme/napf6_dme.traj")},
        "rdf_pairs":      [("Na", "O"), ("Na", "F")],
        "cat_symbol":     "Na",
        "anion_symbol":   "PF6",
        "solvent_symbol": "DME",
    },
    {
        **_SHARED,
        "name":           "NaOTf/Diglyme 1M",
        "traj_paths":     {"original": str(_ROOT / "naotf_diglyme/naotf_diglyme.traj")},
        "rdf_pairs":      [("Na", "S"), ("Na", "O")],
        "cat_symbol":     "Na",
        "anion_symbol":   "OTf",
        "solvent_symbol": "Diglyme",
    },
    {
        **_SHARED,
        "name":           "NaPF6/Diglyme 1M",
        "traj_paths":     {"original": str(_ROOT / "md_omol_napf6_diglyme_pfactor_0.1_1fs/md_omol_napf6_diglyme_pfactor_0.1_1fs.traj")},
        "rdf_pairs":      [("Na", "O"), ("Na", "F")],
        "cat_symbol":     "Na",
        "anion_symbol":   "PF6",
        "solvent_symbol": "Diglyme",
    },
    {
        **_SHARED,
        "name":           "NaPF6/PC 1M",
        "traj_paths":     {"original": str(_ROOT / "md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1/md_omol_napf6_propylene_carbonate_pfactor_0.1_1fs_mask_t_s1p1.traj")},
        "rdf_pairs":      [("Na", "O"), ("Na", "F")],
        "cat_symbol":     "Na",
        "anion_symbol":   "PF6",
        "solvent_symbol": "PC",
    },
]

OUTPUT_DIR = "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/analysis/20ns"
ANALYSES   = ["rdf", "density", "energy", "msd"]
WORKERS    = 5
