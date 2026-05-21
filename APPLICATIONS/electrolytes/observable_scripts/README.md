# Observable Analysis Toolbox

Analysis tools for electrolyte MD trajectories: RDF, density, energy, MSD/diffusivity.

## Directory Layout

```
observable_scripts/
├── eval.py                              ← main dispatcher (system-wise parallel)
│
├── RDF/utils/
│   ├── compute.py                       ← NEW: pure RDF computation functions
│   ├── plot.py                          ← NEW: RDF plotting functions
│   ├── rdf_base.py                      ← single-trajectory RDF CLI tool
│   ├── rdf_single_official.py           ← official single-system RDF
│   ├── rdf_wrapper.py                   ← batch wrapper over rdf_single_official
│   ├── rdf_plots.py                     ← plotting utilities for CSV-based RDF data
│   ├── rdf_plots_batch.py               ← batch plotter over directory trees
│   └── rdf_nr_wr_plots.py               ← n(r) and w(r) specialised plots
│
├── density/
│   ├── compute.py                       ← NEW: density computation functions
│   ├── plot.py                          ← NEW: density timeseries + bar-chart plots
│   └── dens_compute_stride_100.py       ← original density CLI (last-N-frames)
│
├── energy/
│   ├── compute.py                       ← NEW: energy/temperature extraction
│   └── plot.py                          ← NEW: 4-panel energy timeseries plot
│
├── mean_square_displacement/
│   ├── compute.py                       ← NEW: MSD pipeline + convergence analysis
│   ├── plot.py                          ← NEW: MSD plot + 4-panel convergence figure
│   ├── msd_with_com.py                  ← core trajectory streaming / MSD computation
│   ├── analysis_D/fit_D.py             ← standalone diffusivity fitting script
│   └── utils/component_dictionary.py   ← species symbol → atom-list mappings
│
├── cell_size/
│   ├── compute.py                       ← NEW: cell-length (a,b,c) and volume timeseries
│   └── plot.py                          ← NEW: 4-panel cell-parameter timeseries plot
│
├── Green_Kubo/                          ← VACF / Green-Kubo conductivity
├── error_analysis/                      ← force-error analysis outputs
└── old/                                 ← deprecated scripts
```

---

## Compute Modules (pure functions — no file I/O side effects)

### `RDF/utils/compute.py`
| Function | Description |
|---|---|
| `get_rdf_pairs(sys_dir)` | Infer (cation, partner) pairs from system directory name |
| `get_windows(dt_fs, total_ns, skip_ns, window_ns, slide_ns)` | Sliding window frame indices |
| `compute_rdf(traj_path, cation, partner, dt_fs, ...)` | g(r) and n(r) over a fixed time window → DataFrame |
| `compute_rdf_window(traj, f_start, f_end, cation, partner, ...)` | g(r)/n(r) for one frame range (open traj handle) |
| `compute_rdf_sliding(traj_path, cation, partner, dt_fs, ...)` | All sliding windows → dict of arrays + equilibration cutoff |

### `density/compute.py`
| Function | Description |
|---|---|
| `compute_density(traj_path, dt_fs, n_frames, skip_ns, window_ns)` | Mean density (g/cm³) ± std over a time window |
| `extract_density_timeseries(traj_path, dt_fs, n_sample)` | Full trajectory density → (times_ns, densities) |
| `compute_density_tail(traj_path, n_snapshots, stride)` | Mean density from last N frames |

### `energy/compute.py`
| Function | Description |
|---|---|
| `extract_energies(traj_path, n_sample, dt_fs)` | Returns (times_ns, pe, ke, etot, temp) arrays |
| `rolling_mean(x, w)` | Centered rolling average |

### `cell_size/compute.py`
| Function | Description |
|---|---|
| `extract_cell_timeseries(traj_path, dt_fs, n_sample, max_ns, topology, preloaded_frames, analyze_dt_ps)` | Cell lengths a, b, c (Å) and volume (Å³) vs time → dict of arrays |

### `mean_square_displacement/compute.py`
| Function | Description |
|---|---|
| `run_msd_analysis(traj_path, cat, anion, solvent, dt_fs, ...)` | Full pipeline: stream → unwrap → MSD → convergence → fit |
| `compute_convergence(tau_ps, msd_cat, msd_anion, msd_solvent, ...)` | Convergence sweep data for all 4 diagnostic panels |
| `save_msd_pickle(result, slug, output_dir)` | Dump raw MSD arrays to `.pkl` |
| `save_diffusivity_csv(rows, output_dir)` | Save final D values table to CSV |

---

## Plot Modules (save figures to disk, return Path)

### `RDF/utils/plot.py`
| Function | Output |
|---|---|
| `plot_rdf_comparison(rdf_results, sys_name, ...)` | g(r) and n(r) comparison across models |
| `plot_rdf_sliding(result, sys_name, model, pair, ...)` | Sliding-window g(r) and n(r) with colorbar |

### `density/plot.py`
| Function | Output |
|---|---|
| `plot_density_timeseries(times_and_densities, sys_name, ...)` | Instant density + rolling mean, all models overlaid |
| `plot_density_bars(density_df, ...)` | Bar chart of mean density ± std per system/model |

### `energy/plot.py`
| Function | Output |
|---|---|
| `plot_energy_timeseries(times, pe, ke, etot, temp, ...)` | 4-panel PE / KE / E_tot / T vs time |

### `cell_size/plot.py`
| Function | Output |
|---|---|
| `plot_cell_timeseries(times_and_cells, sys_name, ...)` | 4-panel a / b / c / Volume vs time, all models overlaid |

### `mean_square_displacement/plot.py`
| Function | Output |
|---|---|
| `plot_msd(result, sys_name, model, output_dir)` | MSD curves with linear fit overlay + shaded fit window |
| `plot_convergence(result, sys_name, model, output_dir)` | 2×2 diagnostic: MSD / D vs tau_max / ΔD / sliding-window D |

---

## Main Dispatcher: `eval.py`

Runs any combination of analyses for a list of systems, **one process per system**.

```bash
python eval.py --config myconfig.py \
               --analyses rdf,density,energy,msd,cell_size \
               --output-dir ./results \
               --workers 8
```

### Config file (`myconfig.py`)

```python
SYSTEMS = [
    {
        "name": "NaPF6/DME 0.1M",
        "traj_paths": {
            "UMA":     "/path/to/uma.traj",
            "student": "/path/to/student.traj",
        },

        # ── shared ──────────────────────────────────────────────────────────
        "dt_fs": {"UMA": 10.0, "student": 100.0},  # fs; or single float
```

### Trajectory formats

`eval.py` auto-detects the format from each `traj_paths` entry:

| Entry value | Detected as | TPR resolved from |
|---|---|---|
| `"/path/to.traj"` | ASE | — |
| `"/path/to.xtc"` or `"/path/to.trr"` | GROMACS | `tpr_paths[model]`, then same-stem sibling `.tpr` |
| `{"xtc": "...", "tpr": "..."}` | GROMACS | dict value (explicit) |

**GROMACS example — dict form (self-contained):**

```python
"traj_paths": {
    "OPLS": {"xtc": "/path/to/npt_1M_napf6_dme.xtc",
             "tpr": "/path/to/npt_1M_napf6_dme.tpr"},
},
"dt_fs": {"OPLS": 100.0},   # 0.1 ps = 100 fs per frame
```

**GROMACS example — string path + `tpr_paths` key:**

```python
"traj_paths": {"OPLS": "/path/to/npt_1M_napf6_dme.xtc"},
"tpr_paths":  {"OPLS": "/path/to/npt_1M_napf6_dme.tpr"},
"dt_fs":      {"OPLS": 100.0},
```

**GROMACS example — sibling `.tpr` auto-discovery (XTC and TPR share the same stem):**

```python
"traj_paths": {"OPLS": "/path/to/npt_1M_napf6_dme.xtc"},  # .tpr must sit alongside
"dt_fs":      {"OPLS": 100.0},
```

> **Note on `dt_fs` for GROMACS:** the OPLS baseline trajectories use `dt = 0.1 ps = 100 fs`
> per frame. The frame-counting step uses `n_frames × dt_fs × 1e-6` to derive `analysis_ns`,
> consistent with ASE trajectories.

---

```python
SYSTEMS = [
    {
        "name": "NaPF6/DME 0.1M",
        "traj_paths": {
            "UMA":     "/path/to/uma.traj",
            "student": "/path/to/student.traj",
        },

        # ── shared ──────────────────────────────────────────────────────────
        "dt_fs": {"UMA": 10.0, "student": 100.0},  # fs; or single float
        "model_colors": {"UMA": "#1f77b4", "student": "#ff7f0e"},

        # ── RDF ─────────────────────────────────────────────────────────────
        "rdf_pairs":      [("Na", "O"), ("Na", "F")],
        "skip_ns":        0.1,    # equilibration skip for RDF/density windows
        "window_ns":      0.9,    # analysis window width
        "n_frames":       1000,   # frames sampled per window
        "sliding_window": False,  # enable sliding-window RDF

        # ── density ──────────────────────────────────────────────────────────
        "density_roll_window_ns": 0.5,  # smoothing window for timeseries plot

        # ── MSD / diffusivity ────────────────────────────────────────────────
        "cat_symbol":      "Na",   # key in cation_dict   (msd_with_com utils)
        "anion_symbol":    "PF6",  # key in anion_dict
        "solvent_symbol":  "DME",  # key in solvent_dict
        "eq_cut_ns":       0.0,    # skip from trajectory start before computing MSD
        "fit_pct":         0.8,    # fit up to this fraction of max available lag
        "tau_min_fit_ns":  1.0,    # lower bound of linear fit (skips ballistic/cage)
        "slide_window_ns": 10.0,   # Panel 4: fixed sliding-window width (ns)
        "slide_step_ns":   0.1,    # Panel 4: sliding step (ns)
        "n_conv_points":   200,    # resolution of the convergence D vs tau_max sweep

        # ── cell size ────────────────────────────────────────────────────────────
        # Tracks a, b, c cell lengths (Å) and box volume (Å³) along the trajectory.
        # cell_size_analyze_dt_ps: evaluate one frame every N ps (preferred over n_frames).
        # When omitted, n_frames frames are sampled uniformly (default: 2×n_frames).
        "cell_size_analyze_dt_ps": 100.0,   # optional; e.g. one frame per 100 ps
        # "cell_size_n_frames": 2000,        # fallback when analyze_dt_ps is not set
    },
]

OUTPUT_DIR = "/global/homes/y/yuejian/project/MLFF-distill/yuejian/electrolyte_application/analysis/20ns"
ANALYSES   = ["rdf", "density", "energy", "msd", "cell_size"]  # any subset
WORKERS    = 4   # or None → one worker per system
```

### Parallelism

- **System-wise only**: each system is one subprocess.
- Models and analyses within a system run sequentially.
- Set `WORKERS` in the config or pass `--workers N` on the CLI.
- `--output-dir` on the CLI overrides `OUTPUT_DIR` in the config.

### Output layout

You provide a root directory (`OUTPUT_DIR` in the config or `--output-dir` on the CLI). Each run automatically creates a timestamped subdirectory named `YYYYMMDD_HHMMSS_<analyses>` inside that root, so multiple runs never overwrite each other. Each system then gets its own subdirectory under the run dir.

```
OUTPUT_DIR/                                        ← e.g. .../analysis/20ns/
└── 20250508_143022_rdf_density_energy_msd/        ← auto-created: timestamp + analyses
    ├── NaPF6_DME_0.1M/                            ← one dir per system
    │   ├── rdf_NaPF6_DME_0.1M.png                ← g(r)/n(r) comparison across models
    │   ├── sliding_gr_*.png                       ← (if sliding_window=True)
    │   ├── sliding_nr_*.png
    │   ├── rdf_sliding_*.npz                      ← raw sliding-window arrays
    │   ├── density.csv                            ← mean ± std per model
    │   ├── density_timeseries_*.png               ← instant density + rolling mean
    │   ├── energy_{sys}_{model}.png               ← PE/KE/E_tot/T vs time
    │   ├── energy_{model}.npz                     ← raw energy arrays
    │   ├── msd_{sys}_{model}.pkl                  ← raw MSD arrays
    │   ├── msd_{sys}_{model}.png                  ← MSD curves with fit overlay
    │   ├── convergence_{sys}_{model}.png          ← 4-panel diagnostic figure
    │   ├── diffusivity.csv                        ← final D for all models
    │   ├── cell_size_{sys}.png                    ← 4-panel a/b/c/Volume vs time
    │   └── cell_size_{model}.npz                  ← raw cell arrays (times_ns, a, b, c, volume)
    ├── NaOTf_DME_1M/
    │   └── ...
    └── LiPF6_DME_0.5M/
        └── ...
```

### MSD convergence figure panels

| Panel | What it shows | Good sign |
|---|---|---|
| 1 (top-left) | MSD vs lag time with linear fit | Fit line follows MSD in Einstein regime |
| 2 (top-right) | D vs tau_max (fit grows rightward) | D plateaus quickly |
| 3 (bot-left) | ΔD per step (derivative of Panel 2) | Flat near zero in plateau |
| 4 (bot-right) | Sliding window D (fixed-width window slides) | Flat across all window positions |

---

## Extending the Toolbox

To add a new observable:
1. Create `new_obs/compute.py` with pure computation functions.
2. Create `new_obs/plot.py` with plotting functions that return a `Path`.
3. In `eval.py/_run_system()`, add `if "new_obs" in analyses:` block, import locally, call compute + plot + save.
4. Add the key to `ANALYSES` in user configs.
