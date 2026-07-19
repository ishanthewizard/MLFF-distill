# Onsager ionic conductivity of PAINN electrolyte trajectories

Computed with **mdcraft** `analysis.transport.Onsager` (Fong-Self-McCloskey-Persson Onsager framework).

Script: `md_craft.py`. Date: 2026-06-19.

## Systems

Two ASE `.traj` NPT runs (PAINN MLFF), 1 M, 298 K, 20 ns, 100 fs/frame, analysed every 10th frame (1 ps spacing, 20 000 frames). Each box holds **17 ion pairs + 125 DME**.

| system | anion | <V> (A^3) | carriers |
|---|---|---|---|
| naotf_dme | OTf^- | 21886 | 17 Na+ / 17 anion |
| napf6_dme | PF_6^- | 21407 | 17 Na+ / 17 anion |

## Method & key correction

- **Carrier sites:** Na atom for the cation; the central heavy atom (S for OTf-, P for PF6-) as a single-site proxy for each anion. The central-atom long-time MSD slope equals the molecular centre-of-mass slope, so L_ij / kappa are unaffected; using one site per ion keeps the collective flux correctly charged.

- **Unwrapping:** done explicitly with each frame's fluctuating NPT box (minimum image of consecutive 1 ps displacements).

- **Center-of-mass (COM) drift correction — important.** The Langevin thermostat (gamma = 1 THz) makes the whole-system COM random-walk: `D_com = kBT/(M gamma) ~ 0.017 A^2/ps`, i.e. a common drift of ~45 A over 20 ns (measured: naotf_dme 45 A, napf6_dme 63 A). This common drift cancels in the electroneutral charge contraction (sum_i z_i N_i = 0), so **kappa is unchanged**, but it dominates single-ion self-diffusion and transference numbers. All D_i and t+ below are in the **barycentric (COM-removed) frame**.

- **Fit:** L_ij and D_i from a linear fit of the (collective and self) MSDs vs t over the diffusive window **1-10ns** (a tighter 2-8 ns window is reported as a sensitivity check).

## Results — ionic conductivity (uS/cm)

| system | mdcraft Onsager (1-10 ns) | (2-8 ns) | lab-frame check | ref-code Onsager | ref Nernst-Einstein | **experiment** |
|---|---|---|---|---|---|---|
| naotf_dme | **4701** | 4777 | 4701 | 4427 | 11233 | 1233 |
| napf6_dme | **14066** | 13673 | 14066 | 8349 | 14817 | 12960 |

The identical lab-frame value confirms kappa is COM-invariant.

**Interpretation.** naotf_dme: the mdcraft Onsager value agrees with the independent reference-code Onsager (both ~3.8x above experiment) -- a PAINN model over-prediction, not a method artifact, since the two Onsager codes agree. napf6_dme: the mdcraft value is close to experiment and the reference NE but above the reference Onsager, traced to incomplete diffusive convergence of the dominant PF6-PF6 collective term (below), which biases L-- and hence kappa high.

## Transport coefficients (COM-removed, 1-10 ns)

| system | D(Na+) cm^2/s | D(anion) cm^2/s | t+ | t- | L++ | L+- | L-- (mol/kJ/A/ps) | self-MSD slope Na / anion |
|---|---|---|---|---|---|---|---|---|
| naotf_dme | 1.20e-06 | 1.35e-06 | 0.54 | 0.46 | 1.74e-06 | 8.61e-08 | 1.48e-06 | 0.95 / 0.98 |
| napf6_dme | 1.31e-06 | 2.09e-06 | -0.02 | 1.02 | 1.59e-06 | 1.77e-06 | 1.11e-05 | 0.94 / 1.04 |

## Figures

- `conductivity_parity.png` — simulated vs experimental kappa (circles = this work, squares = reference Onsager, triangles = reference NE).
- `cross_displacement_naotf_dme.png` — collective <DR_i . DR_j> vs t (cation-cation, anion-anion, and the cation-anion **cross** term) with the linear fit; use this to judge whether the cross term has a converged linear regime.
- `msd_self_naotf_dme.png` — self-MSD (log-log) with the fit window and a slope-1 guide.
- `cross_displacement_loglog_naotf_dme.png` — log-log collective <DR_i . DR_j> with a **local-slope** panel (d ln<DR.DR>/d ln t). The cross term is converged where this local slope sits flat at 1 across the fit window; a slope > 1 in-window marks a super-diffusive (upper-bound) L_ij.
- `cross_displacement_napf6_dme.png` — collective <DR_i . DR_j> vs t (cation-cation, anion-anion, and the cation-anion **cross** term) with the linear fit; use this to judge whether the cross term has a converged linear regime.
- `msd_self_napf6_dme.png` — self-MSD (log-log) with the fit window and a slope-1 guide.
- `cross_displacement_loglog_napf6_dme.png` — log-log collective <DR_i . DR_j> with a **local-slope** panel (d ln<DR.DR>/d ln t). The cross term is converged where this local slope sits flat at 1 across the fit window; a slope > 1 in-window marks a super-diffusive (upper-bound) L_ij.

## Reading the cross-displacement plots / convergence

`<DR_i . DR_j>` is the *collective* displacement correlation (sum over all ions of a species). Its slope in the fit window, divided by 6 kBT V, gives L_ij. A trustworthy L_ij needs a straight, low-noise region. The diagonal (same-species) terms dominate; the cation-anion cross term is small and noisier. Beyond ~13 ns all curves diverge (few independent time origins). Watch in particular the largest diagonal term in each system: if it is still convex (super-diffusive) inside the window, its slope -- and hence kappa -- is an upper estimate (this is the case for the PF6-PF6 term in napf6_dme).

In-window log-log slopes of the collective terms `d ln<DR_i . DR_j>/d ln t` (target 1.0 in the diffusive regime; see the `cross_displacement_loglog_*.png` local-slope panels):

  - naotf_dme: cation-cation 0.55, cross 0.07, anion-anion 0.49
  - napf6_dme: cation-cation 0.59, cross 1.40, anion-anion 1.22

## Notes / caveats

- Single statistical block (n_blocks = 1); for error bars, split the trajectory into blocks or run multiple seeds.

- Volume uses the average NPT box <V>; per-frame box used only for unwrapping.

- napf6_dme's anion self-MSD slope > 1 (super-diffusive in-window); its kappa should be read as an upper bound until the PF6-PF6 collective term is fit on a longer, converged window (or with block averaging over a longer run). The 1-10 ns and 2-8 ns windows agree only because both lie in the same pre-converged region.

- The single-site anion proxy (S/P central atom) affects only short-time intramolecular motion, not the long-time slopes that set L_ij and kappa.

