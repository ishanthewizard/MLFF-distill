# Onsager / Nernst–Einstein conductivity — math + code walkthrough

A line-by-line companion to
`submodule/byteff2/byteff2/md_utils/onsager_conductivity.py`.

The file turns an **unwrapped MD trajectory** into:

| output key | symbol | meaning |
|---|---|---|
| `conductivity_onsager` | $\sigma$ | full ionic conductivity (incl. ion–ion correlations) |
| `conductivity_NE` | $\sigma_{\text{NE}}$ | Nernst–Einstein conductivity (self-diffusion only) |
| `Lambda_onsager` | $\Lambda_{ij}$ | full Onsager coefficient matrix |
| `Dself_inf` | $D_i^{\infty}$ | self-diffusivities, finite-size corrected |

The whole pipeline is two physical formulas evaluated from the trajectory:

$$
\boxed{\;\sigma=\frac{e^2}{6\,V k_B T}\lim_{t\to\infty}\frac{d}{dt}\Big\langle\Big|\sum_i z_i\,\Delta R_i(t)\Big|^2\Big\rangle\;}
\qquad\text{(Einstein–Helfand, full Onsager)}
$$

$$
\boxed{\;\sigma_{\text{NE}}=\frac{e^2}{V k_B T}\sum_i z_i^2\,N_i\,D_i\;}
\qquad\text{(Nernst–Einstein)}
$$

where $R_i=\sum_{k\in i} r_k$ is the **collective** (summed) center-of-mass coordinate of
species $i$, $z_i$ its charge, $N_i$ its molecule count, $V$ the box volume, and $D_i$ the
self-diffusivity. Everything else in the file is machinery to estimate the two limits
$\frac{d}{dt}\langle\dots\rangle$ robustly and to fix finite-size / center-of-mass artifacts.

> **Internal time unit.** The code treats **one frame = 1 ps** everywhere (see `dts_ps`,
> the slope fits, and `OnsagerUnit.ps = 1.0`). If your real frame spacing is not 1 ps, the
> caller (`conductivity/compute.py`) rescales the outputs by `dt_correction = 1/load_dt_ps`.

---

## 0. The Einstein relation we are estimating everywhere

For a diffusing particle in 3D,

$$
\text{MSD}(\tau)=\big\langle |r(t+\tau)-r(t)|^2\big\rangle \;\xrightarrow{\ \tau\ \text{large}\ }\; 6D\,\tau .
$$

So a **diffusivity is the slope of an MSD divided by 6**. The two transport quantities differ
only in *which* displacement goes into the MSD:

- **self**: single-molecule displacement $\Delta r_k$ → self-diffusivity $D_i$.
- **collective / cross**: summed displacement $\Delta R_i = \sum_k \Delta r_k$ correlated
  against $\Delta R_j$ → Onsager coefficient $\Lambda_{ij}$.

Both reduce to: *build a displacement time series, form an MSD-vs-lag curve, fit a line, take the slope.*

---

## 1. `correlate_xy(in1, in2)` — the MSD/cross-MSD engine (FFT)

Computes, for every lag $m=0,\dots,N-1$:

$$
\text{ans}[m]=\frac{1}{N-m}\sum_{t=0}^{N-1-m}\big(x_{t+m}-x_t\big)\cdot\big(y_{t+m}-y_t\big).
$$

With `in1 is in2` this is the ordinary $\text{MSD}(m)$; with two different series it is the
**cross**-displacement correlation needed for $\Lambda_{ij}$.

### The trick (why FFT)

A naïve double loop is $O(N^2)$. Expand the integrand:

$$
(x_{t+m}-x_t)(y_{t+m}-y_t)=\underbrace{x_t y_t + x_{t+m}y_{t+m}}_{\text{"diagonal" }f_1[m]}
-\underbrace{x_{t+m}y_t}_{c_{12}[m]}-\underbrace{x_t y_{t+m}}_{c_{21}[m]} .
$$

- $f_1[m]=\sum_{t=0}^{N-1-m}(D_t+D_{t+m})$ with $D_t=x_t\!\cdot\!y_t$ — done in $O(N)$ with prefix/suffix
  cumulative sums (`cm1`, `cm2`). Note $f_1[0]=2\sum_t D_t$ and
  $f_1[m]=2\sum_t D_t-\sum_{t<m}D_t-\sum_{t\ge N-m}D_t$.
- $c_{12},c_{21}$ are cross-correlations, computed by the **correlation theorem** in $O(N\log N)$:
  $\text{ifft}(X\,\overline{Y})[m]=\sum_t x_{t+m}y_t$ and $\text{ifft}(\overline{X}\,Y)[m]=\sum_t x_t y_{t+m}$.
  Zero-padding to a power of two $\ge 2N-1$ (`n=2**(N*2-1).bit_length()`) prevents circular wraparound.
- `div = [N, N-1, …, 1]` is the number of valid time origins $(N-m)$ per lag — the average's denominator.

### Reproduce it in plain NumPy

```python
import numpy as np

def correlate_xy_bruteforce(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    N = len(x)
    out = np.empty(N)
    for m in range(N):
        out[m] = np.mean((x[m:] - x[:N-m]) * (y[m:] - y[:N-m]))
    return out

# matches correlate_xy() to machine precision in float64 (verified ~1e-15)
```

The shipped version is the FFT form of exactly this. (When `in1`/`in2` are 2D, the dot `·`
runs over the trailing feature dimension — but in `onsager_calc` it is always called per axis,
so that path is unused.)

---

## 2. `polyfit(x, y, deg)` — least-squares line fit

Builds the Vandermonde matrix $X=[1,\,x,\,x^2,\dots]$ and solves $X\beta=y$ in the least-squares
sense (`torch.linalg.lstsq`). Used with `deg=1` to get the **slope** of MSD vs lag over the fit
window. Only the slope (`kslope`, the last coefficient) is kept; the intercept is discarded.

$$
\text{MSD}(\tau)\approx \text{intercept} + k\,\tau,\qquad k=\text{kslope}.
$$

---

## 3. `OnsagerUnit` — a self-consistent internal unit system

Rather than carry SI constants around, the file defines a unit system anchored at
**ps, nm, amu, K, A** and derives everything else:

```python
ps = 1.0;  s = 1e9 * 1e3 * ps          # 1 s  = 1e12 ps
nm = 1.0;  angstrom = 0.1; m = 1e9     # 1 m  = 1e9 nm
amu = 1.0; mol = 6.022e23; kg = 1e3*mol*amu
J  = kg*(m/s)**2                       # energy from mass·(length/time)²
e  = 1.602e-19 * (s*A)                 # elementary charge in (ps·A)
kB = 1.380649e-23 * J                  # Boltzmann constant
Siemens = (s*A)*A / J
pbc_xi  = 2.837297                     # Yeh–Hummer lattice constant ξ
```

The only thing you need from this section: conversions like
`(angstrom**2/ps)/(1e-10*m**2/s)` evaluate to a pure number. In particular

$$
1\ \frac{\text{Å}^2}{\text{ps}} = 100\times 10^{-10}\ \frac{\text{m}^2}{\text{s}},
$$

which is the factor multiplying every raw MSD slope below. (Check:
`(0.1**2/1)/(1e-10*(1e9)**2/1e12) == 100`.)

---

## 4. `remove_center_of_mass_error(L, masses)` — COM-constraint fix (Eq A.6)

The collective coordinates $R_i$ are referenced to the **system** center of mass, so their
fluxes are not independent: the mass-weighted combination $\sum_i m_i R_i$ is (proportional to)
the system COM, which must carry **zero** transport. The raw $\Lambda$ leaks a spurious COM
contribution into every entry. The fix is a similarity projection that removes the mass mode:

$$
\Lambda' = P\,\Lambda\,P^{\top},\qquad P = I-\frac{\mathbf 1\,\mathbf m^{\top}}{M},\quad M=\textstyle\sum_i m_i,
$$

which expands exactly to the code:

$$
\Lambda'_{ij}=\Lambda_{ij}-\frac{(\Lambda m)_i}{M}-\frac{(\Lambda m)_j}{M}+\frac{m^{\top}\Lambda m}{M^2}.
$$

```python
def remove_center_of_mass_error(L, m):
    M   = m.sum()
    Lm  = L @ m
    MLM = (m @ Lm) / M**2          # scalar  mᵀLm / M²
    Lm  = Lm / M                   # vector  (Lm)/M
    return L - Lm[:,None] - Lm[None,:] + MLM
```

Here `masses` = per-species **molecular** masses (e.g. `[23, 100, 50]`), matching the unweighted
sum $R_i=\sum_k r_k$. After this, $\Lambda' m = 0$: the COM mode is annihilated.

---

## 5. From slopes to conductivity

### 5a. `Lambda_to_ionic_conductivity` — full Onsager σ (Eq 3.1)

$$
\sigma=\frac{e^2\,N}{V k_B T}\;\mathbf z^{\top}\Lambda_{\text{SI}}\,\mathbf z,
\qquad \Lambda_{\text{SI}}=\Lambda\times 10^{-10}\,\tfrac{\text{m}^2}{\text{s}} .
$$

```python
Lambda_si = Lambda * (1e-10 * m**2/s)
L_si      = Lambda_si * N / (V*angstrom**3) / (kB*T*K)   # N = total #molecules
sigma     = e**2 * z @ L_si @ z / (1e-3 * Siemens/cm)    # express in mS/cm
```

The factor `N` reappears because $\Lambda$ was defined *per particle* (divided by $N$ when
built). Folding it back gives $N\,\mathbf z^\top\Lambda\,\mathbf z=\tfrac1{6}\frac{d}{dt}\langle|\sum_i z_i\Delta R_i|^2\rangle$,
i.e. the **charge-displacement MSD** — the Einstein–Helfand conductivity from the box intro.

### 5b. `Dself_to_ionic_conductivity` — Nernst–Einstein σ (Eq 3.2)

Drops all $i\neq j$ correlations and the distinct same-species terms; keeps only self-diffusion:

$$
\sigma_{\text{NE}}=\frac{e^2}{V k_B T}\sum_i z_i^2\,N_i\,D_i .
$$

```python
sigma_NE = e**2 * sum(z**2 * Dself * counts) * (1e-10*m**2/s) / (kB*T*K * V*angstrom**3)
sigma_NE /= (1e-3 * Siemens/cm)
```

The **ratio** $\sigma/\sigma_{\text{NE}}$ is the *ionicity* / inverse Haven ratio: it equals 1 only
when ions move independently; deviations measure ion–ion correlation (ion pairing, etc.).

### 5c. `fsc_self_diffusivity` — Yeh–Hummer finite-size correction (Eq 4.6)

Periodic boundaries systematically *lower* self-diffusivities. The leading correction is

$$
\Delta D_{\text{YH}}=\frac{k_B T\,\xi}{6\pi\,\eta\,L},\qquad \xi=2.837297,
$$

with $\eta$ the viscosity and $L$ the (cubic-equivalent) box edge. It is a **single scalar added
to every species'** self-diffusivity to extrapolate to infinite box size:

```python
DYH1     = fsc_self_diffusivity(T, viscosity_cP, BoxLen)   # one number
DselfInf = Dself_MD + DYH1
```

Because $\Delta D_{\text{YH}}$ depends on $\eta$, and $\eta$ is usually a placeholder (`1.0 cP`)
in this pipeline, treat `Dself_inf` as approximate. It does **not** affect the conductivities.

---

## 6. `onsager_calc(...)` — the driver, part by part

### 6.1 Species bookkeeping (lines 405–439)

Inputs are dicts keyed by species. The code flattens them into tensors and records where each
species' atoms live along the atom axis of `positions`.

```python
# per species i:
Masses[i]        = sum(species_mass[sp])          # molecular mass        m_i
AtomMFrac[i]     = [a/Masses[i] for a in masses]  # atom mass fractions   w_a = m_a/m_i  (Σ w_a = 1)
SpeciesCounts[i] = species_number[sp]             # molecule count        N_i
Charges[i]       = species_charge[sp]             # molecular charge      z_i
AtomRanges[i]    = [start, end)                   # slice in the flat atom axis
```

`AtomMFrac` are the weights that turn atom positions into a **molecular center of mass**:
$r^{\text{COM}} = \sum_a w_a\,r_a$ with $w_a=m_a/m_i$. The flat `AtomMasses` (every atom of the
whole system) is used only for the **system** COM.

> ⚠️ `positions` **must** be ordered species-by-species, molecule-by-molecule, consistent with
> `species_order`. The caller (`compute.py`) builds a `reorder_idx` to guarantee this.

### 6.2 Equilibration cut and the system COM (lines 441–454)

```python
np_xyz = positions[200:]            # drop 200 frames as extra equilibration
nt_start, nt_end = 50, 200          # fit MSD slope over lags [50, 200) "ps"
origx = (xu * AtomMasses).sum(axis=atoms) / TotalAtomMass    # system COM_x(t), per frame
```

$$
R^{\text{sys}}_\alpha(t)=\frac{1}{M_{\text{tot}}}\sum_{a} m_a\,r_{a,\alpha}(t),\quad \alpha\in\{x,y,z\}.
$$

Subtracting it removes global drift (e.g. from a thermostat) before measuring diffusion.

### 6.3 Self-MSD → self-diffusivity (lines 461–485)

For each species, reshape its atom block to `(frames, molecules, atoms_per_molecule)`, form each
molecule's COM relative to the system COM, then sum single-molecule MSDs:

$$
r^{(k)}_\alpha(t)=\sum_a w_a\,r^{(k)}_{a,\alpha}(t)-R^{\text{sys}}_\alpha(t),
\qquad
\text{msd1r}(m)=\sum_{k=1}^{N_i}\sum_{\alpha}\text{MSD}\big(r^{(k)}_\alpha\big)(m).
$$

```python
cmsx1 = einsum("hij,j->hi", x1, AtomMFrac[i]) - origx[:,None]   # (frames, molecules)
for j in range(N_i):                                            # NO cross-molecule terms
    msd1x += correlate_xy(cmsx1[:,j], cmsx1[:,j])
msd1r = msd1x + msd1y + msd1z
kmsd_i[i] = polyfit(dts_ps, msd1r[50:200], 1).slope            # Å²/ps, summed over molecules
```

Since `msd1r` is summed over the $N_i$ molecules, $k^{\text{self}}_i=N_i\cdot 6D_i$, hence the later
division by $6N_i$.

### 6.4 Collective coordinates → Onsager matrix (lines 487–509)

The **same** per-molecule COMs are now *summed* over molecules into one collective coordinate per
species, then cross-correlated between species:

$$
R_{i,\alpha}(t)=\sum_{k=1}^{N_i}\Big(\sum_a w_a r^{(k)}_{a,\alpha}\Big)-N_i R^{\text{sys}}_\alpha,
\qquad
\text{msd2r}_{ij}(m)=\sum_\alpha \big\langle\Delta R_{i,\alpha}\,\Delta R_{j,\alpha}\big\rangle(m).
$$

```python
Rxt[i] = einsum("hij,j->h", x1, AtomMFrac[i]) - N_i*origx       # collective coord, (frames,)
for i in range(nsp):
    for j in range(i, nsp):                                     # symmetric, upper triangle
        msd2r = correlate_xy(Rxt[i],Rxt[j]) + (y) + (z)
        kmsd_xy[i,j] = polyfit(dts_ps, msd2r[50:200], 1).slope
kmsd_xy[j,i] = kmsd_xy[i,j]                                     # mirror
```

**Key contrast** (why $\Lambda_{ii}\neq$ self-diffusivity): expanding the collective MSD,

$$
\frac{d}{dt}\langle|\Delta R_i|^2\rangle
=\underbrace{\sum_k \tfrac{d}{dt}\langle|\Delta r_k|^2\rangle}_{\text{self }=\,k^{\text{self}}_i}
+\underbrace{\sum_{k\neq l}\tfrac{d}{dt}\langle\Delta r_k\!\cdot\!\Delta r_l\rangle}_{\text{distinct, same-species}} .
$$

So $\Lambda_{ii}$ contains self **plus** distinct correlations and is normalized by $N_{\text{tot}}$
(not $N_i$); it cannot be reduced to $D_i$. `Dself` is therefore measured separately in 6.3.

### 6.5 Unit conversion via Einstein relation (lines 516–521)

```python
Dself_MD  = kmsd_i  * 100 / (6 * SpeciesCounts)        # 100 = (Å²/ps)->(1e-10 m²/s)
RawLambda = kmsd_xy * 100 / (6 * SpeciesCountsTotal)
```

$$
D_i=\frac{k^{\text{self}}_i}{6N_i}\cdot100,\qquad
\Lambda_{ij}=\frac{k^{\text{xy}}_{ij}}{6N_{\text{tot}}}\cdot100 \quad[10^{-10}\,\text{m}^2/\text{s}].
$$

### 6.6 COM fix, conductivities, finite-size (lines 522–546)

```python
Lambda__md   = remove_center_of_mass_error(RawLambda, Masses)        # §4
sigma__o__md = Lambda_to_ionic_conductivity(Lambda__md, Charges, N_tot, T, V)  # §5a
sigma_ne__md = Dself_to_ionic_conductivity(Dself_MD, Charges, N_i,   T, V)     # §5b
DselfInf     = Dself_MD + fsc_self_diffusivity(T, Viscosity, BoxLen)           # §5c
```

These populate the return dict: `conductivity_onsager`, `conductivity_NE`, `Lambda_onsager`
(+ `_raw`, `_unit`), `species_order`, `Dself_inf`.

---

## 7. End-to-end sanity check (independent walkers → σ ≈ σ_NE)

For non-interacting Brownian particles, all distinct/cross correlations vanish, so the full
Onsager conductivity must collapse onto Nernst–Einstein:

```python
import numpy as np, sys
sys.path.insert(0, ".../submodule/byteff2")
from byteff2.md_utils.onsager_conductivity import onsager_calc

np.random.seed(3); nf = 8000
order = ["cat","ani","sol"]
mass  = {"cat":[23.],"ani":[100.],"sol":[50.]}
num   = {"cat":60,"ani":60,"sol":180}
chg   = {"cat":1,"ani":-1,"sol":0}
pos   = np.cumsum(np.random.randn(nf, sum(num.values()), 3)*0.1, axis=0)  # independent walks

out = onsager_calc(order, mass, num, chg, volume_angstrom3=3e4,
                   viscosity_cP=1e12, T_K=298.0, positions=pos)
print(out["conductivity_onsager"] / out["conductivity_NE"])   # ~1.0  (finite-N noise)
```

Observed ratio ≈ 1.09 at this trajectory length — the residual is statistical noise in the
cross-correlation estimate, and shrinks with more frames/particles. For real electrolytes the
ratio departs from 1 and *that departure is the physics* (ion correlation / pairing).

---

## 8. Auxiliary (not on the conductivity path)

These implement the **Maxwell–Stefan / Fick** transport picture and a matrix-level finite-size
correction. They are present and tested but **commented out** inside `onsager_calc`
(lines 528–531), so they do not affect `conductivity_onsager`/`_NE`:

| function | eq tag | role |
|---|---|---|
| `build_Delta_index`, `build_B_index` | — | index maps for the $(n{-}1)$-dim reduced matrices |
| `X_matrices` | 2.5 | linear map $\Delta = X\,\Lambda$ between Onsager and Δ representations |
| `Delta_matrices` | 2.2, 2.6 | Δ matrix and its inverse from $\Lambda$ and mole fractions |
| `MS_matrix` | 2.6 | Maxwell–Stefan diffusivities $Đ_{ij}$ from $B=X(1/Đ)$ |
| `fsc_full_Lambda` | 4.3, 4.4 | full-matrix Yeh–Hummer correction (vs the scalar §5c) |

To use them you would also need mole fractions `XFrac = SpeciesCounts/SpeciesCountsTotal`
(defined but commented out at line 432) and to uncomment the block in `onsager_calc`.

> The `# MyNoteEq: X.Y` tags throughout the source reference equations in the upstream byteff2
> derivation note (transport-coefficient methods); section numbers above mirror those tags.

---

## TL;DR mental model

1. **Displacements → MSD curves** via `correlate_xy` (FFT, $O(N\log N)$).
2. **Slope ÷ 6** (Einstein) → diffusivities; per-molecule gives $D_i$, collective gives $\Lambda_{ij}$.
3. **Fix artifacts**: remove system-COM mode from $\Lambda$ (§4); add Yeh–Hummer to $D_i$ (§5c).
4. **Contract with charges**: $\mathbf z^\top\Lambda\mathbf z$ → full $\sigma$; self-only → $\sigma_{\text{NE}}$.
5. **$\sigma/\sigma_{\text{NE}}$** is the ion-correlation (ionicity) factor.
