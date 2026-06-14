# Ionic Conductivity via the Onsager / Green-Kubo Formalism

## Overview

The goal is to compute the ionic conductivity $\kappa$ of an electrolyte from a molecular dynamics trajectory. The route is:

$$\text{atom velocities} \;\longrightarrow\; \text{velocity ACF} \;\longrightarrow\; L^{ij} \;\longrightarrow\; \kappa$$

---

## Step 1 — Collect ion velocities

For each ionic species $i$, sum the velocities of all $N_i$ individual ions at every time step:

$$J_i(t) = \sum_{\alpha=1}^{N_i} \mathbf{v}_{i,\alpha}(t)$$

This gives a 3-component vector time series for each species (cation $+$, anion $-$).

> **Why sum, not average?** The Onsager coefficient is an *extensive* quantity proportional to $N$, so summing avoids multiplying $N$ back in later.

---

## Step 2 — Compute velocity cross-correlation functions (ACF)

For each pair of species $(i, j)$ and each spatial direction, compute the time cross-correlation:

$$C^{ij}(t) = \langle \mathbf{J}_i(t) \cdot \mathbf{J}_j(0) \rangle$$

In practice this is estimated from the finite trajectory using the **FFT convolution trick** (Wiener–Khinchin theorem) for efficiency:

$$C^{ij} = \mathcal{F}^{-1}\!\left[\mathcal{F}[\mathbf{J}_i] \cdot \mathcal{F}[\mathbf{J}_j]^{*}\right]$$

Zero-padding to the next power of 2 avoids circular wrap-around artifacts. The result is normalised by the number of independent time-origin pairs at each lag $t$.

For a binary electrolyte there are three distinct pairs:

| ACF | Meaning |
|-----|---------|
| $C^{++}(t)$ | cation–cation velocity correlation |
| $C^{+-}(t)$ | cation–anion cross-correlation |
| $C^{--}(t)$ | anion–anion velocity correlation |

---

## Step 3 — Integrate to get Onsager transport coefficients $L^{ij}$

The **Green-Kubo relation** connects the time integral of the velocity ACF to the Onsager phenomenological coefficient:

$$L^{ij} = \frac{1}{3 k_B T V} \int_0^{\infty} C^{ij}(t)\, dt$$

where:
- $k_B = 1.38 \times 10^{-23}\ \text{J K}^{-1}$ — Boltzmann constant
- $T$ — temperature in Kelvin
- $V$ — simulation box volume

The factor $\frac{1}{3}$ comes from averaging over the three independent spatial directions $(x, y, z)$.

**Units:** velocities in $\text{Å/fs}$, time in $\text{fs}$, volume in $\text{Å}^3$. A conversion factor of $(\text{Å} \to \text{m})^{-1} \times (\text{fs} \to \text{s})^{-1}$ brings $L^{ij}$ to SI units of $(\text{J\,s\,m})^{-1}$.

In code, the integral is accumulated with the trapezoid rule (`cumtrapz`), producing $L^{ij}(t)$ as a running integral — a plateau indicates convergence.

**Physical meaning of $L^{ij}$:**

| Coefficient | Physical interpretation |
|-------------|------------------------|
| $L^{++}$ | cation self-transport (correlated cation motion) |
| $L^{--}$ | anion self-transport |
| $L^{+-}$ | cross-transport (cation–anion coupling; usually negative) |

---

## Step 4 — Compute ionic conductivity $\kappa$

Once $L^{ij}$ has converged, conductivity follows from the **Onsager formula**:

$$\kappa = F^2 \sum_i \sum_j z_i\, z_j\, L^{ij}$$

where:
- $F = 96\,485\ \text{C\,mol}^{-1}$ — Faraday's constant
- $z_i$ — charge number of species $i$ (e.g. $+1$ for Li$^+$, $-1$ for Cl$^-$)

For a 1:1 electrolyte ($z_+ = +1$, $z_- = -1$):

$$\kappa = F^2 \!\left( L^{++} + L^{--} - 2\,L^{+-} \right)$$

The $-2L^{+-}$ term arises because $z_+ z_- = -1$, which *adds* to conductivity when cations and anions move in opposite directions (as expected physically).

**Unit conversion:**

$$\kappa\ [\text{mS/cm}] = \kappa\ [\text{S/m}] \times 10$$

In code (using per-ion velocities, so $F \to e$):

```python
e = 1.60217662e-19  # elementary charge in Coulombs
conductivity = (L_plusplus + L_minusminus - 2*L_plusminus) * 10 * e**2  # mS/cm
```

---

## Full pipeline summary

| Step | Operation | Key equation |
|------|-----------|-------------|
| 1 | Sum atom velocities per species | $J_i(t) = \sum_\alpha \mathbf{v}_{i,\alpha}(t)$ |
| 2 | Cross-correlate via FFT | $C^{ij}(t) = \langle \mathbf{J}_i(t)\cdot\mathbf{J}_j(0)\rangle$ |
| 3 | Integrate (Green-Kubo) | $L^{ij} = \dfrac{1}{3k_BTV}\displaystyle\int_0^\infty C^{ij}(t)\,dt$ |
| 4 | Onsager conductivity | $\kappa = e^2\!\left(L^{++} + L^{--} - 2L^{+-}\right)$ |

---

## Key physical insights

**$L^{+-} < 0$ increases conductivity.** When cations and anions anti-correlate (move in opposite directions), conductivity is enhanced. A large $|L^{+-}|$ relative to $L^{++}$ and $L^{--}$ signals good ion dissociation.

**Ionicity** $= \kappa_\text{Onsager} / \kappa_\text{NE}$. The Nernst-Einstein conductivity $\kappa_\text{NE}$ ignores cross-correlations (sets $L^{+-} = 0$). The ratio quantifies ion-pairing: values near 1 mean fully dissociated ions, values near 0 mean strongly paired.

**Convergence check.** Plot $L^{ij}(t)$ vs integration time — it should reach a stable plateau well before the end of the trajectory. If it does not, the simulation is too short.
