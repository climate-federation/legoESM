# Silvestri et al. (2024) -- WENO-ILES Momentum Advection: Audit & Implementation Plan

**Target paper:** Silvestri, S., Wagner, G. L., Campin, J.-M., Constantinou, N. C.,
Hill, C. N., Souza, A., & Ferrari, R. (2024). A new WENO-based momentum advection
scheme for simulations of ocean mesoscale turbulence. *Journal of Advances in
Modeling Earth Systems*, 16, e2023MS004130.
DOI: 10.1029/2023MS004130

**Local copy:** `docs/references/weno-ILES.pdf`

**Reference implementation:** Oceananigans.jl v0.84.0
(https://github.com/CliMA/Oceananigans.jl/releases/tag/v0.84.0).
Visualizations via Makie.jl.

---

## 1. Paper Summary

The paper introduces a **rotational WENO-based momentum advection scheme** for ocean
models on a staggered C-grid. The scheme provides implicit (built-in) dissipation
that replaces the need for explicit viscous closures (Smagorinsky, Leith, etc.),
functioning as an **Implicit Large Eddy Simulation (ILES)**.

### 1.1 Key Innovation

Standard ocean models use centered (energy/enstrophy conserving) advection + explicit
closures to damp grid-scale noise. This is problematic because:
- Explicit closures require tuning unknown parameters (C_smag, C_leith, etc.)
- Explicit closures act at all scales, not just the grid scale, damping the
  mesoscale inverse energy cascade
- The dispersion error from centered advection produces grid-scale noise that the
  closure must then fight

The WENO approach builds dissipation into the reconstruction, eliminating the need for
explicit closures. The key innovation is using **velocity-based smoothness indicators**
({zeta;u} stencils) for the vorticity reconstruction, which dramatically reduces
unnecessary dissipation because the velocity field is smoother than vorticity.

### 1.2 Rotational Form Decomposition

The momentum equations in rotational form on a C-grid:

```
D_t u = d_t u - Z_u + D_u + C_u + K_u     (x-momentum, Eq. 25)
D_t v = d_t v + Z_v + D_v + C_v + K_v     (y-momentum, Eq. 26)
```

where:
- **Z** = vorticity flux: WENO reconstruction of zeta at velocity points
- **D** = divergence flux: WENO reconstruction of horizontal divergence d
- **C** = conservative vertical advection: WENO reconstruction of u, v in vertical
- **K** = kinetic energy gradient: WENO reconstruction of delta(u^2), delta(v^2)

The vertical advection is split into C + D (Eqs. 23-24) to separate the conservative
part from the divergent part. This is critical because upwinding the full vertical
advection can inject energy at the grid scale (Appendix C).

### 1.3 Smoothness-Optimized Stencils {phi; psi}

Standard WENO computes smoothness from the field being reconstructed ({phi} = {phi; phi}).
The paper's innovation (Eqs. 39-43) uses a **different, smoother field** psi for
smoothness assessment:

```
{phi; psi}^i = sum_r  xi_{r,psi} [phi]^i_r
```

where xi_{r,psi} uses the Sobolev norm of the polynomials of psi (velocity), but
the actual reconstruction uses polynomials of phi (vorticity/divergence).

The specific stencils used (Eqs. 41-42):
```
u equation:  {zeta;u}^j,  {u}^k,  {D;D}^i,  {delta_i u^2; <u>}^i
v equation:  {zeta;u}^i,  {v}^k,  {D;D}^j,  {delta_j v^2; <v>}^j
```

### 1.4 Test Cases in the Paper

**Table 1: 2D Decaying Turbulence schemes**

| Name  | Vorticity flux      | Smoothness   | Explicit closure |
|-------|---------------------|--------------|------------------|
| DNS   | Energy conserving   | --           | --               |
| Leith1| Energy conserving   | --           | Leith, C=1       |
| Leith2| Energy conserving   | --           | Leith, C=2       |
| W5D   | WENO 5th order      | {zeta}       | --               |
| W9D   | WENO 9th order      | {zeta}       | --               |
| W5V   | WENO 5th order      | {zeta;u}     | --               |
| W9V   | WENO 9th order      | {zeta;u}     | --               |

**Table 3: 3D Baroclinic Jet schemes**

| Name | Advection   | Smoothness        | Explicit closure            |
|------|-------------|-------------------|-----------------------------|
| UP3  | Upwind 3rd  | --                | --                          |
| W9V  | WENO        | Eqs. 41-42        | --                          |
| W9D  | WENO        | Eqs. 37-38        | --                          |
| SM2  | Dispersive  | --                | Smagorinsky (OM4p25)        |
| QG2  | Dispersive  | --                | QG Leith, C=2               |

**Table 2: Advection discretization detail for 3D runs**

| Term         | Dispersive          | Upwind        | WENO          |
|--------------|---------------------|---------------|---------------|
| Z (vor flux) | E-conserving (17)   | --            | WENO 9th      |
| V (vert adv) | Centered 2nd (21)  | --            | --            |
| D (div flux) | --                  | --            | WENO 9th      |
| C (cons vert)| --                  | --            | WENO 5th      |
| K (KE grad)  | Centered 2nd (19)  | --            | WENO 5th      |
| Flux-form    | --                  | Upwind 3rd    | --            |
| Tracer       | WENO 7th            | WENO 7th      | WENO 7th      |

---

## 2. Codebase Audit: What We Already Have

### 2.1 Momentum Advection

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| Vector-invariant form (Z + K) | **Have** (2nd-order) | `ocean_pe_latlon_cgrid.py:396-453` | 2-point zeta average, not 4-point PV |
| Vorticity at vertices | **Have** | `latlon_cgrid_operators.py` `curl_vertex_cgrid()` | |
| KE at cell centers | **Have** | `ocean_pe_latlon_cgrid.py:397-405` | Uses perturbation velocity (issue #160) |
| Vertical momentum advection | **Have** (1st-order upwind) | `ocean_pe_latlon_cgrid.py:462-478` | Not split into C + D |
| TRiSK PV flux (MPAS) | **Have** (energy + enstrophy) | `ocean_pe_mpas.py:190-234` | Uses total velocity correctly |
| Flux-form momentum (UP3) | **Missing** | -- | No flux-form momentum anywhere |
| High-order vorticity reconstruction | **Missing** | -- | Only 2-point average |
| Rotational C + D splitting | **Missing** | -- | Vertical advection not decomposed |

**Known issue #160**: The lat-lon C-grid vorticity flux uses perturbation velocity
(not total), the zeta average is 2-point (not Sadourny/Arakawa-Hsu 4-point PV), and
the scheme conserves neither energy nor enstrophy. Must be fixed as a prerequisite
for WENO momentum work.

### 2.2 Tracer Advection

| Scheme | Order | Location | Available on |
|--------|-------|----------|-------------|
| Upwind (donor cell) | 1st | `ocean_pe_latlon_cgrid.py` | latlon, mpas |
| TVD (Van Leer) | 2nd | `ocean_pe_latlon_cgrid.py` | latlon, mpas |
| DST-3 | 3rd | `advection.py` | latlon |
| DST-3 multidim | 3rd | `advection.py` | latlon |
| PPM (Colella-Woodward) | 4th | `advection.py` | latlon |
| PPM + FCT (Zalesak) | 4th | `advection.py` | latlon |
| SOM (Prather 1986) | 2nd-moment | `advection_som.py` | latlon |
| **WENO (any order)** | **Missing** | -- | -- |

### 2.3 Explicit Dissipation / Closures

| Closure | Status | Location | Notes |
|---------|--------|----------|-------|
| Laplacian viscosity (A_h) | **Have** | `ocean_pe_latlon_cgrid.py:527-531` | Constant coefficient |
| Biharmonic viscosity (B_h) | **Have** | `ocean_pe_latlon_cgrid.py:533-541` | cos^4 polar scaling |
| Smagorinsky biharmonic | **Have** | `latlon_cgrid_operators.py:1414-1501` | C_smag parameter |
| Laplacian Smagorinsky | **Missing** | -- | Only biharmonic exists |
| **Combined Smag (OM4p25)** | **Missing** | -- | Lap + bilap + Burger-number scaling |
| **Leith (2D)** | **Missing** | -- | nu = (C*Delta/pi)^3 |grad(zeta)| |
| **QG Leith** | **Missing** | -- | Bachman et al. 2017 formulation |
| Laplacian tracer diffusion | **Have** | `ocean_pe_latlon_cgrid.py` | |
| Biharmonic tracer diffusion | **Have** | `ocean_pe_latlon_cgrid.py` | |
| Bottom drag (linear) | **Have** | `ocean_pe_latlon_cgrid.py` | |
| Vertical viscosity (A_v) | **Have** | `ocean_pe_latlon_cgrid.py` | Explicit centered |

### 2.4 WENO Kernels in Codebase

WENO reconstruction exists, but only in atmosphere radiation:

| Kernel | Location | Notes |
|--------|----------|-------|
| WENO3-JS | `atmosphere/physics/radiation/rrtmgp/interpolation.py:186-290` | 1D node-to-face |
| WENO5-JS | `atmosphere/physics/radiation/rrtmgp/interpolation.py:308-395` | 1D, epsilon=1e-15 |
| WENO5-Z | `atmosphere/physics/radiation/rrtmgp/interpolation.py:398-478` | 1D, epsilon=1e-20 |
| WENO9 | **Missing** | -- |
| C-grid WENO | **Missing** | -- |
| Smoothness-optimized {phi;psi} | **Missing** | -- |

### 2.5 Time Integration

| Integrator | Status | Used by |
|------------|--------|---------|
| Forward-Euler + Matsuno | **Have** | latlon, mpas baroclinic stepping |
| Forward-backward barotropic | **Have** | BEBT + cosine filter |
| SSP-RK3 | **Have** | Spectral ocean only |
| SSP-RK34, SSP-RK54, RK4 | **Have** | Spectral ocean only |
| **AB2 (Adams-Bashforth 2)** | **Missing** | Used by Silvestri 3D baroclinic jet |
| **RK3 (low-storage)** | **Missing** (ocean) | Used by Silvestri 2D turbulence |

### 2.6 Test Cases

| Test case | Status | Location | Gap vs. paper |
|-----------|--------|----------|---------------|
| Eady baroclinic instability | **Have** | `eady_instability.py` | Different domain/depth/stratification |
| ACC channel | **Have** | `acc_channel.py` | Wind-driven, different physics |
| **2D decaying turbulence** | **Missing** | -- | Doubly-periodic, RK3, FFT pressure |
| **Baroclinic jet (Silvestri)** | **Missing** | -- | Spherical sector 60S-40S, 1km, 1000 days |

### 2.7 Diagnostics

| Diagnostic | Status | Notes |
|------------|--------|-------|
| Global energy budget | **Have** | `diagnostics/energy_budget.py` |
| **Energy spectra (KE vs k)** | **Missing** | 2D FFT of velocity on lat-lon |
| **Enstrophy spectra** | **Missing** | 2D FFT of vorticity^2 |
| **Deformation radius** | **Missing** | L_d from N^2 profile and f |
| **Vertical buoyancy flux (w'b')** | **Missing** | Needed for APE/EKE conversion |

---

## 3. Differentiability Analysis

### 3.1 Overall Assessment: **WENO is differentiable and well-suited for legoESM**

All operations in the WENO scheme are differentiable:
- Polynomial reconstructions (smooth everywhere)
- Rational smoothness indicators beta (smooth everywhere)
- WENO weights alpha = f(beta) (smooth with epsilon regularization)
- Upwind splitting via max(u,0), min(u,0) (subgradient at kink, like ReLU)
- Convex combination of sub-stencil reconstructions (smooth)

**No `custom_vjp` needed.** Standard JAX reverse-mode AD works throughout.

### 3.2 Gradient Quality: Better Than Existing Limiters

| Scheme | Gradient character | Kink points per cell | Dead zones |
|--------|-------------------|---------------------|------------|
| Centered 2nd | Linear, smooth | 0 | No |
| Van Leer TVD | Piecewise linear | 3 | Yes (at limiter saturation) |
| FCT (Zalesak) | Nested min/max | 6+ | Yes (at flux bounds) |
| SOM (Prather) | Smooth polynomial | 0 | No |
| **WENO** | **Smooth rational** | **0** | **No** |

WENO provides higher-order accurate gradients in smooth regions and continuous
gradient signal near discontinuities, unlike TVD/FCT which have hard kinks that
create "dead zones" where gradient signal stops propagating.

### 3.3 Critical: Epsilon Selection

The WENO weight formula involves `1 / (beta + epsilon)^p`. The epsilon value is
critical for AD stability:

| Variant | Exponent p | Float64 epsilon | Float32 epsilon | Notes |
|---------|-----------|-----------------|-----------------|-------|
| WENO-JS | 2 | 1e-12 | 1e-6 | Larger p means steeper weight function |
| **WENO-Z** | **1** | **1e-16** | **1e-7** | **Preferred**: better-conditioned for AD |

**Paper's epsilon = 1e-40 is catastrophic for AD** — below float32 subnormal range,
causes Inf when inverted. Our existing codebase already knows this:
- `interpolation.py` weno5_z: epsilon = 1e-20
- `advection_som.py`: `_EPS = 1e-20` with comment about float32 overflow

**Recommendation: Use WENO-Z everywhere.** The existing `weno5_z` kernel uses the
better-conditioned p=1 formulation.

### 3.4 Memory for Reverse-Mode AD

Each WENO5 reconstruction stores ~32 intermediate arrays. The full rotational scheme
with 12 horizontal WENO reconstructions produces ~384 intermediates per timestep.

At 1-degree (180 x 360 x 50) float64:
- ~26 MB per array
- ~10 GB total for WENO intermediates

**Mitigation: `jax.checkpoint`** (already used extensively in legoESM):
- Wrap WENO momentum advection in `jax.checkpoint`
- Trades ~2x compute for ~10x memory savings
- WENO is O(n) with no iterations, so recomputation is cheap

### 3.5 The {zeta; u} Smoothness Stencil

The dual pathway (velocity feeds weights, vorticity feeds reconstruction) is handled
correctly by JAX reverse-mode AD. No gradient pathologies:

- Perturbation in u affects weights (through beta(psi)) and reconstruction
  (through zeta = curl(u)) at **different spatial locations and scales**
- Systematic cancellation is unlikely
- Near sharp fronts where both channels could amplify, WENO is maximally
  dissipative, which acts as a **gradient regularizer**

### 3.6 Implications for Training

| Use case | Impact | Assessment |
|----------|--------|------------|
| Physics parameter estimation | Neutral | WENO doesn't add/remove tunable params |
| Neural closure training | **Positive** | Richer gradient signal, higher-order accuracy |
| Implicit vs explicit dissipation | Trade-off | WENO removes viscosity coefficient from search space; explicit closures give cleaner parameter path |

For online training (Ross et al. 2023): WENO's implicit dissipation provides a
better forward model with less systematic bias for the neural network to learn.

### 3.7 Smooth WENO Variants to Consider

| Variant | Formula | AD benefit | Drawback |
|---------|---------|-----------|----------|
| WENO-JS | c_r / (beta + eps)^2 | Baseline | Steep weight function |
| **WENO-Z** | c_r * (1 + tau/(beta + eps)) | **Better** | Additive "1+" bounds weights |
| WENO-eta | exp(-C * beta^2 / max_beta^2) | **Best** for AD | Slightly more diffusive |
| TENO | Hard threshold on weights | **Worst** | Introduces discontinuities |

**Recommendation: Start with WENO-Z, consider WENO-eta if gradient issues arise.**

### 3.8 Pre-Implementation Validation Tests (2-3 days)

Before committing to full implementation, run these quick tests:

1. **AD through existing WENO5-Z kernel** (~1 hr):
   `jax.grad` through `weno5_z` in `interpolation.py`. Taylor test for correctness.

2. **1D WENO advection gradient test** (~2 hrs):
   Implement minimal 1D WENO5-Z flux. Compare gradient convergence (Taylor test)
   against upwind, centered, Van Leer TVD over 10-100 steps.

3. **Prototype WENO vorticity flux on C-grid** (~1 day):
   Replace 2-point zeta average in `ocean_pe_latlon_cgrid.py` with WENO5-Z
   reconstruction. Test `jax.grad` through 5 steps on 8x16 differentiability fixture.

4. **Memory profiling** (~2 hrs):
   Measure AD memory with and without `jax.checkpoint`. Extrapolate to production.

5. **Float32 epsilon sweep** (~1 hr):
   Test epsilon = 1e-5, 1e-6, 1e-7, 1e-8 in float32 for finite gradients + accuracy.

---

## 4. Implementation Plan

### Phase 1: Infrastructure & Quick Wins (1-2 weeks)

#### 1a. Core WENO Module — `src/legoesm/core/weno.py`

Factor WENO reconstruction from RRTMGP into a shared module. This is the foundation
for all subsequent work.

**What to implement:**
- `weno5_z(stencil, epsilon)` — 5th-order WENO-Z reconstruction (from `interpolation.py`)
- `weno9_z(stencil, epsilon)` — 9th-order WENO-Z (5 sub-stencils, new)
- `weno7_z(stencil, epsilon)` — 7th-order WENO-Z (for tracer advection)
- `weno_reconstruct(phi, psi, order, direction)` — smoothness-optimized {phi; psi} variant
- Left-biased and right-biased reconstructions for upwinding
- Configurable epsilon with sensible defaults for float32/float64
- Full test suite including Taylor tests for AD correctness

**Estimated LOC:** ~300
**Dependencies:** None
**Tests:** Unit tests + Taylor test for `jax.grad` through each kernel

#### 1b. Leith Closure (2D) — `src/legoesm/ocean/physics/lateral_mixing/`

The simplest missing closure from the paper. All ingredients exist.

**Formula** (Eq. A1):
```
nu_star = (C * Delta / pi)^3 * |grad(zeta)|
```

**What to implement:**
- `leith_viscosity_cgrid(u, v, grid, C_leith)` — compute effective viscosity
- `leith_tendency_cgrid(u, v, grid, C_leith)` — full tendency (Laplacian with Leith nu)
- Config field `C_leith` in `LatLonCGridOceanConfig`
- Dispatch in `ocean_pe_latlon_cgrid.py`

**Ingredients already available:**
- `curl_vertex_cgrid(u, v)` for zeta
- `gradient_x_cgrid`, `gradient_y_cgrid` for |grad(zeta)|
- `vector_laplacian_cgrid` for the diffusion operator

**Estimated LOC:** ~80
**Dependencies:** None

#### 1c. Diagnostics — Deformation Radius

**Formula** (Eq. 54):
```
L_d = (1 / (pi * |f|)) * integral_{-H}^{0} sqrt(d_z b) dz
```

**Estimated LOC:** ~50
**Dependencies:** EOS for buoyancy, grid for f

#### 1d. Diagnostics — Energy & Enstrophy Spectra

2D FFT of velocity (for KE spectra) and vorticity (for enstrophy spectra) on the
lat-lon grid. Compute isotropic 1D spectra by binning in wavenumber shells.

**Estimated LOC:** ~200
**Dependencies:** `jnp.fft.fft2`

### Phase 2: WENO Tracer Advection (1 week)

#### 2a. WENO5/WENO7 Tracer Flux — `src/legoesm/ocean/advection.py`

Apply the core WENO module to C-grid tracer reconstruction. Follow the existing
DST-3/PPM pattern (stencil assembly, boundary ghost cells, flux computation).

**What to implement:**
- `weno5_tracer_flux_x/y(T, u_face, grid)` — 5th-order WENO flux
- `weno7_tracer_flux_x/y(T, u_face, grid)` — 7th-order WENO flux (paper's tracer scheme)
- Wall boundary conditions via ghost-cell extrapolation (follow DST-3 pattern)
- Config dispatch: `tracer_advection = "weno5"` or `"weno7"`

**Estimated LOC:** ~200
**Dependencies:** Phase 1a (core WENO module)

### Phase 3: Test Cases (1-2 weeks)

#### 3a. Baroclinic Jet — Silvestri Configuration

Modify existing `eady_instability.py` or create a new experiment matching the paper's
exact setup.

**Paper setup (Section 5):**
- Spherical sector: 60S to 40S, 20 degrees wide
- Depth: 1 km
- Stratification: N^2 = 4e-6 s^-2
- Meridional buoyancy front (Eqs. 52-53):
  ```
  b(phi, z) = N^2 * z + Delta_b * [smooth front profile]
  phi_0 = 50S, Delta_phi = 20 deg, Delta_b = 5e-3 m/s^2
  ```
- Initial velocity in thermal-wind balance
- Weak white noise perturbation to kick-start instability
- Boundary conditions: no-flux, free-slip on all solid walls
- Background vertical viscosity: nu = 1e-4 m^2/s
- Background vertical diffusivity: kappa = 1e-5 m^2/s
- Zonal-mean restoring to initial profiles, timescale 50 days
- Total integration: 1000 days
- Resolutions: 1/8, 1/16, 1/32 degree (~14, 7, 3.5 km)
- Vertical: 20 m fixed spacing (50 levels)
- Time stepping: AB2 + free-surface subcycling

**Estimated LOC:** ~150
**Dependencies:** Existing infrastructure; optionally Phase 4d (AB2)

#### 3b. 2D Decaying Homogeneous Turbulence

New experiment — doubly periodic domain with random vorticity initial condition.

**Paper setup (Section 4):**
- Domain: 2*pi x 2*pi (nondimensional), doubly periodic
- Re = 3.3e4
- Initial condition: narrow-band energy spectrum (Ishiko et al. 2009, Eq. 48-50)
  ```
  E(k) = 0.5 * a_0 * k_p^{-1} * (k/k_p)^7 * exp(-7/6 * (k/k_p)^6)
  k_p = 12 (energy peak wavenumber)
  ```
- Vorticity field from energy spectrum with random phases (Eq. 49-50)
- Time stepping: RK3 (low-storage) + FFT pressure projection
- Duration: t = 6 (nondimensional, ~18 eddy turnovers)
- Resolutions: 64x64, 128x128, 256x256, 1024x1024

**Estimated LOC:** ~250
**Dependencies:** FFT pressure solver (new), or adapt spectral ocean path

### Phase 4: Momentum Advection — Core Implementation (2-4 weeks)

#### 4a. Fix Issue #160: Total Velocity in Vorticity Flux

**Prerequisite for WENO momentum advection.**

Refactor `ocean_pe_latlon_cgrid.py` to:
- Use total velocity (not perturbation) in vorticity flux computation
- Implement proper 4-point PV averaging (Arakawa & Hsu 1990 or Sadourny 1975)
- Ensure energy or enstrophy conservation with centered scheme as baseline

This is a correctness fix independent of WENO. It establishes a correct baseline
against which WENO can be compared.

**Estimated LOC:** ~100 (refactor)
**Dependencies:** None, but requires careful validation

#### 4b. WENO Rotational Momentum Advection

The main implementation. Replace centered vorticity reconstruction with WENO.

**What to implement for each term:**

**Z (vorticity flux, Eq. 18 → Eq. 27):**
- Reconstruct zeta from vertices to u-faces (j-direction) and v-faces (i-direction)
- WENO5 or WENO9 with {zeta; u} smoothness stencils
- Upwind direction: v (for u-equation), u (for v-equation)

**D (divergence flux, Eqs. 31-32):**
- Compute D = delta_i(U) + delta_j(V) at cell centers
- Reconstruct D to u-faces (i-direction) and v-faces (j-direction)
- Upwind reconstruction with {D; D} or upwind {delta_i U; delta_i U}
- WENO9 for Z-direction, WENO5 for D, C, K (as in Table 2)

**C (conservative vertical advection, Eqs. 23-24):**
- Reconstruct u, v to vertical cell faces
- Upwind direction: vertical velocity w
- WENO5 in vertical

**K (kinetic energy gradient, Eqs. 19 + 33):**
- Replace centered <delta_i u^2> with upwind {delta_i u^2; <u>}
- WENO5

**Estimated LOC:** ~400
**Dependencies:** Phase 1a (core WENO), Phase 4a (issue #160 fix)

#### 4c. QG Leith Closure

**Formula** (Eqs. A2-A3):
```
nu_star = (C*Delta/pi)^3 * (min(|grad q1|, |grad q2|, |grad q3|)^2 + |grad(div u)|^2)^{1/2}

q1 = grad(q) + d_z(f/N^2 * grad(b))
q2 = grad(q) * (1 + 1/Bu)
q3 = grad(q) * (1 + 1/Ro^2)

Bu = Delta^2 / L_d^2,  Ro = V / (|f| * Delta)
```

Requires N^2 from EOS, potential vorticity gradient, Burger and Rossby numbers.

**Estimated LOC:** ~200
**Dependencies:** Phase 1c (deformation radius), Phase 1b (Leith base)

#### 4d. OM4p25-Style Combined Smagorinsky

**Formulas** (Eqs. A5-A8):
```
nu_4 = max[C4 * |D| * Delta^4,  C4^u * Delta^3]     (bilaplacian)
nu_2 = max[C2 * |D| * Delta^2,  C2^u * Delta] * F    (laplacian)
F = (1 + 0.25 * Bu^{-2})^{-1}

C4 = 0.06,  C4^u = 0.01,  C2 = 0.15,  C2^u = 0.01
```

Extends existing `smagorinsky_biharmonic_tendency_cgrid` with:
- Background viscosity floors (C4^u, C2^u terms)
- Laplacian component with Burger-number-dependent reduction
- Combined application of both operators

**Estimated LOC:** ~150
**Dependencies:** Existing Smagorinsky, Phase 1c (deformation radius for Bu)

#### 4e. AB2 Time Integrator (Optional)

Adams-Bashforth 2nd order: `u^{n+1} = u^n + dt * (1.5*F^n - 0.5*F^{n-1})`

Requires storing previous tendency in the scan carry (SegmentCarry).

**Estimated LOC:** ~80
**Dependencies:** SegmentCarry extension (cross-cutting, see CLAUDE.md rules)

#### 4f. UP3 Flux-Form Momentum Advection (Optional)

3rd-order upwind-biased flux-form momentum advection, as used in ROMS.

This is an alternative to vector-invariant form. Would require a new momentum
advection pathway. Lower priority unless specifically needed for the UP3 comparison
case.

**Estimated LOC:** ~200
**Dependencies:** Significant refactoring of momentum update pathway

### Phase 5: Full Paper Reproduction (2+ weeks)

#### 5a. 2D Turbulence Comparison Matrix

Run all 7 schemes from Table 1 at 4 resolutions:
```
Schemes: DNS, Leith1, Leith2, W5D, W9D, W5V, W9V
Resolutions: 64x64, 128x128, 256x256, 1024x1024
```

**Diagnostics to produce:**
- Vorticity snapshots at t = 3.6 (Figure 3)
- KE and enstrophy time series (Figure 4)
- Energy and enstrophy spectra at t = 3.6 (Figure 5)

#### 5b. 3D Baroclinic Jet Comparison Matrix

Run all 5 schemes from Table 3 at 3 resolutions:
```
Schemes: W9V, W9D, SM2, QG2, UP3
Resolutions: 1/8 (14 km), 1/16 (7 km), 1/32 (3.5 km)
Duration: 1000 days each
```

**Diagnostics to produce:**
- Near-surface vorticity snapshots at day 220 (Figure 7)
- Total KE, eddy KE, eddy APE time series (Figure 8)
- Time-averaged energy, enstrophy, w'b' spectra (Figure 9)
- Time-and-zonal-mean buoyancy contours (Figure 10)
- Deformation radius evolution (Figure 6)

---

## 5. Dependency Graph

```
Phase 1a: Core WENO ──┬──> Phase 2a: WENO tracer advection
                       │
                       ├──> Phase 4b: WENO momentum advection
                       │         ↑
Phase 4a: Fix #160 ────┘
                       
Phase 1b: Leith ──────────> Phase 4c: QG Leith

Phase 1c: Deformation ───┬─> Phase 4c: QG Leith
                          └─> Phase 4d: OM4p25 Smagorinsky

Phase 1d: Spectra ────────> Phase 5a/5b: Reproduction

Phase 3a: Baroclinic jet ──> Phase 5b: Reproduction
Phase 3b: 2D turbulence ──> Phase 5a: Reproduction
```

---

## 6. Conversation Workflow

This plan spans ~2000 LOC across ~13 components. Implementing it in a single
conversation would exhaust context and degrade quality. Each conversation should
target a self-contained deliverable with its own tests.

### Recommended conversation sequence

**Conversation 1: Core WENO module + AD validation** (Phase 1a + Section 3.8 tests)
- Factor WENO5-Z from RRTMGP into `src/legoesm/core/weno.py`
- Add WENO7-Z and WENO9-Z kernels
- Add smoothness-optimized {phi; psi} variant
- Write unit tests for all kernels
- Run AD Taylor tests (Section 3.8, tests 1-2): `jax.grad` through WENO kernels
- **Gate**: if Taylor tests fail or epsilon needs rethinking, stop here and reassess
  before investing further

**Conversation 2: Leith closure + diagnostics** (Phases 1b + 1c + 1d)
- 2D Leith closure on lat-lon C-grid
- Deformation radius diagnostic
- Energy/enstrophy spectra diagnostic (2D FFT)
- Tests for each

**Conversation 3: WENO tracer advection** (Phase 2a)
- WENO5 and WENO7 tracer flux on C-grid
- Config dispatch (`tracer_advection = "weno5"` / `"weno7"`)
- Validate against DST-3/PPM on existing test cases
- AD Taylor test through WENO tracer advection

**Conversation 4: Test cases** (Phases 3a + 3b)
- Baroclinic jet with Silvestri-specific parameters
- 2D decaying turbulence experiment (requires FFT pressure solver decision)
- Run baseline cases with existing schemes (centered + Smagorinsky)

**Conversation 5: Fix issue #160** (Phase 4a)
- Refactor vorticity flux to total velocity
- Implement 4-point PV averaging (Arakawa-Hsu or Sadourny)
- Careful validation: rest state, gyre, Eady must still pass
- This is a correctness fix independent of WENO — do NOT bundle with WENO work
- **Gate**: validate baseline centered scheme is energy/enstrophy conserving before
  proceeding

**Conversation 6: WENO rotational momentum advection** (Phase 4b)
- The big one — ~400 LOC
- Implement all 4 terms (Z, D, C, K) with WENO reconstruction
- Both {phi} (standard) and {phi; psi} (smoothness-optimized) variants
- AD Taylor test through 5-10 ocean steps with WENO momentum
- Memory profiling with `jax.checkpoint`
- May need to split into 6a (Z + K terms) and 6b (C + D terms) if too large

**Conversation 7: Alternative closures** (Phases 4c + 4d + optionally 4e)
- QG Leith closure
- OM4p25 combined Smagorinsky
- AB2 time integrator (if needed for baroclinic jet)

**Conversation 8+: Paper reproduction runs** (Phase 5)
- Run comparison matrices
- Generate diagnostic figures
- Iterate on parameters/resolution as needed
- Multiple conversations likely needed as experiments reveal issues

### Decision gates

| After conversation | Gate question | If NO |
|--------------------|--------------|-------|
| 1 | Do AD Taylor tests pass for WENO5-Z? | Investigate epsilon, try WENO-eta, reassess |
| 1 | Is WENO9 stencil width compatible with halo infrastructure? | Stick with WENO5 only |
| 4 | Does 2D turbulence FFT solver work? | Use spectral ocean path instead |
| 5 | Is centered scheme energy-conserving after #160 fix? | Debug before adding WENO |
| 6 | Does AD through WENO momentum fit in memory with checkpoint? | Reduce to WENO5 only, or gradient accumulation |

### Grid priority

Implement on **lat-lon C-grid first** throughout. The lat-lon grid is simpler
(regular stencils, no panel boundaries) and matches the paper's setup. Port to
MPAS/cubed-sphere only after the lat-lon implementation is validated.

---

## 8. Risk Assessment

### High Risk
- **Issue #160 (vorticity flux correctness)**: If the current formulation has deep
  structural issues, fixing it could cascade into significant refactoring. Mitigated
  by the fact that MPAS already has a correct TRiSK implementation as reference.

### Medium Risk
- **WENO9 stencil width**: 9th-order requires a 10-point stencil (halo = 5). Current
  lat-lon halo infrastructure may need extension. Check halo width constraints.
- **2D turbulence FFT pressure solver**: Requires an elliptic solve not currently
  available in the lat-lon ocean. Could use spectral ocean path instead, or
  implement a simple FFT-based Poisson solver for doubly-periodic domains.
- **Memory at high resolution**: WENO9 at 1/32 degree with 50 levels and AD could
  exceed single-GPU memory even with checkpointing. May need gradient accumulation
  or model parallelism.

### Low Risk
- **Differentiability**: Analysis confirms WENO-Z is well-conditioned for AD with
  appropriate epsilon. No custom_vjp needed.
- **Leith/QG Leith/OM4p25 Smagorinsky**: Straightforward implementations building
  on existing operators.
- **WENO tracer advection**: Follows established pattern of DST-3/PPM in `advection.py`.

---

## 9. Success Criteria

### Minimum viable comparison (Phase 1-3)
- [ ] Core WENO module with unit tests and AD Taylor tests
- [ ] Leith closure working on lat-lon C-grid
- [ ] Energy/enstrophy spectra diagnostic
- [ ] WENO7 tracer advection working
- [ ] Baroclinic jet experiment running with at least centered + Leith + Smagorinsky

### Full paper reproduction (Phase 4-5)
- [ ] WENO rotational momentum advection (W5V, W9V, W5D, W9D variants)
- [ ] All 5 dissipation approaches (W9V, W9D, SM2, QG2, UP3)
- [ ] 2D turbulence test at 4 resolutions
- [ ] 3D baroclinic jet at 3 resolutions, 1000-day integrations
- [ ] Spectra matching paper's Figures 4, 5, 8, 9 qualitatively
- [ ] AD verified through WENO momentum advection (Taylor test, 10+ steps)
- [ ] `jax.checkpoint` memory scaling validated at target resolution

### Differentiability milestones
- [ ] `jax.grad` through WENO5-Z kernel (Taylor test, O(h^2) convergence)
- [ ] `jax.grad` through 5 ocean steps with WENO vorticity flux
- [ ] `jax.grad` through full rotational WENO momentum (12 reconstructions)
- [ ] Memory profile: AD through WENO momentum < 2x baseline (with checkpoint)
- [ ] Float32 stability confirmed for Metal backend
