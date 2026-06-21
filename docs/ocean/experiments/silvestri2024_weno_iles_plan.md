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

## Implementation Status (updated 2026-04-25)

The **WENO-ILES momentum advection is fully implemented** and validated on the
lat-lon C-grid. All four rotational momentum terms (Z, D, K, C) use WENO
reconstruction with smoothness-optimized stencils, gated by
`config.momentum_advection = "weno5" | "weno7"`.

### What is done

| Phase | Description | Status | Date |
|-------|-------------|--------|------|
| 1a | Core WENO module (`core/weno.py`) | **DONE** | 2026-04-23 |
| 1b | 2D Leith closure | **DONE** | 2026-04-23 |
| 1c | Deformation radius diagnostic | **DONE** | 2026-04-24 |
| 1d | Energy/enstrophy spectra diagnostic | **DONE** | 2026-04-24 |
| 2a | WENO tracer advection (WENO5/7 horizontal + vertical) | **DONE** | 2026-04-24 |
| 2b | WENO momentum advection — Z (vorticity flux) + C (vertical) | **DONE** | 2026-04-24 |
| 3a | Silvestri baroclinic jet experiment script | **DONE** | 2026-04-24 |
| 3b | 2D decaying turbulence experiment | **SKIPPED** | — (requires FFT pressure solver; out of scope) |
| 4a | Fix issue #160 (total-velocity PV flux, Sadourny EC) | **DONE** | 2026-04-25 |
| 4b | WENO D (divergence flux) + K (KE gradient) terms | **DONE** | 2026-04-25 |
| 4c | QG Leith closure | NOT DONE | — |
| 4d | OM4p25 combined Smagorinsky | NOT DONE | — |
| 4e | AB2 time integrator | **SKIPPED** | — (only needed for UP3; out of scope) |
| 4f | UP3 flux-form momentum advection | **SKIPPED** | — (requires horizontal flux-form momentum pathway; out of scope) |
| 5a | 2D turbulence comparison matrix | **SKIPPED** | — (depends on 3b) |
| 5b | 3D baroclinic jet comparison matrix | NOT DONE | — |

### WENO momentum implementation detail

All in `src/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py`:

| Term | Function(s) | Stencil | Order | Notes |
|------|-------------|---------|-------|-------|
| Z (vorticity flux) | `_weno_zeta_at_u`, `_weno_zeta_at_v` | {ζ; v/u} | config (5 or 7) | Smoothness-optimized split |
| D (divergence flux) | `_weno_cell_to_uface`, `_weno_cell_to_vface` | {D; D} | always 5 | Self-smoothness |
| K (KE gradient) | `_weno_usq_to_cell`, `_weno_vsq_to_cell` | {u²; u}, {v²; v} | always 5 | Cross-stencil |
| C (vertical advection) | `_flux_form_vertical_momentum_advection_weno` | standard | config (5 or 7) | Delegates to tracer WENO |

Test coverage: 42 tests in `tests/ocean/unit/test_weno_momentum.py` (shapes,
constant/linear exactness, periodic/wall BC, AD gradient finiteness, Taylor test,
full tendency integration, WENO-vs-centered comparison).

### Validation results (baroclinic jet, 40x40, 100 days)

- Both centered+Smagorinsky and WENO5 (no explicit viscosity) are stable
- WENO5 produces ~28x more eddy KE — Smagorinsky overdamps the baroclinic instability
- D-term sign confirmed correct (positive: `du_dt += D_at_u * u`)
- WENO5 without any explicit viscosity (C_smag=0) runs stably at 40x40 for 100 days
- AD through full Z+D+K+C WENO path validated (gradient finiteness + Taylor tests)

### What remains for full Silvestri paper reproduction

The remaining phases (4c–5b) are **not part of the WENO implementation** — they
implement competing closure schemes and run the paper's comparison experiments:

1. **Phase 1c: Deformation radius** (~50 LOC) — prerequisite for QG Leith and OM4p25
2. **Phase 1d: Energy/enstrophy spectra** (~200 LOC) — needed for paper Figures 4,5,9
3. **Phase 3b: 2D turbulence experiment** (~250 LOC) — doubly-periodic, needs FFT pressure
4. **Phase 4c: QG Leith closure** (~200 LOC) — "QG2" scheme in paper's Table 3
5. **Phase 4d: OM4p25 combined Smagorinsky** (~150 LOC) — "SM2" scheme in Table 3
6. **Phase 4e: AB2 time integrator** (~80 LOC) — paper uses AB2 for baroclinic jet
7. **Phase 4f: UP3 flux-form momentum** (~200 LOC) — "UP3" scheme in Table 3
8. **Phase 5: Full comparison matrix** — 7 schemes x 4 resolutions (2D) + 5 schemes x 3 resolutions (3D)

---

## 2. Codebase Audit: What We Already Have

### 2.1 Momentum Advection

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| Vector-invariant form (Z + K) | **Have** (centered + WENO) | `ocean_pe_latlon_cgrid.py` | Sadourny EC + WENO5/7 option |
| WENO Z (vorticity flux) | **Have** (WENO5/7) | `_weno_zeta_at_u/v` | {ζ; v/u} smoothness-optimized |
| WENO D (divergence flux) | **Have** (WENO5) | `_weno_cell_to_uface/vface` | {D; D} self-smoothness |
| WENO K (KE gradient) | **Have** (WENO5) | `_weno_usq_to_cell`, `_weno_vsq_to_cell` | {u²; u} cross-stencil |
| WENO C (vertical advection) | **Have** (WENO5/7) | `_flux_form_vertical_momentum_advection_weno` | Delegates to tracer WENO |
| Vorticity at vertices | **Have** | `latlon_cgrid_operators.py` `curl_vertex_cgrid()` | |
| KE at cell centers | **Have** | `ocean_pe_latlon_cgrid.py` | Total velocity (#160 fixed) |
| Vertical momentum advection | **Have** (1st-order + WENO) | `ocean_pe_latlon_cgrid.py` | Config-gated |
| TRiSK PV flux (MPAS) | **Have** (energy + enstrophy) | `ocean_pe_mpas.py:190-234` | Uses total velocity correctly |
| Flux-form momentum (UP3) | **Missing** | -- | No flux-form momentum anywhere |

**Issue #160**: Fixed in commit `ad9757a`. Total-velocity Sadourny EC baseline
established with 4-point PV averaging.

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
| WENO5 | **Have** | `advection.py` | latlon C-grid, horizontal + vertical |
| WENO7 | **Have** | `advection.py` | latlon C-grid, horizontal + vertical |

### 2.3 Explicit Dissipation / Closures

| Closure | Status | Location | Notes |
|---------|--------|----------|-------|
| Laplacian viscosity (A_h) | **Have** | `ocean_pe_latlon_cgrid.py:527-531` | Constant coefficient |
| Biharmonic viscosity (B_h) | **Have** | `ocean_pe_latlon_cgrid.py:533-541` | cos^4 polar scaling |
| Smagorinsky biharmonic | **Have** | `latlon_cgrid_operators.py:1414-1501` | C_smag parameter |
| Laplacian Smagorinsky | **Missing** | -- | Only biharmonic exists |
| **Combined Smag (OM4p25)** | **Missing** | -- | Lap + bilap + Burger-number scaling |
| Leith (2D) | **Have** | `latlon_cgrid_operators.py` | `leith_biharmonic_tendency_cgrid`, C_leith config |
| **QG Leith** | **Missing** | -- | Bachman et al. 2017 formulation |
| Laplacian tracer diffusion | **Have** | `ocean_pe_latlon_cgrid.py` | |
| Biharmonic tracer diffusion | **Have** | `ocean_pe_latlon_cgrid.py` | |
| Bottom drag (linear) | **Have** | `ocean_pe_latlon_cgrid.py` | |
| Vertical viscosity (A_v) | **Have** | `ocean_pe_latlon_cgrid.py` | Explicit centered |

### 2.4 WENO Kernels in Codebase

| Kernel | Location | Notes |
|--------|----------|-------|
| WENO5-Z | **Have** | `core/weno.py` | Shared module, WENO-Z weights, AD-safe epsilon |
| WENO7-Z | **Have** | `core/weno.py` | 8-point stencil |
| WENO9-Z | **Have** | `core/weno.py` | 10-point stencil |
| Smoothness-optimized {phi;psi} | **Have** | `core/weno.py` `weno_reconstruct_split()` | Silvestri Eqs. 39-43 |
| Upwind selection | **Have** | `core/weno.py` `weno_upwind()` | |
| WENO3-JS (radiation) | **Have** | `atmosphere/physics/radiation/rrtmgp/interpolation.py` | Legacy, not used by ocean |

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
| Baroclinic jet (Silvestri) | **Have** | `run_silvestri_baroclinic_jet.py` | Validated 100 days at 40x40; 1000-day high-res not yet run |
| **2D decaying turbulence** | **Missing** | -- | Doubly-periodic, RK3, FFT pressure |

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

#### 1a. Core WENO Module — `src/legoesm/core/weno.py` ✅ DONE

Factored WENO reconstruction into a shared module with WENO5/7/9-Z kernels,
smoothness-optimized {phi; psi} variant (`weno_reconstruct_split`), and
upwind selection (`weno_upwind`). AD-safe epsilon defaults (1e-16 float64,
1e-7 float32). Full unit + Taylor test suite in `tests/core/test_weno.py`.

#### 1b. Leith Closure (2D) ✅ DONE

Implemented as `leith_biharmonic_tendency_cgrid` in `latlon_cgrid_operators.py`.
Config field `C_leith` in `LatLonCGridOceanConfig`.

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

### Phase 2: WENO Tracer Advection (1 week) ✅ DONE

#### 2a. WENO5/WENO7 Tracer Flux ✅ DONE

Implemented in `ocean/advection.py`: `weno5_to_u/v_points`, `weno7_to_u/v_points`
for horizontal, `flux_form_vertical_tracer_advection_weno5/7` for vertical.
Config dispatch: `tracer_advection = "weno5" | "weno7"`.
30 tests in `tests/ocean/unit/test_advection_weno.py`.

### Phase 3: Test Cases (1-2 weeks)

#### 3a. Baroclinic Jet — Silvestri Configuration ✅ DONE

Implemented in `scripts/run/run_silvestri_baroclinic_jet.py`. Matches paper setup:
60S-40S spherical sector, 1 km depth, N²=4e-6, tanh front, thermal-wind IC,
sponge restoring, 50 vertical levels. Supports `centered`, `leith`, `weno5`,
`weno5_leith` schemes. Validated at 40x40 for 100 days (centered vs weno5).
1000-day high-resolution runs not yet attempted.

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

#### 4a. Fix Issue #160: Total Velocity in Vorticity Flux ✅ DONE

Fixed in commit `ad9757a`. Total-velocity Sadourny EC scheme with 4-point PV
averaging at vertices. Uses total velocity for KE and vorticity; Coriolis
handled separately in step function via Matsuno stepping.

#### 4b. WENO Rotational Momentum Advection ✅ DONE

All four terms implemented in `ocean_pe_latlon_cgrid.py`, gated by
`config.momentum_advection in ("weno5", "weno7")`:

**Z (vorticity flux)** — `_weno_zeta_at_u`, `_weno_zeta_at_v`
- {ζ; v/u} smoothness-optimized stencil, config order (5 or 7)
- Done in Phase 2b (2026-04-24)

**D (divergence flux)** — `_weno_cell_to_uface`, `_weno_cell_to_vface`
- {D; D} self-smoothness, always WENO5
- Done in Phase 4b (2026-04-25)

**K (KE gradient)** — `_weno_usq_to_cell`, `_weno_vsq_to_cell`
- {u²; u}/{v²; v} cross-stencil, always WENO5
- Replaces centered (avg(u))² with upwind avg(u²) for shock-capturing
- Done in Phase 4b (2026-04-25)

**C (vertical advection)** — `_flux_form_vertical_momentum_advection_weno`
- Delegates to tracer WENO5/7, config order
- Done in Phase 2b (2026-04-24)

42 tests passing. AD validated (Taylor test + gradient finiteness).
Baroclinic jet stable 100 days at 40x40 without explicit viscosity.

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
Phase 1a: Core WENO ──┬──> Phase 2a: WENO tracer advection     ✅ ALL DONE
                       │
                       ├──> Phase 4b: WENO momentum (Z+D+K+C)  ✅ DONE
                       │         ↑
Phase 4a: Fix #160 ────┘                                        ✅ DONE
                       
Phase 1b: Leith ──────────> Phase 4c: QG Leith                 ⬜ remaining
                                  ↑
Phase 1c: Deformation ───┬───────┘                              ⬜ remaining
                          └─> Phase 4d: OM4p25 Smagorinsky      ⬜ remaining

Phase 1d: Spectra ────────> Phase 5a/5b: Reproduction           ⬜ remaining

Phase 3a: Baroclinic jet ──> Phase 5b: Reproduction             ✅ script done
Phase 3b: 2D turbulence ──> Phase 5a: Reproduction              ⬜ remaining
```

---

## 6. Conversation History

The WENO-ILES implementation was completed across multiple conversations:

| Conv | What was done | Status |
|------|---------------|--------|
| 1 | Core WENO module (Phase 1a) + AD validation | ✅ Done |
| 2 | Leith closure (Phase 1b) | ✅ Done |
| 3 | WENO tracer advection (Phase 2a) | ✅ Done |
| 4 | WENO momentum Z+C terms (Phase 2b) + baroclinic jet script (Phase 3a) | ✅ Done |
| 5 | Fix issue #160 (Phase 4a) + WENO D+K terms (Phase 4b) | ✅ Done |

### Implementation bugs found (2026-04-25 audit)

A deep audit by both the dycore and ocean-model expert agents found that
the legoESM WENO momentum implementation does not faithfully match the
Silvestri paper. These bugs are the most likely cause of the surprising
result that legoESM-WENO5 (no explicit viscosity) appears MORE dissipative
than centered + B_h=2.3e11 on the Eady experiment — opposite of the paper's
claim that WENO produces ~28x more EKE than Smagorinsky.

**Critical bugs:**

1. **D-term: wrong split (Silvestri Eqs. 31-32)** —
   `_weno_cell_to_uface/vface` in `ocean_pe_latlon_cgrid.py:885-890`.
   Paper prescribes `{D}_i = {δ_i U}_i + ⟨δ_j V⟩_i`: only the
   matching-direction divergence is WENO-upwinded; the cross-direction
   component must be **centered**. Our code WENO-reconstructs the full
   divergence, which is exactly the non-energy-dissipative operation
   Appendix C explicitly warns against. Plus a likely sign error
   (paper has `du/dt -= D·u` per Eq. 25 evolution form; code has `+=`).
   Disabled via `weno_d_term=False` default; needs proper fix.

2. **K-term: reconstruction order wrong (Silvestri Eq. 33)** —
   `_weno_usq_to_cell`, `_weno_vsq_to_cell` in `ocean_pe_latlon_cgrid.py:
   551-645` and assembly at lines 766-789. Paper: compute `δ_i u²` at
   cell centers FIRST (small magnitudes), then WENO upwind these
   differenced fields back to faces. Our code: WENO `u²` to cell
   centers (full magnitudes), then centered gradient. Reconstructing
   `u²` directly causes WENO weights to be more upwind-biased
   everywhere there's shear, **adding dissipation** instead of
   reducing it as the paper intends. Plus the smoothness field uses
   raw face values where paper uses `⟨u⟩_i` (cell-centered averages).
   This is the most likely explanation for the over-dissipation
   observed in Eady.

3. **Z-term smoothness stencil possibly incomplete (Eq. 43)** —
   `_weno_zeta_at_u/v` in `ocean_pe_latlon_cgrid.py:271-385`. Paper
   may prescribe `{ζ; u} = ({ζ; ⟨u⟩_j} + {ζ; ⟨v⟩_i}) / 2` — average
   of two WENO reconstructions with different smoothness fields. Our
   code does only one. Verify against Oceananigans v0.84 reference.

4. **Land mask not applied inside WENO stencils** — all helpers.
   Currently harmless for Silvestri jet (no internal coasts) but a
   problem for Eady, ACC channel, AMIP. Easy fix: `_neumann_fill_cgrid`
   inputs before stencil construction.

**Suggested test additions** (current 42 tests don't catch these):
- K-term consistency with Eq. 33 (`{δ_i u²}_i` form)
- D-term Eq. 30 anti-diffusion test (divergence-free input)
- Energy-budget regression: WENO should be LESS dissipative than
  centered+B_h on a smooth Taylor-Green vortex
- 2-step parity: energy decay rate vs. analytical

### What remains for full paper reproduction

**Infrastructure needed before production Silvestri runs:**

- **Fix WENO momentum bugs above** (highest priority — invalidates
  prior comparison runs). Order: K-term first (most likely explains
  over-dissipation), then D-term split + sign, then Z-term Eq. 43
  averaging, then add regression tests.
- **Zonal-mean restoring** (~40 LOC): The paper restores the zonal-mean b and u
  to the initial profiles everywhere in the domain with a 50-day timescale
  (Soufflet et al. 2016 approach). This is fundamentally different from our
  boundary sponge — it maintains the mean jet while allowing eddies to develop
  freely. Without it, the mean flow accelerates unchecked and WENO5 blows up
  at 160x160 by day 34. Implementation: each timestep compute zonal mean of
  T, u; apply tendency `dX/dt += (1/tau) * (X_init_zonalmean - X_zonalmean)`
  uniformly across longitudes. Only the zonal-mean component is restored.
- **Front profile** (Eq. 52-53): The paper uses a smooth piecewise sin/cos
  profile spanning the full 20° domain width, not a narrow tanh. Should be
  updated to match.
- **WENO order**: Paper uses 9th-order for Z and D terms; we currently only
  support up to 7th. Consider adding `weno9` config option or testing with
  `weno7` as a close approximation.

**Next conversation(s)** would cover:

- Zonal-mean restoring implementation + Silvestri front profile fix
- Phase 4c (QG Leith) + Phase 4d (OM4p25 Smagorinsky) — competing schemes
- Phase 5 (comparison matrix) — production runs + figure generation

### Grid priority

Implemented on **lat-lon C-grid**. Port to MPAS/cubed-sphere only after the
lat-lon implementation is validated and comparison study complete.

---

## 8. Risk Assessment

### Resolved Risks
- ~~**Issue #160 (vorticity flux correctness)**~~ — Fixed. Total-velocity Sadourny EC baseline.
- ~~**Differentiability**~~ — Confirmed. WENO-Z well-conditioned for AD. Taylor tests pass.
- ~~**WENO tracer advection**~~ — Done. 30 tests passing.
- ~~**Leith closure**~~ — Done.

### Remaining Risks (for paper reproduction only)
- **2D turbulence FFT pressure solver**: Requires an elliptic solve not currently
  available in the lat-lon ocean. Could use spectral ocean path instead, or
  implement a simple FFT-based Poisson solver for doubly-periodic domains.
- **Memory at high resolution**: WENO at 1/32 degree with 50 levels and AD could
  exceed single-GPU memory even with checkpointing. May need gradient accumulation
  or model parallelism. Not yet profiled.
- **1000-day stability**: WENO5 validated for 100 days at 40x40. Longer integrations
  at higher resolution may reveal stability issues, especially without AB2 time stepping.

---

## 9. Success Criteria

### Minimum viable comparison (Phase 1-3)
- [x] Core WENO module with unit tests and AD Taylor tests
- [x] Leith closure working on lat-lon C-grid
- [ ] Energy/enstrophy spectra diagnostic
- [x] WENO7 tracer advection working
- [x] Baroclinic jet experiment running with centered + WENO5

### WENO-ILES implementation (Phase 4a-4b) — COMPLETE
- [x] Fix issue #160: total-velocity PV flux, Sadourny EC baseline
- [x] WENO Z term (vorticity flux): {ζ; v/u} smoothness-optimized, WENO5/7
- [x] WENO D term (divergence flux): {D; D} self-smoothness, WENO5
- [x] WENO K term (KE gradient): {u²; u}/{v²; v} cross-stencil, WENO5
- [x] WENO C term (vertical advection): WENO5/7, flux-form
- [x] AD validated through full Z+D+K+C path (Taylor tests + gradient finiteness)
- [x] Baroclinic jet stable at 40x40 for 100 days without explicit viscosity

### Full paper reproduction (Phase 4c-5) — remaining work
- [ ] QG Leith closure (Phase 4c) — "QG2" scheme
- [ ] OM4p25 combined Smagorinsky (Phase 4d) — "SM2" scheme
- [ ] UP3 flux-form momentum (Phase 4f) — "UP3" scheme
- [ ] Deformation radius diagnostic (Phase 1c) — needed for QG Leith + OM4p25
- [ ] Energy/enstrophy spectra diagnostic (Phase 1d) — needed for paper figures
- [ ] 2D turbulence experiment (Phase 3b) — needs FFT pressure solver
- [ ] AB2 time integrator (Phase 4e) — paper uses AB2 for baroclinic jet
- [ ] All 5 dissipation approaches (W9V, W9D, SM2, QG2, UP3)
- [ ] 2D turbulence test at 4 resolutions
- [ ] 3D baroclinic jet at 3 resolutions, 1000-day integrations
- [ ] Spectra matching paper's Figures 4, 5, 8, 9 qualitatively

### Differentiability milestones
- [x] `jax.grad` through WENO5-Z kernel (Taylor test, O(h^2) convergence)
- [x] `jax.grad` through WENO vorticity flux (Taylor test + gradient finiteness)
- [x] `jax.grad` through full rotational WENO momentum (Z+D+K+C, all 8 helpers)
- [ ] Memory profile: AD through WENO momentum < 2x baseline (with checkpoint)
- [ ] Float32 stability confirmed for Metal backend
