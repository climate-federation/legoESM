# legoESM Ocean Model Technical Reference

**Version**: 1.0 (April 2026)
**Scope**: Complete technical reference (Sections 1--17)
**Authors**: legoESM development team

This document is the primary technical reference for the ocean component of
legoESM, a JAX-native differentiable Earth system model. It is intended for
developers extending the ocean model, researchers coupling it with ML or data
assimilation workflows, and scientists interpreting simulation output. The
material covers continuous equations, their discrete approximation, and the
specific algorithmic choices made in the code -- with explicit references to
source files and functions throughout.

---

## 1. Introduction

### 1.1 Scope and Purpose

This document describes the mathematical formulation, numerical methods, and
software architecture of the legoESM ocean component. It covers:

- The continuous Boussinesq hydrostatic primitive equations and equation of state.
- The z-star vertical coordinate and its discrete implementation.
- Split-explicit barotropic/baroclinic time stepping.
- Spatial discretisation on four supported grids (lat-lon C-grid, cubed-sphere
  C-D grid, MPAS Voronoi mesh, spectral/Gaussian).
- Tracer advection schemes (upwind, TVD Van Leer, DST-3, PPM with FCT).
- Subgrid-scale physics: vertical mixing, lateral mixing, convection, surface
  forcing, bottom drag.
- Boundary conditions, conservation properties, and known limitations.

The document is *not* a user manual or experiment tutorial. For experiment
setup, see `docs/ocean_experiments_reference.md`. For the broader legoESM
architecture (atmosphere, land, sea ice, coupler), see
`docs/implementation_summary.md`.


### 1.2 Model Heritage and Design Philosophy

legoESM is not a port of MOM6, MITgcm, NEMO, or MPAS-Ocean. It is a clean
reimplementation of ocean primitive equation dynamics in JAX, designed from the
outset for three capabilities that traditional Fortran models lack:

1. **End-to-end differentiability.** Every operator -- from the equation of
   state through the barotropic solver to the physics parameterisations -- is
   compatible with `jax.grad` and `jax.jit`. This enables gradient-based
   parameter estimation, adjoint sensitivity analysis, and neural-network
   coupling without tangent-linear or adjoint model maintenance.

2. **Multi-grid modularity.** The same physical equations are discretised on
   four independent grid types. Grid-specific operators (gradient, divergence,
   curl, interpolation, halo exchange) are isolated in per-grid modules; the
   equation of state, vertical coordinate, physics parameterisations, and
   conservation fixers are shared. This "lego" architecture allows mixing grids
   across components (e.g., cubed-sphere atmosphere coupled to lat-lon ocean)
   and testing the same physics on different meshes.

3. **Hardware portability.** A single Python/JAX codebase runs on CPU, GPU
   (CUDA), TPU, and Apple Silicon (Metal) without source changes. MPI-parallel
   distributed execution is supported via `mpi4jax` with custom VJP rules for
   halo exchange.

The ocean component currently comprises approximately 25,000 lines of Python
across dynamics, physics, biogeochemistry, experiments, and initialisation
modules. The four dynamical cores share a common split-explicit time-stepping
framework with configurable barotropic substep counts, BEBT semi-implicit
pressure gradient, and cosine-filtered time averaging.

Where algorithmic choices are borrowed from existing models, the source is
cited: the Wright (1997) equation of state follows MOM6's implementation
(`MOM_EOS_Wright.F90`); the vector-invariant momentum formulation follows
Arakawa and Lamb (1977) and Sadourny (1975); the split-explicit barotropic
solver follows Hallberg (1997) and Higdon (2005); DST-3 advection follows
MITgcm scheme 33.


### 1.3 Notation and Conventions

The table below defines all symbols used throughout this document. Unless
otherwise noted, SI units are used. Code-level variable names are given in
the rightmost column where they differ from the mathematical symbol.

| Symbol | Description | Units | Typical Value | Code Name |
|--------|------------|-------|---------------|-----------|
| `eta` | Sea surface height | m | O(1) | `eta` |
| `u` | Zonal velocity (eastward) | m/s | O(0.1) | `u` |
| `v` | Meridional velocity (northward) | m/s | O(0.1) | `v` |
| `w` | Vertical velocity (upward positive) | m/s | O(10^-4) | `w` |
| `T` | Potential temperature | degC | -2 to 35 | `T` |
| `S` | Salinity | PSU | 0 to 42 | `S` |
| `rho` | In-situ density | kg/m^3 | 1020--1030 | `rho` |
| `rho_0` | Reference density | kg/m^3 | 1025.0 | `rho_0` |
| `p` | Pressure | Pa | 0--6x10^7 | `p` |
| `p'` | Pressure perturbation (baroclinic) | Pa | O(10^3) | `p_prime` |
| `f` | Coriolis parameter `2*Omega*sin(phi)` | 1/s | -1.5x10^-4 to 1.5x10^-4 | `f` |
| `g` | Gravitational acceleration | m/s^2 | 9.80616 | `g` |
| `H` | Bathymetry depth (positive downward) | m | 0--5500 | `H_bathy` |
| `H_max` | Maximum/reference ocean depth | m | 5500.0 | `H_max` |
| `J` | Dynamic z-star Jacobian `(eta+H)/H_max` | -- | ~1.0 | `jacobian` |
| `h_k` | Layer thickness at level k | m | 10--200 | `h_k` |
| `dz_k` | Reference layer thickness at level k | m | 10--200 | `dz_ref[k]` |
| `K_v` | Vertical tracer diffusivity | m^2/s | 10^-5 to 10^-4 | `K_v` |
| `K_h` | Horizontal tracer diffusivity | m^2/s | 0 to 10^3 | `K_h` |
| `A_v` | Vertical viscosity | m^2/s | 10^-4 to 10^-3 | `A_v` |
| `A_h` | Horizontal (harmonic) viscosity | m^2/s | 10^3 to 10^4 | `A_h` |
| `B_h` | Horizontal biharmonic viscosity | m^4/s | 0 to 10^10 | `B_h` |
| `Omega` | Earth rotation rate | rad/s | 7.292x10^-5 | `Omega` |
| `a` | Earth mean radius | m | 6.371229x10^6 | `R_earth` |
| `phi` | Latitude | rad | -pi/2 to pi/2 | `lat` |
| `lambda` | Longitude | rad | 0 to 2*pi | `lon` |
| `alpha_T` | Thermal expansion coefficient | 1/K | ~2x10^-4 | `alpha_T` |
| `beta_S` | Haline contraction coefficient | 1/PSU | ~7.4x10^-4 | `beta_S` |
| `N^2` | Brunt-Vaisala (buoyancy) frequency | 1/s^2 | 10^-7 to 10^-3 | `N2` |
| `zeta` | Relative vorticity | 1/s | O(10^-6) | `zeta` |
| `KE` | Kinetic energy per unit mass | m^2/s^2 | O(10^-2) | `KE` |
| `tau_x, tau_y` | Surface wind stress components | Pa | O(0.1) | `tau_x`, `tau_y` |

**Index and sign conventions:**

- **Vertical indexing.** `k = 0` is the surface (shallowest); `k = nlev - 1`
  is the bottom (deepest). Interface `k` sits above level `k`.
- **Vertical direction.** `z` is positive upward. Depths below sea level are
  negative values. Reference cell-center depths `z_full_ref` are negative;
  `z_half_ref[0] = 0` at the surface, `z_half_ref[nlev] = -H_max` at the
  bottom.
- **Layer thickness.** `dz_ref[k]` and `h_k` are positive quantities
  (physical thickness).
- **Bathymetry.** `H_bathy` is positive downward: a point with 4000 m of water
  has `H_bathy = 4000`.
- **Land mask.** `land_mask = 1` is ocean, `land_mask = 0` is land. (The name
  is historical; it is effectively an ocean mask.)
- **Density sign.** Density increases downward in a stably stratified water
  column.


### 1.4 How to Read This Document

The document is organised in a progression from continuous equations to discrete
implementation:

- **Section 2** (Governing Equations) presents the continuous Boussinesq
  hydrostatic primitive equations, the equation of state, and boundary
  conditions. Read this for the physics.
- **Section 3** (Vertical Coordinate) describes the z-star coordinate
  transformation, Jacobian, and layer thickness construction. Read this before
  any section involving vertical operators.
- **Section 4** (Horizontal Grids) describes the five supported grids.
- **Section 5** (Time Stepping) covers split-explicit barotropic/baroclinic
  decomposition and all solver features.
- **Sections 6--7** (Tracer Advection, Horizontal Mixing) detail the transport
  and subgrid-scale closure schemes.
- **Sections 8--10** (Vertical Mixing, Other Physics, Conservation) cover
  parameterizations and budget closure.
- **Sections 11--13** (Biogeochemistry, Initialization, Differentiability)
  cover additional components.
- **Sections 14--17** (Configuration, Validation, Limitations, References)
  provide reference material.

---

## 2. Governing Equations

### 2.1 Boussinesq Hydrostatic Primitive Equations

The ocean model solves the rotating Boussinesq hydrostatic primitive equations
in a free-surface formulation. This section presents the continuous equations;
their discrete counterparts are developed in later sections.


#### 2.1.1 Momentum Equations (Vector-Invariant Form)

The horizontal momentum equations are written in *vector-invariant* form:

> **du/dt = -(zeta + f) v - d(KE)/dx - (1/rho_0) dp'/dx + F_u** ... **(2.1a)**
>
> **dv/dt = +(zeta + f) u - d(KE)/dy - (1/rho_0) dp'/dy + F_v** ... **(2.1b)**

where `zeta = dv/dx - du/dy` is the vertical component of relative vorticity,
`f = 2 Omega sin(phi)` is the Coriolis parameter, `KE = (u^2 + v^2)/2` is the
kinetic energy per unit mass, `p'` is the baroclinic pressure perturbation
(Section 2.1.4), and `F_u`, `F_v` represent subgrid-scale forces (viscosity,
bottom drag, wind stress).

**Why vector-invariant form?** The identity `(u . nabla)u = nabla(KE) + zeta x u`
allows the nonlinear advection to be decomposed into a gradient of kinetic energy
and a vorticity flux. This decomposition has three advantages over the flux
form `d(uu)/dx + d(vu)/dy`:

1. **No metric terms.** On curvilinear grids (cubed-sphere, lat-lon), the flux
   form generates Christoffel-symbol metric source terms that are awkward to
   discretise consistently. The vector-invariant form eliminates them.

2. **Conservation control.** The vorticity-flux term `(zeta + f) x u` can be
   discretised to conserve energy, enstrophy, or potential enstrophy by
   choosing different interpolation stencils (Sadourny 1975; Arakawa and
   Lamb 1977). The kinetic energy gradient is naturally conservative.

3. **Natural split-explicit coupling.** The barotropic pressure gradient
   `g * grad(eta)` enters only through the depth-averaged momentum equation,
   while the baroclinic equations use only the perturbation pressure `p'`.
   This separation is clean in vector-invariant form.

In the code, the baroclinic momentum tendency is computed in:
- `ocean_pe_latlon_cgrid.py::latlon_cgrid_ocean_baroclinic_tendencies` (lat-lon
  C-grid), Sections 5--7b.
- `ocean_pe_cdgrid.py::ocean_baroclinic_tendencies_cdgrid` (cubed-sphere C-D
  grid), Sections 7--13.

The Coriolis term `f x u` is *not* part of the returned baroclinic tendency on
the C-grid. Instead, it is applied as a forward-backward (Matsuno) step in the
time-stepping function for unconditional stability of inertial oscillations,
which would amplify under forward Euler by a factor of `sqrt(1 + (f dt)^2)`
per step.


#### 2.1.2 Continuity Equation (Free-Surface Form)

Under the Boussinesq approximation, volume (not mass) is conserved. Integrating
the three-dimensional continuity equation `du/dx + dv/dy + dw/dz = 0` over the
full water column from `z = -H` to `z = eta` gives the free-surface equation:

> **d(eta)/dt + nabla_h . integral_{-H}^{eta} u dz = 0** ... **(2.2)**

In the discrete model, this takes the layer-summed form:

> **d(eta)/dt = -sum_k div_h(h_k u_k)** ... **(2.2')**

where `h_k = dz_k * J` is the dynamic layer thickness (Section 3.2) and
`div_h` is the horizontal divergence operator. Equation (2.2') is the
prognostic equation for sea surface height. In the split-explicit framework,
the barotropic solver substeps this equation at the fast gravity-wave CFL, while
the baroclinic step provides the slow forcing.

Source: `barotropic_latlon_cgrid.py` (lines 1--14) for the C-grid formulation.


#### 2.1.3 Tracer Equations

Tracers (potential temperature `T` and salinity `S`) are advected in flux form
to ensure conservation:

> **d(h T)/dt + nabla_h . (h u T) + d(w T)/dz = nabla_h . (K_h nabla_h T) + d/dz(K_v dT/dz) + Q_T** ... **(2.3)**

where `Q_T` represents source terms (surface heat flux, shortwave penetration,
restoring). The same equation applies to `S` with `Q_S` representing
freshwater-salinity coupling (virtual salt flux or real freshwater; see
`freshwater.py`).

The flux-form formulation `d(hT)/dt + ...` ensures that the global integral of
`h T` over the domain is exactly conserved (up to sources `Q_T`) in the
discrete model, provided the tracer advection scheme is consistent with the
continuity equation (2.2'). This consistency is maintained by using the
*same* velocity and layer thickness in both the tracer transport and the
continuity equation -- a requirement enforced by the Hallberg (1997) transport
averaging in the split-explicit framework.

Available tracer advection schemes (configured via `LatLonCGridOceanConfig.tracer_advection`):
- `"upwind"` -- First-order upwind. Maximum numerical diffusion.
- `"tvd"` -- Van Leer TVD (second-order, monotone). Default.
- `"ppm"` -- Piecewise Parabolic Method (third-order, unlimited).
- `"ppm_fct"` -- PPM with Zalesak flux-corrected transport (third-order, monotone).
- `"dst3"` -- Direct Space-Time 3rd-order with Sweby limiter.
- `"dst3_multidim"` -- DST-3 with multi-dimensional predictor correction.

Source: `ocean_pe_latlon_cgrid.py` (Section 9), `advection.py`, `vertical.py`.


#### 2.1.4 Hydrostatic Relation

The vertical momentum equation reduces to the hydrostatic balance:

> **dp/dz = -rho g** ... **(2.4)**

Integrating downward from the free surface:

> **p(z) = rho_0 g eta + integral_{z}^{0} rho(T, S, p') g dz'** ... **(2.4')**

The first term is the *barotropic* pressure, proportional to sea surface height.
The second term is the *baroclinic* pressure, driven by density variations.
In the split-explicit framework, only the baroclinic pressure gradient enters
the baroclinic momentum equations; the barotropic gradient `g grad(eta)` is
handled by the barotropic substeps.

The discrete hydrostatic integration is implemented in
`eos.py::compute_hydrostatic_pressure`. It computes pressure at cell centers
(midpoints of each layer) by accumulating from the top:

```
p_top[k] = rho_0 * g * eta + sum_{k'=0}^{k-1} rho[k'] * g * dz_actual[k']
p_center[k] = p_top[k] + 0.5 * rho[k] * g * dz_actual[k]
```

**Important implementation detail.** The baroclinic pressure gradient uses
a *reference* Jacobian `J = 1` (corresponding to `eta = 0`) rather than the
actual dynamic Jacobian. This avoids double-counting the free-surface
contribution, which is already handled by the barotropic solver's
`g grad(eta)` term. See issue #109 and the comments at Section 2 of
`ocean_pe_latlon_cgrid.py`.


#### 2.1.5 The Boussinesq Approximation

The Boussinesq approximation replaces density `rho` with the constant reference
density `rho_0` everywhere except in the buoyancy (pressure gradient) term.
Specifically:

1. **Inertial terms.** The mass per unit volume in `rho Du/Dt` is replaced by
   `rho_0`. This is why the pressure gradient appears as `(1/rho_0) grad(p')`.

2. **Continuity equation.** Since `rho` is effectively constant, the continuity
   equation becomes a *volume* conservation statement `div(u) = 0` rather than
   a mass conservation statement. The free-surface equation (2.2) is therefore
   a volume budget.

3. **Buoyancy term.** The full density `rho(T, S, p)` is retained in the
   hydrostatic integral (2.4'), which drives baroclinic pressure gradients.
   This is the only place where density variations matter dynamically.

**Implications.** The Boussinesq ocean conserves *volume* exactly but conserves
*mass* only approximately (with errors of order `(rho - rho_0)/rho_0 ~ 10^-3`).
Energy conservation is similarly approximate. For climate-scale applications,
the Boussinesq approximation introduces a systematic bias in global mean sea
level of order 1 cm/century, which is small compared to other model errors.
See Griffies and Greatbatch (2012) for a detailed analysis.

The reference density `rho_0 = 1025.0 kg/m^3` is defined in `eos.py` and
propagated to all config NamedTuples via their `rho_0` field (default value
1025.0 in `OceanConfig`, `LatLonCGridOceanConfig`, `MPASOceanConfig`, etc.).


### 2.2 Equation of State

The equation of state (EOS) closes the system by relating density to the
prognostic tracers (temperature, salinity) and the diagnostic pressure. legoESM
provides two EOS options: the nonlinear Wright (1997) polynomial and a
configurable linear approximation.


#### 2.2.1 Wright (1997) Nonlinear EOS

The default EOS is the Wright (1997) formulation, following the MOM6
implementation in `MOM_EOS_Wright.F90`. The density is computed as:

> **rho = (p + p0(T, S)) / (lambda(T, S) + alpha0(T, S) (p + p0(T, S)))** ... **(2.5)**

where the three coefficient functions are polynomials of temperature `T` [degC]
and salinity `S` [PSU]:

**Specific volume parameter:**

> **alpha0(T, S) = a0 + a1 T + a2 S** ... **(2.5a)**

**Pressure offset [Pa]:**

> **p0(T, S) = (b0 + b4 S) + T (b1 + T (b2 + b3 T) + b5 S)** ... **(2.5b)**

**Lambda [m^2/s^2]:**

> **lambda(T, S) = (c0 + c4 S) + T (c1 + T (c2 + c3 T) + c5 S)** ... **(2.5c)**

The polynomial coefficients, taken from MOM6, are:

| Coefficient | Value | Group |
|-------------|-------|-------|
| `a0` | 7.057924e-4 | alpha0 |
| `a1` | 3.480336e-7 | alpha0 |
| `a2` | -1.112733e-7 | alpha0 |
| `b0` | 5.790749e+8 | p0 |
| `b1` | 3.516535e+6 | p0 |
| `b2` | -4.002714e+4 | p0 |
| `b3` | 2.084372e+2 | p0 |
| `b4` | 5.944068e+5 | p0 |
| `b5` | -9.643486e+3 | p0 |
| `c0` | 1.704853e+5 | lambda |
| `c1` | 7.904722e+2 | lambda |
| `c2` | -7.984422 | lambda |
| `c3` | 5.140652e-2 | lambda |
| `c4` | -2.302158e+2 | lambda |
| `c5` | -3.079464 | lambda |

**Valid range.** The polynomial is nominally valid for `T` in [-2, 40] degC
and `S` in [0, 42] PSU. Inputs are not clipped: silent clipping would zero
gradients at the boundary and mask unphysical states from advection overshoots
or coupler bugs (see issue #165 in the codebase).

**Precision.** The intermediate polynomial evaluation is promoted to `float64`
to avoid precision loss from the large coefficients (e.g., `b0 ~ 5.79e8`). On
backends that lack `float64` (e.g., Metal), the promotion is a no-op. The
result is cast back to the input dtype. This dtype promotion is fully
differentiable in JAX.

**Reference:** Wright, D. G. (1997): "An Equation of State for Use in Ocean
Models: Ockham's Razor Revisited." *J. Atmos. Oceanic Tech.*, 14(3), 735--740.

Source: `eos.py::wright_eos` (lines 65--122).


#### 2.2.2 Linear EOS

For idealised experiments, a linear equation of state is available:

> **rho = rho_ref [1 - alpha_T (T - T_ref) + beta_S (S - S_ref)]** ... **(2.6)**

with configurable parameters:

| Parameter | Symbol | Default | Units |
|-----------|--------|---------|-------|
| Reference density | `rho_ref` | 1025.0 | kg/m^3 |
| Thermal expansion coefficient | `alpha_T` | 2.0e-4 | 1/K |
| Haline contraction coefficient | `beta_S` | 7.4e-4 | 1/PSU |
| Reference temperature | `T_ref` | 10.0 | degC |
| Reference salinity | `S_ref` | 35.0 | PSU |

The linear EOS accepts a pressure argument for API compatibility but ignores
it (density is independent of pressure). This is consistent with the
incompressibility implicit in the linear approximation.

Configuration is via `LinearEOSConfig` (a NamedTuple in `eos.py`). The EOS
selection is controlled by the `eos` field of the ocean config: `"wright"`
(default) or `"linear"`.

Source: `eos.py::linear_eos` (lines 218--247), `eos.py::LinearEOSConfig`
(lines 206--215).


#### 2.2.3 Derived Quantities

**Thermal expansion coefficient.** The thermal expansion coefficient is
defined as:

> **alpha = -(1/rho) d(rho)/dT** ... **(2.7a)**

In legoESM, `d(rho)/dT` is computed exactly via `jax.grad` applied to the
scalar Wright EOS function, then `vmap`-ed over the array. This avoids
hand-coding the polynomial derivative and guarantees consistency between the
EOS and its linearisation -- a property that is critical for adjoint
correctness.

Source: `eos.py::thermal_expansion_coeff` (lines 144--170).

**Haline contraction coefficient:**

> **beta = (1/rho) d(rho)/dS** ... **(2.7b)**

Computed identically via `jax.grad` with `argnums=1`.

Source: `eos.py::haline_contraction_coeff` (lines 173--199).

**Buoyancy frequency (Brunt-Vaisala frequency).** The buoyancy frequency is
defined at interior interfaces as:

> **N^2 = -(g / rho_0) d(rho)/dz** ... **(2.8)**

Discretised at interface `k + 1/2` (between levels `k` and `k+1`) as:

```
dz_interface = 0.5 * (dz_actual[k] + dz_actual[k+1])
N^2[k] = -(g / rho_0) * (rho[k] - rho[k+1]) / dz_interface
```

This returns `nlev - 1` values for `nlev` levels. Positive `N^2` indicates
stable stratification; negative `N^2` indicates gravitational instability
(subject to convective adjustment).

Source: `eos.py::compute_buoyancy_frequency` (lines 354--391).


#### 2.2.4 Density-Pressure Iteration

The hydrostatic pressure `p` depends on density `rho`, which in turn depends on
pressure through the nonlinear EOS (2.5). This circular dependence is resolved
by a fixed-point iteration:

1. **Initial guess.** Compute density at zero pressure:
   `rho^(0) = EOS(T, S, p=0)`.

2. **Iterate** (2 passes):
   - Compute hydrostatic pressure from the current density:
     `p^(n) = hydrostatic_integral(rho^(n-1), eta=0, J=1)`.
   - Recompute density at the new pressure:
     `rho^(n) = EOS(T, S, p^(n))`.

3. **Final pressure.** Compute a consistent hydrostatic pressure from the
   converged density: `p = hydrostatic_integral(rho^(2), eta=0, J=1)`.

Two iterations suffice for convergence to machine precision for the Wright
(1997) polynomial (the pressure dependence is weak: `d(rho)/dp ~ 4.5e-10
kg/(m^3 Pa)`, so the density correction from pressure is only ~0.5 kg/m^3
at 5000 m depth).

**Critical detail.** The iteration uses the *reference* Jacobian `J = 1` and
`eta = 0`, not the actual dynamic values. The barotropic solver independently
handles the free-surface pressure gradient `g grad(eta)`. Using the actual `J`
would double-count this contribution, creating a spatially varying pressure
even for uniform `T` and `S`. See issue #109.

Source: `eos.py::compute_ocean_rho` (lines 398--429) and the EOS blocks in
`ocean_pe_latlon_cgrid.py` (Section 2) and `ocean_pe_cdgrid.py` (Section 2).


### 2.3 Boundary Conditions

#### 2.3.1 Surface Boundary Conditions

**Kinematic condition.** The vertical velocity at the surface satisfies the
kinematic free-surface condition `w|_{z=eta} = D(eta)/Dt`. In the discrete
model, the diagnosed `w` at the surface interface (`w_half[..., 0]`) represents
`d(eta)/dt`, consistent with the z-star correction applied in
`diagnose_w_from_flux_div`.

**Wind stress.** Surface wind stress `(tau_x, tau_y)` enters the momentum
equations as a body force distributed over the top layer:

> **F_wind_u = tau_x / (rho_0 h_1)** ... **(2.9a)**
>
> **F_wind_v = tau_y / (rho_0 h_1)** ... **(2.9b)**

where `h_1` is the thickness of the surface layer. Wind stress is provided via
`OceanSurfaceForcing.tau_x` and `OceanSurfaceForcing.tau_y`.

**Heat flux.** The net surface heat flux `Q_net` (positive into the ocean)
enters the temperature equation as:

> **Q_T|_{surface} = Q_net / (rho_0 c_sw h_1)** ... **(2.9c)**

where `c_sw = 3994.0 J/(kg K)` is the specific heat of seawater (defined in
`eos.py`). Shortwave radiation can optionally penetrate below the surface layer
via a double-exponential profile (see `physics/shortwave_penetration.py`).

**Freshwater flux.** Net freshwater flux (precipitation minus evaporation plus
runoff, `E - P - R`) affects sea surface height and salinity. Two closure
options are available (configured via `freshwater_closure`):

- `"virtual_salt_flux"`: the freshwater mass flux modifies `eta`; an equivalent
  virtual salt flux `Q_S = -S_ref * (E-P-R) / rho_0` modifies salinity to
  represent dilution/concentration without actually adding water mass. This is
  the default and avoids the numerical difficulties of real freshwater forcing.
- `"real_freshwater"`: both `eta` and `S` are modified by the actual mass flux.

Source: `freshwater.py`, `physics/surface_forcing/`.


#### 2.3.2 Bottom Boundary Conditions

**No-normal-flow.** At the ocean bottom, the normal velocity vanishes:
`u . n|_{bottom} = 0`. In the z-star framework with a rigid bottom, this is
enforced by setting `w = 0` at the bottom interface (`w_half[..., nlev] = 0`).
The bottom-up integration in `diagnose_w_from_flux_div` starts from this zero
boundary condition.

**Bottom drag.** Two formulations are available:

- *Linear drag:* `F_drag = -r * u / dz_bot`, where `r` [m/s] is the linear
  drag coefficient (`bottom_drag_r` in the config) and `dz_bot` is the bottom
  layer thickness. This acts only at the deepest level.
- *Quadratic drag:* `F_drag = -C_d |u| u / dz_bot`, where `C_d` is
  dimensionless (see `physics/bottom_drag/quadratic.py`).

Both formulations act on the *total* velocity (not the perturbation), since
the ocean floor sees the full flow.

Source: `ocean_pe_latlon_cgrid.py` (Section 10, bottom drag block),
`physics/bottom_drag/`.


#### 2.3.3 Lateral Boundary Conditions

**Current implementation.** The ocean model uses binary land masks
(`land_mask = 1` for ocean, `0` for land) to define lateral boundaries. The
treatment is a three-step process:

1. **Neumann fill.** Before computing pressure gradients, land cells are filled
   with nearest-neighbor ocean values via 3 iterations of a diffusive fill
   (`_neumann_fill_cgrid` in `ocean_pe_latlon_cgrid.py`). This prevents large
   pressure jumps across land-ocean boundaries from producing spurious
   pressure gradient forces.

2. **Operator computation.** All operators (gradient, divergence, Laplacian,
   vorticity) are computed over the full domain including filled land cells.

3. **Post-hoc masking.** Tendencies are multiplied by the land mask after
   all computations to zero out land-cell contributions.

For the C-grid lat-lon formulation, face masks (`u_mask`, `v_mask`) are
derived from the cell-center mask: a face is wet only if *both* adjacent cells
are wet. This ensures no mass flux through land boundaries.

**Grid-specific boundary conditions:**
- *Lat-lon*: Periodic in longitude. Solid wall at poles (`v = 0` at `i = 0`
  and `i = n_lat`).
- *Cubed-sphere*: Halo exchange across the 6 panel boundaries.
- *MPAS*: Boundary cells defined by mesh connectivity; masked edges at
  land-ocean interfaces.

**Known limitations.** The Neumann-fill + post-hoc-masking approach is
*not mathematically rigorous*. It has several failure modes documented in
`docs/ocean_boundary_conditions_analysis.md`:

1. **Corner cell inconsistencies.** At diagonal coastlines, the 4-neighbor
   Neumann fill does not guarantee smooth values in all directions, leading to
   residual pressure gradient errors.

2. **Metric amplification.** Near the poles on the lat-lon grid, the
   `1/cos^2(phi)` metric factor amplifies tiny gradient errors created by
   the fill.

3. **No partial cells.** Unlike MITgcm's `hFac` system, legoESM does not
   support fractional cell volumes at topographic boundaries. The boundary
   is either fully wet or fully dry. This produces a staircase representation
   of sloping topography and prevents exact rest-state preservation (small
   but persistent `eta` drift of order 10^-6 m has been observed in rest-state
   tests).

4. **Ordering dependence.** The combination of Neumann fill, operator
   computation, and post-hoc masking is sensitive to the order of operations
   and does not guarantee consistent flux cancellation at boundaries.

A planned refactor (tracked in the boundary conditions analysis document) would
replace this approach with MITgcm-style `hFac` face-area masking that bakes
land boundaries directly into flux computations, eliminating the need for
Neumann fill and post-hoc masking.

Source: `ocean_pe_latlon_cgrid.py::_neumann_fill_cgrid` (lines 213--261),
`docs/ocean_boundary_conditions_analysis.md`.

---

## 3. Vertical Coordinate

### 3.1 Z-star Formulation

The ocean model uses a z-star (`z*`) vertical coordinate that accommodates
free-surface motion by stretching the entire water column proportionally.
The coordinate transformation is:

> **z*(z, eta, H) = H_max (z + H) / (eta + H)** ... **(3.1)**

where `z` is the physical depth (negative below sea level), `H` is the local
bathymetry depth (positive), `eta` is the sea surface height, and `H_max` is
the maximum ocean depth.

The key property of z-star is that *every* layer stretches or compresses
uniformly in response to changes in `eta`. Unlike a pure z-coordinate (where
only the surface layer changes thickness) or a sigma-coordinate (which
introduces pressure gradient errors over topography), z-star distributes the
free-surface displacement across all layers proportionally to their reference
thickness. This prevents thin surface layers from developing extreme aspect
ratios under large `eta` variations.

**Comparison with the atmosphere.** The atmosphere's z-star coordinate has a
*static* terrain-following Jacobian that depends only on orography. The ocean's
z-star Jacobian is *dynamic* -- it must be recomputed every timestep as `eta`
evolves. This is the fundamental difference between the atmospheric and oceanic
vertical coordinate implementations in legoESM.

Source: `vertical.py` (module docstring and lines 1--14).


### 3.2 Dynamic Jacobian

The Jacobian of the z-star coordinate transformation relates physical-space
volumes to coordinate-space volumes:

> **J(x, y, t) = (eta + H_bathy) / H_max** ... **(3.2)**

The Jacobian transforms reference layer thicknesses into actual (physical)
thicknesses:

> **h_k = dz_ref[k] * J** ... **(3.2')**

where `dz_ref[k]` is the reference thickness of layer `k` (at `eta = 0`, flat
bottom `H = H_max`). The total water column depth at any point is:

> **eta + H_bathy = J * H_max = sum_k h_k** ... **(3.2'')**

At rest (`eta = 0`, `H_bathy = H_max`), `J = 1` and `h_k = dz_ref[k]`.

**Bathymetry shallower than H_max.** When `H_bathy < H_max`, the resting
Jacobian is `J_rest = H_bathy / H_max < 1`, and all layers are thinner than
their reference values. The z-star coordinate does *not* mask out bottom layers;
instead, all `nlev` layers exist everywhere, but they become very thin over
shallow bathymetry. The minimum water column treatment (Section 3.6) prevents
the total column depth from becoming dangerously small.

**Optional minimum water column.** When `min_water_column_m` is set (default
0.5 m), the water column `eta + H_bathy` is clipped to this minimum before
computing `J`. This prevents division by near-zero in shallow regions.

Source: `vertical.py::compute_ocean_jacobian` (lines 164--196),
`vertical.py::compute_layer_thickness` (lines 128--161).


### 3.3 Layer Thickness Construction

The reference vertical grid is created by `create_ocean_z_star`, which builds
a stretched grid with fine resolution near the surface and coarse resolution at
depth. The algorithm is:

1. **Linearly growing layer thickness profile.** For `k = 0, 1, ..., nlev-1`:

   > **dz_raw[k] = dz_surface + k * (dz_deep - dz_surface) / max(nlev - 1, 1)** ... **(3.3)**

   This produces layers that grow linearly from `dz_surface` at the top to
   `dz_deep` at the bottom.

2. **Normalisation to H_max.** The raw thicknesses are scaled so their sum
   equals `H_max`:

   > **dz_ref[k] = dz_raw[k] * H_max / sum(dz_raw)** ... **(3.3')**

3. **Interface depths.** Computed as a cumulative sum from the surface:

   > **z_half[0] = 0** (surface)
   >
   > **z_half[k+1] = z_half[k] - dz_ref[k]** for k = 0, ..., nlev-1
   >
   > **z_half[nlev] = -H_max** (bottom)

4. **Cell-center depths.** Midpoints of adjacent interfaces:

   > **z_full[k] = 0.5 * (z_half[k] + z_half[k+1])** ... **(3.3'')**

5. **Inter-level spacing.** Distance between adjacent cell centers (used for
   vertical gradient computation):

   > **dz_half[k] = z_full[k] - z_full[k+1]** for k = 0, ..., nlev-2

   This produces `nlev - 1` values.

**Example.** For the default parameters (`nlev = 50`, `H_max = 5500 m`,
`dz_surface = 10 m`, `dz_deep = 200 m`): the surface layer is approximately
5.2 m thick (after normalisation), the bottom layer is approximately 104 m
thick, and the total depth sums to exactly 5500 m. The actual thicknesses
after normalisation differ from the raw values because the normalisation
factor `H_max / sum(dz_raw)` is approximately 1.05 for these parameters.

Source: `vertical.py::create_ocean_z_star` (lines 54--125).


### 3.4 Level Convention

The vertical indexing convention is:

```
    Interface 0     === Sea surface     z_half[0] = 0
    ------- Level 0 --- (shallowest)    z_full[0] ~ -dz_ref[0]/2
    Interface 1
    ------- Level 1 ---                 z_full[1]
    Interface 2
    ...
    Interface nlev-1
    ------- Level nlev-1 (deepest) ---  z_full[nlev-1]
    Interface nlev  === Ocean bottom    z_half[nlev] = -H_max
```

**Static arrays** (stored in `OceanZStarCoordinate`):

| Array | Shape | Description |
|-------|-------|-------------|
| `z_full_ref` | `(nlev,)` | Reference cell-center depths [m]. Negative. |
| `z_half_ref` | `(nlev+1,)` | Reference interface depths [m]. `z_half_ref[0] = 0`, `z_half_ref[nlev] = -H_max`. |
| `dz_ref` | `(nlev,)` | Reference layer thicknesses [m]. Positive. |
| `dz_half_ref` | `(nlev-1,)` | Distance between adjacent cell centers [m]. Positive. |

**Dynamic arrays** (recomputed every timestep):

| Array | Shape | Description |
|-------|-------|-------------|
| `J` (Jacobian) | `(n_lat, n_lon)` or `(6, n, n)` | `(eta + H_bathy) / H_max` |
| `h_k` | `(n_lat, n_lon, nlev)` or `(6, n, n, nlev)` | `dz_ref * J` |

Source: `vertical.py::OceanZStarCoordinate` (lines 23--51).


### 3.5 Vertical Velocity Diagnosis

The vertical velocity `w` is not a prognostic variable; it is diagnosed from
the horizontal flux divergence to satisfy the discrete continuity equation
exactly. The diagnosis proceeds bottom-up via cumulative summation.

**Continuous form.** From the continuity equation in z-star coordinates:

> **w(k-1/2) = w(k+1/2) - dz_k nabla_h . u_k + (dz_k / J) dJ/dt** ... **(3.4)**

The last term is the z-star correction accounting for the time-varying
coordinate.

**Discrete algorithm** (implemented in `diagnose_w_from_flux_div`):

1. Start from `w[nlev] = 0` at the bottom.

2. Integrate upward:

   > **w[k] = w[k+1] + div_h(h_k u_k)** ... **(3.4')**

   where `div_h(h_k u_k)` is the discrete horizontal mass flux divergence at
   level `k`. This is implemented as a reversed cumulative sum.

3. Apply the z-star sigma correction to enforce `w = 0` at the bottom:

   > **sigma[k] = (z_half[k] + H_max) / H_max** ... **(3.4'')**
   >
   > **w_corrected[k] = w[k] - sigma[k] * w[0]** ... **(3.4''')**

   Here `w[0]` is the surface value (which equals `d(eta)/dt`), and `sigma`
   is a linear profile from `sigma[0] = 1` (surface) to `sigma[nlev] = 0`
   (bottom). The correction redistributes the free-surface motion across all
   levels proportionally to `sigma`, ensuring that the diagnosed `w` is
   consistent with both the bottom boundary condition and the free-surface
   evolution.

**Thickness-weighted mode.** When `thickness_weighted=True`, the input
`flux_div_k` is assumed to already contain `div(h u)` (layer-thickness-weighted
flux divergence) and is not multiplied by `dz_ref`. This is the mode used by
the lat-lon C-grid formulation, where face-normal mass fluxes `h_u * u` are
computed explicitly.

Source: `vertical.py::diagnose_w_from_flux_div` (lines 245--296).


### 3.6 Minimum Water Column Treatment

When `eta` drops below `eta_floor = min_water_column_m - H_bathy`, the water
column depth `eta + H_bathy` approaches zero, causing numerical singularities
(division by near-zero in the Jacobian, vanishing layer thicknesses). The
ocean model applies two levels of protection:

1. **Jacobian clipping.** In `compute_ocean_jacobian` and
   `compute_layer_thickness`, the water column `eta + H_bathy` is clipped to
   `min_water_column_m` (default 0.5 m) before computing `J`. This is a
   pointwise, non-conservative operation that prevents immediate blow-up.

2. **Mass-conserving eta redistribution.** In the barotropic solver, after each
   substep, `clamp_and_redistribute` (in `eta_floor.py`) applies a globally
   conservative correction:

   a. Clamp `eta` to the floor: `eta_new = max(eta, eta_floor)`.

   b. Compute the total mass injected by the clamp:
      `mass_added = sum((eta_new - eta_old) * area * mask)`.

   c. Redistribute this mass over cells with headroom (where
      `eta > eta_floor + epsilon`): subtract `mass_added / area_headroom`
      uniformly from those cells.

   d. Repeat for `n_iter = 3` passes to handle cascading violations.

   This ensures that the global volume integral is preserved to machine
   precision, even when individual cells hit the floor. The redistribution
   is compatible with MPI via `global_sum_mpi`.

Source: `dynamics/eta_floor.py::clamp_and_redistribute` (lines 26--46),
`vertical.py::compute_ocean_jacobian` (lines 164--196).


### 3.7 Configuration Reference

The vertical coordinate is configured through `create_ocean_z_star` parameters
and the ocean config NamedTuple. The following table summarises the key
parameters:

| Parameter | Default | Range / Guidance | Source |
|-----------|---------|------------------|--------|
| `n_levels` | 50 | 10--100. Fewer levels save memory; more resolve the thermocline. 10 is minimum for reasonable baroclinic modes. | `create_ocean_z_star` |
| `H_max` | 5500.0 m | Must exceed the deepest bathymetry. Standard value for global ocean. | `create_ocean_z_star` |
| `dz_surface` | 10.0 m | 1--20 m. Controls surface-layer resolution. Smaller values resolve the mixed layer but require shorter timesteps for vertical CFL. | `create_ocean_z_star` |
| `dz_deep` | 200.0 m | 50--500 m. Controls deep-ocean resolution. Larger values save levels for the upper ocean. | `create_ocean_z_star` |
| `min_water_column_m` | 0.5 m | 0.1--10 m. Safety floor for `eta + H_bathy`. Too small risks numerical instability; too large wastes resolution in shallow regions. | `OceanConfig` et al. |

**Interaction with other parameters.** The ratio `dz_deep / dz_surface`
controls the stretching: a ratio of 20 (the default) produces strong
concentration of levels near the surface. The normalisation step ensures that
`H_max` is exactly spanned regardless of the ratio or `nlev`.

The number of barotropic substeps (`n_barotropic_substeps`, default 30) must
be chosen to resolve the external gravity wave at `c = sqrt(g H_max)`. For
`H_max = 5500 m`, `c ~ 232 m/s`. With a baroclinic timestep `dt` and grid
spacing `dx`, the substep count should satisfy
`n_substeps > c * dt / dx` to keep the barotropic CFL below 1.

---

## 4. Horizontal Grids

The ocean component supports five horizontal discretisations, each with distinct
mesh geometry, variable staggering, operator stencils, and trade-offs. This
section describes them in detail and connects the continuous equations from
Section 2 to their grid-specific discrete forms.

All grids share the same vertical coordinate (Section 3), equation of state
(`ocean/eos.py`), conservation fixers, and physics parameterisations. Grid-specific
code is confined to per-grid operator modules and dynamics files; the integration
layer selects the appropriate dynamics module at initialisation via the factory
function `create_ocean_component` in `driver/component_factory.py`.


### 4.1 Grid Overview and State Representations

The table below summarises the five grid types. "Maturity" reflects testing and
production readiness as of version 0.1.

| Grid | Staggering | Prognostic velocity | State container | Typical resolution | Maturity |
|------|-----------|--------------------|-----------------|--------------------|----------|
| **Cubed-sphere** | C-D grid (baroclinic), A-grid (barotropic) | `u, v` at cell centres `(6, n, n, nlev)` | `OceanState` | C16--C128 (1500--180 km) | Production |
| **Lat-lon C-grid** | C-grid throughout | `u` at lon faces `(n_lat, n_lon+1, nlev)`, `v` at lat faces `(n_lat+1, n_lon, nlev)` | `LatLonCGridOceanState` | 1--5 deg | Production |
| **MPAS Voronoi** | C-grid (TRiSK) throughout | Edge-normal `u` `(nEdges, nlev)` | `MPASOceanState` | 60--300 km (variable) | Production |
| **Spectral Gaussian** | Vorticity-divergence spectral; A-grid in grid space | Spectral coefficients `(n_sh, nlev)` | `SpectralOceanState` | T21--T85 | Experimental |
| **SFNO Neural Operator** | Grid-space Gaussian (learned) | Spectral state + SFNO weights | `SpectralOceanState` | T21--T85 | Experimental |


### 4.2 Cubed-Sphere C-D Grid

The cubed-sphere C-D grid discretisation follows the FV3 dynamical core design
of Lin (2004) and Putman and Lin (2007). The dynamics code lives in
`ocean/dynamics/ocean_pe_cdgrid.py`, with shared operators in
`core/operators_cdgrid.py`.

#### 4.2.1 Gnomonic projection and 6-face structure

The cubed-sphere grid tiles the sphere with six faces of a gnomonic (central)
projection. Each face is a regular `n x n` grid in computational coordinates,
giving `6 n^2` cells globally. The grid is non-orthogonal: the angle between the
two computational coordinate directions (stored as `cosa_corner` in the
`CubedSphereCDGrid` object) varies from 90 degrees at face centres to approximately
70 degrees at cube corners. All metric terms (cell area, edge lengths, angles)
are precomputed from 3D Cartesian geometry and stored in the `CubedSphereCDGrid`
NamedTuple.

#### 4.2.2 C-D grid staggering

The C-D grid uses two staggerings simultaneously for different purposes:

- **D-grid (corners):** Velocity components `u_d, v_d` are defined at cell
  corners, shape `(6, n+1, n+1, nlev)`. The D-grid is used for vorticity
  computation (circulation integral) and the Arakawa-Lamb gradient.

- **C-grid (edges):** Edge-normal velocities `u_c, v_c` are diagnosed from
  D-grid corners for mass and tracer transport. `u_c` lives on x-faces (constant
  i-index) with shape `(6, n+1, n, nlev)`, and `v_c` on y-faces (constant
  j-index) with shape `(6, n, n+1, nlev)`. The D-to-C conversion accounts for
  non-orthogonality via the function `dgrid_to_cgrid` in `operators_cdgrid.py`.

- **A-grid (cell centres):** The prognostic state container `OceanState` stores
  velocities at cell centres `(6, n, n, nlev)` for compatibility with the rest
  of the ocean infrastructure (barotropic solver, physics, conservation fixers,
  coupler). Conversion between A-grid and D-grid is performed by
  `center_to_dgrid_vector` and `dgrid_to_center_vector` in
  `operators_cdgrid.py`.

The barotropic solver operates on A-grid cell-centre data using
centered-difference gradient and divergence operators. This is a deliberate
trade-off: the baroclinic mode needs accurate vorticity dynamics (where the C-D
grid excels), while the barotropic mode primarily resolves fast gravity waves
over many substeps, where simpler operators with tunable diffusion suffice. See
`docs/ocean_grid_staggering.md` for the full rationale.

#### 4.2.3 D-grid vorticity

Relative vorticity is computed from the integral circulation around each cell,
exact on the D-grid and free of the Hollingsworth-Kallberg instability
(function: `dgrid_vorticity` in `operators_cdgrid.py`):

> **zeta_c = (1/A_c) * oint v . dl** ... **(4.1)**

where the line integral is taken CCW around cell `c` using D-grid corner
velocities projected onto the covariant basis of each edge:

> **zeta = (u_S dx_S + v_E dy_E - u_N dx_N - v_W dy_W) / A** ... **(4.2)**

where subscripts S, N, E, W denote south, north, east, west edges and edge
velocities are 2-point averages of the adjacent corner values.

#### 4.2.4 Arakawa-Lamb pressure gradient

The gradient of a cell-centre scalar `B` (kinetic energy or pressure) at D-grid
corners uses the 4-point Arakawa-Lamb stencil (function: `_arakawa_lamb_gradient`
in `operators_cdgrid.py`):

> **dB_raw_x = (B_SE + B_NE) - (B_SW + B_NW)** ... **(4.3a)**
>
> **dB_raw_y = (B_NW + B_NE) - (B_SW + B_SE)** ... **(4.3b)**

The raw differences are then transformed to the face-local coordinate system
using a precomputed 2x2 matrix derived from 3D Cartesian geometry:

> **dB/dx = c_00 dB_raw_x + c_01 dB_raw_y** ... **(4.4a)**
>
> **dB/dy_perp = c_10 dB_raw_x + c_11 dB_raw_y** ... **(4.4b)**

This Cartesian-based transformation eliminates the separate non-orthogonality
correction and gives correct gradients at face boundaries and cube vertices
where face-local metrics are inconsistent across faces.

#### 4.2.5 PPM transport

Mass and tracer transport uses the Piecewise Parabolic Method (Colella and
Woodward 1984) for 4th-order accuracy with monotonicity constraints. The
implementation in `operators_cdgrid.py` provides:

- `cgrid_mass_flux_divergence(h, u_c, v_c, cdgrid)` for layer-thickness flux
  divergence, requiring halo=2 data for the PPM stencil.
- `cgrid_tracer_advection_fct(q, u_c, v_c, cdgrid)` for monotone tracer
  transport using PPM with face-value clipping and Zalesak-style FCT.

The PPM reconstruction computes face values via 4th-order interpolation:

> **q_face = (7(q_i + q_{i+1}) - (q_{i-1} + q_{i+2})) / 12** ... **(4.5)**

followed by Colella-Woodward monotonicity constraints. The FCT scheme uses
`jax.custom_jvp` to ensure differentiability: the forward pass uses the full
limiter, while the JVP linearises through the unlimited PPM scheme.

#### 4.2.6 A-grid to D-grid conversion

**Cell-centre to D-grid corners** (`center_to_dgrid_vector`): Uses
`pad_halo_vector` for proper rotation of vector components across cubed-sphere
face boundaries, then 4-point averages to corners.

**D-grid corners to cell-centres** (`dgrid_to_center_vector`): Simple 4-point
average of the four surrounding corners.

**D-grid to C-grid edges** (`dgrid_to_cgrid`): Projects D-grid corner velocities
onto face-normal directions with non-orthogonality correction.

#### 4.2.7 Halo exchange across face boundaries

The cubed-sphere requires halo exchange to communicate data across the six
face boundaries. The exchange handles scalar fields (`pad_halo`, `pad_halo_4d`),
vector fields with rotation (`pad_halo_vector`), and variable halo widths
(halo=1 for standard operators, halo=2 for PPM).

Under MPI-parallel execution, halo exchange uses `mpi4jax.sendrecv` wrapped in
a `custom_vjp` (`_sendrecv_vjp` in `halo_exchange.py`) that swaps source and
destination in the backward pass, enabling `jax.grad` through MPI
communication. The 4D path exchanges all vertical levels in a single MPI
message.

#### 4.2.8 State representation

`OceanState` (defined in `ocean/state.py`):

| Field | Shape | Description |
|-------|-------|-------------|
| `u` | `(6, n, n, nlev)` | Zonal velocity at cell centres [m/s] |
| `v` | `(6, n, n, nlev)` | Meridional velocity at cell centres [m/s] |
| `T` | `(6, n, n, nlev)` | Potential temperature [degC] |
| `S` | `(6, n, n, nlev)` | Salinity [PSU] |
| `eta` | `(6, n, n)` | Sea surface height [m] |
| `H_bathy` | `(6, n, n)` | Bathymetry depth [m] (positive downward, static) |
| `land_mask` | `(6, n, n)` | Ocean mask (1=ocean, 0=land, static) |

**Known limitations:**
- The A-grid barotropic solver has a 2*dx checkerboard null space requiring
  explicit diffusion (`barotropic_diffusion_alpha`) to control.
- Face-boundary halo interpolation introduces `O(dx)` errors that the
  Arakawa-Lamb gradient amplifies; float64 pressure computation mitigates but
  does not eliminate this.
- Cube vertices are singular points of the gnomonic projection.


### 4.3 Lat-Lon C-Grid

The lat-lon C-grid discretisation is the most mature ocean pathway and uses
compact stencils on a regular latitude-longitude grid. The dynamics code lives
in `ocean/dynamics/ocean_pe_latlon_cgrid.py` and
`ocean/dynamics/ocean_model_latlon_cgrid.py`, with operators in
`ocean/dynamics/latlon_cgrid_operators.py`.

#### 4.3.1 Regular grid with compact stencils

The latitude-longitude grid is a regular mesh with `n_lat` cells meridionally
and `n_lon` cells zonally. Cell centres are at `(lat[i], lon[j])`, with uniform
spacing `dlat = pi/n_lat` and `dlon = 2*pi/n_lon`. The grid object `LatLonGrid`
stores cell-centre coordinates, edge lengths, cell areas
`A = R^2 cos(phi) dlat dlon`, and the Coriolis parameter `f = 2 Omega sin(phi)`.

The key advantage of C-grid staggering on the lat-lon grid is the use of
compact (single-cell) stencils for gradient and divergence, eliminating the
checkerboard null space of A-grid formulations.

#### 4.3.2 C-grid staggering

Variables are placed on an Arakawa C-grid:

- **u at longitude interfaces** (east/west cell faces): shape
  `(n_lat, n_lon+1, nlev)`. Face `j` sits between cell `j-1` (west) and cell
  `j` (east), with periodic wrapping `u[:, n_lon, :] = u[:, 0, :]`.

- **v at latitude interfaces** (north/south cell faces): shape
  `(n_lat+1, n_lon, nlev)`. Face `i` sits between cell `i-1` (south) and cell
  `i` (north). Pole faces (`i=0` and `i=n_lat`) have `v = 0` (solid wall).

- **Scalars (eta, T, S, H_bathy, land_mask) at cell centres**: shape
  `(n_lat, n_lon [, nlev])`.

#### 4.3.3 Divergence, gradient, Laplacian operators

All operators are defined in `ocean/dynamics/latlon_cgrid_operators.py`.

**Zonal gradient** (`gradient_x_cgrid`): At u-point (face `j`):

> **df/dx|_j = (f_j - f_{j-1}) / (R cos(phi) dlon)** ... **(4.6)**

**Meridional gradient** (`gradient_y_cgrid`): At v-point (face `i`):

> **df/dy|_i = (f_i - f_{i-1}) / (R dlat)** ... **(4.7)**

**Divergence** (`divergence_cgrid`): At cell centre `(i, j)`:

> **div = (1/A) [u_E dy_E - u_W dy_W + v_N dx_N - v_S dx_S]** ... **(4.8)**

**Laplacian** (`laplacian_cgrid`): Computed as `div(grad(f))` -- a compact
5-point Laplacian. The **vector Laplacian** (`vector_laplacian_cgrid`) uses the
Hodge decomposition `del^2 u = grad(div u) - curl(curl u)`.

#### 4.3.4 Boundary conditions

- **Longitude:** Periodic. `u[:, n_lon, :] = u[:, 0, :]`.
- **Latitude (poles):** Solid wall. `v = 0` at `i = 0` and `i = n_lat`.

#### 4.3.5 Face masks

The `LatLonCGridOceanState` carries three masks:

- `land_mask`: cell-centre ocean mask `(n_lat, n_lon)`, 1 = ocean.
- `u_mask`: ocean mask at u-points `(n_lat, n_lon+1)`. Wet only if both
  adjacent cells are wet.
- `v_mask`: ocean mask at v-points `(n_lat+1, n_lon)`. Wet only if both
  adjacent cells are wet.

**Critical constraint:** Replacing `land_mask` without updating `u_mask` and
`v_mask` creates stale face masks that allow mass flux through walls. Use
`replace_land_mask(state, new_mask)` from `init_latlon_cgrid.py`.

#### 4.3.6 Coriolis discretisation

The Coriolis term uses the Sadourny (1975) energy-conserving 4-point average:

> **(f v)_u = f_u * (1/4)(v_{i,j-1} + v_{i+1,j-1} + v_{i,j} + v_{i+1,j})** ... **(4.9a)**
>
> **(-f u)_v = -f_v * (1/4)(u_{i,j} + u_{i,j+1} + u_{i+1,j} + u_{i+1,j+1})** ... **(4.9b)**

Time stepping uses forward-backward (Matsuno) for unconditional inertial
stability. **Caveat (issue #160):** The current implementation uses only
perturbation velocity in the vorticity flux, missing cross-terms with the
barotropic velocity.

#### 4.3.7 State representation

`LatLonCGridOceanState` (defined in `ocean/state.py`):

| Field | Shape | Description |
|-------|-------|-------------|
| `u` | `(n_lat, n_lon+1, nlev)` | Zonal velocity at lon interfaces [m/s] |
| `v` | `(n_lat+1, n_lon, nlev)` | Meridional velocity at lat interfaces [m/s] |
| `T` | `(n_lat, n_lon, nlev)` | Potential temperature [degC] |
| `S` | `(n_lat, n_lon, nlev)` | Salinity [PSU] |
| `eta` | `(n_lat, n_lon)` | Sea surface height [m] |
| `H_bathy` | `(n_lat, n_lon)` | Bathymetry depth [m] (positive downward, static) |
| `land_mask` | `(n_lat, n_lon)` | Ocean mask at cell centres (static) |
| `u_mask` | `(n_lat, n_lon+1)` | Ocean mask at u-points (static) |
| `v_mask` | `(n_lat+1, n_lon)` | Ocean mask at v-points (static) |
| `w` | `(n_lat, n_lon, nlev)` | Vertical velocity [m/s] (diagnostic) |

**Strengths:** Compact stencils eliminate checkerboard modes. Simple structured
grid enables efficient vectorisation. Face masks handle irregular coastlines.
Most extensively tested pathway.

**Limitations:** Meridian convergence at poles restricts the time step. No
tripolar or bipolar pole treatment.


### 4.4 MPAS Voronoi (TRiSK)

The MPAS discretisation uses an unstructured centroidal Voronoi tessellation with
TRiSK mimetic operators (Ringler et al. 2010). Dynamics code is in
`ocean/dynamics/ocean_pe_mpas.py`, barotropic solver in
`ocean/dynamics/barotropic_mpas.py`, operators in `core/operators_voronoi.py`.

#### 4.4.1 Unstructured dual mesh

The mesh consists of three entity types:

- **Cells** (`nCells`): Voronoi polygons. Scalars (`eta, T, S`) live at cell
  centres. Each cell has area `areaCell`.
- **Edges** (`nEdges`): Segments connecting adjacent cell centres. The
  prognostic normal velocity `u` is defined at edge midpoints. Each edge has
  dual-mesh length `dvEdge` and primal-mesh length `dcEdge`.
- **Vertices** (`nVertices`): Intersections of Voronoi polygon edges. Vorticity
  is computed at vertices with area `areaTriangle`.

Connectivity arrays (`edgesOnCell`, `cellsOnEdge`, `verticesOnEdge`,
`edgesOnVertex`, etc.) and sign arrays enable gather-scatter operations in JAX.

#### 4.4.2 TRiSK mimetic operators

**Divergence at cells** (`divergence_cell`, Eq. 21 of Ringler et al. 2010):

> **div(c) = (1/A_c) sum_e n_{e,c} u_e l_e** ... **(4.10)**

**Gradient at edges** (`gradient_edge`, Eq. 22):

> **grad(e) = (phi_{c2} - phi_{c1}) / d_e** ... **(4.11)**

**Curl at vertices** (`curl_vertex`, Eq. 23):

> **zeta(v) = (1/A_v) sum_e t_{e,v} u_e d_e** ... **(4.12)**

**Tangential velocity** (`tangential_velocity`, Eq. 24):

> **v_t(e) = sum_{e'} w(e, e') u(e')** ... **(4.13)**

The Thuburn et al. (2009) weights `w(e, e')` satisfy `curl(v_t) = div(u)` on
the dual, essential for the energy-conserving property.

**Kinetic energy at cells** (`kinetic_energy_cell`, Eq. 63):

> **KE(c) = (1/(2 A_c)) sum_e (A_e / 2) u_e^2** ... **(4.14)**

#### 4.4.3 Energy and enstrophy conservation

Two PV-flux formulations are available:

**Energy-conserving** (`pv_flux_energy_conserving`, Eq. 49):

> **F_q(e) = sum_{e'} w(e, e') q_{e'} h_{e'} u(e')** ... **(4.15)**

Conserves total energy but not enstrophy.

**Enstrophy-conserving** (`pv_flux_enstrophy_conserving`, Eqs. 71-72):

> **F_q(e) = q_e * sum_{e'} w(e, e') h_{e'} u(e')** ... **(4.16)**

Conserves potential enstrophy but not total energy. Selected via
`config.pv_scheme` in `MPASOceanConfig`.

#### 4.4.4 State representation

`MPASOceanState`:

| Field | Shape | Description |
|-------|-------|-------------|
| `u` | `(nEdges, nlev)` | Edge-normal velocity [m/s] |
| `T` | `(nCells, nlev)` | Potential temperature [degC] |
| `S` | `(nCells, nlev)` | Salinity [PSU] |
| `eta` | `(nCells,)` | Sea surface height [m] |
| `w` | `(nCells, nlev+1)` | Vertical velocity [m/s] (diagnostic, half levels) |
| `H_bathy` | `(nCells,)` | Bathymetry depth [m] (positive downward, static) |
| `land_mask` | `(nCells,)` | Ocean mask (1=ocean, 0=land, static) |

**Strengths:** Consistent C-grid staggering throughout. TRiSK operators satisfy
exact mimetic identities. Unstructured mesh enables local refinement. No polar
singularity.

**Limitations:** Gather-scatter operations less cache-friendly. Edge-normal
velocity requires reconstruction for geographic (u,v) diagnostics.


### 4.5 Spectral Gaussian Grid

The spectral discretisation uses spherical harmonic transforms in a
vorticity-divergence formulation. The dynamics code lives in
`ocean/dynamics/spectral_ocean_pe.py`.

#### 4.5.1 Spherical harmonic transforms

Prognostic variables are stored as SH coefficients: `vor_hat, div_hat, T_hat,
S_hat` of shape `(n_sh, nlev)` (complex128) and `eta_hat` of shape `(n_sh,)`,
where `n_sh = (T+1)(T+2)/2` for truncation `T`.

The tendency computation follows a pseudospectral workflow:
1. Transform to grid space (`sh_synthesis`).
2. Apply land mask in grid space.
3. Compute nonlinear products (EOS, advection, pressure gradient, KE) in grid space.
4. Transform tendencies back to spectral space (`sh_analysis`).
5. Apply spectral hyperdiffusion.

The vorticity and divergence tendencies:

> **d(vor)/dt = -div((vor + f) v) + curl(F)** ... **(4.17a)**
>
> **d(div)/dt = +curl((vor + f) v) - lap(KE + p'/rho_0) + div(F)** ... **(4.17b)**

#### 4.5.2 Unsplit SSP-RK3 time stepping

Unlike finite-volume grids, the spectral ocean does **not** use split-explicit
barotropic/baroclinic time stepping. Instead, it uses SSP-RK3 that integrates
all prognostic equations simultaneously. The barotropic gravity-wave mode is
controlled through spectral hyperdiffusion rather than fast subcycling.

#### 4.5.3 Spectral hyperdiffusion for eta stability

> **d(eta_hat)/dt += -nu_eta * (-n(n+1)/a^2)^p * eta_hat** ... **(4.19)**

where `nu_eta = eta_hyperdiff_coeff` (default `2.5e18`), `p = hyperdiff_order`
(default 2, biharmonic). Without this, high-wavenumber barotropic modes exceed
the RK3 stability limit.

#### 4.5.4 Land masking limitations (Gibbs ringing)

A sharp land-ocean boundary generates Gibbs oscillations (~9% overshoot of the
jump) that cannot be eliminated by increasing resolution. Current mitigation:
grid-space masking before SH analysis, spectral hyperdiffusion, and post-step
conservation fixers. This limits the spectral pathway to idealised
configurations with simple or no coastlines.

#### 4.5.5 State representation

`SpectralOceanState`:

| Field | Shape | Description |
|-------|-------|-------------|
| `vor_hat` | `(n_sh, nlev)` | Vorticity SH coefficients (complex) |
| `div_hat` | `(n_sh, nlev)` | Divergence SH coefficients (complex) |
| `T_hat` | `(n_sh, nlev)` | Temperature SH coefficients (complex) |
| `S_hat` | `(n_sh, nlev)` | Salinity SH coefficients (complex) |
| `eta_hat` | `(n_sh,)` | SSH SH coefficients (complex) |
| `land_mask_grid` | `(n_lat, n_lon)` | Ocean mask in grid space (static) |

Spectral computations require `JAX_ENABLE_X64=1` (float64 and complex128).


### 4.6 SFNO Neural Operator (Experimental)

The Spherical Fourier Neural Operator pathway provides a learned dynamical core
as an alternative to traditional discretisations (`ocean/dynamics/sfno_ocean.py`).

The SFNO (Bonev et al. 2023) uses spherical harmonic transforms as the spectral
convolution layer, operating on the Gaussian grid. Two modes: `state_update`
(direct prediction) and `hybrid_tendencies` (SFNO provides tendencies integrated
with SSP-RK3). Channel packing via `pack_ocean_state` / `unpack_ocean_output`
from `ml/channel_packing.py`. Post-hoc conservation corrections for volume,
heat, and salt since the SFNO does not intrinsically conserve.


### 4.7 Grid Staggering Summary and Trade-offs

| Property | Cubed-sphere C-D | Lat-lon C-grid | MPAS TRiSK | Spectral |
|----------|-----------------|----------------|------------|----------|
| **Checkerboard suppression** | Baroclinic: yes. Barotropic: no (A-grid, needs diffusion) | Yes (compact stencils) | Yes (C-grid throughout) | N/A |
| **Energy conservation** | Partial (Arakawa-Lamb) | Not formally conserved (issue #160) | Energy-conserving option | Not formally conserved |
| **Enstrophy conservation** | No | No | Option available | No |
| **Land masking** | Cell mask; halo fill for PGF | Face masks (u_mask, v_mask) | Cell mask; edge masking via connectivity | Grid-space mask; Gibbs ringing |
| **Polar singularity** | None | Solid wall; CFL restriction | None | None |
| **Local refinement** | Uniform within face | Uniform only | Variable resolution | Uniform (truncation) |
| **Differentiability** | Full (including halo exchange) | Full (custom VJP for limiters) | Full (JAX scatter-gather) | Full (requires float64) |
| **Barotropic solver** | Split-explicit, A-grid | Split-explicit, C-grid (BEBT) | Split-explicit, C-grid | Unsplit SSP-RK3 |

**Which grid for which purpose:**

- **Lat-lon C-grid:** Best default for ocean-only experiments, especially with
  complex coastlines. Most mature pathway.
- **Cubed-sphere C-D grid:** Primary choice when coupling to cubed-sphere
  atmosphere.
- **MPAS Voronoi:** Best for variable-resolution and regional experiments,
  formal energy/enstrophy conservation.
- **Spectral:** Idealised experiments without coastlines, SFNO training
  baseline.
- **SFNO:** ML-accelerated ocean modelling research.

---

## 5. Time Stepping

The time-stepping algorithm is the computational heart of the legoESM ocean
model. Its primary challenge is the large separation between the speed of
barotropic gravity waves (phase speed `c = sqrt(gH)`, typically 100--230 m/s)
and the much slower baroclinic motions (internal waves at 1--3 m/s, advection at
0.1--1 m/s). The split-explicit method resolves this by subcycling the
barotropic mode at a small substep `dt_s = dt / N_s` while advancing the
baroclinic mode at the full `dt`.

The canonical implementation lives in `ocean_model_latlon_cgrid.py` (class
`LatLonCGridOceanModel`, method `step()`), with barotropic solvers in
`barotropic_latlon_cgrid.py`, `barotropic_mpas.py`, and `barotropic.py`.


### 5.1 Split-Explicit Barotropic/Baroclinic Decomposition

#### 5.1.1 Motivation

For `H = 5500 m`, `c ~ 232 m/s`. An unsplit explicit scheme at 10 km resolution
requires `dt < 40 s`, whereas baroclinic CFL allows `dt ~ 300--3600 s`. The
split-explicit method (Hallberg 1997, Higdon 2005, Shchepetkin and McWilliams
2005) isolates the barotropic mode into a 2D sub-problem subcycled at `dt_s`.

#### 5.1.2 Algorithm Overview

The full baroclinic time step proceeds in nine stages:

1. **Baroclinic tendencies**: Compute 3D tendencies from pressure gradient,
   advection, viscosity, diffusion, vertical mixing, surface forcing, sponge.
   Coriolis is *excluded* (Section 5.3).

2. **Tracer preliminary update**:

> **T* = T^n + dt * dT/dt, S* = S^n + dt * dS/dt** ... **(5.1)**

3. **Slow-forcing decomposition**: Depth-average the 3D momentum tendency for
   the barotropic solver:

> **F_slow = sum_k(h_k * du_k/dt) / sum_k(h_k)** ... **(5.2)**
>
> **(du_k/dt)' = du_k/dt - F_slow** (perturbation tendency)

4. **Forward-backward Coriolis** on perturbation velocity (Section 5.3).

5. **Pre-barotropic layer thickness** from current eta.

6. **Barotropic substeps**: N_s forward-backward substeps for (eta, U_bar,
   V_bar) with slow-forcing, BEBT, cosine filter (Section 5.4).

7. **Velocity reconciliation and flux-form tracer transport** (Sections 5.5, 5.6).

8. **Freshwater forcing**: Virtual salt flux to surface layer.

9. **Conservation fixers** (optional, Section 10).


### 5.2 Baroclinic Step

#### 5.2.1 Tendency Computation

`latlon_cgrid_ocean_baroclinic_tendencies()` returns `du_dt, dv_dt, dT_dt,
dS_dt` from: baroclinic pressure gradient (from rho' using reference-Jacobian
layer thicknesses), kinetic energy gradient, relative vorticity flux, horizontal
viscosity (Laplacian/biharmonic/Smagorinsky), vertical momentum advection,
vertical viscosity, tracer vertical advection/diffusion, surface forcing, sponge,
and physics parameterizations. Coriolis is *excluded* -- handled separately.

The free-surface pressure gradient `g * grad(eta)` is handled entirely by the
barotropic solver, avoiding double-counting.

#### 5.2.2 Slow-Forcing Coupling

The MOM6-style slow-forcing coupling passes the depth-averaged tendency
`F_slow` as a constant forcing at *every* barotropic substep, coupling the slow
physics to the evolving barotropic state:

> **F_slow_u = sum_k(h_{k,u} * du_k/dt) / sum_k(h_{k,u})** ... **(5.3)**

Without slow-forcing injection, the Eady instability experiment blew up at day 2
with `dt = 300 s`, whereas MOM6 runs the same case at `dt = 3600 s`
(`OCEAN_DEVELOPMENT_LOG.md`, 2026-04-19).


### 5.3 Forward-Backward Coriolis

Forward Euler Coriolis amplifies inertial oscillations by `sqrt(1 + (f dt)^2)`
per step -- at 80 degrees latitude with `dt = 300 s`, this produces e-folding
in ~13 hours. The Matsuno (forward-backward) scheme is unconditionally neutral:

> **u'^{n+1} = u'^n + dt * f_u * avg(v'^n)** ... **(5.5)**
>
> **v'^{n+1} = v'^n - dt * f_v * avg(u'^{n+1})** ... **(5.6)**

The amplification matrix has determinant exactly 1 for any `f dt`, with
eigenvalues on the unit circle. Applied to *perturbation* velocity only (not
barotropic); the barotropic mode has its own Coriolis in the substeps.

Source: `_forward_backward_coriolis_3d()` in `ocean_model_latlon_cgrid.py`.


### 5.4 Barotropic Substeps

The barotropic solver advances `(eta, U_bar, V_bar)` over `N_s` substeps of
size `dt_s = dt / N_s`. The governing equations:

> **d(eta)/dt = -div(H_total * U_bar) + F_slow_eta** ... **(5.8)**
>
> **d(U_bar)/dt = f * avg(V_bar) - g * d(eta)/dx + F_slow_u** ... **(5.9)**
>
> **d(V_bar)/dt = -f * avg(U_bar) - g * d(eta)/dy + F_slow_v** ... **(5.10)**

#### 5.4.1 Forward-Backward Substep Equations

**Step 1 -- Continuity (forward):**

> **eta^{n+1} = eta^n - dt_s * div(H_u U_bar^n, H_v V_bar^n) + dt_s * F_slow_eta** ... **(5.11)**

followed by mass-conserving eta-floor clamp.

**Step 2 -- BEBT pressure gradient:**

> **eta_pgf = (1 - beta) eta^{n+1} + beta eta^n** ... **(5.12)**

**Step 3 -- Forward-backward momentum:**

> **U_bar^{n+1} = U_bar^n + dt_s (f_u avg(V_bar^n) - g d(eta_pgf)/dx + F_slow_u)** ... **(5.14)**
>
> **V_bar^{n+1} = V_bar^n + dt_s (-f_v avg(U_bar^{n+1}) - g d(eta_pgf)/dy + F_slow_v)** ... **(5.15)**

Note Eq. 5.15 uses the *just-updated* `U_bar^{n+1}` (forward-backward Coriolis).

#### 5.4.2 BEBT Semi-Implicit Pressure Gradient

The `bebt` parameter (default 0.2) controls the implicit weighting:

- `bebt = 0`: Fully forward-backward (most accurate, strictest CFL).
- `bebt = 1`: Fully backward Euler (unconditionally stable, overdamps).
- `bebt = 0.2` (MOM6 default): Mild blending that relaxes the CFL by ~30-50%.

Named after MOM6's `BEBT` variable (Backward Euler BaroTropic), from Hallberg (1997).

#### 5.4.3 Cosine (Hanning) Time Filter

The cosine window for time-averaging:

> **w_i = 1 + cos(2 pi (i - N_s/2) / N_s)** ... **(5.16)**

has much smaller side lobes than box averaging, suppressing aliasing of
high-frequency barotropic modes. Applied to `eta, U_bar, V_bar` for baroclinic
coupling. **Transport** accumulators `H*U_bar` always use box filtering for
consistency with the continuity equation.

Config: `barotropic_time_filter` (default `"cosine"`; alternative `"box"`).

#### 5.4.4 Barotropic Diffusion on Eta

Flux-form Laplacian diffusion:

> **eta^{n+1} <- eta^{n+1} + div(nu_face * grad(eta^{n+1}))** ... **(5.18)**

where `nu_face = alpha * (dt_s/dt_ref) * avg(A_left, A_right)`. The flux-form
discretization ensures exact volume conservation. Config: `barotropic_diffusion_alpha`
(default 0.01).

#### 5.4.5 Divergence Damping on Barotropic Velocity

Targets the divergent mode while leaving geostrophic flow untouched:

> **U_bar^{n+1} <- U_bar^{n+1} + nu_div * A_u * d(div(U_bar))/dx** ... **(5.20)**

Config: `barotropic_div_damp` (default 0.0, typical 0.05 when enabled).

#### 5.4.6 Velocity Clipping

> **U_bar^{n+1} <- clip(U_bar^{n+1}, -v_max, v_max)** ... **(5.21)**

Config: `maxvel_barotropic` (default 0.0 = disabled; MOM6 uses 6.0 m/s). Non-conservative safety valve.

#### 5.4.7 Bottom Drag in Barotropic Substeps

> **U_bar^{n+1} <- U_bar^{n+1} * (1 - r dt_s / H_u)** ... **(5.22)**

Applied every substep for continuous damping. Config: `bottom_drag_r`.

#### 5.4.8 Differentiable Mode

- **`jax.lax.fori_loop`** (default): No intermediate storage. Not AD-compatible.
- **`jax.lax.scan`** (`differentiable_barotropic=True`): Stores all carries.
  Memory O(N_s) but fully differentiable. Required for training workflows.


### 5.5 Velocity Reconciliation

After barotropic substeps:

> **u_k^{n+1} = (u_k* - U_bar^n) + U_bar_avg** ... **(5.23)**

Ensures the depth-average matches the barotropic solution while preserving
baroclinic shear. A uniform correction ensures transport consistency:

> **delta_U = (H*U_bar_avg - sum_k h_k u_k) / sum_k h_k** ... **(5.24)**


### 5.6 Flux-Form Tracer Transport

Per-layer mass fluxes from corrected 3D velocity:

> **mf_u_k = h_{k,u}^old * u_k^corrected * u_mask** ... **(5.26)**

Vertical velocity diagnosed from flux divergence (bottom-up integration):

> **w_{k-1/2} = w_{k+1/2} + div(mf_k)** ... **(5.28)**

Flux-form tracer update:

> **h_k^new T_k^new = h_k^old T_k* - dt div(mf_k T_face) - dt (w T)_{k-1/2..k+1/2}** ... **(5.29)**

Conservation is exact to machine precision: horizontal flux divergence
telescopes, vertical flux telescopes. The face value `T_face` is computed by
the selected advection scheme (Section 6).


### 5.7 Spectral Time Stepping

The spectral ocean uses unsplit SSP-RK3:

> **s^(1) = s^n + dt F(s^n)**
> **s^(2) = 3/4 s^n + 1/4 [s^(1) + dt F(s^(1))]** ... **(5.30)**
> **s^{n+1} = 1/3 s^n + 2/3 [s^(2) + dt F(s^(2))]**

Barotropic gravity waves stabilized by spectral hyperdiffusion
(`eta_hyperdiff_coeff = 2.5e18`). No split-explicit subcycling.


### 5.8 CFL Considerations

**Barotropic CFL:**

> **N_s > sqrt(g H_max) * dt / dx_min** ... **(5.32)**

For Eady (`H = 5500 m`, `c = 232 m/s`, `dx = 10 km`, `dt = 300 s`):
`N_s > 7`. Default `N_s = 30` gives CFL ~ 0.23.

**Baroclinic CFL:** `|u|_max * dt / dx_min < 1`. Rarely binding except at
eddy-resolving resolution (`dx ~ 1 km`).

**Vertical CFL:** `|w|_max * dt / dz_min < 1`. Typically CFL_v ~ 0.003,
never binding.

| Resolution | c_baro | dt | N_s | CFL_baro | CFL_bc |
|-----------|--------|-----|-----|----------|--------|
| 1 deg (~100 km) | 230 m/s | 3600 s | 30 | 0.28 | 0.04 |
| 1/4 deg (~25 km) | 230 m/s | 900 s | 30 | 0.28 | 0.04 |
| 1/10 deg (~10 km) | 230 m/s | 300 s | 30 | 0.23 | 0.03 |


### 5.9 Grid-Specific Variations

**Lat-lon C-grid** (`barotropic_latlon_cgrid.py`): Compact C-grid stencils,
Sadourny Coriolis, full slow-forcing + BEBT + cosine filter + div damping.

**MPAS Voronoi** (`barotropic_mpas.py`): Normal-velocity on edges, Heun
predictor-corrector Coriolis, Neumann land fill, BEBT + cosine filter.

**Cubed-sphere A-grid** (`barotropic.py`): Collocated staggering requiring
Laplacian diffusion on both eta and velocity, Crank-Nicolson Coriolis. Does
not yet return barotropic-averaged transport (known limitation).

**References:**
- Hallberg, R. (1997). "Stable split time stepping schemes for large-scale
  ocean modeling." *J. Comput. Phys.*, 135(1), 54--65.
- Higdon, R. L. (2005). "A two-level time-stepping method for layered ocean
  circulation models." *J. Comput. Phys.*, 208(2), 629--648.
- Shchepetkin, A. F. and J. C. McWilliams (2005). "The regional oceanic
  modeling system (ROMS)." *Ocean Modelling*, 9(4), 347--404.

---

## 6. Tracer Advection

### 6.1 Overview and Scheme Selection

The continuous tracer equation from Section 2, equation (2.3), is:

> **d(h T)/dt + div_h(h u T) + d(w* T)/dz* = sources** ... **(6.1)**

The discrete flux-form update for each layer is:

> **h_k^{n+1} T_k^{n+1} = h_k^n T_k^n - dt * div_h(F^h_k) - dt * (F^v_{k-1/2} - F^v_{k+1/2})** ... **(6.2)**

Horizontal and vertical advection are treated separately, each with its own
scheme. The `tracer_advection` field in `LatLonCGridOceanConfig` selects among
six combinations:

| Config string | Horizontal scheme | Vertical scheme | Order | Monotone | Cost |
|---|---|---|---|---|---|
| `"upwind"` | 1st-order upwind | 1st-order upwind | 1 | Yes | Lowest |
| `"tvd"` (default) | TVD Van Leer | TVD Van Leer | 2 | Yes | Low |
| `"dst3"` | DST-3 Van Leer | DST-3 Van Leer | 3 | Yes | Medium |
| `"dst3_multidim"` | DST-3 multidim | DST-3 Van Leer | 3 | Yes | High |
| `"ppm"` | PPM (unlimited) | PPM (unlimited) | 4 | No* | Medium |
| `"ppm_fct"` | PPM + Zalesak FCT | PPM + Zalesak FCT | 4 | Yes | Highest |

*PPM without FCT does not guarantee global monotonicity; intended for testing.


### 6.2 First-Order Upwind

The simplest scheme assigns the face value to the upwind cell value:

> **T^u_j = T_{j-1} if (h u)_j > 0, else T_j** ... **(6.3)**
>
> **F^u_j = (h u)_j * T^u_j** ... **(6.4)**

**Properties:** Conservative, monotone. Maximum numerical diffusivity:
`K_num ~ |u| dx / 2`. At typical ocean parameters (`w = 1e-5 m/s`, `dz = 110 m`),
vertical `K_num ~ 5.5e-4 m^2/s` -- 55x the physical `K_v ~ 1e-5 m^2/s`.

Source: `ocean_pe_latlon_cgrid.py`, `vertical.py::flux_form_vertical_tracer_advection`.


### 6.3 TVD Van Leer (Second-Order)

Adds an anti-diffusive correction to upwind, limited by the Van Leer limiter:

> **psi(r) = (r + |r|) / (1 + |r|)** ... **(6.7)**

where `r` is the smoothness ratio (upwind gradient / local gradient).

**Horizontal TVD flux at u-face j** (positive flow, donor = cell j-1):

> **r+ = (T_{j-1} - T_{j-2}) / (T_j - T_{j-1})** ... **(6.8)**
>
> **T^u_j = T_{j-1} + (1/2) psi(r+) (T_j - T_{j-1})** ... **(6.9)**

**Vertical TVD flux** includes CFL-dependent decompression:

> **F_k = F_k^upwind + (1/2) |w_k| (1 - c_k) psi(r_k) delta_k** ... **(6.12)**

where `c_k = |w_k| dt / h_donor` is the local vertical CFL number.

**Properties:** Second-order in smooth regions. `K_num -> 0` in smooth regions.
Default scheme for lat-lon C-grid. Dramatically reduces spurious deep-ocean
warming vs. upwind (issue #209).

Source: `ocean_pe_latlon_cgrid.py`, `vertical.py::flux_form_vertical_tracer_advection_tvd`.


### 6.4 Piecewise Parabolic Method (PPM)

Colella and Woodward (1984) reconstruction achieving 4th-order accuracy:

**Step 1: Fourth-order edge values:**

> **T_hat_{i+1/2} = (7(T_i + T_{i+1}) - (T_{i-1} + T_{i+2})) / 12** ... **(6.13)**

**Step 2: Colella-Woodward monotonicity limiter** -- extremum flattening and
overshoot limiting on the parabola defined by `(T_L, T_bar, T_R)`.

**Step 3: Upwind face value selection** from the PPM reconstruction.

Source: `advection.py::ppm_to_u_points`, `advection.py::ppm_to_v_points`.


### 6.5 PPM with Flux-Corrected Transport (PPM-FCT)

Combines PPM accuracy with strict monotonicity via Zalesak (1979) limiting:

1. **Low-order flux** `F^L`: first-order upwind (monotone).
2. **High-order flux** `F^H`: PPM (fourth-order).
3. **Anti-diffusive flux:** `A = F^H - F^L`.
4. **Zalesak limiter:** scales `A` by `alpha in [0, 1]` ensuring the updated
   tracer stays within local bounds:

> **T_min = min(T_{i,j,k}, T_neighbors)** ... **(6.19)**
>
> **T_max = max(T_{i,j,k}, T_neighbors)** ... **(6.20)**

> **alpha = min(1, R+ / (A dt)) if A dt > 0; min(1, R- / (-A dt)) if A dt < 0** ... **(6.22)**

**Properties:** Fourth-order accuracy, strictly monotone, conservative. Most
expensive (two full advection passes). Recommended for long integrations.

Source: `advection.py::fct_tracer_advection`, `core/operators_cdgrid.py::cgrid_tracer_advection_fct`.


### 6.6 DST-3 (Direct Space-Time Third-Order)

MITgcm scheme 33 (Adcroft, Hill, and Marshall, 1997): third-order in space and
time simultaneously through CFL-dependent coefficients.

**Face value** (positive flow at face j+1/2, donor cell j):

> **T^face = T_j + psi(r) [d_0(c) (T_{j+1} - T_j) + d_1(c) (T_{j-1} - T_j)]** ... **(6.24)**

**CFL-dependent downstream coefficient** with stability cap:

> **d_0(c) = min((1-c)(4-2c)/6, (1-c)/2)** ... **(6.29)**

**Upwind-of-upwind coefficient:**

> **d_1(c) = (1-c)(1-2c)/6** ... **(6.30)**

The CFL dependence in `d_0` and `d_1` encodes temporal information, giving
formal third-order accuracy from a single forward Euler step without multi-stage
Runge-Kutta.

Source: `advection.py::dst3_to_u_points`, `advection.py::dst3_to_v_points`,
`advection.py::_dst3_d0`, `advection.py::_dst3_d1`.


### 6.7 DST-3 Multidimensional

Transverse correction to eliminate operator-splitting errors:

**Pass 1 (predictor):** Half-step update with upwind fluxes:

> **T* = T^n - (dt/2) div_h(F^upw) / h_k** ... **(6.31)**

**Pass 2 (corrector):** Full DST-3 fluxes using the corrected field `T*`.

Removes leading-order splitting error `O(dt u v d^2T/dxdy)` at the cost of
two advection passes.

Source: `advection.py::multidim_tracer_advection`.


### 6.8 Vertical Tracer Advection

All vertical schemes return the flux divergence `F_top[k] - F_bot[k]` with
units [tracer] x [m/s]:

| Scheme | Function | Order | K_num (vertical) | Activated by |
|---|---|---|---|---|
| Upwind | `flux_form_vertical_tracer_advection` | 1 | `|w| dz / 2` | `"upwind"` |
| TVD Van Leer | `flux_form_vertical_tracer_advection_tvd` | 2 | ~0 (smooth) | `"tvd"` |
| DST-3 | `flux_form_vertical_tracer_advection_dst3` | 3 | ~0 (smooth) | `"dst3"` |
| PPM | `flux_form_vertical_tracer_advection_ppm` | 4 | ~0 (smooth) | `"ppm"`, `"ppm_fct"` |

Switching from upwind to TVD reduces effective vertical numerical diffusion by
~55x -- the primary motivation for higher-order vertical advection (issue #209).


### 6.9 Momentum Advection

**Vertical momentum advection** uses first-order upwind exclusively
(`flux_form_vertical_momentum_advection` in `vertical.py`). The implicit
numerical viscosity `A_v^num ~ |w| dz / 2` provides essential shear damping.
Development experiments confirmed TVD momentum blew up at day 29, while upwind
momentum survived beyond day 76 (issue #209).

**Horizontal momentum advection** uses the vector-invariant form (Section 2),
not flux-form advection -- the KE gradient and vorticity flux handle nonlinear
advection.


### 6.10 Scheme Selection Guide

| Experiment type | Recommended | Rationale |
|---|---|---|
| Quick debugging | `"upwind"` | Cheapest, most stable |
| Standard ocean (default) | `"tvd"` | Good accuracy/cost balance |
| Eddy-resolving, long runs | `"ppm_fct"` | Best monotonicity, 4th-order |
| MITgcm-comparable | `"dst3"` | Matches MITgcm scheme 33 |
| Diagonal-flow-sensitive | `"dst3_multidim"` | Reduced splitting errors |
| Accuracy studies | `"ppm"` | Highest formal accuracy (test only) |

**References:**
- Colella, P. and P. R. Woodward (1984). "The Piecewise Parabolic Method (PPM)
  for gas-dynamical simulations." *J. Comput. Phys.*, 54(1), 174--201.
- Zalesak, S. T. (1979). "Fully multidimensional flux-corrected transport
  algorithms for fluids." *J. Comput. Phys.*, 31(3), 335--362.
- Adcroft, A., C. Hill, and J. Marshall (1997). "Representation of topography
  by shaved cells in a height coordinate ocean model." *MWR*, 125(9), 2293--2315.
- Van Leer, B. (1977). "Towards the ultimate conservative difference scheme.
  IV." *J. Comput. Phys.*, 23(3), 276--299.

---

## 7. Horizontal Mixing

Horizontal mixing parameterizes the effects of unresolved lateral turbulence on
momentum and tracers. legoESM provides four lateral mixing schemes: harmonic
(Laplacian), biharmonic (constant-coefficient), biharmonic Smagorinsky
(flow-dependent), and Gent-McWilliams/Redi isopycnal mixing.


### 7.1 Harmonic (Laplacian) Viscosity and Diffusion

**Momentum:**

> **du/dt|_hmix = A_h nabla^2 u** ... **(7.1)**

**Tracers:**

> **dT/dt|_hmix = K_h nabla^2 T** ... **(7.2)**

where `A_h` [m^2/s] is horizontal viscosity and `K_h` [m^2/s] is horizontal
tracer diffusivity.

On the lat-lon C-grid, momentum uses `vector_laplacian_cgrid()` (grad(div) -
curl(curl)); tracer diffusion uses `laplacian_cgrid()` (5-point compact div(grad)).
On MPAS, `vector_laplacian_del2_3d()` from TRiSK operators.

**Configuration:** `A_h = 1e4` (default), `K_h = 0.0` (default; rely on
advection scheme for implicit diffusion).

Source: `physics/lateral_mixing/harmonic.py`, `latlon_cgrid_operators.py`.


### 7.2 Biharmonic Viscosity (Constant Coefficient)

> **du/dt|_bih = -B_h nabla^4 u** ... **(7.3)**

Scale-selective: damps at rate `B_h k^4` vs. `A_h k^2` for harmonic. Strong at
grid scale, negligible at large scales. Preferred for eddy-permitting models.

**Pole scaling** (lat-lon grid): `alpha(phi) = (cos(phi) / cos(phi_max))^4` ...
**(7.4)** to maintain CFL at high latitudes where `dx -> 0`.

Source: `latlon_cgrid_operators.py::vector_bilaplacian_cgrid`,
`operators_voronoi.py::vector_laplacian_del4_3d`.


### 7.3 Biharmonic Smagorinsky (Flow-Dependent)

The Smagorinsky (1963) closure makes viscosity proportional to the local strain
rate:

**Strain rate decomposition:**

> **D_T = du/dx - dv/dy** (tension), **D_S = du/dy + dv/dx** (shearing) ... **(7.5)**
>
> **|D| = sqrt(D_T^2 + D_S^2)** ... **(7.6)**

**Smagorinsky viscosity coefficient:**

> **A_smag = (C_s Delta)^2 |D|** ... **(7.7)**

where `C_s` is the dimensionless Smagorinsky coefficient and
`Delta = sqrt(A_cell)`.

**Two-pass stress-tensor formulation (lat-lon):** Following MOM6:

> **du/dt|_smag = -S^T(A_smag * S(u))** ... **(7.8)**

where `S` is the discrete strain operator and `S^T` its exact discrete adjoint.
This guarantees the energy identity:

> **sum u . tend_u * A_face = -sum A_smag_h D_T^2 A_h - sum A_smag_q D_S^2 A_q <= 0** ... **(7.9)**

holding to machine precision.

**MPAS variant:** Sandwich form `tend = -del^2(A_smag del^2 u)` ... **(7.10)**
via `smagorinsky_biharmonic_3d()` in `operators_voronoi.py`.

**AD safety:** `sqrt(D_T^2 + D_S^2 + 1e-30)` prevents NaN gradients at
quiescent/masked points.

**Configuration:** `C_smag = 0.0` (default); typical 0.1--0.25.

Source: `latlon_cgrid_operators.py::smagorinsky_biharmonic_tendency_cgrid`,
`operators_voronoi.py::smagorinsky_biharmonic_3d`.


### 7.4 Biharmonic Tracer Diffusion

> **dT/dt|_bih = -K_bih nabla^4 T** ... **(7.11)**

Independent coefficient from momentum biharmonic. Default `K_bih = 0`; modern
practice relies on implicit diffusion from advection schemes.

Source: `physics/lateral_mixing/biharmonic.py`, `latlon_cgrid_operators.py::bilaplacian_cgrid`.


### 7.5 Gent-McWilliams / Redi Isopycnal Mixing

At coarse resolution, GM/Redi parameterizes mesoscale eddy effects: Redi (1982)
rotates diffusion to isopycnal surfaces; GM (Gent and McWilliams 1990) adds
bolus transport to flatten isopycnals.

#### 7.5.1 Redi isopycnal diffusion tensor

In the small-slope approximation:

> **F_Redi = kappa_R [[1, 0, S_x], [0, 1, S_y], [S_x, S_y, S_x^2+S_y^2]] nabla q** ... **(7.12)**

#### 7.5.2 GM bolus transport (skew-flux form, Griffies 1998)

> **F_GM = kappa_G [[0, 0, -S_x], [0, 0, -S_y], [S_x, S_y, 0]] nabla q** ... **(7.13)**

#### 7.5.3 Combined fluxes

**Horizontal:** `F_x = kappa_R dq/dx + (kappa_R - kappa_G) S_x dq/dz` ... **(7.14)**

**Vertical:** `F_z = (kappa_R + kappa_G)(S_x dq/dx + S_y dq/dy) + kappa_R |S|^2 dq/dz` ... **(7.15)**

When `kappa_G = kappa_R = kappa` (default), the horizontal off-diagonal terms
vanish and the scheme reduces to standard horizontal diffusion plus enhanced
vertical mixing proportional to isopycnal slope.

#### 7.5.4 Isopycnal slope computation

> **S_x = -(drho/dx) / (drho/dz), S_y = -(drho/dy) / (drho/dz)** ... **(7.16)**

Slopes computed at vertical interfaces. Vertical density gradient clamped to
`min(drho/dz, -eps)` to prevent division by zero.

#### 7.5.5 DM95 slope tapering

> **tau = (1/2)(1 + tanh((S_max - |S|) / (0.1 S_max)))** ... **(7.17)**

Smooth `tanh` transition avoids discontinuities (important for AD compatibility).

#### 7.5.6 Configuration

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| `kappa_GM` | 1000 | m^2/s | GM bolus transport coefficient |
| `kappa_Redi` | 1000 | m^2/s | Redi isopycnal diffusivity |
| `S_max` | 0.01 | -- | Maximum slope for DM95 tapering |

Available on lat-lon C-grid and cubed-sphere. Not yet on MPAS. Implemented but
not yet validated against benchmark experiments.

Source: `physics/lateral_mixing/gm_redi.py`, `physics/lateral_mixing/gm_redi_latlon.py`.


### 7.6 Summary Table

| Scheme | Coefficient | Units | Config field | Grid support | Typical values |
|--------|-------------|-------|-------------|--------------|----------------|
| Harmonic viscosity | A_h | m^2/s | `A_h` | All | 1e3--1e5 |
| Harmonic tracer | K_h | m^2/s | `K_h` | All | 0--1e3 |
| Biharmonic viscosity | B_h | m^4/s | `B_h` | Lat-lon, CS, MPAS | 1e10--1e11 |
| Biharmonic tracer | K_bih | m^4/s | `K_bih` | Lat-lon, CS | 0 |
| Smagorinsky biharmonic | C_s | -- | `C_smag` | Lat-lon, MPAS | 0.1--0.25 |
| GM thickness | kappa_G | m^2/s | `gm_redi.kappa_GM` | Lat-lon, CS | 500--2000 |
| Redi isopycnal | kappa_R | m^2/s | `gm_redi.kappa_Redi` | Lat-lon, CS | 500--2000 |

**References:**
- Smagorinsky, J. (1963). *Mon. Wea. Rev.*, 91, 99--164.
- Redi, M. H. (1982). *J. Phys. Oceanogr.*, 12, 1154--1158.
- Gent, P. R. and J. C. McWilliams (1990). *J. Phys. Oceanogr.*, 20, 150--155.
- Danabasoglu, G. and J. C. McWilliams (1995). *J. Climate*, 8, 2967--2987.
- Griffies, S. M. (1998). *J. Phys. Oceanogr.*, 28, 831--841.

---

## 8. Vertical Mixing

Three schemes are available via `VerticalMixingConfig.scheme`: constant, Richardson-number dependent, and KPP. All share the same underlying discretization (Eq. 8.2) with zero-flux boundary conditions at surface and bottom.

The vertical diffusion operator:

> **d(phi)/dt = d/dz(K(z) d(phi)/dz)** ... **(8.1)**

discretized at full levels using interface fluxes:

> **(d(phi)/dt)_k = (F_{k-1/2} - F_{k+1/2}) / dz_k, F_{k+1/2} = K_{k+1/2} (phi_k - phi_{k+1}) / dz_{k+1/2}** ... **(8.2)**

Source: `ocean/physics/mixing.py` (`vertical_diffusion`, `vertical_diffusion_variable_K`).


### 8.1 Constant Diffusivity

> **du/dt = d/dz(A_v d(u)/dz), dT/dt = d/dz(K_v d(T)/dz)** ... **(8.3)**

Defaults: `A_v = 1e-3 m^2/s`, `K_v = 1e-4 m^2/s`.

Source: `physics/vertical_mixing/constant.py`.


### 8.2 Richardson-Number Dependent Mixing

Following Pacanowski & Philander (1981):

> **K_v = K_0 / (1 + alpha Ri)^n + K_bg** ... **(8.4)**
>
> **A_v = K_v * Pr_t + A_bg** ... **(8.5)**

where Ri = N^2 / S^2 at each interface. Negative Ri (unstable) clipped to zero,
giving maximum mixing. Division by zero protected by machine epsilon.

Defaults: `K_0 = 5e-3`, `alpha = 5`, `n = 2`, `K_bg = 1e-5`, `A_bg = 1e-4`,
`Pr_t = 10`.

Source: `physics/vertical_mixing/richardson.py`.


### 8.3 K-Profile Parameterization (KPP)

Large, McWilliams & Doney (1994). Diagnoses a boundary-layer depth `h`, applies
enhanced diffusivity within it, and adds non-local transport for convective
conditions.

#### 8.3.1 Boundary layer depth

Diagnosed where the bulk Richardson number first exceeds `Ri_crit`:

> **Ri_b(z) = g Delta_rho(z) z / (rho_0 (Delta_V^2(z) + V_t^2(z)))** ... **(8.7)**

where `V_t^2` includes unresolved turbulent velocity from the previous step.
**Differentiable depth finding**: sigmoid-weighted average (sharpness s=20)
instead of non-differentiable `argmax`.

Source: `_boundary_layer_depth()` in `kpp.py`.

#### 8.3.2 Shape function and velocity scales

> **K_bl(z) = h w_s(sigma) G(sigma)** ... **(8.10)**

where `G(sigma) = sigma (1-sigma)^2` ... **(8.11)** and `w_s` depends on the
stability regime (stable, weakly unstable, strongly convective). Friction
velocity from wind stress: `u_* = sqrt(|tau|/rho_0)`.

#### 8.3.3 Non-local transport

For convective layers, counter-gradient flux:

> **(dT/dt)_NL = -d/dz[C_s Q_T^0 G(sigma)]** ... **(8.16)**

where `C_s = gamma_T = 6.33`.

#### 8.3.4 Interior mixing

Below the boundary layer, shear-instability and convective-instability mixing:

> **K_shear = K_0^shear [1 - (Ri/Ri_0)^2]^3 + K_bg** ... **(8.17)**

### 8.4 Configuration Reference

| Parameter | Symbol | Default | Units | Scheme |
|-----------|--------|---------|-------|--------|
| `A_v` | A_v | 1e-3 | m^2/s | Constant |
| `K_v` | K_v | 1e-4 | m^2/s | Constant |
| `K_0` | K_0 | 5e-3 | m^2/s | Richardson |
| `alpha` | alpha | 5.0 | -- | Richardson |
| `Ri_crit` | Ri_crit | 0.25 | -- | KPP |
| `K_max` | K_max | 1.0 | m^2/s | KPP |
| `gamma_T` | gamma_T | 6.33 | -- | KPP (non-local) |
| `K_conv` | K_conv | 1.0 | m^2/s | KPP (interior) |

Source: `physics/vertical_mixing/config.py`.

---

## 9. Other Physics Parameterizations

### 9.1 Bottom Drag

**Source**: `ocean/physics/bottom_drag/`

#### 9.1.1 Linear

> **du/dt|_drag = -r u_bot / dz_bot** ... **(9.1)**

where `r` [m/s] so that bottom stress `tau = rho_0 r u_bot` is
resolution-independent. Default: `r = 1.1e-3 m/s`.

#### 9.1.2 Quadratic

> **du/dt|_drag = -C_d |u_bot| u_bot / dz_bot** ... **(9.2)**

where `|u_bot| = sqrt(u^2 + v^2 + eps)` for AD safety. Default: `C_d = 2.5e-3`.

#### 9.1.3 Application

Tendency nonzero only at bottom level. Division by `dz_bot` converts stress
to acceleration (issue #198 fixed the units). In barotropic substeps, drag
divides by total column `H + eta` rather than bottom-layer thickness.


### 9.2 Surface Forcing

**Source**: `ocean/physics/surface_forcing/`

#### 9.2.1 Prescribed

Momentum: `du/dt|_{k=0} = tau / (rho_0 dz_0)` ... **(9.3)**

Heat: `dT/dt|_{k=0} = Q_net / (rho_0 c_sw dz_0)` ... **(9.4)**

Wind profiles available: `"constant"`, `"cosine_latitude"`, `"single_gyre"`,
`"double_gyre"`, `"double_gyre_sin2"`, `"double_gyre_tapered"`,
`"channel_sine"`, `"global_wind"`.

Source: `prescribed.py`, `wind_profiles.py`.

#### 9.2.2 Restoring

Newtonian relaxation:

> **dT/dt|_{k=0} = -(T_sfc - T*(phi)) / tau_T** ... **(9.6)**

Default: `tau_T = 30 days`, `T*` from `T_eq = 25` to `T_pole = 0` degC.
Combined mode applies prescribed wind + restoring T/S simultaneously.

Source: `restoring.py`.

#### 9.2.3 Bulk Formulas

Three sub-schemes: `"constant"` (fixed transfer coefficients), `"coare3"`
(COARE 3.0, Fairall et al. 2003), `"large_yeager"` (Large & Yeager 2004).

> **Q_sh = rho_a c_pa C_H U_a (T_s - T_a)** ... **(9.9)**
>
> **Q_lh = rho_a L_v C_E U_a (q_sat(T_s) - q_a)** ... **(9.10)**
>
> **tau = rho_a C_D |U_a| U_a** ... **(9.11)**

Source: `bulk_formulas.py`.

#### 9.2.4 Coupler-provided

When coupled to the legoESM atmosphere, surface fluxes bypass standalone
schemes and are computed by the coupler's air-sea flux module.


### 9.3 Shortwave Penetration

Paulson & Simpson (1977) two-band exponential absorption:

> **I(z) = Q_sw [R exp(z/zeta_1) + (1-R) exp(z/zeta_2)]** ... **(9.13)**

Default: Jerlov type II (`R = 0.77`, `zeta_1 = 1.5 m`, `zeta_2 = 14.0 m`).

Source: `physics/shortwave_penetration.py`.


### 9.4 Convection

**Enhanced diffusion** (`"enhanced_diffusion"`): Where `N^2 < 0`, apply large
`K_conv = 1.0 m^2/s`. Smooth sigmoid transition for AD compatibility.

**Entraining plume** (`"plume"`): Mass-flux plume descends from surface with
entrainment rate `eps = 1e-3 m^-1`. Implemented as `jax.lax.scan` over levels.

Source: `physics/convection/`.


### 9.5 Sponge Layers

> **d(phi)/dt|_sponge = gamma(x,y) [phi_ref - phi]** ... **(9.20)**

where `gamma` ramps quadratically: `gamma = (1/tau)(1 - d/w)^2` for `d < w`.

Source: `ocean/sponge.py`.


### 9.6 Freshwater Forcing

Net freshwater: `F_fw = P - E + R + M` ... **(9.22)**

**Virtual salt flux:** `dS/dt|_{k=0} = -S_ref F_fw / (rho_0 dz_0)` ... **(9.23)**

**Real freshwater mass:** `d(eta)/dt|_fw = F_fw / rho_0` ... **(9.24)**

Source: `ocean/freshwater.py`.

---

## 10. Conservation

### 10.1 What is Conserved

Three global integrals: volume (`integral eta dA`), heat
(`integral T h_k dA`), salt (`integral S h_k dA`). Numerical errors from time
splitting, advection truncation, and the z-star coordinate introduce small
violations each step.


### 10.2 Post-hoc Uniform Additive Fixers

**Volume:**

> **eta^new_fixed = eta^new + delta_eta * m** ... **(10.1)**
>
> **delta_eta = sum((eta^old - eta^new) A m) / sum(A m)**

**Heat:**

> **T^new_fixed = T^new + delta_T * m** ... **(10.3)**
>
> **delta_T = sum_k((T^old h^old - T^new h^new) A m) / sum_k(h^new A m)**

**Salt:** Identical structure to heat fixer.

All global reductions upcast to float64 for precision. MPI-aware via
`global_sum_mpi`.

Source: `ocean/conservation.py`.


### 10.3 Limitations

1. **Not locally conservative**: single global scalar applied uniformly.
2. **Spurious diapycnal mixing**: uniform `delta_T` mixes heat across density
   surfaces.
3. **Min water column interaction**: the floor breaks exact conservation.
4. **AD through floor**: `jnp.maximum` has zero gradient at the floor.


### 10.4 Configuration

| Field | Default | Description |
|-------|---------|-------------|
| `use_conservation_fixer` | False | Master switch |
| `fix_volume` | True | Uniform eta correction |
| `fix_heat` | True | Uniform T correction |
| `fix_salt` | True | Uniform S correction |
| `min_water_column_m` | 0.5 | Minimum column [m] |


### 10.5 Path Forward

Proper flux-form tracer advection with barotropic-baroclinic flux
reconciliation (Hallberg 1997, Higdon 2005) would guarantee local conservation
by construction, eliminating the need for post-hoc fixers. Tracked as issue #59.

---

## 11. Biogeochemistry

The biogeochemistry module (`ocean/biogeochemistry/`) provides two tracer schemes:
an abiotic carbon cycle (DIC + ALK) and a full NPZD ecosystem model. Both are
pure-JAX, differentiable, and JIT-compatible. Advection/diffusion uses the same
operators as T and S; biogeochemistry computes only source-sink tendencies.


### 11.1 Abiotic Carbon Cycle

Carries DIC [mol C m^-3] and ALK [mol eq m^-3]. Air-sea CO2 gas exchange at
the surface; no biological source-sink terms.

**Carbonate chemistry** (`carbonate.py`): Single-step analytic solver following
Follows et al. (2006). Equilibrium constants: K0 (Weiss 1974), K1/K2 (Lueker
et al. 2000), K_B (Dickson 1990).

**Air-sea CO2 flux** (`gas_exchange.py`): Wanninkhof (2014) gas transfer:

> **k_w = 0.251 U_10^2 (Sc/660)^{-1/2}** ... **(11.2)**
>
> **F_CO2 = k_w K_0 rho_sw (pCO2_atm - pCO2_ocean) * 1e-6** ... **(11.3)**

Applied to surface layer as `dDIC/dt = F_CO2 / dz_0`.


### 11.2 NPZD Ecosystem Model

Four biological tracers: NO3 (nitrate), P (phytoplankton), Z (zooplankton),
D (detritus), all in mol N m^-3. Fasham et al. (1990) / Oschlies & Garcon (1999).

**Light field:** Beer-Lambert with self-shading:

> **I(z) = I_0 exp(-integral_0^z [k_w + k_chl P(z')] dz')** ... **(11.4)**

**Growth:** Temperature, nutrient, and light limitation:

> **mu = mu_max Q_10^T min(N/(N+k_N), 1-exp(-alpha_P I/mu_max))** ... **(11.5)**

**Grazing:** Holling type II with quadratic prey dependence:

> **G = g_max P^2/(P^2 + k_P^2) Z** ... **(11.6)**

**Detritus sinking:** Upwind interface-flux at velocity `w_sink`.

**Stoichiometric coupling** to DIC/ALK via Redfield ratios (R_CN = 6.625).
CaCO3 rain ratio R_CaP = 0.07. Non-negativity via differentiable softplus.

### 11.3 Configuration

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| `scheme` | `"none"` | -- | `"none"`, `"abiotic"`, `"npzd"` |
| `pCO2_atm` | 400 | uatm | Atmospheric CO2 |
| `mu_max` | 1.5 | day^-1 | Max phytoplankton growth |
| `g_max` | 0.6 | day^-1 | Max grazing rate |
| `w_sink` | 10.0 | m/day | Detritus sinking speed |
| `remin_rate` | 0.05 | day^-1 | Remineralization rate |

Source: `biogeochemistry/config.py`, `npzd.py`, `carbonate.py`, `gas_exchange.py`.

---

## 12. Bathymetry and Initialization

### 12.1 Idealized Bathymetry

Flat bottom at `H_max` (default 5500 m) with land at `|phi| > 80 deg`.
`H_bathy = H_max` everywhere (including land) for smooth Jacobian. Flow
suppressed by `land_mask`.

Source: `init.py`, `init_latlon_cgrid.py`.

### 12.2 Realistic Bathymetry

Six-stage pipeline (`bathymetry.py`):

1. **Loading**: ETOPO/GEBCO NetCDF files.
2. **Regridding**: Bilinear interpolation to target grid.
3. **Ocean fraction**: Sub-grid sampling on `n_sub x n_sub` stencil.
4. **Depth bounds**: Cells shallower than `H_min` (10 m) become land.
5. **Smoothing**: Laplacian smoothing (default 2 passes).
6. **Strait enforcement**: Critical straits widened (Section 12.4).

### 12.3 Initial Conditions

**Rest state:** Zero velocity, zero eta, exponential temperature:

> **T(z) = T_deep + (T_surf - T_deep) exp(z/d_s)** ... **(12.1)**

where `d_s = 1000 m`, `T_surf = 20 degC`, `T_deep = 2 degC`. Salinity uniform
at 35 PSU.

**WOA18 climatology** (`init_woa.py`): World Ocean Atlas annual-mean T/S on
57 standard levels, regridded and vertically interpolated to model levels.
Analytical fallback: `T_surf = 28 cos^2(phi)` with 500 m e-folding.

### 12.4 Critical Straits

16 straits enforced when `enforce_straits = True`:

| Strait | Lat | Lon | Min width [km] | Min depth [m] |
|--------|-----|-----|-----------------|----------------|
| Drake Passage | -60 | -67 | 300 | 3000 |
| Gibraltar | 36 | -5.5 | 50 | 300 |
| Indonesian Throughflow | -3 | 120 | 200 | 1500 |
| Mozambique Channel | -17 | 41 | 200 | 2500 |
| Denmark Strait | 66 | -27 | 150 | 600 |
| Faroe Bank Channel | 61.5 | -8.5 | 100 | 800 |
| Bering Strait | 65.8 | -169 | 60 | 40 |
| Florida Strait | 25.5 | -79.5 | 100 | 800 |
| Luzon Strait | 20.5 | 121.5 | 100 | 2000 |
| *(+ 7 more)* | | | | |

Source: `bathymetry.py`.

---

## 13. Differentiability

End-to-end differentiability via `jax.grad` is a first-class design requirement.
All operators are pure JAX to preserve autodiff compatibility.


### 13.1 Design Principles

All state containers are `NamedTuple`s wrapping `Field` pytrees -- compatible
with `jax.grad`, `jax.jit`, `jax.vmap`, and `jax.lax.scan`. The
`build_segment_fn` pattern provides a `.raw` non-donating variant for use
inside `eqx.filter_value_and_grad` (buffer donation conflicts with reverse AD).


### 13.2 Custom VJPs

**Barotropic solver:** `differentiable_barotropic = True` uses `jax.lax.scan`
(stores O(N_s) intermediates); `False` (default) uses `jax.lax.fori_loop`
(O(1) memory, approximate backward pass).

**MPI halo exchange:** `_sendrecv_vjp` (`halo_exchange.py`) provides
`@jax.custom_vjp` that swaps source/dest in the backward pass, enabling
`jax.grad` through MPI communication.

**EOS derivatives:** `alpha` and `beta` computed via `jax.grad` of the scalar
Wright EOS function, vmapped over all grid points:

> **alpha(T,S,p) = -(1/rho) d(rho_Wright)/dT** ... **(13.1)**

Guarantees exact consistency with the EOS.


### 13.3 AD-Safe Numerics

**Sigmoid transitions:** Discrete switches replaced by smooth sigmoid:
- Convective adjustment: `K_v = K_bg + (K_conv - K_bg) sigmoid(-N^2 * s)` ... **(13.2)**
- KPP boundary layer: sigmoid-weighted depth average instead of `argmax`
- Plume convection: continuous activity via `lax.scan`

**Softplus for non-negativity:** Biogeochemical tracers use:

> **[c]+ = alpha ln(1 + exp(c/alpha))**, alpha = 1e-6 ... **(13.3)**

**Epsilon guards:** `sqrt(D_T^2 + D_S^2 + 1e-30)` in Smagorinsky,
`maximum(H, min_water_col)` before divisions, `clip(discriminant, 0)` before
sqrt in carbonate solver.

**EOS precision:** Float64 promotion for Wright polynomial intermediate
evaluation; `astype` calls are differentiable.


### 13.4 Gradient Checkpointing

Not currently automated. Memory scales as O(N_steps * N_substeps) under AD.
Recommended: differentiate single segments (6-24 hours) via `build_segment_fn`,
handle inter-segment checkpointing at training loop level.


### 13.5 Known Non-Differentiable Paths

| Operation | Location | Issue |
|-----------|----------|-------|
| Velocity clipping `clip(U, -maxvel, maxvel)` | `barotropic_latlon_cgrid.py` | Zero gradient when clipped |
| Eta floor `maximum(eta, eta_floor)` | `eta_floor.py` | Zero gradient when floored |
| `global_max_mpi`, `global_min_mpi` | `parallel/reductions.py` | Not AD-safe; diagnostics only |
| `fori_loop` barotropic | `barotropic_latlon_cgrid.py` | Approximate backward pass |
| Isolated basin fill (scipy) | `bathymetry.py` | NumPy; init time only |

Set `differentiable_barotropic = True` for any training/parameter-estimation
workflow. `global_sum_mpi` (allreduce SUM) has full VJP support and is safe in
loss functions.

---

## 14. Configuration Reference

All configuration is specified through immutable `NamedTuple` objects in
`state.py`, `mpas_config.py`, and `physics/combined.py`. Pytree-friendly for
`jax.jit`, `jax.grad`, and `lax.scan`.


### 14.1 LatLonCGridOceanConfig (Primary)

| Parameter | Default | Units | Description | Sec |
|---|---|---|---|---|
| `g` | 9.80616 | m/s^2 | Gravitational acceleration | 2.1 |
| `rho_0` | 1025.0 | kg/m^3 | Reference density | 2.1 |
| `A_h` | 1.0e4 | m^2/s | Harmonic viscosity | 7.1 |
| `B_h` | 0.0 | m^4/s | Biharmonic viscosity | 7.2 |
| `C_smag` | 0.0 | -- | Smagorinsky coefficient | 7.3 |
| `bottom_drag_r` | 0.0 | m/s | Linear bottom drag | 9.1 |
| `K_h` | 0.0 | m^2/s | Harmonic tracer diffusivity | 7.1 |
| `K_bih` | 0.0 | m^4/s | Biharmonic tracer diffusivity | 7.4 |
| `A_v` | 1.0e-3 | m^2/s | Vertical viscosity | 8.1 |
| `K_v` | 1.0e-4 | m^2/s | Vertical tracer diffusivity | 8.1 |
| `n_barotropic_substeps` | 30 | -- | Barotropic substeps per dt | 5.4 |
| `bebt` | 0.2 | -- | Semi-implicit barotropic PGF [0,1] | 5.4 |
| `barotropic_time_filter` | `"cosine"` | -- | `"box"` or `"cosine"` | 5.4 |
| `barotropic_diffusion_alpha` | 0.01 | -- | Laplacian damping on eta | 5.4 |
| `barotropic_div_damp` | 0.0 | -- | Divergence damping on U_bar | 5.4 |
| `maxvel_barotropic` | 0.0 | m/s | Velocity clipping (0=off) | 5.4 |
| `tracer_advection` | `"tvd"` | -- | `"upwind"/"tvd"/"ppm"/"ppm_fct"/"dst3"/"dst3_multidim"` | 6 |
| `eos` | `"wright"` | -- | `"wright"` or `"linear"` | 2.2 |
| `freshwater_closure` | `"virtual_salt_flux"` | -- | `"virtual_salt_flux"` or `"none"` | 9.6 |
| `S_ref` | 35.0 | PSU | Reference salinity for VSF | 9.6 |
| `use_conservation_fixer` | False | -- | Enable global fixers | 10 |
| `differentiable_barotropic` | False | -- | lax.scan (True) vs fori_loop | 13.2 |
| `min_water_column_m` | 0.5 | m | Minimum water column | 3.6 |
| `gm_redi` | None | -- | `GMRediConfig` or None | 7.5 |
| `physics` | None | -- | `OceanPhysicsConfig` or None | 14.5 |

Source: `state.py`.

### 14.2 OceanConfig (Cubed-Sphere)

Shares most parameters with lat-lon. Key differences:

| Parameter | Default | Notes |
|---|---|---|
| `barotropic_diffusion_alpha` | 0.05 | Higher to suppress face-boundary instability |
| `barotropic_staggering` | `"a_grid"` | `"a_grid"` or `"c_grid"` |
| `div_damp_2` | 0.0 | 2nd-order divergence damping |
| `div_damp_4` | 0.0 | 4th-order divergence damping |

Missing from cubed-sphere: `B_h`, `C_smag`, `bebt`, `tracer_advection`,
`gm_redi` (less mature pathway).

### 14.3 MPASOceanConfig

MPAS-specific additions:

| Parameter | Default | Notes |
|---|---|---|
| `pv_scheme` | `"energy"` | `"energy"` or `"enstrophy"` PV flux |
| `semi_implicit_coriolis` | True | Prevents Coriolis double-counting |
| `barotropic_damping` | 0.0 | Rayleigh damping (1/s) |

Full parameter set mirrors lat-lon with all barotropic controls (bebt, cosine
filter, div damp, maxvel). Source: `mpas_config.py`.

### 14.4 SpectralOceanConfig

Spectral-specific:

| Parameter | Default | Notes |
|---|---|---|
| `hyperdiff_coeff` | 1.0e15 | Spectral hyperdiffusion |
| `hyperdiff_order` | 2 | Biharmonic |
| `eta_hyperdiff_coeff` | 2.5e18 | SSH stabilization (unsplit RK3) |
| `time_integrator` | `"ssp_rk3"` | No split-explicit; SSP-RK3 |

Source: `state.py`.

### 14.5 OceanPhysicsConfig Sub-Configs

`OceanPhysicsConfig` (`physics/combined.py`) bundles six sub-configurations:

| Module | Config | Schemes | Default | Sec |
|---|---|---|---|---|
| Vertical mixing | `VerticalMixingConfig` | constant, richardson, kpp, none | constant | 8 |
| Lateral mixing | `LateralMixingConfig` | harmonic, biharmonic, gm_redi, none | harmonic | 7 |
| Surface forcing | `SurfaceForcingConfig` | prescribed, restoring, combined, bulk_formulas, none | none | 9.2 |
| Bottom drag | `BottomDragConfig` | linear, quadratic, none | none | 9.1 |
| Convection | `OceanConvectionConfig` | enhanced_diffusion, plume, none | none | 9.4 |
| Shortwave | `ShortwavePenetrationConfig` | Jerlov I/IA/IB/II/III | II | 9.3 |


### 14.6 Common Configuration Recipes

| Experiment | tracer_adv | Viscosity | Drag | Forcing | EOS |
|---|---|---|---|---|---|
| Rest state | `"tvd"` | A_h=1e4 | none | none | wright |
| Wind-driven gyre | `"tvd"` | A_h=5e5 | r=1e-4 | prescribed | wright |
| Baroclinic gyre | `"tvd"` | B_h=1e11, C_smag=0.1 | r=1e-4 | combined | wright |
| Eady instability | `"tvd"` | C_smag=0.2 | r=1e-3 | none | wright |
| ACC channel | `"tvd"` | A_h=1e3, B_h=1e10 | r=1e-3 | prescribed sine | linear |
| Lock exchange | `"upwind"` | A_h=0 | none | none | linear |
| AD training | `"tvd"` | per expt | per expt | per expt | wright + `differentiable_barotropic=True` |

---

## 15. Validation Experiments

All experiments reside in `src/legoesm/ocean/experiments/`. Full details in
`docs/ocean_experiments_reference.md`. The test matrix runner is
`scripts/run_ocean_test_matrix.py`.

### 15.1 Experiment Summary

| # | Experiment | Grids | Physics tested | Key metric |
|---|---|---|---|---|
| 1 | Rest state (no land) | CS, LL, MPAS, Sp | PGF, EOS, time stepping | eta drift < 1e-10 |
| 2 | Rest state (with land) | CS, LL, MPAS, Sp | Land masking, BCs | eta drift < 1e-10 (FV) |
| 3 | Barotropic wave | CS, LL, MPAS | Gravity wave dispersion | Amplitude 0.8-1.2x |
| 4 | Wind-driven gyre | CS, LL | Sverdrup balance, WBC | Max speed 0.05-0.5 m/s |
| 5 | Baroclinic adjustment | CS, LL, MPAS, Sp | Thermal wind balance | T drift < 1e-3 |
| 6 | Phillips two-layer | CS, LL, MPAS, Sp | Baroclinic instability | eta growth 0.8-10x |
| 7 | Inertia-gravity wave | CS, LL, MPAS, Sp | Poincare wave (analytical) | L2 error < 0.1 |
| 8 | Lock exchange | CS, LL | Gravity currents | RPE drift < 1% |
| 9 | Overflow | CS, LL | Topographic gravity current | RPE drift < 1% |
| 10 | Stommel gyre tracer | CS, LL | Passive tracer conservation | Integral drift < 0.1% |
| 11 | Geostrophic adjustment | CS, LL, MPAS | Rossby adjustment | Drift vs baseline |
| 12 | Global barotropic wind | CS, LL | Subtropical/subpolar gyres | WBC formation |
| 13 | Eady instability | LL | Eddy generation | EKE growth rate |
| 14 | ACC channel | LL | Topography, sponge, 1yr | Zonal transport |
| 15 | Baroclinic gyre | LL, MPAS | Wind + thermal, MOC | Heat transport |

Grid abbreviations: CS=cubed-sphere, LL=lat-lon C-grid, MPAS=Voronoi, Sp=spectral.

### 15.2 Grid Coverage

| Experiment | CS | LL | MPAS | Sp |
|---|---|---|---|---|
| rest_state | yes | yes | yes | yes (land issues) |
| barotropic_wave | yes | yes | yes | skipped |
| wind-driven gyre | yes | yes | no (#55) | no |
| baroclinic | yes | yes | yes | yes (no land) |
| inertia_gravity_wave | yes | yes | yes | yes |
| lock_exchange | yes | yes | no | no |
| eady/acc/baroclinic_gyre | no | yes | partial | no |

### 15.3 Experiment Hierarchy

Ordered simplest to most complex — if an earlier test fails, later ones are
unlikely to succeed:

1. rest_state_no_land → 2. rest_state → 3. inertia_gravity_wave →
4. barotropic_wave → 5. geostrophic_adjustment → 6. baroclinic →
7. phillips_two_layer → 8. wind-driven gyres → 9. lock_exchange/overflow →
10. stommel_gyre_tracer → 11. eady_instability → 12. baroclinic_gyre →
13. acc_channel

See `docs/ocean_experiments_reference.md` for full setup details.

---

## 16. Known Limitations and Open Issues

### 16.1 Spectral Land Masking (Gibbs Ringing)

Gibbs oscillations at land-ocean boundaries produce SSH drift O(0.3 m) and
land leakage O(0.6 m). Most spectral experiments run with no land. Issue #99.

### 16.2 Boundary Conditions: Post-Hoc Masking

Binary land mask applied after tendency computation rather than partial-cell
(hFac) methods. Staircase coastline approximation; reduced accuracy near coasts.

### 16.3 Missing Vorticity Cross-Terms (Issue #153, #160)

Lat-lon C-grid omits nonlinear vorticity advection (`zeta * u`). Coriolis
applied to perturbation velocity only, missing barotropic-baroclinic cross-terms.

### 16.4 MPAS Surface Forcing Gap (Issue #55, #113)

Only prescribed wind + linear drag on MPAS. Temperature restoring, bulk
formulas, shortwave penetration not wired.

### 16.5 Vertical Momentum: Forced Upwind

First-order upwind only. Higher-order requires implicit viscosity solver
(issue #204). MPAS missing vertical momentum advection entirely (#152).

### 16.6 Conservation Fixers Not Locally Conservative

Global uniform additive corrections. Not flux-form, introduces spurious
diapycnal mixing. Disabled by default. Issue #101.

### 16.7 No ALE Remapping

Z-star only. No isopycnal, sigma, or hybrid coordinates. No adaptive vertical
refinement.

### 16.8 No Tidal Forcing

No astronomical tides, boundary tides, or SAL. Standard for climate-scale
simulations; limiting for coastal/regional applications.

### 16.9 No Sea-Ice Coupling from Ocean

Ocean freezing point defined (`T_freeze_ocean = 271.35 K`) but no
freezing/melting logic acts on the ocean state.

### 16.10 KPP Untested in Production

Implemented with full LMD94 algorithm but not validated in long integrations.
Default is constant-coefficient mixing.

### 16.11 Cubed-Sphere Instabilities

- Face-boundary exponential instability (issue #100)
- A-grid barotropic checkerboard (issue #182)

---

## 17. References

### Governing Equations and Coordinates

- Griffies, S. M. (2004): *Fundamentals of Ocean Climate Models*. Princeton UP.
- Campin, J.-M., A. Adcroft, C. Hill, J. Marshall (2004): Conservation of
  properties in a free-surface model. *Ocean Modelling*, 6, 221-244.

### Equation of State

- Wright, D. G. (1997): An equation of state for use in ocean models.
  *JAOT*, 14(3), 735-740.

### Grid Discretization

- Arakawa, A., V. R. Lamb (1977): Computational design of the basic dynamical
  processes of the UCLA GCM. *Methods Comput. Phys.*, 17, 173-265.
- Sadourny, R. (1975): The dynamics of finite-difference models of the
  shallow-water equations. *JAS*, 32, 680-689.
- Lin, S.-J. (2004): A "vertically Lagrangian" finite-volume dynamical core.
  *MWR*, 132, 2293-2307.
- Putman, W. M., S.-J. Lin (2007): Finite-volume transport on various
  cubed-sphere grids. *JCP*, 227, 55-78.
- Ringler, T. D., J. Thuburn, J. B. Klemp, W. C. Skamarock (2010): A unified
  approach to energy conservation and PV dynamics for C-grids. *JCP*, 229,
  3065-3090.

### Time Stepping

- Hallberg, R. (1997): Stable split time stepping schemes for large-scale
  ocean modeling. *JCP*, 135, 54-65.
- Higdon, R. L. (2005): A two-level time-stepping method for layered ocean
  circulation models. *JCP*, 208, 629-648.
- Shchepetkin, A. F., J. C. McWilliams (2005): The regional oceanic modeling
  system (ROMS). *Ocean Modelling*, 9, 347-404.

### Tracer Advection

- Van Leer, B. (1977): Towards the ultimate conservative difference scheme.
  IV. *JCP*, 23, 276-299.
- Colella, P., P. R. Woodward (1984): The piecewise parabolic method (PPM).
  *JCP*, 54, 174-201.
- Zalesak, S. T. (1979): Fully multidimensional flux-corrected transport.
  *JCP*, 31, 335-362.

### Mixing and Parameterizations

- Smagorinsky, J. (1963): General circulation experiments with the primitive
  equations. *MWR*, 91, 99-164.
- Gent, P. R., J. C. McWilliams (1990): Isopycnal mixing in ocean circulation
  models. *JPO*, 20, 150-155.
- Redi, M. H. (1982): Oceanic isopycnal mixing by coordinate rotation.
  *JPO*, 12, 1154-1158.
- Griffies, S. M. (1998): The Gent-McWilliams skew flux. *JPO*, 28, 831-841.
- Danabasoglu, G., J. C. McWilliams (1995): Sensitivity of the global ocean
  to mesoscale tracer transport parameterizations. *J. Climate*, 8, 2967-2987.
- Large, W. G., J. C. McWilliams, S. C. Doney (1994): Oceanic vertical
  mixing: a review and a model with KPP. *Rev. Geophys.*, 32, 363-403.
- Pacanowski, R. C., S. G. H. Philander (1981): Parameterization of vertical
  mixing. *JPO*, 11, 1443-1451.

### Surface Forcing

- Large, W. G., S. Yeager (2004): Diurnal to decadal global forcing. NCAR
  Tech. Note NCAR/TN-460+STR.
- Paulson, C. A., J. J. Simpson (1977): Irradiance measurements in the upper
  ocean. *JPO*, 7(6), 952-956.

### Biogeochemistry

- Wanninkhof, R. (2014): Relationship between wind speed and gas exchange
  over the ocean revisited. *Limnol. Oceanogr. Methods*, 12, 351-362.
- Fasham, M. J. R., H. W. Ducklow, S. M. McKelvie (1990): A nitrogen-based
  model of plankton dynamics. *J. Marine Res.*, 48, 591-639.

### Validation Test Cases

- Eady, E. T. (1949): Long waves and cyclone waves. *Tellus*, 1, 33-52.
- Phillips, N. A. (1954): Energy transformations and meridional circulations.
  *Tellus*, 6, 273-286.
- Zhang, Y., et al. (2024): Influence of topographic form stress on the ACC.
  *JPO*, 54, 1565-1581.
- Gill, A. E. (1982): *Atmosphere-Ocean Dynamics*. Academic Press.
- Vallis, G. K. (2017): *Atmospheric and Oceanic Fluid Dynamics*, 2nd ed.
  Cambridge UP.
