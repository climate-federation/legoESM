# legoESM: A Differentiable Earth System Model
## Technical Specification v2.0

**Project**: legoESM
**License**: MIT
**Authors**: Pierre Gentine + Claude
**Date**: 2026-03-02 (updated from v1.0, 2026-02-26)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Core Infrastructure](#3-core-infrastructure)
4. [Component Specifications](#4-component-specifications)
5. [Differentiability Design](#5-differentiability-design)
6. [Hardware & Parallelism](#6-hardware--parallelism)
7. [I/O & Data Pipeline](#7-io--data-pipeline)
8. [Software Engineering](#8-software-engineering)
9. [API Design](#9-api-design)
10. [Testing & Validation](#10-testing--validation)
11. [Milestone Roadmap](#11-milestone-roadmap)
12. [Repository Structure](#12-repository-structure)

---

## 1. Project Overview

### 1.1 Vision

legoESM is a next-generation, fully differentiable Earth System Model built from
scratch in JAX. It spans weather-to-climate timescales, couples atmosphere, ocean,
land, and cryosphere through a unified interface, and enables end-to-end gradient
computation for data assimilation, parameter estimation, and hybrid AI-physics modeling.

### 1.2 Design Principles

| Principle | Description |
|-----------|-------------|
| **Differentiable-first** | Every operation is JAX-traceable; end-to-end `jax.grad` through the full coupled model |
| **Conservation as hard constraint** | Mass, energy, and momentum are conserved via projection/fixer steps, not soft penalties |
| **Modular & swappable** | Each physics module has a standard tensor-in/tendency-out interface; AI or physics implementations are interchangeable |
| **Hardware-portable** | Runs on CPU, GPU (NVIDIA multi-GPU primary), TPU, and Apple Silicon (Metal) |
| **Functional purity** | No in-place mutations; all state transitions are pure functions returning new state |
| **Performance-oriented** | Fully JIT-compiled; mixed precision; optimized for throughput |

### 1.3 Milestone Roadmap (Summary)

| # | Milestone | Target |
|---|-----------|--------|
| M1 | Dynamical core MVP (shallow-water on cubed-sphere) | Week 1 |
| M2 | 3D primitive equations + Held-Suarez | Week 2-3 |
| M3 | Swappable physics parameterizations | Week 4-5 |
| M4 | ML-driven parameterization training | Week 6-8 |
| M5 | Adjoint data assimilation (4D-Var) | Week 9-12 |
| M6 | Ocean + land + ice coupling | Week 13-20 |
| M7 | Climate-scale simulations | Week 21+ |

### 1.4 Key Reference Models

| Model | Inspiration |
|-------|-------------|
| **NeuralGCM / Dinosaur** | JAX dycore architecture, IMEX time integration, pytree state, ML/physics coupling |
| **MPAS** | Variable-resolution Voronoi meshes, C-grid staggering, fully compressible non-hydrostatic equations, split-explicit time integration |
| **Veros** | JAX ocean model, mpi4jax parallelism, differentiable ocean dynamics |
| **DifferLand** | JAX land model, carbon-water coupling, `jax.lax.scan` time integration pattern |
| **MOM6** | Ocean component architecture, ALE vertical coordinate |

---

## 2. Architecture Overview

### 2.1 High-Level Architecture

```
+------------------------------------------------------------------+
|                        legoESM Runner                           |
|  (Python API / YAML Config / CLI)                                |
+------------------------------------------------------------------+
         |                    |                    |
         v                    v                    v
+------------------+  +---------------+  +------------------+
|    Coupler       |  |  Diagnostics  |  |   I/O Manager    |
|  (flux exchange, |  |  (online      |  |  (Zarr, ERA5     |
|   regridding,    |  |   statistics) |  |   initialization)|
|   conservation)  |  |               |  |                  |
+------------------+  +---------------+  +------------------+
   |     |     |     |
   v     v     v     v
+------+ +-----+ +------+ +------+
| Atm  | | Ocn | | Land | | Ice  |
+------+ +-----+ +------+ +------+
   |        |        |        |
   v        v        v        v
+------------------------------------------------------------------+
|                    Core Infrastructure                            |
|  Grid System | State/Field | Time Integration | Conservation     |
|  Operators   | Parallelism | Mixed Precision  | Smooth Approx    |
+------------------------------------------------------------------+
         |
         v
+------------------------------------------------------------------+
|                    JAX Runtime                                    |
|  jit | grad | vmap | scan | checkpoint | sharding | pjit        |
+------------------------------------------------------------------+
         |
         v
+------------------------------------------------------------------+
|                    Hardware Backends                              |
|  CPU | NVIDIA GPU (multi) | TPU | Apple Metal (M-series)        |
+------------------------------------------------------------------+
```

### 2.2 Data Flow

Each component follows the same pattern per timestep:

```
state(t) --> [Physics Module A] --> tendencies_A \
         --> [Physics Module B] --> tendencies_B  |--> sum --> [Time Integrator] --> state(t+dt)
         --> [Physics Module C] --> tendencies_C /
                                                          |
                                                          v
                                                   [Conservation Fixer]
                                                          |
                                                          v
                                                   state_conserved(t+dt)
```

### 2.3 Component Coupling

```
                    +----------+
                    |  Coupler |
                    +----------+
                   /   |    |   \
        heat,     /    |    |    \  heat,
        moisture /     |    |     \ freshwater
        momentum/      |    |      \stress
               v       |    |       v
         +-------+     |    |    +-------+
         |  Atm  |<--->|    |<-->|  Ocn  |
         +-------+  SST|    |SSS +-------+
              |    rad, |    | runoff  |
              | precip  |    |         |
              v         v    v         v
         +-------+            +-------+
         | Land  |            |  Ice  |
         +-------+            +-------+
```

Coupling is flexible: exchange interval configurable from 30 minutes to 1+ day.

---

## 3. Core Infrastructure

### 3.1 Grid System

#### 3.1.1 Design: Multi-Grid Abstraction

All grids implement a common `Grid` protocol:

```python
class Grid(Protocol):
    """Abstract grid interface. All grids are JAX pytrees."""

    @property
    def n_cells(self) -> int: ...

    @property
    def n_edges(self) -> int: ...

    @property
    def n_vertices(self) -> int: ...

    @property
    def cell_areas(self) -> jax.Array: ...

    @property
    def edge_lengths(self) -> jax.Array: ...

    def cell_to_edge(self, field: jax.Array) -> jax.Array: ...
    def edge_to_cell(self, field: jax.Array) -> jax.Array: ...
    def gradient(self, scalar_field: jax.Array) -> jax.Array: ...
    def divergence(self, vector_field: jax.Array) -> jax.Array: ...
    def curl(self, vector_field: jax.Array) -> jax.Array: ...
    def laplacian(self, field: jax.Array) -> jax.Array: ...
    def interpolate_to(self, field: jax.Array, target_grid: 'Grid') -> jax.Array: ...
```

#### 3.1.2 Implemented Grids

| Grid | Status | Notes |
|------|--------|-------|
| **Cubed-sphere** | Implemented | 6 faces, gnomonic equidistant, FV operators |
| **Gaussian (spectral)** | Implemented | Triangular-truncated spherical harmonics, dealiased |

#### 3.1.3 Cubed-Sphere Grid (`grids/cubed_sphere.py`)

`CubedSphereGrid(NamedTuple)` — gnomonic equidistant projection, registered as JAX pytree.

- 6 faces x N x N cells per face (N configurable)
- Resolution examples: C8 (~500 km, testing), C48 (~200 km), C192 (~50 km), C384 (~25 km)
- Ghost cells via halo padding for inter-face communication
- All metric arrays are shape `(6, n, n)`:
  - `lon, lat` — cell-center geographic coordinates
  - `area, dx, dy` — cell areas and spacings
  - `f` — Coriolis parameter
  - `angle` — grid rotation angle (for vector halo exchange)
  - `cos_angle_padded, sin_angle_padded` — padded rotation for vector halo
  - `hx_ext, hy_ext` — extrapolated half-metrics for divergence operator
  - `x_cart, y_cart, z_cart` — Cartesian coordinates

#### 3.1.4 Gaussian Grid (`grids/gaussian.py`)

`GaussianGrid(NamedTuple)` — for pseudospectral methods with spherical harmonic (SH) transforms.

- Triangular truncation T_N: `n_max` spectral modes, `n_sh = (n_max+1)(n_max+2)/2` coefficients
- Grid resolution: `n_lat = 3(n_max+1)/2` (dealiasing), `n_lon = 2*n_lat`
- Precomputed SH matrices: `Pnm` (associated Legendre), `Hnm` (weighted), `Pnm_oc2` (cos²-weighted), `Dnm` (dP/dμ)
- Spectral eigenvalues: `lap = -n(n+1)/a²`, `ilap` (inverse Laplacian)
- Key transforms:
  - `sh_analysis(grid, field)` → spectral coefficients (FFT + Legendre)
  - `sh_synthesis(grid, coeffs)` → grid-space field (Legendre + IFFT)
  - `uv_from_vordiv(grid, vor_hat, div_hat)` → (u·cosθ, v·cosθ) in grid space
  - `sh_analysis_3d / sh_synthesis_3d` — batched over vertical levels via vmap
  - `spectral_hyperdiffusion(coeffs, grid, coeff, order)` — scale-selective damping

#### 3.1.5 Vertical Coordinates (`grids/vertical.py`, `ocean/vertical.py`)

**Atmosphere — Sigma Coordinate** (`SigmaCoordinate`):

```
σ = p / p_s    (σ_top ≤ σ ≤ 1)
```

- `sigma_full` (nlev,), `sigma_half` (nlev+1,), `dsigma` (nlev,)
- Simmons-Burridge coefficients: `ln_ratio`, `alpha`, `fractional_sigma`
- Used by: hydrostatic PE (FV and spectral)

**Atmosphere — Height Coordinate** (`HeightCoordinate`):

```
z* = H · (z − z_s) / (H − z_s)
```

- `z_full` (nlev,), `z_half` (nlev+1,), `dz` (nlev,), `dz_half` (nlev-1,)
- Reference state: `rho_ref`, `theta_ref`, `exner_ref` (1D profiles at init)
- `TerrainMetric`: Jacobian J = (H−z_s)/H, physical z at all levels
- Used by: non-hydrostatic compressible Euler (FV and spectral)

**Ocean — z-star Coordinate** (`OceanZStarCoordinate`):

```
z* = H_max · (z + H) / (η + H)
```

- **Dynamic Jacobian**: J = (η + H_bathy) / H_max, recomputed every timestep
- Stretched grid: ~10 m near surface, ~200 m at depth, default 50 levels over 5500 m
- `z_full_ref` (nlev,), `z_half_ref` (nlev+1,), `dz_ref` (nlev,)
- Level convention: k=0 is surface, k=nlev−1 is deepest; reference z values are negative
- Layer thickness: h_k = dz_ref[k] · J

### 3.2 State Representation

We use a **custom lightweight `Field` dataclass** registered as a JAX pytree:

```python
@jax.tree_util.register_pytree_class
@dataclass(frozen=True)
class Field:
    """A coordinate-aware array that is a JAX pytree leaf."""
    data: jax.Array           # The actual numerical data
    name: str                 # Variable name (e.g., "potential_temperature")
    dims: tuple[str, ...]     # Dimension names (e.g., ("face", "x", "y", "z"))
    units: str                # Physical units (e.g., "K")
    long_name: str = ""       # Human-readable description
    staggering: str = "cell"  # "cell", "edge", or "vertex"
```

**Rationale**: Lighter than Coordax (no external dependency), full pytree compatibility
for `jit`/`grad`/`vmap`/`scan`, carries metadata for I/O and diagnostics, and
enables compile-time dimension checking.

**Model state** is a frozen dataclass containing Fields:

```python
@dataclass(frozen=True)
class AtmosphereState:
    # Prognostic variables (fully compressible non-hydrostatic)
    rho: Field              # Dry air density [kg/m^3]
    theta: Field            # Potential temperature [K]
    u: Field                # Zonal wind [m/s] (edge-normal)
    v: Field                # Meridional wind [m/s] (edge-normal)
    w: Field                # Vertical velocity [m/s]
    q_vapor: Field          # Specific humidity [kg/kg]
    q_cloud: Field          # Cloud water mixing ratio [kg/kg]
    q_ice: Field            # Cloud ice mixing ratio [kg/kg]
    q_rain: Field           # Rain mixing ratio [kg/kg]
    q_snow: Field           # Snow mixing ratio [kg/kg]
    p_surface: Field        # Surface pressure [Pa]

    # Diagnostic (derived, not time-stepped)
    pressure: Field         # Full pressure [Pa]
    temperature: Field      # Temperature [K]
    geopotential: Field     # Geopotential height [m^2/s^2]
```

For hydrostatic mode, `w` is diagnostic and `rho` is derived from the equation of state.

### 3.3 Discrete Operators

#### 3.3.1 2D Operators (`core/operators.py`)

All spatial operators are pure functions operating on `(6, n, n)` cubed-sphere arrays:

```python
gradient_x(field, grid)           # d/dx via centered differences
gradient_y(field, grid)           # d/dy via centered differences
divergence(u_field, v_field, grid) # div(u,v) with vector halo exchange
curl_z(u_field, v_field, grid)    # vorticity: dv/dx − du/dy
laplacian(field, grid)            # 2nd-order ∇²
advect_upwind(q, u, v, grid)     # 1st-order upwind advection
advect_centered(q, u, v, grid)   # 2nd-order centered advection
hyperdiffusion(field, grid, coeff) # 4th-order ∇⁴ damping (two Laplacians)
global_integral(field, grid)      # Area-weighted integral (MPI-aware via allreduce)
global_mean(field, grid)          # Area-weighted mean
```

#### 3.3.2 3D Operators (`core/operators_3d.py`)

Vmapped over vertical levels, operating on `(6, n, n, nlev)` arrays:

```python
vorticity_3d(u_3d, v_3d, grid)
gradient_x_3d(field_3d, grid)
gradient_y_3d(field_3d, grid)
divergence_3d(u_3d, v_3d, grid)
hyperdiffusion_3d(field_3d, grid, coeff)
vertical_gradient_half_to_full(field, dz, dz_half)
vertical_gradient_full_to_half(field, dz_half)
vertical_advection_height(field, w, dz, dz_half, J)
vertical_divergence_height(flux, dz, J)
```

#### 3.3.3 Halo Exchange (`grids/halo.py`, `parallel/halo_exchange.py`)

Inter-face boundary data exchange with pluggable backend:
- **Local backend** (default): direct array indexing within single process
- **MPI backend**: `mpi4jax.sendrecv` for multi-node; edges packed per neighbor rank to reduce MPI messages
- `pad_halo(data)` → `(6, n+2, n+2)` — scalar halo padding
- `pad_halo_vector(u, v, ...)` → vector halo with grid-angle rotation at face boundaries
- Canonical edge ordering for packed exchange: sender sorts by `(face, edge)`, receiver by `(nbr_face, nbr_edge)`

### 3.4 Time Integration

#### 3.4.1 Integrator Interface

```python
class TimeIntegrator(Protocol):
    def step(
        self,
        state: State,
        tendencies_fn: Callable[[State], State],
        dt: float,
    ) -> State: ...
```

#### 3.4.2 Implemented Integrators

| Integrator | File | Use Case |
|------------|------|----------|
| **SSP-RK3** | `ssp_rk3.py` | Shallow water, hydrostatic PE, ocean spectral |
| **Split-explicit RK3** | `split_explicit.py` | Non-hydrostatic (acoustic substeps), ocean (barotropic substeps) |
| **Semi-implicit** | `semi_implicit.py` | Hoskins-Simmons method for spectral PE |
| **Tridiagonal solver** | `tridiagonal.py` | Semi-implicit acoustic substeps (vertical) |

#### 3.4.3 SSP-RK3

```
k1 = state + dt·F(state)
k2 = ¾·state + ¼·(k1 + dt·F(k1))
k3 = ⅓·state + ⅔·(k2 + dt·F(k2))
```

#### 3.4.4 Split-Explicit Strategy

For non-hydrostatic atmosphere and ocean:
- **Slow tendencies**: Coriolis, advection, horizontal PGF, diffusion (once per RK3 stage)
- **Fast substeps**: Acoustic (atmosphere) or barotropic (ocean) modes via `fori_loop` or `scan`
- `dt_fast = dt_slow / N_substeps` (atmosphere: N=6 default, ocean: N=30 default)

### 3.5 Conservation

Conservation is a **hard constraint** enforced via projection after each timestep.

#### 3.5.1 Conserved Quantities

| Quantity | Method | Scope |
|----------|--------|-------|
| **Dry air mass** | Surface pressure fixer (global integral preserved) | Global |
| **Total water mass** | Column-wise: `sum(q_v + q_c + q_i + q_r + q_s) * dp/g` preserved | Column + Global |
| **Total energy** | Global fixer: kinetic + internal + potential + latent | Global |
| **Momentum** | Angular momentum conservation via Coriolis discretization | Global |
| **Tracer mass** | Positive-definite flux-form advection (Zalesak limiter) | Global per tracer |

#### 3.5.2 Conservation Fixer

Applied after each full timestep:

```python
def apply_conservation_fixer(state_new: State, state_old: State, grid: Grid) -> State:
    """Project state_new onto the conservation manifold."""

    # 1. Compute global integrals
    mass_old = global_integral(state_old.p_surface, grid)
    mass_new = global_integral(state_new.p_surface, grid)

    # 2. Apply uniform correction (preserves gradients for differentiability)
    correction = (mass_old - mass_new) / grid.total_area
    p_surface_fixed = state_new.p_surface + correction

    # 3. Similarly for energy and moisture
    # Energy fixer: scale temperature uniformly to restore total energy
    # Moisture fixer: scale q uniformly to restore total moisture

    return state_new.replace(p_surface=p_surface_fixed, ...)
```

The fixer uses **uniform additive corrections** (not multiplicative) to preserve
gradients for automatic differentiation.

### 3.6 Smooth Approximations

All discontinuous operations are replaced with smooth, differentiable alternatives:

```python
# Instead of: jnp.where(q > q_sat, condensation, 0)
# Use:        sigmoid_switch(q - q_sat, sharpness=100) * condensation

def sigmoid_switch(x: jax.Array, sharpness: float = 100.0) -> jax.Array:
    """Smooth approximation to Heaviside step function."""
    return jax.nn.sigmoid(sharpness * x)

def smooth_max(a: jax.Array, b: jax.Array, sharpness: float = 100.0) -> jax.Array:
    """Differentiable approximation to max(a, b)."""
    return jax.nn.logsumexp(jnp.stack([a * sharpness, b * sharpness]), axis=0) / sharpness

def smooth_clamp(x: jax.Array, lo: float, hi: float, sharpness: float = 100.0) -> jax.Array:
    """Differentiable clamp."""
    return smooth_max(smooth_min(x, hi, sharpness), lo, sharpness)
```

---

## 4. Component Specifications

### 4.1 Atmosphere

The atmosphere has six implemented dynamical cores spanning two discretization families (finite-volume on cubed-sphere, pseudospectral on Gaussian grid) and three equation sets (shallow water, hydrostatic PE, non-hydrostatic compressible Euler).

| Model | Grid | Equations | Time Integration |
|-------|------|-----------|-----------------|
| `ShallowWaterModel` | Cubed-sphere | Shallow water | SSP-RK3 |
| `PrimitiveEquationModel` | Cubed-sphere | Hydrostatic PE (σ) | SSP-RK3 |
| `CompressibleEulerModel` | Cubed-sphere | Non-hydrostatic (z*) | Split-explicit RK3 |
| `SpectralShallowWaterModel` | Gaussian | Shallow water (vor-div) | SSP-RK3 |
| `SpectralPrimitiveEquationModel` | Gaussian | Hydrostatic PE (vor-div-σ) | SSP-RK3 |
| `SpectralCompressibleEulerModel` | Gaussian | Non-hydrostatic (vor-div-z*) | Split-explicit RK3 |

All models follow the same API: `state = model.step(state, dt)`, with `integrate()` and `integrate_scan()` (differentiable via `lax.scan`) methods.

#### 4.1.1 Shallow Water Equations (`atmosphere/dynamics/shallow_water.py`)

Vector-invariant form on the cubed-sphere:

```
dh/dt = −div(h·v)
du/dt = (ζ+f)·v − ∂B/∂x + D_u
dv/dt = −(ζ+f)·u − ∂B/∂y + D_v
```

where B = K + g(h + h_s), K = ½(u² + v²), ζ = ∂v/∂x − ∂u/∂y.

**State**: `ShallowWaterState(h, u, v, h_s)` — all `(6, n, n)`.
**Config**: `g=9.81`, `hyperdiff_coeff=0.0`, conservation fixer flags.

#### 4.1.2 Hydrostatic Primitive Equations (`atmosphere/dynamics/primitive_eq.py`)

σ-coordinate PE in vector-invariant form:

```
dp_s/dt = −p_s · Σ[div(v_k)·Δσ_k] / (1−σ_top)
du/dt   = (ζ+f)·v − ∂B/∂x − R_d·T·∂(ln p_s)/∂x
dT/dt   = −v·∇T − σ̇·∂T/∂σ + κ·T·ω/p
```

Geopotential via Simmons-Burridge hydrostatic integration with α_k correction.
Sigma-dot diagnosed from continuity with proper BCs (σ̇=0 at top/bottom).

**State**: `HydrostaticState(u, v, T, p_s, phis)` — 3D fields `(6,n,n,nlev)`, surface fields `(6,n,n)`.

#### 4.1.3 Non-Hydrostatic Compressible Euler (`atmosphere/dynamics/compressible_euler.py`)

Height z* coordinates with reference-state subtraction to avoid cancellation:

```
dρ'/dt  = −(1/J)[div_h(J·ρ·v_h) + ∂(ρ·w)/∂z*]
dθ'/dt  = −v·∇θ − (w/J)·∂θ/∂z*
du/dt   = (ζ+f)·v − ∂K/∂x − c_p·θ·∂π'/∂x
dw/dt   = −c_p·θ·(1/J)·∂π'/∂z* − g·θ'/θ₀
```

Exner perturbation: π' = π₀·[((1+ρ'/ρ₀)(1+θ'/θ₀))^(R_d/c_v) − 1] (ratio form).

**Split-explicit time stepping**: RK3 outer loop (slow tendencies: Coriolis, horizontal PGF, advection, hyperdiffusion, sponge) with N acoustic substeps (vertical PGF, buoyancy, continuity). Options for explicit forward-backward or semi-implicit (tridiagonal solve for w).

**State**: `NonHydrostaticState(u, v, w, theta_prime, rho_prime, phis, tracers)`.
**Config**: `n_acoustic_substeps=6`, `sponge_width=10000m`, `sponge_coeff=0.05`.

#### 4.1.4 Spectral Variants (`atmosphere/dynamics/spectral_sw.py`, `spectral_pe.py`, `spectral_nh.py`)

Pseudospectral vorticity-divergence formulation on the Gaussian grid:

```
∂ζ/∂t = −div((ζ+f)·v) + curl(...)
∂D/∂t = curl((ζ+f)·v) − ∇²(E+Φ+...) + div(...)
```

Workflow: SH synthesis → grid-space nonlinear products → SH analysis → spectral tendencies.
Spectral hyperdiffusion: −ν·[n(n+1)/a²]^order per coefficient.
Metal backend: automatic CPU fallback for complex128 SH transforms.

#### 4.1.5 Physics Parameterizations

Implemented physics modules:

| Module | File | Description |
|--------|------|-------------|
| Held-Suarez | `physics/held_suarez.py` | Newtonian relaxation + Rayleigh friction |
| Baroclinic wave | `physics/baroclinic_wave.py` | Jablonowski-Williamson initial conditions |
| Kessler | `physics/kessler.py` | Warm-rain microphysics |

Standard interface: `tendencies = physics_fn(state, grid, dt, config)`.
Models support `step_with_physics()` for coupled dynamics+physics stepping.

### 4.2 Ocean

#### 4.2.1 Overview

Boussinesq hydrostatic ocean primitive equation solver with:
- Wright (1997) equation of state
- z-star vertical coordinate with dynamic Jacobian
- Split-explicit barotropic/baroclinic time stepping
- Both cubed-sphere FV and spectral (Gaussian grid) discretizations
- Land masking via boolean mask + zero-fill
- Global ocean support (C48 ~2°, 50 vertical levels)

Architecture: `OceanModel` (FV on cubed-sphere), `SpectralOceanModel` (pseudospectral on Gaussian grid).

#### 4.2.2 Governing Equations

Boussinesq hydrostatic primitive equations in vector-invariant form:

```
du/dt = (ζ+f)·v − ∂B/∂x − (1/ρ₀)·∂p'/∂x + A_h·∇²u + ∂/∂z(A_v·∂u/∂z) − w·∂u/∂z
dv/dt = −(ζ+f)·u − ∂B/∂y − (1/ρ₀)·∂p'/∂y + A_h·∇²v + ∂/∂z(A_v·∂v/∂z) − w·∂v/∂z
dT/dt = −u·∇T − w·∂T/∂z + K_h·∇²T + ∂/∂z(K_v·∂T/∂z)
dS/dt = −u·∇S − w·∂S/∂z + K_h·∇²S + ∂/∂z(K_v·∂S/∂z)
dη/dt = −Σ_k div(h_k·v_k)
```

**Diagnostic relations:**
- Density: ρ = wright_eos(T, S, p_hydro)
- Hydrostatic pressure: p(z) = ρ₀gη + ∫₀^z ρ'g dz' (top-down cumsum)
- Vertical velocity: w(z) = −∫_{-H}^z div(v) dz' (bottom-up, w=0 at floor)
- Layer thickness: h_k = dz_ref[k] · J, where J = (η + H_bathy) / H_max
- Kinetic energy: B = ½(u² + v²)
- Vorticity: ζ = ∂v/∂x − ∂u/∂y

#### 4.2.3 Equation of State (`ocean/eos.py`)

Wright (1997) 9-term polynomial EOS (J. Atmos. Oceanic Tech., 14(3), 735–740):

```
ρ = (p + p₀) / (λ + α₀·(p + p₀))
```

where α₀(T,S), p₀(T,S), λ(T,S) are polynomial functions of temperature and salinity.
Reference values: T=25°C, S=35 PSU, p=0 → ρ ≈ 1023.3 kg/m³.

Additional functions:
- `density_perturbation(T, S, p)` → ρ' = ρ − ρ₀
- `compute_hydrostatic_pressure(rho, eta, dz, jacobian)` — top-down cumulative integral
- `compute_buoyancy_frequency(rho, dz, jacobian)` → N² = −(g/ρ₀)·dρ/dz

Ocean constants: ρ₀ = 1025.0 kg/m³, c_sw = 3994.0 J/(kg·K).

#### 4.2.4 State Containers (`ocean/state.py`)

**FV Ocean State (cubed-sphere):**

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| u | (6,n,n,nlev) | m/s | Zonal velocity |
| v | (6,n,n,nlev) | m/s | Meridional velocity |
| T | (6,n,n,nlev) | °C | Potential temperature |
| S | (6,n,n,nlev) | PSU | Salinity |
| eta | (6,n,n) | m | Sea surface height |
| H_bathy | (6,n,n) | m | Bathymetry depth (static, positive) |
| land_mask | (6,n,n) | 0/1 | Ocean=1, land=0 (static) |

**OceanConfig defaults:**

| Parameter | Default | Units | Purpose |
|-----------|---------|-------|---------|
| g | 9.80616 | m/s² | Gravity |
| rho_0 | 1025.0 | kg/m³ | Reference density |
| A_h | 1.0e4 | m²/s | Horizontal viscosity |
| K_h | 1.0e3 | m²/s | Horizontal tracer diffusivity |
| A_v | 1.0e-3 | m²/s | Vertical viscosity |
| K_v | 1.0e-4 | m²/s | Vertical tracer diffusivity |
| n_barotropic_substeps | 30 | — | Barotropic subcycles per step |
| barotropic_diffusion_alpha | 0.01 | — | Barotropic damping coefficient |
| barotropic_diffusion_dt_ref | 60.0 | s | Reference dt for damping scaling |

**Spectral Ocean State (Gaussian grid):**
- 3D spectral fields `(n_sh, nlev)` complex: `vor_hat, div_hat, T_hat, S_hat`
- 2D spectral fields `(n_sh,)` complex: `eta_hat, H_bathy_hat`
- Grid-space field: `land_mask_grid` (masking applied in grid space)

#### 4.2.5 Split-Explicit Time Stepping

**Baroclinic (slow) step** (`ocean/dynamics/ocean_pe.py`):
1. Compute layer thickness h_k and Jacobian J from η, H_bathy
2. Density from Wright EOS and hydrostatic pressure (top-down cumsum)
3. Diagnose w from continuity (bottom-up integral of div(v))
4. Coriolis split: planetary f on baroclinic shear (u' = u − U_bar), relative ζ on full velocity
5. Pressure gradient, vertical advection (upwind), tracer advection
6. Horizontal/vertical mixing, optional hyperdiffusion
7. Land masking of all tendencies
8. Free-surface tendency from depth-integrated flux divergence

**Barotropic (fast) substeps** (`ocean/dynamics/barotropic.py`):

Forward-backward substeps for 2D free-surface gravity waves:

```
Forward:  η_new = η − dt_s · div(H_total · U_bar, H_total · V_bar)
Backward: U_bar_new, V_bar_new from semi-implicit Coriolis + g·∇η_new + F_slow
```

Semi-implicit Coriolis at substep level:
```
α = 0.5·f·dt_s
U_new = (U_c + α·V_c − dt_s·g·∂η/∂x + ...) / (1 + α²)
V_new = (V_c − α·U_c − dt_s·g·∂η/∂y − ...) / (1 + α²)
```

After substeps: velocity correction u_new = (u − U_bar_old) + U_bar_new.
Loop via `jax.lax.fori_loop` (fast) or `jax.lax.scan` (differentiable).
Optional compact Laplacian diffusion with dt-scaled damping.

#### 4.2.6 OceanModel (`ocean/dynamics/ocean_model.py`)

```python
class OceanModel:
    def __init__(self, grid, z_coord, config=None)

    def step(self, state, dt) -> OceanState           # JIT-compiled
    def step_checked(self, state, dt) -> OceanState    # with runtime assertions
    def integrate(self, state, duration, dt, save_every)
    def integrate_scan(self, state, n_steps, dt)       # differentiable via lax.scan
```

Runtime checks (if enabled): finite state, water column thickness, |η| bounds, T/S bounds, land cells strictly zero.

#### 4.2.7 Mixing (`ocean/physics/mixing.py`)

- `laplacian_viscosity_3d(field_3d, grid, coeff)` — A_h·∇² vmapped over levels
- `vertical_diffusion(field, z_coord, jacobian, coeff)` — ∂/∂z(K_v·∂f/∂z) with zero-flux BCs at surface and bottom

#### 4.2.8 Spectral Ocean (`ocean/dynamics/spectral_ocean_pe.py`)

Pseudospectral vorticity-divergence formulation, paralleling the atmospheric spectral PE:

```
∂ζ/∂t = −div((ζ+f)·v) + curl(vertical advection + mixing)
∂D/∂t = curl((ζ+f)·v) − ∇²(K + p'/ρ₀) + div(vertical advection + mixing)
∂T/∂t = −div(T·v) + T·D − w·∂T/∂z + K_h·∇²T + vertical diffusion
∂S/∂t = −div(S·v) + S·D − w·∂S/∂z + K_h·∇²S + vertical diffusion
∂η/∂t = −Σ_k(D·h_k)
```

Land masking in spectral: mask applied in grid space before every SH analysis call.
Gibbs oscillations at coastlines controlled by spectral hyperdiffusion.

`SpectralOceanModel`: same API as `OceanModel`, uses SSP-RK3.
Auto-routes complex128 computation to CPU on Metal backend.

#### 4.2.9 Conservation (`ocean/conservation.py`)

| Quantity | Method |
|----------|--------|
| Volume | Uniform additive η correction: ∫∫ η·area = const |
| Heat | Uniform additive T correction: ∫∫∫ T·h_k·area = const |
| Salt | Uniform additive S correction: ∫∫∫ S·h_k·area = const |

MPI-aware global sums via `global_sum_mpi()` in distributed mode.

#### 4.2.10 Initialization (`ocean/init.py`)

- `idealized_bathymetry(grid, H_max=5500, land_lat_threshold=80)` — land at high latitudes
- `rest_state_ocean(grid, z_coord, T_surface=20, T_deep=2, S_uniform=35)` — exponential stratification
- `wind_driven_gyre_init(...)` — double-gyre test case

#### 4.2.11 Validated Test Cases

| Test | Resolution | Duration | Validation |
|------|-----------|----------|------------|
| Stommel gyre | C16–C32 | 300 days | Steady western boundary current |
| Baroclinic adjustment | C16–C32 | 300 days | Thermocline flattening |
| Equatorial Kelvin wave | C16–C32 | 300 days | Eastward propagation |
| Rest state | C8 | 50 steps | Tendencies < 1e-6, T/S bounded |
| Differentiability | C8 | 1 step | `jax.grad` through model.step produces finite gradients |

### 4.3 Land

#### 4.3.1 Overview

Inspired by DifferLand (Fang & Gentine, 2024), rewritten for the legoESM interface with
added energy balance and soil physics.

#### 4.3.2 Processes

**Milestone 1 (Bucket Model):**
- Surface energy balance: `R_net = H + LE + G`
- Single-layer soil moisture bucket with field capacity and wilting point
- Simple snow accumulation/melt (temperature threshold)
- Prescribed vegetation (LAI from climatology)
- Penman-Monteith evapotranspiration

**Phase 2 (DifferLand-inspired):**
- Multi-layer soil temperature (diffusion equation)
- Multi-layer soil moisture (Richards equation)
- Carbon cycling: 6 pools (labile, foliar, root, wood, litter, SOM)
- FvCB-Medlyn photosynthesis (from DifferLand)
- Dynamic LAI from foliar carbon
- Stomatal conductance

**Phase 3 (Full LSM):**
- Vegetation dynamics and competition
- Carbon-nitrogen coupling
- Groundwater and runoff routing
- Urban / lake / glacier tiles

#### 4.3.3 State

```python
@dataclass(frozen=True)
class LandState:
    # Energy
    T_surface: Field          # Surface temperature [K]
    T_soil: Field             # Soil temperature profile [K] (n_soil_layers)
    snow_depth: Field         # Snow water equivalent [kg/m^2]

    # Hydrology
    soil_moisture: Field      # Volumetric soil moisture [m^3/m^3] (n_soil_layers)
    snow_cover: Field         # Fractional snow cover [-]

    # Carbon (Phase 2)
    C_pools: Field            # Carbon pools [gC/m^2] (n_pools)
    LAI: Field                # Leaf area index [m^2/m^2]
```

### 4.4 Cryosphere

#### 4.4.1 Sea Ice

- Thermodynamic: Semtner 3-layer model (ice + snow)
- Dynamic: Elastic-viscous-plastic (EVP) rheology
- Transport: Advection of ice area and volume

#### 4.4.2 Ice Sheets (Phase 3+)

- Shallow-ice/shallow-shelf approximation
- Thermomechanical coupling
- Calving parameterization

#### 4.4.3 Permafrost (Phase 3+)

- Deep soil temperature extension
- Phase change (freeze/thaw) with smooth approximation

### 4.5 Coupler

#### 4.5.1 Design

```python
class Coupler:
    """Mediates flux exchange between components."""

    def __init__(
        self,
        atmosphere: AtmosphereComponent,
        ocean: OceanComponent,
        land: LandComponent,
        ice: IceComponent,
        coupling_dt: float,           # Exchange interval [s]
        regridder: Regridder,         # For different component grids
    ): ...

    def exchange_fluxes(
        self,
        atm_state: AtmosphereState,
        ocn_state: OceanState,
        land_state: LandState,
        ice_state: IceState,
    ) -> CoupledFluxes:
        """Compute and distribute fluxes between components.

        Fluxes computed:
        - Atm -> Ocn: wind stress, heat flux, freshwater flux, radiation
        - Ocn -> Atm: SST, surface currents
        - Atm -> Land: precipitation, radiation, temperature, humidity, wind
        - Land -> Atm: sensible heat, latent heat, albedo, roughness
        - Atm <-> Ice: similar to ocean but over ice fraction
        - Ocn <-> Ice: basal melt, salt rejection, heat flux
        """
        ...
```

#### 4.5.2 Conservation in Coupling

The coupler enforces that fluxes are conservative:
- Heat flux leaving atmosphere = heat flux entering ocean + land + ice
- Freshwater is conserved across the atmosphere-surface interface
- Momentum transfer is balanced (Newton's 3rd law)

---

## 5. Differentiability Design

### 5.1 End-to-End Gradient Flow

The entire model is a composition of differentiable functions:

```
loss = L(observe(integrate(initialize(params), n_steps)), observations)
grad_loss = jax.grad(loss)(params)
```

where `params` can include:
- Physical parameters (diffusion coefficients, drag coefficients, etc.)
- Neural network weights (for ML parameterizations)
- Initial conditions (for 4D-Var data assimilation)

### 5.2 Gradient Computation Strategy

| Technique | Purpose |
|-----------|---------|
| `jax.grad` | Reverse-mode AD for parameter gradients |
| `jax.jvp` | Forward-mode for tangent linear model |
| `jax.checkpoint` | Rematerialization to reduce memory for long rollouts |
| `jax.lax.scan` | Efficient sequential time integration with automatic checkpointing |
| Gradient clipping | Prevent exploding gradients in long rollouts |
| Truncated BPTT | Optional: backprop through fixed windows (e.g., 5 days) |

### 5.3 Checkpointing Strategy

For a 10-day forecast at 15-min timesteps = 960 steps.
Without checkpointing: must store all 960 intermediate states.
With checkpointing: store every Kth state, recompute the rest.

```python
def integrate(state, n_steps, dt, model):
    def scan_fn(state, _):
        state = jax.checkpoint(model.step)(state, dt)
        return state, state
    final_state, trajectory = jax.lax.scan(scan_fn, state, jnp.arange(n_steps))
    return final_state, trajectory
```

Configurable checkpoint interval: every 10-50 steps (tradeoff: memory vs recomputation).

### 5.4 4D-Var Data Assimilation

```python
def cost_4dvar(x0: State, observations: list[Observation], model, B_inv, R_inv):
    """4D-Var cost function — differentiable end-to-end."""
    # Background term
    J_b = 0.5 * (x0 - x_b).T @ B_inv @ (x0 - x_b)

    # Observation term: integrate forward, compute misfits
    trajectory = model.integrate(x0, n_steps)
    J_o = 0.0
    for obs in observations:
        H_x = obs.operator(trajectory[obs.time_index])
        d = obs.values - H_x
        J_o += 0.5 * d.T @ R_inv @ d

    return J_b + J_o

# Minimize with gradient descent
grad_J = jax.grad(cost_4dvar)
x0_optimal = optimize(cost_4dvar, grad_J, x0_initial)
```

---

## 6. Hardware & Parallelism

### 6.1 Target Platforms

| Platform | Priority | Backend | Notes |
|----------|----------|---------|-------|
| NVIDIA GPU (single) | P0 | CUDA via JAX | Development & testing |
| NVIDIA GPU (multi, single node) | P0 | JAX sharding | Primary production target |
| Apple M-series (Metal/MPS) | P1 | jax-metal | Development on M4 MacBook |
| CPU (multi-core) | P1 | JAX default | Fallback, CI/CD |
| NVIDIA GPU (multi-node) | P2 | mpi4jax | Large-scale production |
| Google TPU | P2 | JAX native | Cloud scaling |

### 6.2 Parallelism Strategy

#### 6.2.1 Single-Node: JAX Device Mesh (`parallel/mesh.py`)

Face-dimension sharding across 1–6 devices:

```python
config = create_device_mesh(n_devices="auto")  # auto-detects, clamps to 1/2/3/6
```

- `DeviceConfig(mesh, face_sharding, replicated_sharding, n_devices, backend, is_distributed)`
- Single device → `mesh=None`, no sharding overhead
- Multi-device → `Mesh(devices, ("face",))`, `PartitionSpec("face")` on first axis
- `shard_pytree(state, config)` — shards `(6,n,n,...)` arrays by face dimension
- `replicate_pytree(data, config)` — replicates data across all devices (e.g., grid metrics)
- Backend fallback: if requested backend unavailable, falls back to default with warning

#### 6.2.2 Multi-Node: MPI via mpi4jax (`parallel/distributed.py`, `parallel/halo_exchange.py`)

```python
config, topology = initialize_distributed(return_topology=True)
```

Initializes JAX distributed runtime + MPI, builds `CommTopology` for this rank.
Double-initialization guard returns existing config with warning.

**CommTopology** (`parallel/comm.py`):
- Deterministic face-to-rank mapping: `face_to_rank(face, n_processes) = face // (6 // n_processes)`
- Valid decompositions: 1, 2, 3, or 6 MPI ranks (must divide 6 faces evenly)
- `neighbor_ranks[(face, edge)]` — which rank owns each neighbor
- `neighbor_info[(face, edge)]` → `(nbr_face, nbr_edge, is_reversed)` — full connectivity

**MPI Halo Exchange** (`parallel/halo_exchange.py`):
- Edges to same neighbor rank packed into single send buffer → reduces MPI messages from ~4/face to ~2–3/rank
- Canonical edge ordering for packed correctness: sender sorts by `(face, edge)`, receiver by `(nbr_face, nbr_edge)`
- Tag encoding: `send_tag = rank * 1000 + nbr_rank` for uniqueness
- Local edges handled without MPI (direct array indexing)

**State management**:
- `partition_state(state, topology)` — zeros non-local faces, keeps full `(6,n,n)` shape
- `gather_state(state, topology)` — `mpi4jax.allreduce(SUM)` reconstructs global state

### 6.3 Mixed Precision

```python
from jax import dtypes

# Dynamics: float32 for accuracy
dynamics_dtype = jnp.float32

# ML parameterizations: bfloat16 for speed
ml_dtype = jnp.bfloat16

# Conservation fixer: float64 for global sums (accumulated round-off)
conservation_dtype = jnp.float64

# Automatic casting at module boundaries
def cast_for_physics(state: State) -> State:
    return jax.tree.map(lambda x: x.astype(ml_dtype), state)

def cast_for_dynamics(state: State) -> State:
    return jax.tree.map(lambda x: x.astype(dynamics_dtype), state)
```

### 6.4 Apple Silicon Support (`parallel/metal.py`, `core/hardware.py`)

**Backend detection**: `detect_devices()` returns backend, n_devices, supports_f64, distributed status.
Backend priority: METAL → GPU → TPU → CPU.

**Metal limitations**:
- No float64 or complex128 support (MPS backend)
- Spectral transforms require complex128 → automatically routed to CPU device
- `get_metal_config()` → `MetalConfig(is_metal, metal_device, cpu_device)`
- `to_cpu(array)` — transfers array to CPU device for unsupported operations

**Spectral on Metal**: Grid and SH transforms batched on CPU device; state transferred once, all steps computed, result transferred back to avoid per-step GPU↔CPU overhead.

---

## 7. I/O & Data Pipeline

### 7.1 Primary Format: Zarr

```python
import zarr
import xarray as xr

def save_state(state: ModelState, path: str, time_index: int):
    """Save model state to Zarr store."""
    ds = state_to_xarray(state)
    ds.to_zarr(path, mode='a', append_dim='time')

def load_state(path: str, time_index: int = -1) -> ModelState:
    """Load model state from Zarr store."""
    ds = xr.open_zarr(path)
    return xarray_to_state(ds.isel(time=time_index))
```

### 7.2 ERA5 Initialization

```python
class ERA5Initializer:
    """Initialize legoESM from ERA5 or IFS analysis data."""

    def __init__(self, era5_path: str, grid: Grid):
        self.era5_path = era5_path
        self.grid = grid
        self.regridder = build_regridder(era5_grid='N320', target_grid=grid)

    def initialize(self, date: str) -> AtmosphereState:
        """Create initial state from ERA5 for a given date.

        Steps:
        1. Load ERA5 fields (u, v, T, q, sp, z) from Zarr
        2. Regrid from ERA5 lat-lon to model grid
        3. Interpolate vertically to model levels
        4. Compute derived fields (rho, theta, etc.)
        5. Apply conservation fixer to balance
        """
        ...
```

### 7.3 Online Diagnostics

Computed during runtime at configurable intervals:

| Diagnostic | Variables | Level |
|------------|-----------|-------|
| Surface maps | T_2m, u_10m, v_10m, MSLP, precipitation | Surface |
| Pressure-level maps | Z, T, q, u, v | 500 hPa, 850 hPa |
| Global means | Total energy, total mass, total moisture | Scalar |
| Conservation budget | Energy/mass/moisture residuals per timestep | Scalar |
| Spectra | Kinetic energy spectrum vs. wavenumber | Global |

```python
class DiagnosticsManager:
    def __init__(self, config: DiagConfig):
        self.output_interval = config.output_interval  # seconds
        self.variables = config.variables

    def compute(self, state: ModelState, grid: Grid, time: float) -> dict:
        diags = {}
        diags['T_2m'] = interpolate_to_2m(state.temperature, state.geopotential, grid)
        diags['mslp'] = compute_mslp(state.p_surface, state.temperature, state.geopotential)
        diags['precip'] = state.precipitation_rate
        diags['total_energy'] = global_integral(total_energy(state), grid)
        diags['total_mass'] = global_integral(state.p_surface, grid) / g
        return diags
```

---

## 8. Software Engineering

### 8.1 Language & Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| **Python** | >= 3.11 | Runtime |
| **JAX** | >= 0.4.35 | Core compute framework |
| **jaxlib** | matching JAX | XLA backends |
| **jax-metal** | latest | Apple Silicon GPU support |
| **Equinox** | >= 0.11 | Neural network modules (pytree-based) |
| **Optax** | >= 0.2 | Optimizers for training/DA |
| **xarray** | >= 2024.0 | Data structures for I/O |
| **zarr** | >= 2.18 | Storage format |
| **matplotlib** | >= 3.9 | Visualization |
| **cartopy** | >= 0.23 | Map projections |
| **mpi4jax** | >= 0.4 | Multi-node parallelism (Phase 2) |
| **pytest** | >= 8.0 | Testing |
| **pyyaml** | >= 6.0 | Configuration |

### 8.2 Dependency Management

**uv** with `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "legoesm"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "jax>=0.4.35",
    "jaxlib>=0.4.35",
    "equinox>=0.11",
    "optax>=0.2",
    "xarray>=2024.0",
    "zarr>=2.18",
    "matplotlib>=3.9",
    "cartopy>=0.23",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
metal = ["jax-metal"]
mpi = ["mpi4jax>=0.4"]
dev = ["pytest>=8.0", "pytest-xdist", "ruff", "mypy", "pre-commit"]
docs = ["sphinx", "myst-parser"]

[project.scripts]
legoesm = "legoesm.cli:main"
```

### 8.3 Docker Support

```dockerfile
# Multi-stage build
FROM nvidia/cuda:12.4-cudnn9-runtime-ubuntu22.04 AS base
# ... JAX + CUDA setup ...

FROM base AS dev
# ... dev dependencies, testing tools ...

FROM base AS production
# ... minimal runtime ...
```

Also provide: `docker-compose.yml` for multi-GPU and a lightweight CPU-only image.

### 8.4 Code Quality

- **Formatter**: `ruff format`
- **Linter**: `ruff check`
- **Type checking**: `mypy` with strict mode
- **Pre-commit hooks**: ruff + mypy + pytest (fast subset)
- **CI/CD**: GitHub Actions (lint, type check, unit tests, integration tests)

---

## 9. API Design

### 9.1 Python API

```python
import legoesm as desm

# ---- Quick start: Create and run a model ----
grid = desm.grids.CubedSphere(resolution="C48", n_levels=37)
atmosphere = desm.atmosphere.DynamicalCore(
    grid=grid,
    equations="fully_compressible",  # or "hydrostatic", "anelastic"
    advection="ppm",                 # or "weno5"
    time_integrator="imex_ark",      # or "ssp_rk3", "split_explicit"
)

# Initialize from ERA5
initializer = desm.io.ERA5Initializer(grid=grid, path="/data/era5/")
state = initializer.initialize(date="2024-01-15T00:00")

# Run a 24-hour forecast
trajectory = atmosphere.integrate(
    state,
    duration=desm.hours(24),
    dt=desm.minutes(15),
    diagnostics=desm.diagnostics.default_weather(),
)

# Save output
trajectory.to_zarr("output/forecast_20240115.zarr")

# ---- Differentiable: compute gradients ----
def forecast_loss(params, state, observations):
    atm = desm.atmosphere.DynamicalCore(grid=grid, params=params)
    trajectory = atm.integrate(state, duration=desm.hours(24), dt=desm.minutes(15))
    return desm.losses.rmse(trajectory, observations)

grads = jax.grad(forecast_loss)(params, state, observations)

# ---- Coupled model ----
model = desm.CoupledModel(
    atmosphere=atmosphere,
    ocean=desm.ocean.PrimitiveEquation(grid=ocean_grid),
    land=desm.land.BucketModel(grid=land_grid),
    ice=desm.ice.ThermodynamicSeaIce(grid=ice_grid),
    coupler=desm.coupler.FluxCoupler(coupling_dt=desm.hours(1)),
)
model.run(duration=desm.years(1), dt_atm=desm.minutes(15), dt_ocn=desm.hours(1))
```

### 9.2 YAML Configuration

```yaml
# config/weather_forecast.yaml
model:
  name: "legoESM weather forecast"
  type: coupled  # or atmosphere_only, ocean_only

grid:
  type: cubed_sphere
  resolution: C384           # ~25 km
  n_levels: 91
  vertical_coord: hybrid_sigma_pressure
  model_top_pa: 1.0          # 0.01 hPa

atmosphere:
  equations: fully_compressible
  advection: ppm
  time_integrator: imex_ark
  dt_seconds: 300

  physics:
    radiation:
      type: placeholder      # or "rrtmgp", "neural_emulator"
    convection:
      type: placeholder
    microphysics:
      type: placeholder
    boundary_layer:
      type: placeholder

ocean:
  type: prescribed_sst
  sst_path: "/data/sst_climatology.zarr"

land:
  type: bucket
  soil_layers: 4
  snow: true

ice:
  type: thermodynamic
  layers: 3

coupler:
  coupling_dt_seconds: 1800  # 30 minutes

initialization:
  source: era5
  path: "/data/era5/"
  date: "2024-01-15T00:00"

time:
  duration_hours: 240        # 10 days
  output_interval_hours: 6

diagnostics:
  - T_2m
  - u_10m
  - v_10m
  - mslp
  - precipitation
  - geopotential_500hPa
  - temperature_850hPa
  - total_energy
  - total_mass

output:
  format: zarr
  path: "output/"

hardware:
  precision:
    dynamics: float32
    ml: bfloat16
    conservation: float64
  devices: auto              # auto-detect GPUs
```

### 9.3 Command-Line Interface

```bash
# Run a forecast
legoesm run config/weather_forecast.yaml

# Run with overrides
legoesm run config/weather_forecast.yaml --grid.resolution C48 --time.duration_hours 24

# Initialize from ERA5
legoesm init --source era5 --date 2024-01-15 --grid C384 --output initial_state.zarr

# Run Williamson test case
legoesm test williamson --case 2 --resolution C48 --days 5

# Benchmark performance
legoesm benchmark --grid C384 --n-steps 100 --devices 4
```

### 9.4 Jupyter Notebook Support

```python
# Inline visualization
model = desm.from_config("config/weather_forecast.yaml")
state = model.initialize()

# Step-by-step with visualization
for i in range(10):
    state = model.step(state)
    if i % 5 == 0:
        desm.plot.global_map(state.temperature, level=0, title=f"Step {i}")
```

---

## 10. Testing & Validation

### 10.1 Testing Pyramid

```
                /\
               /  \     Validation vs ERA5/Obs
              /    \    (weekly, GPU)
             /------\
            /        \   Integration tests
           /          \  (component runs, conservation checks)
          /            \ (per-PR, GPU)
         /--------------\
        /                \  Unit tests
       /                  \ (every function, operators, grids)
      /                    \ (per-commit, CPU)
     /________________________\
```

### 10.2 Unit Tests

```python
# tests/test_operators.py
def test_divergence_theorem(cubed_sphere_grid):
    """Integral of divergence over domain = boundary flux (= 0 for global)."""
    vector_field = random_vector_field(grid)
    div_integral = global_integral(divergence(vector_field, grid), grid)
    assert jnp.abs(div_integral) < 1e-10

def test_gradient_curl_identity(cubed_sphere_grid):
    """curl(gradient(f)) = 0 for any scalar field."""
    scalar = random_scalar_field(grid)
    curl_grad = curl(gradient(scalar, grid), grid)
    assert jnp.allclose(curl_grad, 0, atol=1e-10)

def test_conservation_fixer_preserves_mass(state, grid):
    """Conservation fixer must exactly preserve global mass."""
    mass_before = global_integral(state.p_surface, grid)
    state_fixed = apply_conservation_fixer(state, state, grid)
    mass_after = global_integral(state_fixed.p_surface, grid)
    assert jnp.abs(mass_before - mass_after) < 1e-12

def test_differentiability_through_step():
    """jax.grad must work through a single model step."""
    def loss(params):
        state = make_test_state(params)
        state_new = model.step(state, dt=300.0)
        return jnp.mean(state_new.theta.data ** 2)
    grads = jax.grad(loss)(params)
    assert all(jnp.isfinite(g).all() for g in jax.tree.leaves(grads))
```

### 10.3 Williamson Test Cases (Milestone 1)

| Test | Description | Validation Criteria |
|------|-------------|-------------------|
| **Test 2** | Steady-state nonlinear zonal geostrophic flow | Height field error < 1e-5 after 5 days |
| **Test 5** | Zonal flow over isolated mountain | Conservation: mass < 1e-12, energy < 1e-8; qualitative comparison to reference |

### 10.4 Held-Suarez Test (Milestone 2)

- 1200-day integration with Newtonian relaxation + Rayleigh friction
- Validate: zonal-mean zonal wind (jet structure), temperature, eddy statistics
- Compare to published Held-Suarez benchmarks

### 10.5 Bitwise Reproducibility

```python
def test_reproducibility_across_runs():
    """Same inputs must produce identical outputs."""
    state1 = model.step(initial_state, dt=300.0)
    state2 = model.step(initial_state, dt=300.0)
    assert jnp.array_equal(state1.theta.data, state2.theta.data)
```

### 10.6 Conservation Tests (Continuous)

Run alongside every integration test:

```python
def check_conservation(trajectory, grid, tolerances):
    mass = [global_integral(s.p_surface, grid) for s in trajectory]
    energy = [global_integral(total_energy(s), grid) for s in trajectory]
    assert max(mass) - min(mass) < tolerances['mass']
    assert max(energy) - min(energy) < tolerances['energy']
```

---

## 11. Milestone Roadmap (Detailed)

### Milestone 1: Dynamical Core MVP (Week 1)

**Goal**: Shallow-water equations on cubed-sphere, differentiable, validated.

| Day | Task |
|-----|------|
| 1 | Repository setup, project structure, CI/CD, Docker |
| 1 | Field/State dataclasses, pytree registration |
| 2 | Cubed-sphere grid: coordinates, connectivity, metrics, areas |
| 2 | Discrete operators: gradient, divergence, curl, Laplacian |
| 3 | Shallow-water equations: tendencies, SSP-RK3 time integration |
| 3 | Conservation fixer (mass + energy) |
| 4 | Williamson Test 2 (steady-state) + Test 5 (mountain) |
| 4 | `jax.grad` validation through short rollout |
| 5 | Visualization tools (global maps with Cartopy) |
| 5 | YAML config + CLI + Jupyter demo notebook |
| 5 | Apple Silicon (M4) testing + performance benchmarks |
| 5 | Documentation of architecture and API |

**Deliverables**:
- Working shallow-water model on cubed-sphere
- Williamson Test 2 and 5 passing
- `jax.grad` through 10-step rollout producing finite gradients
- Conservation: mass to machine precision, energy to < 1e-8
- Runs on CPU, NVIDIA GPU, and Apple M-series
- YAML config, Python API, CLI, and Jupyter notebook all working

### Milestone 2: 3D Primitive Equations (Week 2-3)

- Extend to full 3D with hybrid sigma-pressure vertical coordinate
- Add vertical advection (PPM), IMEX time integration
- Fully compressible + hydrostatic + anelastic options
- Held-Suarez test case validation
- Multi-GPU sharding (single node)

### Milestone 3: Physics Parameterizations (Week 4-5)

- Implement physics module interface
- Simple radiation (gray atmosphere)
- Simple boundary layer (bulk aerodynamic)
- Simple microphysics (Kessler warm rain)
- Verify AI-physics swapping works

### Milestone 4: ML Parameterization Training (Week 6-8)

- Neural network parameterization using Equinox
- Online training: differentiate through dynamics + ML physics
- ERA5 initialization pipeline
- Train ML convection/radiation emulators

### Milestone 5: Data Assimilation (Week 9-12)

- 4D-Var cost function
- Adjoint model via `jax.grad`
- Observation operators (radiance, in-situ)
- Assimilation cycling experiments

### Milestone 6: Component Coupling (Week 13-20)

- Ocean: 3D primitive equations (Veros-inspired)
- Land: Bucket + energy balance + DifferLand carbon
- Sea ice: Thermodynamic + EVP dynamics
- Coupler: Flux exchange, regridding, conservation

### Milestone 7: Climate Scale (Week 21+)

- Multi-century stability
- Carbon cycle coupling
- Ice sheet dynamics
- Multi-node scaling (mpi4jax)
- Full physics suite

---

## 12. Repository Structure (Actual)

```
legoESM/
├── pyproject.toml                     # Project metadata, dependencies (uv)
├── SPECIFICATION.md                   # This document
├── CLAUDE.md                          # AI assistant context
│
├── src/legoesm/
│   ├── __init__.py                    # Public API
│   ├── cli.py                         # Command-line interface
│   ├── config.py                      # YAML config loading + validation
│   ├── constants.py                   # Physical constants (g, R_d, c_p, Omega, etc.)
│   │
│   ├── core/                          # Core infrastructure
│   │   ├── field.py                   # Field dataclass (JAX pytree leaf)
│   │   ├── state.py                   # State containers (ShallowWaterState, etc.)
│   │   ├── operators.py               # 2D FV operators (grad, div, curl, laplacian, advection)
│   │   ├── operators_3d.py            # 3D operators (vmap over levels)
│   │   ├── conservation.py            # Conservation fixers (mass, energy, enstrophy)
│   │   ├── smooth.py                  # Smooth approximations (sigmoid_switch, etc.)
│   │   └── hardware.py                # Device detection, backend info
│   │
│   ├── grids/                         # Grid implementations
│   │   ├── cubed_sphere.py            # CubedSphereGrid (gnomonic equidistant, 6 faces)
│   │   ├── gaussian.py                # GaussianGrid + spherical harmonic transforms
│   │   ├── halo.py                    # Halo exchange (local backend, dispatch to MPI)
│   │   └── vertical.py                # SigmaCoordinate, HeightCoordinate, TerrainMetric
│   │
│   ├── timestepping/                  # Time integration
│   │   ├── ssp_rk3.py                # SSP-RK3 (3-stage strong stability preserving)
│   │   ├── split_explicit.py          # Split-explicit RK3 + acoustic/barotropic substeps
│   │   ├── semi_implicit.py           # Semi-implicit Hoskins-Simmons (spectral PE)
│   │   └── tridiagonal.py             # Tridiagonal solver (semi-implicit acoustic)
│   │
│   ├── atmosphere/                    # Atmosphere component
│   │   ├── dynamics/
│   │   │   ├── shallow_water.py       # ShallowWaterModel (FV cubed-sphere)
│   │   │   ├── primitive_eq.py        # PrimitiveEquationModel (FV hydrostatic σ)
│   │   │   ├── compressible_euler.py  # CompressibleEulerModel (FV non-hydrostatic z*)
│   │   │   ├── spectral_sw.py         # SpectralShallowWaterModel (Gaussian grid)
│   │   │   ├── spectral_pe.py         # SpectralPrimitiveEquationModel (Gaussian grid σ)
│   │   │   ├── spectral_nh.py         # SpectralCompressibleEulerModel (Gaussian grid z*)
│   │   │   ├── tracer_transport.py    # Passive tracer advection
│   │   │   ├── dcmip_transport.py     # DCMIP transport test cases
│   │   │   ├── williamson.py          # Williamson shallow-water test cases
│   │   │   └── dcmip2025/             # DCMIP-2025 test cases (1, 2, 3)
│   │   └── physics/
│   │       ├── held_suarez.py         # Held-Suarez forcing
│   │       ├── baroclinic_wave.py     # Jablonowski-Williamson baroclinic wave
│   │       └── kessler.py             # Kessler warm-rain microphysics
│   │
│   ├── ocean/                         # Ocean component
│   │   ├── __init__.py                # Re-exports (OceanModel, SpectralOceanModel, etc.)
│   │   ├── eos.py                     # Wright (1997) equation of state
│   │   ├── vertical.py                # OceanZStarCoordinate, layer thickness, Jacobian
│   │   ├── state.py                   # OceanState, OceanConfig, SpectralOceanState
│   │   ├── init.py                    # Bathymetry, initial conditions, test cases
│   │   ├── conservation.py            # Volume/heat/salt fixers (MPI-aware)
│   │   ├── dynamics/
│   │   │   ├── ocean_pe.py            # Baroclinic tendencies (FV cubed-sphere)
│   │   │   ├── barotropic.py          # Barotropic substeps (forward-backward)
│   │   │   ├── ocean_model.py         # OceanModel class (split-explicit step)
│   │   │   └── spectral_ocean_pe.py   # SpectralOceanModel (Gaussian grid)
│   │   └── physics/
│   │       └── mixing.py              # Laplacian viscosity/diffusivity + vertical diffusion
│   │
│   ├── parallel/                      # Parallelism infrastructure
│   │   ├── mesh.py                    # DeviceConfig, create_device_mesh, shard/replicate_pytree
│   │   ├── comm.py                    # CommTopology, build_comm_topology
│   │   ├── distributed.py             # MPI initialization, partition/gather state
│   │   ├── halo_exchange.py           # MPI halo exchange (packed per neighbor rank)
│   │   ├── metal.py                   # Apple Metal detection, CPU fallback
│   │   └── reductions.py              # MPI-aware global reductions
│   │
│   ├── visualization/
│   │   └── maps.py                    # Global maps (Cartopy + Mollweide projection)
│   │
│   ├── coupler/                       # Component coupling (placeholder)
│   ├── da/                            # Data assimilation (placeholder)
│   ├── diagnostics/                   # Online diagnostics (placeholder)
│   ├── io/                            # I/O (placeholder)
│   ├── land/                          # Land component (placeholder)
│   ├── ice/                           # Cryosphere (placeholder)
│   └── ml/                            # ML integration (placeholder)
│
├── tests/
│   ├── conftest.py                    # Shared fixtures
│   └── unit/
│       ├── test_core.py               # Field, operators, conservation
│       ├── test_grid.py               # Cubed-sphere grid creation, metrics
│       ├── test_shallow_water.py      # SW model: Williamson tests, differentiability
│       ├── test_primitive_eq.py       # PE model: Held-Suarez, conservation
│       ├── test_compressible_euler.py # CE model: acoustic substeps, mountain waves
│       ├── test_spectral.py           # Spectral: SH transforms, spectral SW/PE/NH
│       ├── test_parallel.py           # Device mesh, comm topology, halo dispatch, Metal
│       └── test_ocean.py              # Ocean: EOS, z-star, model step, conservation, spectral
│
├── scripts/
│   ├── run_williamson.py              # Williamson test case runner
│   ├── run_held_suarez.py             # Held-Suarez experiment
│   ├── run_dycore_tests.py            # Comprehensive dycore test suite
│   ├── run_dcmip2025.py              # DCMIP-2025 intercomparison
│   └── run_ocean_realistic.py         # Stommel gyre, baroclinic adj., Kelvin wave
│
└── notebooks/                         # Jupyter notebooks
```

---

## Appendix A: Physical Constants

```python
# legoesm/constants.py
import jax.numpy as jnp

# Fundamental
g = 9.80616                # Gravitational acceleration [m/s^2]
Omega = 7.292e-5           # Earth rotation rate [rad/s]
R_earth = 6.371229e6       # Earth mean radius [m]

# Dry air
R_d = 287.05               # Gas constant for dry air [J/(kg*K)]
c_pd = 1004.64             # Specific heat at const pressure [J/(kg*K)]
c_vd = 717.56              # Specific heat at const volume [J/(kg*K)]
kappa = R_d / c_pd         # Poisson constant (~0.2857)
p_ref = 1.0e5              # Reference pressure [Pa]

# Water
R_v = 461.51               # Gas constant for water vapor [J/(kg*K)]
c_pv = 1846.0              # Specific heat of water vapor [J/(kg*K)]
L_v = 2.501e6              # Latent heat of vaporization [J/kg]
L_s = 2.834e6              # Latent heat of sublimation [J/kg]
L_f = 3.337e5              # Latent heat of fusion [J/kg]
rho_water = 1000.0         # Density of liquid water [kg/m^3]
rho_ice = 917.0            # Density of ice [kg/m^3]

# Radiation
sigma_sb = 5.670374419e-8  # Stefan-Boltzmann constant [W/(m^2*K^4)]
S_0 = 1361.0               # Solar constant [W/m^2]

# Derived
epsilon = R_d / R_v         # Molecular weight ratio (~0.622)
```

---

## Appendix B: Williamson Test Case 2 — Specification

**Global steady-state nonlinear zonal geostrophic flow.**

Initial conditions:
- `u = u_0 * cos(lat)` where `u_0 = 2*pi*R_earth / (12 days)`
- `v = 0`
- `h = h_0 - (R_earth * Omega * u_0 + u_0^2 / 2) * sin(lat)^2 / g`
  where `h_0 = 2.94e4 / g`

Validation: After 5 days, the solution should be identical to the initial condition.
Height field L2 error norm should be < 1e-5.

## Appendix C: Williamson Test Case 5 — Specification

**Zonal flow over an isolated mountain.**

Initial conditions: Same as Test 2 but with:
- Mountain: `h_s = h_s0 * (1 - r/R)` for `r < R`, where
  - `h_s0 = 2000 m`
  - `R = pi/9` (20 degrees)
  - Center: `(3*pi/2, pi/6)` (30N, 90W in rotated coords)
  - `r = min(R, sqrt((lon - lon_c)^2 + (lat - lat_c)^2))`

Run for 15 days. Validate:
- Mass conservation to machine precision
- Total energy conservation to < 1e-6
- Qualitative comparison to published solutions

---

*End of Specification*
