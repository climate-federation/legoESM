# legoESM: A Differentiable Earth System Model
## Technical Specification v1.0

**Project**: legoESM
**License**: MIT
**Authors**: Pierre Gentine + Claude
**Date**: 2026-02-26

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

#### 3.1.2 Supported Grids (Priority Order)

| Grid | Milestone | Notes |
|------|-----------|-------|
| **Cubed-sphere** | M1 (primary) | 6 faces, regular arrays, refinement via panel nesting |
| **Icosahedral** | M3+ | Voronoi dual, variable resolution |
| **Lat-lon** | M3+ | Simple, good for testing; polar filtering needed |
| **Spectral element** | M4+ | High-order accuracy, hp-refinement |

#### 3.1.3 Cubed-Sphere Specification (Milestone 1)

- **Gnomonic equidistant** projection on each face
- 6 faces x N x N cells per face (N configurable)
- **C-grid staggering**: scalars at cell centers, normal velocities at cell edges
- Resolution examples:
  - C48 (N=48): ~200 km
  - C192 (N=192): ~50 km
  - C384 (N=384): ~25 km (target)
  - C768 (N=768): ~13 km
  - C3072 (N=3072): ~3 km (stretch goal)
- Ghost cells / halo regions for inter-face communication
- All connectivity arrays (neighbors, edges, vertices) stored as static arrays in the pytree

#### 3.1.4 Vertical Coordinate

**Hybrid sigma-pressure** for the atmosphere:

```
p(k, i, j) = A(k) * p_top + B(k) * p_s(i, j)
```

where:
- `A(k)`, `B(k)` are level-dependent coefficients (static)
- `p_top` is model top pressure
- `p_s(i, j)` is surface pressure (prognostic)
- Near surface: terrain-following (B ~ 1)
- Near top: pure pressure levels (A ~ 1)

**Default vertical level configurations:**

| Name | Levels | Model Top | Use Case |
|------|--------|-----------|----------|
| L37 | 37 | 1 hPa | ERA5-like, testing |
| L91 | 91 | 0.01 hPa | IFS-like, weather |
| L137 | 137 | 0.01 hPa | HRES-like, high-resolution |

#### 3.1.5 Grid Refinement

- **Static refinement**: Cubed-sphere with stretched grids (Schmidt transform)
- **Panel nesting**: Higher resolution on selected faces
- Future: adaptive mesh refinement via solution-dependent criteria

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

All spatial operators are implemented as **pure functions** on the cubed-sphere C-grid:

```python
# Finite-volume operators
def divergence(flux_edge: Field, grid: CubedSphereGrid) -> Field: ...
def gradient(scalar_cell: Field, grid: CubedSphereGrid) -> Field: ...
def curl(vector_edge: Field, grid: CubedSphereGrid) -> Field: ...
def laplacian(scalar_cell: Field, grid: CubedSphereGrid) -> Field: ...

# Interpolation
def cell_to_edge(cell_field: Field, grid: CubedSphereGrid) -> Field: ...
def edge_to_cell(edge_field: Field, grid: CubedSphereGrid) -> Field: ...

# Reconstruction (for finite-volume accuracy)
def reconstruct_ppm(cell_field: Field, grid: CubedSphereGrid) -> Field:
    """Piecewise Parabolic Method (PPM) reconstruction for high-order FV."""
    ...

def reconstruct_weno(cell_field: Field, grid: CubedSphereGrid, order: int = 5) -> Field:
    """Weighted Essentially Non-Oscillatory reconstruction."""
    ...
```

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

#### 3.4.2 Available Integrators

| Integrator | Type | Use Case |
|------------|------|----------|
| **SSP-RK3** | Explicit, 3-stage Strong Stability Preserving | Default for advection |
| **IMEX-ARK** | Additive Runge-Kutta (implicit-explicit) | Stiff terms (gravity waves) implicit, rest explicit |
| **Split-explicit** | RK3 outer + forward-backward acoustic substeps | Non-hydrostatic: fast acoustic waves subcycled |
| **Leapfrog + Robert-Asselin** | Classic, with filter | Comparison / legacy |
| **Semi-implicit** | Crank-Nicolson for vertical diffusion | Vertical physics |

#### 3.4.3 IMEX Strategy (Primary)

Following MPAS and NeuralGCM:
- **Explicit part**: Horizontal advection, Coriolis, nonlinear pressure gradient
- **Implicit part**: Vertical acoustic modes (fast gravity waves), vertical diffusion
- The implicit solve is tridiagonal per column (embarrassingly parallel over columns)
- Split-explicit subcycling option for full compressible: acoustic substeps at `dt_acoustic = dt / N_substeps`

#### 3.4.4 Adaptive Timestepping

CFL-based adaptive dt with configurable safety factor:

```python
dt = cfl_safety * min(dx / (|u| + c_s))  # c_s = speed of sound
```

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

#### 4.1.1 Governing Equations (Fully Compressible Non-Hydrostatic)

The Euler equations on the rotating sphere:

```
d(rho)/dt     = -rho * div(v)                                    [continuity]
d(theta)/dt   = Q_theta / (rho * c_p)                            [thermodynamic]
d(v_h)/dt     = -theta * grad(pi') - f x v_h + F_h              [horizontal momentum]
d(w)/dt       = -theta * d(pi')/dz - g * (theta'/theta_0) + F_w [vertical momentum]
d(q_x)/dt     = S_x                                              [moisture species]
```

where:
- `pi = c_p * (p / p_0)^(R/c_p)` is the Exner function
- `theta` is potential temperature
- `f` is the Coriolis parameter
- `Q_theta`, `F_h`, `F_w`, `S_x` are source/tendency terms from physics

**Hydrostatic option**: Set `dw/dt = 0`, diagnose `w` from continuity, use hydrostatic balance for vertical pressure gradient. Activated when grid spacing > ~10 km.

**Anelastic option**: Filter acoustic waves by replacing `rho` with reference profile `rho_0(z)` in the continuity equation. Faster (no acoustic CFL constraint) but less accurate for deep convection.

#### 4.1.2 Numerical Discretization

- **Horizontal**: Finite-volume on cubed-sphere C-grid
  - PPM (Piecewise Parabolic Method) for advection (3rd-order, monotone)
  - WENO-5 option for higher accuracy
  - Compatible discretization of gradient/divergence/curl for conservation
- **Vertical**: Finite-volume in hybrid sigma-pressure with:
  - PPM vertical advection
  - Tridiagonal implicit solver for fast vertical modes

#### 4.1.3 Physics Parameterizations (Placeholder Interface)

Each parameterization follows the standard interface:

```python
class PhysicsModule(Protocol):
    """Standard interface for all physics parameterizations."""

    def __call__(
        self,
        state: AtmosphereState,
        grid: Grid,
        dt: float,
        params: PyTree,          # Learnable or physical parameters
    ) -> AtmosphereTendencies:
        """Compute tendencies from this physics process.

        Returns tendencies (rates of change) that will be summed
        and applied by the time integrator.
        """
        ...

    @property
    def is_column_local(self) -> bool:
        """Whether this module operates on independent columns."""
        ...
```

**Planned physics modules** (post-milestone 1):

| Module | Physics-based | AI-based |
|--------|--------------|----------|
| Radiation (LW/SW) | RRTMGP-like | Neural emulator |
| Deep convection | Zhang-McFarlane or SAS | ML parameterization |
| Shallow convection | CLUBB-like | ML parameterization |
| Microphysics | Morrison 2-moment | ML parameterization |
| Boundary layer | MYNN or TKE-based | ML parameterization |
| Gravity wave drag | Orographic + non-orographic | ML parameterization |
| Surface layer | Monin-Obukhov | ML parameterization |

Each module has both a physics and AI implementation, swappable at config time.

### 4.2 Ocean

#### 4.2.1 Overview

3D primitive-equation ocean model inspired by Veros (JAX) and MOM6.

#### 4.2.2 Equations

Hydrostatic primitive equations in z-coordinates with free surface:

```
d(u)/dt = -u.grad(u) - f x u - (1/rho_0) * grad(p) + A_h * laplacian(u) + d/dz(A_v * du/dz)
d(eta)/dt = -div(integral(u, dz))                    [free surface]
d(T)/dt = -u.grad(T) + K_h * laplacian(T) + d/dz(K_v * dT/dz) + Q_T
d(S)/dt = -u.grad(S) + K_h * laplacian(S) + d/dz(K_v * dS/dz) + Q_S
rho = EOS(T, S, p)                                   [equation of state]
```

#### 4.2.3 Design

- **Grid**: Same cubed-sphere infrastructure as atmosphere (shared `Grid` protocol)
- **Vertical**: z-coordinates with partial bottom cells (future: ALE)
- **Parameterizations**: Mesoscale eddy (GM90), submesoscale (Fox-Kemper), KPP mixing
- **Milestone**: Prescribed SST initially, then slab ocean, then full 3D

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

#### Phase 1: JAX Native Sharding (Milestone 1-4)

```python
from jax.sharding import NamedSharding, PartitionSpec as P, Mesh

# Create device mesh
devices = jax.devices()  # e.g., 4 GPUs
mesh = Mesh(devices, axis_names=('x',))

# Shard cubed-sphere faces across devices
# 6 faces -> shard over face dimension
state_sharding = NamedSharding(mesh, P('x', None, None))  # (face, i, j)

# JIT with sharding constraints
@jax.jit
def step(state, dt):
    ...

step = jax.jit(step, in_shardings=(state_sharding, None),
                      out_shardings=state_sharding)
```

- **Domain decomposition**: Shard the face/horizontal dimensions across devices
- **Vertical stays local**: Physics is column-local, no communication needed
- **Halo exchange**: `jax.lax.ppermute` for neighbor communication between faces/shards

#### Phase 2: mpi4jax for Multi-Node (Milestone 5+)

```python
import mpi4jax

def halo_exchange(field, grid, comm):
    """Exchange halo cells between MPI ranks."""
    for neighbor_rank, send_buf, recv_buf in grid.halo_pairs:
        recv_buf, token = mpi4jax.sendrecv(
            send_buf, recv_buf,
            source=neighbor_rank, dest=neighbor_rank,
            comm=comm
        )
    return field.with_halos(recv_buf)
```

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

### 6.4 Apple Silicon Support

```python
# Metal backend detection
def get_backend():
    available = jax.devices()
    if any(d.platform == 'gpu' for d in available):
        return 'gpu'          # NVIDIA or Metal
    elif any(d.platform == 'tpu' for d in available):
        return 'tpu'
    return 'cpu'

# M-series specific: use jax-metal package
# pip install jax-metal  (for macOS ARM)
# Limitations: some ops may fall back to CPU
```

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

## 12. Repository Structure

```
legoESM/
├── pyproject.toml                   # Project metadata, dependencies (uv)
├── uv.lock                          # Locked dependencies
├── Dockerfile                       # Multi-stage: base, dev, production
├── docker-compose.yml               # Multi-GPU orchestration
├── Makefile                         # Common commands (test, lint, run, benchmark)
├── README.md
├── LICENSE                          # MIT
├── SPECIFICATION.md                 # This document
│
├── config/                          # Example YAML configurations
│   ├── williamson_test2.yaml
│   ├── williamson_test5.yaml
│   ├── held_suarez.yaml
│   ├── weather_forecast.yaml
│   └── coupled_climate.yaml
│
├── src/
│   └── legoesm/                   # Main package
│       ├── __init__.py              # Public API
│       ├── cli.py                   # Command-line interface
│       ├── config.py                # YAML config loading + validation
│       ├── constants.py             # Physical constants (g, R_d, c_p, etc.)
│       ├── types.py                 # Type aliases, protocols
│       │
│       ├── core/                    # Core infrastructure
│       │   ├── __init__.py
│       │   ├── field.py             # Field dataclass (JAX pytree)
│       │   ├── state.py             # State containers
│       │   ├── operators.py         # Discrete operators (grad, div, curl, laplacian)
│       │   ├── interpolation.py     # Spatial interpolation
│       │   ├── conservation.py      # Conservation fixers
│       │   ├── smooth.py            # Smooth approximations (sigmoid_switch, etc.)
│       │   └── precision.py         # Mixed precision utilities
│       │
│       ├── grids/                   # Grid implementations
│       │   ├── __init__.py
│       │   ├── grid_protocol.py     # Abstract Grid protocol
│       │   ├── cubed_sphere.py      # Cubed-sphere grid (primary)
│       │   ├── latlon.py            # Lat-lon grid
│       │   ├── icosahedral.py       # Icosahedral grid (future)
│       │   ├── spectral.py          # Spectral element grid (future)
│       │   └── vertical.py          # Vertical coordinate (hybrid sigma-pressure)
│       │
│       ├── timestepping/            # Time integration
│       │   ├── __init__.py
│       │   ├── integrator.py        # TimeIntegrator protocol
│       │   ├── ssp_rk3.py           # Strong Stability Preserving RK3
│       │   ├── imex_ark.py          # IMEX Additive Runge-Kutta
│       │   ├── split_explicit.py    # Split-explicit (acoustic substeps)
│       │   └── adaptive.py          # CFL-based adaptive dt
│       │
│       ├── atmosphere/              # Atmosphere component
│       │   ├── __init__.py
│       │   ├── component.py         # AtmosphereComponent (standalone runner)
│       │   ├── state.py             # AtmosphereState, AtmosphereTendencies
│       │   ├── dynamics/
│       │   │   ├── __init__.py
│       │   │   ├── shallow_water.py # Shallow-water equations (milestone 1)
│       │   │   ├── primitive_eq.py  # Hydrostatic primitive equations
│       │   │   ├── compressible.py  # Fully compressible non-hydrostatic
│       │   │   └── anelastic.py     # Anelastic approximation
│       │   └── physics/
│       │       ├── __init__.py
│       │       ├── interface.py     # PhysicsModule protocol
│       │       ├── held_suarez.py   # Held-Suarez forcing (test)
│       │       ├── radiation/       # Placeholder
│       │       ├── convection/      # Placeholder
│       │       ├── microphysics/    # Placeholder
│       │       ├── boundary_layer/  # Placeholder
│       │       └── gravity_wave/    # Placeholder
│       │
│       ├── ocean/                   # Ocean component
│       │   ├── __init__.py
│       │   ├── component.py         # OceanComponent
│       │   ├── state.py             # OceanState
│       │   ├── prescribed_sst.py    # Prescribed SST (milestone 1)
│       │   ├── slab_ocean.py        # Slab / mixed-layer ocean
│       │   └── primitive_eq.py      # Full 3D ocean (Veros-inspired)
│       │
│       ├── land/                    # Land component
│       │   ├── __init__.py
│       │   ├── component.py         # LandComponent
│       │   ├── state.py             # LandState
│       │   ├── bucket.py            # Simple bucket model + energy balance
│       │   ├── soil.py              # Multi-layer soil T and moisture
│       │   ├── snow.py              # Snow accumulation/melt
│       │   ├── vegetation.py        # Vegetation dynamics
│       │   └── carbon/              # Carbon cycling (DifferLand-inspired)
│       │       ├── __init__.py
│       │       ├── pools.py         # Carbon pool dynamics
│       │       ├── photosynthesis.py # FvCB-Medlyn (from DifferLand)
│       │       └── respiration.py   # Auto/heterotrophic respiration
│       │
│       ├── ice/                     # Cryosphere component
│       │   ├── __init__.py
│       │   ├── component.py         # IceComponent
│       │   ├── sea_ice.py           # Thermodynamic sea ice
│       │   ├── ice_dynamics.py      # EVP rheology
│       │   ├── ice_sheet.py         # Ice sheet (future)
│       │   └── permafrost.py        # Permafrost (future)
│       │
│       ├── coupler/                 # Component coupling
│       │   ├── __init__.py
│       │   ├── coupler.py           # Flux exchange mediator
│       │   ├── regridder.py         # Grid-to-grid interpolation
│       │   └── flux_calculator.py   # Surface flux computations
│       │
│       ├── da/                      # Data assimilation
│       │   ├── __init__.py
│       │   ├── cost_function.py     # 4D-Var cost function
│       │   ├── observation.py       # Observation operators
│       │   └── minimizer.py         # L-BFGS, gradient descent wrappers
│       │
│       ├── ml/                      # ML integration
│       │   ├── __init__.py
│       │   ├── neural_physics.py    # Equinox-based neural parameterization
│       │   ├── training.py          # Online training loop
│       │   └── architectures.py     # MLP, U-Net, attention for physics
│       │
│       ├── io/                      # Input/Output
│       │   ├── __init__.py
│       │   ├── zarr_io.py           # Zarr read/write
│       │   ├── era5.py              # ERA5 initialization
│       │   └── ifs.py               # IFS analysis initialization
│       │
│       ├── diagnostics/             # Online diagnostics
│       │   ├── __init__.py
│       │   ├── manager.py           # DiagnosticsManager
│       │   ├── derived_fields.py    # T_2m, MSLP, etc.
│       │   └── spectra.py           # Energy spectra
│       │
│       ├── visualization/           # Plotting tools
│       │   ├── __init__.py
│       │   ├── maps.py              # Global/regional maps (Cartopy)
│       │   ├── cross_sections.py    # Vertical cross-sections
│       │   ├── timeseries.py        # Time series plots
│       │   └── conservation.py      # Conservation budget plots
│       │
│       └── parallel/               # Parallelism utilities
│           ├── __init__.py
│           ├── sharding.py          # JAX native sharding setup
│           ├── halo.py              # Halo exchange
│           └── mpi.py               # mpi4jax wrappers (Phase 2)
│
├── tests/                           # Test suite
│   ├── conftest.py                  # Shared fixtures
│   ├── unit/
│   │   ├── test_field.py
│   │   ├── test_grid.py
│   │   ├── test_operators.py
│   │   ├── test_conservation.py
│   │   ├── test_timestepping.py
│   │   ├── test_smooth.py
│   │   └── test_differentiability.py
│   ├── integration/
│   │   ├── test_shallow_water.py
│   │   ├── test_williamson.py
│   │   ├── test_held_suarez.py
│   │   └── test_coupled.py
│   └── validation/
│       ├── test_era5_comparison.py
│       └── test_benchmarks.py
│
├── notebooks/                       # Jupyter notebooks
│   ├── 01_quickstart.ipynb
│   ├── 02_williamson_tests.ipynb
│   ├── 03_differentiable_demo.ipynb
│   └── 04_ml_parameterization.ipynb
│
├── scripts/                         # Utility scripts
│   ├── download_era5.py
│   └── benchmark.py
│
└── docs/                            # Documentation (Sphinx)
    ├── conf.py
    ├── index.rst
    ├── installation.rst
    ├── quickstart.rst
    ├── architecture.rst
    └── api/
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
