# LegoESM 4D-Var Data Assimilation Implementation — Claude Code Prompt

You are implementing a production-grade 4D-Var data assimilation system for legoESM, a fully differentiable Earth system model built on JAX. The key advantage of legoESM is that the adjoint model is **free** — `jax.grad` through `jax.lax.scan`-based time integration gives us the exact adjoint without hand-coding a single line of adjoint code. Your implementation must exploit this fully.

The system must work for **any grid** (spectral Gaussian, lat-lon FV, cubed-sphere C-D grid, MPAS Voronoi) and **both atmosphere and ocean** models, using the existing `GridProtocol` and state NamedTuple abstractions.

---

## CRITICAL DESIGN CONSTRAINTS

1. **Everything must be differentiable end-to-end.** The entire 4D-Var cost function — background term, observation operators, observation-space mismatch, model integration — must be traceable by `jax.grad`. No Python-side loops over time steps inside the differentiated region (use `jax.lax.scan`). No `numpy` (use `jnp`). No host-side conditionals on array values.

2. **Grid-agnostic.** All observation operators, background error covariance, and control-variable transforms must work through `GridProtocol` (`grid.to_columns`, `grid.from_columns`, `grid.grid_area`, `grid.grid_lat`, `grid.grid_lon`, `grid.grid_n_columns`, `grid.grid_shape_2d`). Never hard-code `(6, n, n)` or `(n_lat, n_lon)`.

3. **Memory-efficient for long windows.** Use `jax.checkpoint` (gradient checkpointing / rematerialization) to bound memory for assimilation windows of O(100) time steps. The user must be able to control the checkpoint granularity.

4. **Composable with existing infrastructure.** Use the existing `Field`, `NamedTuple`-based states, `IntegrationMixin.integrate_scan`, conservation fixers, and `ensemble.py` for hybrid methods. Do not duplicate existing functionality.

5. **Support incremental 4D-Var.** The outer loop linearizes around a trajectory; the inner loop minimizes a quadratic approximation. This is how operational NWP centers (ECMWF, Météo-France, JMA) do it — and it maps beautifully onto JAX's AD.

6. **JIT-compile the inner loop.** The L-BFGS or conjugate-gradient minimization in the inner loop must run entirely on-device via `jax.lax.while_loop` or `jax.lax.fori_loop` — no Python-side optimizer loops that break XLA compilation.

---

## REPOSITORY CONTEXT

### Existing infrastructure you MUST use:

**States (all NamedTuple pytrees, fully AD-compatible):**
```
src/legoesm/core/state.py:
  ShallowWaterState(h, u, v, h_s)
  HydrostaticState(u, v, T, p_s, phis, tracers)
  FV3HydrostaticState(u_d, v_d, T, p_s, phis, tracers)
  NonHydrostaticState(u, v, w, theta_prime, rho_prime, phis, tracers)
  MPASShallowWaterState(h, u, h_s)
  MPASHydrostaticState(u, T, p_s, phis)

src/legoesm/ocean/state.py:
  OceanState(u, v, T, S, eta, H_bathy, land_mask)
  SpectralOceanState(vor_hat, div_hat, T_hat, S_hat, eta_hat, H_bathy_hat, land_mask_grid)
  LatLonOceanState(u, v, T, S, eta, H_bathy, land_mask)
  MPASOceanState(u, T, S, eta, H_bathy, land_mask)
```

All state fields are `Field` objects (JAX pytree leaves with `.data` array and metadata). To get/set the raw array: `state.T.data` / `state.T.replace(data=new_array)`.

**Grid protocol** (`src/legoesm/grids/protocol.py`):
```python
class GridProtocol(Protocol):
    grid_lat: jax.Array           # Cell-center latitudes
    grid_lon: jax.Array           # Cell-center longitudes
    grid_area: jax.Array          # Cell areas [m²]
    grid_total_area: jax.Array    # Sum of areas
    grid_coriolis: jax.Array      # f = 2Ω sin(lat)
    grid_radius: float            # Planet radius [m]
    grid_n_columns: int           # Total horizontal DOFs
    grid_shape_2d: tuple[int, ...]  # Horizontal shape before flattening
    def to_columns(self, field) -> jax.Array: ...   # Flatten to (ncol, ...)
    def from_columns(self, cols) -> jax.Array: ...   # Unflatten
```

Grid implementations: `CubedSphereGrid` (shape `(6,n,n)`), `LatLonGrid` (`(n_lat,n_lon)`), `GaussianGrid` (`(n_lat,n_lon)`), `VoronoiMesh` (`(nCells,)`).

**Time integration** (`src/legoesm/timestepping/integration.py`):
```python
class IntegrationMixin:
    def integrate_scan(self, state, n_steps, dt) -> (final_state, trajectory):
        # Uses jax.lax.scan — fully differentiable
```

**Conservation fixers** (`src/legoesm/core/conservation.py`):
- `fix_mass_hydrostatic`, `fix_mass_shallow_water`, `zero_mean_tendency`, etc.
- All grid-agnostic via `_global_area_sum(array, grid)` and `_total_area(grid)`.

**Ensemble infrastructure** (`src/legoesm/parallel/ensemble.py`):
- `perturb_initial_conditions(state, key, n_members, scale)`
- `make_ensemble_step(step_fn)` — vmap over ensemble dimension
- `ensemble_integrate(step_fn, init_states, n_steps, dt, checkpoint=True)`
- `ensemble_mean`, `ensemble_std`, `ensemble_spread`

**Existing loss functions** (`src/legoesm/ml/loss.py`):
- `area_weighted_mse(pred, target, weights, mask)` — area-weighted MSE
- `spectral_loss(pred_hat, target_hat)` — L2 in spectral space

**DA stub** (`src/legoesm/da/__init__.py`): Currently empty. This is where your code goes.

---

## FILE STRUCTURE TO CREATE

```
src/legoesm/da/
├── __init__.py                    # Public API exports
├── control_vector.py              # Control variable ↔ state transforms
├── background_error.py            # B matrix: diagonal, diffusion-based, spectral
├── observation.py                 # Observation containers, operators, R matrix
├── cost_function.py               # J(x) = J_b + J_o, with jax.grad
├── minimizer.py                   # L-BFGS and CG inner-loop solvers (on-device)
├── incremental.py                 # Incremental 4D-Var outer/inner loop driver
├── cycling.py                     # Assimilation cycling (analysis ↔ forecast)
├── preconditioning.py             # Change-of-variable preconditioning (B^{1/2})
└── _diagnostics.py                # Cost function evolution, gradient norms, obs-fit

tests/da/
├── __init__.py
├── test_control_vector.py         # Round-trip, grid-agnostic, differentiability
├── test_background_error.py       # Symmetry, positive-definiteness, spectral decay
├── test_observation.py            # Operator linearity, H(x_true)≈y, R symmetry
├── test_cost_function.py          # Gradient correctness (finite-diff check), J≥0
├── test_minimizer.py              # Rosenbrock, quadratic bowl, convergence rate
├── test_incremental.py            # Single-obs experiment, twin experiment
├── test_cycling.py                # Multi-cycle drift check
├── test_preconditioning.py        # B^{1/2} @ B^{1/2}.T = B, condition number
├── integration/
│   ├── test_sw_4dvar.py           # Shallow water twin experiment (all grids)
│   ├── test_pe_4dvar.py           # Hydrostatic PE twin experiment
│   ├── test_ocean_4dvar.py        # Ocean twin experiment
│   └── test_end_to_end.py         # Full cycling with synthetic obs
```

---

## MODULE-BY-MODULE SPECIFICATION

### 1. `control_vector.py` — Control Variable Transforms

The control vector is a **flat 1D JAX array** that maps bijectively to the model state. This is what the minimizer operates on.

```python
class ControlVectorSpec(NamedTuple):
    """Specification mapping state fields → control vector slices."""
    entries: tuple[ControlEntry, ...]   # Ordered list of (field_name, slice, shape, transform)
    total_size: int                      # Length of flat control vector

class ControlEntry(NamedTuple):
    field_name: str           # e.g. "T", "u", "p_s"
    offset: int               # Start index in flat vector
    size: int                 # Number of elements
    shape: tuple[int, ...]    # Original field shape
    transform: str            # "identity", "log" (for p_s), "bounded" (for q)

def build_control_spec(
    state,                     # Any NamedTuple state
    grid: GridProtocol,
    fields: tuple[str, ...] | None = None,  # Which fields to include (default: all prognostic)
    transforms: dict[str, str] | None = None,  # Per-field transforms
) -> ControlVectorSpec:
    """Build control vector specification from a template state.

    Must handle:
    - Field objects (extract .data, record .shape)
    - Static fields (phis, H_bathy, land_mask) excluded by default
    - dict fields (tracers) flattened with stable key ordering
    - Any grid shape — uses jnp.prod to compute sizes
    """

def state_to_control(state, spec: ControlVectorSpec) -> jax.Array:
    """Flatten state → 1D control vector. Differentiable."""

def control_to_state(x: jax.Array, spec: ControlVectorSpec, template_state) -> state:
    """Unflatten 1D control vector → state NamedTuple. Differentiable.

    Static fields (phis, H_bathy, land_mask) are copied from template_state.
    """

def control_to_increment(dx: jax.Array, spec: ControlVectorSpec, template_state) -> state:
    """Unflatten a control-space increment into a state-shaped increment.

    Like control_to_state but for perturbations (no static field copy).
    """
```

**Key implementation detail**: The transforms must be differentiable. For `"log"` (surface pressure), use `jnp.log` / `jnp.exp`. For `"bounded"` (moisture, 0 ≤ q ≤ q_max), use a sigmoid or softplus transform.

**Tests** (`test_control_vector.py`):
- Round-trip: `control_to_state(state_to_control(s, spec), spec, s)` recovers `s` to atol=1e-6 for all grid types
- Differentiable: `jax.grad(lambda x: jnp.sum(state_to_control(control_to_state(x, spec, s), spec)))(x)` is identity
- Grid-agnostic: test with CubedSphereGrid(N=4), LatLonGrid(16,32), GaussianGrid(T21), VoronoiMesh(level=2)
- Handles HydrostaticState, OceanState, ShallowWaterState, NonHydrostaticState

---

### 2. `background_error.py` — Background Error Covariance

The B matrix is never formed explicitly (too large). Instead, provide `B_sqrt_multiply` (multiply by B^{1/2}) and `B_inv_multiply` (multiply by B^{-1}) in control space.

```python
class DiagonalB(NamedTuple):
    """Diagonal background error covariance (simplest)."""
    sigma: jax.Array   # Standard deviations, shape (control_size,)

    # B^{1/2} x = sigma * x
    # B^{-1} x = x / sigma²

class DiffusionB:
    """Implicit diffusion-based B (Weaver & Courtier 2001).

    B = Σ C Σ  where:
    - Σ = diagonal standard deviation matrix
    - C = correlation matrix implemented as implicit diffusion

    The diffusion is applied grid-agnostically:
    - Flatten to columns via grid.to_columns
    - Apply horizontal diffusion (Laplacian smoothing, n_iter iterations)
    - Apply vertical diffusion (tridiagonal solve)
    - Unflatten via grid.from_columns

    This gives smooth, physically motivated correlations without
    ever forming the full matrix.
    """

    def __init__(
        self,
        grid: GridProtocol,
        sigma: jax.Array,             # Standard deviations per control element
        horizontal_length_scale: float,  # Correlation length [m]
        vertical_length_scale: float | None = None,  # Vertical correlation [levels]
        n_diffusion_iter: int = 10,   # More iterations → closer to Gaussian
        spec: ControlVectorSpec = None,
    ): ...

    def sqrt_multiply(self, x: jax.Array) -> jax.Array:
        """Apply B^{1/2} to control vector x. Differentiable."""

    def inv_multiply(self, x: jax.Array) -> jax.Array:
        """Apply B^{-1} to control vector x. Differentiable."""

class SpectralB:
    """Spectral background error for GaussianGrid (spectral models).

    B is diagonal in spectral space with prescribed power spectrum:
    σ²(n) = σ₀² × (n(n+1)/n₀(n₀+1))^{-α}

    This gives isotropic correlations with length scale ~ R/n₀.
    Very efficient: just element-wise multiply in SH space.
    """

    def __init__(
        self,
        grid: GaussianGrid,
        sigma_0: float,          # Base standard deviation
        n_0: int = 10,           # Decorrelation wavenumber
        alpha: float = 2.0,      # Spectral slope
        spec: ControlVectorSpec = None,
    ): ...

    def sqrt_multiply(self, x: jax.Array) -> jax.Array: ...
    def inv_multiply(self, x: jax.Array) -> jax.Array: ...

class HybridB:
    """Hybrid B = β_s B_static + β_e B_ensemble (Hamill & Snyder 2000).

    B_ensemble comes from an ensemble of forecasts (localized sample covariance).
    B_static is any of the above (Diagonal, Diffusion, Spectral).
    β_s + β_e = 1.
    """

    def __init__(
        self,
        static_B,                    # DiagonalB, DiffusionB, or SpectralB
        ensemble_states,             # (n_members, control_size) — ensemble perturbations
        beta_static: float = 0.5,
        localization_length: float = 1000e3,  # Gaspari-Cohn localization [m]
        grid: GridProtocol = None,
    ): ...

    def sqrt_multiply(self, x: jax.Array) -> jax.Array: ...
    def inv_multiply(self, x: jax.Array) -> jax.Array: ...
```

**Key implementation details**:
- `DiffusionB.sqrt_multiply` applies n_iter half-iterations of implicit diffusion, then scales by σ. This is the square root in the sense that applying it twice gives the full correlation. Use `jax.lax.fori_loop` for the iterations.
- The horizontal diffusion coefficient κ = L² / (2 × n_iter × Δt_diff) where L is the length scale. The diffusion is applied on the column-flattened representation.
- For `HybridB`, localization uses the Gaspari-Cohn 5th-order polynomial as a function of great-circle distance. Precompute the localization matrix (sparse!) at init time.

**Tests** (`test_background_error.py`):
- `DiagonalB`: `B_inv @ B @ x ≈ x` (atol=1e-6)
- `DiffusionB`: correlation decreases with distance; `sqrt_multiply` is differentiable; symmetric (⟨Bx, y⟩ = ⟨x, By⟩)
- `SpectralB`: power spectrum matches prescribed decay; round-trip
- All B types: positive-definite (`x.T @ B_inv @ x > 0` for random x ≠ 0)
- Grid-agnostic: test DiffusionB on all 4 grid types

---

### 3. `observation.py` — Observations and Observation Operators

```python
class Observation(NamedTuple):
    """A single observation batch at one time."""
    values: jax.Array          # Observed values, shape (n_obs,)
    errors: jax.Array          # Observation error std devs, shape (n_obs,)
    time_index: int            # Which time step in the assimilation window
    operator: ObsOperator      # Forward observation operator H
    metadata: dict | None      # Optional (lat, lon, level, variable, instrument, ...)

class ObsOperator(Protocol):
    """Observation operator: state → observation space."""
    def __call__(self, state) -> jax.Array:
        """H(x): map model state to observation-equivalent values.

        Must be differentiable (jax.grad-compatible).
        Returns shape (n_obs,).
        """
        ...

# --- Concrete observation operators ---

class DirectObsOperator:
    """Direct observation of a state field at grid points.

    H(x) = x[field_name].data[indices]

    Simplest case: observations are at model grid points.
    """
    def __init__(
        self,
        field_name: str,           # e.g. "T", "u", "p_s", "eta"
        indices: tuple[jax.Array, ...],  # Multi-index into field.data
        grid: GridProtocol,
    ): ...

class InterpolatingObsOperator:
    """Observation at arbitrary (lat, lon, level) via interpolation.

    H(x) = interpolate(x[field_name].data, obs_lat, obs_lon, obs_level)

    Uses bilinear horizontal interpolation + linear vertical interpolation.
    All interpolation weights precomputed at init (static); only the
    weighted sum is inside the AD tape.
    """
    def __init__(
        self,
        field_name: str,
        obs_lat: jax.Array,        # shape (n_obs,) [radians]
        obs_lon: jax.Array,        # shape (n_obs,) [radians]
        obs_level: jax.Array | None,  # shape (n_obs,) [level index or pressure]
        grid: GridProtocol,
        vertical_coord=None,       # VerticalCoordProtocol for pressure interpolation
    ): ...

    def __call__(self, state) -> jax.Array:
        """Bilinear interpolation — differentiable."""

class ColumnIntegralObsOperator:
    """Observe a column-integrated quantity.

    H(x) = ∫ x[field].data × dp / g

    For: total column water vapor, ozone, etc.
    """
    def __init__(
        self,
        field_name: str,
        grid: GridProtocol,
        vertical_coord,            # VerticalCoordProtocol
        obs_columns: jax.Array,    # Column indices, shape (n_obs,)
    ): ...

class CompositeObsOperator:
    """Combine multiple operators for multi-variable observations."""
    def __init__(self, operators: tuple[ObsOperator, ...], slices: tuple[slice, ...]): ...

    def __call__(self, state) -> jax.Array:
        """Concatenate sub-operator outputs."""

# --- Observation error (R matrix) ---

class DiagonalR(NamedTuple):
    """Diagonal observation error covariance."""
    sigma: jax.Array   # shape (n_obs,)

    # R^{-1} d = d / sigma²

# --- Synthetic observation generation (for twin experiments) ---

def generate_synthetic_obs(
    truth_trajectory,          # List of states or stacked pytree (n_steps, ...)
    operators: tuple[ObsOperator, ...],
    time_indices: tuple[int, ...],
    error_stds: tuple[float, ...],
    key: jax.Array,            # PRNG key for noise
) -> tuple[Observation, ...]:
    """Generate synthetic observations from a truth run.

    y = H(x_true) + ε,  ε ~ N(0, R)

    For twin experiment validation.
    """
```

**Key implementation details**:
- `InterpolatingObsOperator.__init__` must find the enclosing grid cell for each observation point. For cubed-sphere, this means finding the face + local (i,j). For lat-lon, it's straightforward. For Voronoi, use nearest-neighbor lookup. **Precompute all weights at init** — the `__call__` method must be a pure function of the state (no dynamic Python).
- For cubed-sphere interpolation, use the grid's `lon` and `lat` arrays (shape `(6,n,n)`) to find the nearest 4 cell centers and compute bilinear weights. This can be done with `jnp.argmin` on great-circle distances.
- All operators must return `jax.Array` (not `Field`), since observation space has no grid structure.

**Tests** (`test_observation.py`):
- `DirectObsOperator`: `H(x) == x.T.data[indices]` exactly
- `InterpolatingObsOperator`: H(constant_field) == constant; H is differentiable; H at grid point ≈ direct obs (atol=1e-4)
- `CompositeObsOperator`: concatenation correct; differentiable
- `generate_synthetic_obs`: shapes correct; noise has expected variance
- Grid-agnostic: test all operators on all 4 grid types

---

### 4. `cost_function.py` — 4D-Var Cost Function

This is the heart of the system. The cost function J(x₀) must be a **pure function** of the control vector, differentiable by `jax.grad`.

```python
def build_cost_fn(
    model,                         # Any model with .step(state, dt)
    background: jax.Array,         # x_b in control space
    observations: tuple[Observation, ...],
    B,                             # Background error cov (DiagonalB, DiffusionB, SpectralB, HybridB)
    control_spec: ControlVectorSpec,
    template_state,                # Template for static fields
    dt: float,                     # Time step [s]
    n_steps: int,                  # Assimilation window length
    checkpoint_every: int = 1,     # Gradient checkpointing interval
) -> Callable[[jax.Array], jax.Array]:
    """Build a JIT-compilable 4D-Var cost function.

    Returns J: control_vector → scalar

    J(x) = ½ (x - x_b)ᵀ B⁻¹ (x - x_b) + ½ Σᵢ (yᵢ - Hᵢ(Mᵢ(x)))ᵀ Rᵢ⁻¹ (yᵢ - Hᵢ(Mᵢ(x)))

    where Mᵢ(x) is the model state at observation time i.
    """

def build_cost_and_grad_fn(
    model, background, observations, B, control_spec, template_state,
    dt, n_steps, checkpoint_every=1,
) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:
    """Build JIT-compiled (J, ∇J) function using jax.value_and_grad.

    This is what the minimizer calls.
    """
```

**Internal implementation of `build_cost_fn`**:

```python
def _cost(x, background, observations, B, control_spec, template_state, model, dt, n_steps):
    # Background term
    dx = x - background
    J_b = 0.5 * jnp.sum(dx * B.inv_multiply(dx))

    # Forward model integration using scan
    state_0 = control_to_state(x, control_spec, template_state)

    def scan_step(carry, _):
        s = model.step(carry, dt)
        return s, s  # (new_carry, output_to_stack)

    # Apply checkpointing for memory efficiency
    if checkpoint_every > 1:
        scan_step = jax.checkpoint(scan_step, policy=...)

    final_state, trajectory = jax.lax.scan(scan_step, state_0, jnp.arange(n_steps))

    # Observation term: accumulate over all observation batches
    J_o = 0.0
    for obs in observations:
        # Extract state at observation time from trajectory
        state_t = jax.tree.map(lambda arr: arr[obs.time_index], trajectory)
        H_x = obs.operator(state_t)
        d = obs.values - H_x
        R_inv_d = d / (obs.errors ** 2)
        J_o = J_o + 0.5 * jnp.sum(d * R_inv_d)

    return J_b + J_o
```

**CRITICAL**: The `for obs in observations` loop is a Python-time loop over a static (known at trace time) list of observations. This is fine for `jax.jit` as long as the number of observation batches is fixed per window. If you need a dynamic number of observations, use `jax.lax.scan` over a padded observation array with a mask.

**Tests** (`test_cost_function.py`):
- `J(x_b) == J_b_only` when there are no observations (J_o = 0 at background)
- `J(x) ≥ 0` for all x (J is a sum of quadratic forms)
- Gradient finite-difference check: `|∇J_AD - ∇J_FD| / |∇J_AD| < 1e-4` (use `jax.grad` vs. centered finite differences with h=1e-5)
- `J(x_true) < J(x_b)` in a twin experiment (truth should fit observations better)
- Gradient at minimum is approximately zero
- Works for ShallowWaterState (spectral, latlon, cubedsphere, MPAS), HydrostaticState, OceanState
- `jax.jit(build_cost_and_grad_fn(...))` compiles without error
- Checkpoint_every=1 and checkpoint_every=10 give same gradient (to machine precision)

---

### 5. `minimizer.py` — On-Device Minimization

The minimizer must run **entirely on-device** inside `jax.jit`. No Python-side loops.

```python
class MinimizationResult(NamedTuple):
    x: jax.Array              # Solution
    fun: jax.Array            # Final cost value
    grad_norm: jax.Array      # Final gradient norm
    n_iter: int               # Number of iterations performed
    converged: jax.Array      # Boolean: converged?
    history: jax.Array | None # Cost at each iteration (optional)

def minimize_lbfgs(
    cost_and_grad_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    x0: jax.Array,
    max_iter: int = 50,
    gtol: float = 1e-5,        # Gradient norm tolerance
    ftol: float = 1e-8,        # Relative function decrease tolerance
    m: int = 10,               # L-BFGS memory (number of correction pairs)
    line_search: str = "wolfe",  # "wolfe" or "backtracking"
) -> MinimizationResult:
    """L-BFGS minimization implemented with jax.lax.while_loop.

    The entire optimization runs on-device (XLA-compiled).

    L-BFGS stores the last m pairs (s_k, y_k) where:
      s_k = x_{k+1} - x_k
      y_k = ∇J_{k+1} - ∇J_k

    Direction computed by two-loop recursion (Nocedal & Wright, Algorithm 7.4).

    Line search: Wolfe conditions via jax.lax.while_loop.
    """

def minimize_cg(
    cost_and_grad_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    x0: jax.Array,
    max_iter: int = 50,
    gtol: float = 1e-5,
    preconditioner: Callable | None = None,  # P^{-1} @ g
) -> MinimizationResult:
    """Preconditioned conjugate gradient (Polak-Ribière).

    For incremental 4D-Var inner loop with B^{1/2} preconditioning.
    Implemented with jax.lax.while_loop.
    """
```

**Key implementation details**:
- **L-BFGS two-loop recursion**: Store `s` and `y` in fixed-size buffers of shape `(m, control_size)`. Use `jax.lax.fori_loop` for the two-loop recursion. Use a circular buffer index to avoid shifting.
- **Wolfe line search**: `jax.lax.while_loop` that backtracks until sufficient decrease (Armijo) and curvature conditions are met. Start with α=1, backtrack by factor 0.5. Max 20 backtracks.
- **Convergence**: Stop when `||∇J|| < gtol` or `|J_k - J_{k-1}| / max(|J_k|, 1) < ftol` or `k >= max_iter`.
- The `while_loop` body must have fixed-shape state (no dynamic allocation). Pre-allocate history array of shape `(max_iter,)` and fill with `jnp.inf`.

**Tests** (`test_minimizer.py`):
- Rosenbrock function: converges to (1,1) within 100 iterations
- Quadratic bowl `½ x.T @ A @ x - b.T @ x`: exact solution in ≤ n iterations (for CG), near-exact for L-BFGS
- Gradient norm decreases monotonically (for well-conditioned problems)
- JIT-compiles without error
- Returns correct `converged` flag

---

### 6. `preconditioning.py` — Change of Variable

Operational 4D-Var uses a change of variable: minimize in `v`-space where `δx = B^{1/2} v`. This transforms the Hessian from `B^{-1} + H^T R^{-1} H` (poorly conditioned) to `I + B^{1/2} H^T R^{-1} H B^{1/2}` (much better conditioned).

```python
def preconditioned_cost_fn(
    cost_fn: Callable,         # J(x) in x-space
    B,                         # Background error covariance
    x_b: jax.Array,           # Background in control space
) -> Callable[[jax.Array], jax.Array]:
    """Build preconditioned cost function J̃(v) = J(x_b + B^{1/2} v).

    Minimizing J̃ over v is equivalent to minimizing J over x,
    with x = x_b + B^{1/2} v*.

    The gradient: ∇_v J̃ = B^{1/2,T} ∇_x J|_{x=x_b+B^{1/2}v}

    Since B^{1/2} is symmetric for our covariance models,
    this is just B^{1/2} applied to the x-space gradient.
    But jax.grad handles this automatically through the chain rule!
    """
    def J_tilde(v):
        x = x_b + B.sqrt_multiply(v)
        return cost_fn(x)
    return J_tilde
```

**Tests** (`test_preconditioning.py`):
- `J̃(0) == J(x_b)` (identity at background)
- Minimum of J̃ maps to same x* as minimum of J
- Condition number of Hessian of J̃ < condition number of Hessian of J (check numerically for small problem)
- Fewer iterations to converge with preconditioning than without

---

### 7. `incremental.py` — Incremental 4D-Var

Operational centers use incremental 4D-Var (Courtier, Thépaut & Hollingsworth 1994): linearize the model around a trajectory, solve a quadratic inner minimization, update the trajectory, repeat.

```python
class IncrementalConfig(NamedTuple):
    n_outer: int = 3               # Number of outer loop iterations
    n_inner: int = 50              # Max inner loop iterations per outer
    inner_gtol: float = 1e-5       # Inner loop gradient tolerance
    inner_method: str = "cg"       # "cg" or "lbfgs"
    use_preconditioning: bool = True
    checkpoint_every: int = 1      # Gradient checkpointing
    resolution_schedule: tuple[int, ...] | None = None  # Multi-resolution outer loops

def incremental_4dvar(
    model,                         # Model with .step()
    background_state,              # x_b as model state
    observations: tuple[Observation, ...],
    B,                             # Background error covariance
    control_spec: ControlVectorSpec,
    dt: float,
    n_steps: int,
    config: IncrementalConfig = IncrementalConfig(),
    grid: GridProtocol | None = None,
) -> tuple:  # (analysis_state, diagnostics)
    """Run incremental 4D-Var.

    Outer loop (Python, not JIT):
    1. Linearize: run forward model from current guess x^(k) to get trajectory
    2. Inner loop (JIT): minimize quadratic cost in increment space
       J̃(δx) = ½ δx^T B^{-1} δx + ½ Σ (d_i - H_i M_i δx)^T R_i^{-1} (d_i - H_i M_i δx)
       where d_i = y_i - H_i(M_i(x^(k))) is the innovation
    3. Update: x^(k+1) = x^(k) + δx*

    The inner loop's gradient is computed by jax.grad — this IS the adjoint model.
    The tangent linear model M_i δx is computed by jax.jvp (forward mode) for
    the Hessian-vector products, or equivalently jax.grad of the quadratic.

    Returns:
        (analysis_state, IncrementalDiagnostics)
    """

class IncrementalDiagnostics(NamedTuple):
    cost_history: list[jax.Array]        # J at each outer iteration
    grad_norm_history: list[jax.Array]   # ||∇J|| at each outer
    inner_iterations: list[int]          # Inner iter count per outer
    innovation_rms: list[jax.Array]      # RMS(y - H(x)) at each outer
```

**Key implementation details**:
- The outer loop is a Python loop (not JIT'd) because the trajectory changes each iteration and we want to print diagnostics.
- The inner loop is fully JIT-compiled: `jax.jit(minimize_cg(cost_and_grad_inner, ...))`.
- For the inner quadratic cost, the model integration + observation operators are called within `jax.grad`. JAX's reverse mode AD automatically produces the adjoint.
- `resolution_schedule` allows running initial outer loops at lower resolution for speed (truncate spectral coefficients or coarsen grid), then refine. This is advanced — implement as optional.

**Tests** (`test_incremental.py`):
- Single-observation experiment: place one temperature observation, verify analysis shifts toward it with magnitude proportional to B/(B+R)
- Twin experiment: generate truth, perturb IC, assimilate synthetic obs, verify analysis closer to truth than background
- Cost decreases at each outer iteration
- Works for both atmosphere and ocean

---

### 8. `cycling.py` — Assimilation Cycling

```python
class CyclingConfig(NamedTuple):
    window_length: int             # Assimilation window [time steps]
    cycle_length: int              # Forecast between windows [time steps]
    dt: float                      # Model time step [s]
    n_cycles: int = 1              # Number of DA cycles
    incremental: IncrementalConfig = IncrementalConfig()

def run_cycling(
    model,
    initial_state,
    observations_per_cycle: tuple[tuple[Observation, ...], ...],  # Obs for each cycle
    B,
    control_spec: ControlVectorSpec,
    config: CyclingConfig,
    grid: GridProtocol | None = None,
) -> tuple:  # (final_analysis, cycle_diagnostics)
    """Run sequential DA cycling.

    For each cycle:
    1. Analysis: run incremental 4D-Var on current window
    2. Forecast: integrate analysis forward to start of next window
    3. Use forecast as background for next cycle

    Returns:
        (final_analysis_state, list of per-cycle diagnostics)
    """
```

**Tests** (`test_cycling.py`):
- Multi-cycle twin experiment: analysis RMSE decreases over cycles (the system "spins up")
- Background RMSE bounded (doesn't diverge)
- Conservation: global mass drift bounded over N cycles

---

### 9. `_diagnostics.py` — Monitoring

```python
def compute_innovation_statistics(
    state,
    observations: tuple[Observation, ...],
    model, dt, n_steps, control_spec, template_state,
) -> dict:
    """Compute innovation (y - H(x)) statistics.

    Returns: {
        'innovation_mean': ...,
        'innovation_rms': ...,
        'chi_squared': Σ dᵢᵀ Rᵢ⁻¹ dᵢ / n_obs  (should be ≈ 1 for optimal DA)
    }
    """

def log_minimization_progress(result: MinimizationResult, cycle: int, outer: int) -> None:
    """Print cost, grad_norm, iteration count."""
```

---

## INTEGRATION TESTS — THE REAL VALIDATION

These are the most important tests. Each one is a self-contained twin experiment.

### `tests/da/integration/test_sw_4dvar.py`

```python
"""Shallow water 4D-Var twin experiment on all grid types.

Setup:
1. Create truth: Williamson TC2 (geostrophic flow) + small perturbation
2. Run truth forward 24 hours (N time steps)
3. Sample synthetic observations of h at random grid points every 6 hours
4. Create background: unperturbed Williamson TC2
5. Run 4D-Var (3 outer, 30 inner iterations)
6. Verify: analysis RMSE(h) < 0.5 × background RMSE(h)

Test for: SpectralShallowWaterModel, FVShallowWaterLatLonModel,
          CDGridShallowWaterModel, MPASShallowWaterModel
"""
```

### `tests/da/integration/test_pe_4dvar.py`

```python
"""Hydrostatic PE 4D-Var twin experiment.

Setup:
1. Truth: isothermal rest state + localized temperature perturbation (+5K Gaussian blob)
2. Forward 6 hours, 10-minute time steps
3. Observe T at 4 times (every 1.5 hours), 100 random grid-column/level locations
4. Background: unperturbed rest state
5. 4D-Var with DiffusionB (L=500km horizontal, 2-level vertical)
6. Verify: analysis temperature closer to truth than background
7. Verify: analysis is in hydrostatic balance (geopotential consistent)

Test for: SpectralPrimitiveEquationModel, FVLatLonPrimitiveEquationModel,
          CDGridPrimitiveEquationModel
"""
```

### `tests/da/integration/test_ocean_4dvar.py`

```python
"""Ocean 4D-Var twin experiment.

Setup:
1. Truth: rest state + warm SST anomaly (+2K in 30°×30° box)
2. Forward 5 days, 1-hour time steps
3. Observe T at surface every day (5 observation times), all ocean grid points
4. Background: rest state (no anomaly)
5. 4D-Var with DiagonalB (σ_T=1K, σ_S=0.1PSU, σ_u=0.01m/s)
6. Verify: analysis SST RMSE < 0.5 × background SST RMSE
7. Verify: conservation fixer produces bounded η drift

Test for: OceanModel (cubed-sphere), LatLonOceanModel, MPASOceanModel
"""
```

### `tests/da/integration/test_end_to_end.py`

```python
"""Full end-to-end cycling test.

3 DA cycles of atmosphere shallow water on cubed-sphere:
- 12-hour windows, 6-hourly cycling
- Height observations every 3 hours
- Verify: analysis improves cycle over cycle
- Verify: total runtime < 60 seconds on CPU at C8 resolution
- Verify: fully differentiable (jax.grad through a single cycle's cost)
"""
```

---

## WORKFLOW

Proceed in this order. Do not move to the next module until the current one's tests pass.

```
1. control_vector.py + tests → verify round-trip on all grids and state types
2. background_error.py + tests → verify B properties (symmetry, positive-def, differentiability)
3. observation.py + tests → verify H operators on all grids, synthetic obs generation
4. cost_function.py + tests → verify J properties, gradient correctness via finite-diff
5. minimizer.py + tests → verify convergence on analytical problems (Rosenbrock, quadratic)
6. preconditioning.py + tests → verify condition number improvement
7. incremental.py + tests → single-obs and twin experiments
8. cycling.py + tests → multi-cycle validation
9. Integration tests → full twin experiments on all grids

After each module, run ALL previous tests to check for regressions:
  python -m pytest tests/da/ -v --tb=short
```

---

## PHYSICAL REFERENCE: 4D-Var EQUATIONS

### Cost function:
```
J(x₀) = ½ (x₀ - x_b)ᵀ B⁻¹ (x₀ - x_b) + ½ Σᵢ [yᵢ - Hᵢ(xᵢ)]ᵀ Rᵢ⁻¹ [yᵢ - Hᵢ(xᵢ)]
```
where `xᵢ = M_{0→tᵢ}(x₀)` is the model trajectory.

### Gradient (the adjoint identity):
```
∇J = B⁻¹(x₀ - x_b) + Σᵢ Mᵢᵀ Hᵢᵀ Rᵢ⁻¹ [Hᵢ(xᵢ) - yᵢ]
```
where `Mᵢᵀ` is the adjoint of the tangent linear model. **In JAX, `jax.grad(J)` computes this automatically via reverse-mode AD.** This is the entire point.

### Incremental formulation:
At outer iteration k, with trajectory `x^(k)`:
```
δx* = argmin ½ δxᵀ B⁻¹ δx + ½ Σᵢ [dᵢ - Hᵢ Mᵢ δx]ᵀ Rᵢ⁻¹ [dᵢ - Hᵢ Mᵢ δx]
```
where `dᵢ = yᵢ - Hᵢ(x^(k)_tᵢ)` are innovations.

### Preconditioning (change of variable):
Let `δx = B^{1/2} v`. Then:
```
v* = argmin ½ vᵀ v + ½ Σᵢ [dᵢ - Hᵢ Mᵢ B^{1/2} v]ᵀ Rᵢ⁻¹ [dᵢ - Hᵢ Mᵢ B^{1/2} v]
```
The Hessian is now `I + B^{1/2} Σᵢ Mᵢᵀ Hᵢᵀ Rᵢ⁻¹ Hᵢ Mᵢ B^{1/2}` which has condition number ≤ 1 + λ_max(B) × λ_max(H^T R^{-1} H), much better than the unpreconditioned case.

### Diffusion-based B (Weaver & Courtier 2001):
```
B = Σ C Σ
C ≈ (I - κΔt∇²)^{-n}  (n diffusion steps)
```
where κ = L²/(4nΔt) gives correlation length L. The square root is:
```
B^{1/2} = Σ C^{1/2} = Σ (I - κΔt∇²)^{-n/2}
```

### Gaspari-Cohn localization (for hybrid B):
```
C₅(r/c) = { (−¼(r/c)⁵ + ½(r/c)⁴ + ⅝(r/c)³ − ⅚(r/c)² + 1)     if 0 ≤ r ≤ c
           { (1/12(r/c)⁵ − ½(r/c)⁴ + ⅝(r/c)³ + ⅚(r/c)² − 5(r/c) + 4 − ⅔c/r)  if c < r ≤ 2c
           { 0                                                         if r > 2c
```
Compact-support correlation with effective range 2c.

---

## COMMON PITFALLS TO AVOID

1. **Python loops inside JIT**: The `for obs in observations` loop in the cost function is OK because it's a static-length Python loop unrolled at trace time. But do NOT put a Python `while` loop for the minimizer — use `jax.lax.while_loop`.

2. **Non-differentiable operations**: `jnp.argmin` (for nearest-neighbor search in obs operators) is NOT differentiable. Use it only at init time to precompute indices/weights, not inside the forward pass.

3. **In-place mutation**: JAX arrays are immutable. Never do `x[i] = v`. Use `x.at[i].set(v)` (which creates a new array).

4. **Memory blowup**: A 100-step assimilation window with checkpointing=1 stores 100 full states on the backward pass. Use `jax.checkpoint` with `checkpoint_every > 1` to trade compute for memory. At `checkpoint_every=10`, memory = 10 states + recompute 10 steps per checkpoint.

5. **Static vs. traced values**: `n_steps`, `max_iter`, `m` (L-BFGS memory) must be Python ints (static), not JAX arrays. They define loop bounds and buffer sizes.

6. **Field vs. data**: The control vector operates on raw `jax.Array` data (`.data` attribute of `Field`). The `control_to_state` function must reconstruct `Field` objects with the correct metadata from the template state.

7. **Static fields**: Never include `phis`, `H_bathy`, `land_mask`, `h_s` in the control vector. These are boundary conditions, not prognostic variables. Copy them unchanged from the template state.

8. **Conservation fixers in the forward model**: If the model's `.step()` includes conservation fixers, these will be differentiated through. This is correct and desirable — the analysis should be consistent with the fixed model.

9. **float32 vs float64**: DA is sensitive to precision. Default to float32 (JAX default) but provide a flag to enable float64 (`jax.config.update("jax_enable_x64", True)`) for validation. The finite-difference gradient check REQUIRES float64.

Do not stop until every test in `tests/da/` passes, including the integration tests on all grid types.
