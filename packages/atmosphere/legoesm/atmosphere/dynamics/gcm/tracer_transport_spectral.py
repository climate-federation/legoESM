"""Prescribed-wind tracer transport on the spectral (Gaussian) grid.

Advects an arbitrary number of passive tracers with analytically prescribed
winds using the pseudospectral (transform) method, completing the
``tracer_transport`` dynamics row of the solver registry for the
``spectral_gaussian`` backend (cubed-sphere / MPAS / lat-lon C-grid siblings
already exist).

The horizontal transport is the ADVECTIVE form, assembled from the same
Bourke (1972) / Hack & Jakob (1992) spectral flux-divergence machinery the
spectral shallow-water model uses for its mass equation
(``spectral_sw.py``; no re-derived numerics)::

    dq/dt = -V . grad(q) = -div(q V) + q div(V)

with both divergences evaluated by the pole-safe transform pair::

    div_hat(F) = (im/a) sh_analysis_oc2(F_lon cos(lat))
               -  (1/a) sh_analysis_dmu(F_lat cos(lat))

The advective form is correct for prescribed-wind transport because there is
no companion continuity equation to be mass-consistent with, and it
preserves a uniform tracer exactly even for divergent winds (the
``q div(V)`` correction cancels the flux-form source; the lat-lon C-grid
sibling makes the same choice).  For a NON-divergent wind the two forms
coincide and the global mean (the ``(n=0, m=0)`` coefficient) is invariant.

Vertical advection (optional): the shared upwind advective-form operator
``grids.vertical.vertical_advection`` (``-sigma_dot dq/dsigma``), evaluated
in grid space and transformed back — identical operator to the lat-lon
sibling.  ``vertical_scheme="fv3_remap"`` instead runs the horizontal
step alone and then the split FV3 Lagrangian remap
(``grids.vertical.vertical_remap_sigma_fv3``) in grid space, sigma_dot at
the step mid-time, followed by analysis and the Orszag-band truncation.
KNOWN DIFFERENCE (RULE 0): the grid-space remap closes every column's
tracer integral to round-off, but projecting the result back onto the
truncated band keeps only the GLOBAL integral exactly (the (0,0)
coefficient is the Gaussian-quadrature mean; tested), not each column's:
the column budget misses by the field's out-of-band content (measured
7.6e-12 absolute on 0.117 column integrals, DCMIP 1-2 at T21 L30, one
600 s step, vertical wind only).  The same projection follows the upwind
path's grid-space vertical tendency.
Pass ``sigma_dot=None`` from the wind function for pure
horizontal transport (the classic solid-body / Williamson-1 protocols).

Green's-function structure (what the unit tests pin): for solid-body
rotation about the polar axis (``u = Omega a cos(lat)``, ``v = 0``) each
spherical harmonic ``Y_l^m`` is an EIGENFUNCTION of the advection operator,

    d(q_hat_lm)/dt = -i m Omega q_hat_lm ,

so the discrete propagator must stay DIAGONAL (no mode mixing) with the
exact phase rotation ``exp(-i m Omega t)``; and the spectral
hyperdiffusion propagator is diagonal with the exact per-mode rate
``-nu (l(l+1)/a^2)^order``.  Time is embedded in the state so SSP-RK3
evaluates the prescribed wind at the correct stage times (``dtime_dt = 1``),
mirroring the lat-lon sibling.

Dtype: spectral coefficients are complex128 under the repo's
spectral-requires-x64 policy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from legoesm.core.field import Field
from legoesm.core.state import TracerState
from legoesm.grids.gaussian import (
    GaussianGrid,
    dealiasing_mask,
    sh_analysis_3d,
    sh_analysis_dmu_3d,
    sh_analysis_oc2_3d,
    sh_synthesis_3d,
    spectral_hyperdiffusion_3d,
)
from legoesm.grids.vertical import (
    SigmaCoordinate, check_tracer_vertical_scheme, vertical_advection,
    vertical_remap_sigma_fv3)
from legoesm.parallel.metal import place_spectral_grid
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import IntegrationMixin

# Prescribed wind on the Gaussian grid.
# Signature: (time, grid, sigma_coord) -> (u, v, sigma_dot)
#   u, v      : physical winds [m/s], shape (n_lat, n_lon, nlev)
#   sigma_dot : (n_lat, n_lon, nlev+1) at half levels, or None (no vertical
#               transport; sigma_coord may then be None as well).
WindFnSpectral = Callable[
    [jax.Array, GaussianGrid, "SigmaCoordinate | None"],
    tuple[jax.Array, jax.Array, "jax.Array | None"],
]


class TracerTransportSpectralConfig(NamedTuple):
    """Configuration for spectral (Gaussian-grid) tracer transport.

    Fields
    ------
    time_integrator : str
        Integrator name for ``timestepping.dispatch`` (default "ssp_rk3";
        unknown names raise there).
    hyperdiff_coeff : float
        Spectral hyperdiffusion coefficient ``nu`` [m^(2*order)/s] applied to
        the tracer coefficients (default 0.0 = OFF, so the pure-advection
        Green's-function pins hold exactly; the operator itself is
        ``grids.gaussian.spectral_hyperdiffusion_3d``).
    hyperdiff_order : int
        Laplacian power ``order`` in ``-nu (l(l+1)/a^2)^order`` (default 2).
    dealiasing_fraction : float
        Orszag-rule fraction for the quadratic advection products
        (default 0.667, matching the spectral dycores).
    vertical_scheme : str
        "upwind" (default, in-tendency) or "fv3_remap" (split remap after
        the horizontal step; see module docstring).
    """

    time_integrator: str = "ssp_rk3"
    hyperdiff_coeff: float = 0.0
    hyperdiff_order: int = 2
    dealiasing_fraction: float = 0.667
    vertical_scheme: str = "upwind"


def spectral_tracer_state(
    grid: GaussianGrid,
    q_grid: jax.Array,
    time: float = 0.0,
) -> TracerState:
    """Build a :class:`TracerState` from grid-space tracers.

    Parameters
    ----------
    q_grid : (n_lat, n_lon, nlev, n_tracers)
        Grid-space tracer mixing ratios.

    Returns
    -------
    TracerState with ``tracers`` = spectral coefficients
    ``(n_sh, nlev, n_tracers)`` (complex) and scalar ``time``.
    """
    n_lat, n_lon, nlev, n_tracers = q_grid.shape
    q_hat = sh_analysis_3d(
        grid, q_grid.reshape(n_lat, n_lon, nlev * n_tracers)
    ).reshape(-1, nlev, n_tracers)
    return TracerState(
        tracers=Field(data=q_hat, name="tracers_hat",
                      dims=("n_sh", "lev", "tracer"), units="1"),
        time=Field(data=jnp.asarray(float(time)), name="time",
                   dims=(), units="s"),
    )


def spectral_tracer_grid_values(
    grid: GaussianGrid, state: TracerState
) -> jax.Array:
    """Synthesize grid-space tracers ``(n_lat, n_lon, nlev, n_tracers)``."""
    n_sh, nlev, n_tracers = state.tracers.data.shape
    q = sh_synthesis_3d(grid, state.tracers.data.reshape(n_sh, nlev * n_tracers))
    return q.reshape(grid.n_lat, grid.n_lon, nlev, n_tracers)


def tracer_tendencies_spectral(
    state: TracerState,
    grid: GaussianGrid,
    sigma_coord: SigmaCoordinate | None,
    wind_fn: WindFnSpectral,
    config: TracerTransportSpectralConfig = TracerTransportSpectralConfig(),
) -> TracerState:
    """Advective-form spectral tracer tendencies (see module docstring).

    Returns a ``TracerState`` tendency pytree (``dq_hat_dt``, ``dtime_dt=1``).
    """
    check_tracer_vertical_scheme(config.vertical_scheme)
    t = state.time.data
    q_hat = state.tracers.data                     # (n_sh, nlev, n_tracers)
    n_sh, nlev, n_tracers = q_hat.shape
    a = grid.radius
    im_over_a = 1j * grid.ms.astype(jnp.float64) / a   # (n_sh,)
    one_over_a = 1.0 / a
    coslat = grid.cos_lat[:, None, None]           # (n_lat, 1, 1)

    # --- grid-space tracers and prescribed winds ---
    q = sh_synthesis_3d(grid, q_hat.reshape(n_sh, nlev * n_tracers))
    q = q.reshape(grid.n_lat, grid.n_lon, nlev, n_tracers)
    u, v, sigma_dot = wind_fn(t, grid, sigma_coord)
    u_cos = u * coslat                             # (n_lat, n_lon, nlev)
    v_cos = v * coslat

    # --- div(V): one transform pair, shared by every tracer ---
    div_v_hat = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, u_cos)
        - one_over_a * sh_analysis_dmu_3d(grid, v_cos)
    )                                              # (n_sh, nlev)
    div_v = sh_synthesis_3d(grid, div_v_hat)       # (n_lat, n_lon, nlev)

    # --- div(qV) per tracer: fold tracers into the passive trailing axis.
    # The reshape interleaves (lev, tracer) as [lev0/trc0, lev0/trc1, ...],
    # so per-level winds are duplicated n_tracers times with jnp.repeat to
    # stay aligned (jnp.tile would mis-align tracer <-> level; same
    # convention as the lat-lon sibling).
    q_flat = q.reshape(grid.n_lat, grid.n_lon, nlev * n_tracers)
    if n_tracers == 1:
        u_b, v_b, div_b = u_cos, v_cos, div_v
    else:
        u_b = jnp.repeat(u_cos, n_tracers, axis=-1)
        v_b = jnp.repeat(v_cos, n_tracers, axis=-1)
        div_b = jnp.repeat(div_v, n_tracers, axis=-1)
    div_qv_hat = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, q_flat * u_b)
        - one_over_a * sh_analysis_dmu_3d(grid, q_flat * v_b)
    )                                              # (n_sh, nlev*n_tracers)

    # --- advective-form correction + optional vertical advection ---
    corr_grid = q_flat * div_b                     # q * div(V)
    if sigma_dot is not None and sigma_coord is None:
        raise ValueError(
            "wind_fn returned sigma_dot but sigma_coord is None; pass "
            "the SigmaCoordinate to enable vertical transport."
        )
    if sigma_dot is not None and config.vertical_scheme != "fv3_remap":
        q4 = q                                      # (n_lat, n_lon, nlev, ntr)

        def _vert_one(q_one):
            return vertical_advection(q_one, sigma_dot, sigma_coord)

        vert = jax.vmap(_vert_one, in_axes=-1, out_axes=-1)(q4)
        corr_grid = corr_grid + vert.reshape(
            grid.n_lat, grid.n_lon, nlev * n_tracers
        )
    corr_hat = sh_analysis_3d(grid, corr_grid)

    # --- assemble; Orszag-mask the transformed products (all are quadratic
    # in gridded fields, so their transforms carry aliased power) ---
    mask = dealiasing_mask(grid, config.dealiasing_fraction)[:, None]
    dq_hat = (-div_qv_hat + corr_hat) * mask       # (n_sh, nlev*n_tracers)

    # --- optional spectral hyperdiffusion (linear, unmasked) ---
    if config.hyperdiff_coeff > 0.0:
        dq_hat = dq_hat + spectral_hyperdiffusion_3d(
            grid,
            q_hat.reshape(n_sh, nlev * n_tracers),
            config.hyperdiff_coeff,
            config.hyperdiff_order,
        )

    return TracerState(
        tracers=state.tracers.replace(
            data=dq_hat.reshape(n_sh, nlev, n_tracers)
        ),
        time=state.time.replace(data=jnp.ones_like(t)),
    )


class SpectralTracerTransportModel(IntegrationMixin):
    """Prescribed-wind tracer transport on the spectral Gaussian grid.

    Band-limit contract (mirrors ``SpectralShallowWaterModel``): the ACTIVE
    spectral space is the Orszag band ``n <= floor(dealiasing_fraction *
    n_max)``.  The tendency mask alone would FREEZE (not damp) any
    pre-existing upper-band state modes, so ``step`` additionally truncates
    the STATE after each update — for band-limited states the 0/1-mask
    multiply is an exact no-op; upper-band initial content is removed on
    the first step rather than carried frozen.

    Metal/MPS: spectral transforms require float64/complex128, so grid and
    stepping follow the shared ``place_spectral_grid`` CPU routing exactly
    like the spectral dycores (state is moved to the CPU device for the
    step and back afterwards when Metal is the default backend).

    Parameters
    ----------
    grid : GaussianGrid
    sigma_coord : SigmaCoordinate or None
        Required only when the wind function supplies ``sigma_dot``.
    wind_fn : WindFnSpectral
        ``(t, grid, sigma_coord) -> (u, v, sigma_dot)`` with physical winds
        on the Gaussian grid (``sigma_dot`` may be ``None``).
    config : TracerTransportSpectralConfig, optional
    allow_unsupported_backend : bool
        Forwarded to ``place_spectral_grid`` (Metal opt-out escape hatch).
    """

    def __init__(
        self,
        grid: GaussianGrid,
        sigma_coord: SigmaCoordinate | None,
        wind_fn: WindFnSpectral,
        config: TracerTransportSpectralConfig | None = None,
        *,
        allow_unsupported_backend: bool = False,
    ):
        self.config = config or TracerTransportSpectralConfig()
        check_tracer_vertical_scheme(self.config.vertical_scheme)
        # step() is not jitted (Metal CPU routing); jit the remap once so
        # its k-scans are not re-traced every step.
        self._remap_jit = jax.jit(
            lambda q, sd, dt, sc: vertical_remap_sigma_fv3(q, sd, sc, dt))
        placement = place_spectral_grid(
            grid, allow_unsupported=allow_unsupported_backend
        )
        self.grid = placement.grid
        self._use_cpu_for_spectral = placement.use_cpu_for_spectral
        self._cpu_device = placement.cpu_device
        self._default_device = placement.default_device
        self.sigma_coord = sigma_coord
        self.wind_fn = wind_fn
        if self.config.dealiasing_fraction > 0.0:
            self._dealias_state = dealiasing_mask(
                self.grid, self.config.dealiasing_fraction
            )
        else:
            self._dealias_state = None

    def tendencies(self, state: TracerState) -> TracerState:
        return tracer_tendencies_spectral(
            state, self.grid, self.sigma_coord, self.wind_fn, self.config,
        )

    def _truncate(self, state: TracerState) -> TracerState:
        """Orszag-band state truncation (see class docstring)."""
        if self._dealias_state is None:
            return state
        mask = self._dealias_state[:, None, None]
        return state._replace(
            tracers=state.tracers.replace(data=state.tracers.data * mask)
        )

    def step(self, state: TracerState, dt: float) -> TracerState:
        """Advance one time step (CPU-routed on Metal, then truncated)."""

        def tendency_fn(s):
            return tracer_tendencies_spectral(
                s, self.grid, self.sigma_coord, self.wind_fn, self.config,
            )

        integrator = self.config.time_integrator
        if self._use_cpu_for_spectral:
            state_cpu = jax.device_put(state, self._cpu_device)
            result = dispatch_integrator(state_cpu, tendency_fn, dt, integrator)
            result = self._vertical_remap(state_cpu, result, dt)
            return jax.device_put(result, self._default_device)
        result = dispatch_integrator(state, tendency_fn, dt, integrator)
        return self._vertical_remap(state, result, dt)

    def _vertical_remap(self, state0: TracerState, new: TracerState,
                        dt: float) -> TracerState:
        """Split FV3 vertical remap (``vertical_scheme="fv3_remap"``) of the
        horizontally advected, NOT yet truncated state ``new``, then the
        Orszag truncation (the truncation alone for "upwind").  Remapping
        before truncating keeps the out-of-band content in the remap's
        column budget; only the final projection drops it."""
        if self.config.vertical_scheme != "fv3_remap":
            return self._truncate(new)
        _, _, sigma_dot = self.wind_fn(state0.time.data + 0.5 * dt,
                                       self.grid, self.sigma_coord)
        if sigma_dot is None:              # pure horizontal transport
            return self._truncate(new)
        if self.sigma_coord is None:
            raise ValueError(
                "wind_fn returned sigma_dot but sigma_coord is None; pass "
                "the SigmaCoordinate to enable vertical transport.")
        sigma_coord = self.sigma_coord
        if self._use_cpu_for_spectral:
            # Metal CPU routing: every remap input on the CPU device with
            # the (CPU) tracer state, so the jitted remap never mixes
            # committed devices.
            sigma_dot = jax.device_put(sigma_dot, self._cpu_device)
            sigma_coord = jax.device_put(sigma_coord, self._cpu_device)
        q = self._remap_jit(spectral_tracer_grid_values(self.grid, new),
                            sigma_dot, dt, sigma_coord)
        q_hat = spectral_tracer_state(self.grid, q).tracers.data
        return self._truncate(new._replace(
            tracers=new.tracers.replace(data=q_hat)))
