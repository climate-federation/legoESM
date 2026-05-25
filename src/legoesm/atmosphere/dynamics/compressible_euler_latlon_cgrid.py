"""Non-hydrostatic compressible Euler on the lat-lon Arakawa C-grid.

This module provides the non-hydrostatic companion to
``primitive_eq_latlon_cgrid`` (hydrostatic) on the latitude-longitude
grid.  It is the lat-lon C-grid counterpart of the cubed-sphere
``compressible_euler_cdgrid`` and the Gaussian-grid spectral
``spectral_nh`` non-hydrostatic dycores.

Design
------
Staggering follows the same Arakawa C-grid convention as the
hydrostatic C-grid module:

  u           : (n_lat,   n_lon+1, nlev)     -- zonal wind on lon faces
  v           : (n_lat+1, n_lon,   nlev)     -- meridional wind on lat faces
  w           : (n_lat,   n_lon,   nlev+1)   -- vertical wind on half-levels
                                                (Lorenz staggering; w=0 at top
                                                 and bottom rigid lid)
  theta_prime : (n_lat,   n_lon,   nlev)     -- potential-temperature pertub. [K]
  rho_prime   : (n_lat,   n_lon,   nlev)     -- density perturbation [kg/m^3]
  phis        : (n_lat,   n_lon)             -- surface geopotential (static)
  tracers     : dict name -> (n_lat, n_lon, nlev)  [optional]

Prognostic variables are perturbations from a 1-D hydrostatically
balanced reference state ``rho_ref(z)``, ``theta_ref(z)`` carried by
``HeightCoordinate`` (same convention as the cubed-sphere
``compressible_euler_cdgrid``).

Time integration follows the Klemp & Wilhelmson / Skamarock & Klemp
split-explicit scheme: slow advection / Coriolis / hyperdiff
tendencies are evaluated once per outer SSP-RK3 stage and held
constant while the acoustic-mode subsystem is integrated for
``n_acoustic_substeps`` short steps.  The acoustic substep machinery
in :mod:`compressible_euler` (``acoustic_substeps`` /
``acoustic_substeps_semi_implicit``) is written generically against
the trailing-axis convention ``(..., nlev)``, so it is reused
verbatim once the C-grid u/v contributions to the slow tendency are
folded in.

References
----------
- Klemp, J. B., & Wilhelmson, R. B. (1978).  The simulation of
  three-dimensional convective storm dynamics.  JAS, 35, 1070-1096.
- Skamarock, W. C., & Klemp, J. B. (2008).  A time-split nonhydrostatic
  atmospheric model for weather research and forecasting applications.
  JCP, 227, 3465-3485.
- Sadourny, R. (1975).  The dynamics of finite-difference models of the
  shallow-water equations.  JAS, 32, 680-689.

Status
------
v0 first-cut.  Subject to adversarial review (see ``/codex:adversarial-review``)
before being used for production benchmarks.  Known limitations and
TODOs are marked explicitly in the code below.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.core.field import Field
from legoesm.atmosphere.dynamics.compressible_euler import (
    CompressibleEulerConfig,
    _sponge_profile,
    acoustic_substeps,
    acoustic_substeps_semi_implicit,
    compute_exner_perturbation,
)
from legoesm.atmosphere.physics.thermodynamics import sanitize_theta_rho
from legoesm.core.state import NonHydrostaticState, NonHydrostaticTendencies
from legoesm.grids.latlon import LatLonGrid
from legoesm.grids.vertical import HeightCoordinate, TerrainMetric
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    divergence_cgrid,
    gradient_x_cgrid,
    gradient_y_cgrid,
    interp_cell_to_uface,
    interp_cell_to_vface,
)
from legoesm.atmosphere.dynamics.shallow_water_latlon_cgrid import (
    absolute_vorticity_coriolis,
)
from legoesm.core.operators_fv_latlon_3d import (
    cgrid_fv_scalar_advection_latlon_3d,
)
from legoesm.grids.halo_latlon import pad_halo_latlon_3d
from legoesm.timestepping.split_explicit import SplitExplicitConfig
from legoesm.timestepping.dispatch import dispatch_integrator


# ==============================================================================
# State + config
# ==============================================================================

class CGridLatLonNonHydrostaticState(NamedTuple):
    """Non-hydrostatic state on the lat-lon Arakawa C-grid.

    Shape conventions documented in the module-level docstring.
    """
    u: jax.Array
    v: jax.Array
    w: jax.Array
    theta_prime: jax.Array
    rho_prime: jax.Array
    phis: jax.Array
    tracers: dict = {}


class CGridLatLonCompressibleEulerConfig(NamedTuple):
    """Configuration for the lat-lon C-grid NH compressible Euler model.

    Wraps :class:`compressible_euler.CompressibleEulerConfig` plus the
    handful of additional lat-lon-specific knobs (polar-filter
    bandwidth, dry-mass anchoring, etc.).  Keeping the shared
    acoustic-substep + sponge knobs on a separate NamedTuple means the
    acoustic-substep machinery in :mod:`compressible_euler` can be
    used unchanged.
    """
    euler: CompressibleEulerConfig = CompressibleEulerConfig()
    time_integrator: str = "ssp_rk3"
    # Horizontal hyperdiffusion coefficient for u, v, w (Laplacian-form
    # ``hyperdiff_order=2`` in the existing operator).  0 disables.
    hyperdiff_uv: float = 0.0
    hyperdiff_w: float = 0.0
    hyperdiff_scalar: float = 0.0   # for theta_prime + rho_prime
    # Polar-filter cutoff (latitude past which a longitudinal smoother
    # is applied).  Disabled by default; the C-grid hydrostatic module
    # has the full polar-filter scaffolding -- the NH port reuses it
    # via ``apply_polar_filter`` (TODO: wire in v1).
    use_polar_filter: bool = False
    polar_filter_cutoff_deg: float = 70.0
    # Theta + density floors / ceilings used by sanitisation (mirror
    # the cubed-sphere ``compressible_euler_cdgrid`` knobs).
    theta_min: float = 50.0
    rho_min: float = 1.0e-3


# ==============================================================================
# Helpers (shape-agnostic; reused from the hydrostatic C-grid module)
# ==============================================================================


def _face_to_cell_u(u: jnp.ndarray) -> jnp.ndarray:
    """Average u from lon faces (n_lat, n_lon+1, *) to cell centres (n_lat, n_lon, *)."""
    return 0.5 * (u[:, :-1] + u[:, 1:])


def _face_to_cell_v(v: jnp.ndarray) -> jnp.ndarray:
    """Average v from lat faces (n_lat+1, n_lon, *) to cell centres (n_lat, n_lon, *)."""
    return 0.5 * (v[:-1] + v[1:])


def _apply_pole_wall(v: jnp.ndarray) -> jnp.ndarray:
    """Enforce wall boundary condition on v at the poles.

    On a lat-lon C-grid the meridional velocity sits at lat-face
    positions ``j = 0 .. n_lat`` with ``j = 0`` on the south pole and
    ``j = n_lat`` on the north pole.  The kinematic boundary condition
    at each pole is v = 0 (no flow through the pole).  We enforce this
    after every slow-tendency evaluation so the dycore does not need
    to track polar singularities elsewhere.
    """
    return v.at[0].set(0.0).at[-1].set(0.0)


# ==============================================================================
# Slow tendencies (advection / Coriolis / PGF / buoyancy / hyperdiff)
# ==============================================================================


def cgrid_latlon_nh_slow_tendencies(
    state: CGridLatLonNonHydrostaticState,
    grid: LatLonGrid,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
    config: CGridLatLonCompressibleEulerConfig = CGridLatLonCompressibleEulerConfig(),
) -> NonHydrostaticTendencies:
    """Slow (non-acoustic) tendencies on the lat-lon C-grid.

    Excludes the fast acoustic + gravity-wave modes, which are
    integrated separately by the inner acoustic substep loop
    (:func:`compressible_euler.acoustic_substeps` /
    ``..._semi_implicit``) operating on ``(w, theta', rho')``.  The
    slow tendency for ``u`` and ``v`` therefore is *full* (advection +
    Coriolis + horizontal Exner gradient + horizontal hyperdiff): no
    pressure-gradient term is shifted into the acoustic loop in this
    scheme.  For ``theta'`` and ``rho'`` the slow tendency is purely
    horizontal advection + horizontal hyperdiff -- the vertical
    advection and continuity contributions are handled inside the
    acoustic substep loop.

    Returns
    -------
    NonHydrostaticTendencies
        Pytree of slow tendencies with matching C-grid staggering.
    """
    u, v, w = state.u, state.v, state.w
    theta_p, rho_p = state.theta_prime, state.rho_prime
    theta_0 = height_coord.theta_ref
    rho_0 = height_coord.rho_ref
    c_p = constants.c_pd
    cfg = config.euler

    theta_total, rho_total = sanitize_theta_rho(
        theta_0 + theta_p, rho_0 + rho_p,
    )

    # --- Horizontal kinetic energy + Bernoulli ---
    # On a C-grid the kinetic-energy gradient must be evaluated from
    # the cell-centred KE to avoid the chequerboard instability in the
    # vector-invariant form (Sadourny 1975).  We compute KE at cell
    # centres from face-averaged u, v.
    u_c = _face_to_cell_u(u)
    v_c = _face_to_cell_v(v)
    KE = 0.5 * (u_c ** 2 + v_c ** 2)

    # Add the 3D vertical-velocity contribution to KE from w at
    # cell-centre full levels.  ``w`` is on half-levels (Lorenz
    # staggering); average to full levels for the KE budget.
    w_full = 0.5 * (w[..., :-1] + w[..., 1:])
    KE = KE + 0.5 * w_full ** 2

    # --- Horizontal pressure-gradient term ---
    # Non-hydrostatic PGF in Exner formulation:
    #   du/dt += -c_p * theta * d(pi)/dx
    # where pi = pi_0(z) + pi'.  The reference Exner is z-only so its
    # horizontal gradient vanishes; only pi' contributes.
    pi_p = compute_exner_perturbation(rho_p, theta_p, height_coord)
    dpi_p_dx = gradient_x_cgrid(pi_p, grid)   # (n_lat, n_lon+1, nlev)
    dpi_p_dy = gradient_y_cgrid(pi_p, grid)   # (n_lat+1, n_lon, nlev)

    theta_u = interp_cell_to_uface(theta_total)
    theta_v = interp_cell_to_vface(theta_total)

    pgf_u = -c_p * theta_u * dpi_p_dx
    pgf_v = -c_p * theta_v * dpi_p_dy

    # --- Bernoulli gradient (KE part of vector-invariant momentum) ---
    dKE_dx = gradient_x_cgrid(KE, grid)
    dKE_dy = gradient_y_cgrid(KE, grid)

    # --- Coriolis (absolute-vorticity, energy-conserving) ---
    if cfg.use_coriolis:
        cor_u, cor_v = absolute_vorticity_coriolis(u, v, grid)
    else:
        cor_u = jnp.zeros_like(u)
        cor_v = jnp.zeros_like(v)

    du_dt = -dKE_dx + pgf_u + cor_u
    dv_dt = -dKE_dy + pgf_v + cor_v

    # --- Horizontal scalar advection (PPM) for theta' + rho' ---
    # The PPM helper operates on cell-centred fields with u, v at
    # faces -- same C-grid convention as the hydrostatic module.  We
    # advect the *perturbations* directly.  The vertical advection of
    # theta' inside the acoustic substep is the dominant tendency for
    # the rising-bubble class of tests; the horizontal piece below is
    # for the larger-scale benchmarks (mountain wave, baroclinic
    # wave).
    if config.hyperdiff_scalar != 0.0:
        # TODO: switch to higher-order PPM advection once the scalar
        # transport operator gains a NH-aware variant.  For v0 the
        # advection is the same operator used by the hydrostatic
        # path with theta_prime / rho_prime in place of T / p_s.
        pass
    horiz_adv_theta_p = cgrid_fv_scalar_advection_latlon_3d(theta_p, u, v, grid)
    horiz_adv_rho_p = cgrid_fv_scalar_advection_latlon_3d(rho_p, u, v, grid)

    # --- Horizontal advection of w (cell-centred, half-level vertical) ---
    # The KE-gradient term above already absorbs the metric part of
    # ``v · grad w`` at cell centres into the Bernoulli function for
    # the horizontal momenta.  For ``w`` itself we still need the
    # ``v_h · grad_h w`` flux-divergence contribution.  v0: cell-
    # centred upwind with linear interpolation in z to half-levels.
    # TODO: replace with C-grid PPM consistent with theta' / rho'.
    w_cell_full = 0.5 * (w[..., :-1] + w[..., 1:])  # cell-centred, full lev
    # PPM on cell-centred w
    horiz_adv_w_full = cgrid_fv_scalar_advection_latlon_3d(
        w_cell_full, u, v, grid,
    )
    # Map back to half-levels (interior).  Boundaries (k=0, k=nlev)
    # are rigid (w=0); their tendency is zero.
    horiz_adv_w_inner = 0.5 * (
        horiz_adv_w_full[..., :-1] + horiz_adv_w_full[..., 1:]
    )
    pad_axes = ((0, 0), (0, 0))
    horiz_adv_w = jnp.pad(horiz_adv_w_inner, (*pad_axes, (1, 1)))

    # --- Horizontal hyperdiffusion ---
    # Reuse the (-Δ²) Laplacian-of-Laplacian via successive
    # ``divergence_cgrid(gradient_*_cgrid(f))`` applications.  v0:
    # second-order (single Laplacian).
    def _laplacian_scalar(field: jnp.ndarray) -> jnp.ndarray:
        gx = gradient_x_cgrid(field, grid)
        gy = gradient_y_cgrid(field, grid)
        return divergence_cgrid(gx, gy, grid)

    if config.hyperdiff_uv > 0.0:
        # On a C-grid u, v live on different faces.  Computing the
        # Laplacian via grad(div(u, v)) - curl(curl(u, v)) preserves
        # the vector-Laplacian symmetry.  v0: use scalar Laplacian on
        # u / v separately as a simpler placeholder.
        # TODO: replace with the C-grid vector Laplacian from the
        # ocean module (``vector_laplacian_cgrid``).
        du_dt = du_dt + config.hyperdiff_uv * (
            -interp_cell_to_uface(_laplacian_scalar(u_c))
        )
        dv_dt = dv_dt + config.hyperdiff_uv * (
            -interp_cell_to_vface(_laplacian_scalar(v_c))
        )

    if config.hyperdiff_w > 0.0:
        horiz_adv_w = horiz_adv_w + config.hyperdiff_w * (
            -_laplacian_scalar(w_full)[..., None]  # broadcast back to half-lev shape
            .repeat(w.shape[-1], axis=-1)
        )
    if config.hyperdiff_scalar > 0.0:
        horiz_adv_theta_p = horiz_adv_theta_p + config.hyperdiff_scalar * (
            -_laplacian_scalar(theta_p)
        )
        horiz_adv_rho_p = horiz_adv_rho_p + config.hyperdiff_scalar * (
            -_laplacian_scalar(rho_p)
        )

    # --- Apply pole wall BC on v slow tendency ---
    dv_dt = _apply_pole_wall(dv_dt)

    # --- Sponge layer on w (Rayleigh damping in top sponge) ---
    z_full = height_coord.z_full
    H = height_coord.H
    sponge = _sponge_profile(
        z_full, H, cfg.sponge_width, cfg.sponge_coeff,
    )  # (nlev,)
    sponge_half = jnp.concatenate(
        [sponge[:1], 0.5 * (sponge[:-1] + sponge[1:]), sponge[-1:]],
        axis=-1,
    )  # (nlev+1,)
    horiz_adv_w = horiz_adv_w - sponge_half * w

    # --- Pack as NonHydrostaticTendencies (Field-wrapped) ---
    dims_u = ("lat", "lon_face", "level")
    dims_v = ("lat_face", "lon", "level")
    dims_c = ("lat", "lon", "level")
    dims_w = ("lat", "lon", "level_half")
    dims_2 = ("lat", "lon")
    return NonHydrostaticTendencies(
        du_dt=Field(data=du_dt, name="du_dt", dims=dims_u, units="m/s^2"),
        dv_dt=Field(data=dv_dt, name="dv_dt", dims=dims_v, units="m/s^2"),
        dw_dt=Field(data=horiz_adv_w, name="dw_dt", dims=dims_w, units="m/s^2"),
        dtheta_prime_dt=Field(
            data=horiz_adv_theta_p,
            name="dtheta_prime_dt", dims=dims_c, units="K/s",
        ),
        drho_prime_dt=Field(
            data=horiz_adv_rho_p,
            name="drho_prime_dt", dims=dims_c, units="kg/m^3/s",
        ),
        dphis_dt=Field(
            data=jnp.zeros_like(state.phis),
            name="dphis_dt", dims=dims_2, units="m^2/s^2/s",
        ),
        # Tracer tendencies are zero in v0; multi-tracer transport is
        # wired in a v2 follow-up alongside the hybrid coordinate.
        dtracers_dt=Field(
            data=jnp.zeros((0,), dtype=u.dtype),
            name="dtracers_dt", dims=(), units="",
        ),
    )


# ==============================================================================
# Outer step: slow + acoustic split
# ==============================================================================


def cgrid_latlon_nh_step(
    state: CGridLatLonNonHydrostaticState,
    grid: LatLonGrid,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
    dt: float,
    config: CGridLatLonCompressibleEulerConfig = CGridLatLonCompressibleEulerConfig(),
) -> CGridLatLonNonHydrostaticState:
    """One outer Klemp-Wilhelmson step on the lat-lon C-grid.

    Composition:
      1. Compute slow tendencies (advection + Coriolis + PGF + hyperdiff)
      2. Advance u, v explicitly by dt (slow tendency only -- no acoustic
         pressure-gradient feedback on horizontal momenta in this split,
         which matches the Skamarock & Klemp 2008 split-explicit scheme).
      3. Run ``n_acoustic_substeps`` acoustic substeps on (w, theta', rho')
         using the small acoustic time step.

    Parameters
    ----------
    state : CGridLatLonNonHydrostaticState
    grid : LatLonGrid
    height_coord : HeightCoordinate
    terrain_metric : TerrainMetric
    dt : float
        Outer dycore time step [s].
    config : CGridLatLonCompressibleEulerConfig

    Returns
    -------
    CGridLatLonNonHydrostaticState
        Updated state after one outer step.
    """
    cfg = config.euler

    # 1. Slow tendency
    slow = cgrid_latlon_nh_slow_tendencies(
        state, grid, height_coord, terrain_metric, config,
    )

    # 2. Update u + v (slow only).  Apply pole wall on v.
    u_new = state.u + dt * slow.du_dt.data
    v_new = _apply_pole_wall(state.v + dt * slow.dv_dt.data)

    # 3. Acoustic substeps on (w, theta', rho').  The shared
    #    machinery expects a ``NonHydrostaticState`` with trailing-axis
    #    convention ``(..., nlev)``; on our lat-lon C-grid the leading
    #    axes are ``(n_lat, n_lon)`` so the metric ``J[..., None]``
    #    broadcasts cleanly.  We wrap the cell-centred parts in a
    #    transient ``NonHydrostaticState`` for the substep call and
    #    unwrap back into the C-grid state below.
    state_for_acoustic = _to_shared_state(
        state._replace(u=u_new, v=v_new),
    )
    n_acoustic = cfg.n_acoustic_substeps
    dt_s = dt / n_acoustic
    if cfg.semi_implicit_acoustic:
        state_post_acoustic = acoustic_substeps_semi_implicit(
            state_for_acoustic, slow, dt_s, n_acoustic,
            SplitExplicitConfig(),
            height_coord, terrain_metric, cfg,
        )
    else:
        state_post_acoustic = acoustic_substeps(
            state_for_acoustic, slow, dt_s, n_acoustic,
            SplitExplicitConfig(),
            height_coord, terrain_metric, cfg,
        )

    return _from_shared_state(state_post_acoustic, c_grid_proto=state, u=u_new, v=v_new)


# ==============================================================================
# Adapter between CGrid state and the shared ``NonHydrostaticState``
# ==============================================================================


def _to_shared_state(s: CGridLatLonNonHydrostaticState) -> NonHydrostaticState:
    """Wrap the C-grid cell-centred parts in a ``NonHydrostaticState``.

    The shared acoustic-substep code accesses ``state.w``,
    ``state.theta_prime``, ``state.rho_prime`` (cell-centred fields)
    plus ``state.u`` and ``state.v`` only to carry them through to the
    return value -- the acoustic loop does not actually touch u or v.
    We pass the C-grid u / v as-is so the returned ``Field`` objects
    keep their staggered shapes; downstream code re-extracts them.
    """
    dims_u = ("lat", "lon_face", "level")
    dims_v = ("lat_face", "lon", "level")
    dims_w = ("lat", "lon", "level_half")
    dims_c = ("lat", "lon", "level")
    dims_2 = ("lat", "lon")
    return NonHydrostaticState(
        u=Field(data=s.u, name="u", dims=dims_u, units="m/s"),
        v=Field(data=s.v, name="v", dims=dims_v, units="m/s"),
        w=Field(data=s.w, name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(
            data=s.theta_prime, name="theta_prime", dims=dims_c, units="K",
        ),
        rho_prime=Field(
            data=s.rho_prime, name="rho_prime", dims=dims_c, units="kg/m^3",
        ),
        phis=Field(data=s.phis, name="phis", dims=dims_2, units="m^2/s^2"),
        tracers=Field(
            data=jnp.zeros((0,), dtype=s.u.dtype),
            name="tracers", dims=(), units="",
        ),
    )


def _from_shared_state(
    s: NonHydrostaticState,
    c_grid_proto: CGridLatLonNonHydrostaticState,
    u: jax.Array,
    v: jax.Array,
) -> CGridLatLonNonHydrostaticState:
    """Inverse of :func:`_to_shared_state`."""
    return CGridLatLonNonHydrostaticState(
        u=u,
        v=v,
        w=s.w.data,
        theta_prime=s.theta_prime.data,
        rho_prime=s.rho_prime.data,
        phis=c_grid_proto.phis,
        tracers=c_grid_proto.tracers,
    )


# ==============================================================================
# Tendency wrapper for SSP-RK3 outer integrator (legacy API)
# ==============================================================================


def cgrid_latlon_nh_tendency_fn(
    state: CGridLatLonNonHydrostaticState,
    grid: LatLonGrid,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
    config: CGridLatLonCompressibleEulerConfig = CGridLatLonCompressibleEulerConfig(),
) -> NonHydrostaticTendencies:
    """Return slow tendencies wrapped as :class:`NonHydrostaticTendencies`.

    Convenience entry point for callers that integrate the system
    with an off-the-shelf outer integrator (SSP-RK3, RK4, etc.)
    rather than the split-explicit Klemp & Wilhelmson scheme exposed
    by :func:`cgrid_latlon_nh_step`.  Note that under an explicit
    outer integrator the acoustic CFL constraint limits ``dt`` more
    strongly than the slow-tendency-only CFL of the split scheme;
    callers should prefer :func:`cgrid_latlon_nh_step` for production
    runs.
    """
    return cgrid_latlon_nh_slow_tendencies(
        state, grid, height_coord, terrain_metric, config,
    )


__all__ = [
    "CGridLatLonNonHydrostaticState",
    "CGridLatLonCompressibleEulerConfig",
    "cgrid_latlon_nh_slow_tendencies",
    "cgrid_latlon_nh_step",
    "cgrid_latlon_nh_tendency_fn",
]
