"""Non-hydrostatic compressible Euler on the lat-lon Arakawa C-grid.

This module provides the non-hydrostatic companion to
``primitive_eq_latlon_cgrid`` (hydrostatic) on the latitude-longitude
grid.  It is the lat-lon C-grid counterpart of the cubed-sphere
``compressible_euler_cdgrid`` and the Gaussian-grid spectral
``spectral_nh`` non-hydrostatic dycores.

.. note::
   NOT YET WIRED INTO THE SOLVER SELECTION LAYER.  This dycore is
   implemented and unit-tested in isolation but is intentionally absent
   from ``dynamics/__init__.py`` (``_LAZY_IMPORTS`` / ``_SOLVER_TO_CLASS`` /
   ``AVAILABLE_SOLVERS`` / ``_AXIS_TO_SOLVER``) and from
   ``driver/component_factory._DRIVER_SUPPORTED`` — there is no
   ``("nonhydrostatic", "latlon_cgrid")`` selection path, so it cannot be
   chosen through the model factory or driver.  It is a peer dycore pending
   end-to-end driver integration + dycore-progression validation (or
   relocation to ``_future/``).  Do not document it as a selectable solver
   until that wiring + validation lands.

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

Acoustic-substep contract (invariant)
-------------------------------------
The shared :func:`compressible_euler.acoustic_substeps` (and its
semi-implicit variant) updates only the cell-centred fields
``w``, ``theta_prime``, ``rho_prime`` inside its substep body.  The
``u`` and ``v`` ``Field`` objects on the input :class:`NonHydrostaticState`
are passed through to the returned state **with their .data arrays
unmodified**.  This is what makes the C-grid u (n_lat, n_lon+1, nlev)
and v (n_lat+1, n_lon, nlev) staggering safe to ship through the
substep loop: the substep code never re-shapes or interpolates u or
v, it only consumes the cell-centred trio.  If a future refactor of
``acoustic_substeps`` ever touches u or v, this assumption breaks --
the rest-state and shape tests below guard against that.

Status
------
v1 (2026-05-25): adversarial review applied.
  - KE uses Sadourny "square-then-average" form (no double-averaging).
  - Vertical KE removed from horizontal Bernoulli function.
  - w advected on its native half-level staggering by vertically
    averaging the horizontal face velocities to half-levels.
  - Sponge profile built directly on ``z_half`` (no full-to-half
    interpolation hack).
  - Pole-wall projection of v moved to the start of the slow
    tendency so the Coriolis stencil never sees pole leakage.
  - Hyperdiffusion on u, v uses the proper C-grid vector Laplacian
    (grad(div) - curl(curl)) from
    ``ocean.dynamics.latlon_cgrid_operators``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.core.field import Field
from legoesm.atmosphere.dynamics.compressible_euler import (
    CompressibleEulerConfig,
    sponge_profile,
    acoustic_substeps,
    acoustic_substeps_semi_implicit,
    compute_exner_perturbation,
)
from legoesm.atmosphere.physics.thermodynamics import sanitize_theta_rho
from legoesm.core.state import NonHydrostaticState, NonHydrostaticTendencies
from legoesm.grids.latlon import LatLonGrid
from legoesm.grids.vertical import HeightCoordinate, TerrainMetric
from legoesm.grids.operators_latlon_cgrid import (
    divergence_cgrid,
    gradient_x_cgrid,
    gradient_y_cgrid,
    interp_cell_to_uface,
    interp_cell_to_vface,
    vector_laplacian_cgrid,
)
from legoesm.atmosphere.dynamics.shallow_water_latlon_cgrid import (
    absolute_vorticity_coriolis,
)
from legoesm.core.operators_fv_latlon_3d import (
    cgrid_fv_flux_divergence_latlon_3d,
    cgrid_fv_scalar_advection_latlon_3d,
)
from legoesm.timestepping.split_explicit import SplitExplicitConfig


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


def _apply_pole_wall(v: jnp.ndarray) -> jnp.ndarray:
    """Enforce wall boundary condition on v at the poles.

    On a lat-lon C-grid the meridional velocity sits at lat-face
    positions ``j = 0 .. n_lat`` with ``j = 0`` on the south pole and
    ``j = n_lat`` on the north pole.  The kinematic boundary condition
    at each pole is v = 0 (no flow through the pole).  We enforce this
    both on the *incoming* v passed to :func:`cgrid_latlon_nh_slow_tendencies`
    (so the Coriolis stencil never sees pole leakage from an unprojected
    RK3 intermediate stage) **and** on the outgoing slow ``dv_dt``.
    """
    return v.at[0].set(0.0).at[-1].set(0.0)


def _full_to_half_vert(field: jnp.ndarray) -> jnp.ndarray:
    """Average a trailing-axis-(..., nlev) field to (..., nlev+1) half-levels.

    Interior half-levels (k=1..nlev-1) are the arithmetic mean of the
    two flanking full levels.  Boundary half-levels (k=0 and k=nlev)
    copy the nearest full-level value (zero-flux extension), so a
    constant column maps to a constant half-level column.
    """
    inner = 0.5 * (field[..., :-1] + field[..., 1:])
    return jnp.concatenate(
        [field[..., :1], inner, field[..., -1:]], axis=-1,
    )


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
    scheme.  For ``theta'`` the slow tendency is horizontal advection +
    horizontal hyperdiff; for ``rho'`` it is the horizontal leg of
    continuity in conservative flux form on TOTAL density,
    ``-div_h(rho_total v_h)``, + horizontal hyperdiff on ``rho'`` -- the
    vertical advection and continuity contributions are handled inside
    the acoustic substep loop.

    Returns
    -------
    NonHydrostaticTendencies
        Pytree of slow tendencies with matching C-grid staggering.
    """
    u, w = state.u, state.w
    # Project incoming v onto the wall-BC (v=0 at j=0 and j=n_lat)
    # before any stencil consumes it.  Without this, an SSP-RK3
    # intermediate stage can deliver a state whose polar v-faces
    # carry round-off-level non-zero values, contaminating the
    # 4-point ``u_at_v`` / ``v_at_u`` averages inside the
    # absolute-vorticity Coriolis at the pole-adjacent faces.
    v = _apply_pole_wall(state.v)
    theta_p, rho_p = state.theta_prime, state.rho_prime
    theta_0 = height_coord.theta_ref
    rho_0 = height_coord.rho_ref
    c_p = constants.c_pd
    cfg = config.euler

    theta_total, rho_total = sanitize_theta_rho(
        theta_0 + theta_p, rho_0 + rho_p,
    )

    # --- Horizontal kinetic energy + Bernoulli ---
    # Sadourny "square-then-average" form: KE at cell centres is the
    # arithmetic mean of u^2, v^2 over the four faces of the cell.
    # This is the canonical Sadourny (1975) discretisation -- it
    # avoids the spectral gap that arises when KE is first averaged
    # face->cell (u_c) and then differentiated cell->face (the
    # face->cell->face double-averaging that the v0 code did).
    # Vertical KE (0.5 w^2) is NOT included here: w lives on a
    # different vertical stagger and the vertical-momentum equation
    # has its own Bernoulli term inside the acoustic substep.
    KE = 0.25 * (
        u[:, :-1] ** 2 + u[:, 1:] ** 2
        + v[:-1] ** 2 + v[1:] ** 2
    )

    # --- Horizontal pressure-gradient term ---
    # Non-hydrostatic PGF in Exner formulation:
    #   du/dt += -c_p * theta * d(pi)/dx
    # where pi = pi_0(z) + pi'.  ``compute_exner_perturbation``
    # returns pi' alone (the reference Exner is z-only and is
    # subtracted analytically inside the function), so the gradient
    # below is the full horizontal PGF.
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

    # --- Horizontal scalar advection (PPM) for theta' ---
    horiz_adv_theta_p = cgrid_fv_scalar_advection_latlon_3d(
        theta_p, u, v, grid,
    )
    # --- Horizontal continuity for rho': conservative FLUX form ---
    # Sign convention: tendency = d(rho')/dt, applied as rho' += dt*tend.
    # Continuity is d(rho)/dt = -div(rho v); the reference rho_ref(z) is
    # static, so d(rho')/dt = d(rho)/dt, and the horizontal leg is
    #   d(rho')/dt|_h = -div_h(rho_total * v_h)
    # on TOTAL density -- matching the vertical leg inside the acoustic
    # kernel, which is flux form on rho_total (-d(rho_total w)/dz).
    # ``cgrid_fv_flux_divergence_latlon_3d`` returns -div_h(q v) (the
    # minus sign is inside the operator), so it is added with a + sign.
    # Budget closure: the operator telescopes (periodic lon; v = 0 at the
    # polar faces via _apply_pole_wall above), so the global integral
    # sum(tend * area) vanishes per level and dry mass is conserved.
    # The v1 code used the ADVECTIVE form on rho' (-v . grad rho'),
    # which dropped the -rho_total*div_h(v) compression term and leaked
    # mass under divergent flow.
    horiz_cont_rho_p = cgrid_fv_flux_divergence_latlon_3d(
        rho_total, u, v, grid,
    )

    # --- Horizontal advection of w on its native half-level grid ---
    # The PPM helper is shape-generic in the trailing axis, so we can
    # advect w directly on the (n_lat, n_lon, nlev+1) Lorenz-staggered
    # half-level grid once we provide horizontal face velocities at
    # half-levels.  We average the full-level (..., nlev) u, v
    # vertically to (..., nlev+1) half-levels with edge-pad
    # boundaries; the rigid-lid kinematic BC ``w = 0`` is preserved
    # because we explicitly zero the tendency at k=0 and k=nlev.
    u_half = _full_to_half_vert(u)   # (n_lat, n_lon+1, nlev+1)
    v_half = _full_to_half_vert(v)   # (n_lat+1, n_lon, nlev+1)
    horiz_adv_w = cgrid_fv_scalar_advection_latlon_3d(
        w, u_half, v_half, grid,
    )
    # Rigid lid: top and bottom half-levels carry w = 0 always.
    horiz_adv_w = horiz_adv_w.at[..., 0].set(0.0).at[..., -1].set(0.0)

    # --- Horizontal hyperdiffusion ---
    # Sign convention: ``+ K * ∇²f`` is the smoothing form (the v0 code
    # had a stray sign inversion via ``-interp(_laplacian_scalar(...))``
    # that would have driven anti-diffusion).
    if config.hyperdiff_uv > 0.0:
        # Proper C-grid vector Laplacian: grad(div u) - curl(curl u).
        # Avoids the v0 face -> cell -> Laplacian -> face round-trip
        # that smeared the mode structure of vector hyperdiffusion.
        vlap_u, vlap_v = vector_laplacian_cgrid(u, v, grid)
        du_dt = du_dt + config.hyperdiff_uv * vlap_u
        dv_dt = dv_dt + config.hyperdiff_uv * vlap_v

    def _laplacian_scalar(field: jnp.ndarray) -> jnp.ndarray:
        gx = gradient_x_cgrid(field, grid)
        gy = gradient_y_cgrid(field, grid)
        return divergence_cgrid(gx, gy, grid)

    if config.hyperdiff_w > 0.0:
        # w lives at (n_lat, n_lon, nlev+1).  The cell-centred
        # horizontal Laplacian is shape-generic over the trailing
        # vertical axis, so we apply it directly to w on its native
        # half-level grid.
        horiz_adv_w = horiz_adv_w + config.hyperdiff_w * _laplacian_scalar(w)
    if config.hyperdiff_scalar > 0.0:
        horiz_adv_theta_p = (
            horiz_adv_theta_p
            + config.hyperdiff_scalar * _laplacian_scalar(theta_p)
        )
        # Hyperdiffusion stays on the PERTURBATION rho' (smoothing the
        # deviation field, not the balanced reference profile).
        horiz_cont_rho_p = (
            horiz_cont_rho_p
            + config.hyperdiff_scalar * _laplacian_scalar(rho_p)
        )

    # --- Apply pole wall BC on v slow tendency ---
    dv_dt = _apply_pole_wall(dv_dt)

    # --- Sponge layer on w (Rayleigh damping in top sponge) ---
    # Build the Rayleigh damping coefficient directly on the half-level
    # vertical grid where w lives -- this gives the correct sin^2
    # profile at the physical w locations.  The v0 code averaged a
    # full-level sponge profile to half-levels, which mis-located the
    # tapered region by half a layer.
    sponge_half = sponge_profile(
        height_coord.z_half, height_coord.H,
        cfg.sponge_width, cfg.sponge_coeff,
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
            data=horiz_cont_rho_p,
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
      2. Advance ALL prognostic fields (u, v, w, theta', rho') explicitly
         by dt * slow tendency.  The shared acoustic substep machinery
         ignores ``slow_tend`` by contract -- the outer step pre-applies
         it to every leaf, exactly like
         ``split_explicit._rk_stage_with_acoustics`` (``_pytree_axpy``)
         does for the cubed-sphere / MPAS / spectral NH dycores.  u and v
         then receive no acoustic feedback in this split, which matches
         the Skamarock & Klemp 2008 split-explicit scheme.
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

    # 2. Pre-apply the slow tendency to ALL prognostic leaves.
    #    Sign convention: tendency = d(field)/dt, so field += dt * tend
    #    (positive tendency increases the field; the w sponge tendency
    #    -sponge*w therefore damps w).  The shared acoustic substep
    #    machinery (``acoustic_substeps`` / ``..._semi_implicit``)
    #    receives ``slow`` but by contract never consumes it ("state
    #    after slow tendency update"): the outer step must fold the slow
    #    tendencies into EVERY field first, mirroring
    #    ``split_explicit._rk_stage_with_acoustics`` (``_pytree_axpy``).
    #    v1 advanced only u and v here, silently discarding the slow
    #    w / theta' / rho' tendencies (horizontal advection, hyperdiff,
    #    and the w sponge never acted on the prognostic state).
    u_new = state.u + dt * slow.du_dt.data
    v_new = _apply_pole_wall(state.v + dt * slow.dv_dt.data)
    w_slow = state.w + dt * slow.dw_dt.data
    theta_slow = state.theta_prime + dt * slow.dtheta_prime_dt.data
    rho_slow = state.rho_prime + dt * slow.drho_prime_dt.data

    # 3. Acoustic substeps on (w, theta', rho').  The shared
    #    machinery expects a ``NonHydrostaticState`` with trailing-axis
    #    convention ``(..., nlev)``; on our lat-lon C-grid the leading
    #    axes are ``(n_lat, n_lon)`` so the metric ``J[..., None]``
    #    broadcasts cleanly.  We wrap the cell-centred parts in a
    #    transient ``NonHydrostaticState`` for the substep call and
    #    unwrap back into the C-grid state below.
    state_for_acoustic = _to_shared_state(
        state._replace(
            u=u_new, v=v_new, w=w_slow,
            theta_prime=theta_slow, rho_prime=rho_slow,
        ),
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
