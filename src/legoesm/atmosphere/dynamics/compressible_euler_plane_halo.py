"""Halo-aware slow-tendency for the plane NH dycore.

Mirror of :func:`compressible_euler_plane.plane_compressible_euler_slow_tendencies`
but operates on a LOCAL per-rank slab + uses the halo-aware
operators from :mod:`plane_operators_halo`. Halo exchange happens
via :func:`packed_exchange_halo_plane_yxz` calls so MPI message
count is independent of stencil count.

JIT compatibility (Codex iter-2 caveat)
---------------------------------------
``packed_exchange_halo_plane_yxz`` is jit-safe on the single-rank
path (uses ``jnp.pad(mode='wrap')``) but the multi-rank
``mpi4jax.sendrecv`` branch contains Python control flow that
can raise ``ConcretizationTypeError`` when traced. Callers
intending to ``jax.jit`` the multi-rank tendency MUST verify
their mpi4jax version supports tracer-input sendrecv, OR call
the slow-tendency in eager mode (no jit) on multi-rank runs.
Single-rank jit is verified by ``test_halo_jit_compilable``.

Coverage (matches the original):

* C-grid pressure gradient (cell→face interp + face gradient)
* Coriolis (if ``use_coriolis``)
* Mass continuity (-div(ρ·u))
* Theta advective transport (1st-order upwind)
* Momentum advection (Arakawa-C upwind, corner averages)
* Vertical advection (column-local, no halo needed)
* Rayleigh sponge (column-local, no halo)
* Biharmonic hyperdiffusion (`laplacian²`)
* Tracer advective transport (vmap over tracer axis)

Not covered here (yet — separate PR):

* Smagorinsky LES (``smagorinsky_cs > 0``) — strain tensor uses
  half-level w + horizontal shear with a layered halo pattern
  that needs its own halo-aware rewrite. Raises ``NotImplementedError``
  when ``smagorinsky_cs > 0`` so callers see a clear gate.

Single-rank equivalence
-----------------------
With ``layout.n_ranks == 1``, the halo exchange devolves to
``jnp.pad(mode='wrap')`` and the halo-aware operators reduce to
the original ``jnp.roll`` stencils via slice arithmetic — the
return value is BIT-IDENTICAL to the non-halo dycore.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.atmosphere.dynamics import (
    plane_operators_halo as oh,
)
from legoesm.atmosphere.dynamics.compressible_euler import (
    CompressibleEulerConfig, _sponge_profile, compute_exner_perturbation,
)
from legoesm.atmosphere.physics.thermodynamics import sanitize_theta_rho
from legoesm.core.state import (
    PlaneNonHydrostaticState, PlaneNonHydrostaticTendencies,
)
from legoesm.grids.plane import PlaneGrid
from legoesm.grids.vertical import HeightCoordinate, TerrainMetric
from legoesm.parallel.plane_mpi import (
    PlanePencilLayout, packed_exchange_halo_plane_yxz,
)


def _c_pd_constant():
    from legoesm import constants
    return constants.c_pd


def _vertical_advection_plane(
    field_yxz: jax.Array,
    w_yxz_half: jax.Array,
    hc: HeightCoordinate,
    J: jax.Array,
) -> jax.Array:
    """Vertical advection ``-w · df/dz``. Column-local (no halo)."""
    # Centered-difference on full levels, w averaged to full level.
    w_full = 0.5 * (w_yxz_half[..., :-1] + w_yxz_half[..., 1:])
    nlev = field_yxz.shape[-1]
    if nlev <= 2:
        return jnp.zeros_like(field_yxz)
    dz_half = hc.dz_half
    dz_centered = dz_half[:-1] + dz_half[1:]
    inner_grad = (field_yxz[..., :-2] - field_yxz[..., 2:]) / dz_centered
    pad_axes = ((0, 0),) * (field_yxz.ndim - 1)
    dfield_dz = jnp.pad(inner_grad, (*pad_axes, (1, 1)))
    return -w_full / J[..., None] * dfield_dz


def precompute_coriolis_halo(
    grid: PlaneGrid, layout: PlanePencilLayout,
) -> jax.Array:
    """Pre-exchange grid.f_y for the halo path.

    f_y is static — caller should compute this once at model init
    and pass into plane_compressible_euler_slow_tendencies_halo
    via ``f_pad_cached`` so we avoid one MPI round per slow-tendency
    call (Codex iter-3 perf finding).
    """
    f_3d = grid.f_y[:, :, None]
    f_pad, = packed_exchange_halo_plane_yxz(f_3d, layout=layout)
    return f_pad


def plane_compressible_euler_slow_tendencies_halo(
    state: PlaneNonHydrostaticState,
    grid: PlaneGrid,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
    config: CompressibleEulerConfig,
    layout: PlanePencilLayout,
    f_pad_cached: jax.Array | None = None,
) -> PlaneNonHydrostaticTendencies:
    """Halo-aware slow-tendency for the plane NH dycore.

    Inputs: rank-local slabs. Output: rank-local interior-shape
    tendencies. Single packed MPI round per call covers all
    horizontally-coupled prognostic fields.

    Parameters
    ----------
    f_pad_cached : optional pre-exchanged f_y halo array (use
        :func:`precompute_coriolis_halo` once at init). When None
        and ``config.use_coriolis``, f_y is exchanged per call
        (correct but adds 1 MPI round).

    JIT safety (Codex iter-2)
    -------------------------
    Hard guard raises ``RuntimeError`` if called under a JIT
    trace with ``layout.n_ranks > 1``. mpi4jax sendrecv branch
    may raise ``ConcretizationTypeError`` under tracing on some
    mpi4jax versions; we prevent silent crashes deep in the
    compiled graph by failing early.
    """
    if config.smagorinsky_cs > 0.0:
        raise NotImplementedError(
            "Halo-aware slow tendency does not yet support "
            "Smagorinsky LES (smagorinsky_cs > 0). Use the original "
            "plane_compressible_euler_slow_tendencies for "
            "single-process Smag runs, or wait for the follow-up PR "
            "that ports _compute_smagorinsky_K_m_plane to halo-aware "
            "ops."
        )
    # Codex iter-3: multi-rank under JIT is a known crash mode on
    # some mpi4jax versions. Raise early so the failure is at the
    # call site, not deep in the compiled HLO graph.
    if layout.n_ranks > 1 and isinstance(
        state.u.data, jax.core.Tracer,
    ):
        raise RuntimeError(
            "plane_compressible_euler_slow_tendencies_halo cannot "
            "be JIT-compiled with layout.n_ranks > 1: "
            "packed_exchange_halo_plane_yxz uses mpi4jax sendrecv "
            "with Python control flow that raises "
            "ConcretizationTypeError under tracing. Call eagerly on "
            "multi-rank, or jit the local-compute portion only and "
            "drive halo exchange from Python."
        )

    h = layout.halo if hasattr(layout, "halo") else 1

    u = state.u.data
    v = state.v.data
    w = state.w.data
    theta_p = state.theta_prime.data
    rho_p = state.rho_prime.data

    c_p = jnp.asarray(0.0, dtype=u.dtype) + _c_pd_constant()
    theta_0 = height_coord.theta_ref
    rho_0 = height_coord.rho_ref
    J = terrain_metric.jacobian

    theta_total, rho_total = sanitize_theta_rho(
        theta_0 + theta_p, rho_0 + rho_p,
    )
    pi_p = compute_exner_perturbation(rho_p, theta_p, height_coord)

    # ONE packed halo exchange for every horizontally-coupled field
    # used in slow-tendency computations.
    fields_to_exchange = [
        u, v, theta_p, rho_p, pi_p, theta_total, rho_total,
    ]
    (
        u_pad, v_pad, theta_p_pad, rho_p_pad, pi_p_pad,
        theta_total_pad, rho_total_pad,
    ) = packed_exchange_halo_plane_yxz(
        *fields_to_exchange, layout=layout,
    )

    # 2. C-grid pressure gradient.
    grad_pi_x = oh.grad_x_vlast_halo(pi_p_pad, grid, h)
    grad_pi_y = oh.grad_y_vlast_halo(pi_p_pad, grid, h)
    theta_xface = oh.interp_cell_to_xface_vlast_halo(theta_total_pad, grid, h)
    theta_yface = oh.interp_cell_to_yface_vlast_halo(theta_total_pad, grid, h)
    du_pg = -c_p * theta_xface * grad_pi_x
    dv_pg = -c_p * theta_yface * grad_pi_y

    # 3. Coriolis (need v interp to x-face, u interp to y-face).
    if config.use_coriolis:
        # Codex iter-3 (perf): use cached f_pad if caller pre-computed
        # it (preferred — avoids per-step exchange of a static field).
        # Fall back to per-step exchange if not cached.
        if f_pad_cached is not None:
            f_pad = f_pad_cached
        else:
            # Codex iter-2 (correctness): f_y is y-dependent beta-plane
            # field on GLOBAL grid. Local jnp.pad(wrap) would wrap each
            # rank's slab → wrong f on north/south halos under MPI.
            # Use packed_exchange_halo_plane_yxz so the halo carries
            # global-grid f_y from neighbor rank.
            f_3d = grid.f_y[:, :, None]
            f_pad, = packed_exchange_halo_plane_yxz(f_3d, layout=layout)
        f_xface = oh.interp_cell_to_xface_vlast_halo(f_pad, grid, h)
        f_yface = oh.interp_cell_to_yface_vlast_halo(f_pad, grid, h)
        v_xface = oh.interp_yface_to_xface_vlast_halo(v_pad, grid, h)
        u_yface = oh.interp_xface_to_yface_vlast_halo(u_pad, grid, h)
        du_cor = f_xface * v_xface
        dv_cor = -f_yface * u_yface
    else:
        du_cor = jnp.zeros_like(u)
        dv_cor = jnp.zeros_like(v)

    # 4. Mass continuity: -div(ρ·u). rho_xface + rho_yface live at
    #    interior face positions; combine with u_pad (already padded)
    #    by extracting u's interior, multiplying, then re-padding.
    rho_xface_int = oh.interp_cell_to_xface_vlast_halo(
        rho_total_pad, grid, h,
    )
    rho_yface_int = oh.interp_cell_to_yface_vlast_halo(
        rho_total_pad, grid, h,
    )
    u_int = u_pad[h:-h, h:-h, :]
    v_int = v_pad[h:-h, h:-h, :]
    flux_u_int = rho_xface_int * u_int
    flux_v_int = rho_yface_int * v_int
    # Single MPI round to repad both fluxes for divergence.
    flux_u_pad, flux_v_pad = packed_exchange_halo_plane_yxz(
        flux_u_int, flux_v_int, layout=layout,
    )
    drho_p_dt = -oh.divergence_vlast_halo(flux_u_pad, flux_v_pad, grid, h)

    # 5. Theta advection.
    u_center = oh.interp_xface_to_cell_vlast_halo(u_pad, grid, h)
    v_center = oh.interp_yface_to_cell_vlast_halo(v_pad, grid, h)
    # Need u_center, v_center to be padded for use with the halo-aware
    # upwind. Re-pad the interior result.
    u_center_pad = _re_pad_halo(u_center, layout, h)
    v_center_pad = _re_pad_halo(v_center, layout, h)
    dtheta_p_dt = (
        oh.upwind_advection_x_halo(theta_total_pad, u_center_pad, grid.dx, h)
        + oh.upwind_advection_y_halo(theta_total_pad, v_center_pad, grid.dy, h)
    )

    # 6. Horizontal momentum advection — u advected by (u, v_at_xface),
    #    v advected by (u_at_yface, v).
    v_at_xface = oh.interp_yface_to_xface_vlast_halo(v_pad, grid, h)
    u_at_yface = oh.interp_xface_to_yface_vlast_halo(u_pad, grid, h)
    v_at_xface_pad = _re_pad_halo(v_at_xface, layout, h)
    u_at_yface_pad = _re_pad_halo(u_at_yface, layout, h)
    du_adv = (
        oh.upwind_advection_x_halo(u_pad, u_pad, grid.dx, h)
        + oh.upwind_advection_y_halo(u_pad, v_at_xface_pad, grid.dy, h)
    )
    dv_adv = (
        oh.upwind_advection_x_halo(v_pad, u_at_yface_pad, grid.dx, h)
        + oh.upwind_advection_y_halo(v_pad, v_pad, grid.dy, h)
    )

    # 7. Vertical advection (column-local).
    du_vert = _vertical_advection_plane(u, w, height_coord, J)
    dv_vert = _vertical_advection_plane(v, w, height_coord, J)

    du_dt = du_adv + du_vert + du_pg + du_cor
    dv_dt = dv_adv + dv_vert + dv_pg + dv_cor

    # 8. w slow part.
    w_full = 0.5 * (w[..., :-1] + w[..., 1:])
    w_full_pad = _re_pad_halo(w_full, layout, h)
    dw_full = (
        oh.upwind_advection_x_halo(w_full_pad, u_center_pad, grid.dx, h)
        + oh.upwind_advection_y_halo(w_full_pad, v_center_pad, grid.dy, h)
    )
    pad_axes = ((0, 0),) * (dw_full.ndim - 1)
    dw_dt = jnp.pad(
        0.5 * (dw_full[..., :-1] + dw_full[..., 1:]),
        (*pad_axes, (1, 1)),
    )

    # 9. Rayleigh sponge (column-local).
    sponge_full = _sponge_profile(
        height_coord.z_full, height_coord.H,
        config.sponge_width, config.sponge_coeff,
    )
    sponge_half = _sponge_profile(
        height_coord.z_half, height_coord.H,
        config.sponge_width, config.sponge_coeff,
    )
    du_dt = du_dt - sponge_full * u
    dv_dt = dv_dt - sponge_full * v
    dtheta_p_dt = dtheta_p_dt - sponge_full * theta_p
    dw_dt = dw_dt - sponge_half * w

    # 10. Biharmonic hyperdiffusion. Two laplacian passes →
    #     need re-exchange between passes since the first lap output
    #     loses halo freshness.
    if config.hyperdiff_coeff > 0.0:
        lap_u_int = oh.laplacian_vlast_halo(u_pad, grid, h)
        lap_v_int = oh.laplacian_vlast_halo(v_pad, grid, h)
        lap_theta_int = oh.laplacian_vlast_halo(theta_p_pad, grid, h)
        lap_u_pad, lap_v_pad, lap_theta_pad = packed_exchange_halo_plane_yxz(
            lap_u_int, lap_v_int, lap_theta_int, layout=layout,
        )
        du_dt = du_dt - config.hyperdiff_coeff * oh.laplacian_vlast_halo(
            lap_u_pad, grid, h,
        )
        dv_dt = dv_dt - config.hyperdiff_coeff * oh.laplacian_vlast_halo(
            lap_v_pad, grid, h,
        )
        dtheta_p_dt = dtheta_p_dt - config.hyperdiff_coeff * (
            oh.laplacian_vlast_halo(lap_theta_pad, grid, h)
        )
    if config.hyperdiff_rho_coeff > 0.0:
        lap_rho_int = oh.laplacian_vlast_halo(rho_p_pad, grid, h)
        lap_rho_pad, = packed_exchange_halo_plane_yxz(
            lap_rho_int, layout=layout,
        )
        drho_p_dt = drho_p_dt - config.hyperdiff_rho_coeff * (
            oh.laplacian_vlast_halo(lap_rho_pad, grid, h)
        )
    if config.hyperdiff_w_coeff > 0.0:
        w_pad = _re_pad_halo(w, layout, h)
        lap_w_int = oh.laplacian_vlast_halo(w_pad, grid, h)
        lap_w_pad, = packed_exchange_halo_plane_yxz(
            lap_w_int, layout=layout,
        )
        dw_dt = dw_dt - config.hyperdiff_w_coeff * (
            oh.laplacian_vlast_halo(lap_w_pad, grid, h)
        )

    # 11. Smagorinsky LES — gated above (NotImplementedError).

    # 12. Tracer advection.
    tracers = state.tracers.data
    if tracers.shape[-1] > 0:
        # Pad each tracer slot via packed exchange (one MPI round).
        # tracers.shape = (ny_local, nx_local, nlev, n_tracers)
        n_tr = tracers.shape[-1]
        # Treat trailing tracer axis as additional channels by
        # stacking into the trailing dim of packed exchange.
        # tracers already (ny, nx, nlev, n_tr); pack into single
        # exchange directly by reshape into (ny, nx, nlev*n_tr).
        ny_l, nx_l, nlev, _ = tracers.shape
        tracers_flat = tracers.reshape(ny_l, nx_l, nlev * n_tr)
        tracers_flat_pad, = packed_exchange_halo_plane_yxz(
            tracers_flat, layout=layout,
        )
        ny_p, nx_p = tracers_flat_pad.shape[:2]
        tracers_pad = tracers_flat_pad.reshape(ny_p, nx_p, nlev, n_tr)

        def _tracer_tend_one(q_pad, q_int):
            return (
                oh.upwind_advection_x_halo(q_pad, u_center_pad, grid.dx, h)
                + oh.upwind_advection_y_halo(q_pad, v_center_pad, grid.dy, h)
                + _vertical_advection_plane(q_int, w, height_coord, J)
            )
        dtracers_dt = jax.vmap(
            _tracer_tend_one, in_axes=(-1, -1), out_axes=-1,
        )(tracers_pad, tracers)
    else:
        dtracers_dt = jnp.zeros_like(tracers)
    zero_phis = jnp.zeros_like(state.phis.data)

    return PlaneNonHydrostaticTendencies(
        du_dt=state.u.replace(data=du_dt),
        dv_dt=state.v.replace(data=dv_dt),
        dw_dt=state.w.replace(data=dw_dt),
        dtheta_prime_dt=state.theta_prime.replace(data=dtheta_p_dt),
        drho_prime_dt=state.rho_prime.replace(data=drho_p_dt),
        dphis_dt=state.phis.replace(data=zero_phis),
        dtracers_dt=state.tracers.replace(data=dtracers_dt),
    )


def _re_pad_halo(arr_int, layout, halo):
    """Helper: re-exchange halo for an interior-shape result.

    Used when an operator output is interior-shape but a downstream
    operator needs it padded. Single-rank uses jnp.pad(mode='wrap');
    multi-rank goes through packed_exchange_halo_plane_yxz which
    triggers MPI sendrecv.
    """
    arr_pad, = packed_exchange_halo_plane_yxz(arr_int, layout=layout)
    return arr_pad


