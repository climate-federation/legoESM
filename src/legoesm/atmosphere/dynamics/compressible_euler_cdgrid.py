"""FV3-inspired C-D grid Non-Hydrostatic Compressible Euler on the cubed-sphere.

**Fidelity status: stabilized research path, not a faithful FV3 port.**

Uses the same C-D grid discretisation as the shallow water and PE solvers:

* D-grid winds (cell corners) are prognostic (internal conversion from cell-centre).
* C-grid velocities (cell edges) are diagnosed for mass/scalar transport.
* Vorticity from circulation (exact on D-grid).
* Bernoulli/pressure gradient via Arakawa-Lamb at D-grid corners.
* Scalar transport (theta, rho, tracers) via C-grid upwind mass flux.
* Acoustic substeps for vertically propagating sound waves.

**State is stored at cell centres** for compatibility with the existing physics
infrastructure. Velocities are converted to D-grid for momentum computation
and back to cell-centre for output. A faithful FV3 NH path would store winds
on D-grid edges throughout and require a physics coupler that consumes D-grid
winds.

References
----------
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState, NonHydrostaticTendencies
from legoesm.core.operators_3d import (
    divergence_3d,
    gradient_x_3d,
    gradient_y_3d,
    hyperdiffusion_3d,
    laplacian_compact_3d,
    vertical_advection_height,
)
from legoesm.core.operators_cdgrid import (
    dgrid_to_cgrid,
    dgrid_to_center_vector,
    dgrid_vorticity,
    cgrid_mass_flux_divergence,
    cgrid_divergence,
    _arakawa_lamb_gradient,
    _interp_center_to_corner,
    _interp_corner_to_center,
    _laplacian_dgrid,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.cubed_sphere_cdgrid import (
    CubedSphereCDGrid,
    create_cubed_sphere_cdgrid,
)
from legoesm.grids.vertical import HeightCoordinate, TerrainMetric
from legoesm.timestepping.integration import IntegrationMixin
from legoesm.timestepping.split_explicit import (
    split_explicit_step,
    SplitExplicitConfig,
)
from legoesm.atmosphere.dynamics.compressible_euler import (
    CompressibleEulerConfig,
    compute_exner_perturbation,
    _sponge_profile,
    acoustic_substeps,
    acoustic_substeps_semi_implicit,
)
from legoesm.atmosphere.physics.thermodynamics import sanitize_theta_rho
from legoesm.grids.halo import pad_halo_4d
from legoesm import constants


class CDGridCompressibleEulerConfig(NamedTuple):
    """Configuration for the C-D grid non-hydrostatic CE model.

    This is the recommended cubed-sphere non-hydrostatic solver for
    production AMIP/CMIP simulations.  Uses FV3-style C-D grid staggering
    (Lin 2004, Putman & Lin 2007) which eliminates the Hollingsworth-Kallberg
    instability that affects cell-centre solvers.
    """
    g: float = constants.g
    A_h: float = 0.0              # Laplacian viscosity [m^2/s]
    hyperdiff_coeff: float = 0.0
    hyperdiff_rho_coeff: float = 0.0
    hyperdiff_w_coeff: float = 0.0
    sponge_width: float = 10000.0
    sponge_coeff: float = 0.05
    n_acoustic_substeps: int = 6
    small_earth_factor: float = 1.0
    use_coriolis: bool = True
    semi_implicit_acoustic: bool = False
    outer_integrator: str = "ssp_rk3"
    fix_mass: bool = False
    anchor_mass_to_initial: bool = False
    acoustic_off_centering: float = 0.0   # Off-centering beta for acoustic damping
                                          # 0.0 = centered, 0.1 = recommended for long runs


def cdgrid_compressible_euler_slow_tendencies(
    state: NonHydrostaticState,
    grid: CubedSphereGrid,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
    cdgrid: CubedSphereCDGrid,
    config: CDGridCompressibleEulerConfig,
    physics_tendency: NonHydrostaticTendencies | None = None,
) -> NonHydrostaticTendencies:
    """Compute slow (advective) tendencies using C-D grid operators.

    Parameters
    ----------
    state : NonHydrostaticState
    grid : CubedSphereGrid
    height_coord : HeightCoordinate
    terrain_metric : TerrainMetric
    cdgrid : CubedSphereCDGrid
    config : CDGridCompressibleEulerConfig
    physics_tendency : NonHydrostaticTendencies, optional

    Returns
    -------
    NonHydrostaticTendencies
    """
    u = state.u.data
    v = state.v.data
    w = state.w.data
    theta_p = state.theta_prime.data
    rho_p = state.rho_prime.data
    tracers = state.tracers.data

    c_p = constants.c_pd
    rho_0 = height_coord.rho_ref
    theta_0 = height_coord.theta_ref
    dz = height_coord.dz
    dz_half = height_coord.dz_half
    J = terrain_metric.jacobian

    theta_total, rho_total = sanitize_theta_rho(
        theta_0 + theta_p, rho_0 + rho_p,
    )

    # --- 1. Exner perturbation and horizontal PGF ---
    pi_prime = compute_exner_perturbation(rho_p, theta_p, height_coord)

    # --- 2. Convert to D-grid ---
    # Stack (u, v) along a trailing axis and fold into the level dim so
    # a single ``_interp_center_to_corner`` (one halo exchange + one
    # 4-point average) handles both components, replacing two separate
    # calls each with their own halo.  Same passive-trailing-axis
    # pattern as the SH and divergence batching loops.
    n_face_uv, n_i_uv, n_j_uv, nlev_uv = u.shape
    _uv_stack = jnp.stack([u, v], axis=-1)  # (6, n, n, nlev, 2)
    _uv_flat = _uv_stack.reshape(n_face_uv, n_i_uv, n_j_uv, nlev_uv * 2)
    _uv_d_flat = _interp_center_to_corner(_uv_flat, cdgrid)
    _uv_d = _uv_d_flat.reshape(
        _uv_d_flat.shape[0], _uv_d_flat.shape[1], _uv_d_flat.shape[2],
        nlev_uv, 2,
    )
    u_d = _uv_d[..., 0]
    v_d = _uv_d[..., 1]

    # --- 3. C-grid velocities ---
    u_c, v_c = dgrid_to_cgrid(u_d, v_d, cdgrid)

    # --- 4. Vorticity (cell centres) ---
    zeta = dgrid_vorticity(u_d, v_d, cdgrid)

    # --- 5. KE at cell centres from D-grid (orthogonal basis) ---
    u_cc, v_cc = dgrid_to_center_vector(u_d, v_d)
    K = 0.5 * (u_cc ** 2 + v_cc ** 2)

    # --- 6. Gradients at D-grid corners ---
    # Pack K and pi_prime into a single halo exchange under SPMD/MPI;
    # under the local backend each operator does its own exchange (same
    # as before).  ``_arakawa_lamb_gradient`` takes ``padded=`` to skip
    # its internal halo when supplied.
    from legoesm.grids.halo import _halo_backend as _hb_step6
    if _hb_step6 == "spmd":
        from legoesm.parallel.cubesphere_exchange import (
            packed_pad_halo_4d as _packed_4d_spmd, _spmd_mesh as _spmd_mesh_step6,
        )
        _K_pad_step6, _pi_pad_step6 = _packed_4d_spmd(
            K, pi_prime, mesh=_spmd_mesh_step6,
        )
    elif _hb_step6 == "mpi":
        from legoesm.grids.halo import _mpi_topology as _mpi_topo_step6
        from legoesm.parallel.halo_exchange import (
            packed_pad_halo_mpi_4d as _packed_mpi_4d_step6,
        )
        _K_pad_step6, _pi_pad_step6 = _packed_mpi_4d_step6(
            K, pi_prime, topology=_mpi_topo_step6,
        )
    else:
        _K_pad_step6 = _pi_pad_step6 = None

    # Batch the two Arakawa-Lamb gradients (K and pi_prime) into a
    # single call on a stacked tensor — the operator treats the
    # trailing axis as a passive batch (the 4-point finite difference
    # and the metric-matrix multiplication broadcast over the trailing
    # dim).  Stack the pre-padded inputs the same way so the local
    # backend (no pre-pad) issues only one halo exchange instead of
    # two.  2 gradient calls → 1.
    n_face_kp, n_i_kp, n_j_kp, nlev_kp = K.shape
    _kp_stack = jnp.stack([K, pi_prime], axis=-1)  # (6, n, n, nlev, 2)
    _kp_flat = _kp_stack.reshape(n_face_kp, n_i_kp, n_j_kp, nlev_kp * 2)
    if _K_pad_step6 is not None:
        _kp_pad_stack = jnp.stack([_K_pad_step6, _pi_pad_step6], axis=-1)
        _kp_pad_flat = _kp_pad_stack.reshape(
            _kp_pad_stack.shape[0], _kp_pad_stack.shape[1],
            _kp_pad_stack.shape[2], nlev_kp * 2,
        )
    else:
        _kp_pad_flat = None
    _dKpi_dx_flat, _dKpi_dy_perp_flat = _arakawa_lamb_gradient(
        _kp_flat, cdgrid, padded=_kp_pad_flat,
    )
    _dKpi_dx = _dKpi_dx_flat.reshape(
        _dKpi_dx_flat.shape[0], _dKpi_dx_flat.shape[1],
        _dKpi_dx_flat.shape[2], nlev_kp, 2,
    )
    _dKpi_dy_perp = _dKpi_dy_perp_flat.reshape(
        _dKpi_dy_perp_flat.shape[0], _dKpi_dy_perp_flat.shape[1],
        _dKpi_dy_perp_flat.shape[2], nlev_kp, 2,
    )
    dK_dx = _dKpi_dx[..., 0]
    dpi_dx = _dKpi_dx[..., 1]
    dK_dy_perp = _dKpi_dy_perp[..., 0]
    dpi_dy_perp = _dKpi_dy_perp[..., 1]

    # --- 7. D-grid momentum tendencies ---
    # Interpolate ζ only; add f_corner directly (FV3 stores f at corners).
    # Previously abs_vor = ζ + f was interpolated as a single field; because
    # the 4-point interpolator is linear but sin(lat) is not,
    # interp(f_cc) ≠ f_corner introduced an O(dx²) Coriolis error at corners.
    # Matches iter-74 fix in cdgrid_momentum_tendencies.
    zeta_corner = _interp_center_to_corner(zeta, cdgrid)
    if config.use_coriolis:
        abs_vor_corner = zeta_corner + cdgrid.f_corner[..., None]
    else:
        abs_vor_corner = zeta_corner
    theta_corner = _interp_center_to_corner(theta_total, cdgrid)

    du_d_dt = abs_vor_corner * v_d - dK_dx - c_p * theta_corner * dpi_dx
    dv_d_dt = -abs_vor_corner * u_d - dK_dy_perp - c_p * theta_corner * dpi_dy_perp

    # Laplacian viscosity — batch (u_d, v_d) into a single
    # ``_laplacian_dgrid`` call by stacking along a trailing axis and
    # folding into the level dim.  ``_laplacian_dgrid`` is now
    # 4D-native (single ``pad_halo_4d`` for all "levels"), so the
    # paired call shares one halo exchange and one compact ∇² across
    # both wind components — same passive-trailing-axis pattern as the
    # corner interps and other dycore batches.
    if config.A_h > 0:
        n_face_vl, n_id_vl, n_jd_vl, nlev_vl = u_d.shape
        _uv_d_lap_stack = jnp.stack([u_d, v_d], axis=-1)
        _uv_d_lap_flat = _uv_d_lap_stack.reshape(
            n_face_vl, n_id_vl, n_jd_vl, nlev_vl * 2,
        )
        _uv_d_lap_out = _laplacian_dgrid(_uv_d_lap_flat, cdgrid).reshape(
            n_face_vl, n_id_vl, n_jd_vl, nlev_vl, 2,
        )
        du_d_dt = du_d_dt + config.A_h * _uv_d_lap_out[..., 0]
        dv_d_dt = dv_d_dt + config.A_h * _uv_d_lap_out[..., 1]

    # --- 8. Convert back to cell-centre ---
    # Batch (du_d_dt, dv_d_dt) corner-to-center interp.  Same
    # passive-trailing-axis pattern; ``_interp_corner_to_center`` is a
    # 4-point average with no halo, so this saves one kernel launch.
    # ``du_d_dt``/``dv_d_dt`` live on D-grid corners (spatial dims may
    # differ from cell-centre by one in the staggered direction); after
    # ``_interp_corner_to_center`` the result lands on cell-centre
    # ``(n_face_uv, n_i_uv, n_j_uv, nlev_uv)`` from line 155.
    _duv_d_dt = jnp.stack([du_d_dt, dv_d_dt], axis=-1)  # (..., nlev, 2)
    _duv_d_dt_flat = _duv_d_dt.reshape(
        _duv_d_dt.shape[0], _duv_d_dt.shape[1], _duv_d_dt.shape[2],
        nlev_uv * 2,
    )
    _duv_dt = _interp_corner_to_center(_duv_d_dt_flat).reshape(
        n_face_uv, n_i_uv, n_j_uv, nlev_uv, 2,
    )
    du_dt = _duv_dt[..., 0]
    dv_dt = _duv_dt[..., 1]

    # --- 9. Vertical advection of u, v ---
    # Batch the two ``vertical_advection_height`` calls by stacking
    # (u, v) along a new leading axis.  ``w_full`` / ``w_star``
    # depend only on (w, dz, dz_half, J) so they are computed once
    # and the trailing-axis ``[..., :-1] - [..., 1:]`` gradient
    # broadcasts across the new axis.  Two passes through the
    # vertical-advection kernel collapse to one — same trailing/leading
    # axis batching as Loops 137/141.
    _uv_va = jnp.stack([u, v], axis=0)
    _uv_va_adv = vertical_advection_height(_uv_va, w, dz, dz_half, J)
    du_dt = du_dt + _uv_va_adv[0]
    dv_dt = dv_dt + _uv_va_adv[1]

    # --- 10. Theta equation: advective form -v·∇θ ---
    # The θ equation uses advective form (not divergence/flux form) because
    # θ is NOT a conserved density — it satisfies dθ/dt = 0, not ∂(ρθ)/∂t = -∇·(ρθv).
    # Advective form = flux divergence + θ·div(v):  -v·∇θ = -∇·(θv) + θ∇·v
    div_v = cgrid_divergence(u_c, v_c, cdgrid)

    # --- 10/11/12. (theta, rho, tracers) flux divergence (batched) ---
    # ``cgrid_mass_flux_divergence`` issues a ``pad_halo_4d`` on its
    # scalar input and runs the PPM reconstruction along the trailing
    # axis as a passive batch.  Stack ``(theta_total, rho_total)`` and
    # any prognostic tracers along that trailing axis, fold into the
    # level dim, and run a single PPM transport call instead of two.
    # ``u_c``, ``v_c``, and ``div_v`` are shared; ``jnp.repeat`` builds
    # the matching velocity broadcast for the interleaved (level ×
    # scalar) trailing axis.
    n_face_tr, n_i_tr, n_j_tr, nlev_tr = theta_total.shape
    n_tracers = tracers.shape[-1] if tracers.ndim > 3 else 0
    n_total = 2 + n_tracers  # theta + rho + tracers

    if n_tracers > 0:
        combined_stack = jnp.concatenate(
            [
                jnp.stack([theta_total, rho_total], axis=-1),  # (..., nlev, 2)
                tracers,  # (..., nlev, n_tracers)
            ], axis=-1,
        )  # (..., nlev, n_total)
    else:
        combined_stack = jnp.stack(
            [theta_total, rho_total], axis=-1,
        )  # (..., nlev, 2)

    combined_flat = combined_stack.reshape(
        n_face_tr, n_i_tr, n_j_tr, nlev_tr * n_total,
    )
    if n_total == 1:
        u_c_b, v_c_b = u_c, v_c
    else:
        u_c_b = jnp.repeat(u_c, n_total, axis=-1)
        v_c_b = jnp.repeat(v_c, n_total, axis=-1)
    flux_combined_flat = cgrid_mass_flux_divergence(
        combined_flat, u_c_b, v_c_b, cdgrid,
    )
    flux_combined = flux_combined_flat.reshape(
        n_face_tr, n_i_tr, n_j_tr, nlev_tr, n_total,
    )
    # Theta uses advective form: -∇·(θv) + θ·∇·v.
    dtheta_p_dt = flux_combined[..., 0] + theta_total * div_v
    # Rho uses pure flux form: -∇·(ρv) + 0 (continuity).
    drho_p_dt = flux_combined[..., 1]

    # Tracer advection (advective form).  Vertical advection still runs
    # per-tracer via ``vmap`` over the trailing axis so JAX produces one
    # batched kernel.
    if n_tracers > 0:
        horiz = (
            flux_combined[..., 2:]                # (..., nlev, n_tracers)
            + tracers * div_v[..., None]          # advective-form correction
        )

        def _vert_one(q):
            return vertical_advection_height(q, w, dz, dz_half, J)

        vert = jax.vmap(_vert_one, in_axes=-1, out_axes=-1)(tracers)
        dtracers_dt = horiz + vert
    else:
        dtracers_dt = jnp.zeros_like(tracers)

    # --- 13. Hyperdiffusion ---
    # ``hyperdiffusion_3d(field, grid, coeff)`` is defined as
    # ``-coeff * ∇²(∇²(field))`` where each ∇² runs the cubed-sphere
    # 4D-native compact stencil (single ``pad_halo_4d`` per call).
    # When *both* ``hyperdiff_coeff`` (applied to (u, v, theta_p))
    # and ``hyperdiff_rho_coeff`` (applied to rho_p) are non-zero,
    # the two operator chains share the same biharmonic structure
    # but use different coefficients on the outer step.  Inline the
    # operator and stack ALL four fields along a trailing axis: a
    # single inner ∇² and a single outer ∇² serve all four fields,
    # then per-field coefficients are applied at the very end.
    # 4 ∇² evaluations → 2 (one inner, one outer) per RHS evaluation
    # when both coefficients are active.  When only one coefficient
    # is active, fall back to the existing path (3-field or 1-field).
    _coeff_uvT = config.hyperdiff_coeff
    _coeff_rho = config.hyperdiff_rho_coeff
    if _coeff_uvT > 0 and _coeff_rho > 0:
        n_face_h, n_i_h, n_j_h, nlev_h = u.shape
        _hyper_stack = jnp.stack(
            [u, v, theta_p, rho_p], axis=-1,
        )  # (6, n, n, nlev, 4)
        _hyper_flat = _hyper_stack.reshape(n_face_h, n_i_h, n_j_h, nlev_h * 4)
        # Inner ∇² (compact stencil) — shared across all four fields.
        _lap1 = laplacian_compact_3d(_hyper_flat, grid)
        # Outer ∇² = div(grad).  Pad ``_lap1`` once and feed it to
        # both gradient ops (saves 1 ``pad_halo_4d`` per call).
        _dg = getattr(grid, 'duogrid', None)
        _offsets = None if _dg is not None else grid.halo_interp_offsets
        _lap1_pad = pad_halo_4d(_lap1, interp_offsets=_offsets, duogrid=_dg)
        _gx = gradient_x_3d(_lap1, grid, padded=_lap1_pad)
        _gy = gradient_y_3d(_lap1, grid, padded=_lap1_pad)
        _lap2 = divergence_3d(_gx, _gy, grid).reshape(
            n_face_h, n_i_h, n_j_h, nlev_h, 4,
        )
        # Apply per-field hyperdiffusion coefficients.
        du_dt = du_dt - _coeff_uvT * _lap2[..., 0]
        dv_dt = dv_dt - _coeff_uvT * _lap2[..., 1]
        dtheta_p_dt = dtheta_p_dt - _coeff_uvT * _lap2[..., 2]
        drho_p_dt = drho_p_dt - _coeff_rho * _lap2[..., 3]
    elif _coeff_uvT > 0:
        n_face_h, n_i_h, n_j_h, nlev_h = u.shape
        hyper_stack = jnp.stack(
            [u, v, theta_p], axis=-1,
        )  # (6, n, n, nlev, 3)
        hyper_flat = hyper_stack.reshape(n_face_h, n_i_h, n_j_h, nlev_h * 3)
        hyper_out_flat = hyperdiffusion_3d(
            hyper_flat, grid, _coeff_uvT,
        )
        hyper_out = hyper_out_flat.reshape(n_face_h, n_i_h, n_j_h, nlev_h, 3)
        du_dt = du_dt + hyper_out[..., 0]
        dv_dt = dv_dt + hyper_out[..., 1]
        dtheta_p_dt = dtheta_p_dt + hyper_out[..., 2]
    elif _coeff_rho > 0:
        drho_p_dt = drho_p_dt + hyperdiffusion_3d(
            rho_p, grid, _coeff_rho,
        )

    # --- 14. Sponge layer ---
    sponge = _sponge_profile(
        height_coord.z_full, height_coord.H,
        config.sponge_width, config.sponge_coeff,
    )
    du_dt = du_dt - sponge * u
    dv_dt = dv_dt - sponge * v
    dtheta_p_dt = dtheta_p_dt - sponge * theta_p

    sponge_half = _sponge_profile(
        height_coord.z_half, height_coord.H,
        config.sponge_width, config.sponge_coeff,
    )

    # --- 15. w tendency (slow: horizontal advection) ---
    # Pre-pad ``w_full`` once and pass to both gradient_x_3d /
    # gradient_y_3d via ``padded=`` so they share the halo MPI exchange.
    w_full = 0.5 * (w[..., :-1] + w[..., 1:])
    _dg_w = getattr(grid, 'duogrid', None)
    _offsets_w = None if _dg_w is not None else grid.halo_interp_offsets
    _w_full_pad = pad_halo_4d(w_full, interp_offsets=_offsets_w, duogrid=_dg_w)
    dw_dx = gradient_x_3d(w_full, grid, padded=_w_full_pad)
    dw_dy = gradient_y_3d(w_full, grid, padded=_w_full_pad)
    horiz_adv_w = -(u * dw_dx + v * dw_dy)

    # Pad zero at top/bottom interfaces (rigid BC).  Single Pad HLO op
    # replaces alloc-zeros + scatter.
    pad_axes_w = ((0, 0),) * (w.ndim - 1)
    horiz_adv_w_half = jnp.pad(
        0.5 * (horiz_adv_w[..., :-1] + horiz_adv_w[..., 1:]),
        (*pad_axes_w, (1, 1)),
    )

    dw_dt = horiz_adv_w_half - sponge_half * w
    if config.hyperdiff_w_coeff > 0:
        dw_dt = dw_dt + hyperdiffusion_3d(w, grid, config.hyperdiff_w_coeff)

    # --- 16. Physics ---
    if physics_tendency is not None:
        du_dt = du_dt + physics_tendency.du_dt.data
        dv_dt = dv_dt + physics_tendency.dv_dt.data
        dw_dt = dw_dt + physics_tendency.dw_dt.data
        dtheta_p_dt = dtheta_p_dt + physics_tendency.dtheta_prime_dt.data
        drho_p_dt = drho_p_dt + physics_tendency.drho_prime_dt.data
        dtracers_dt = dtracers_dt + physics_tendency.dtracers_dt.data

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    dims_tr = ("face", "x", "y", "level", "tracer")

    return NonHydrostaticTendencies(
        du_dt=Field(data=du_dt, name="du_dt", dims=dims_3d, units="m/s^2"),
        dv_dt=Field(data=dv_dt, name="dv_dt", dims=dims_3d, units="m/s^2"),
        dw_dt=Field(data=dw_dt, name="dw_dt", dims=dims_w, units="m/s^2"),
        dtheta_prime_dt=Field(
            data=dtheta_p_dt, name="dtheta_prime_dt", dims=dims_3d, units="K/s",
        ),
        drho_prime_dt=Field(
            data=drho_p_dt, name="drho_prime_dt", dims=dims_3d, units="kg/m^3/s",
        ),
        dphis_dt=Field(
            data=jnp.zeros_like(state.phis.data),
            name="dphis_dt", dims=dims_2d, units="m^2/s^3",
        ),
        dtracers_dt=Field(
            data=dtracers_dt, name="dtracers_dt", dims=dims_tr, units="1/s",
        ),
    )


class CDGridCompressibleEulerModel(IntegrationMixin):
    """FV3-style C-D grid non-hydrostatic compressible Euler model.

    Parameters
    ----------
    grid : CubedSphereGrid
    height_coord : HeightCoordinate
    terrain_metric : TerrainMetric
    config : CDGridCompressibleEulerConfig, optional
    """

    def __init__(
        self,
        grid: CubedSphereGrid,
        height_coord: HeightCoordinate,
        terrain_metric: TerrainMetric,
        config: CDGridCompressibleEulerConfig | None = None,
    ):
        self.height_coord = height_coord
        self.terrain_metric = terrain_metric
        self.config = config or CDGridCompressibleEulerConfig()
        self._target_mass = None

        if self.config.small_earth_factor != 1.0:
            from legoesm.grids.cubed_sphere import apply_small_earth_scaling
            grid = apply_small_earth_scaling(grid, self.config.small_earth_factor)
        self.grid = grid
        self.cdgrid = create_cubed_sphere_cdgrid(grid)

        # Acoustic CFL check at construction time (outside JIT)
        from legoesm.core.cfl import estimate_min_dx_cubed_sphere
        import logging as _logging
        _ce_logger = _logging.getLogger("legoesm.compressible_euler")
        dx_min = estimate_min_dx_cubed_sphere(
            grid.n, getattr(grid, 'radius', constants.R_earth),
        )
        c_sound = float(jnp.sqrt(
            constants.c_pd / constants.c_vd * constants.R_d * 300.0
        ))
        self._dx_min = dx_min
        self._c_sound = c_sound
        _ce_logger.info(
            f"  Acoustic check: dx_min={dx_min/1000:.0f}km, c_s={c_sound:.0f}m/s, "
            f"n_substeps={self.config.n_acoustic_substeps}"
        )

    def tendencies(
        self,
        state: NonHydrostaticState,
        physics_tendency: NonHydrostaticTendencies | None = None,
    ) -> NonHydrostaticTendencies:
        return cdgrid_compressible_euler_slow_tendencies(
            state, self.grid, self.height_coord, self.terrain_metric,
            self.cdgrid, self.config, physics_tendency,
        )

    def step(self, state: NonHydrostaticState, dt: float, physics_fn=None) -> NonHydrostaticState:
        """Advance one time step using split-explicit RK3 with C-D grid transport.

        This non-jitted wrapper precomputes target mass outside the JIT
        boundary, then delegates to the jitted ``_step_jitted``.
        """
        # Precompute target mass outside JIT boundary (host-side only).
        if (self.config.fix_mass
                and self.config.anchor_mass_to_initial
                and self._target_mass is None):
            from legoesm.core.conservation import compute_nh_dry_mass
            self._target_mass = compute_nh_dry_mass(
                state.rho_prime.data, self.height_coord,
                self.terrain_metric, self.grid,
            )
        return self._step_jitted(state, dt, physics_fn=physics_fn)

    @partial(jax.jit, static_argnums=(0, 3))
    def _step_jitted(self, state: NonHydrostaticState, dt: float, physics_fn=None) -> NonHydrostaticState:
        """JIT-compiled step core."""
        acoustic_cfg = CompressibleEulerConfig(
            g=self.config.g,
            n_acoustic_substeps=self.config.n_acoustic_substeps,
            semi_implicit_acoustic=self.config.semi_implicit_acoustic,
            acoustic_off_centering=self.config.acoustic_off_centering,
        )

        se_config = SplitExplicitConfig(
            n_substeps=self.config.n_acoustic_substeps,
            outer_integrator=self.config.outer_integrator,
        )

        def slow_tendency_fn(s):
            phys = None
            if physics_fn is not None:
                _phys_result = physics_fn(
                    s, self.grid, self.height_coord, self.terrain_metric,
                )
                phys = _phys_result[0] if type(_phys_result) is tuple else _phys_result
            tend = cdgrid_compressible_euler_slow_tendencies(
                s, self.grid, self.height_coord, self.terrain_metric,
                self.cdgrid, self.config, phys,
            )
            return NonHydrostaticState(
                u=s.u.replace(data=tend.du_dt.data),
                v=s.v.replace(data=tend.dv_dt.data),
                w=s.w.replace(data=tend.dw_dt.data),
                theta_prime=s.theta_prime.replace(data=tend.dtheta_prime_dt.data),
                rho_prime=s.rho_prime.replace(data=tend.drho_prime_dt.data),
                phis=s.phis.replace(data=jnp.zeros_like(s.phis.data)),
                tracers=s.tracers.replace(data=tend.dtracers_dt.data),
            )

        def acoustic_update_fn(s, slow_tend, dt_s, n_sub, cfg):
            if self.config.semi_implicit_acoustic:
                return acoustic_substeps_semi_implicit(
                    s, slow_tend, dt_s, n_sub, cfg,
                    self.height_coord, self.terrain_metric, acoustic_cfg,
                )
            return acoustic_substeps(
                s, slow_tend, dt_s, n_sub, cfg,
                self.height_coord, self.terrain_metric, acoustic_cfg,
            )

        state_new = split_explicit_step(
            state, slow_tendency_fn, acoustic_update_fn, dt, se_config,
        )

        if self.config.fix_mass:
            from legoesm.core.conservation import (
                fix_mass_nonhydrostatic, compute_nh_dry_mass,
            )
            # _target_mass is precomputed in step() outside the JIT boundary.
            target = self._target_mass if self.config.anchor_mass_to_initial else (
                compute_nh_dry_mass(
                    state.rho_prime.data, self.height_coord,
                    self.terrain_metric, self.grid,
                )
            )
            state_new = fix_mass_nonhydrostatic(
                state_new, target, self.height_coord,
                self.terrain_metric, self.grid,
            )

        return state_new

    def step_with_physics(self, state, dt, physics_fn=None):
        return self.step(state, dt, physics_fn=physics_fn)

    # integrate() and integrate_scan() inherited from IntegrationMixin
