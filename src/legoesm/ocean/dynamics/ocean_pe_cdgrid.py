"""Boussinesq Hydrostatic Primitive Equations on the C-D grid cubed-sphere.

FV3-style C-D grid discretisation for the ocean:

* D-grid winds ``u_d, v_d`` at cell corners (shape ``(6, n+1, n+1, nlev)``)
  are the prognostic velocity variables.
* C-grid velocities at cell edges are diagnosed for mass and tracer transport.
* Vorticity is computed from the integral circulation, exact on the D-grid.
* Bernoulli gradient uses the Arakawa-Lamb 4-point formula.
* Scalars (T, S, h, eta) live at cell centres.

The ocean state containers (``OceanState``, ``OceanTendencies``) use cell-centre
velocity storage ``(6, n, n, nlev)`` for compatibility with the rest of the
ocean infrastructure (barotropic solver, conservation fixers, etc.).
Conversion between cell-centre and D-grid is done at the tendency interface.

References
----------
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core
- Griffies (2004): Fundamentals of Ocean Climate Models
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.operators_cdgrid import (
    center_to_dgrid_vector,
    dgrid_to_center_vector,
    dgrid_to_cgrid,
    dgrid_vorticity,
    cgrid_divergence,
    cgrid_mass_flux_divergence,
    cgrid_tracer_advection_fct,
    _arakawa_lamb_gradient,
    _interp_center_to_corner,
    _extrapolate_boundary_corners,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.cubed_sphere_cdgrid import CubedSphereCDGrid
from legoesm.ocean.eos import make_eos_fn
from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
    compute_layer_thickness,
    compute_ocean_jacobian,
)
from legoesm.ocean.state import OceanState, OceanTendencies, OceanConfig
from legoesm.ocean.dynamics.ocean_tendency_common import (
    iterate_eos_and_pressure_anomaly,
)


# ==============================================================================
# Vertical velocity diagnosis
# ==============================================================================

from legoesm.ocean.vertical import (
    diagnose_w_from_flux_div as _diagnose_w_from_flux_div,
    vertical_advection_ocean as _vertical_advection_ocean,
)
from legoesm.ocean.dynamics.barotropic import fill_land_cells
from legoesm.ocean.physics.mixing import laplacian_viscosity_3d, vertical_diffusion
from legoesm.grids.halo import pad_halo_4d
from legoesm.core.operators_3d import hyperdiffusion_3d


# ==============================================================================
# Main tendency function
# ==============================================================================

def ocean_baroclinic_tendencies_cdgrid(
    state: OceanState,
    grid: CubedSphereGrid,
    z_coord: OceanZStarCoordinate,
    cdgrid: CubedSphereCDGrid,
    config: OceanConfig = OceanConfig(),
    physics_fn=None,
    surface_forcing=None,
) -> OceanTendencies:
    """Compute 3D baroclinic tendencies using C-D grid operators.

    The state uses cell-centre storage. Velocities are converted to D-grid
    for the momentum computation, then converted back.

    Parameters
    ----------
    state : OceanState
    grid : CubedSphereGrid
    z_coord : OceanZStarCoordinate
    cdgrid : CubedSphereCDGrid
    config : OceanConfig
    physics_fn : callable, optional

    Returns
    -------
    OceanTendencies
    """
    u_a = state.u.data       # (6, n, n, nlev)
    v_a = state.v.data
    T = state.T.data
    S = state.S.data
    eta = state.eta.data      # (6, n, n)
    H_bathy = state.H_bathy.data
    mask = state.land_mask.data
    mask_3d = mask[..., jnp.newaxis]

    g = config.g
    rho_0 = config.rho_0
    min_water_col = jnp.asarray(config.min_water_column_m, dtype=eta.dtype)
    eta_floor = min_water_col - H_bathy
    eta_safe = jnp.maximum(eta, eta_floor) * mask

    # --- 1. Layer thickness and Jacobian ---
    J = compute_ocean_jacobian(
        eta_safe, H_bathy, z_coord,
        min_water_column_m=config.min_water_column_m,
    )
    h_k = compute_layer_thickness(
        eta_safe, H_bathy, z_coord,
        min_water_column_m=config.min_water_column_m,
    )

    # --- 2. Density from EOS + 3. Baroclinic pressure anomaly ---
    # Reference Jacobian (J=1, eta=0): the barotropic solver handles
    # -g*grad(eta) and using the actual J here would double-count the
    # free-surface contribution (see #109).
    #
    # The cubed-sphere path runs the cumsum in float64: at depth p has
    # ULP = 0.0625 Pa in float32, so the halo-exchange interpolation
    # of float32 values at face boundaries leaks O(ULP/dx) ≈ 2e-7 Pa/m
    # — a spurious PGF that drives rest-state instability.  Keeping
    # p_prime in float64 reduces the leak by 9 orders of magnitude.
    # ``fill_land_cells`` is ndim-aware: it uses ``pad_halo_4d`` for 4D
    # input so all vertical levels share one MPI halo exchange per pass
    # (instead of nlev separate exchanges under the prior vmap).
    fill_TS = lambda field: fill_land_cells(field, mask, grid)
    eos_fn = make_eos_fn(config.eos, getattr(config, 'eos_linear', None))
    rho, rho_prime, p_prime = iterate_eos_and_pressure_anomaly(
        T, S, mask, fill_TS, eos_fn, z_coord.dz_ref, rho_0, g,
        n_iter=2, hi_precision_pressure=True,
    )

    # --- 4. Convert to D-grid ---
    u_d, v_d = center_to_dgrid_vector(u_a * mask_3d, v_a * mask_3d, cdgrid)

    # --- 5. C-grid velocities for mass transport ---
    u_c, v_c = dgrid_to_cgrid(u_d, v_d, cdgrid)

    # --- 6. Flux divergence for vertical velocity ---
    # Use cell-centre for flux divergence (cell-centre h_k and velocities)
    flux_div_k = cgrid_mass_flux_divergence(
        h_k, u_c, v_c, cdgrid,
    )
    w = _diagnose_w_from_flux_div(flux_div_k, z_coord)

    # --- 7. Velocity divergence for skew-symmetric correction ---
    div_v = cgrid_divergence(u_c, v_c, cdgrid)

    # --- 8. Vorticity ---
    zeta = dgrid_vorticity(u_d, v_d, cdgrid)

    # --- 9. KE at cell centres from D-grid (orthogonal basis) ---
    u_cc_ke, v_cc_ke = dgrid_to_center_vector(u_d, v_d)
    KE = 0.5 * (u_cc_ke ** 2 + v_cc_ke ** 2)

    # --- 10. Bernoulli and pressure gradients at D-grid corners ---
    # Fill land cells in p_prime before gradient so the 4-point stencil
    # sees smooth values at coastlines instead of the ocean-to-zero jump.
    # ``fill_land_cells`` natively handles 4D input (single halo exchange
    # across all levels), so call it directly.
    p_prime_filled = fill_land_cells(p_prime, mask, grid)
    # Batch the two Arakawa-Lamb gradients (KE, p_prime_filled) into a
    # single call — both are 3D scalar fields on (face, n, n, nlev) and
    # the operator treats the trailing axis as a passive batch.  Same
    # exploit as Loop 119 (CD-grid CE) for K and pi_prime.  2 gradients
    # → 1 (one halo exchange + one 4-point finite-difference + one 2x2
    # metric-matrix multiply on the thicker tensor).
    n_face_kp, n_i_kp, n_j_kp, nlev_kp = KE.shape
    _kp_stack = jnp.stack([KE, p_prime_filled], axis=-1)
    _kp_flat = _kp_stack.reshape(n_face_kp, n_i_kp, n_j_kp, nlev_kp * 2)
    _dkp_dx_flat, _dkp_dy_perp_flat = _arakawa_lamb_gradient(_kp_flat, cdgrid)
    _dkp_dx = _dkp_dx_flat.reshape(
        _dkp_dx_flat.shape[0], _dkp_dx_flat.shape[1],
        _dkp_dx_flat.shape[2], nlev_kp, 2,
    )
    _dkp_dy_perp = _dkp_dy_perp_flat.reshape(
        _dkp_dy_perp_flat.shape[0], _dkp_dy_perp_flat.shape[1],
        _dkp_dy_perp_flat.shape[2], nlev_kp, 2,
    )
    dKE_dx = _dkp_dx[..., 0]
    dp_dx = _dkp_dx[..., 1]
    dKE_dy_perp = _dkp_dy_perp[..., 0]
    dp_dy_perp = _dkp_dy_perp[..., 1]
    # Downcast PGF results back to working precision
    dp_dx = dp_dx.astype(T.dtype)
    dp_dy_perp = dp_dy_perp.astype(T.dtype)

    # --- 11. Vorticity + divergence at corners (batched) ---
    # Batch the (zeta, div_v) center-to-corner interpolation: both are
    # cell-centre (face, n, n, nlev) fields and the operator treats
    # the trailing axis as a passive batch.  Stack and fold so a
    # single halo + 4-point average serves both interps.  Same
    # passive-trailing-axis pattern as the CD-grid PE corner interps.
    n_face_zd, n_i_zd, n_j_zd, nlev_zd = zeta.shape
    _zd_stack = jnp.stack([zeta, div_v], axis=-1)
    _zd_corner_flat = _interp_center_to_corner(
        _zd_stack.reshape(n_face_zd, n_i_zd, n_j_zd, nlev_zd * 2), cdgrid,
    )
    _zd_corner = _zd_corner_flat.reshape(
        _zd_corner_flat.shape[0], _zd_corner_flat.shape[1],
        _zd_corner_flat.shape[2], nlev_zd, 2,
    )
    zeta_corner = _zd_corner[..., 0]
    div_corner = _zd_corner[..., 1]
    f_corner_3d = cdgrid.f_corner[:, :, :, None]   # (6, n+1, n+1, 1)

    # --- 12. Baroclinic Coriolis split ---
    # Planetary Coriolis: barotropic part (f*v_bar) handled by barotropic
    # substeps; here only the baroclinic deviation is included.
    H_total = jnp.maximum(jnp.sum(h_k, axis=-1), min_water_col)
    U_bar_a = jnp.sum(u_a * h_k, axis=-1) / H_total * mask
    V_bar_a = jnp.sum(v_a * h_k, axis=-1) / H_total * mask
    u_prime_a = (u_a - U_bar_a[..., jnp.newaxis]) * mask_3d
    v_prime_a = (v_a - V_bar_a[..., jnp.newaxis]) * mask_3d
    u_prime_d, v_prime_d = center_to_dgrid_vector(u_prime_a, v_prime_a, cdgrid)

    # --- 13. D-grid momentum tendencies ---
    # ζ*v + f*v' (relative vorticity × full velocity, Coriolis × deviation)
    du_d_dt = (zeta_corner * v_d + f_corner_3d * v_prime_d
               - dKE_dx - dp_dx / rho_0)
    dv_d_dt = (-zeta_corner * u_d - f_corner_3d * u_prime_d
               - dKE_dy_perp - dp_dy_perp / rho_0)

    # Skew-symmetric correction (``div_corner`` was already computed
    # alongside ``zeta_corner`` via the batched corner interpolation).
    du_d_dt = du_d_dt - 0.5 * u_d * div_corner
    dv_d_dt = dv_d_dt - 0.5 * v_d * div_corner

    # Boundary-corner fix: replace face-boundary corner tendencies with
    # nearest-interior values to eliminate O(dx) halo interpolation error
    # (mirrors atmosphere fix from commit f3f9a86).
    du_d_dt, dv_d_dt = _extrapolate_boundary_corners(du_d_dt, dv_d_dt, cdgrid.n)

    # --- 14. Convert D-grid tendencies back to cell-centre ---
    du_dt, dv_dt = dgrid_to_center_vector(du_d_dt, dv_d_dt)

    # --- 15. Vertical advection of u, v (cell-centre) ---
    # Batch the two ``_vertical_advection_ocean`` calls by stacking
    # (u_a, v_a) along a new leading axis.  ``w_full`` / ``jac_safe`` /
    # ``dz_half`` depend only on (w, z_coord, J), so they are
    # computed once and the trailing-axis ``[..., :-1] - [..., 1:]``
    # upwind gradient broadcasts across the new axis.  Same
    # leading-axis batching as Loop 142 in CD-grid CE / PE.
    _uv_a_va = jnp.stack([u_a, v_a], axis=0)
    _uv_a_va_adv = _vertical_advection_ocean(_uv_a_va, w, z_coord, J)
    du_dt = du_dt + _uv_a_va_adv[0]
    dv_dt = dv_dt + _uv_a_va_adv[1]

    # --- 16. Tracer tendencies ---
    # Use C-grid velocities for upwind advection of tracers at cell centres.
    # Stack T, S along a trailing tracer axis and fold it into the level
    # axis so the halo-issuing operators (cgrid_tracer_advection_fct,
    # laplacian_viscosity_3d) run ONCE for both tracers instead of being
    # called twice under vmap-over-(T,S) — each vmap'd call would emit
    # its own pad_halo_4d MPI exchange.  Vertical operators stay
    # per-tracer because they hard-code the vertical axis at -1.
    tracer_stack = jnp.stack([T, S], axis=-1)  # (6, n, n, nlev, 2)
    n_face, n_i, n_j, nlev_t, n_tracers = tracer_stack.shape
    tracer_flat = tracer_stack.reshape(n_face, n_i, n_j, nlev_t * n_tracers)
    # Broadcast C-grid velocities across the combined (level × tracer)
    # axis.  ``tracer_flat`` reshape interleaves levels and tracers as
    # ``[lev0/trc0, lev0/trc1, ..., lev1/trc0, ...]`` — each level's
    # velocity must be duplicated ``n_tracers`` times to align, which
    # ``jnp.repeat`` does directly.  ``jnp.tile`` would instead
    # concatenate the entire array and mis-align tracer ↔ level.
    if n_tracers == 1:
        u_c_b, v_c_b = u_c, v_c
    else:
        u_c_b = jnp.repeat(u_c, n_tracers, axis=-1)
        v_c_b = jnp.repeat(v_c, n_tracers, axis=-1)

    horiz_flat = cgrid_tracer_advection_fct(tracer_flat, u_c_b, v_c_b, cdgrid)
    if config.K_h > 0:
        horiz_flat = horiz_flat + laplacian_viscosity_3d(
            tracer_flat, grid, config.K_h,
        )
    horiz_stack = horiz_flat.reshape(n_face, n_i, n_j, nlev_t, n_tracers)

    # Vertical advection per-tracer (vmap over the trailing tracer axis
    # so JAX produces one batched kernel rather than n_tracers unrolled
    # stencils).
    def _vert_adv(q):
        return _vertical_advection_ocean(q, w, z_coord, J)

    vert_adv_stack = jax.vmap(_vert_adv, in_axes=-1, out_axes=-1)(tracer_stack)

    if config.K_v > 0:

        def _vdiff(q):
            return vertical_diffusion(q, z_coord, J, config.K_v)

        vdiff_stack = jax.vmap(_vdiff, in_axes=-1, out_axes=-1)(tracer_stack)
        tracer_tend_stack = horiz_stack + vert_adv_stack + vdiff_stack
    else:
        tracer_tend_stack = horiz_stack + vert_adv_stack

    dT_dt = tracer_tend_stack[..., 0]
    dS_dt = tracer_tend_stack[..., 1]

    # --- 17. Mixing (always applied from config, grid-native operators) ---
    # Stack u, v along a trailing axis and fold it into the level dim so
    # the halo-issuing horizontal viscosity operators
    # (``laplacian_viscosity_3d``, ``hyperdiffusion_3d``) run ONCE on the
    # thicker (6, n, n, nlev*2) field instead of issuing two separate
    # pad_halo_4d MPI exchanges per call.  Vertical diffusion stays
    # per-component (axis -1 = nlev hard-coded, no halo).
    n_face_v, n_i_v, n_j_v, nlev_v = u_a.shape
    if config.A_h > 0 or config.hyperdiff_coeff > 0:
        vel_masked_stack = jnp.stack(
            [u_a * mask_3d, v_a * mask_3d], axis=-1,
        )  # (6, n, n, nlev, 2)
        vel_masked_flat = vel_masked_stack.reshape(
            n_face_v, n_i_v, n_j_v, nlev_v * 2,
        )
        # Pre-pad the (u, v)-stack ONCE so the explicit Laplacian
        # (``laplacian_viscosity_3d``) and the inner Laplacian of the
        # biharmonic hyperdiffusion (``hyperdiffusion_3d``) share the
        # same halo on ``vel_masked_flat`` instead of issuing two
        # independent ``pad_halo_4d`` collectives on the same input.
        # Saves 1 MPI message per RHS evaluation when both A_h and
        # hyperdiff_coeff are non-zero — the dominant ocean test config.
        _dg_oc = getattr(grid, 'duogrid', None)
        _offsets_oc = None if _dg_oc is not None else grid.halo_interp_offsets
        vel_masked_pad = pad_halo_4d(
            vel_masked_flat, interp_offsets=_offsets_oc, duogrid=_dg_oc,
        )
    if config.A_h > 0:
        vel_lap_flat = laplacian_viscosity_3d(
            vel_masked_flat, grid, config.A_h, padded=vel_masked_pad,
        )
        vel_lap = vel_lap_flat.reshape(n_face_v, n_i_v, n_j_v, nlev_v, 2)
        du_dt = du_dt + vel_lap[..., 0]
        dv_dt = dv_dt + vel_lap[..., 1]
    if config.A_v > 0:

        def _vdiff_uv(q):
            return vertical_diffusion(q, z_coord, J, config.A_v)

        vel_uv = jnp.stack([u_a, v_a], axis=-1)  # (6, n, n, nlev, 2)
        vel_vdiff = jax.vmap(_vdiff_uv, in_axes=-1, out_axes=-1)(vel_uv)
        du_dt = du_dt + vel_vdiff[..., 0]
        dv_dt = dv_dt + vel_vdiff[..., 1]
    if config.hyperdiff_coeff > 0:
        vel_hyper_flat = hyperdiffusion_3d(
            vel_masked_flat, grid, config.hyperdiff_coeff,
            padded=vel_masked_pad,
        )
        vel_hyper = vel_hyper_flat.reshape(n_face_v, n_i_v, n_j_v, nlev_v, 2)
        du_dt = du_dt + vel_hyper[..., 0]
        dv_dt = dv_dt + vel_hyper[..., 1]

    # --- 17b. Physics tendencies (surface forcing, bottom drag, etc.) ---
    if physics_fn is not None:
        phys = physics_fn(state, grid, z_coord, surface_forcing)
        du_dt = du_dt + phys.du_dt.data
        dv_dt = dv_dt + phys.dv_dt.data
        dT_dt = dT_dt + phys.dT_dt.data
        dS_dt = dS_dt + phys.dS_dt.data

    # --- 18. Land masking ---
    du_dt = du_dt * mask_3d
    dv_dt = dv_dt * mask_3d
    dT_dt = dT_dt * mask_3d
    dS_dt = dS_dt * mask_3d

    # --- 19. Free-surface tendency ---
    deta_dt = -jnp.sum(flux_div_k, axis=-1) * mask

    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")

    return OceanTendencies(
        du_dt=Field(data=du_dt, name="du_dt", dims=dims_3d, units="m/s^2"),
        dv_dt=Field(data=dv_dt, name="dv_dt", dims=dims_3d, units="m/s^2"),
        dT_dt=Field(data=dT_dt, name="dT_dt", dims=dims_3d, units="degC/s"),
        dS_dt=Field(data=dS_dt, name="dS_dt", dims=dims_3d, units="PSU/s"),
        deta_dt=Field(data=deta_dt, name="deta_dt", dims=dims_2d, units="m/s"),
        dH_bathy_dt=Field(
            data=jnp.zeros_like(H_bathy), name="dH_bathy_dt",
            dims=dims_2d, units="m/s",
        ),
        dland_mask_dt=Field(
            data=jnp.zeros_like(mask), name="dland_mask_dt",
            dims=dims_2d, units="1/s",
        ),
    )
