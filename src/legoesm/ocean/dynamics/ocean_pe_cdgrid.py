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
    from legoesm.ocean.dynamics.barotropic import fill_land_cells
    fill_TS = lambda field: jax.vmap(
        lambda f: fill_land_cells(f, mask, grid), in_axes=-1, out_axes=-1,
    )(field)
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
    dKE_dx, dKE_dy_perp = _arakawa_lamb_gradient(KE, cdgrid)
    # Fill land cells in p_prime before gradient so the 4-point stencil
    # sees smooth values at coastlines instead of the ocean-to-zero jump.
    p_prime_filled = jax.vmap(
        lambda f: fill_land_cells(f, mask, grid), in_axes=-1, out_axes=-1,
    )(p_prime)
    dp_dx, dp_dy_perp = _arakawa_lamb_gradient(p_prime_filled, cdgrid)
    # Downcast PGF results back to working precision
    dp_dx = dp_dx.astype(T.dtype)
    dp_dy_perp = dp_dy_perp.astype(T.dtype)

    # --- 11. Vorticity at corners (relative only) ---
    zeta_corner = _interp_center_to_corner(zeta, cdgrid)
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

    # Skew-symmetric correction
    div_corner = _interp_center_to_corner(div_v, cdgrid)
    du_d_dt = du_d_dt - 0.5 * u_d * div_corner
    dv_d_dt = dv_d_dt - 0.5 * v_d * div_corner

    # Boundary-corner fix: replace face-boundary corner tendencies with
    # nearest-interior values to eliminate O(dx) halo interpolation error
    # (mirrors atmosphere fix from commit f3f9a86).
    du_d_dt, dv_d_dt = _extrapolate_boundary_corners(du_d_dt, dv_d_dt, cdgrid.n)

    # --- 14. Convert D-grid tendencies back to cell-centre ---
    du_dt, dv_dt = dgrid_to_center_vector(du_d_dt, dv_d_dt)

    # --- 15. Vertical advection of u, v (cell-centre) ---
    du_dt = du_dt + _vertical_advection_ocean(u_a, w, z_coord, J)
    dv_dt = dv_dt + _vertical_advection_ocean(v_a, w, z_coord, J)

    # --- 16. Tracer tendencies ---
    # Use C-grid velocities for upwind advection of tracers at cell centres
    tracers = jnp.stack([T, S], axis=0)

    def tracer_tendency(tr):
        # Horizontal: FCT-limited advection with C-grid velocities
        # Uses Zalesak (1979) flux-corrected transport to ensure
        # monotonicity — eliminates overshoot/undershoot that unlimited
        # PPM produces on the cubed-sphere near panel boundaries.
        dtr_dt = cgrid_tracer_advection_fct(tr, u_c, v_c, cdgrid)
        # Vertical advection
        dtr_dt = dtr_dt + _vertical_advection_ocean(tr, w, z_coord, J)

        if config.K_h > 0:
            from legoesm.ocean.physics.mixing import laplacian_viscosity_3d
            dtr_dt = dtr_dt + laplacian_viscosity_3d(tr, grid, config.K_h)
        if config.K_v > 0:
            from legoesm.ocean.physics.mixing import vertical_diffusion
            dtr_dt = dtr_dt + vertical_diffusion(tr, z_coord, J, config.K_v)
        return dtr_dt

    tracer_tend = jax.vmap(tracer_tendency, in_axes=0, out_axes=0)(tracers)
    dT_dt = tracer_tend[0]
    dS_dt = tracer_tend[1]

    # --- 17. Mixing (always applied from config, grid-native operators) ---
    if config.A_h > 0:
        from legoesm.ocean.physics.mixing import laplacian_viscosity_3d
        vel_masked = jnp.stack([u_a * mask_3d, v_a * mask_3d], axis=0)
        vel_lap = jax.vmap(
            lambda q: laplacian_viscosity_3d(q, grid, config.A_h),
            in_axes=0, out_axes=0,
        )(vel_masked)
        du_dt = du_dt + vel_lap[0]
        dv_dt = dv_dt + vel_lap[1]
    if config.A_v > 0:
        from legoesm.ocean.physics.mixing import vertical_diffusion
        vel = jnp.stack([u_a, v_a], axis=0)
        vel_vdiff = jax.vmap(
            lambda q: vertical_diffusion(q, z_coord, J, config.A_v),
            in_axes=0, out_axes=0,
        )(vel)
        du_dt = du_dt + vel_vdiff[0]
        dv_dt = dv_dt + vel_vdiff[1]
    if config.hyperdiff_coeff > 0:
        from legoesm.core.operators_3d import hyperdiffusion_3d
        du_dt = du_dt + hyperdiffusion_3d(u_a * mask_3d, grid, config.hyperdiff_coeff)
        dv_dt = dv_dt + hyperdiffusion_3d(v_a * mask_3d, grid, config.hyperdiff_coeff)

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
