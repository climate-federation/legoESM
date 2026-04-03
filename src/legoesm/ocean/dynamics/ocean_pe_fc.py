"""FC-Gram Boussinesq Ocean PE on the cubed-sphere.

Replaces all horizontal operators with FC spectral versions.
Everything else is identical to ocean_pe_cdgrid: Wright EOS, hydrostatic
pressure, layer thickness, w diagnosis, skew-symmetric momentum,
vertical advection, vertical diffusion, land masking.

Divergence damping is applied when fc_config.div_damp_2 or
fc_config.div_damp_4 are nonzero.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.operators_fc import FCOperatorConfig
from legoesm.core.operators_fc_3d import (
    fc_curl_z_3d,
    fc_gradient_x_3d,
    fc_gradient_y_3d,
    fc_divergence_3d,
    fc_hyperdiffusion_3d,
    fc_scalar_advection_3d,
    fc_laplacian_3d,
    fc_divergence_damping_3d,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.eos import wright_eos, compute_hydrostatic_pressure
from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
    compute_layer_thickness,
    compute_ocean_jacobian,
)
from legoesm.ocean.state import OceanState, OceanTendencies, OceanConfig
from legoesm.ocean.physics.mixing import vertical_diffusion
from legoesm.ocean.vertical import (
    diagnose_w_from_flux_div as _diagnose_w_from_flux_div,
    vertical_advection_ocean as _vertical_advection_ocean,
)


def ocean_baroclinic_tendencies_fc(
    state: OceanState,
    grid: CubedSphereGrid,
    z_coord: OceanZStarCoordinate,
    fc_config: FCOperatorConfig,
    config: OceanConfig = OceanConfig(),
    physics_fn=None,
    surface_forcing=None,
) -> OceanTendencies:
    """Compute 3D baroclinic tendencies with FC spectral operators.

    Parameters
    ----------
    state : OceanState
    grid : CubedSphereGrid
    z_coord : OceanZStarCoordinate
    fc_config : FCOperatorConfig
    config : OceanConfig
    physics_fn : callable, optional

    Returns
    -------
    OceanTendencies
    """
    u = state.u.data
    v = state.v.data
    T = state.T.data
    S = state.S.data
    eta = state.eta.data
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
        eta_safe, H_bathy, z_coord, min_water_column_m=config.min_water_column_m,
    )
    h_k = compute_layer_thickness(
        eta_safe, H_bathy, z_coord, min_water_column_m=config.min_water_column_m,
    )

    # --- 2. Density from EOS ---
    p_hydro = compute_hydrostatic_pressure(
        jnp.full_like(T, rho_0), eta_safe, z_coord.dz_ref, J, rho_0, g,
    )
    rho = wright_eos(T, S, p_hydro)
    rho_prime = rho - rho_0

    # --- 3. Baroclinic pressure gradient ---
    dz_actual = z_coord.dz_ref * J[..., jnp.newaxis]
    dp_layer = rho_prime * g * dz_actual
    p_prime = jnp.cumsum(dp_layer, axis=-1) - dp_layer
    p_prime = p_prime + 0.5 * dp_layer

    dp_dx = fc_gradient_x_3d(p_prime, grid, fc_config)
    dp_dy = fc_gradient_y_3d(p_prime, grid, fc_config)

    # --- 4. Diagnose w from flux divergence ---
    flux_div_k = fc_divergence_3d(
        h_k * u * mask_3d, h_k * v * mask_3d, grid, fc_config,
    )
    w = _diagnose_w_from_flux_div(flux_div_k, z_coord)

    div_v = fc_divergence_3d(u * mask_3d, v * mask_3d, grid, fc_config)

    # --- 5. Vorticity ---
    zeta = fc_curl_z_3d(u * mask_3d, v * mask_3d, grid, fc_config)

    # --- 6. Kinetic energy gradient ---
    K = 0.5 * (u**2 + v**2)
    dK_dx = fc_gradient_x_3d(K, grid, fc_config)
    dK_dy = fc_gradient_y_3d(K, grid, fc_config)

    # --- 7. Vector-invariant momentum (skew-symmetric) ---
    H_total = jnp.maximum(jnp.sum(h_k, axis=-1), min_water_col)
    U_bar = jnp.sum(u * h_k, axis=-1) / H_total * mask
    V_bar = jnp.sum(v * h_k, axis=-1) / H_total * mask
    u_prime = (u - U_bar[..., jnp.newaxis]) * mask_3d
    v_prime = (v - V_bar[..., jnp.newaxis]) * mask_3d
    f_3d = grid.f[..., jnp.newaxis]

    du_dt = (zeta * v + f_3d * v_prime - dK_dx
             - 0.5 * u * div_v - dp_dx / rho_0)
    dv_dt = (-zeta * u - f_3d * u_prime - dK_dy
             - 0.5 * v * div_v - dp_dy / rho_0)

    # --- Divergence damping (only if fc_config requests it) ---
    if fc_config.div_damp_2 > 0 or fc_config.div_damp_4 > 0:
        du_damp, dv_damp = fc_divergence_damping_3d(
            u * mask_3d, v * mask_3d, grid, fc_config)
        du_dt = du_dt + du_damp
        dv_dt = dv_dt + dv_damp

    # --- 8. Vertical advection of u, v ---
    du_dt = du_dt + _vertical_advection_ocean(u, w, z_coord, J)
    dv_dt = dv_dt + _vertical_advection_ocean(v, w, z_coord, J)

    # --- 9. Tracer tendencies ---
    tracers = jnp.stack([T, S], axis=0)

    def tracer_tendency(tr: jnp.ndarray) -> jnp.ndarray:
        dtr_dt = fc_scalar_advection_3d(tr, u * mask_3d, v * mask_3d, grid, fc_config)
        dtr_dt = dtr_dt + _vertical_advection_ocean(tr, w, z_coord, J)

        if physics_fn is None:
            if config.K_h > 0:
                dtr_dt = dtr_dt + fc_laplacian_3d(tr, grid, fc_config) * config.K_h
            if config.K_v > 0:
                dtr_dt = dtr_dt + vertical_diffusion(tr, z_coord, J, config.K_v)
            if config.hyperdiff_coeff > 0:
                dtr_dt = dtr_dt + fc_hyperdiffusion_3d(tr, grid, fc_config, config.hyperdiff_coeff)

        return dtr_dt

    tracer_tend = jax.vmap(tracer_tendency, in_axes=0, out_axes=0)(tracers)
    dT_dt = tracer_tend[0]
    dS_dt = tracer_tend[1]

    # --- 10. Mixing ---
    if physics_fn is None:
        if config.A_h > 0:
            vel_masked = jnp.stack([u * mask_3d, v * mask_3d], axis=0)
            vel_lap = jax.vmap(
                lambda q: fc_laplacian_3d(q, grid, fc_config) * config.A_h,
                in_axes=0, out_axes=0,
            )(vel_masked)
            du_dt = du_dt + vel_lap[0]
            dv_dt = dv_dt + vel_lap[1]
        if config.A_v > 0:
            vel = jnp.stack([u, v], axis=0)
            vel_vdiff = jax.vmap(
                lambda q: vertical_diffusion(q, z_coord, J, config.A_v),
                in_axes=0, out_axes=0,
            )(vel)
            du_dt = du_dt + vel_vdiff[0]
            dv_dt = dv_dt + vel_vdiff[1]

        if config.hyperdiff_coeff > 0:
            du_dt = du_dt + fc_hyperdiffusion_3d(u * mask_3d, grid, fc_config, config.hyperdiff_coeff)
            dv_dt = dv_dt + fc_hyperdiffusion_3d(v * mask_3d, grid, fc_config, config.hyperdiff_coeff)
    else:
        phys = physics_fn(state, grid, z_coord, surface_forcing)
        du_dt = du_dt + phys.du_dt.data
        dv_dt = dv_dt + phys.dv_dt.data
        dT_dt = dT_dt + phys.dT_dt.data
        dS_dt = dS_dt + phys.dS_dt.data

    # --- 12. Land masking ---
    du_dt = du_dt * mask_3d
    dv_dt = dv_dt * mask_3d
    dT_dt = dT_dt * mask_3d
    dS_dt = dS_dt * mask_3d

    # --- 13. Free-surface tendency ---
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
