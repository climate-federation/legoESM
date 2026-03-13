"""FC-Gram Ocean PE with divergence damping on the cubed-sphere.

Same as ocean_pe_fc.py but adds divergence damping to momentum.
Valuable for the ocean because barotropic gravity waves are divergent
modes that can cause ringing at panel boundaries.

Note: div damping applied to masked velocities (same pattern as
laplacian_viscosity_3d in ocean_pe.py step 10).
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
from legoesm.ocean.dynamics.ocean_pe import (
    _diagnose_w_from_flux_div,
    _vertical_advection_ocean,
)


def ocean_baroclinic_tendencies_fc_cgrid(
    state: OceanState,
    grid: CubedSphereGrid,
    z_coord: OceanZStarCoordinate,
    fc_config: FCOperatorConfig,
    config: OceanConfig = OceanConfig(),
    physics_fn=None,
) -> OceanTendencies:
    """Compute 3D baroclinic tendencies with FC operators + div damping.

    Parameters
    ----------
    state : OceanState
    grid : CubedSphereGrid
    z_coord : OceanZStarCoordinate
    fc_config : FCOperatorConfig
        Must have div_damp_2 and/or div_damp_4 set for damping to activate.
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

    J = compute_ocean_jacobian(
        eta_safe, H_bathy, z_coord, min_water_column_m=config.min_water_column_m,
    )
    h_k = compute_layer_thickness(
        eta_safe, H_bathy, z_coord, min_water_column_m=config.min_water_column_m,
    )

    p_hydro = compute_hydrostatic_pressure(
        jnp.full_like(T, rho_0), eta_safe, z_coord.dz_ref, J, rho_0, g,
    )
    rho = wright_eos(T, S, p_hydro)
    rho_prime = rho - rho_0

    dz_actual = z_coord.dz_ref * J[..., jnp.newaxis]
    dp_layer = rho_prime * g * dz_actual
    p_prime = jnp.cumsum(dp_layer, axis=-1) - dp_layer
    p_prime = p_prime + 0.5 * dp_layer

    dp_dx = fc_gradient_x_3d(p_prime, grid, fc_config)
    dp_dy = fc_gradient_y_3d(p_prime, grid, fc_config)

    flux_div_k = fc_divergence_3d(
        h_k * u * mask_3d, h_k * v * mask_3d, grid, fc_config,
    )
    w = _diagnose_w_from_flux_div(flux_div_k, z_coord)

    div_v = fc_divergence_3d(u * mask_3d, v * mask_3d, grid, fc_config)

    zeta = fc_curl_z_3d(u * mask_3d, v * mask_3d, grid, fc_config)

    K = 0.5 * (u**2 + v**2)
    dK_dx = fc_gradient_x_3d(K, grid, fc_config)
    dK_dy = fc_gradient_y_3d(K, grid, fc_config)

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

    # --- Divergence damping (applied to masked velocities) ---
    du_damp, dv_damp = fc_divergence_damping_3d(
        u * mask_3d, v * mask_3d, grid, fc_config)
    du_dt = du_dt + du_damp
    dv_dt = dv_dt + dv_damp

    du_dt = du_dt + _vertical_advection_ocean(u, w, z_coord, J)
    dv_dt = dv_dt + _vertical_advection_ocean(v, w, z_coord, J)

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
        phys = physics_fn(state, grid, z_coord)
        du_dt = du_dt + phys.du_dt.data
        dv_dt = dv_dt + phys.dv_dt.data
        dT_dt = dT_dt + phys.dT_dt.data
        dS_dt = dS_dt + phys.dS_dt.data

    du_dt = du_dt * mask_3d
    dv_dt = dv_dt * mask_3d
    dT_dt = dT_dt * mask_3d
    dS_dt = dS_dt * mask_3d

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
