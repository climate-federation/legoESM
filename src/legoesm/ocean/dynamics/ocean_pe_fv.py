"""Consistent FV Ocean PE on the cubed-sphere.

Uses a compatible face-based discretization where tracer transport,
vertical velocity, and momentum damping share the same FV interface layer:

- Tracer transport (T, S): FV scalar advection (PPM)
- Vertical velocity: diagnosed from FV-consistent divergence (same
  interface velocities as PPM), ensuring discrete compatibility with
  the horizontal transport
- Divergence damping: selective damping using the same FV divergence
- Momentum: vector-invariant form with centered gradient (unchanged)

This is the ocean analog of atmosphere/dynamics/primitive_eq_fv.py.

References
----------
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core (FV3)
- Harris & Lin (2013): A Two-Way Nested Global-Regional Dynamical Core
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.operators_3d import (
    vorticity_3d,
    gradient_x_3d,
    gradient_y_3d,
    divergence_3d,
    hyperdiffusion_3d,
    fv_scalar_advection_3d,
)
from legoesm.core.operators_fv_cubed import (
    fv_divergence_3d as _fv_divergence_3d,
    fv_divergence_damping_3d as _fv_divergence_damping_3d,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.eos import wright_eos, compute_hydrostatic_pressure
from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
    compute_layer_thickness,
    compute_ocean_jacobian,
)
from legoesm.ocean.state import OceanState, OceanTendencies, OceanConfig
from legoesm.ocean.physics.mixing import laplacian_viscosity_3d, vertical_diffusion
from legoesm.ocean.dynamics.ocean_pe import (
    _diagnose_w_from_flux_div,
    _vertical_advection_ocean,
)


def ocean_baroclinic_tendencies_fv(
    state: OceanState,
    grid: CubedSphereGrid,
    z_coord: OceanZStarCoordinate,
    config: OceanConfig = OceanConfig(),
    physics_fn=None,
) -> OceanTendencies:
    """Compute 3D baroclinic tendencies with FV-consistent operators.

    Key differences from ocean_pe.py (centered):
    - Vertical velocity diagnosed from FV divergence (consistent with PPM)
    - PPM always used for tracer transport
    - FV divergence damping for cube-sphere stabilization

    Parameters
    ----------
    state : OceanState
    grid : CubedSphereGrid
    z_coord : OceanZStarCoordinate
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
    rho = wright_eos(T, S, jnp.zeros_like(T))
    for _ in range(2):
        p_hydro = compute_hydrostatic_pressure(
            rho, eta_safe, z_coord.dz_ref, J, rho_0, g,
        )
        rho = wright_eos(T, S, p_hydro)
    p_hydro = compute_hydrostatic_pressure(
        rho, eta_safe, z_coord.dz_ref, J, rho_0, g,
    )
    rho_prime = rho - rho_0

    # --- 3. Baroclinic pressure gradient ---
    dz_actual = z_coord.dz_ref * J[..., jnp.newaxis]
    dp_layer = rho_prime * g * dz_actual
    p_prime = jnp.cumsum(dp_layer, axis=-1) - dp_layer
    p_prime = p_prime + 0.5 * dp_layer

    dp_dx = gradient_x_3d(p_prime, grid)
    dp_dy = gradient_y_3d(p_prime, grid)

    # --- 4. Diagnose vertical velocity from FV-consistent flux divergence ---
    # Use FV divergence (same interface velocities as PPM transport) instead
    # of centered divergence, ensuring w is discretely compatible with
    # horizontal tracer transport.
    flux_div_k = _fv_divergence_3d(
        h_k * u * mask_3d, h_k * v * mask_3d, grid,
    )
    w = _diagnose_w_from_flux_div(flux_div_k, z_coord)

    # Velocity divergence for skew-symmetric momentum
    div_v = _fv_divergence_3d(u * mask_3d, v * mask_3d, grid)

    # --- 5. Vorticity ---
    zeta = vorticity_3d(u * mask_3d, v * mask_3d, grid)

    # --- 6. Bernoulli function ---
    K = 0.5 * (u**2 + v**2)
    dK_dx = gradient_x_3d(K, grid)
    dK_dy = gradient_y_3d(K, grid)

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

    # --- 8. FV divergence damping ---
    if config.div_damp_2 > 0 or config.div_damp_4 > 0:
        du_damp, dv_damp = _fv_divergence_damping_3d(
            u * mask_3d, v * mask_3d, grid,
            config.div_damp_2, config.div_damp_4,
        )
        du_dt = du_dt + du_damp
        dv_dt = dv_dt + dv_damp

    # --- 9. Vertical advection of u, v ---
    du_dt = du_dt + _vertical_advection_ocean(u, w, z_coord, J)
    dv_dt = dv_dt + _vertical_advection_ocean(v, w, z_coord, J)

    # --- 10. Tracer tendencies (PPM horizontal + vertical) ---
    tracers = jnp.stack([T, S], axis=0)

    def tracer_tendency(tr: jnp.ndarray) -> jnp.ndarray:
        # Always use PPM for FV discretization
        dtr_dt = fv_scalar_advection_3d(tr, u * mask_3d, v * mask_3d, grid)
        dtr_dt = dtr_dt + _vertical_advection_ocean(tr, w, z_coord, J)

        if physics_fn is None:
            if config.K_h > 0:
                dtr_dt = dtr_dt + laplacian_viscosity_3d(tr, grid, config.K_h)
            if config.K_v > 0:
                dtr_dt = dtr_dt + vertical_diffusion(tr, z_coord, J, config.K_v)
            if config.hyperdiff_coeff > 0:
                dtr_dt = dtr_dt + hyperdiffusion_3d(tr, grid, config.hyperdiff_coeff)

        return dtr_dt

    tracer_tend = jax.vmap(tracer_tendency, in_axes=0, out_axes=0)(tracers)
    dT_dt = tracer_tend[0]
    dS_dt = tracer_tend[1]

    # --- 11. Mixing ---
    if physics_fn is None:
        if config.A_h > 0:
            vel_masked = jnp.stack([u * mask_3d, v * mask_3d], axis=0)
            vel_lap = jax.vmap(
                lambda q: laplacian_viscosity_3d(q, grid, config.A_h),
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
            du_dt = du_dt + hyperdiffusion_3d(u * mask_3d, grid, config.hyperdiff_coeff)
            dv_dt = dv_dt + hyperdiffusion_3d(v * mask_3d, grid, config.hyperdiff_coeff)
    else:
        phys = physics_fn(state, grid, z_coord)
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
