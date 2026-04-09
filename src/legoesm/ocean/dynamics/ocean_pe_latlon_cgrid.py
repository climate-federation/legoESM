"""Boussinesq Hydrostatic Primitive Equations on the lat-lon C-grid (FV).

C-grid staggering:
  u at lon interfaces  (n_lat, n_lon+1, nlev)
  v at lat interfaces  (n_lat+1, n_lon, nlev)
  eta, T, S at cell centers (n_lat, n_lon [, nlev])

Key advantages over the A-grid formulation (ocean_pe_latlon.py):
- Pressure gradient uses compact 1-cell stencil -> no 2*dx null space
- Divergence sums actual face fluxes -> no checkerboard mode
- Coriolis coupling is exact at face points with Sadourny averaging

Boundary conditions:
- Longitude: periodic (u wraps at j=0 and j=n_lon)
- Latitude: solid wall at poles (v=0 at i=0 and i=n_lat)

References
----------
- Griffies (2004): Fundamentals of Ocean Climate Models (MOM framework)
- Sadourny (1975): The Dynamics of Finite-Difference Models of the
  Shallow-Water Equations
- Arakawa & Lamb (1977): Computational Design of the Basic Dynamical
  Processes of the UCLA GCM
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.grids.latlon import LatLonGrid
from legoesm.ocean.eos import compute_hydrostatic_pressure, make_eos_fn
from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
    compute_layer_thickness,
    compute_ocean_jacobian,
)
from legoesm.ocean.state import (
    LatLonCGridOceanState,
    LatLonCGridOceanTendencies,
    LatLonCGridOceanConfig,
)
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    divergence_cgrid,
    gradient_x_cgrid,
    gradient_y_cgrid,
    laplacian_cgrid,
    vector_laplacian_cgrid,
)
from legoesm.ocean.vertical import (
    diagnose_w_from_flux_div as _diagnose_w_from_flux_div,
    vertical_advection_ocean as _vertical_advection_ocean,
)


def _interp_to_u_points(f: jnp.ndarray) -> jnp.ndarray:
    """Interpolate cell-center field to u-points (lon interfaces).

    Parameters
    ----------
    f : array, shape (n_lat, n_lon, ...) at cell centers.

    Returns
    -------
    f_u : array, shape (n_lat, n_lon+1, ...) at u-points.
    """
    # Face j is between cell (j-1) mod n_lon and cell j
    f_left = jnp.roll(f, 1, axis=1)
    f_avg = 0.5 * (f_left + f)
    if f.ndim >= 3:
        return jnp.concatenate([f_avg, f_avg[:, 0:1, :]], axis=1)
    else:
        return jnp.concatenate([f_avg, f_avg[:, 0:1]], axis=1)


def _interp_to_v_points(f: jnp.ndarray) -> jnp.ndarray:
    """Interpolate cell-center field to v-points (lat interfaces).

    Parameters
    ----------
    f : array, shape (n_lat, n_lon, ...) at cell centers.

    Returns
    -------
    f_v : array, shape (n_lat+1, n_lon, ...) at v-points.
    """
    f_interior = 0.5 * (f[:-1] + f[1:])  # (n_lat-1, n_lon, ...)
    if f.ndim >= 3:
        zero = jnp.zeros((1, f.shape[1], f.shape[2]), dtype=f.dtype)
    else:
        zero = jnp.zeros((1, f.shape[1]), dtype=f.dtype)
    return jnp.concatenate([zero, f_interior, zero], axis=0)


def _neumann_fill_cgrid(
    f: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Fill land cells with nearest ocean-neighbor (Neumann BC).

    Same algorithm as latlon_operators.neumann_fill_latlon.
    """
    m = mask
    filled = f
    for _ in range(3):
        f_s = jnp.concatenate([filled[0:1], filled[:-1]], axis=0)
        m_s = jnp.concatenate([m[0:1], m[:-1]], axis=0)
        f_n = jnp.concatenate([filled[1:], filled[-1:]], axis=0)
        m_n = jnp.concatenate([m[1:], m[-1:]], axis=0)
        f_w = jnp.roll(filled, 1, axis=1)
        m_w = jnp.roll(m, 1, axis=1)
        f_e = jnp.roll(filled, -1, axis=1)
        m_e = jnp.roll(m, -1, axis=1)

        is_land = m < 0.5

        if f.ndim > 2:
            m_s_e = m_s[..., jnp.newaxis]
            m_n_e = m_n[..., jnp.newaxis]
            m_w_e = m_w[..., jnp.newaxis]
            m_e_e = m_e[..., jnp.newaxis]
            is_land_e = is_land[..., jnp.newaxis]
        else:
            m_s_e = m_s
            m_n_e = m_n
            m_w_e = m_w
            m_e_e = m_e
            is_land_e = is_land

        nbr_sum = f_s * m_s_e + f_n * m_n_e + f_w * m_w_e + f_e * m_e_e
        nbr_count = m_s_e + m_n_e + m_w_e + m_e_e
        nbr_avg = nbr_sum / jnp.maximum(nbr_count, 1.0)

        has_any_nbr = (m_s + m_n + m_w + m_e) > 0.0
        if f.ndim > 2:
            has_any_nbr_e = has_any_nbr[..., jnp.newaxis]
        else:
            has_any_nbr_e = has_any_nbr

        filled = jnp.where(is_land_e & has_any_nbr_e, nbr_avg, filled)
        m = jnp.where(is_land & has_any_nbr, 1.0, m)

    return filled


def latlon_cgrid_ocean_baroclinic_tendencies(
    state: LatLonCGridOceanState,
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    config: LatLonCGridOceanConfig = LatLonCGridOceanConfig(),
    physics_fn=None,
    surface_forcing=None,
) -> LatLonCGridOceanTendencies:
    """Compute 3D baroclinic tendencies on a C-grid lat-lon grid.

    Parameters
    ----------
    state : LatLonCGridOceanState
    grid : LatLonGrid
    z_coord : OceanZStarCoordinate
    config : LatLonCGridOceanConfig
    physics_fn : callable, optional
    surface_forcing : optional

    Returns
    -------
    LatLonCGridOceanTendencies
    """
    u = state.u.data       # (n_lat, n_lon+1, nlev)
    v = state.v.data       # (n_lat+1, n_lon, nlev)
    T = state.T.data       # (n_lat, n_lon, nlev)
    S = state.S.data
    eta = state.eta.data   # (n_lat, n_lon)
    H_bathy = state.H_bathy.data
    mask = state.land_mask.data
    u_mask = state.u_mask.data
    v_mask = state.v_mask.data
    mask_3d = mask[..., jnp.newaxis]
    u_mask_3d = u_mask[..., jnp.newaxis]
    v_mask_3d = v_mask[..., jnp.newaxis]

    g_val = config.g
    rho_0 = config.rho_0
    min_water_col = jnp.asarray(config.min_water_column_m, dtype=eta.dtype)
    eta_floor = min_water_col - H_bathy
    eta_safe = jnp.maximum(eta, eta_floor) * mask

    n_lat = grid.n_lat
    n_lon = grid.n_lon
    nlev = z_coord.n_levels

    # --- 1. Layer thickness and Jacobian ---
    J = compute_ocean_jacobian(
        eta_safe, H_bathy, z_coord,
        min_water_column_m=config.min_water_column_m,
    )
    h_k = compute_layer_thickness(
        eta_safe, H_bathy, z_coord,
        min_water_column_m=config.min_water_column_m,
    )

    # --- 2. Density from EOS ---
    T_filled = _neumann_fill_cgrid(T, mask)
    S_filled = _neumann_fill_cgrid(S, mask)
    eos_fn = make_eos_fn(config.eos, getattr(config, 'eos_linear', None))
    # Use REFERENCE Jacobian (J=1, eta=0) for the hydrostatic pressure
    # in the EOS iteration.  The barotropic solver handles the
    # free-surface pressure gradient g*grad(eta); using the actual J
    # here would create a spatially-varying pressure even for uniform
    # T/S, double-counting the barotropic forcing.
    J_ref = jnp.ones_like(J)
    eta_ref = jnp.zeros_like(eta_safe)
    rho = eos_fn(T_filled, S_filled, jnp.zeros_like(T))
    for _ in range(2):
        p_hydro = compute_hydrostatic_pressure(
            rho, eta_ref, z_coord.dz_ref, J_ref, rho_0, g_val,
        )
        rho = eos_fn(T_filled, S_filled, p_hydro)
    p_hydro = compute_hydrostatic_pressure(
        rho, eta_ref, z_coord.dz_ref, J_ref, rho_0, g_val,
    )
    rho_prime = rho - rho_0

    # --- 3. Baroclinic pressure gradient (compact C-grid stencil) ---
    # Use REFERENCE layer thickness (dz_ref, corresponding to eta=0)
    # rather than the actual thickness (dz_ref * J) which includes the
    # free-surface contribution.  The barotropic solver handles
    # g*grad(eta); using J here would double-count that forcing.
    dp_layer = rho_prime * g_val * z_coord.dz_ref
    p_prime = jnp.cumsum(dp_layer, axis=-1) - dp_layer
    p_prime = p_prime + 0.5 * dp_layer

    p_prime_filled = _neumann_fill_cgrid(p_prime, mask)

    # C-grid gradient: compact stencil at face points
    # vmap over levels for 3D gradient
    p_t = jnp.moveaxis(p_prime_filled, -1, 0)  # (nlev, n_lat, n_lon)

    dp_dx_t = jax.vmap(lambda p2d: gradient_x_cgrid(p2d, grid))(p_t)
    dp_dy_t = jax.vmap(lambda p2d: gradient_y_cgrid(p2d, grid))(p_t)

    dp_dx = jnp.moveaxis(dp_dx_t, 0, -1)  # (n_lat, n_lon+1, nlev)
    dp_dy = jnp.moveaxis(dp_dy_t, 0, -1)  # (n_lat+1, n_lon, nlev)

    # --- 4. Vertical velocity from FV flux divergence ---
    # Divergence needs face fluxes: h*u at u-points, h*v at v-points.
    # Uses FULL velocity (barotropic + baroclinic) for mass transport.
    h_u = _interp_to_u_points(h_k)
    h_v = _interp_to_v_points(h_k)
    flux_div_k = divergence_cgrid(
        h_u * u * u_mask_3d, h_v * v * v_mask_3d, grid,
    )
    w = _diagnose_w_from_flux_div(flux_div_k, z_coord)

    # --- 4b. Baroclinic perturbation velocity ---
    # The barotropic solver handles the depth-averaged momentum.
    # The baroclinic step must operate on the PERTURBATION velocity
    # u' = u - U_bar to avoid double-counting the barotropic tendency.
    U_bar = jnp.sum(u * h_u, axis=-1) / jnp.maximum(
        jnp.sum(h_u, axis=-1), 1e-10) * u_mask  # (n_lat, n_lon+1)
    V_bar = jnp.sum(v * h_v, axis=-1) / jnp.maximum(
        jnp.sum(h_v, axis=-1), 1e-10) * v_mask  # (n_lat+1, n_lon)
    u_prime = u - U_bar[..., jnp.newaxis]
    v_prime = v - V_bar[..., jnp.newaxis]

    # --- 5. Coriolis ---
    # Coriolis is NOT included in the returned momentum tendencies.
    # It is applied as a forward-backward (Matsuno) step in the step
    # function (ocean_model_latlon_cgrid.py), which is unconditionally
    # stable for inertial oscillations.  Forward Euler Coriolis amplifies
    # by sqrt(1 + (f*dt)^2) per step and blows up within ~1 day at
    # high latitudes.

    # --- 6. Kinetic energy gradient (from perturbation velocity) ---
    up_cell = 0.5 * (u_prime[:, :-1, :] + u_prime[:, 1:, :])
    vp_cell = 0.5 * (v_prime[:-1, :, :] + v_prime[1:, :, :])
    KE = 0.5 * (up_cell**2 + vp_cell**2)

    KE_t = jnp.moveaxis(KE, -1, 0)
    dKE_dx_t = jax.vmap(lambda ke2d: gradient_x_cgrid(ke2d, grid))(KE_t)
    dKE_dy_t = jax.vmap(lambda ke2d: gradient_y_cgrid(ke2d, grid))(KE_t)
    dKE_dx = jnp.moveaxis(dKE_dx_t, 0, -1)
    dKE_dy = jnp.moveaxis(dKE_dy_t, 0, -1)

    # --- 7. Momentum tendencies (non-Coriolis only) ---
    du_dt = -dKE_dx - dp_dx / rho_0
    dv_dt = -dKE_dy - dp_dy / rho_0

    # --- 8. Vertical advection of u, v (perturbation velocity) ---
    w_u = _interp_to_u_points(w)
    w_v = _interp_to_v_points(w)
    du_dt = du_dt + _vertical_advection_ocean(
        u_prime, w_u, z_coord, _interp_to_u_points(J))
    dv_dt = dv_dt + _vertical_advection_ocean(
        v_prime, w_v, z_coord, _interp_to_v_points(J))

    # --- 9. Tracer tendencies (diffusion + physics only) ---
    # Horizontal AND vertical tracer advection are handled in the step()
    # function using barotropic-averaged transport (Hallberg 1997, #102).
    # Vertical velocity w is diagnosed from the barotropic-averaged
    # per-layer divergence, ensuring 3D transport consistency.
    #
    # The tendency here includes only: diffusion and physics.
    h_safe = jnp.maximum(h_k, 1e-10)
    tracers = jnp.stack([T, S], axis=0)

    def tracer_tendency(tr: jnp.ndarray) -> jnp.ndarray:
        dtr_dt = jnp.zeros_like(tr)

        if config.K_h > 0:
            dtr_dt = dtr_dt + config.K_h * laplacian_cgrid(tr, grid, mask=mask)
        if physics_fn is None:
            if config.K_v > 0 and tr.shape[-1] >= 2:
                jac_v = jnp.maximum(J[..., jnp.newaxis], 1e-10)
                dz_actual_loc = z_coord.dz_ref * jac_v
                dtr_dz_half = jnp.diff(tr, axis=-1) / (
                    z_coord.dz_half_ref * jac_v
                )
                flux = config.K_v * dtr_dz_half
                zeros_face = jnp.zeros(
                    (*tr.shape[:-1], 1), dtype=tr.dtype,
                )
                flux_full = jnp.concatenate(
                    [zeros_face, flux, zeros_face], axis=-1,
                )
                dtr_dt = dtr_dt + (
                    flux_full[..., :-1] - flux_full[..., 1:]
                ) / dz_actual_loc
        return dtr_dt

    tracer_tend = jax.vmap(tracer_tendency, in_axes=0, out_axes=0)(tracers)
    dT_dt = tracer_tend[0]
    dS_dt = tracer_tend[1]

    # --- 10. Mixing (viscosity on perturbation velocity) ---
    # Uses the proper vector Laplacian grad(div) - k×grad(curl) directly
    # on face velocities, avoiding the lossy cell-center detour.
    # See issue #105 for details.
    if config.A_h > 0:
        vlap_u, vlap_v = vector_laplacian_cgrid(
            u_prime, v_prime, grid,
            mask=mask, u_mask=u_mask, v_mask=v_mask)
        du_dt = du_dt + config.A_h * vlap_u
        dv_dt = dv_dt + config.A_h * vlap_v

    if config.A_v > 0 and u.shape[-1] >= 2:
        jac_v_u = jnp.maximum(_interp_to_u_points(J)[..., jnp.newaxis], 1e-10)
        jac_v_v = jnp.maximum(_interp_to_v_points(J)[..., jnp.newaxis], 1e-10)
        for vel, jac, is_u in [(u_prime, jac_v_u, True), (v_prime, jac_v_v, False)]:
            dv_dz_half = jnp.diff(vel, axis=-1) / (
                z_coord.dz_half_ref * jac
            )
            flux = config.A_v * dv_dz_half
            zeros_face = jnp.zeros((*vel.shape[:-1], 1), dtype=vel.dtype)
            flux_full = jnp.concatenate([zeros_face, flux, zeros_face], axis=-1)
            vdiff = (flux_full[..., :-1] - flux_full[..., 1:]) / (
                z_coord.dz_ref * jac
            )
            if is_u:
                du_dt = du_dt + vdiff
            else:
                dv_dt = dv_dt + vdiff

    # --- 10b. Physics tendencies (surface forcing, bottom drag, etc.) ---
    # The physics pipeline expects cell-center u/v shapes (shared with
    # A-grid and cubed-sphere).  Create a cell-center proxy state so
    # the physics functions produce (n_lat, n_lon, nlev) output, then
    # interpolate momentum tendencies to C-grid face points.
    if physics_fn is not None:
        u_cell = 0.5 * (u[:, :-1, :] + u[:, 1:, :])  # (n_lat, n_lon, nlev)
        v_cell = 0.5 * (v[:-1, :, :] + v[1:, :, :])   # (n_lat, n_lon, nlev)
        cc_state = state._replace(
            u=state.u.replace(data=u_cell),
            v=state.v.replace(data=v_cell),
        )
        phys = physics_fn(cc_state, grid, z_coord, surface_forcing)
        du_dt = du_dt + _interp_to_u_points(phys.du_dt.data)
        dv_dt = dv_dt + _interp_to_v_points(phys.dv_dt.data)
        dT_dt = dT_dt + phys.dT_dt.data
        dS_dt = dS_dt + phys.dS_dt.data

    # --- 11. Land masking ---
    du_dt = du_dt * u_mask_3d
    dv_dt = dv_dt * v_mask_3d
    dT_dt = dT_dt * mask_3d
    dS_dt = dS_dt * mask_3d

    # --- 12. Free-surface tendency ---
    deta_dt = -jnp.sum(flux_div_k, axis=-1) * mask

    dims_u = ("lat", "lon_u", "level")
    dims_v = ("lat_v", "lon", "level")
    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")

    return LatLonCGridOceanTendencies(
        du_dt=Field(data=du_dt, name="du_dt", dims=dims_u, units="m/s^2"),
        dv_dt=Field(data=dv_dt, name="dv_dt", dims=dims_v, units="m/s^2"),
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


