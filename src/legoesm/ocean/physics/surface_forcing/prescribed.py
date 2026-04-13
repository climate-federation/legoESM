"""Prescribed surface forcing: fixed wind stress and heat/freshwater fluxes."""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.eos import rho_0 as rho_0_ref, c_sw
from legoesm.ocean.physics.surface_forcing.config import PrescribedForcingConfig
from legoesm.ocean.physics.surface_forcing.output import SurfaceForcingOutput
from legoesm.ocean.vertical import OceanZStarCoordinate


def prescribed_surface_forcing(
    u: jnp.ndarray,
    v: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    grid: CubedSphereGrid,
    cfg: PrescribedForcingConfig,
) -> SurfaceForcingOutput:
    """Apply prescribed surface forcing to the top ocean layer.

    Parameters
    ----------
    u, v : array (6, n, n, nlev)
    T, S : array (6, n, n, nlev)
    z_coord : OceanZStarCoordinate
    jacobian : array (6, n, n)
    grid : CubedSphereGrid
    cfg : PrescribedForcingConfig

    Returns
    -------
    SurfaceForcingOutput
    """
    nlev = u.shape[-1]
    shape_3d = u.shape
    dtype = u.dtype

    # Top layer thickness
    dz_0 = z_coord.dz_ref[0] * jacobian  # (6, n, n)
    inv_rho_dz = 1.0 / (rho_0_ref * jnp.maximum(dz_0, 1e-10))

    # Wind stress
    # Use grid_lat (GridProtocol property) for correct shape on all grid types:
    # cubed_sphere (6, n, n), latlon (n_lat, n_lon).
    if cfg.wind_profile == "cosine_latitude":
        lat = grid.grid_lat
        lat_range = jnp.pi / 2.0  # 90 degrees
        tau_x = -cfg.tau_max * jnp.cos(jnp.pi * lat / lat_range)
        tau_y = jnp.zeros_like(tau_x)
    elif cfg.wind_profile == "single_gyre":
        lat = grid.grid_lat
        # Basin-relative single-gyre wind stress (Stommel 1948, Munk 1950).
        # tau_x = -tau_max * cos(pi * (lat - lat_s) / (lat_n - lat_s))
        # Easterlies at southern boundary, westerlies at northern boundary.
        # One sign of curl → one anticyclonic (subtropical) gyre.
        lat_s = cfg.lat_south_deg * jnp.pi / 180.0
        lat_n = cfg.lat_north_deg * jnp.pi / 180.0
        basin_width = lat_n - lat_s
        tau_x = -cfg.tau_max * jnp.cos(jnp.pi * (lat - lat_s) / basin_width)
        tau_y = jnp.zeros_like(tau_x)
    elif cfg.wind_profile == "double_gyre":
        lat = grid.grid_lat
        # Basin-relative double-gyre wind stress (Holland & Lin 1975).
        # tau_x = -tau_max * cos(2*pi * (lat - lat_s) / (lat_n - lat_s))
        # Easterlies at both boundaries, westerly jet at mid-basin.
        # Curl changes sign at mid-basin → subtropical gyre (south)
        # + subpolar gyre (north).
        lat_s = cfg.lat_south_deg * jnp.pi / 180.0
        lat_n = cfg.lat_north_deg * jnp.pi / 180.0
        basin_width = lat_n - lat_s
        tau_x = -cfg.tau_max * jnp.cos(2.0 * jnp.pi * (lat - lat_s) / basin_width)
        tau_y = jnp.zeros_like(tau_x)
    elif cfg.wind_profile == "double_gyre_sin2":
        lat = grid.grid_lat
        # Mid-latitude westerly jet with sin^2 profile.
        # tau_x = +tau_max * sin^2(pi * (lat - lat_s') / (lat_n' - lat_s'))
        # where lat_s', lat_n' are inset by wind_buffer_deg from the basin
        # walls to ensure zero wind stress and zero Ekman transport at
        # the boundaries.
        #
        # Properties:
        #   - tau_x >= 0 everywhere (eastward, representing westerly jet)
        #   - tau_x = 0 at lat_s' and lat_n' (and in buffer zones)
        #   - Peak westerly at mid-basin
        #   - Curl changes sign at mid-basin → double gyre
        #   - Positive curl (Ekman suction) in southern half → subtropical gyre
        #   - Negative curl (Ekman pumping) in northern half → subpolar gyre
        buf = cfg.wind_buffer_deg * jnp.pi / 180.0
        lat_s = cfg.lat_south_deg * jnp.pi / 180.0 + buf
        lat_n = cfg.lat_north_deg * jnp.pi / 180.0 - buf
        wind_width = lat_n - lat_s
        y_norm = (lat - lat_s) / wind_width  # 0 at lat_s', 1 at lat_n'
        tau_x = cfg.tau_max * jnp.sin(jnp.pi * y_norm) ** 2
        # Zero outside the wind region
        tau_x = jnp.where((lat >= lat_s) & (lat <= lat_n), tau_x, 0.0)
        tau_y = jnp.zeros_like(tau_x)
    elif cfg.wind_profile == "double_gyre_tapered":
        lat = grid.grid_lat
        # Cosine double-gyre wind with smooth taper to zero at walls.
        # Base profile: -tau_max * cos(2*pi*(lat-lat_s)/(lat_n-lat_s))
        # Taper: sin^2(pi/2 * distance_from_wall / buffer) in the buffer zone.
        #
        # Properties:
        #   - tau_x = 0 at lat_s and lat_n (smooth zero at walls)
        #   - Easterlies near walls, westerly jet at mid-basin (like cosine)
        #   - Basin-integrated wind ≈ 0 (small O(buffer/basin)^2 bias)
        #   - Curl changes sign at mid-basin → double gyre
        #   - No spurious Ekman transport at coastal walls
        buf = cfg.wind_buffer_deg * jnp.pi / 180.0
        lat_s = cfg.lat_south_deg * jnp.pi / 180.0
        lat_n = cfg.lat_north_deg * jnp.pi / 180.0
        basin_width = lat_n - lat_s
        y_norm = (lat - lat_s) / basin_width  # 0 at south wall, 1 at north wall
        # Cosine base: easterlies at walls, westerlies at center
        tau_base = -cfg.tau_max * jnp.cos(2.0 * jnp.pi * y_norm)
        # Smooth taper: sin^2 ramp in buffer zones
        dist_south = (lat - lat_s) / buf  # 0 at wall, 1 at buffer edge
        dist_north = (lat_n - lat) / buf
        taper_south = jnp.where(dist_south < 1.0,
                                jnp.sin(0.5 * jnp.pi * jnp.clip(dist_south, 0, 1))**2,
                                1.0)
        taper_north = jnp.where(dist_north < 1.0,
                                jnp.sin(0.5 * jnp.pi * jnp.clip(dist_north, 0, 1))**2,
                                1.0)
        taper = taper_south * taper_north
        tau_x = tau_base * taper
        # Zero outside basin
        tau_x = jnp.where((lat >= lat_s) & (lat <= lat_n), tau_x, 0.0)
        tau_y = jnp.zeros_like(tau_x)
    elif cfg.wind_profile == "global_wind":
        lat = grid.grid_lat
        # Realistic 3-belt zonal wind stress following
        # Nikurashin & Vallis (2012, JPO) style profile.
        # Polynomial in sin^2(phi) with cos(phi) envelope:
        #   tau_x = tau_max * (a + b*s^2 + c*s^4 + d*s^6) * cos(phi)
        # where s = sin(phi). Coefficients tuned so that:
        #   phi=0:  tau_x = -0.08 Pa  (easterly trades)
        #   phi=30: tau_x = 0         (zero crossing)
        #   phi=50: tau_x = +0.10 Pa  (westerly peak)
        #   phi=70: tau_x = 0         (returns to zero)
        # Scaled by tau_max/0.1 so the default tau_max=0.1 gives
        # the reference amplitudes above.
        s2 = jnp.sin(lat) ** 2
        scale = cfg.tau_max / 0.1
        tau_x = scale * (
            -0.08 - 0.0397 * s2 + 1.9487 * s2**2 - 2.0397 * s2**3
        ) * jnp.cos(lat)
        tau_y = jnp.zeros_like(tau_x)
    else:
        tau_x = jnp.full_like(dz_0, cfg.tau_x, dtype=dtype)
        tau_y = jnp.full_like(dz_0, cfg.tau_y, dtype=dtype)

    du_dt = jnp.zeros(shape_3d, dtype=dtype)
    dv_dt = jnp.zeros(shape_3d, dtype=dtype)
    du_dt = du_dt.at[..., 0].set(tau_x * inv_rho_dz)
    dv_dt = dv_dt.at[..., 0].set(tau_y * inv_rho_dz)

    # Heat flux: dT/dt = Q_net / (rho_0 * c_sw * dz_0)
    Q_net = jnp.full_like(dz_0, cfg.Q_net, dtype=dtype)
    inv_rho_csw_dz = 1.0 / (rho_0_ref * c_sw * jnp.maximum(dz_0, 1e-10))
    dT_dt = jnp.zeros(shape_3d, dtype=dtype)
    dT_dt = dT_dt.at[..., 0].set(Q_net * inv_rho_csw_dz)

    # Freshwater (virtual salt flux): dS/dt = +S * E_minus_P / dz_0
    # Positive E-P means net evaporation → water leaves → salt concentrates → dS/dt > 0
    dS_dt = jnp.zeros(shape_3d, dtype=dtype)
    if cfg.E_minus_P != 0.0:
        inv_dz = 1.0 / jnp.maximum(dz_0, 1e-10)
        dS_dt = dS_dt.at[..., 0].set(S[..., 0] * cfg.E_minus_P * inv_dz)

    return SurfaceForcingOutput(
        du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, dS_dt=dS_dt,
        Q_net=Q_net, tau_x=tau_x, tau_y=tau_y,
    )
