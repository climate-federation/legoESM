"""MPAS-specific ocean physics: surface forcing and bottom drag.

MPAS uses edge-normal velocity on a TRiSK C-grid, so cell-centered
wind stress (tau_x, tau_y) must be projected onto edge normals.  This
module mirrors the ``make_ocean_physics`` pipeline but returns
``MPASOceanTendencies`` compatible with the MPAS tendency function.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import MPASOceanState, MPASOceanTendencies
from legoesm.grids.voronoi import VoronoiMesh
from legoesm.ocean.eos import rho_0 as rho_0_ref
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_ocean_jacobian


def make_mpas_ocean_physics(config) -> Callable:
    """Build a combined physics function for MPAS ocean.

    Parameters
    ----------
    config : OceanPhysicsConfig

    Returns
    -------
    Callable
        ``physics_fn(state, mesh, z_coord, surface_forcing=None)``
        returning ``MPASOceanTendencies``.
    """
    from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
    from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig

    sf_config = config.surface_forcing
    bd_config = config.bottom_drag

    # Warn about unsupported physics schemes that would be silently ignored.
    import warnings
    _unsupported = []
    for attr in ("vertical_mixing", "lateral_mixing", "convection",
                 "shortwave_penetration"):
        sub = getattr(config, attr, None)
        if sub is not None and getattr(sub, "scheme", "none") != "none":
            _unsupported.append(f"{attr}={getattr(sub, 'scheme', '?')!r}")
    if _unsupported:
        warnings.warn(
            f"MPAS ocean physics: ignoring unsupported schemes: "
            + ", ".join(_unsupported),
            RuntimeWarning,
            stacklevel=2,
        )

    has_surface_forcing = (
        isinstance(sf_config, SurfaceForcingConfig)
        and sf_config.scheme != "none"
    )
    has_bottom_drag = (
        isinstance(bd_config, BottomDragConfig)
        and bd_config.scheme != "none"
    )

    def physics_fn(
        state: MPASOceanState,
        mesh: VoronoiMesh,
        z_coord: OceanZStarCoordinate,
        surface_forcing=None,
    ) -> MPASOceanTendencies:
        u_3d = state.u.data        # (nEdges, nlev)
        T_3d = state.T.data        # (nCells, nlev)
        eta = state.eta.data        # (nCells,)
        H_bathy = state.H_bathy.data
        mask = state.land_mask.data  # (nCells,)
        dtype = u_3d.dtype

        du_dt = jnp.zeros_like(u_3d)
        dT_dt = jnp.zeros_like(T_3d)
        dS_dt = jnp.zeros_like(T_3d)
        deta_dt = jnp.zeros_like(eta)

        # --- Prescribed surface forcing ---
        if has_surface_forcing and sf_config.scheme == "prescribed":
            cfg = sf_config.prescribed

            # Jacobian for top-layer thickness
            jacobian = compute_ocean_jacobian(eta, H_bathy, z_coord)
            dz_0_cell = z_coord.dz_ref[0] * jacobian  # (nCells,)

            # Compute cell-centered wind stress from latitude
            lat = mesh.grid_lat  # (nCells,) radians
            if cfg.wind_profile == "cosine_latitude":
                lat_range = jnp.pi / 2.0
                tau_x = -cfg.tau_max * jnp.cos(jnp.pi * lat / lat_range)
                tau_y = jnp.zeros_like(tau_x)
            elif cfg.wind_profile == "single_gyre":
                lat_s = cfg.lat_south_deg * jnp.pi / 180.0
                lat_n = cfg.lat_north_deg * jnp.pi / 180.0
                basin_width = lat_n - lat_s
                tau_x = -cfg.tau_max * jnp.cos(
                    jnp.pi * (lat - lat_s) / basin_width)
                tau_y = jnp.zeros_like(tau_x)
            elif cfg.wind_profile == "double_gyre":
                lat_s = cfg.lat_south_deg * jnp.pi / 180.0
                lat_n = cfg.lat_north_deg * jnp.pi / 180.0
                basin_width = lat_n - lat_s
                tau_x = -cfg.tau_max * jnp.cos(
                    2.0 * jnp.pi * (lat - lat_s) / basin_width)
                tau_y = jnp.zeros_like(tau_x)
            elif cfg.wind_profile == "double_gyre_sin2":
                buf = cfg.wind_buffer_deg * jnp.pi / 180.0
                lat_s = cfg.lat_south_deg * jnp.pi / 180.0 + buf
                lat_n = cfg.lat_north_deg * jnp.pi / 180.0 - buf
                wind_width = lat_n - lat_s
                y_norm = (lat - lat_s) / wind_width
                tau_x = cfg.tau_max * jnp.sin(jnp.pi * y_norm) ** 2
                tau_x = jnp.where(
                    (lat >= lat_s) & (lat <= lat_n), tau_x, 0.0)
                tau_y = jnp.zeros_like(tau_x)
            elif cfg.wind_profile == "double_gyre_tapered":
                # Cosine double-gyre with smooth taper to zero at walls.
                # See prescribed.py for full documentation.
                buf = cfg.wind_buffer_deg * jnp.pi / 180.0
                lat_s = cfg.lat_south_deg * jnp.pi / 180.0
                lat_n = cfg.lat_north_deg * jnp.pi / 180.0
                basin_width = lat_n - lat_s
                y_norm = (lat - lat_s) / basin_width
                tau_base = -cfg.tau_max * jnp.cos(2.0 * jnp.pi * y_norm)
                dist_south = (lat - lat_s) / buf
                dist_north = (lat_n - lat) / buf
                taper_south = jnp.where(dist_south < 1.0,
                    jnp.sin(0.5 * jnp.pi * jnp.clip(dist_south, 0, 1))**2, 1.0)
                taper_north = jnp.where(dist_north < 1.0,
                    jnp.sin(0.5 * jnp.pi * jnp.clip(dist_north, 0, 1))**2, 1.0)
                taper = taper_south * taper_north
                tau_x = tau_base * taper
                tau_x = jnp.where(
                    (lat >= lat_s) & (lat <= lat_n), tau_x, 0.0)
                tau_y = jnp.zeros_like(tau_x)
            elif cfg.wind_profile == "global_wind":
                # Nikurashin & Vallis (2012) style 3-belt wind.
                # See prescribed.py for full documentation.
                s2 = jnp.sin(lat) ** 2
                scale = cfg.tau_max / 0.1
                tau_x = scale * (
                    -0.08 - 0.0397 * s2 + 1.9487 * s2**2 - 2.0397 * s2**3
                ) * jnp.cos(lat)
                tau_y = jnp.zeros_like(tau_x)
            else:  # "constant"
                tau_x = jnp.full(mesh.nCells, cfg.tau_x, dtype=dtype)
                tau_y = jnp.full(mesh.nCells, cfg.tau_y, dtype=dtype)

            # Project cell-centered wind stress onto edge normals.
            # Average tau from the two cells sharing each edge, then dot
            # with the edge-normal direction (angleEdge).
            c1 = mesh.cellsOnEdge[0]  # (nEdges,)
            c2 = mesh.cellsOnEdge[1]  # (nEdges,)
            tau_x_e = 0.5 * (tau_x[c1] + tau_x[c2])
            tau_y_e = 0.5 * (tau_y[c1] + tau_y[c2])
            tau_n = (tau_x_e * jnp.cos(mesh.angleEdge)
                     + tau_y_e * jnp.sin(mesh.angleEdge))

            # Edge top-layer thickness
            dz_0_e = 0.5 * (dz_0_cell[c1] + dz_0_cell[c2])
            inv_rho_dz_e = 1.0 / (rho_0_ref * jnp.maximum(dz_0_e, 1e-10))

            # Apply wind stress to top layer only
            du_dt = du_dt.at[:, 0].add(tau_n * inv_rho_dz_e)

            # Heat flux: dT/dt = Q_net / (rho_0 * c_sw * dz_0)
            if cfg.Q_net != 0.0:
                from legoesm.ocean.eos import c_sw
                inv_rho_csw_dz = 1.0 / (
                    rho_0_ref * c_sw * jnp.maximum(dz_0_cell, 1e-10))
                dT_dt = dT_dt.at[:, 0].add(cfg.Q_net * inv_rho_csw_dz * mask)

            # E-P virtual salt flux
            if cfg.E_minus_P != 0.0:
                inv_dz = 1.0 / jnp.maximum(dz_0_cell, 1e-10)
                dS_dt = dS_dt.at[:, 0].add(
                    state.S.data[:, 0] * cfg.E_minus_P * inv_dz * mask)

        # --- Bottom drag ---
        if has_bottom_drag and bd_config.scheme == "linear":
            r = bd_config.linear.r
            du_dt = du_dt.at[:, -1].add(-r * u_3d[:, -1])

        return MPASOceanTendencies(
            du_dt=Field(data=du_dt, name="du_dt",
                        dims=("nEdges", "nlev"), units="m/s²"),
            dT_dt=Field(data=dT_dt, name="dT_dt",
                        dims=("nCells", "nlev"), units="degC/s"),
            dS_dt=Field(data=dS_dt, name="dS_dt",
                        dims=("nCells", "nlev"), units="PSU/s"),
            deta_dt=Field(data=deta_dt, name="deta_dt",
                          dims=("nCells",), units="m/s"),
        )

    return physics_fn
