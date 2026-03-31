"""Dedicated lat-band sharded step for FV lat-lon dynamics.

Uses ``shard_map`` with explicit ppermute north/south halo exchange
per RK stage, replacing the implicit XLA-managed collectives that
produced ~870 collective-permutes.  The 1D chain topology needs just
2 ppermute calls per RK stage (shift-north + shift-south).

Design
------
1. Latitude axis is partitioned into n_devices equal bands.
2. Each device holds ``lats_per = n_lat // n_devices`` rows.
3. Before computing tendencies, each device exchanges ``halo`` rows
   with its north and south neighbors via ``jax.lax.ppermute``.
4. Pole devices apply pole-folding BCs for their outer boundary.
5. Per-device local grids are pre-built with correct metric terms.
6. The SSP-RK3 integrator runs inside a single JIT, refreshing halos
   at each stage.
"""

from __future__ import annotations

import logging

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.atmosphere.dynamics.primitive_eq_fv_latlon import (
    fv_latlon_hydrostatic_tendencies,
)
from legoesm.grids.latlon import LatLonGrid
from legoesm.grids.polar_filter import compute_polar_filter_mask

logger = logging.getLogger(__name__)


# ============================================================================
# Local grid builder
# ============================================================================

def _build_local_grid(
    global_grid: LatLonGrid,
    dev_id: int,
    n_dev: int,
    halo: int,
) -> LatLonGrid:
    """Build a local LatLonGrid for device ``dev_id`` with halo rows.

    Returns a grid with ``lats_per + 2*halo`` latitude rows whose
    metric terms (dx, area, f) correspond to the correct latitudes.
    For pole-adjacent halos the latitude is clamped so cos(lat) > 0.
    """
    n_lat = global_grid.n_lat
    n_lon = global_grid.n_lon
    lats_per = n_lat // n_dev
    R = global_grid.radius
    dlon = global_grid.dlon
    dlat = global_grid.dlat

    lat_global = np.asarray(global_grid.lat)  # (n_lat,)
    i_start = dev_id * lats_per
    i_end = i_start + lats_per

    # Extend with halo, mirroring at poles
    lat_list = []
    for i in range(i_start - halo, i_end + halo):
        if 0 <= i < n_lat:
            lat_list.append(lat_global[i])
        elif i < 0:
            mi = min(max(-1 - i, 0), n_lat - 1)
            lat_list.append(-np.pi - lat_global[mi])
        else:
            mi = min(max(2 * n_lat - 1 - i, 0), n_lat - 1)
            lat_list.append(np.pi - lat_global[mi])

    lat_local = jnp.array(lat_list)
    n_lat_local = len(lat_list)
    lon = global_grid.lon

    lat2d = jnp.broadcast_to(lat_local[:, None], (n_lat_local, n_lon))
    lon2d = jnp.broadcast_to(lon[None, :], (n_lat_local, n_lon))
    cos_lat = jnp.maximum(jnp.abs(jnp.cos(lat_local)), 1e-10)
    sin_lat = jnp.sin(lat_local)

    omega_val = float(
        jnp.asarray(global_grid.f[0, 0])
        / (2.0 * jnp.sin(global_grid.lat[0]))
    )
    f = 2.0 * omega_val * sin_lat[:, None] * jnp.ones((1, n_lon))
    dx = R * 2.0 * dlon * cos_lat[:, None] * jnp.ones((1, n_lon))
    dy = float(R * 2.0 * dlat)
    area = R**2 * dlat * dlon * cos_lat[:, None] * jnp.ones((1, n_lon))

    return LatLonGrid(
        n_lat=n_lat_local, n_lon=n_lon, radius=R,
        lat=lat_local, lon=lon, lat2d=lat2d, lon2d=lon2d,
        cos_lat=cos_lat, sin_lat=sin_lat, f=f,
        dx=dx, dy=dy, area=area, total_area=jnp.sum(area),
        dlon=float(dlon), dlat=float(dlat),
    )


# ============================================================================
# Main entry point
# ============================================================================

def make_latlon_sharded_step(model, config):
    """Build a multi-GPU step for FV lat-lon with explicit halo exchange.

    Parameters
    ----------
    model : FVLatLonPrimitiveEquationModel
    config : DeviceConfig

    Returns
    -------
    callable : ``(state, dt) -> state``
    """
    grid = model.grid
    sigma_coord = model.sigma_coord
    cfg = model.config
    n_dev = config.n_devices
    jax_mesh = config.mesh

    n_lat = grid.n_lat
    n_lon = grid.n_lon
    lats_per = n_lat // n_dev

    if n_lat % n_dev != 0:
        raise ValueError(
            f"n_lat={n_lat} must be divisible by n_devices={n_dev}")

    halo = 2  # PPM needs halo=2
    nlev = len(sigma_coord.sigma_full)
    lats_local = lats_per + 2 * halo
    n_channels = 3 * nlev + 2  # u, v, T (each nlev), p_s, phis

    logger.info(
        "Building lat-lon sharded step: n_dev=%d, lats_per=%d, halo=%d, nlev=%d",
        n_dev, lats_per, halo, nlev,
    )

    # --- Pre-build local grids ---
    local_grids = [_build_local_grid(grid, d, n_dev, halo) for d in range(n_dev)]

    # Stack grid metric arrays for device-indexing inside shard_map
    stacked = {
        "lat": jnp.stack([g.lat for g in local_grids]),
        "dx": jnp.stack([g.dx for g in local_grids]),
        "area": jnp.stack([g.area for g in local_grids]),
        "f": jnp.stack([g.f for g in local_grids]),
        "cos_lat": jnp.stack([g.cos_lat for g in local_grids]),
        "sin_lat": jnp.stack([g.sin_lat for g in local_grids]),
    }

    rep_sharding = NamedSharding(jax_mesh, P())
    stacked = jax.tree.map(lambda x: jax.device_put(x, rep_sharding), stacked)
    sigma_coord_rep = jax.tree.map(
        lambda x: jax.device_put(x, rep_sharding) if hasattr(x, "shape") else x,
        sigma_coord,
    )

    # Polar filter masks per device
    if cfg.use_polar_filter:
        pm_list = [
            compute_polar_filter_mask(
                lg, 600.0, cfg.polar_filter_max_wave_speed,
                cfg.polar_filter_cutoff_deg)
            for lg in local_grids
        ]
        stacked_polar_mask = jnp.stack(pm_list)
        stacked_polar_mask = jax.device_put(stacked_polar_mask, rep_sharding)
    else:
        stacked_polar_mask = None

    # --- ppermute patterns (2 calls: shift-north + shift-south) ---
    # Fill south halo: d-1 sends its north boundary → d receives
    perm_fill_south = [(d, d + 1) for d in range(n_dev - 1)]
    # Fill north halo: d+1 sends its south boundary → d receives
    perm_fill_north = [(d, d - 1) for d in range(1, n_dev)]

    # Sign mask for pole-folding: negate u, v; keep T, p_s, phis
    _sign_mask = jnp.concatenate([
        -jnp.ones(nlev), -jnp.ones(nlev),  # u, v
        jnp.ones(nlev), jnp.ones(1), jnp.ones(1),  # T, p_s, phis
    ])

    # Area for mass fixer (stays sharded)
    _area_owned = grid.area
    _total_area = grid.total_area

    # Capture scalars for the shard kernel
    _n_dev = n_dev
    _halo = halo
    _lats_per = lats_per
    _lats_local = lats_local
    _n_lon = n_lon
    _nlev = nlev
    _n_channels = n_channels

    # --- Shard-map kernel ---
    def _local_tendency(u_s, v_s, T_s, ps_s, phis_s, dt_val):
        """Halo exchange + local tendency for one lat band."""
        dev_idx = jax.lax.axis_index("lat")

        # Pack all fields: (lats_per, n_lon, 3*nlev+2)
        pack = jnp.concatenate([
            u_s, v_s, T_s, ps_s[:, :, None], phis_s[:, :, None],
        ], axis=-1)

        # Send buffers: boundary rows
        send_north = pack[-_halo:]   # my north boundary
        send_south = pack[:_halo]    # my south boundary

        # Fill south halo: receive north boundary from device d-1
        recv_for_south = jax.lax.ppermute(send_north, "lat", perm_fill_south)
        # Fill north halo: receive south boundary from device d+1
        recv_for_north = jax.lax.ppermute(send_south, "lat", perm_fill_north)

        # Pole-fold for boundary devices
        half = _n_lon // 2
        # South pole fold: reflect first halo owned rows
        south_fold = jnp.roll(pack[:_halo][::-1], half, axis=1) * _sign_mask
        # North pole fold: reflect last halo owned rows
        north_fold = jnp.roll(pack[-_halo:][::-1], half, axis=1) * _sign_mask

        # Select: pole-fold for endpoints, ppermute for interior
        south_halo = jnp.where(
            (dev_idx > 0)[None, None, None],
            recv_for_south, south_fold,
        )
        north_halo = jnp.where(
            (dev_idx < _n_dev - 1)[None, None, None],
            recv_for_north, north_fold,
        )

        # Assemble padded local array
        local = jnp.concatenate([south_halo, pack, north_halo], axis=0)

        # Unpack
        u_local = local[:, :, :_nlev]
        v_local = local[:, :, _nlev:2*_nlev]
        T_local = local[:, :, 2*_nlev:3*_nlev]
        ps_local = local[:, :, 3*_nlev]
        phis_local = local[:, :, 3*_nlev + 1]

        # Build local grid
        my_grid = LatLonGrid(
            n_lat=_lats_local, n_lon=_n_lon, radius=grid.radius,
            lat=stacked["lat"][dev_idx],
            lon=grid.lon,
            lat2d=jnp.broadcast_to(
                stacked["lat"][dev_idx][:, None], (_lats_local, _n_lon)),
            lon2d=jnp.broadcast_to(grid.lon[None, :], (_lats_local, _n_lon)),
            cos_lat=stacked["cos_lat"][dev_idx],
            sin_lat=stacked["sin_lat"][dev_idx],
            f=stacked["f"][dev_idx],
            dx=stacked["dx"][dev_idx],
            dy=grid.dy, area=stacked["area"][dev_idx],
            total_area=_total_area, dlon=grid.dlon, dlat=grid.dlat,
        )

        my_polar_mask = (
            stacked_polar_mask[dev_idx] if stacked_polar_mask is not None
            else None
        )

        dims_3d = ("lat", "lon", "level")
        dims_2d = ("lat", "lon")
        local_state = HydrostaticState(
            u=Field(data=u_local, name="u", dims=dims_3d, units="m/s"),
            v=Field(data=v_local, name="v", dims=dims_3d, units="m/s"),
            T=Field(data=T_local, name="T", dims=dims_3d, units="K"),
            p_s=Field(data=ps_local, name="p_s", dims=dims_2d, units="Pa"),
            phis=Field(data=phis_local, name="phis", dims=dims_2d,
                       units="m^2/s^2"),
        )

        tend = fv_latlon_hydrostatic_tendencies(
            local_state, my_grid, sigma_coord_rep, cfg,
            physics_tendency=None, polar_mask=my_polar_mask,
        )

        # Return only owned portion
        h = _halo
        return (
            tend.du_dt.data[h:h + _lats_per],
            tend.dv_dt.data[h:h + _lats_per],
            tend.dT_dt.data[h:h + _lats_per],
            tend.dp_s_dt.data[h:h + _lats_per],
        )

    _shard_tendency = shard_map(
        _local_tendency,
        mesh=jax_mesh,
        in_specs=(
            P("lat", None, None),  # u
            P("lat", None, None),  # v
            P("lat", None, None),  # T
            P("lat", None),        # p_s
            P("lat", None),        # phis
            P(),                   # dt
        ),
        out_specs=(
            P("lat", None, None),
            P("lat", None, None),
            P("lat", None, None),
            P("lat", None),
        ),
        check_rep=False,
    )

    # --- SSP-RK3 step ---
    @jax.jit
    def _step(state, dt):
        u = state.u.data
        v = state.v.data
        T = state.T.data
        ps = state.p_s.data
        phis = state.phis.data

        # Stage 1
        du1, dv1, dT1, dps1 = _shard_tendency(u, v, T, ps, phis, dt)
        u1 = u + dt * du1
        v1 = v + dt * dv1
        T1 = T + dt * dT1
        ps1 = ps + dt * dps1

        # Stage 2
        du2, dv2, dT2, dps2 = _shard_tendency(u1, v1, T1, ps1, phis, dt)
        u2 = 0.75 * u + 0.25 * (u1 + dt * du2)
        v2 = 0.75 * v + 0.25 * (v1 + dt * dv2)
        T2 = 0.75 * T + 0.25 * (T1 + dt * dT2)
        ps2 = 0.75 * ps + 0.25 * (ps1 + dt * dps2)

        # Stage 3
        du3, dv3, dT3, dps3 = _shard_tendency(u2, v2, T2, ps2, phis, dt)
        u_new = (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * du3)
        v_new = (1.0 / 3.0) * v + (2.0 / 3.0) * (v2 + dt * dv3)
        T_new = (1.0 / 3.0) * T + (2.0 / 3.0) * (T2 + dt * dT3)
        ps_new = (1.0 / 3.0) * ps + (2.0 / 3.0) * (ps2 + dt * dps3)

        # Floors
        if cfg.T_min > 0:
            T_new = jnp.maximum(T_new, cfg.T_min)
        if cfg.p_floor > 0:
            ps_new = jnp.maximum(ps_new, cfg.p_floor)

        # Mass conservation
        if cfg.use_conservation_fixer and cfg.fix_mass:
            mass_old = jnp.sum(ps * _area_owned)
            mass_new = jnp.sum(ps_new * _area_owned)
            ps_new = ps_new + (mass_old - mass_new) / _total_area

        dims_3d = ("lat", "lon", "level")
        dims_2d = ("lat", "lon")
        return HydrostaticState(
            u=Field(data=u_new, name="u", dims=dims_3d, units="m/s"),
            v=Field(data=v_new, name="v", dims=dims_3d, units="m/s"),
            T=Field(data=T_new, name="T", dims=dims_3d, units="K"),
            p_s=Field(data=ps_new, name="p_s", dims=dims_2d, units="Pa"),
            phis=state.phis,
        )

    logger.info(
        "Lat-lon sharded step ready: %d devices, %d lats/device, "
        "halo=%d, 2 ppermute calls/stage",
        n_dev, lats_per, halo,
    )
    return _step
