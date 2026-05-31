"""Surface-flux applicator for the OMIP-2 / Bryan-THC drivers.

Wraps the Large & Yeager 2009 bulk-flux module
(``legoesm.ocean.bulk_flux_omip``) and applies the resulting
``(tau_x, tau_y, shflx, lhflx)`` to the ocean state's top layer as a
per-timestep forward-Euler update of u / v / T. Designed to be called
once per model step from the long-run drivers.

State conventions handled
-------------------------

* Lat-lon C-grid (``LatLonCGridOceanState``): ``u`` on east-west
  faces (shape ``(n_lat, n_lon+1, n_z)``), ``v`` on north-south
  faces (shape ``(n_lat+1, n_lon, n_z)``). The applicator
  interpolates cell-centred tau_x / tau_y to the matching faces
  via the C-grid 0.5*(left+right) stencil already used by the
  prescribed-forcing path.
* MPAS Voronoi: ``u`` on edges; ``tau_normal`` = tau_x*cos(angleEdge)
  + tau_y*sin(angleEdge).
* Cube C-D grid: ``u``, ``v`` collocated on cells; direct
  application.

All paths preserve land masking through ``state.land_mask`` /
``state.u_mask`` / ``state.v_mask``.

Sign conventions
----------------
``shflx`` and ``lhflx`` are defined as positive INTO the ocean
(matching ``bulk_flux_omip.air_sea_fluxes``). The top-cell
temperature tendency is

.. math::
    \\frac{dT}{dt}\\bigg|_{\\rm top} = \\frac{Q_{\\rm net}}{\\rho_0 c_p \\Delta z_0}

with ``Q_net = shflx + lhflx + sw_down - lw_up`` (sw / lw passed
in directly; the smoke path adds only the turbulent components).
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.core.field import Field
from legoesm.thermo import saturation_vapor_pressure
from legoesm.ocean.bulk_flux_omip import air_sea_fluxes


def _bolton_q_sat(T_K, p_hpa: float = 1013.25):
    """Saturation specific humidity [kg/kg] at temperature T_K [K], pressure p_hpa [hPa].

    Uses the canonical saturation vapour pressure from :mod:`legoesm.thermo`
    (no re-derived Magnus/Tetens coefficients) and ``constants.epsilon`` for
    the specific-humidity conversion ``q = eps e / (p - (1 - eps) e)``.
    """
    e_s = saturation_vapor_pressure(T_K) / 100.0  # Pa -> hPa
    return constants.epsilon * e_s / (p_hpa - (1.0 - constants.epsilon) * e_s)


# Cache of precomputed nearest-neighbour (lat_idx, lon_idx) maps, keyed by a
# (src, dst) signature. The target points (grid cell centres) are fixed within
# a run, so the O(src*dst) index search runs once; per-step calls are a single
# fancy-index.
_NN_INDEX_CACHE: dict = {}


def _nn_interp_to_points(field, src_lat_deg, src_lon_deg,
                         dst_lat_deg_pts, dst_lon_deg_pts):
    """True nearest-neighbour lookup from a regular 1-D ``(src_lat, src_lon)``
    grid to a 1-D set of target points ``(dst_lat_pts, dst_lon_pts)``.

    Latitude: nearest by ``|Δlat|``. Longitude: nearest by *periodic*
    (wrap-around) distance, so a target near the 0/360 seam picks the truly
    closest source column. Source axes need not be pre-sorted. Used for cube,
    MPAS, and tripole targets (unstructured / 2-D cell centres); the regular
    lat-lon path uses ``conservative_regrid`` instead.
    """
    src_lat = np.asarray(src_lat_deg, dtype=np.float64)
    src_lon = np.asarray(src_lon_deg, dtype=np.float64) % 360.0
    dst_lat = np.asarray(dst_lat_deg_pts, dtype=np.float64)
    dst_lon = np.asarray(dst_lon_deg_pts, dtype=np.float64) % 360.0
    # Robust signature of BOTH source and destination coords (codex: the dst
    # longitude checksum was missing, so two same-shape targets differing only
    # in interior lon could collide and reuse the wrong index map).
    key = (
        src_lat.size, src_lon.size, dst_lat.size,
        float(src_lat[0]), float(src_lat[-1]), float(src_lat.sum()),
        float(src_lon[0]), float(src_lon[-1]), float(src_lon.sum()),
        float(dst_lat[0]), float(dst_lat[-1]), float(dst_lat.sum()),
        float(dst_lon[0]), float(dst_lon[-1]), float(dst_lon.sum()),
    )
    idx = _NN_INDEX_CACHE.get(key)
    if idx is None:
        # nearest source latitude (absolute difference)
        i = np.argmin(np.abs(src_lat[:, None] - dst_lat[None, :]), axis=0)
        # nearest source longitude (periodic distance on the circle)
        dlon = np.abs(src_lon[:, None] - dst_lon[None, :])
        dlon = np.minimum(dlon, 360.0 - dlon)
        j = np.argmin(dlon, axis=0)
        idx = (i, j)
        _NN_INDEX_CACHE[key] = idx
    i, j = idx
    return np.asarray(field)[i, j]


# Cache of pre-computed conservative-regrid weights, keyed by
# (src_lat_shape, src_lon_shape, src_lat_first, src_lon_first,
#  dst_lat_shape, dst_lon_shape, dst_lat_first, dst_lon_first).
_REGRID_WEIGHTS_CACHE: dict = {}


def _edges_from_centers_deg(centers_deg, *, periodic: bool = False):
    """Derive uniform-spaced cell edges (radians) from cell centres.

    Assumes the centres are uniformly spaced. ``periodic`` only changes
    the conventional first / last edge offsets.
    """
    c = np.asarray(centers_deg, dtype=np.float64)
    if c.size < 2:
        raise ValueError("Need >= 2 cell centres to infer edges.")
    dc = float(c[1] - c[0])
    edges = np.empty(c.size + 1, dtype=np.float64)
    edges[:-1] = c - 0.5 * dc
    edges[-1] = c[-1] + 0.5 * dc
    return np.radians(edges)


def _conservative_regrid_to_latlon(
    field_2d, src_lat_deg, src_lon_deg, dst_lat_deg, dst_lon_deg,
):
    """Conservatively regrid a 2-D ``(n_src_lat, n_src_lon)`` field to a
    regular lat-lon model grid.

    Builds + caches the overlap weights on first call for each
    (src_shape, dst_shape) pair; subsequent calls are a sparse matmul.
    """
    from legoesm.grids.conservative_regrid import (
        compute_overlap_weights, apply_conservative_regrid,
    )
    import jax.numpy as jnp_local
    key = (
        np.asarray(src_lat_deg).shape,
        np.asarray(src_lon_deg).shape,
        float(src_lat_deg[0]), float(src_lon_deg[0]),
        np.asarray(dst_lat_deg).shape,
        np.asarray(dst_lon_deg).shape,
        float(dst_lat_deg[0]), float(dst_lon_deg[0]),
    )
    if key not in _REGRID_WEIGHTS_CACHE:
        src_lat_edges = _edges_from_centers_deg(src_lat_deg)
        src_lon_edges = _edges_from_centers_deg(src_lon_deg, periodic=True)
        dst_lat_edges = _edges_from_centers_deg(dst_lat_deg)
        dst_lon_edges = _edges_from_centers_deg(dst_lon_deg, periodic=True)
        # Clamp lat edges into [-pi/2, pi/2] in case the inferred edge
        # spills over the pole due to rounding.
        src_lat_edges = np.clip(src_lat_edges, -np.pi / 2, np.pi / 2)
        dst_lat_edges = np.clip(dst_lat_edges, -np.pi / 2, np.pi / 2)
        _REGRID_WEIGHTS_CACHE[key] = compute_overlap_weights(
            src_lat_edges, src_lon_edges,
            dst_lat_edges, dst_lon_edges,
        )
    weights = _REGRID_WEIGHTS_CACHE[key]
    return np.asarray(apply_conservative_regrid(
        jnp_local.asarray(field_2d), weights,
    ))


def _sample_forcing_latlon(forcing, idx_t, dst_lat_deg, dst_lon_deg):
    """Conservative-regrid the seven channels at time ``idx_t`` onto a
    regular destination lat-lon grid."""
    out = {}
    for name in ("u10", "v10", "T_air", "q_air",
                  "sw_down", "lw_down", "precip"):
        out[name] = _conservative_regrid_to_latlon(
            getattr(forcing, name)[idx_t],
            forcing.lat, forcing.lon,
            dst_lat_deg, dst_lon_deg,
        )
    return out


def _sample_forcing_points(forcing, idx_t, lat_pts_deg, lon_pts_deg):
    """Nearest-neighbour sample the seven channels at a set of points
    (cube / MPAS cell centres)."""
    out = {}
    for name in ("u10", "v10", "T_air", "q_air",
                  "sw_down", "lw_down", "precip"):
        out[name] = _nn_interp_to_points(
            getattr(forcing, name)[idx_t],
            forcing.lat, forcing.lon,
            lat_pts_deg, lon_pts_deg,
        )
    return out


def _apply_cgrid_surface_fluxes(state, forc, *, dz_0, rho_0, c_p,
                                rho_air, sigma_sb, dt, grid=None):
    """Apply bulk fluxes to a lat-lon C-grid state from already-sampled forcing.

    Shared by the ``latlon`` / ``latlon_regional`` path (forcing conservatively
    regridded onto the regular T grid) and the curvilinear ``tripole`` path
    (forcing nearest-neighbour sampled onto the 2-D T grid) -- both use the
    identical ``LatLonCGridOceanState`` layout (T cell-centred; u on EW faces
    ``(n_lat, n_lon+1)``; v on NS faces ``(n_lat+1, n_lon)``). ``forc`` holds
    the seven channels as 2-D ``(n_lat, n_lon)`` arrays. Forward-Euler update of
    the top layer of T, u, v; land masking via state masks.

    ``grid`` (optional) supplies ``cos_alpha_u`` / ``sin_alpha_u`` /
    ``cos_alpha_v`` / ``sin_alpha_v`` for the geographic->grid wind-stress
    rotation on curvilinear (tripole) geometries; absent on a regular lat-lon
    grid, where the rotation is the identity.

    NOTE (inherited tech debt, pre-existing in the latlon branch this factors):
    ``_bolton_q_sat`` re-implements saturation and ``273.15`` / ``0.97`` are
    hardcoded -- should move to ``legoesm.thermo`` / ``constants`` / a config in
    a dedicated cleanup; kept verbatim here to preserve the heat-flux results.
    The wind-stress *sign* (ocean reaction = -tau), by contrast, is corrected
    here vs the previous (+tau) latlon code -- see the momentum comment below.
    """
    T_sfc_K = np.asarray(state.T.data, dtype=np.float64)[..., 0] + 273.15
    q_sfc = np.asarray(_bolton_q_sat(jnp.asarray(T_sfc_K)), dtype=np.float64)
    tau_x, tau_y, sh, lh = air_sea_fluxes(
        u10=jnp.asarray(forc["u10"]),
        v10=jnp.asarray(forc["v10"]),
        T_air_K=jnp.asarray(forc["T_air"]),
        q_air=jnp.asarray(forc["q_air"]),
        T_sfc_K=jnp.asarray(T_sfc_K),
        q_sfc=jnp.asarray(q_sfc),
        rho_air=jnp.asarray(rho_air),
    )
    # --- Heat: Q_net positive into the ocean warms the top cell. ---
    lw_up = 0.97 * sigma_sb * T_sfc_K ** 4
    Q_net = (np.asarray(sh) + np.asarray(lh)
             + forc["sw_down"] - lw_up + forc["lw_down"])
    mask = np.asarray(state.land_mask.data, dtype=np.float64)
    T_new = np.asarray(state.T.data, dtype=np.float64).copy()
    T_new[..., 0] = T_new[..., 0] + Q_net / (rho_0 * c_p * dz_0) * dt * mask

    # --- Momentum: OCEAN REACTION force. ``air_sea_fluxes`` returns tau in the
    # ATMOSPHERIC convention (tau = -rho_air Cd |U| U, i.e. opposing the wind),
    # so the ocean feels -tau (Newton's 3rd law). This sign flip mirrors the
    # dynamics-core external-tau block (ocean_pe_latlon_cgrid.py:1892); without
    # it the wind drives the ocean BACKWARDS. Geographic east/north stress is
    # then rotated to grid-aligned (i, j) via cos_alpha_u / sin_alpha_u when the
    # grid is curvilinear (tripole); a regular lat-lon C-grid lacks those
    # attributes, so the rotation is the identity (grid-i == east). ---
    tau_e = -np.asarray(tau_x, dtype=np.float64)   # eastward stress ON ocean
    tau_n = -np.asarray(tau_y, dtype=np.float64)   # northward stress ON ocean
    n_lat, n_lon = tau_e.shape

    def _cell_to_uface(a):
        f = np.zeros((n_lat, n_lon + 1), dtype=np.float64)
        f[:, 1:-1] = 0.5 * (a[:, :-1] + a[:, 1:])
        f[:, 0] = a[:, 0]
        f[:, -1] = a[:, -1]
        return f

    def _cell_to_vface(a):
        f = np.zeros((n_lat + 1, n_lon), dtype=np.float64)
        f[1:-1, :] = 0.5 * (a[:-1, :] + a[1:, :])
        f[0, :] = a[0, :]
        f[-1, :] = a[-1, :]
        return f

    tau_e_u, tau_n_u = _cell_to_uface(tau_e), _cell_to_uface(tau_n)
    tau_e_v, tau_n_v = _cell_to_vface(tau_e), _cell_to_vface(tau_n)

    ca_u = getattr(grid, "cos_alpha_u", None) if grid is not None else None
    sa_u = getattr(grid, "sin_alpha_u", None) if grid is not None else None
    ca_v = getattr(grid, "cos_alpha_v", None) if grid is not None else None
    sa_v = getattr(grid, "sin_alpha_v", None) if grid is not None else None
    if ca_u is not None and sa_u is not None:
        tau_i_u = (tau_e_u * np.asarray(ca_u, dtype=np.float64)
                   + tau_n_u * np.asarray(sa_u, dtype=np.float64))
    else:
        tau_i_u = tau_e_u
    if ca_v is not None and sa_v is not None:
        tau_j_v = (-tau_e_v * np.asarray(sa_v, dtype=np.float64)
                   + tau_n_v * np.asarray(ca_v, dtype=np.float64))
    else:
        tau_j_v = tau_n_v

    u_face = np.asarray(state.u.data, dtype=np.float64).copy()
    v_face = np.asarray(state.v.data, dtype=np.float64).copy()
    u_mask = np.asarray(state.u_mask.data, dtype=np.float64)
    v_mask = np.asarray(state.v_mask.data, dtype=np.float64)
    u_face[..., 0] = u_face[..., 0] + tau_i_u / (rho_0 * dz_0) * dt * u_mask
    v_face[..., 0] = v_face[..., 0] + tau_j_v / (rho_0 * dz_0) * dt * v_mask

    return state._replace(
        T=Field(jnp.asarray(T_new), name=state.T.name,
                dims=state.T.dims, units=state.T.units),
        u=Field(jnp.asarray(u_face), name=state.u.name,
                dims=state.u.dims, units=state.u.units),
        v=Field(jnp.asarray(v_face), name=state.v.name,
                dims=state.v.dims, units=state.v.units),
    )


def apply_omip2_surface_fluxes(state, *, forcing, idx_t: int,
                                 z_coord, grid, grid_type: str,
                                 dt: float,
                                 rho_0: Optional[float] = None,
                                 c_p: Optional[float] = None,
                                 rho_air: float = 1.225):
    """Apply one timestep of JRA55-do / CORE-II forcing to ``state``.

    Supported ``grid_type``: ``latlon`` / ``latlon_regional`` (conservative
    regrid onto the regular T grid), ``cubed_sphere`` and ``mpas`` /
    ``mpas_regional`` (nearest-neighbour onto cell centres), and ``tripole``
    (curvilinear lat-lon C-grid; nearest-neighbour onto the 2-D ``lat_T``/
    ``lon_T`` grid, sharing the latlon C-grid application via
    ``_apply_cgrid_surface_fluxes``). The ``spectral`` grid is NOT supported
    (its state carries no grid-space u/v faces -- forcing it needs a
    spectral-space path; tracked as a follow-up in OMIP_faithful.md).

    Returns a new ``state`` with updated top-layer u, v, T fields.
    """
    if rho_0 is None:
        rho_0 = float(constants.rho_ocean)
    if c_p is None:
        c_p = float(constants.c_sw)

    sigma_sb = float(constants.sigma_sb)
    dz_0 = float(np.asarray(z_coord.dz_ref)[0])

    if grid_type in ("latlon", "latlon_regional"):
        lat_deg = np.degrees(np.asarray(grid.lat))
        lon_deg = np.degrees(np.asarray(grid.lon))
        forc = _sample_forcing_latlon(forcing, idx_t, lat_deg, lon_deg)
        return _apply_cgrid_surface_fluxes(
            state, forc, dz_0=dz_0, rho_0=rho_0, c_p=c_p,
            rho_air=rho_air, sigma_sb=sigma_sb, dt=dt, grid=grid,
        )

    if grid_type == "tripole":
        # Curvilinear tripolar grid: same C-grid state layout as latlon
        # (T cell-centred; u/v on EW/NS faces) but the T-point coordinates are
        # 2-D (lat_T, lon_T). Conservative regrid needs a regular destination
        # grid, so sample CORE-II at each T cell by nearest-neighbour (as the
        # cube / MPAS paths do). NOTE: tau_x/tau_y are geographic (eastward /
        # northward); south of the ~50 deg N tripole join the grid is regular
        # lat-lon (grid-i == east exactly), so only in the largely ice-masked
        # Arctic fold is the unrotated stress application an approximation
        # (tracked in OMIP_faithful.md).
        lat_pts = np.degrees(np.asarray(grid.lat_T))   # (n_lat, n_lon)
        lon_pts = np.degrees(np.asarray(grid.lon_T))
        shp = lat_pts.shape
        forc = _sample_forcing_points(
            forcing, idx_t, lat_pts.reshape(-1), lon_pts.reshape(-1),
        )
        for _k, _v in forc.items():
            forc[_k] = _v.reshape(shp)
        return _apply_cgrid_surface_fluxes(
            state, forc, dz_0=dz_0, rho_0=rho_0, c_p=c_p,
            rho_air=rho_air, sigma_sb=sigma_sb, dt=dt, grid=grid,
        )

    if grid_type == "cubed_sphere":
        # Cube C-D grid stores u, v collocated on cell centres -- no
        # face interpolation needed. Sample forcing at the (lat, lon)
        # of each cell centre via nearest-neighbour.
        lat_pts = np.degrees(np.asarray(grid.lat))   # (6, n, n)
        lon_pts = np.degrees(np.asarray(grid.lon))
        flat_shape = lat_pts.shape
        forc = _sample_forcing_points(
            forcing, idx_t, lat_pts.reshape(-1), lon_pts.reshape(-1),
        )
        for k, v in forc.items():
            forc[k] = v.reshape(flat_shape)
        T_sfc_K = np.asarray(state.T.data, dtype=np.float64)[..., 0] + constants.T_freeze
        q_sfc = np.asarray(_bolton_q_sat(jnp.asarray(T_sfc_K)),
                           dtype=np.float64)
        tau_x, tau_y, sh, lh = air_sea_fluxes(
            u10=jnp.asarray(forc["u10"]),
            v10=jnp.asarray(forc["v10"]),
            T_air_K=jnp.asarray(forc["T_air"]),
            q_air=jnp.asarray(forc["q_air"]),
            T_sfc_K=jnp.asarray(T_sfc_K),
            q_sfc=jnp.asarray(q_sfc),
            rho_air=jnp.asarray(rho_air),
        )
        lw_up = 0.97 * sigma_sb * T_sfc_K ** 4
        Q_net = (np.asarray(sh) + np.asarray(lh)
                 + forc["sw_down"] - lw_up + forc["lw_down"])
        mask = np.asarray(state.land_mask.data, dtype=np.float64)
        T_new = np.asarray(state.T.data, dtype=np.float64).copy()
        T_new[..., 0] = T_new[..., 0] + (
            Q_net / (rho_0 * c_p * dz_0) * dt * mask
        )
        u_new = np.asarray(state.u.data, dtype=np.float64).copy()
        v_new = np.asarray(state.v.data, dtype=np.float64).copy()
        # Ocean reaction force: air_sea_fluxes returns atmospheric-convention
        # tau (opposing the wind), so the ocean feels -tau (Newton's 3rd law),
        # consistent with the lat-lon/tripole helper above. NOTE: applied
        # without geographic->cube-frame rotation (pre-existing limitation;
        # tracked in OMIP_faithful.md).
        u_new[..., 0] = u_new[..., 0] + (
            -np.asarray(tau_x) / (rho_0 * dz_0) * dt * mask
        )
        v_new[..., 0] = v_new[..., 0] + (
            -np.asarray(tau_y) / (rho_0 * dz_0) * dt * mask
        )
        return state._replace(
            T=Field(jnp.asarray(T_new), name=state.T.name,
                    dims=state.T.dims, units=state.T.units),
            u=Field(jnp.asarray(u_new), name=state.u.name,
                    dims=state.u.dims, units=state.u.units),
            v=Field(jnp.asarray(v_new), name=state.v.name,
                    dims=state.v.dims, units=state.v.units),
        )

    if grid_type in ("mpas", "mpas_regional"):
        # MPAS stores u on edge-normals; cell-centred tau is projected
        # via ``tau_normal = tau_x * cos(angleEdge) + tau_y *
        # sin(angleEdge)`` then averaged from neighbouring cells onto
        # the shared edge.
        lat_pts = np.degrees(np.asarray(grid.latCell))
        lon_pts = np.degrees(np.asarray(grid.lonCell))
        forc = _sample_forcing_points(
            forcing, idx_t, lat_pts, lon_pts,
        )
        T_sfc_K = np.asarray(state.T.data, dtype=np.float64)[:, 0] + constants.T_freeze
        q_sfc = np.asarray(_bolton_q_sat(jnp.asarray(T_sfc_K)),
                           dtype=np.float64)
        tau_x, tau_y, sh, lh = air_sea_fluxes(
            u10=jnp.asarray(forc["u10"]),
            v10=jnp.asarray(forc["v10"]),
            T_air_K=jnp.asarray(forc["T_air"]),
            q_air=jnp.asarray(forc["q_air"]),
            T_sfc_K=jnp.asarray(T_sfc_K),
            q_sfc=jnp.asarray(q_sfc),
            rho_air=jnp.asarray(rho_air),
        )
        lw_up = 0.97 * sigma_sb * T_sfc_K ** 4
        Q_net = (np.asarray(sh) + np.asarray(lh)
                 + forc["sw_down"] - lw_up + forc["lw_down"])
        mask = np.asarray(state.land_mask.data, dtype=np.float64)
        T_new = np.asarray(state.T.data, dtype=np.float64).copy()
        T_new[:, 0] = T_new[:, 0] + (
            Q_net / (rho_0 * c_p * dz_0) * dt * mask
        )
        # Project tau onto edge normal via the two adjacent cells.
        c1 = np.asarray(grid.cellsOnEdge[0], dtype=np.int64)
        c2 = np.asarray(grid.cellsOnEdge[1], dtype=np.int64)
        angle = np.asarray(grid.angleEdge, dtype=np.float64)
        tau_x_edge = 0.5 * (np.asarray(tau_x)[c1] + np.asarray(tau_x)[c2])
        tau_y_edge = 0.5 * (np.asarray(tau_y)[c1] + np.asarray(tau_y)[c2])
        # Ocean reaction force (-tau): air_sea_fluxes returns atmospheric
        # convention. The angleEdge projection rotates geographic E/N onto the
        # edge normal; the leading minus is the wind->ocean sign flip.
        tau_n = -(tau_x_edge * np.cos(angle) + tau_y_edge * np.sin(angle))
        edge_wet = ((mask[c1] > 0.5) & (mask[c2] > 0.5)).astype(np.float64)
        u_new = np.asarray(state.u.data, dtype=np.float64).copy()
        u_new[:, 0] = u_new[:, 0] + (
            tau_n / (rho_0 * dz_0) * dt * edge_wet
        )
        return state._replace(
            T=Field(jnp.asarray(T_new), name=state.T.name,
                    dims=state.T.dims, units=state.T.units),
            u=Field(jnp.asarray(u_new), name=state.u.name,
                    dims=state.u.dims, units=state.u.units),
        )

    raise NotImplementedError(
        f"OMIP-2 applicator does not support grid_type={grid_type!r}"
    )


def compute_omip2_surface_forcing(state, *, forcing, idx_t: int,
                                  grid, grid_type: str,
                                  rho_air: float = 1.225):
    """Build an :class:`OceanSurfaceForcing` (tau_x, tau_y, q_net, sw_down) on
    the model grid from CORE-II / JRA55 forcing, for INTEGRATION INSIDE
    ``model.step(state, dt, surface_forcing=...)`` -- the dynamics-core
    external-tau block -- rather than the operator-split forward-Euler
    :func:`apply_omip2_surface_fluxes`. Integrating the forcing within the
    timestep is energetically consistent with the dynamics (the split applicator
    pumps spurious KE -> runaway velocities over a multi-month global run).

    Conventions (matching the dynamics-core consumer, ocean_pe_latlon_cgrid):
    * ``tau_x``/``tau_y`` are left in the ATMOSPHERIC convention exactly as
      ``air_sea_fluxes`` returns them; the core applies the ``-tau`` ocean
      reaction + the geographic->grid rotation (so no sign/rotation here).
    * ``q_net`` is the TOTAL net surface heat flux into the ocean (turbulent +
      longwave + shortwave); the core subtracts the penetrating ``sw_down`` and
      distributes it over depth.

    Supports the lat-lon C-grid family (``latlon`` / ``latlon_regional`` via
    conservative regrid; ``tripole`` via nearest-neighbour on the 2-D T grid).
    """
    from legoesm.ocean.state import OceanSurfaceForcing
    sigma_sb = float(getattr(constants, "sigma_sb", 5.67e-8))
    T_freeze = float(constants.T_freeze)

    if grid_type in ("latlon", "latlon_regional"):
        lat_deg = np.degrees(np.asarray(grid.lat))
        lon_deg = np.degrees(np.asarray(grid.lon))
        forc = _sample_forcing_latlon(forcing, idx_t, lat_deg, lon_deg)
    elif grid_type == "tripole":
        lat_pts = np.degrees(np.asarray(grid.lat_T))
        lon_pts = np.degrees(np.asarray(grid.lon_T))
        shp = lat_pts.shape
        forc = _sample_forcing_points(
            forcing, idx_t, lat_pts.reshape(-1), lon_pts.reshape(-1),
        )
        forc = {k: v.reshape(shp) for k, v in forc.items()}
    else:
        raise NotImplementedError(
            f"compute_omip2_surface_forcing does not support grid_type={grid_type!r}"
        )

    # Top-cell ocean temperature (state stored in degC) -> K.
    T_sfc_K = np.asarray(state.T.data, dtype=np.float64)[..., 0] + T_freeze
    q_sfc = np.asarray(_bolton_q_sat(jnp.asarray(T_sfc_K)), dtype=np.float64)
    tau_x, tau_y, sh, lh = air_sea_fluxes(
        u10=jnp.asarray(forc["u10"]),
        v10=jnp.asarray(forc["v10"]),
        T_air_K=jnp.asarray(forc["T_air"]),
        q_air=jnp.asarray(forc["q_air"]),
        T_sfc_K=jnp.asarray(T_sfc_K),
        q_sfc=jnp.asarray(q_sfc),
        rho_air=jnp.asarray(rho_air),
    )
    lw_up = 0.97 * sigma_sb * T_sfc_K ** 4
    q_net = (np.asarray(sh) + np.asarray(lh)
             + forc["lw_down"] - lw_up + forc["sw_down"])
    return OceanSurfaceForcing(
        tau_x=jnp.asarray(np.asarray(tau_x)),
        tau_y=jnp.asarray(np.asarray(tau_y)),
        q_net=jnp.asarray(q_net),
        sw_down=jnp.asarray(forc["sw_down"]),
    )


__all__ = ["apply_omip2_surface_fluxes", "compute_omip2_surface_forcing"]
