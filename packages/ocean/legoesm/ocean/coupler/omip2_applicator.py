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


def _bolton_q_sat(T_K, p_hpa: float = constants.p_atm_std / 100.0):
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
        # LONGITUDE WRAP: pad the source with one ghost column on each side
        # (last column shifted -360, first column shifted +360) so a
        # destination cell straddling the 0/360 seam sees full source
        # coverage.  Without this the seam-adjacent destination column was
        # only partially covered (under-weighted forcing stripe at the last
        # longitude; with the coarse synthetic test forcing the column came
        # back HALVED, and the 10-m pressure iteration then NaN'd on the
        # resulting garbage air temperature).
        src_lon = np.asarray(src_lon_deg, dtype=np.float64)
        src_lon_padded = np.concatenate(
            [[src_lon[-1] - 360.0], src_lon, [src_lon[0] + 360.0]])
        src_lat_edges = _edges_from_centers_deg(src_lat_deg)
        src_lon_edges = _edges_from_centers_deg(src_lon_padded, periodic=True)
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
    f = jnp_local.asarray(field_2d)
    f_padded = jnp_local.concatenate([f[:, -1:], f, f[:, :1]], axis=1)
    return np.asarray(apply_conservative_regrid(f_padded, weights))


_BASE_FORCING_CHANNELS = ("u10", "v10", "T_air", "q_air",
                          "sw_down", "lw_down", "precip")
# Channels added for NEMO-parity surface fluxes (2026-06): solid
# precipitation (snow-fusion + snow heat-content terms of q_ns) and
# sea-level pressure (moist-air density + Goff saturation humidity).
# OPTIONAL: forcing sets built before the schema extension lack them and
# the flux assembly falls back to snow=0 / slp=standard atmosphere.
_OPTIONAL_FORCING_CHANNELS = ("snow", "slp")


def _forcing_channels(forcing):
    """Base channels + whichever optional channels ``forcing`` carries."""
    names = list(_BASE_FORCING_CHANNELS)
    for name in _OPTIONAL_FORCING_CHANNELS:
        if getattr(forcing, name, None) is not None:
            names.append(name)
    return tuple(names)


def _sample_forcing_latlon(forcing, idx_t, dst_lat_deg, dst_lon_deg):
    """Conservative-regrid the forcing channels at time ``idx_t`` onto a
    regular destination lat-lon grid."""
    out = {}
    for name in _forcing_channels(forcing):
        out[name] = _conservative_regrid_to_latlon(
            getattr(forcing, name)[idx_t],
            forcing.lat, forcing.lon,
            dst_lat_deg, dst_lon_deg,
        )
    return out


def _sample_forcing_points(forcing, idx_t, lat_pts_deg, lon_pts_deg):
    """Nearest-neighbour sample the forcing channels at a set of points
    (cube / MPAS cell centres)."""
    out = {}
    for name in _forcing_channels(forcing):
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

    Physical constants are routed through :mod:`legoesm.constants`
    (``emissivity_ocean``, ``rho_air``, ``sigma_sb``, ``p_atm_std``,
    ``T_freeze``) and saturation through :mod:`legoesm.thermo`
    (``_bolton_q_sat`` wraps ``saturation_vapor_pressure``) -- no hardcoded
    thermodynamic literals remain here.  The wind-stress *sign* (ocean reaction
    = -tau), by contrast, is corrected here vs the previous (+tau) latlon code
    -- see the momentum comment below.
    """
    T_sfc_K = np.asarray(state.T.data, dtype=np.float64)[..., 0] + constants.T_freeze
    q_sfc = np.asarray(_bolton_q_sat(jnp.asarray(T_sfc_K)), dtype=np.float64)
    # Legacy operator-split path: pinned to the pre-NCAR two-coefficient
    # scheme (algo='ly09_2coeff') so its regression tests stay bit-exact.
    # The FAITHFUL in-step path (compute_omip2_surface_forcing) uses the
    # NEMO NCAR algorithm.
    tau_x, tau_y, sh, lh, _evap = air_sea_fluxes(
        u10=jnp.asarray(forc["u10"]),
        v10=jnp.asarray(forc["v10"]),
        T_air_K=jnp.asarray(forc["T_air"]),
        q_air=jnp.asarray(forc["q_air"]),
        T_sfc_K=jnp.asarray(T_sfc_K),
        q_sfc=jnp.asarray(q_sfc),
        rho_air=jnp.asarray(rho_air),
        algo="ly09_2coeff",
    )
    # --- Heat: Q_net positive into the ocean warms the top cell. ---
    lw_up = constants.emissivity_ocean * sigma_sb * T_sfc_K ** 4
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
                                 rho_air: float = constants.rho_air):
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
        tau_x, tau_y, sh, lh, _evap = air_sea_fluxes(
            u10=jnp.asarray(forc["u10"]),
            v10=jnp.asarray(forc["v10"]),
            T_air_K=jnp.asarray(forc["T_air"]),
            q_air=jnp.asarray(forc["q_air"]),
            T_sfc_K=jnp.asarray(T_sfc_K),
            q_sfc=jnp.asarray(q_sfc),
            rho_air=jnp.asarray(rho_air),
            algo="ly09_2coeff",
        )
        lw_up = constants.emissivity_ocean * sigma_sb * T_sfc_K ** 4
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
        tau_x, tau_y, sh, lh, _evap = air_sea_fluxes(
            u10=jnp.asarray(forc["u10"]),
            v10=jnp.asarray(forc["v10"]),
            T_air_K=jnp.asarray(forc["T_air"]),
            q_air=jnp.asarray(forc["q_air"]),
            T_sfc_K=jnp.asarray(T_sfc_K),
            q_sfc=jnp.asarray(q_sfc),
            rho_air=jnp.asarray(rho_air),
            algo="ly09_2coeff",
        )
        lw_up = constants.emissivity_ocean * sigma_sb * T_sfc_K ** 4
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


def _sample_omip2_forcing(forcing, idx_t, grid, grid_type):
    """Sample the CORE-II/JRA forcing channels onto the model grid for a record.

    Shared by :func:`compute_omip2_surface_forcing` (momentum/heat) and
    :func:`compute_omip2_freshwater_forcing` (P - E) so both see the SAME
    spatially-sampled fields.  ``latlon``/``latlon_regional`` use the
    conservative regrid onto the regular T grid; ``tripole``/``cubed_sphere``/
    ``mpas`` use nearest-neighbour onto the (possibly 2-D / 1-D) cell centres.
    """
    if grid_type in ("latlon", "latlon_regional"):
        lat_deg = np.degrees(np.asarray(grid.lat))
        lon_deg = np.degrees(np.asarray(grid.lon))
        return _sample_forcing_latlon(forcing, idx_t, lat_deg, lon_deg)
    if grid_type == "tripole":
        lat_pts = np.degrees(np.asarray(grid.lat_T))
        lon_pts = np.degrees(np.asarray(grid.lon_T))
        shp = lat_pts.shape
        forc = _sample_forcing_points(
            forcing, idx_t, lat_pts.reshape(-1), lon_pts.reshape(-1),
        )
        return {k: v.reshape(shp) for k, v in forc.items()}
    if grid_type == "cubed_sphere":
        lat_pts = np.degrees(np.asarray(grid.lat))
        lon_pts = np.degrees(np.asarray(grid.lon))
        shp = lat_pts.shape
        forc = _sample_forcing_points(
            forcing, idx_t, lat_pts.reshape(-1), lon_pts.reshape(-1),
        )
        return {k: v.reshape(shp) for k, v in forc.items()}
    if grid_type in ("mpas", "mpas_regional"):
        lat_pts = np.degrees(np.asarray(grid.latCell))
        lon_pts = np.degrees(np.asarray(grid.lonCell))
        return _sample_forcing_points(forcing, idx_t, lat_pts, lon_pts)
    raise NotImplementedError(
        f"_sample_omip2_forcing does not support grid_type={grid_type!r}"
    )


def sample_omip2_forcing(forcing, idx_t, grid, grid_type):
    """Public wrapper for :func:`_sample_omip2_forcing`.

    Returns the CORE-II / JRA forcing channels (``u10``, ``v10``, ``T_air`` [K],
    ``q_air``, ``sw_down``, ``lw_down``, ``precip``, and the optional ``snow`` /
    ``slp``) spatially sampled onto the model grid for one time record — the
    SAME sampling :func:`compute_omip2_surface_forcing` and
    :func:`compute_omip2_freshwater_forcing` consume.  Exposed so an external
    coupler driver (e.g. the prognostic sea-ice glue in
    ``scripts/run/run_omip_core2.py``) can populate an ``AtmToSurface`` from the
    identical sampled fields rather than re-deriving the regrid (cross-module
    callers must not import the private ``_sample_omip2_forcing``)."""
    return _sample_omip2_forcing(forcing, idx_t, grid, grid_type)


def compute_omip2_freshwater_forcing(state, *, forcing, idx_t: int,
                                     grid, grid_type: str,
                                     runoff_R=None,
                                     emp: bool = True,
                                     ramp: float = 1.0,
                                     rho_air: float = constants.rho_air):
    """Build a :class:`FreshwaterForcing` (P, E, runoff) for the OMIP-2 run.

    Delivered to the ocean via the in-core channel
    ``model.step(state, dt, surface_forcing=sf, freshwater=fw)`` so the surface
    freshwater enters the SAME tested path the model uses for its own freshwater:
    the virtual-salt tendency ``dS_top/dt = -S_ref * F_fw / (rho_0 * dz_0)`` with
    ``config.S_ref`` AND the free-surface source ``deta/dt = F_fw/rho_0`` inside
    the barotropic solve (NOT an operator-split post-step NumPy update, which
    would bypass the eta solve, break AD/JIT device-residency, and double-apply
    on the cube's 'external' physics that already consumes
    ``surface_forcing.freshwater``).

    Components (all [kg/m^2/s], + INTO ocean):
    * ``precip`` -- prescribed CORE-II precipitation (rain+snow).
    * ``evap``   -- INTERACTIVE evaporation, positive UP, back-derived from the
      latent heat flux (``air_sea_fluxes`` returns ``lh`` +INTO ocean, so an
      evaporating column has lh<0 and E = -lh/L_v >= 0).  ``net_freshwater_flux``
      forms ``precip - evap + runoff`` => ``precip + lh/L_v + runoff``.
    * ``runoff`` -- optional Dai-Trenberth river/ice-shelf/iceberg field.

    Without P - E the ocean only sees runoff (a one-sided freshwater SOURCE) and
    freshens ~0.3 PSU/yr -- the multi-year drift that strong SSS restoring was
    masking by injecting net salt (-> AMOC suppression).

    Parameters
    ----------
    emp : bool
        When False, zero the P - E contribution (ablation / runoff-only).
    ramp : float
        Cold-start spin-up scale in [0, 1] applied to ALL freshwater components
        (matches the tau/q_net ramp).
    """
    from legoesm.ocean.freshwater import FreshwaterForcing

    forc = _sample_omip2_forcing(forcing, idx_t, grid, grid_type)
    # MPAS state is (nCells,) at the surface; cube/latlon/tripole are (..., 0).
    T_sfc_K = np.asarray(state.T.data, dtype=np.float64)[..., 0] + float(constants.T_freeze)
    slp = forc.get("slp")
    # NCAR algo (NEMO-faithful): q_sfc = 0.98*q_sat_goff(SST, slp) computed
    # internally; evap returned directly (= -lh / L_vap(SST), consistent with
    # the latent flux by construction -- no separate constants.L_v division).
    _, _, _, _, evap_j = air_sea_fluxes(
        u10=jnp.asarray(forc["u10"]), v10=jnp.asarray(forc["v10"]),
        T_air_K=jnp.asarray(forc["T_air"]), q_air=jnp.asarray(forc["q_air"]),
        T_sfc_K=jnp.asarray(T_sfc_K),
        slp_Pa=None if slp is None else jnp.asarray(slp),
    )
    if emp:
        precip = np.asarray(forc["precip"], dtype=np.float64)
        evap = np.asarray(evap_j, dtype=np.float64)   # positive UP
    else:
        precip = np.zeros_like(np.asarray(forc["precip"], dtype=np.float64))
        evap = np.zeros_like(precip)
    if runoff_R is not None:
        runoff = np.asarray(runoff_R, dtype=np.float64)
    else:
        runoff = np.zeros_like(precip)
    r = float(ramp)
    z = jnp.zeros_like(jnp.asarray(precip))
    return FreshwaterForcing(
        precip=jnp.asarray(precip * r),
        evap=jnp.asarray(evap * r),
        runoff=jnp.asarray(runoff * r),
        ice_fw=z,
        restoring=z,
    )


def dm2dc_sw_factor(grid, dm2dc_window):
    """Analytic diurnal-cycle shortwave factor field (NEMO ln_dm2dc / sbcdcy,
    Bernie et al. 2007) for one step window, on the model grid's T points.

    ``dm2dc_window = (day_of_year, year_len_days, t_frac_lo, t_frac_up)``.
    Mean-preserving over a day by construction.  ONE shared implementation so
    every consumer of the CORE-II daily-mean SW (the ocean surface forcing AND
    the prognostic sea-ice AtmToSurface) sees the SAME diurnal modulation —
    codex r1 #2: the ice tile previously received the raw daily-mean SW while
    the ocean saw the modulated one.  Lat-lon / tripole C-grid families only
    (matches the run drivers, which reject --dm2dc elsewhere)."""
    from legoesm.ocean.forcing.diurnal_cycle import diurnal_sw_factor
    _day_of_year, _year_len, _t_lo, _t_up = dm2dc_window
    _lat_T = getattr(grid, "lat_T", None)
    if _lat_T is not None:            # tripole family (2-D, radians)
        _lat_deg = np.degrees(np.asarray(_lat_T))
        _lon_deg = np.degrees(np.asarray(grid.lon_T))
    else:                             # regular lat-lon (1-D, radians)
        _lat_deg = np.degrees(np.asarray(grid.lat))[:, None]
        _lon_deg = np.degrees(np.asarray(grid.lon))[None, :]
    return np.asarray(diurnal_sw_factor(
        _lon_deg, _lat_deg,
        day_of_year=_day_of_year, year_len_days=_year_len,
        t_frac_lo=_t_lo, t_frac_up=_t_up))


def _sw_albedo_factor(sw_down, ice_albedo):
    """Reduce downwelling SW by the effective surface albedo.

    ``albedo_eff = alpha_ocean*(1-siconc) + alpha_ice*siconc`` where ``siconc``
    (= ``ice_albedo`` arg, the prescribed sea-ice concentration in [0,1]) weights
    open-ocean vs sea-ice broadband albedo.  Returns ``sw_down*(1-albedo_eff)``.
    ``ice_albedo=None`` -> unchanged ``sw_down`` (bit-exact default).  This is the
    ONLY place SW albedo is applied; the cores' 0.94 penetration/surface split is
    a vertical-distribution split (NOT an albedo) and is left untouched, so the
    albedoed ``sw_net`` flows once into both ``q_net`` and the penetrating SW.
    """
    if ice_albedo is None:
        return sw_down
    a_oc = float(constants.alpha_ocean_broadband)
    a_ice = float(constants.alpha_ice_broadband_cold)
    # Defensive clip: the loader already returns siconc in [0,1], but guard the
    # API contract (albedo_eff in [0,1], 0<=sw_net<=sw_down) against a NaN /
    # out-of-range caller (codex) so a bad siconc can never amplify SW.
    sic = np.clip(np.nan_to_num(np.asarray(ice_albedo, dtype=np.float64),
                                nan=0.0), 0.0, 1.0)
    albedo_eff = a_oc * (1.0 - sic) + a_ice * sic
    return np.asarray(sw_down, dtype=np.float64) * (1.0 - albedo_eff)


def _ice_surface_heat(sw_down, q_non_sw, ice_albedo, *,
                      under_ice: bool = False, tau_ice_sw: float = 0.03):
    """Combine SW + non-SW surface heat into ``(sw_ocean, q_net)`` [W/m², +into
    ocean] with an optional PRESCRIBED-ICE thermodynamic boundary (codex HIGH).

    Three regimes (``ice_albedo`` = prescribed siconc in [0,1], or ``None``):

    * ``ice_albedo is None`` -> NO albedo (bit-exact legacy default):
      ``sw_ocean = sw_down``; ``q_net = q_non_sw + sw_down``.
    * ``under_ice=False`` (albedo-only surrogate): ``sw_ocean`` = open/ice albedo-
      weighted SW (:func:`_sw_albedo_factor`); full open-ocean ``q_non_sw``.
    * ``under_ice=True`` (prescribed-ice boundary): under sea ice almost no SW
      reaches the ocean and the open-ocean turbulent+LW fluxes do not act on the
      ice-covered fraction.  Per-cell ice fraction ``sic`` blends open water and
      ice::

          sw_ocean = (1-sic)·sw_down·(1-α_ocean) + sic·sw_down·τ_ice_sw
          q_net    = sw_ocean + (1-sic)·q_non_sw

      ``τ_ice_sw`` (~0.03) is the small SW transmittance through ice/snow into the
      ocean (vs the albedo-only surrogate's ``1-α_ice=0.35``, which over-warms the
      under-ice ocean -- the Southern-Ocean warm bias).  At ``sic=0`` this equals
      the albedo-only open-water value, so only ice-covered cells change.  The
      under-ice relaxation toward the freezing point is applied as a post-step
      state nudge (:func:`under_ice_freeze_relax`), NOT here, so it needs no
      top-layer thickness in this flux producer."""
    q_non_sw = np.asarray(q_non_sw, dtype=np.float64)
    if not under_ice or ice_albedo is None:
        sw_ocean = _sw_albedo_factor(sw_down, ice_albedo)
        return sw_ocean, q_non_sw + sw_ocean
    a_oc = float(constants.alpha_ocean_broadband)
    sic = np.clip(np.nan_to_num(np.asarray(ice_albedo, dtype=np.float64),
                                nan=0.0), 0.0, 1.0)
    swd = np.asarray(sw_down, dtype=np.float64)
    sw_open = swd * (1.0 - a_oc)
    sw_ice = swd * float(tau_ice_sw)
    sw_ocean = (1.0 - sic) * sw_open + sic * sw_ice
    q_net = sw_ocean + (1.0 - sic) * q_non_sw
    return sw_ocean, q_net


def under_ice_freeze_relax(T_top_C, ice_concentration, dt: float, *,
                           tau_ice_days: float = 20.0, T_freeze_C=None):
    """Relax the under-ice surface ocean temperature toward the freezing point
    (prescribed-ice thermodynamic boundary; codex HIGH).  Returns the updated
    top-cell temperature [°C].

        T_top <- T_top + (dt/τ_ice)·sic·(T_freeze - T_top)

    The ice-ocean interface sits at the freezing point, so under prescribed ice
    the surface ocean is nudged toward ``T_freeze`` with timescale ``τ_ice`` (days)
    weighted by the ice fraction ``sic``.  TWO-SIDED: it COOLS a too-warm under-ice
    cell (the Southern-Ocean / Antarctic warm bias) and HOLDS a cold Arctic cell
    near freezing, while ``sic=0`` open water is untouched -> leaves the seasonal-
    albedo NH-summer warming intact.  Intentionally non-conservative for the ocean
    alone (the missing reservoir is the prescribed ice's latent heat).  Host-loop
    helper (NumPy; the ``--ice-thermo`` host path, NOT the lax.scan path -- scan
    refuses it).  A convex relaxation, unconditionally stable: ``dt/τ`` is clipped
    to <=1 so the update never overshoots ``T_freeze``."""
    if T_freeze_C is None:
        T_freeze_C = float(constants.T_freeze_ocean) - float(constants.T_freeze)
    sic = np.clip(np.nan_to_num(np.asarray(ice_concentration, dtype=np.float64),
                                nan=0.0), 0.0, 1.0)
    tau_s = float(tau_ice_days) * 86400.0
    alpha = np.minimum(float(dt) / max(tau_s, 1.0e-30), 1.0) * sic
    T = np.asarray(T_top_C, dtype=np.float64)
    return T + alpha * (float(T_freeze_C) - T)


def compute_omip2_surface_forcing(state, *, forcing, idx_t: int,
                                  grid, grid_type: str,
                                  rho_air: float = constants.rho_air,
                                  ice_albedo=None,
                                  under_ice: bool = False,
                                  tau_ice_sw: float = 0.03,
                                  dm2dc_window=None):
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

    ``ice_albedo`` (optional, shape of the model surface field): prescribed
    sea-ice concentration in [0,1].  When given, the downwelling SW is reduced by
    the effective open-ocean/sea-ice albedo (see :func:`_sw_albedo_factor`) in
    BOTH ``q_net`` and the returned ``sw_down``.  ``None`` is bit-exact default.

    ``under_ice`` (prescribed-ice thermodynamic boundary; default False keeps the
    albedo-only surrogate): under sea ice, cut the SW reaching the ocean to
    ``tau_ice_sw`` (~0.03, vs the albedo-only 0.35) and suppress the open-ocean
    turbulent+LW fluxes by ``(1-sic)`` (see :func:`_ice_surface_heat`).  The
    companion under-ice freezing relaxation (:func:`under_ice_freeze_relax`) is a
    post-step state nudge.  Together they cool the over-warm under-ice Southern
    Ocean while leaving seasonal open water to benefit from the albedo correction.

    Supports the lat-lon C-grid family (``latlon`` / ``latlon_regional`` via
    conservative regrid; ``tripole`` via nearest-neighbour on the 2-D T grid).
    """
    from legoesm.ocean.state import OceanSurfaceForcing
    sigma_sb = float(constants.sigma_sb)
    T_freeze = float(constants.T_freeze)

    forc = _sample_omip2_forcing(forcing, idx_t, grid, grid_type)

    if dm2dc_window is not None:
        # NEMO ln_dm2dc (sbcdcy, Bernie et al. 2007): modulate the DAILY-MEAN
        # downwelling SW with the analytic diurnal shape for this step's
        # window.  Mean-preserving by construction; applied to the raw
        # sw_down so BOTH q_net and the penetrative channel see it (exactly
        # where NEMO applies sbc_dcy to qsr).
        _fac = dm2dc_sw_factor(grid, dm2dc_window)
        forc = dict(forc)
        forc["sw_down"] = np.asarray(forc["sw_down"], dtype=np.float64) * _fac

    # Top-cell ocean temperature (state stored in degC) -> K.
    T_sfc_K = np.asarray(state.T.data, dtype=np.float64)[..., 0] + T_freeze
    slp = forc.get("slp")
    tau_x, tau_y, sh, lh, evap = air_sea_fluxes(
        u10=jnp.asarray(forc["u10"]),
        v10=jnp.asarray(forc["v10"]),
        T_air_K=jnp.asarray(forc["T_air"]),
        q_air=jnp.asarray(forc["q_air"]),
        T_sfc_K=jnp.asarray(T_sfc_K),
        slp_Pa=None if slp is None else jnp.asarray(slp),
    )
    # Non-solar open-water heat flux, assembled EXACTLY like NEMO blk_oce_2:
    #   qns = eps_w*(LW_down - sigma*T_s^4)            net LW (Kirchhoff: the
    #                                                  SAME eps_w=0.98 weights
    #                                                  absorption AND emission)
    #       + sensible + latent                        NCAR turbulent fluxes
    #       - snow*L_fus                               melt freshly-fallen snow
    #       - evap*c_p_sw*T_s[degC]                    evap removes heat at SST
    #       + rain*c_p_sw*(theta_air[degC])            rain heat content
    #       + snow*c_p_ice*min(theta_air[degC], 0)     snow heat content
    # (NEMO blk_oce_2 receives theta_air_zt for the rain/snow terms and the
    # ABSOLUTE skin/bulk SST for LW + evap heat content; rcp/rcpi/rLfus are
    # the NEMO-parity values.)
    lw_net = constants.emissivity_seawater_lw * (
        np.asarray(forc["lw_down"], dtype=np.float64) - sigma_sb * T_sfc_K ** 4
    )
    precip_np = np.asarray(forc["precip"], dtype=np.float64)
    snow_np = (np.asarray(forc["snow"], dtype=np.float64)
               if forc.get("snow") is not None else np.zeros_like(precip_np))
    # CORE-II precip = RAIN+SNOW; guard tiny negative rain from regridding.
    rain_np = np.maximum(precip_np - snow_np, 0.0)
    from legoesm.ocean.bulk_flux_omip import potential_air_temperature_10m
    theta_air_j, _ = potential_air_temperature_10m(
        jnp.asarray(forc["T_air"]), jnp.asarray(forc["q_air"]),
        None if slp is None else jnp.asarray(slp))
    theta_air_C = np.asarray(theta_air_j, dtype=np.float64) - T_freeze
    T_sfc_C = T_sfc_K - T_freeze
    q_precip_evap = (
        - snow_np * float(constants.L_fus_nemo)
        - np.asarray(evap, dtype=np.float64) * float(constants.c_p_seawater) * T_sfc_C
        + rain_np * float(constants.c_p_seawater) * theta_air_C
        + snow_np * float(constants.c_p_ice_nemo) * np.minimum(theta_air_C, 0.0)
    )
    # Under a prescribed-ice boundary (under_ice) ``_ice_surface_heat``
    # suppresses this on the ice-covered fraction (snow falling on ice does
    # NOT cool the ocean) and cuts the under-ice SW; otherwise it is the
    # albedo-only surrogate.
    q_non_sw = np.asarray(sh) + np.asarray(lh) + lw_net + q_precip_evap
    sw_net, q_net = _ice_surface_heat(
        forc["sw_down"], q_non_sw, ice_albedo,
        under_ice=under_ice, tau_ice_sw=tau_ice_sw)
    # NOTE: the surface freshwater flux P - E is NOT returned here.  It is built
    # by :func:`compute_omip2_freshwater_forcing` and delivered to the ocean via
    # the in-core ``model.step(..., freshwater=FreshwaterForcing)`` channel (which
    # applies the virtual salt with ``config.S_ref`` + the eta free-surface source
    # in the barotropic solve).  Routing it through ``surface_forcing.freshwater``
    # would double-apply on the cube ('external' physics consumes that field) and
    # bypass the eta/free-surface treatment on latlon/MPAS.
    return OceanSurfaceForcing(
        tau_x=jnp.asarray(np.asarray(tau_x)),
        tau_y=jnp.asarray(np.asarray(tau_y)),
        q_net=jnp.asarray(q_net),
        sw_down=jnp.asarray(np.asarray(sw_net)),
    )


# ===========================================================================
# JAX-traceable / lax.scan-fusable forcing path (issue #354)
# ===========================================================================
#
# ``compute_omip2_surface_forcing`` (above) pulls ``state.T`` to the host
# every step (np.asarray) to compute the saturation humidity, which forces
# a device sync per step and prevents wrapping the time loop in
# ``jax.lax.scan``.  The functions below provide a pure-JAX equivalent for
# the tripole grid: the forcing is lifted to device arrays ONCE and the
# spatial nearest-neighbour sample is a static gather, so the whole block
# of steps fuses on-device.  At 1/4 deg the raw CORE-II forcing
# (1460 x 94 x 192 x 7ch, ~1.4 GB) fits on device, whereas pre-sampling all
# records to the model grid (1206 x 1440) would not — hence the gather.


def core2_forcing_nn_indices(forcing, lat_pts_deg, lon_pts_deg):
    """Fixed nearest-neighbour source indices ``(i, j)`` from the CORE-II
    forcing grid to a set of model points.

    Geometry-only (time-independent), so the spatial sample inside a
    ``lax.scan`` becomes a static gather.  Replicates the ``(i, j)``
    computed in :func:`_nn_interp_to_points` EXACTLY (nearest latitude by
    ``|Δlat|``; nearest longitude by periodic wrap distance).
    """
    src_lat = np.asarray(forcing.lat, dtype=np.float64)
    src_lon = np.asarray(forcing.lon, dtype=np.float64) % 360.0
    dst_lat = np.asarray(lat_pts_deg, dtype=np.float64)
    dst_lon = np.asarray(lon_pts_deg, dtype=np.float64) % 360.0
    i = np.argmin(np.abs(src_lat[:, None] - dst_lat[None, :]), axis=0)
    dlon = np.abs(src_lon[:, None] - dst_lon[None, :])
    dlon = np.minimum(dlon, 360.0 - dlon)
    j = np.argmin(dlon, axis=0)
    return i.astype(np.int32), j.astype(np.int32)


def build_core2_forcing_device_stack(forcing, grid, grid_type: str):
    """Lift CORE-II forcing to device arrays + fixed NN indices for the
    scan kernel (issue #354).  Tripole only (the faithful OMIP path);
    other grids keep the host per-step sampling.

    Returns
    -------
    forcing_stack : dict[str, jax.Array]
        6 channels (u10, v10, T_air, q_air, sw_down, lw_down), each
        ``(n_rec, n_lat_f, n_lon_f)`` on device.
    nn_i, nn_j : jax.Array (int32, ``(n_lat*n_lon,)``)
        Nearest-neighbour source indices into the forcing grid.
    grid_shape : tuple
        ``(n_lat, n_lon)`` of the model T grid.
    """
    if grid_type != "tripole":
        raise NotImplementedError(
            "build_core2_forcing_device_stack: the lax.scan forcing path is "
            "implemented for grid_type='tripole' (the faithful OMIP path); "
            f"got {grid_type!r}.  Use the host compute_omip2_surface_forcing."
        )
    lat_pts = np.degrees(np.asarray(grid.lat_T)).reshape(-1)
    lon_pts = np.degrees(np.asarray(grid.lon_T)).reshape(-1)
    nn_i, nn_j = core2_forcing_nn_indices(forcing, lat_pts, lon_pts)
    channels = ("u10", "v10", "T_air", "q_air", "sw_down", "lw_down",
                "precip")
    forcing_stack = {
        name: jnp.asarray(np.asarray(getattr(forcing, name)))
        for name in channels
    }
    # Optional NEMO-parity channels: snow (q_ns fusion/heat-content terms)
    # and slp (moist-air density + Goff saturation).  Missing -> the same
    # fallbacks as the host path (snow=0, slp=standard atmosphere), built
    # as full records so the scan gather stays shape-uniform.
    ref = np.asarray(forcing.precip)
    snow = getattr(forcing, "snow", None)
    slp = getattr(forcing, "slp", None)
    forcing_stack["snow"] = jnp.asarray(
        np.zeros_like(ref) if snow is None else np.asarray(snow))
    forcing_stack["slp"] = jnp.asarray(
        np.full_like(ref, float(constants.p_atm_std)) if slp is None
        else np.asarray(slp))
    grid_shape = tuple(np.asarray(grid.lat_T).shape)
    return forcing_stack, jnp.asarray(nn_i), jnp.asarray(nn_j), grid_shape


def compute_omip2_surface_forcing_jax(
    state, *, forcing_stack, nn_i, nn_j, grid_shape, idx_t, rho_air=constants.rho_air,
):
    """Pure-JAX, ``lax.scan``-traceable form of
    :func:`compute_omip2_surface_forcing` for the tripole grid (issue #354).

    ``idx_t`` is the (possibly traced) 6-hourly CORE-II record index.  The
    spatial sample is a static nearest-neighbour gather via the fixed
    ``(nn_i, nn_j)`` map, so there is NO host roundtrip per step.
    Bit-equivalent (to fp tolerance) to the host function for tau/q_net/sw_down:
    identical ``air_sea_fluxes`` (NCAR), identical NEMO-form non-solar assembly
    (Kirchhoff LW + snow fusion + precip/evap heat content), identical
    nearest-neighbour spatial sample.  NO ice (``ice_albedo``/``under_ice``)
    support — the prescribed-ice runs use the host loop.

    LIMITATION: this scan variant covers ONLY tau/q_net/sw_down.  The device
    stack DOES carry precip/snow/slp (the q_ns heat-content terms need them),
    but the P - E / runoff ``FreshwaterForcing`` is still NOT built here: the
    faithful surface-freshwater forcing is applied only on the HOST run loop
    (``compute_omip2_freshwater_forcing`` -> ``model.step(freshwater=...)``),
    and the run driver refuses ``--scan-block`` for salinity-faithful runs
    until the freshwater channel is wired into the scan body.
    """
    from legoesm.ocean.state import OceanSurfaceForcing
    sigma_sb = float(constants.sigma_sb)
    T_freeze = float(constants.T_freeze)

    def _sample(name):
        rec = forcing_stack[name][idx_t]            # (n_lat_f, n_lon_f)
        return rec[nn_i, nn_j].reshape(grid_shape)  # (n_lat, n_lon)

    u10 = _sample("u10")
    v10 = _sample("v10")
    T_air = _sample("T_air")
    q_air = _sample("q_air")
    sw_down = _sample("sw_down")
    lw_down = _sample("lw_down")
    precip = _sample("precip")
    snow = _sample("snow")
    slp = _sample("slp")

    # Top-cell ocean temperature (state in degC) -> K, pure JAX (no host pull).
    T_sfc_K = state.T.data[..., 0] + T_freeze
    tau_x, tau_y, sh, lh, evap = air_sea_fluxes(
        u10=u10, v10=v10, T_air_K=T_air, q_air=q_air,
        T_sfc_K=T_sfc_K, slp_Pa=slp,
    )
    # Non-solar assembly: EXACT jnp mirror of the host
    # ``compute_omip2_surface_forcing`` (NEMO blk_oce_2 form) -- net-LW
    # Kirchhoff eps_w, snow fusion, rain/snow heat content at theta_air,
    # evap heat content at the absolute SST.
    from legoesm.ocean.bulk_flux_omip import potential_air_temperature_10m
    lw_net = constants.emissivity_seawater_lw * (
        lw_down - sigma_sb * T_sfc_K ** 4)
    rain = jnp.maximum(precip - snow, 0.0)
    theta_air, _ = potential_air_temperature_10m(T_air, q_air, slp)
    theta_air_C = theta_air - T_freeze
    T_sfc_C = T_sfc_K - T_freeze
    q_precip_evap = (
        - snow * constants.L_fus_nemo
        - evap * constants.c_p_seawater * T_sfc_C
        + rain * constants.c_p_seawater * theta_air_C
        + snow * constants.c_p_ice_nemo * jnp.minimum(theta_air_C, 0.0)
    )
    q_net = sh + lh + lw_net + q_precip_evap + sw_down
    return OceanSurfaceForcing(
        tau_x=tau_x, tau_y=tau_y, q_net=q_net, sw_down=sw_down,
    )


def build_omip2_scan_block_fn(
    model, dt, grid_shape, *, rho_air=constants.rho_air, ramp_s=0.0,
):
    """Build a JIT-compiled ``lax.scan`` block-step function for the tripole
    OMIP time loop (issue #354).

    The returned
    ``block_fn(state, forcing_stack, nn_i, nn_j, idx_t_block, step0)``
    advances ``state`` over ``len(idx_t_block)`` steps; each step samples
    the CORE-II forcing on-device via
    :func:`compute_omip2_surface_forcing_jax`, optionally applies the
    spin-up wind/heat ramp, and calls ``model._step_impl`` (the un-wrapped
    core step, so no nested JIT).  Because the forcing has no host
    roundtrip, XLA fuses the whole block.

    ``forcing_stack`` / ``nn_i`` / ``nn_j`` are explicit JIT arguments
    (NOT closure captures), so the ~1.3 GB 1/4-deg forcing is a device
    INPUT rather than an embedded compiled constant.  ``grid_shape`` is a
    static (closure) shape used only in ``reshape``.

    Host-side per-step pieces of the Python loop (diagnostics, snapshots,
    non-finite abort, optional WOA nudging / spin-up drag) are intentionally
    NOT inside the scan — the caller runs them at block boundaries and must
    not enable nudging/drag on the scan path.
    """
    import jax
    from jax import lax

    apply_ramp = ramp_s > 0.0  # static gate (CLAUDE.md feature-gating)

    @jax.jit
    def block_fn(state, forcing_stack, nn_i, nn_j, idx_t_block, step0):
        def _body(carry, idx_t):
            st, step = carry
            sf = compute_omip2_surface_forcing_jax(
                st, forcing_stack=forcing_stack, nn_i=nn_i, nn_j=nn_j,
                grid_shape=grid_shape, idx_t=idx_t, rho_air=rho_air,
            )
            if apply_ramp:
                ramp = jnp.minimum(
                    1.0, (step.astype(sf.tau_x.dtype) * dt) / ramp_s)
                sf = sf._replace(
                    tau_x=sf.tau_x * ramp, tau_y=sf.tau_y * ramp,
                    q_net=sf.q_net * ramp, sw_down=sf.sw_down * ramp,
                )
            st = model._step_impl(st, dt, surface_forcing=sf)
            # The scan body calls ``_step_impl`` directly (no nested JIT), so it
            # bypasses ``model.step``'s post-step freezing-point floor.  Re-apply
            # it here under the same STATIC config gate so ``--scan-block`` and
            # the default Python loop are physically identical (issue #354 +
            # freeze_floor). ``_step_impl`` returns a plain state here
            # (outer_integrator='forward_euler'; scan does not support ab2).
            if getattr(model.config, "freeze_floor", False):
                st = model._apply_freeze_floor(st)
            return (st, step + 1), None

        (state, _), _ = lax.scan(_body, (state, step0), idx_t_block)
        return state

    return block_fn


__all__ = [
    "apply_omip2_surface_fluxes",
    "compute_omip2_surface_forcing",
    "compute_omip2_freshwater_forcing",
    "compute_omip2_surface_forcing_jax",
    "build_core2_forcing_device_stack",
    "build_omip2_scan_block_fn",
    "core2_forcing_nn_indices",
]
