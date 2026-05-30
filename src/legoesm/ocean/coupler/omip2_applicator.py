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


def _nn_interp_to_points(field, src_lat_deg, src_lon_deg,
                         dst_lat_deg_pts, dst_lon_deg_pts):
    """Nearest-neighbour 1-D-axis lookup from (src_lat, src_lon) to a
    1-D set of target points ``(dst_lat_pts, dst_lon_pts)``.

    Used for cube + MPAS grids where the target lives on unstructured
    cell centres rather than a regular lat-lon mesh; the lat-lon-
    coupled path goes through ``conservative_regrid`` instead.
    """
    src_lat = np.asarray(src_lat_deg)
    src_lon = np.asarray(src_lon_deg) % 360.0
    dst_lat = np.asarray(dst_lat_deg_pts)
    dst_lon = np.asarray(dst_lon_deg_pts) % 360.0
    i = np.clip(np.searchsorted(src_lat, dst_lat), 0, src_lat.size - 1)
    j = np.clip(np.searchsorted(src_lon, dst_lon), 0, src_lon.size - 1)
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


def apply_omip2_surface_fluxes(state, *, forcing, idx_t: int,
                                 z_coord, grid, grid_type: str,
                                 dt: float,
                                 rho_0: Optional[float] = None,
                                 c_p: Optional[float] = None,
                                 rho_air: float = 1.225):
    """Apply one timestep of JRA55-do / CORE-II forcing to ``state``.

    Currently supports ``grid_type="latlon"`` and
    ``grid_type="latlon_regional"`` -- the two grids the OMIP-2 and
    Bryan-THC drivers use. MPAS + cube support stubbed (raises
    ``NotImplementedError``) because the production OMIP-2 spec uses
    lat-lon; MPAS extension is a follow-up.

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
        tau_x_np = np.asarray(tau_x)
        tau_y_np = np.asarray(tau_y)
        lw_up = 0.97 * sigma_sb * T_sfc_K ** 4
        Q_net = (np.asarray(sh) + np.asarray(lh)
                 + forc["sw_down"] - lw_up + forc["lw_down"])
        mask = np.asarray(state.land_mask.data, dtype=np.float64)
        dT_top = Q_net / (rho_0 * c_p * dz_0) * dt * mask
        T_new = np.asarray(state.T.data, dtype=np.float64).copy()
        T_new[..., 0] = T_new[..., 0] + dT_top
        # C-grid face interpolation for tau_x, tau_y.
        u_face = np.asarray(state.u.data, dtype=np.float64).copy()
        v_face = np.asarray(state.v.data, dtype=np.float64).copy()
        n_lat, n_lon = tau_x_np.shape
        tau_x_face = np.zeros((n_lat, n_lon + 1), dtype=np.float64)
        tau_x_face[:, 1:-1] = 0.5 * (tau_x_np[:, :-1] + tau_x_np[:, 1:])
        tau_x_face[:, 0] = tau_x_np[:, 0]
        tau_x_face[:, -1] = tau_x_np[:, -1]
        u_mask = np.asarray(state.u_mask.data, dtype=np.float64)
        u_face[..., 0] = u_face[..., 0] + (
            tau_x_face / (rho_0 * dz_0) * dt * u_mask
        )
        tau_y_face = np.zeros((n_lat + 1, n_lon), dtype=np.float64)
        tau_y_face[1:-1, :] = 0.5 * (tau_y_np[:-1, :] + tau_y_np[1:, :])
        tau_y_face[0, :] = tau_y_np[0, :]
        tau_y_face[-1, :] = tau_y_np[-1, :]
        v_mask = np.asarray(state.v_mask.data, dtype=np.float64)
        v_face[..., 0] = v_face[..., 0] + (
            tau_y_face / (rho_0 * dz_0) * dt * v_mask
        )
        return state._replace(
            T=Field(jnp.asarray(T_new), name=state.T.name,
                    dims=state.T.dims, units=state.T.units),
            u=Field(jnp.asarray(u_face), name=state.u.name,
                    dims=state.u.dims, units=state.u.units),
            v=Field(jnp.asarray(v_face), name=state.v.name,
                    dims=state.v.dims, units=state.v.units),
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
        u_new[..., 0] = u_new[..., 0] + (
            np.asarray(tau_x) / (rho_0 * dz_0) * dt * mask
        )
        v_new[..., 0] = v_new[..., 0] + (
            np.asarray(tau_y) / (rho_0 * dz_0) * dt * mask
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
        tau_n = tau_x_edge * np.cos(angle) + tau_y_edge * np.sin(angle)
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


__all__ = ["apply_omip2_surface_fluxes"]
