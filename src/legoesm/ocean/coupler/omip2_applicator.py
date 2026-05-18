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
from legoesm.ocean.bulk_flux_omip import air_sea_fluxes


def _bolton_q_sat(T_K, p_hpa: float = 1013.25):
    """Bolton (1980) saturation specific humidity at temperature T_K."""
    T_C = T_K - 273.15
    e_s = 6.112 * jnp.exp(17.67 * T_C / (T_C + 243.5))
    return 0.622 * e_s / (p_hpa - 0.378 * e_s)


def _interp_to_model_grid(field, src_lat_deg, src_lon_deg,
                          dst_lat_deg, dst_lon_deg):
    """Nearest-neighbour 1-D interpolation from (src_lat, src_lon) to
    (dst_lat, dst_lon). Smoke-grade; production uses
    ``legoesm.grids.conservative_regrid``.

    ``field`` has shape ``(n_lat_src, n_lon_src)``.
    """
    src_lat = np.asarray(src_lat_deg)
    src_lon = np.asarray(src_lon_deg) % 360.0
    dst_lat = np.asarray(dst_lat_deg)
    dst_lon = np.asarray(dst_lon_deg) % 360.0
    i = np.clip(
        np.searchsorted(src_lat, dst_lat), 0, src_lat.size - 1,
    )
    j = np.clip(
        np.searchsorted(src_lon, dst_lon), 0, src_lon.size - 1,
    )
    return np.asarray(field)[i[:, None], j[None, :]]


def _sample_forcing_at_time(forcing, idx_t: int, lat_deg, lon_deg):
    """Sample one time slice of the seven-channel forcing onto a
    target lat-lon grid (nearest-neighbour). Returns a dict of arrays
    of shape ``(n_lat, n_lon)``.
    """
    out = {}
    for name in ("u10", "v10", "T_air", "q_air",
                  "sw_down", "lw_down", "precip"):
        src = getattr(forcing, name)[idx_t]
        out[name] = _interp_to_model_grid(
            src, forcing.lat, forcing.lon, lat_deg, lon_deg,
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

    if grid_type not in ("latlon", "latlon_regional"):
        raise NotImplementedError(
            f"OMIP-2 applicator currently lat-lon only; got {grid_type!r}. "
            "MPAS + cube support requires edge / corner stress projection "
            "and is a follow-up commit."
        )

    lat_deg = np.degrees(np.asarray(grid.lat))
    lon_deg = np.degrees(np.asarray(grid.lon))
    forc = _sample_forcing_at_time(forcing, idx_t, lat_deg, lon_deg)
    # Diagnostic surface state (SST in Kelvin, sea-surface q from
    # Bolton saturation at SST).
    T_sfc_C = np.asarray(state.T.data, dtype=np.float64)[..., 0]
    T_sfc_K = T_sfc_C + 273.15
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
    sh_np = np.asarray(sh)
    lh_np = np.asarray(lh)
    # Net heat flux into the ocean = turbulent + radiative (SW down -
    # LW up). LW up is sigma * T_sfc^4 (Stefan-Boltzmann); for the
    # smoke path we use the canonical surface emissivity = 0.97.
    sigma_sb = float(getattr(constants, "sigma_sb", 5.67e-8))
    lw_up = 0.97 * sigma_sb * T_sfc_K ** 4
    Q_net = sh_np + lh_np + forc["sw_down"] - lw_up + forc["lw_down"]

    # Top-layer thickness.
    dz_0 = float(np.asarray(z_coord.dz_ref)[0])

    # ---- Temperature update (cell-centred) ----
    mask = np.asarray(state.land_mask.data, dtype=np.float64)
    dT_top = Q_net / (rho_0 * c_p * dz_0) * dt * mask
    T_new = np.asarray(state.T.data, dtype=np.float64).copy()
    T_new[..., 0] = T_new[..., 0] + dT_top

    # ---- Velocity update (C-grid face interpolation) ----
    # Cell-centred tau_x -> east-west face: 0.5*(left + right).
    # tau_x_np shape (n_lat, n_lon); pad zonally with periodic /
    # zero on edge cells to match the (n_lat, n_lon+1) face grid.
    u_face = np.asarray(state.u.data, dtype=np.float64).copy()
    v_face = np.asarray(state.v.data, dtype=np.float64).copy()
    n_lat, n_lon = tau_x_np.shape
    tau_x_face = np.zeros((n_lat, n_lon + 1), dtype=np.float64)
    tau_x_face[:, 1:-1] = 0.5 * (tau_x_np[:, :-1] + tau_x_np[:, 1:])
    tau_x_face[:, 0] = tau_x_np[:, 0]
    tau_x_face[:, -1] = tau_x_np[:, -1]
    u_mask = np.asarray(state.u_mask.data, dtype=np.float64)
    du_top = tau_x_face / (rho_0 * dz_0) * dt * u_mask
    u_face[..., 0] = u_face[..., 0] + du_top

    tau_y_face = np.zeros((n_lat + 1, n_lon), dtype=np.float64)
    tau_y_face[1:-1, :] = 0.5 * (tau_y_np[:-1, :] + tau_y_np[1:, :])
    tau_y_face[0, :] = tau_y_np[0, :]
    tau_y_face[-1, :] = tau_y_np[-1, :]
    v_mask = np.asarray(state.v_mask.data, dtype=np.float64)
    dv_top = tau_y_face / (rho_0 * dz_0) * dt * v_mask
    v_face[..., 0] = v_face[..., 0] + dv_top

    return state._replace(
        T=Field(jnp.asarray(T_new), name=state.T.name,
                dims=state.T.dims, units=state.T.units),
        u=Field(jnp.asarray(u_face), name=state.u.name,
                dims=state.u.dims, units=state.u.units),
        v=Field(jnp.asarray(v_face), name=state.v.name,
                dims=state.v.dims, units=state.v.units),
    )


__all__ = ["apply_omip2_surface_fluxes"]
