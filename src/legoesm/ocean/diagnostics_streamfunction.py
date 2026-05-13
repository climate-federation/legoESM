"""Streamfunction diagnostics for the lat-lon C-grid ocean.

Used by the global-overturning and tropical-OMIP progress plotters.
Both quantities are NumPy-only (consume the data already extracted from
restart files); they are not part of any hot loop, so they are kept
out of the JIT-compiled paths.

Functions
---------
moc_streamfunction
    Eulerian-mean meridional overturning ψ(lat, z) [Sv].
barotropic_streamfunction
    Depth-integrated transport ψ_bt(lat, lon) [Sv].

History
-------
The two helpers used to live inline in
``scripts/global_overturning/plot_realistic_geometry_progress.py``.
They were moved here once a second consumer (the tropical-OMIP
progress plotter) appeared, since copy-and-rename across plot scripts
is forbidden by the project's CLAUDE.md.
"""

from __future__ import annotations

import numpy as np

from legoesm import constants


def moc_streamfunction(v, h_partial, eta, H_bathy, mask, grid):
    """Eulerian-mean meridional overturning streamfunction [Sv].

    ψ(j, k) = − ∫_{z[k]}^0 V_zonal(lat[j], z') dz'

    where ``V_zonal = Σ_lon h_v · v · dx_v`` is the zonally-integrated
    thickness-weighted meridional volume flux at v-faces. Sign
    convention: positive ψ in lat-z space corresponds to clockwise
    circulation (warm rising, cold sinking).

    Parameters
    ----------
    v : ndarray, shape (n_lat+1, n_lon, nlev)
        Meridional velocity at v-faces [m/s].
    h_partial : ndarray, shape (n_lat, n_lon, nlev)
        Layer thickness on cell centres [m].
    eta : ndarray, shape (n_lat, n_lon)
        Free surface elevation [m] (currently unused — accepted for
        API symmetry with future free-surface-aware variants).
    H_bathy : ndarray, shape (n_lat, n_lon)
        Bathymetric depth [m] (currently unused; same reason).
    mask : ndarray, shape (n_lat, n_lon)
        Ocean mask (1 = ocean, 0 = land).
    grid : LatLonGrid-like
        Must expose ``radius`` (or default to Earth's radius).

    Returns
    -------
    psi : ndarray, shape (n_lat+1, nlev) [Sv]
    """
    del eta, H_bathy  # accepted for API symmetry
    n_lat_v, n_lon, _ = v.shape  # n_lat_v = n_lat + 1
    R = getattr(grid, "radius", constants.R_earth)
    # Derive v-face latitudes from grid metadata when available so
    # regional grids get the correct zonal face lengths.  Falls back
    # to the legacy global ``linspace(-π/2, π/2)`` only when the
    # grid object does not expose ``lat_v`` / ``lat`` / ``dlat``.
    # Codex iter-36 #1.
    lat_v = getattr(grid, "lat_v", None)
    if lat_v is None:
        grid_lat = getattr(grid, "lat", None)
        grid_dlat = getattr(grid, "dlat", None)
        if grid_lat is not None and grid_dlat is not None:
            # Cell centres + half-cell offset → v-face latitudes.
            lat_v = np.concatenate([
                [grid_lat[0] - 0.5 * grid_dlat],
                grid_lat + 0.5 * grid_dlat,
            ])
        else:
            lat_v = np.linspace(-np.pi / 2, np.pi / 2, n_lat_v)
    cos_lat_v = np.cos(np.asarray(lat_v))
    # Longitudinal spacing: prefer ``grid.dlon`` (correct on regional
    # grids); fall back to the global 2π/n_lon.
    dlon = getattr(grid, "dlon", 2.0 * np.pi / n_lon)
    dx_v = R * dlon * cos_lat_v[:, None]                        # (n_lat+1, 1)

    h_v = np.zeros_like(v)                                      # (n_lat+1, n_lon, nlev)
    h_v[1:-1] = 0.5 * (h_partial[:-1] + h_partial[1:])
    # Pole rows stay zero — no flux across the polar cap.

    v_mask = np.zeros((n_lat_v, n_lon))
    if n_lat_v >= 2:
        v_mask[1:-1] = mask[:-1] * mask[1:]
    v_mask = v_mask[:, :, None]                                  # (n_lat+1, n_lon, 1)

    Vh = (v * h_v * v_mask) * dx_v[:, :, None]                   # (n_lat+1, n_lon, nlev)
    V_zonal = Vh.sum(axis=1)                                     # (n_lat+1, nlev)

    # ψ(j, k) = − ∫_{z[k]}^{z=0} V_zonal · dz'
    # Level 0 is surface, level −1 is bottom; cumsum from the surface
    # downward gives the cumulative transport above each level.
    psi = -np.cumsum(V_zonal, axis=1) / 1.0e6                    # m^3/s → Sv
    return psi


def barotropic_streamfunction(u, h_partial, mask, grid):
    """Barotropic streamfunction ψ_bt(lat, lon) [Sv].

    ψ_bt(j, i) = − ∫_{south_wall}^{lat[j]} U_zonal(lat', i) · dy

    where ``U_zonal = Σ_z h_u · u`` is the depth-integrated zonal
    transport at u-faces. Sign convention: positive ψ_bt corresponds
    to clockwise circulation as viewed from above.

    Parameters
    ----------
    u : ndarray, shape (n_lat, n_lon+1, nlev)
        Zonal velocity at u-faces [m/s] (periodic in longitude — the
        last column is the wrap of the first).
    h_partial : ndarray, shape (n_lat, n_lon, nlev)
        Layer thickness on cell centres [m].
    mask : ndarray, shape (n_lat, n_lon)
        Ocean mask.
    grid : LatLonGrid-like
        Must expose ``radius`` (default Earth).

    Returns
    -------
    psi_bt : ndarray, shape (n_lat, n_lon) [Sv]
        Cell-centre values (the wrap column is dropped).
    """
    n_lat, n_lon_u, _ = u.shape
    n_lon = n_lon_u - 1
    R = getattr(grid, "radius", constants.R_earth)
    # Use ``grid.dlat`` when available so regional grids integrate
    # transport with their actual meridional spacing rather than the
    # global ``π / n_lat``.  Codex iter-36 #2.
    dlat = getattr(grid, "dlat", np.pi / n_lat)
    dy = R * dlat                                                # uniform

    # MOM6/MITgcm "min-rule" thickness at u-faces:
    h_E = h_partial
    h_W = np.roll(h_partial, 1, axis=1)
    h_u_int = np.minimum(h_E, h_W)                               # (n_lat, n_lon, nlev)
    h_u = np.concatenate([h_u_int, h_u_int[:, 0:1, :]], axis=1)  # periodic wrap

    mask_E = mask
    mask_W = np.roll(mask, 1, axis=1)
    u_mask_int = (mask_E * mask_W) > 0.5
    u_mask = np.concatenate([u_mask_int, u_mask_int[:, 0:1]], axis=1)

    U_dz = np.sum(u * h_u, axis=-1) * u_mask                     # (n_lat, n_lon+1)
    psi_bt = -np.cumsum(U_dz, axis=0) * dy / 1.0e6               # (n_lat, n_lon+1) [Sv]

    return psi_bt[:, :-1]
