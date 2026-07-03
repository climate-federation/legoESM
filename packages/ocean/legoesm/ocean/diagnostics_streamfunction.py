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
``scripts/run/global_overturning/plot_realistic_geometry_progress.py``.
They were moved here once a second consumer (the tropical-OMIP
progress plotter) appeared, since copy-and-rename across plot scripts
is forbidden by the project's CLAUDE.md.
"""

from __future__ import annotations

import numpy as np

from legoesm import constants


def partial_cell_thickness(H_bathy, dz_ref):
    """Partial-cell layer thickness (n_lat, n_lon, nlev) [m] reconstructed
    from the bottom depth + z* reference thicknesses: full ``dz_ref`` above
    the floor, a clipped bottom cell, zero below. The eta-driven z* stretch
    (~0.1% of the column) is neglected — intended for depth-integrated
    transport diagnostics from snapshots that store ``H_bathy`` but not the
    instantaneous layer thicknesses.
    """
    dz_ref = np.asarray(dz_ref)
    z_bot = np.cumsum(dz_ref)                       # (nlev,) interface depths
    z_top = z_bot - dz_ref
    H = np.asarray(H_bathy)[..., None]              # (n_lat, n_lon, 1)
    return np.clip(np.minimum(z_bot, H) - z_top, 0.0, dz_ref)


def _v_face_geometry(v, h_partial, mask, grid):
    """Shared v-face geometry for the meridional diagnostics (MOC + MHT).

    Returns ``(dx_v, h_v, v_mask, lat_v)`` for v-faces of shape ``(n_lat+1, ...)``:
      * ``dx_v``  : zonal face width [m], shape ``(n_lat+1, n_lon)`` (tripole, from
        ``grid.dx_v``) or ``(n_lat+1, 1)`` (regular ``R·dlon·cos(lat_v)``).
      * ``h_v``   : centred v-face thickness [m], ``(n_lat+1, n_lon, nlev)`` (pole
        rows zero — no flux across the cap).
      * ``v_mask``: ``(n_lat+1, n_lon, 1)`` = ``mask[j-1]·mask[j]`` (poles zero).
      * ``lat_v`` : v-face latitudes [rad], ``(n_lat+1,)``.
    Factored out so ``moc_streamfunction`` and ``meridional_heat_transport`` share
    one geometry (no duplicated v-face metric code).
    """
    n_lat_v, n_lon, _ = v.shape  # n_lat_v = n_lat + 1
    R = getattr(grid, "radius", constants.R_earth)
    lat_v = getattr(grid, "lat_v", None)
    if lat_v is None:
        grid_lat = getattr(grid, "lat", None)
        grid_dlat = getattr(grid, "dlat", None)
        if grid_lat is not None and grid_dlat is not None:
            lat_v = np.concatenate([
                [grid_lat[0] - 0.5 * grid_dlat],
                grid_lat + 0.5 * grid_dlat,
            ])
        else:
            lat_v = np.linspace(-np.pi / 2, np.pi / 2, n_lat_v)
    lat_v = np.asarray(lat_v)
    cos_lat_v = np.cos(lat_v)
    # Longitudinal spacing: prefer ``grid.dlon``; tripole sets ``dlon=0`` sentinel
    # and exposes the per-v-face physical width ``grid.dx_v`` [m] (curvilinear).
    dlon = getattr(grid, "dlon", 2.0 * np.pi / n_lon)
    dx_v_metric = getattr(grid, "dx_v", None)
    if (not dlon) and dx_v_metric is not None and np.asarray(dx_v_metric).size > 1:
        dx_v = np.asarray(dx_v_metric)
        if dx_v.ndim == 1:
            dx_v = dx_v[:, None]
    else:
        if not dlon:
            dlon = 2.0 * np.pi / n_lon
        dx_v = R * dlon * cos_lat_v[:, None]

    h_v = np.zeros_like(v)
    h_v[1:-1] = 0.5 * (h_partial[:-1] + h_partial[1:])

    v_mask = np.zeros((n_lat_v, n_lon))
    if n_lat_v >= 2:
        v_mask[1:-1] = mask[:-1] * mask[1:]
    v_mask = v_mask[:, :, None]
    return dx_v, h_v, v_mask, lat_v


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
    dx_v, h_v, v_mask, _ = _v_face_geometry(v, h_partial, mask, grid)

    Vh = (v * h_v * v_mask) * dx_v[:, :, None]                   # (n_lat+1, n_lon, nlev)
    V_zonal = Vh.sum(axis=1)                                     # (n_lat+1, nlev)

    # ψ(j, k) = − ∫_{z[k]}^{z=0} V_zonal · dz'
    # Level 0 is surface, level −1 is bottom; cumsum from the surface
    # downward gives the cumulative transport above each level.
    psi = -np.cumsum(V_zonal, axis=1) / 1.0e6                    # m^3/s → Sv
    return psi


def meridional_heat_transport(v, theta, h_partial, mask, grid,
                              *, rho0=constants.rho_ocean, cp=constants.c_sw):
    """Global meridional ocean heat transport MHT(lat) [PW] on a C-grid.

    MHT(j) = ρ0·cp · Σ_x Σ_z v · θ_v · h_v · dx_v   [W]   (θ_v in degC at v-faces)

    Reuses the shared v-face geometry (:func:`_v_face_geometry`); θ is averaged
    from cell centres onto the v-faces.  The full zonal integral at a latitude has
    ~zero net mass flux, so degC is the conventional (reference-independent)
    choice — matching the offline NEMO reader ``nemo_transports.mht_core``.

    Parameters
    ----------
    v : ndarray ``(n_lat+1, n_lon, nlev)`` — meridional velocity at v-faces [m/s].
    theta : ndarray ``(n_lat, n_lon, nlev)`` — potential temperature [degC].
    h_partial, mask, grid : as in :func:`moc_streamfunction`.
    rho0, cp : reference seawater density / heat capacity (default
        ``constants.rho_ocean`` / ``constants.c_sw``).

    Returns
    -------
    mht_PW : ndarray ``(n_lat+1,)`` — MHT at each v-row [PW].
    lat_v_deg : ndarray ``(n_lat+1,)`` — v-face latitudes [°].
    """
    dx_v, h_v, v_mask, lat_v = _v_face_geometry(v, h_partial, mask, grid)
    theta_v = np.zeros_like(v)
    theta_v[1:-1] = 0.5 * (theta[:-1] + theta[1:])              # cells -> v-faces
    Hf = (rho0 * cp) * (v * h_v * v_mask * theta_v) * dx_v[:, :, None]
    # A masked-land NaN tracer would poison a whole row (0*NaN=NaN after v_mask);
    # zero non-finite contributions so one land cell can't NaN the MHT curve.
    Hf = np.nan_to_num(Hf, nan=0.0, posinf=0.0, neginf=0.0)
    mht_W = Hf.sum(axis=(1, 2))                                 # (n_lat+1,) [W]
    return mht_W / 1.0e15, np.degrees(lat_v)


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
    n_lon_u - 1
    R = getattr(grid, "radius", constants.R_earth)
    # Cell-row meridional extent — prefer ``grid.dy`` (1D array,
    # Mercator-safe) but fall back to a uniform ``R * dlat`` if absent
    # (lightweight grid proxies in tests sometimes lack ``dy``).
    if hasattr(grid, "dy"):
        dy = np.asarray(grid.dy) * 0.5                            # (n_lat,)
    else:
        dlat = getattr(grid, "dlat", np.pi / n_lat)
        dy = np.full((n_lat,), R * dlat)

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
    # Per-row dy weighting before cumsum so non-uniform grids integrate
    # the correct meridional transport.
    psi_bt = -np.cumsum(U_dz * dy[:, None], axis=0) / 1.0e6      # (n_lat, n_lon+1) [Sv]

    return psi_bt[:, :-1]
