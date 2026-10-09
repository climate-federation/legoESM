"""Topography generators, real topography loading, and land-mask utilities.

Provides functions to generate idealized topography fields on a
cubed-sphere grid (6, n, n) or Gaussian grid (n_lat, n_lon),
load real topography from NetCDF files (ETOPO, GEBCO, etc.),
and derive surface geopotential and land/ocean masks.

All topography generators return surface elevation z_s in meters.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.halo import pad_halo_local

logger = logging.getLogger(__name__)

# Elevation-magnitude sanity threshold [m] for the geopotential-vs-meters
# "double-g" trap: no Earth surface elevation exceeds ~8850 m (Everest) and no
# ocean depth exceeds ~11000 m (Mariana), but the GEOPOTENTIAL of Everest is
# ~86800 m²/s².  An "elevation" field whose magnitude exceeds this threshold is
# almost certainly geopotential [m²/s²] mistakenly passed as elevation [m] —
# multiplying it by g again in load_real_topography would silently inflate
# phis ~9.8×.
_MAX_PLAUSIBLE_ELEV_M = 12000.0


def _grid_lat_lon_2d(grid):
    """Per-cell (lat, lon) in the grid's native horizontal layout [rad].

    Cubed-sphere exposes 2-D ``(6, n, n)`` ``lat``/``lon`` directly; the lat-lon
    grid stores 1-D axes (``lat`` shape ``(n_lat,)``, ``lon`` shape ``(n_lon,)``)
    plus 2-D meshes ``lat2d``/``lon2d``.  The analytic-mountain generators below
    need per-cell 2-D fields, so prefer ``lat2d``/``lon2d`` when present and fall
    back to ``lat``/``lon`` (already 2-D on the cubed sphere, 1-D per-cell on
    unstructured meshes).  Without this, ``grid.lat - grid.lon`` style broadcasts
    fail on lat-lon as ``(n_lat,) + (n_lon,)``.
    """
    lat = getattr(grid, "lat2d", None)
    lon = getattr(grid, "lon2d", None)
    return (grid.lat if lat is None else lat,
            grid.lon if lon is None else lon)


def gaussian_mountain(
    grid: CubedSphereGrid,
    h0: float = 2500.0,
    lat0: float = 40.0 * jnp.pi / 180.0,
    lon0: float = 255.0 * jnp.pi / 180.0,
    sigma_lat: float = 15.0 * jnp.pi / 180.0,
    sigma_lon: float = 15.0 * jnp.pi / 180.0,
) -> jnp.ndarray:
    """Generate a single Gaussian mountain.

    z_s = h0 * exp(-((lat-lat0)^2/(2*sigma_lat^2) + (lon-lon0)^2/(2*sigma_lon^2)))

    Parameters
    ----------
    grid : CubedSphereGrid
        Cubed-sphere grid.
    h0 : float
        Peak height [m]. Default 2500 (Rocky Mountains scale).
    lat0 : float
        Center latitude [rad]. Default 40 deg N.
    lon0 : float
        Center longitude [rad]. Default 255 deg E.
    sigma_lat : float
        Latitudinal half-width [rad]. Default 15 deg.
    sigma_lon : float
        Longitudinal half-width [rad]. Default 15 deg.

    Returns
    -------
    jnp.ndarray
        Surface elevation z_s, shape (6, n, n).
    """
    lat, lon = _grid_lat_lon_2d(grid)

    dlat = lat - lat0

    # Periodic longitude wrapping to [-pi, pi]
    dlon = jnp.mod(lon - lon0 + jnp.pi, 2.0 * jnp.pi) - jnp.pi

    z_s = h0 * jnp.exp(
        -(dlat ** 2 / (2.0 * sigma_lat ** 2) + dlon ** 2 / (2.0 * sigma_lon ** 2))
    )
    return z_s


def zonal_ridge(
    grid: CubedSphereGrid,
    h0: float = 2500.0,
    lat0: float = 45.0 * jnp.pi / 180.0,
    sigma_lat: float = 10.0 * jnp.pi / 180.0,
) -> jnp.ndarray:
    """Generate a zonally symmetric ridge.

    z_s = h0 * exp(-(lat-lat0)^2/(2*sigma_lat^2))

    Parameters
    ----------
    grid : CubedSphereGrid
        Cubed-sphere grid.
    h0 : float
        Peak height [m].
    lat0 : float
        Ridge center latitude [rad]. Default 45 deg N.
    sigma_lat : float
        Latitudinal half-width [rad]. Default 10 deg.

    Returns
    -------
    jnp.ndarray
        Surface elevation z_s, shape (6, n, n).
    """
    lat, _ = _grid_lat_lon_2d(grid)
    dlat = lat - lat0
    return h0 * jnp.exp(-(dlat ** 2) / (2.0 * sigma_lat ** 2))


def schaer_mountain(
    grid: CubedSphereGrid,
    h0: float = 2000.0,
    lat0: float = 0.0,
    lon0: float = jnp.pi,
    half_width: float = 72.0e3,
) -> jnp.ndarray:
    """Generate Schaer-type cosine-bell mountain topography.

    Uses great-circle distance: z_s = h0/2 * (1 + cos(pi*r/a)) for r < a,
    where r is the distance from the mountain center and a is the half-width.

    Parameters
    ----------
    grid : CubedSphereGrid
        Cubed-sphere grid.
    h0 : float
        Peak height [m].
    lat0 : float
        Mountain center latitude [rad].
    lon0 : float
        Mountain center longitude [rad].
    half_width : float
        Mountain half-width [m].

    Returns
    -------
    jnp.ndarray
        Surface elevation z_s, shape (6, n, n).
    """
    lat, lon = _grid_lat_lon_2d(grid)

    # Great-circle distance via haversine
    dlat = lat - lat0
    dlon = lon - lon0
    a = (
        jnp.sin(dlat / 2.0) ** 2
        + jnp.cos(lat) * jnp.cos(lat0) * jnp.sin(dlon / 2.0) ** 2
    )
    angular_dist = 2.0 * jnp.arcsin(jnp.sqrt(jnp.clip(a, 0.0, 1.0)))
    dist = angular_dist * grid.radius

    # Cosine bell: h0/2 * (1 + cos(pi*r/a)) for r < a, 0 otherwise
    z_s = jnp.where(
        dist < half_width,
        0.5 * h0 * (1.0 + jnp.cos(jnp.pi * dist / half_width)),
        0.0,
    )
    return z_s


def land_mask_from_topography(
    z_s: jnp.ndarray,
    threshold: float = 0.0,
) -> jnp.ndarray:
    """Derive a land/ocean mask from surface elevation.

    Parameters
    ----------
    z_s : jnp.ndarray
        Surface elevation [m], shape (6, n, n).
    threshold : float
        Elevation threshold [m]. Grid cells with z_s > threshold are land.

    Returns
    -------
    jnp.ndarray
        Land fraction f_land (1.0 = land, 0.0 = ocean), shape (6, n, n).
    """
    return jnp.where(z_s > threshold, 1.0, 0.0)


def phis_from_topography(z_s: jnp.ndarray) -> jnp.ndarray:
    """Convert surface elevation to surface geopotential.

    phis = g * z_s

    Parameters
    ----------
    z_s : jnp.ndarray
        Surface elevation [m].

    Returns
    -------
    jnp.ndarray
        Surface geopotential [m^2/s^2], same shape as z_s.
    """
    return constants.g * z_s


# ==============================================================================
# Topography configuration
# ==============================================================================


class TopographyConfig(NamedTuple):
    """Configuration for topography loading.

    Parameters
    ----------
    source : str
        Topography source: "flat", "gaussian", "file".
    path : str
        Path to topography NetCDF file (for source="file").
    elev_var : str
        Elevation variable name in NetCDF.
    lat_var : str
        Latitude variable name in NetCDF.
    lon_var : str
        Longitude variable name in NetCDF.
    smoothing_passes : int
        Passes of the masked (zero_ocean) flux-form diffusion on the grid.
    land_mask_path : str
        Optional path to a separate land-sea-mask NetCDF file (e.g. CMIP6
        ``sftlf`` or an ERA5 ``lsm`` invariant).  When set, the land
        fraction is taken from this file instead of being derived from
        ``elevation > 0`` — which avoids misclassifying below-sea-level
        land (Caspian/Dead Sea/Netherlands) as ocean.
    land_mask_var : str
        Variable name in ``land_mask_path``.  Empty → auto-detect among
        ``sftlf``, ``lsm``, ``land_sea_mask``, ``land``,
        ``land_area_fraction``.
    """
    source: str = "flat"
    path: str = ""
    elev_var: str = ""
    lat_var: str = ""
    lon_var: str = ""
    smoothing_passes: int = 4
    land_mask_path: str = ""
    land_mask_var: str = ""


# ==============================================================================
# Real topography loading
# ==============================================================================


def _detect_variables(ds) -> tuple[str, str, str]:
    """Auto-detect elevation, latitude, and longitude variable names."""
    all_vars = set(ds.data_vars.keys()) | set(ds.coords.keys())

    # Elevation: try common names
    elev_candidates = ["z", "elevation", "topo", "Band1", "ROSE",
                       "bedrock_topography", "surface_elevation"]
    elev_var = None
    for c in elev_candidates:
        if c in all_vars:
            elev_var = c
            break
    if elev_var is None:
        # Use first data variable
        data_vars = list(ds.data_vars.keys())
        if data_vars:
            elev_var = data_vars[0]
        else:
            raise KeyError(
                f"Cannot detect elevation variable. "
                f"Available: {sorted(all_vars)}"
            )

    # Latitude
    lat_candidates = ["lat", "latitude", "y", "Y"]
    lat_var = None
    for c in lat_candidates:
        if c in all_vars:
            lat_var = c
            break
    if lat_var is None:
        raise KeyError(
            f"Cannot detect latitude variable. Available: {sorted(all_vars)}"
        )

    # Longitude
    lon_candidates = ["lon", "longitude", "x", "X"]
    lon_var = None
    for c in lon_candidates:
        if c in all_vars:
            lon_var = c
            break
    if lon_var is None:
        raise KeyError(
            f"Cannot detect longitude variable. Available: {sorted(all_vars)}"
        )

    return elev_var, lat_var, lon_var


def _drop_duplicate_periodic_lon(lon_src, data):
    """A file carrying both periodic ends (-180 and 180, or 0 and 360) maps
    them to one longitude after the callers' ``% 360``: scipy refuses the
    repeated point, and the elevation binning would count that meridian
    twice (two source columns averaged into one cell, and a ``2 pi / n_lon``
    source width that includes the duplicate).  Keep the first occurrence:
    the callers' STABLE sort puts the file's first column ahead of its end
    column.  ``lon_src`` must already be sorted ascending."""
    lon_src = np.asarray(lon_src)
    keep = np.concatenate([[True], np.diff(lon_src) != 0.0])
    if not keep.all():
        logger.info("Dropping %d duplicated periodic longitude column(s) at "
                    "%s deg (file's end column)", int((~keep).sum()),
                    lon_src[~keep])
        lon_src = lon_src[keep]
        data = np.asarray(data)[:, keep]
    return lon_src, data


def _build_latlon_interpolator(
    lat_src: np.ndarray,
    lon_src: np.ndarray,
    data: np.ndarray,
):
    """Bilinear interpolator with longitudinal wrap AND polar edge padding.

    Used by the land-mask-file loader (:func:`_load_land_fraction_file` via
    ``_regrid_to_target``); the elevation product bins by cell ownership
    (:func:`bin_latlon_to_cells`) instead.

    * Longitude is wrapped by one column on each side for periodicity.
    * The latitude axis is padded to exactly ±90° by EDGE REPLICATION
      (prepend/append a copy of the first/last latitude row).  Without this,
      ``bounds_error=False, fill_value=0.0`` silently assigns 0 (ocean /
      sea level) to target cells POLEWARD of the source grid's outermost
      latitude CENTER — e.g. a 1° sftlf spanning ±89.5° left cubed-sphere or
      Gaussian cells near ±90° misclassified as f_land=0 open ocean over
      Antarctica.  With the padding, pole-adjacent targets get the nearest
      real source value instead.
    """
    from scipy.interpolate import RegularGridInterpolator

    lat_src = np.asarray(lat_src)
    lon_src, data = _drop_duplicate_periodic_lon(lon_src, data)
    # Longitudinal wrap (unchanged behavior).
    lon_wrapped = np.concatenate([
        lon_src[-1:] - 360.0, lon_src, lon_src[:1] + 360.0
    ])
    data_wrapped = np.concatenate([
        data[:, -1:], data, data[:, :1]
    ], axis=1)

    # Polar edge replication (lat_src is ascending by the callers' contract).
    if lat_src[0] > -90.0:
        lat_src = np.concatenate([[-90.0], lat_src])
        data_wrapped = np.concatenate([data_wrapped[:1], data_wrapped], axis=0)
    if lat_src[-1] < 90.0:
        lat_src = np.concatenate([lat_src, [90.0]])
        data_wrapped = np.concatenate([data_wrapped, data_wrapped[-1:]], axis=0)

    return RegularGridInterpolator(
        (lat_src, lon_wrapped), data_wrapped,
        method="linear", bounds_error=False, fill_value=0.0,
    )


def _regrid_to_target(
    lat_src: np.ndarray,
    lon_src: np.ndarray,
    elev_data: np.ndarray,
    target_lat_deg: np.ndarray,
    target_lon_deg: np.ndarray,
) -> np.ndarray:
    """Regrid elevation data from regular lat-lon to target grid points.

    Uses bilinear interpolation with longitude wrapping for periodicity
    and polar edge replication (see ``_build_latlon_interpolator``).

    Parameters
    ----------
    lat_src : (nlat_src,)
        Source latitude in degrees.
    lon_src : (nlon_src,)
        Source longitude in degrees, in [0, 360).
    elev_data : (nlat_src, nlon_src)
        Source elevation in meters.
    target_lat_deg : array
        Target latitude in degrees (any shape).
    target_lon_deg : array
        Target longitude in degrees (any shape).

    Returns
    -------
    np.ndarray
        Regridded elevation, same shape as target_lat_deg.
    """
    interp = _build_latlon_interpolator(lat_src, lon_src, elev_data)

    target_shape = target_lat_deg.shape
    points = np.stack([
        target_lat_deg.ravel(),
        target_lon_deg.ravel(),
    ], axis=-1)

    result = interp(points).reshape(target_shape)
    return result


def _load_land_fraction_file(
    path: str,
    var_name: str,
    target_lat_deg: np.ndarray,
    target_lon_deg: np.ndarray,
) -> np.ndarray:
    """Load a land-sea-mask file and regrid (bilinear) to the target grid.

    Accepts any regular lat-lon NetCDF land-fraction field — CMIP6
    ``sftlf`` (percent), ERA5 ``lsm`` (fraction), etc.  Percent fields
    are auto-detected (max value > 1.5) and rescaled to [0, 1].

    Parameters
    ----------
    path : str
        NetCDF file containing the land-sea mask.
    var_name : str
        Mask variable name; empty → auto-detect.
    target_lat_deg, target_lon_deg : np.ndarray
        Target grid centers in degrees (any shape).

    Returns
    -------
    np.ndarray
        Land fraction in [0, 1], same shape as ``target_lat_deg``.
    """
    import xarray as xr

    ds = xr.open_dataset(path)
    try:
        all_names = set(ds.data_vars) | set(ds.coords)
        if var_name:
            mask_var = var_name
        else:
            mask_var = None
            for c in ("sftlf", "lsm", "land_sea_mask", "land",
                      "land_area_fraction", "LSM"):
                if c in all_names:
                    mask_var = c
                    break
            if mask_var is None:
                data_vars = list(ds.data_vars)
                if not data_vars:
                    raise KeyError(
                        f"No data variables in land-mask file {path}"
                    )
                mask_var = data_vars[0]

        lat_var = None
        for c in ("lat", "latitude", "y", "Y"):
            if c in all_names:
                lat_var = c
                break
        lon_var = None
        for c in ("lon", "longitude", "x", "X"):
            if c in all_names:
                lon_var = c
                break
        if lat_var is None or lon_var is None:
            raise KeyError(
                f"Cannot detect lat/lon in land-mask file {path}; "
                f"available: {sorted(all_names)}"
            )

        lat_src = ds[lat_var].values.astype(np.float64)
        lon_src = ds[lon_var].values.astype(np.float64)
        mask_data = ds[mask_var].values.astype(np.float64)
    finally:
        ds.close()

    while mask_data.ndim > 2:
        mask_data = mask_data[0]

    # Longitude in [0, 360), ascending
    lon_src = lon_src % 360.0
    lon_order = np.argsort(lon_src, kind="stable")
    lon_src = lon_src[lon_order]
    mask_data = mask_data[:, lon_order]

    # Latitude ascending
    if lat_src[0] > lat_src[-1]:
        lat_src = lat_src[::-1]
        mask_data = mask_data[::-1, :]

    mask_data = np.where(np.isnan(mask_data), 0.0, mask_data)

    # Percent → fraction (CMIP6 sftlf is 0-100; ERA5 lsm is 0-1)
    if np.nanmax(mask_data) > 1.5:
        mask_data = mask_data / 100.0

    f_land = _regrid_to_target(
        lat_src, lon_src, mask_data, target_lat_deg, target_lon_deg
    )
    return np.clip(f_land, 0.0, 1.0)


# ==============================================================================
# ONE terrain product per grid (decision C, 2026-10-02)
# ==============================================================================
# GFDL fv_surf_map.F90 is the reference: cell-mean elevation and land
# fraction from ONE dataset, then grid-native filtering with zero_ocean --
# every diffusive edge flux multiplied by max(0, min(oro_l, oro_r)), oro the
# fractional land cover (:768-777, :896-905).  Here the elevation is binned
# by nearest cell centre on the unit sphere (exact Voronoi ownership on an
# MPAS mesh, the same ownership rule conservative_regrid_cubedsphere uses),
# with each 0.25-degree source cell weighted by its own spherical area.

_SMOOTH_K_PER_PASS = 0.1   # today's per-pass strength: 0.5 blend x 1/5 mean


def _source_cell_areas(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Unit-sphere area of every cell of a regular lat-lon source grid
    (``(n_lat, n_lon)``): cell edges at the midpoints, clamped to the poles,
    longitude periodic."""
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    mid = 0.5 * (lat[1:] + lat[:-1])
    pole_first, pole_last = (-0.5 * np.pi, 0.5 * np.pi) if lat[0] <= lat[-1] else (
        0.5 * np.pi, -0.5 * np.pi)
    edges = np.concatenate([[pole_first], mid, [pole_last]])
    edges = np.clip(edges, -0.5 * np.pi, 0.5 * np.pi)
    dlon = 2.0 * np.pi / int(np.asarray(lon_deg).shape[0])
    return np.abs(np.sin(edges[1:]) - np.sin(edges[:-1]))[:, None] * dlon * np.ones(
        (1, int(np.asarray(lon_deg).shape[0])))


def _xyz(lat_rad, lon_rad):
    lat = np.asarray(lat_rad, dtype=np.float64).ravel()
    lon = np.asarray(lon_rad, dtype=np.float64).ravel()
    cl = np.cos(lat)
    return np.stack([cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)], axis=-1)


_EDGE_TOL = 1e-12


def _inside_quads(p, quads, centres):
    """``p`` ``(m, 3)`` points, ``quads`` ``(m, 4, 3)`` spherically convex
    quads with great-circle edges (gnomonic cells: straight lines on the
    face plane, any orientation), ``centres`` ``(m, 3)`` a point strictly
    inside each: True where ``p`` is on the centre's side of all 4 edges
    (a point on an edge belongs to both cells; the first claim wins).
    A centre ON an edge plane would make that edge unconstrained: error."""
    ok = np.ones(p.shape[0], dtype=bool)
    for k in range(4):
        nrm = np.cross(quads[:, k, :], quads[:, (k + 1) % 4, :])
        side_c = np.einsum("ij,ij->i", nrm, centres)
        if np.any(np.abs(side_c) <= _EDGE_TOL):
            raise ValueError("_inside_quads: a cell centre lies on one of its edges")
        side_p = np.einsum("ij,ij->i", nrm, p)
        ok &= side_p * np.sign(side_c) >= -_EDGE_TOL
    return ok


def owner_by_quads(xyz_s, xyz_c, corners_xyz, nb):
    """Exact polygon ownership for cells with great-circle edges (the
    cubed-sphere's): the nearest centre first, then that cell's quad, then
    its neighbours' quads (``nb`` ``(K, n)``).  Returns ``(owner,
    n_unclaimed)``: the caller refuses any unclaimed point (the corner
    arrays do not describe the grid, or the source is coarser than a
    cell's corner-neighbourhood)."""
    from scipy.spatial import cKDTree
    owner = cKDTree(xyz_c).query(xyz_s, k=1)[1].astype(np.int64)
    inside = _inside_quads(xyz_s, corners_xyz[owner], xyz_c[owner])
    todo = np.nonzero(~inside)[0]
    for k in range(nb.shape[0]):
        if todo.size == 0:
            break
        cand = nb[k, owner[todo]]
        valid = cand >= 0
        hit = np.zeros(todo.size, dtype=bool)
        hit[valid] = _inside_quads(xyz_s[todo[valid]], corners_xyz[cand[valid]],
                                   xyz_c[cand[valid]])
        owner[todo[hit]] = cand[hit]
        todo = todo[~hit]
    return owner, int(todo.size)


def owner_by_boxes(lat_s, lon_s, lat_c_1d, lon_c_1d, lat_edges_1d=None):
    """Exact ownership on a structured lat-lon grid: latitude edges
    ``lat_edges_1d`` ``(n_lat+1,)`` when the grid defines them (Gaussian:
    the quadrature-weight edges that make the boxes equal ``grid_area``),
    else the midpoints of the centres; poles clamped, longitude periodic
    with equal spacing."""
    lat_c = np.asarray(lat_c_1d, dtype=np.float64)
    lon_c = np.asarray(lon_c_1d, dtype=np.float64)
    asc = lat_c[0] <= lat_c[-1]
    la = lat_c if asc else lat_c[::-1]
    if lat_edges_1d is None:
        lat_e = np.concatenate([[-0.5 * np.pi], 0.5 * (la[1:] + la[:-1]), [0.5 * np.pi]])
    else:
        lat_e = np.asarray(lat_edges_1d, dtype=np.float64)
        lat_e = lat_e if lat_e[0] <= lat_e[-1] else lat_e[::-1]
        if lat_e.shape != (la.size + 1,) or not (np.diff(lat_e) > 0).all():
            raise ValueError("owner_by_boxes: lat edges must be (n_lat+1,) and increasing")
    dlon = 2.0 * np.pi / lon_c.size
    # absolute tolerance: float32 grid coordinates carry ~4e-7 rad of
    # rounding whatever the spacing (a relative test on dlon rejects fine grids)
    if not np.allclose(lon_c, lon_c[0] + dlon * np.arange(lon_c.size), rtol=0.0, atol=1e-5):
        raise ValueError("owner_by_boxes: longitudes must be equally spaced over 2 pi "
                         "without a repeated end point")
    lon_e0 = lon_c[0] - 0.5 * dlon
    j = np.clip(np.searchsorted(lat_e, lat_s, side="right") - 1, 0, la.size - 1)
    if not asc:
        j = la.size - 1 - j
    i = np.floor(np.mod(lon_s - lon_e0, 2.0 * np.pi) / dlon).astype(np.int64) % lon_c.size
    return (j * lon_c.size + i).astype(np.int64)


def bin_latlon_to_cells(lat_src_deg, lon_src_deg, fields, cell_lat_rad, cell_lon_rad,
                        cell_area=None, *, corners_xyz=None, nb=None, axes_1d=None):
    """Area-weighted cell means of regular lat-lon ``fields`` (each
    ``(n_lat, n_lon)``) on the cells of a grid: every source cell belongs
    to ONE cell and contributes its own spherical area (first order in the
    source resolution).  Ownership: ``corners_xyz`` ``(n_cells, 4, 3)`` +
    ``nb`` -> exact spherical quads (cubed-sphere); ``axes_1d`` ``(lat_c,
    lon_c[, lat_edges])`` radians -> exact lat-lon boxes; neither -> the NEAREST centre
    (exact for a Voronoi mesh).  Returns a list of ``(n_cells,)`` arrays.
    A cell no source cell lands in is an error (the source must be finer);
    with ``cell_area`` the assigned source area is checked against the
    cell's own area (ratio in [0.5, 2])."""
    from scipy.spatial import cKDTree
    lat_s = np.deg2rad(np.asarray(lat_src_deg, dtype=np.float64))
    lon_s = np.deg2rad(np.asarray(lon_src_deg, dtype=np.float64))
    lon2, lat2 = np.meshgrid(lon_s, lat_s)
    xyz_s = _xyz(lat2.ravel(), lon2.ravel())
    xyz_c = _xyz(cell_lat_rad, cell_lon_rad)
    n = xyz_c.shape[0]
    if corners_xyz is not None:
        owner, n_fb = owner_by_quads(xyz_s, xyz_c, np.asarray(corners_xyz), np.asarray(nb))
        if n_fb > 0:
            raise ValueError(f"bin_latlon_to_cells: {n_fb} source cells inside no quad -- "
                             "the corner arrays do not describe this grid")
    elif axes_1d is not None:
        owner = owner_by_boxes(lat2.ravel(), lon2.ravel(), *axes_1d)
    else:
        owner = cKDTree(xyz_c).query(xyz_s, k=1)[1].astype(np.int64)
    w = _source_cell_areas(lat_src_deg, lon_src_deg).ravel()
    denom = np.bincount(owner, weights=w, minlength=n)
    if np.any(denom <= 0.0):
        raise ValueError(
            f"bin_latlon_to_cells: {int(np.sum(denom <= 0.0))} target cells received "
            "no source cell -- the source grid is coarser than the target")
    if cell_area is not None:
        ca = np.asarray(cell_area, dtype=np.float64).ravel()
        ratio = (denom / denom.sum()) / (ca / ca.sum())
        if ratio.min() < 0.5 or ratio.max() > 2.0:
            raise ValueError(
                "bin_latlon_to_cells: assigned source area / cell area in "
                f"[{ratio.min():.2f}, {ratio.max():.2f}] -- the source is too coarse "
                "or the cell centres do not describe this grid")
    return [np.bincount(owner, weights=w * np.asarray(f, dtype=np.float64).ravel(),
                        minlength=n) / denom for f in fields]


def _neighbour_table(grid):
    """``(nb, area)`` -- ``nb`` ``(K, n_cells)`` int neighbour indices (``-1``
    unused), ``area`` ``(n_cells,)`` -- for a cubed-sphere ``(6, n, n)`` grid
    (cross-face halo), a structured lat-lon ``(n_lat, n_lon)`` grid
    (periodic longitude, no neighbour across a pole) or a Voronoi mesh
    (``cellsOnCell``); ``None`` for a mesh without neighbour information
    (the duo column mesh: its terrain is filtered on the duo faces by
    ``terrain_filter_duo``)."""
    coc = getattr(grid, "cellsOnCell", None)
    if coc is not None:
        return np.asarray(coc, dtype=np.int64), np.asarray(grid.grid_area, dtype=np.float64)
    lat = np.asarray(grid.grid_lat)
    if lat.ndim == 3:
        six, n, _ = lat.shape
        ids = jnp.asarray(np.arange(six * n * n, dtype=np.float64).reshape(six, n, n))
        p = np.asarray(pad_halo_local(ids, None))
        nb = np.stack([p[:, :-2, 1:-1], p[:, 2:, 1:-1], p[:, 1:-1, :-2], p[:, 1:-1, 2:]])
        nb = nb.reshape(4, -1)
        # the pad must COPY ids (an interpolating pad would average two ids
        # at a cube corner); the relation must be symmetric (conservation
        # rests on it) and each cell must have 4 distinct neighbours
        if not np.all(nb == np.rint(nb)):
            raise ValueError("_neighbour_table: the cube halo pad interpolated cell ids")
        nb = nb.astype(np.int64)
        n_cells = six * n * n
        pairs = {(i, int(j)) for i in range(n_cells) for j in nb[:, i]}
        if len(pairs) != 4 * n_cells or any((j, i) not in pairs for i, j in pairs):
            raise ValueError("_neighbour_table: cube neighbour relation is not "
                             "symmetric / 4-regular")
        return nb, np.asarray(grid.area, dtype=np.float64).reshape(-1)
    if lat.ndim == 2:
        n_lat, n_lon = lat.shape
        ids = np.arange(n_lat * n_lon).reshape(n_lat, n_lon)
        south = np.where(np.arange(n_lat)[:, None] > 0, np.roll(ids, 1, axis=0), -1)
        north = np.where(np.arange(n_lat)[:, None] < n_lat - 1, np.roll(ids, -1, axis=0), -1)
        nb = np.stack([south, north, np.roll(ids, 1, axis=1), np.roll(ids, -1, axis=1)])
        return nb.reshape(4, -1).astype(np.int64), np.asarray(
            grid.grid_area, dtype=np.float64).reshape(-1)
    return None


def masked_diffusion(q, f_land, nb, area, *, passes: int = 4,
                     k: float = _SMOOTH_K_PER_PASS):
    """``passes`` of explicit flux-form diffusion on a cell field with the
    zero_ocean rule: ``q_i += k * sum_j w_ij * (abar_ij / a_i) * (q_j - q_i)``
    with ``w_ij = max(0, min(f_land_i, f_land_j))`` and ``abar_ij`` the mean
    of the two cell areas, so the area integral ``sum_i a_i q_i`` is conserved
    exactly (antisymmetric edge terms) and an ocean cell (``f_land == 0``)
    never changes.  Monotone (positive weights) while ``k * K <= 1``."""
    q = np.asarray(q, dtype=np.float64).copy()
    f = np.asarray(f_land, dtype=np.float64)
    a = np.asarray(area, dtype=np.float64)
    nb = np.asarray(nb, dtype=np.int64)
    valid = nb >= 0
    j = np.where(valid, nb, 0)
    w = np.where(valid, np.maximum(0.0, np.minimum(f[None, :], f[j])), 0.0)
    coef = w * 0.5 * (a[None, :] + a[j]) / a[None, :]
    # max principle: the self weight 1 - k*sum_j coef must stay >= 0 on
    # every cell (the COEFFICIENT sum, not the neighbour count: a small
    # cell beside large ones carries (a_i + a_j)/(2 a_i) > 1 per edge)
    worst = float(np.max(k * np.sum(coef, axis=0)))
    if worst > 1.0:
        raise ValueError(f"masked_diffusion: k * sum_j coef = {worst:.3f} > 1 on "
                         "some cell -- the explicit pass is not monotone")
    for _ in range(int(passes)):
        q = q + k * np.sum(coef * (q[j] - q[None, :]), axis=0)
    return q


def _corner_neighbours(corners_xyz, k_max: int = 8):
    """``(k_max, n)`` neighbour table from shared corners: two cells are
    neighbours when they share a corner point (edge AND corner neighbours,
    enough for the quad ownership search)."""
    n = corners_xyz.shape[0]
    key = np.round(corners_xyz.reshape(-1, 3), 9)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(n, 4)
    by_corner = {}
    for c in range(n):
        for v in inv[c]:
            by_corner.setdefault(int(v), []).append(c)
    nb = np.full((k_max, n), -1, dtype=np.int64)
    for c in range(n):
        s = sorted({d for v in inv[c] for d in by_corner[int(v)] if d != c})
        nb[:min(k_max, len(s)), c] = s[:k_max]
    return nb


def grid_terrain_product(grid, lat_src_deg, lon_src_deg, elev_m, *,
                         f_land_override=None, smoothing_passes: int = 4):
    """THE terrain product of ``grid`` from one elevation field (metres,
    ocean negative or 0): cell-mean elevation of ``max(elev, 0)`` and the
    land fraction (area fraction with ``elev > 0``) from the SAME binning,
    then the masked diffusion.  ``f_land_override`` (an sftlf-type field
    already on the grid) replaces the land fraction BEFORE the filter and
    masks it; where it says ocean the elevation is pinned to 0.  Returns
    ``(z_s [m], f_land)`` flattened ``(n_cells,)``; a grid without
    neighbour information gets the UNSMOOTHED binned elevation (the duo
    column lane filters on its faces)."""
    elev = np.asarray(elev_m, dtype=np.float64)
    area = getattr(grid, "grid_area", None)
    if area is None:
        area = getattr(grid, "area", None)
    own = {}
    c_lat = getattr(grid, "cornerLat", None)
    if c_lat is not None:
        # cells with great-circle edges and known corners (the duo column
        # mesh): exact quad ownership, neighbours for the boundary search
        own["corners_xyz"] = _xyz(c_lat, grid.cornerLon).reshape(-1, 4, 3)
        own["nb"] = _corner_neighbours(own["corners_xyz"])
    elif np.asarray(grid.grid_lat).ndim == 2:
        lat_v = getattr(grid, "lat_v", None)
        own["axes_1d"] = (np.asarray(grid.lat, dtype=np.float64),
                          np.asarray(grid.lon, dtype=np.float64),
                          None if lat_v is None else np.asarray(lat_v, dtype=np.float64))
    # else: nearest centre -- exact for a Voronoi mesh; the cube lane grid
    # (no corner arrays) is served to that approximation (follow-up)
    z_mean, f_land = bin_latlon_to_cells(
        lat_src_deg, lon_src_deg, [np.maximum(elev, 0.0), (elev > 0.0).astype(np.float64)],
        np.asarray(grid.grid_lat), np.asarray(grid.grid_lon),
        cell_area=None if area is None else np.asarray(area), **own)
    if f_land_override is not None:
        f_land = np.clip(np.asarray(f_land_override, dtype=np.float64).ravel(), 0.0, 1.0)
    z_s = np.where(f_land > 0.0, z_mean, 0.0)
    table = _neighbour_table(grid)
    if table is not None and smoothing_passes > 0:
        z_s = masked_diffusion(z_s, f_land, table[0], table[1], passes=smoothing_passes)
    return z_s, f_land




def voronoi_cell_spacing_deg(ncells: int) -> float:
    """Mean angular cell size [deg] of a quasi-uniform Voronoi/MPAS mesh: the
    sphere (4π sr) split over ``ncells`` cells, ~sqrt(4π/ncells) rad per cell.
    The one definition of "cell" for the loader and the subgrid-orography
    builder, so a file built for a mesh and the loader's scale check agree."""
    return float(np.sqrt(4.0 * np.pi / max(int(ncells), 1)) * 180.0 / np.pi)


def _target_grid_degrees(grid):
    """Return target grid centers in degrees and grid metadata.

    Returns ``(target_lat_2d, target_lon_2d, is_gaussian, grid_spacing)``
    where the lat/lon arrays are in degrees and longitude is in [0, 360).

    Unstructured Voronoi/MPAS meshes expose ``grid_lat``/``grid_lon`` as
    rank-1 ``(nCells,)`` cell centres (``latCell``/``lonCell``); they have
    no structured ``n``/``n_lat``, so the cell-count is used to estimate a
    mean angular cell spacing.  ``is_gaussian`` is False for them (they are
    not a regular lat-lon mesh); callers that smooth must additionally guard
    on the rank-1 target shape (no structured/cube smoother applies).
    """
    grid_lat = np.asarray(grid.grid_lat)
    grid_lon = np.asarray(grid.grid_lon)
    # Classify by coordinate rank, not attribute presence.  Cubed-sphere stores
    # grid_lat as (6, n, n) (ndim 3); the structured lat-lon meshes — Gaussian
    # AND the regular lat-lon grid — store it as (n_lat, n_lon) (ndim 2).  The
    # old ``hasattr(grid, 'n_lat') and not hasattr(grid, 'n')`` heuristic
    # mis-classified the lat-lon grid (which also exposes ``n``) as
    # cubed-sphere, so its 2-D field was routed through the cubed-sphere
    # smoother + cube-edge blend instead of the structured lat-lon (periodic-lon,
    # pole-clamped) smoother.  ``is_structured_latlon`` keeps the variable's
    # downstream meaning (1-D lat/lon meshgrid + gaussian smoother, no cube-edge
    # blend) — Gaussian and regular lat-lon share that path.
    is_gaussian = grid_lat.ndim == 2

    if is_gaussian:
        target_lat = np.asarray(grid.lat) * 180.0 / np.pi
        target_lon = (np.asarray(grid.lon) * 180.0 / np.pi) % 360.0
        target_lon_2d, target_lat_2d = np.meshgrid(target_lon, target_lat)
        grid_spacing = 180.0 / grid.n_lat
    else:
        target_lat_2d = grid_lat * 180.0 / np.pi
        target_lon_2d = (grid_lon * 180.0 / np.pi) % 360.0
        if hasattr(grid, "n"):
            grid_spacing = 90.0 / grid.n
        else:
            # Unstructured Voronoi/MPAS: no structured ``n``.  Estimate the
            # mean angular cell size from the cell count — the whole sphere
            # (4π sr) split over nCells cells gives a linear angular extent
            # ~sqrt(4π/nCells) rad per cell (used only as the sub-grid
            # land-fraction sampling box width).
            grid_spacing = voronoi_cell_spacing_deg(np.asarray(grid_lat).size)

    return target_lat_2d, target_lon_2d, is_gaussian, grid_spacing


def load_land_fraction(
    grid,
    path: str,
    var_name: str = "",
) -> jnp.ndarray:
    """Load a land-sea mask from a NetCDF file, regridded to the model grid.

    A standalone entry point for the land fraction alone (independent of
    topography), so a real land-sea mask can be used with any
    ``topography`` setting — including ``flat``.

    Parameters
    ----------
    grid : CubedSphereGrid or GaussianGrid
        Target model grid.
    path : str
        Land-sea-mask NetCDF (CMIP6 ``sftlf`` percent, ERA5 ``lsm``
        fraction, etc.).
    var_name : str, optional
        Mask variable name; empty → auto-detect.

    Returns
    -------
    jax.Array
        Land fraction in [0, 1], shape ``(6, n, n)`` for cubed-sphere or
        ``(n_lat, n_lon)`` for Gaussian.
    """
    target_lat_2d, target_lon_2d, _, _ = _target_grid_degrees(grid)
    f_land = _load_land_fraction_file(
        path, var_name, target_lat_2d, target_lon_2d
    )
    return jnp.asarray(f_land)


def load_land_albedo(
    grid,
    path: str,
    var_name: str = "",
    month: int | None = None,
) -> jnp.ndarray:
    """Load a land surface albedo field, regridded to the model grid.

    Accepts a regular lat-lon NetCDF albedo (e.g. the ICON-extpar ALB
    broadband albedo remapped to 0.25 deg lat-lon at
    ``/work/bd1083/b309178/diffESM/land_data/
    extpar_albedo_latlon_0p25deg.nc``).  When the field carries a time
    axis (monthly climatology), ``month`` (1-12) selects a single month;
    otherwise the annual mean is used.

    Auto-detects percent vs. fraction (max > 1.5 → divide by 100).

    Parameters
    ----------
    grid : CubedSphereGrid or GaussianGrid
        Target model grid.
    path : str
        NetCDF file containing land albedo.
    var_name : str, optional
        Albedo variable name; empty → auto-detect (ALB, al, alb, albedo,
        surface_albedo, fal).
    month : int, optional
        1-12 to pick a single month from a monthly file; None → annual
        mean (or the only field if the file is already time-collapsed).

    Returns
    -------
    jax.Array
        Land albedo (fraction in [0, 1]), shape ``(6, n, n)`` for
        cubed-sphere or ``(n_lat, n_lon)`` for Gaussian.
    """
    import xarray as xr

    ds = xr.open_dataset(path)
    try:
        all_names = set(ds.data_vars) | set(ds.coords)
        if var_name:
            alb_var = var_name
        else:
            alb_var = None
            for c in ("ALB", "al", "alb", "albedo", "surface_albedo",
                      "fal", "Albedo"):
                if c in all_names:
                    alb_var = c
                    break
            if alb_var is None:
                data_vars = [
                    v for v in ds.data_vars
                    if ds[v].ndim >= 2
                    and not str(v).endswith(("_bnds", "_bounds"))
                ]
                if not data_vars:
                    raise KeyError(
                        f"No data variables in albedo file {path}"
                    )
                alb_var = data_vars[0]

        lat_var = None
        for c in ("lat", "latitude", "y", "Y"):
            if c in all_names:
                lat_var = c
                break
        lon_var = None
        for c in ("lon", "longitude", "x", "X"):
            if c in all_names:
                lon_var = c
                break
        if lat_var is None or lon_var is None:
            raise KeyError(
                f"Cannot detect lat/lon in albedo file {path}; "
                f"available: {sorted(all_names)}"
            )

        lat_src = ds[lat_var].values.astype(np.float64)
        lon_src = ds[lon_var].values.astype(np.float64)
        alb_data = ds[alb_var].values.astype(np.float64)
    finally:
        ds.close()

    # Reduce time / level dims to 2-D
    if alb_data.ndim >= 3 and month is not None:
        if not 1 <= month <= alb_data.shape[0]:
            raise ValueError(
                f"month={month} out of range [1, {alb_data.shape[0]}]"
            )
        alb_data = alb_data[month - 1]
    while alb_data.ndim > 2:
        # Time-mean over leading axis; squeeze any singleton level.
        if alb_data.shape[0] == 1:
            alb_data = alb_data[0]
        else:
            alb_data = np.nanmean(alb_data, axis=0)

    # Longitude in [0, 360), ascending
    lon_src = lon_src % 360.0
    lon_order = np.argsort(lon_src, kind="stable")
    lon_src = lon_src[lon_order]
    alb_data = alb_data[:, lon_order]

    # Latitude ascending
    if lat_src[0] > lat_src[-1]:
        lat_src = lat_src[::-1]
        alb_data = alb_data[::-1, :]

    alb_data = np.where(np.isnan(alb_data), 0.0, alb_data)

    # Percent → fraction (extpar ALB is 0-100; ERA5 fal is 0-1)
    if np.nanmax(alb_data) > 1.5:
        alb_data = alb_data / 100.0

    target_lat_2d, target_lon_2d, _, _ = _target_grid_degrees(grid)
    alb_grid = _regrid_to_target(
        lat_src, lon_src, alb_data, target_lat_2d, target_lon_2d
    )
    alb_grid = np.clip(alb_grid, 0.0, 1.0)
    return jnp.asarray(alb_grid)


# The effective resolution of a finite-volume dynamical core: the shortest
# wave it actually represents rather than damps, ~3-4 grid spacings (GLM
# review on #1712; the standard spectral-fidelity result for this class of
# core).  Orographic variance ABOVE this scale is in the model's own
# topography, so launching gravity-wave drag from it a second time
# double-counts.
EFFECTIVE_RESOLUTION_DX = 3.5


def _sso_file_construction(ds) -> dict:
    """Machine-readable record of how a subgrid-orography file was built.

    Prefers real attributes; falls back to parsing the ``history`` string for
    files written before those attributes existed.  Returns ``{}`` when the
    file says nothing -- an unstamped file is a file whose scale decomposition
    is unknown, which is itself worth reporting (#1712).
    """
    out = {}
    attrs = dict(getattr(ds, "attrs", {}) or {})
    for key in ("block_deg", "fine_res_deg", "resolved_cutoff_deg"):
        if key in attrs:
            try:
                out[key] = float(attrs[key])
            except (TypeError, ValueError):
                pass
    # PER-FIELD precedence, not wholesale: a file that stamps only
    # ``fine_res_deg`` used to suppress the history fallback entirely, and the
    # guard then had no scale to check and passed in silence (codex review).
    history = str(attrs.get("history", ""))
    for key, flag in (("block_deg", "--block-deg"),
                      ("fine_res_deg", "--fine-res-deg"),
                      ("resolved_cutoff_deg", "--resolved-cutoff-deg")):
        if key in out or flag not in history:
            continue
        tail = history.split(flag, 1)[1].lstrip()
        # accept both `--block-deg 2.0` and `--block-deg=2.0`
        if tail.startswith("="):
            tail = tail[1:]
        token = tail.split()[0] if tail.split() else ""
        try:
            out[key] = float(token)
        except ValueError:
            pass
    return out


# One report per (file, grid) pair per process.  A guard that fires on every
# load of every run is alarm fatigue rather than a guard (GLM review), and the
# condition is a property of the PAIR, not of the call.
_SSO_SCALE_REPORTED: set = set()

# Relative orographic drag against block size, measured on one fixed 0.25 deg
# source, area-weighted 40-60S, normalised to the 2 deg block (#1712, job
# 9633446).  Quoted in the guard message so the reader sees the SIZE of the
# mismatch rather than only its existence.
_SSO_BLOCK_DRAG_SWEEP = ((0.5, 0.06), (1.0, 0.31), (2.0, 1.00), (4.0, 1.99))


def _sso_drag_hint(block_deg: float, cell_deg: float) -> str:
    """Plain statement of what the block/grid mismatch is worth."""
    lo = min(_SSO_BLOCK_DRAG_SWEEP, key=lambda r: abs(r[0] - block_deg))
    hi = min(_SSO_BLOCK_DRAG_SWEEP, key=lambda r: abs(r[0] - cell_deg))
    if lo[1] <= 0.0 or hi[1] <= 0.0 or lo[0] == hi[0]:
        return ""
    return (f" For scale: the measured block sweep puts {lo[0]:g} deg at "
            f"{lo[1]:.2f}x and {hi[0]:g} deg at {hi[1]:.2f}x the Southern-Ocean "
            f"drag, i.e. this mismatch is worth roughly "
            f"{lo[1] / hi[1]:.1f}x in launch stress, not a rounding detail.")


def _check_sso_scale_decomposition(built: dict, grid_spacing_deg: float,
                                   path: str, mode: str) -> str | None:
    """Compare a subgrid-orography file's scale cutoff with the model grid.

    ``sgh`` is meant to be the stddev of orography the model does NOT resolve,
    so the file's upper cutoff has to sit at or below the model's effective
    resolution.  Nothing used to check that: the loader interpolated and
    clipped, and the same 2 deg file was read at every resolution (#1712,
    defects 2 and 3).  Returns the message (also emitted per ``mode``), or
    ``None`` when the file and the grid agree.
    """
    if mode not in ("warn", "error", "off"):
        raise ValueError(
            f"scale_check must be 'warn', 'error' or 'off'; got {mode!r}.")
    if mode == "off" or not np.isfinite(grid_spacing_deg) or grid_spacing_deg <= 0:
        return None
    cell = float(grid_spacing_deg)
    effective = EFFECTIVE_RESOLUTION_DX * cell

    if not built:
        msg = (
            f"subgrid orography {path}: the file records no scale "
            f"decomposition (no block_deg / resolved_cutoff_deg attribute and "
            f"nothing parseable in its history), so whether it double-counts "
            f"orography this {cell:.3f} deg grid already resolves cannot be "
            f"checked (#1712).")
    else:
        # WHERE the cutoff belongs is not settled, and this guard does not
        # settle it -- it reports the band and names both readings (#1712):
        #   * the model's TOPOGRAPHY carries features down to one cell, so
        #     variance above ``cell`` is already in the resolved field and the
        #     drag launched from it is counted twice;
        #   * the model cannot PROPAGATE waves shorter than ~3.5 cells, so a
        #     stricter reading puts the cutoff there and calls everything below
        #     it subgrid.
        # The trigger is the first (conservative) reading, because that is the
        # one #1712 measured; the message carries the second so nobody has to
        # rediscover the ambiguity.
        band = (f"this grid: cells {cell:.3f} deg, effective resolution "
                f"~{effective:.3f} deg ({EFFECTIVE_RESOLUTION_DX:g} cells)")
        cutoff = built.get("resolved_cutoff_deg")
        if cutoff is not None:
            # A file built with an EXPLICIT cutoff has already made the
            # decomposition on purpose, so it is judged against the LOOSER of
            # the two readings -- the effective resolution.  Judging it against
            # the cell size would reject the very construction this guard's own
            # message recommends (codex review).
            if cutoff <= effective * 1.05:
                return None
            msg = (
                f"subgrid orography {path}: built with an explicit resolved "
                f"cutoff of {cutoff:.3f} deg, coarser than anything this grid "
                f"could call subgrid ({band}). Variance above the effective "
                f"resolution is in the model's own topography, so the "
                f"gravity-wave drag launches from it a SECOND time (#1712). "
                f"Rebuild with --resolved-cutoff-deg between {cell:.2f} and "
                f"{effective:.2f}, or pass scale_check='off' to accept it.")
        else:
            block = built.get("block_deg")
            if block is None or block <= cell:
                return None
            msg = (
                f"subgrid orography {path}: built with block_deg="
                f"{block:.3f} deg and NO explicit resolved cutoff, so its "
                f"variance runs up to {block:.3f} deg, coarser than one cell "
                f"of the grid it is being read on ({band}). The model carries "
                f"orography between {cell:.3f} and {block:.3f} deg in its own "
                f"topography, so SOME of this variance is resolved and the "
                f"drag launched from it is counted a SECOND time. How much "
                f"depends on where the cutoff belongs -- at one cell, or at "
                f"the effective resolution -- and #1712 does not settle that, "
                f"which is why this is a mismatch report and not a verdict. "
                f"Rebuild with `prep_subgrid_orography.py "
                f"--resolved-cutoff-deg` between {cell:.2f} and "
                f"{effective:.2f} to make the decomposition explicit, or pass "
                f"scale_check='off' to accept the file as built."
                + _sso_drag_hint(block, cell))
    if mode == "error":
        raise ValueError(msg)
    key = (str(path), round(float(grid_spacing_deg), 6))
    if key not in _SSO_SCALE_REPORTED:
        _SSO_SCALE_REPORTED.add(key)
        logger.warning(msg)
    return msg


def load_subgrid_orography(
    grid,
    path: str,
    var_name: str = "SSO_STDH",
    *,
    scale_check: str = "warn",
) -> jnp.ndarray:
    """Load the subgrid orographic standard deviation, regridded to the grid.

    The orographic gravity-wave-drag schemes (Lindzen, McFarlane) launch a
    stress ``tau_0 ∝ h_topo²`` where ``h_topo`` is the standard deviation of
    the unresolved (subgrid) orography in each model cell.  Without a real
    per-column field the schemes fall back to a single global
    ``config.h_topo = 500 m`` everywhere — i.e. a 500 m mountain over the open
    ocean too — which is unphysical.  This loader reads a real subgrid
    orographic stddev (ICON-extpar ``SSO_STDH`` remapped to a regular lat-lon
    grid, e.g. ``extpar_sso_latlon_0p25deg.nc``) so the drag is localized to
    real mountains and is ~0 over flat ocean.

    Parameters
    ----------
    grid : CubedSphereGrid or GaussianGrid
        Target model grid.
    path : str
        NetCDF file containing the subgrid orographic stddev [m].
    var_name : str, optional
        Variable name (default ``SSO_STDH``; ICON-extpar convention).
    scale_check : {"warn", "error", "off"}, optional
        What to do when the file's scale decomposition does not match this
        grid (#1712).  ``sgh`` is by definition the variance the model does
        NOT resolve, so a file whose variance runs up to 2 deg read on a
        ~1.1 deg mesh hands the drag scheme orography the model already has in
        its own topography, and the wave is launched from it twice.  Nothing
        checked this before: the loader interpolated and clipped, ignoring the
        file's construction and the grid entirely.  Default ``"warn"`` reports
        it with both numbers and continues (every run today is in this state,
        so refusing by default would stop production without a decision);
        ``"error"`` refuses; ``"off"`` states that the double count is
        accepted.

    Returns
    -------
    jax.Array
        Subgrid orographic stddev [m], clipped to >= 0, shape ``(6, n, n)``
        for cubed-sphere or ``(n_lat, n_lon)`` for Gaussian.
    """
    import xarray as xr

    ds = xr.open_dataset(path)
    try:
        all_names = set(ds.data_vars) | set(ds.coords)
        sso_var = var_name if var_name in all_names else None
        if sso_var is None:
            for c in ("SSO_STDH", "sso_stdh", "stddev_orography", "sgh"):
                if c in all_names:
                    sso_var = c
                    break
        if sso_var is None:
            raise KeyError(
                f"Cannot find subgrid orography variable in {path}; "
                f"available: {sorted(ds.data_vars)}"
            )

        lat_var = None
        for c in ("lat", "latitude", "y", "Y"):
            if c in all_names:
                lat_var = c
                break
        lon_var = None
        for c in ("lon", "longitude", "x", "X"):
            if c in all_names:
                lon_var = c
                break
        if lat_var is None or lon_var is None:
            raise KeyError(
                f"Cannot detect lat/lon in SSO file {path}; "
                f"available: {sorted(all_names)}"
            )

        lat_src = ds[lat_var].values.astype(np.float64)
        lon_src = ds[lon_var].values.astype(np.float64)
        sso_data = ds[sso_var].values.astype(np.float64)
        built = _sso_file_construction(ds)
    finally:
        ds.close()

    # #1712: does this file's scale decomposition belong to THIS grid?
    _, _, _, _grid_spacing_deg = _target_grid_degrees(grid)
    _check_sso_scale_decomposition(
        built, float(_grid_spacing_deg), path, scale_check)

    while sso_data.ndim > 2:
        sso_data = sso_data[0]

    lon_src = lon_src % 360.0
    lon_order = np.argsort(lon_src, kind="stable")
    lon_src = lon_src[lon_order]
    sso_data = sso_data[:, lon_order]

    if lat_src[0] > lat_src[-1]:
        lat_src = lat_src[::-1]
        sso_data = sso_data[::-1, :]

    sso_data = np.where(np.isnan(sso_data), 0.0, sso_data)

    target_lat_2d, target_lon_2d, _, _ = _target_grid_degrees(grid)
    sso_grid = _regrid_to_target(
        lat_src, lon_src, sso_data, target_lat_2d, target_lon_2d
    )
    sso_grid = np.maximum(sso_grid, 0.0)
    return jnp.asarray(sso_grid)


def load_real_topography(
    grid,
    config: TopographyConfig | None = None,
    data_path: str | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Load real topography from a NetCDF file and regrid to model grid.

    Supports any regular lat-lon elevation dataset (ETOPO, GEBCO,
    GMTED2010, etc.).  THE terrain product (decision C, 2026-10-02):
    cell-mean of the ocean-clipped elevation and the land fraction from
    the SAME binning (:func:`grid_terrain_product`), then the grid's
    masked flux-form diffusion (fv_surf_map's zero_ocean rule).

    Land fraction from ``config.land_mask_path`` when set (CMIP6 ``sftlf``
    / ERA5 ``lsm``) replaces the binned one BEFORE the filter and masks it.

    Parameters
    ----------
    grid : CubedSphereGrid or GaussianGrid
        Target model grid.
    config : TopographyConfig, optional
        Configuration. If None, uses defaults.
    data_path : str, optional
        Override path to NetCDF file (takes precedence over config.path).

    Returns
    -------
    phis : jax.Array
        Surface geopotential [m^2/s^2].
        Shape (6, n, n) for cubed-sphere or (n_lat, n_lon) for Gaussian.
    f_land : jax.Array
        Land fraction [0, 1], same shape.
    """
    import xarray as xr

    if config is None:
        config = TopographyConfig(source="file")

    path = data_path or config.path
    if not path:
        raise ValueError(
            "No topography data path provided. "
            "Set config.path or pass data_path."
        )

    # Open dataset
    ds = xr.open_dataset(path)

    # Detect or use configured variable names
    if config.elev_var and config.lat_var and config.lon_var:
        elev_var, lat_var, lon_var = config.elev_var, config.lat_var, config.lon_var
    else:
        elev_var, lat_var, lon_var = _detect_variables(ds)

    # Extract source data
    lat_src = ds[lat_var].values.astype(np.float64)
    lon_src = ds[lon_var].values.astype(np.float64)
    elev_data = ds[elev_var].values.astype(np.float64)
    elev_units = str(ds[elev_var].attrs.get("units", ""))

    # Handle multi-dimensional data (squeeze extra dims)
    while elev_data.ndim > 2:
        elev_data = elev_data[0]

    # Ensure longitude in [0, 360)
    lon_src = lon_src % 360.0
    lon_order = np.argsort(lon_src, kind="stable")
    lon_src = lon_src[lon_order]
    elev_data = elev_data[:, lon_order]
    lon_src, elev_data = _drop_duplicate_periodic_lon(lon_src, elev_data)

    # Ensure latitude is sorted ascending
    if lat_src[0] > lat_src[-1]:
        lat_src = lat_src[::-1]
        elev_data = elev_data[::-1, :]

    # Replace NaN with 0 (ocean)
    elev_data = np.where(np.isnan(elev_data), 0.0, elev_data)

    ds.close()

    # --- geopotential-vs-meters "double-g" guard -----------------------------
    # This loader multiplies elevation [m] by constants.g below; an ERA5-style
    # invariant where 'z' is GEOPOTENTIAL [m²/s²] would be silently inflated
    # ~9.8×.  Detect it by the units attribute (when present) and by magnitude
    # (see _MAX_PLAUSIBLE_ELEV_M).
    _double_g_msg = (
        f"Topography variable {elev_var!r} in {path!r} looks like surface "
        f"GEOPOTENTIAL [m**2 s**-2], not elevation [m] "
        f"(units={elev_units!r}, max |value| = "
        f"{float(np.max(np.abs(elev_data))):.0f}; no Earth elevation exceeds "
        f"~8850 m, threshold {_MAX_PLAUSIBLE_ELEV_M:.0f} m). Multiplying it "
        "by g again (the double-g trap) would inflate phis ~9.8x. Divide the "
        "field by g (legoesm.constants.g) first, or pass the correct "
        "elevation variable via config.elev_var."
    )
    _units_norm = elev_units.replace(" ", "").replace("**", "^").lower()
    if _units_norm in ("m^2s^-2", "m^2/s^2", "m2s-2", "m2/s2", "m^2s-2"):
        raise ValueError(_double_g_msg)
    if float(np.max(np.abs(elev_data))) > _MAX_PLAUSIBLE_ELEV_M:
        raise ValueError(_double_g_msg)

    shape = tuple(np.asarray(grid.grid_lat).shape)
    f_over = None
    if config.land_mask_path:
        target_lat_2d, target_lon_2d, _, _ = _target_grid_degrees(grid)
        f_over = _load_land_fraction_file(
            config.land_mask_path, config.land_mask_var, target_lat_2d, target_lon_2d)
    z_s, f_land = grid_terrain_product(
        grid, lat_src, lon_src, elev_data, f_land_override=f_over,
        smoothing_passes=config.smoothing_passes)
    phis = jnp.asarray(constants.g * z_s.reshape(shape))
    f_land = jnp.asarray(np.clip(f_land, 0.0, 1.0).reshape(shape))
    return phis, f_land
