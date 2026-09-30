"""Nearest-wet-neighbour regrid from a curvilinear source grid.

Shared by the NEMO static-forcing loaders (zdfiwm power maps, ISF 'spe'
melt fields): map each target cell to the closest WET source cell on
the unit sphere.  One KDTree per (source grid, wet mask); the returned
indexer applies to any number of fields (single-owner regrid numerics —
no per-loader KDTree re-derivation).
"""

from __future__ import annotations

import numpy as np


def unit_sphere_xyz(lon_deg, lat_deg) -> np.ndarray:
    """(..., 3) unit vectors from degrees lon/lat."""
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    return np.stack([np.cos(lat) * np.cos(lon),
                     np.cos(lat) * np.sin(lon),
                     np.sin(lat)], axis=-1)


class NearestWetRegridder:
    """Precomputed nearest-WET-source-cell lookup for a fixed target grid."""

    def __init__(self, src_lon_deg, src_lat_deg, src_wet, tgt_lon_deg,
                 tgt_lat_deg, structured: bool = True):
        from scipy.spatial import cKDTree

        src_wet = np.asarray(src_wet, dtype=bool)
        if not np.any(src_wet):
            raise ValueError("curvilinear regrid: source has no wet cells")
        self._src_wet = src_wet
        tree = cKDTree(unit_sphere_xyz(
            np.asarray(src_lon_deg)[src_wet], np.asarray(src_lat_deg)[src_wet]))
        tgt_lat = np.asarray(tgt_lat_deg, dtype=np.float64)
        tgt_lon = np.asarray(tgt_lon_deg, dtype=np.float64)
        # structured=True: 1-D lat/lon are regular grid AXES -> cross into a
        # 2-D mesh.  structured=False: 1-D lat/lon are PAIRED unstructured cell
        # centres (MPAS Voronoi, nCells) -> use as-is (no outer product).
        if structured and tgt_lat.ndim == 1 and tgt_lon.ndim == 1:
            tgt_lat, tgt_lon = np.meshgrid(tgt_lat, tgt_lon, indexing="ij")
        if tgt_lat.shape != tgt_lon.shape:
            raise ValueError(
                f"target lat {tgt_lat.shape} and lon {tgt_lon.shape} differ")
        self.target_shape = tgt_lat.shape
        self.target_lat = tgt_lat
        _, self._idx = tree.query(
            unit_sphere_xyz(tgt_lon, tgt_lat).reshape(-1, 3), k=1)

    def __call__(self, field) -> np.ndarray:
        """Regrid one 2-D source field to the target grid."""
        f = np.asarray(field, dtype=np.float64)
        return f[self._src_wet][self._idx].reshape(self.target_shape)


def coords_match(src_lat, src_lon, tgt_lat, tgt_lon,
                 tol_deg: float = 1.0e-3, valid=None) -> bool:
    """True when the target grid IS the source grid (same-mesh passthrough).

    ``valid`` (optional, bool array of the source shape) restricts the
    comparison to coordinate-carrying source cells: some NEMO input files
    (e.g. SI3 ``Ice_initialization.nc``) write (0, 0) lat/lon on land, so
    a full-array compare would spuriously fail against the model grid's
    real land coordinates.  ``valid=None`` keeps the exact legacy
    full-array behaviour.
    """
    tgt_lat = np.asarray(tgt_lat)
    tgt_lon = np.asarray(tgt_lon)
    src_lat = np.asarray(src_lat)
    src_lon = np.asarray(src_lon)
    if tgt_lat.shape != src_lat.shape:
        return False
    dlat = np.abs(tgt_lat - src_lat)
    dlon = np.abs((tgt_lon - src_lon + 180.0) % 360.0 - 180.0)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != src_lat.shape:
            raise ValueError(
                f"coords_match: valid mask shape {valid.shape} does not "
                f"match source coords {src_lat.shape}")
        if not valid.any():
            return False
        dlat = dlat[valid]
        dlon = dlon[valid]
    return (float(np.max(dlat)) < tol_deg
            and float(np.max(dlon)) < tol_deg)


def estimate_curvilinear_cell_area(lat_deg, lon_deg):
    """Estimate per-cell areas [m^2] of a curvilinear grid from its 2-D nav
    coordinates: R^2 cos(lat) |dlon| |dlat| with wrap-aware longitude
    differences (an eORCA seam column jumps ~360 deg; the raw gradient there
    would inflate the area by two orders of magnitude — codex 2026-09-01).

    An estimate for RENORMALISATION WEIGHTS only (global integral scalars),
    not a metric: tripolar fold rows keep O(1 deg) magnitudes and contribute
    a few of ~300 rows to a global sum.
    """
    lat = np.asarray(lat_deg, dtype=np.float64)
    lon = np.asarray(lon_deg, dtype=np.float64)
    from legoesm import constants
    # Unwrap BEFORE differencing: a central difference halves a mid-row
    # 360-deg jump to ~180 deg, which a post-gradient min(d, 360-d) misses.
    dlon = np.abs(np.gradient(np.unwrap(lon, period=360.0, axis=-1), axis=-1))
    dlat = np.abs(np.gradient(lat, axis=-2))
    return (constants.R_earth ** 2 * np.cos(np.deg2rad(lat))
            * np.deg2rad(dlon) * np.deg2rad(dlat))
