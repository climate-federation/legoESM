"""Regridding between CubedSphere and Gaussian grids.

Provides offline weight computation (numpy, done once at init) and
JIT-compatible application for field transfer between grids.
Supports scalar fields and vector fields with rotation correction.

Methods:
- Face-aware bilinear interpolation (cubed-sphere → lat-lon):
  projects each target point onto the correct cube face via gnomonic
  inversion and interpolates from the 4 surrounding cell centres.
- Inverse-distance weighting of K nearest neighbors (KD-tree) for
  general unstructured-to-structured transfers.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.gaussian import GaussianGrid


class RegridWeights(NamedTuple):
    """Precomputed regridding weights.

    Attributes
    ----------
    src_indices : jax.Array, shape (n_target, k_neighbors)
        Flat indices into the source field for each target point.
    weights : jax.Array, shape (n_target, k_neighbors)
        Interpolation weights (sum to 1 for each target point).
    target_shape : tuple
        Shape of the target grid (for reshaping output).
    src_flat_size : int
        Total number of source grid points.
    """
    src_indices: jnp.ndarray
    weights: jnp.ndarray
    target_shape: tuple
    src_flat_size: int


def _latlon_to_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Convert lat/lon [rad] to Cartesian (x, y, z) on unit sphere."""
    cos_lat = np.cos(lat)
    return np.stack([
        cos_lat * np.cos(lon),
        cos_lat * np.sin(lon),
        np.sin(lat),
    ], axis=-1)


def _inverse_distance_weights(src_xyz, tgt_xyz, k_neighbors: int):
    """KD-tree k-nearest-neighbour inverse-distance weights from source to target
    Cartesian points — the SHARED core of every ``compute_*_weights`` builder so the
    KD-tree + IDW math is written ONCE.  Returns ``(indices (n_tgt, k), weights
    (n_tgt, k))`` with weights summing to 1 per target."""
    from scipy.spatial import cKDTree

    tree = cKDTree(src_xyz)
    distances, indices = tree.query(tgt_xyz, k=k_neighbors)
    if k_neighbors == 1:
        # cKDTree.query collapses the neighbour axis for k=1; restore (n_tgt, 1)
        # so the axis=-1 normalization and (n_tgt, k) contract hold for all k.
        distances = distances[:, None]
        indices = indices[:, None]
    distances = np.maximum(distances, 1e-12)        # guard exact matches (dist 0)
    inv_dist = 1.0 / distances
    weights = inv_dist / inv_dist.sum(axis=-1, keepdims=True)
    return indices, weights


def compute_latlon_to_cs_weights(
    src_lat, src_lon, cs_grid: CubedSphereGrid, k_neighbors: int = 4,
) -> RegridWeights:
    """Regridding weights from a REGULAR lat-lon grid (1-D ``src_lat``/``src_lon`` in
    radians, e.g. ERA5) to a cubed-sphere grid via KD-tree inverse-distance.

    Built from the ACTUAL source point locations.  Going through a Gaussian PROXY of
    the source instead (the old ERA5→cubed-sphere path) is WRONG: the proxy's
    quadrature latitudes do not coincide with a uniform lat-lon grid, and its latitude
    COUNT generally differs (e.g. 72 vs a pole-inclusive 73), so the ``src_indices``
    — computed for the proxy's flattened layout — gather the WRONG source cells (the
    regrid pulled data from near-antipodal latitudes).  The source is flattened
    ``(lat, lon)`` in C-order (lat slowest) to match a field reshaped
    ``(n_lat, n_lon, ...) → (n_lat·n_lon, ...)``.
    """
    src_lat = np.asarray(src_lat)
    src_lon = np.asarray(src_lon)
    lat2d, lon2d = np.meshgrid(src_lat, src_lon, indexing="ij")   # (n_lat, n_lon)
    src_xyz = _latlon_to_xyz(lat2d.ravel(), lon2d.ravel())
    tgt_xyz = _latlon_to_xyz(
        np.asarray(cs_grid.lat).ravel(), np.asarray(cs_grid.lon).ravel())
    indices, weights = _inverse_distance_weights(src_xyz, tgt_xyz, k_neighbors)
    return RegridWeights(
        src_indices=jnp.array(indices, dtype=jnp.int32),
        weights=jnp.array(weights, dtype=jnp.float32),
        target_shape=tuple(int(s) for s in cs_grid.lat.shape),
        src_flat_size=int(src_lat.shape[0] * src_lon.shape[0]),
    )


def compute_cs_to_gauss_weights(
    cs_grid: CubedSphereGrid,
    gauss_grid: GaussianGrid,
    k_neighbors: int = 4,
) -> RegridWeights:
    """Compute regridding weights from CubedSphere to Gaussian grid.

    Uses KD-tree in Cartesian coordinates for nearest-neighbor lookup,
    then inverse-distance weighting for interpolation.

    Parameters
    ----------
    cs_grid : CubedSphereGrid
        Source cubed-sphere grid.
    gauss_grid : GaussianGrid
        Target Gaussian grid.
    k_neighbors : int
        Number of nearest neighbors for interpolation.

    Returns
    -------
    RegridWeights
        Precomputed weights for regridding.
    """
    # Source points: flatten cubed-sphere (6, n, n); target: Gaussian grid.
    src_xyz = _latlon_to_xyz(
        np.asarray(cs_grid.lat).ravel(), np.asarray(cs_grid.lon).ravel())
    tgt_xyz = _latlon_to_xyz(
        np.asarray(gauss_grid.lat2d).ravel(), np.asarray(gauss_grid.lon2d).ravel())
    indices, weights = _inverse_distance_weights(src_xyz, tgt_xyz, k_neighbors)
    return RegridWeights(
        src_indices=jnp.array(indices, dtype=jnp.int32),
        weights=jnp.array(weights, dtype=jnp.float64),
        target_shape=(gauss_grid.n_lat, gauss_grid.n_lon),
        src_flat_size=int(np.prod(np.array(cs_grid.lat.shape))),
    )


def compute_gauss_to_cs_weights(
    gauss_grid: GaussianGrid,
    cs_grid: CubedSphereGrid,
    k_neighbors: int = 4,
) -> RegridWeights:
    """Compute regridding weights from Gaussian to CubedSphere grid.

    Parameters
    ----------
    gauss_grid : GaussianGrid
        Source Gaussian grid.
    cs_grid : CubedSphereGrid
        Target cubed-sphere grid.
    k_neighbors : int
        Number of nearest neighbors for interpolation.

    Returns
    -------
    RegridWeights
    """
    src_xyz = _latlon_to_xyz(
        np.asarray(gauss_grid.lat2d).ravel(), np.asarray(gauss_grid.lon2d).ravel())
    tgt_xyz = _latlon_to_xyz(
        np.asarray(cs_grid.lat).ravel(), np.asarray(cs_grid.lon).ravel())
    indices, weights = _inverse_distance_weights(src_xyz, tgt_xyz, k_neighbors)
    return RegridWeights(
        src_indices=jnp.array(indices, dtype=jnp.int32),
        weights=jnp.array(weights, dtype=jnp.float64),
        target_shape=tuple(int(s) for s in cs_grid.lat.shape),
        src_flat_size=gauss_grid.n_lat * gauss_grid.n_lon,
    )


def compute_latlon_to_voronoi_weights(
    src_lat_1d: np.ndarray,
    src_lon_1d: np.ndarray,
    tgt_lat: np.ndarray,
    tgt_lon: np.ndarray,
    k_neighbors: int = 4,
    src_valid: np.ndarray | None = None,
) -> RegridWeights:
    """Compute regridding weights from regular lat-lon to unstructured points.

    Uses KD-tree in Cartesian coordinates for nearest-neighbor lookup,
    then inverse-distance weighting for interpolation.

    Parameters
    ----------
    src_lat_1d : array, shape (n_lat,)
        Source latitude centres in **radians**.
    src_lon_1d : array, shape (n_lon,)
        Source longitude centres in **radians**.
    tgt_lat : array, shape (n_target,)
        Target point latitudes in **radians** (e.g. ``mesh.latCell``).
    tgt_lon : array, shape (n_target,)
        Target point longitudes in **radians** (e.g. ``mesh.lonCell``).
    k_neighbors : int
        Number of nearest neighbors for interpolation.
    src_valid : array (n_lat, n_lon) or (n_lat*n_lon,), optional
        Boolean mask of VALID source cells.  When given, the KD-tree is built
        from valid cells ONLY, so every target's k neighbours are guaranteed
        valid — each target maps to its nearest *actual* valid source, and no
        target can end up with all-missing neighbours (the returned
        ``src_indices`` still index the FULL flattened grid, so the field regrid
        is unchanged).  Used for a land-only source (e.g. CRU-JRA ocean = NaN) so
        coastal targets never draw only ocean neighbours.  ``None`` = every source
        cell participates (the original behaviour).

    Returns
    -------
    RegridWeights
        Precomputed weights with ``target_shape = (n_target,)``.
    """
    from scipy.spatial import cKDTree

    # Source: regular lat-lon meshgrid → (n_lat*n_lon, 3)
    lon2d, lat2d = np.meshgrid(
        np.asarray(src_lon_1d), np.asarray(src_lat_1d),
    )
    src_xyz = _latlon_to_xyz(lat2d.ravel(), lon2d.ravel())

    # Target: unstructured points → (n_target, 3)
    tgt_xyz = _latlon_to_xyz(
        np.asarray(tgt_lat).ravel(), np.asarray(tgt_lon).ravel(),
    )

    if src_valid is not None:
        # Build the tree from VALID source cells only; map neighbour indices back
        # to full-grid coordinates so the downstream field regrid is unchanged.
        valid_flat = np.asarray(src_valid).ravel().astype(bool)
        valid_idx = np.nonzero(valid_flat)[0]
        if valid_idx.size == 0:
            raise ValueError("compute_latlon_to_voronoi_weights: src_valid masks out all source cells")
        k = min(int(k_neighbors), int(valid_idx.size))
        tree = cKDTree(src_xyz[valid_idx])
        distances, local = tree.query(tgt_xyz, k=k)
        if k == 1:
            distances = distances[:, None]
            local = local[:, None]
        indices = valid_idx[local]
    else:
        tree = cKDTree(src_xyz)
        distances, indices = tree.query(tgt_xyz, k=k_neighbors)

    distances = np.maximum(distances, 1e-12)
    inv_dist = 1.0 / distances
    weights = inv_dist / inv_dist.sum(axis=-1, keepdims=True)

    n_target = tgt_xyz.shape[0]
    src_flat_size = int(len(src_lat_1d)) * int(len(src_lon_1d))

    return RegridWeights(
        src_indices=jnp.array(indices, dtype=jnp.int32),
        weights=jnp.array(weights, dtype=jnp.float64),
        target_shape=(n_target,),
        src_flat_size=src_flat_size,
    )


def regrid_scalar(
    field: jnp.ndarray,
    regrid_weights: RegridWeights,
) -> jnp.ndarray:
    """Apply precomputed regridding weights to a scalar field.

    Parameters
    ----------
    field : array
        Source field. Spatial dimensions will be flattened; extra
        trailing dimensions (e.g. vertical levels) are preserved.
    regrid_weights : RegridWeights
        Precomputed weights from ``compute_*_weights``.

    Returns
    -------
    array
        Regridded field with shape ``target_shape + trailing_dims``.
    """
    # Flatten spatial dimensions
    spatial_size = regrid_weights.src_flat_size

    # Handle different source shapes
    if field.size == spatial_size:
        flat = field.ravel()
        extra_dims = ()
    else:
        # Flatten spatial dims, keep trailing
        n_spatial = spatial_size
        n_trailing = field.size // n_spatial
        flat = field.reshape(n_spatial, n_trailing)
        extra_dims = flat.shape[1:]

    indices = regrid_weights.src_indices    # (n_target, k)
    # Weights are stored fp64. For a FLOATING field, downcast to the field dtype
    # so an fp64 field keeps full-precision gradients through the IDW sum while
    # an fp32 field stays fp32 (no fp64 footprint at run time). For a NON-floating
    # field (int/bool mask, categorical), keep the float weights so the fractional
    # IDW weights are not truncated to 0/1 — the product then promotes the result
    # to float, exactly as before this change.
    if jnp.issubdtype(field.dtype, jnp.floating):
        weights = regrid_weights.weights.astype(field.dtype)    # (n_target, k)
    else:
        weights = regrid_weights.weights                        # (n_target, k)

    if len(extra_dims) == 0:
        # Simple scalar: (n_target,)
        gathered = flat[indices]                    # (n_target, k)
        result = jnp.sum(gathered * weights, axis=-1)  # (n_target,)
        return result.reshape(regrid_weights.target_shape)
    else:
        # With extra dimensions: (n_target, k, n_extra)
        gathered = flat[indices]                    # (n_target, k, n_extra)
        result = jnp.sum(
            gathered * weights[..., None], axis=-2
        )  # (n_target, n_extra)
        return result.reshape(regrid_weights.target_shape + extra_dims)


def regrid_vector(
    u: jnp.ndarray,
    v: jnp.ndarray,
    regrid_weights: RegridWeights,
    src_angle: jnp.ndarray | None = None,
    tgt_angle: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Regrid vector components with optional rotation correction.

    If grid rotation angles are provided, vector components are rotated
    from the source grid orientation to true east-north before regridding,
    then rotated to the target grid orientation.

    Parameters
    ----------
    u, v : arrays
        Zonal and meridional components on source grid.
    regrid_weights : RegridWeights
        Precomputed weights.
    src_angle : array, optional
        Rotation angle of source grid relative to east [rad].
    tgt_angle : array, optional
        Rotation angle of target grid relative to east [rad].

    Returns
    -------
    u_tgt, v_tgt : arrays
        Vector components on target grid.
    """
    # Rotate to true east-north if source has grid rotation
    if src_angle is not None:
        cos_a = jnp.cos(src_angle)
        sin_a = jnp.sin(src_angle)
        u_en = u * cos_a - v * sin_a
        v_en = u * sin_a + v * cos_a
    else:
        u_en, v_en = u, v

    # Regrid each component
    u_tgt = regrid_scalar(u_en, regrid_weights)
    v_tgt = regrid_scalar(v_en, regrid_weights)

    # Rotate to target grid orientation
    if tgt_angle is not None:
        cos_a = jnp.cos(tgt_angle)
        sin_a = jnp.sin(tgt_angle)
        u_out = u_tgt * cos_a + v_tgt * sin_a
        v_out = -u_tgt * sin_a + v_tgt * cos_a
        return u_out, v_out

    return u_tgt, v_tgt


# ==============================================================================
# Face-aware cubed-sphere to lat-lon regridding
# ==============================================================================

# Face connectivity (same as halo.py).
# CONNECTIVITY[face][edge] = (neighbor_face, neighbor_edge, reversed)
_WEST, _EAST, _SOUTH, _NORTH = 0, 1, 2, 3
_CONNECTIVITY = {
    0: {_WEST: (3, _EAST, False), _EAST: (1, _WEST, False),
        _SOUTH: (5, _NORTH, False), _NORTH: (4, _SOUTH, False)},
    1: {_WEST: (0, _EAST, False), _EAST: (2, _WEST, False),
        _SOUTH: (5, _EAST, True), _NORTH: (4, _EAST, False)},
    2: {_WEST: (1, _EAST, False), _EAST: (3, _WEST, False),
        _SOUTH: (5, _SOUTH, True), _NORTH: (4, _NORTH, True)},
    3: {_WEST: (2, _EAST, False), _EAST: (0, _WEST, False),
        _SOUTH: (5, _WEST, False), _NORTH: (4, _WEST, True)},
    4: {_WEST: (3, _NORTH, True), _EAST: (1, _NORTH, False),
        _SOUTH: (0, _NORTH, False), _NORTH: (2, _NORTH, True)},
    5: {_WEST: (3, _SOUTH, False), _EAST: (1, _SOUTH, True),
        _SOUTH: (2, _SOUTH, True), _NORTH: (0, _SOUTH, False)},
}


def _pad_faces_for_regrid(field: np.ndarray, *, strip_inset: int) -> np.ndarray:
    """Pad a ``(6, m, m)`` per-face field into ``(6, m+2, m+2)`` with the
    neighbouring faces' boundary cells (CONNECTIVITY reversal-aware); the
    four corners are the mean of the two adjacent halo values.

    ``strip_inset`` selects which neighbour row/column is copied into the
    halo ring: ``0`` = the shared-boundary cell (cell-centre regrid), ``1``
    = one gnomonic cell inside it (D-grid corner data, whose index-0/-n
    points sit ON the shared boundary).  Output is sized off
    ``field.shape`` so it serves both the ``(6,n,n)`` and ``(6,n+1,n+1)``
    layouts.
    """
    m = field.shape[1]
    padded = np.zeros((6, m + 2, m + 2), dtype=field.dtype)
    padded[:, 1:-1, 1:-1] = field

    def _nbr_strip(f: int, edge: int) -> np.ndarray:
        if edge == _WEST:
            return field[f, strip_inset, :]
        if edge == _EAST:
            return field[f, -1 - strip_inset, :]
        if edge == _SOUTH:
            return field[f, :, strip_inset]
        return field[f, :, -1 - strip_inset]  # NORTH

    for face in range(6):
        for edge in (_WEST, _EAST, _SOUTH, _NORTH):
            nbr_face, nbr_edge, rev = _CONNECTIVITY[face][edge]
            strip = _nbr_strip(nbr_face, nbr_edge)
            if rev:
                strip = strip[::-1]
            if edge == _WEST:
                padded[face, 0, 1:-1] = strip
            elif edge == _EAST:
                padded[face, -1, 1:-1] = strip
            elif edge == _SOUTH:
                padded[face, 1:-1, 0] = strip
            else:
                padded[face, 1:-1, -1] = strip

    # Corners: average of the two adjacent halo neighbours.
    for face in range(6):
        padded[face, 0, 0] = 0.5 * (padded[face, 0, 1] + padded[face, 1, 0])
        padded[face, 0, -1] = 0.5 * (padded[face, 0, -2] + padded[face, 1, -1])
        padded[face, -1, 0] = 0.5 * (padded[face, -1, 1] + padded[face, -2, 0])
        padded[face, -1, -1] = 0.5 * (padded[face, -1, -2] + padded[face, -2, -1])

    return padded


def _pad_field_for_regrid(field: np.ndarray, n: int) -> np.ndarray:
    """Create (6, n+2, n+2) padded field with neighbor boundary cells.

    Copies the outermost row/column of each neighbouring face into the
    halo ring (see :func:`_pad_faces_for_regrid`).
    """
    return _pad_faces_for_regrid(field, strip_inset=0)


class CubedSphereToLatLonWeights(NamedTuple):
    """Precomputed face-aware bilinear interpolation weights.

    Indices (``i0``, ``j0``) refer to the **padded** (n+2)×(n+2) field
    produced by :func:`_pad_field_for_regrid`, so valid values run from
    0 (halo) through n+1 (opposite halo).

    Attributes
    ----------
    face : int32 array, shape (n_target,)
        Cube face index (0-5) for each target point.
    i0, j0 : int32 arrays, shape (n_target,)
        Lower-left indices in the padded field.
    wi, wj : float64 arrays, shape (n_target,)
        Bilinear weights in the i and j directions (0 ≤ w ≤ 1).
    n_lat, n_lon : int
        Output grid dimensions.
    n : int
        Cubed-sphere tile size (cells per face edge).
    lon_cent : 1-D array, shape (n_lon,)
        Longitude centres [degrees].
    lat_cent : 1-D array, shape (n_lat,)
        Latitude centres [degrees].
    """
    face: np.ndarray
    i0: np.ndarray
    j0: np.ndarray
    wi: np.ndarray
    wj: np.ndarray
    n_lat: int
    n_lon: int
    n: int
    lon_cent: np.ndarray
    lat_cent: np.ndarray


def _target_lat(n_lat: int, lat_cent) -> np.ndarray:
    """Target latitudes [deg]: pole-to-pole ``linspace(-90, 90, n_lat)`` by
    default, or the caller's own row centres (e.g. the CMIP writer's
    cell-centred labels) so samples and labels come from ONE definition."""
    if lat_cent is None:
        return np.linspace(-90.0, 90.0, n_lat)
    lat = np.asarray(lat_cent, dtype=np.float64)
    if (lat.shape != (n_lat,) or n_lat == 0 or not np.all(np.isfinite(lat))
            or np.any(np.diff(lat) <= 0.0) or lat[0] < -90.0 or lat[-1] > 90.0):
        raise ValueError(
            f"lat_cent must be {n_lat} finite, strictly increasing latitudes "
            f"in [-90, 90]; got shape {lat.shape}")
    return lat


def compute_cubedsphere_to_latlon_weights(
    n: int,
    n_lon: int = 360,
    n_lat: int = 181,
    lat_cent: np.ndarray | None = None,
) -> CubedSphereToLatLonWeights:
    """Precompute face-aware bilinear weights for CS → lat-lon.

    For each target lat-lon point:
      1. Convert to Cartesian on the unit sphere.
      2. Select the cube face whose outward normal has the largest projection.
      3. Invert the gnomonic projection to recover face-local (alpha_x, alpha_y).
      4. Map gnomonic coordinates to fractional cell-centre indices.
      5. Shift indices into **padded** (n+2)×(n+2) coordinates so that
         bilinear interpolation extends smoothly across face boundaries
         using halo cells from the neighbouring face.

    Parameters
    ----------
    n : int
        Cubed-sphere tile size (cells per face edge).
    n_lon, n_lat : int
        Output regular lat-lon grid dimensions.
    lat_cent : 1-D array, optional
        Row latitudes [deg] to sample at (default pole-to-pole linspace).

    Returns
    -------
    CubedSphereToLatLonWeights
    """
    # Target grid
    lon_cent = np.linspace(-180.0, 180.0, n_lon, endpoint=False) + 180.0 / n_lon
    lat_cent = _target_lat(n_lat, lat_cent)
    lon2d, lat2d = np.meshgrid(lon_cent, lat_cent)

    lon_r = np.deg2rad(lon2d.ravel())
    lat_r = np.deg2rad(lat2d.ravel())
    cos_lat = np.cos(lat_r)
    x = cos_lat * np.cos(lon_r)
    y = cos_lat * np.sin(lon_r)
    z = np.sin(lat_r)

    # Face assignment: pick the face whose outward normal has max projection.
    # Face ordering: 0→+x, 1→+y, 2→−x, 3→−y, 4→+z, 5→−z.
    proj = np.column_stack([x, y, -x, -y, z, -z])
    face = np.argmax(proj, axis=1).astype(np.int32)

    # Inverse gnomonic projection per face (vectorised).
    alpha_x = np.empty(len(x), dtype=np.float64)
    alpha_y = np.empty(len(x), dtype=np.float64)

    for f in range(6):
        m = face == f
        if not np.any(m):
            continue
        xm, ym, zm = x[m], y[m], z[m]
        if f == 0:
            alpha_x[m] = np.arctan(ym / xm)
            alpha_y[m] = np.arctan(zm / xm)
        elif f == 1:
            alpha_x[m] = np.arctan(-xm / ym)
            alpha_y[m] = np.arctan(zm / ym)
        elif f == 2:
            alpha_x[m] = np.arctan(ym / xm)
            alpha_y[m] = np.arctan(-zm / xm)
        elif f == 3:
            alpha_x[m] = np.arctan(-xm / ym)
            alpha_y[m] = np.arctan(-zm / ym)
        elif f == 4:
            alpha_x[m] = np.arctan(ym / zm)
            alpha_y[m] = np.arctan(-xm / zm)
        elif f == 5:
            alpha_x[m] = np.arctan(-ym / zm)
            alpha_y[m] = np.arctan(-xm / zm)

    # Map gnomonic coords to fractional cell-centre indices.
    # Cell centres: alpha[k] = -pi/4 + (k + 0.5) * dalpha, k = 0..n-1.
    dalpha = np.pi / (2.0 * n)
    alpha_min = -np.pi / 4.0 + dalpha / 2.0  # centre of first cell

    fi = (alpha_x - alpha_min) / dalpha  # ~ [-0.5, n-0.5]
    fj = (alpha_y - alpha_min) / dalpha

    # Shift to padded coordinates (interior sits at indices 1..n).
    fi_pad = fi + 1.0  # ~ [0.5, n+0.5]
    fj_pad = fj + 1.0

    # Clip to valid padded range.
    fi_pad = np.clip(fi_pad, 0.0, n + 1.0)
    fj_pad = np.clip(fj_pad, 0.0, n + 1.0)

    i0 = np.minimum(np.floor(fi_pad).astype(np.int32), n)
    j0 = np.minimum(np.floor(fj_pad).astype(np.int32), n)

    wi = fi_pad - i0.astype(np.float64)
    wj = fj_pad - j0.astype(np.float64)

    return CubedSphereToLatLonWeights(
        face=face, i0=i0, j0=j0, wi=wi, wj=wj,
        n_lat=n_lat, n_lon=n_lon, n=n,
        lon_cent=lon_cent, lat_cent=lat_cent,
    )


_cs_weights_cache: dict[tuple, CubedSphereToLatLonWeights] = {}


def get_cubedsphere_to_latlon_weights(
    n: int, n_lon: int = 360, n_lat: int = 181,
    lat_cent: np.ndarray | None = None,
) -> CubedSphereToLatLonWeights:
    """Cached version of :func:`compute_cubedsphere_to_latlon_weights`."""
    key = (n, n_lat, n_lon, None if lat_cent is None
           else tuple(np.asarray(lat_cent, dtype=np.float64).tolist()))
    if key not in _cs_weights_cache:
        _cs_weights_cache[key] = compute_cubedsphere_to_latlon_weights(
            n, n_lon=n_lon, n_lat=n_lat, lat_cent=lat_cent)
    return _cs_weights_cache[key]


def apply_cubedsphere_to_latlon(
    field_faces: np.ndarray,
    weights: CubedSphereToLatLonWeights,
) -> np.ndarray:
    """Apply face-aware bilinear weights to regrid (6, n, n) → (n_lat, n_lon).

    The source field is first padded with one ring of halo cells copied
    from neighbouring faces (via :func:`_pad_field_for_regrid`) so that
    bilinear interpolation extends smoothly across cube-face boundaries.

    Parameters
    ----------
    field_faces : array, shape (6, n, n)
        Source cubed-sphere field.
    weights : CubedSphereToLatLonWeights
        Precomputed weights from :func:`compute_cubedsphere_to_latlon_weights`.

    Returns
    -------
    array, shape (n_lat, n_lon)
    """
    n = weights.n
    field = np.asarray(field_faces, dtype=np.float64).reshape(6, n, n)
    padded = _pad_field_for_regrid(field, n)  # (6, n+2, n+2)

    i1 = weights.i0 + 1
    j1 = weights.j0 + 1

    v00 = padded[weights.face, weights.i0, weights.j0]
    v10 = padded[weights.face, i1, weights.j0]
    v01 = padded[weights.face, weights.i0, j1]
    v11 = padded[weights.face, i1, j1]

    result = (v00 * (1.0 - weights.wi) * (1.0 - weights.wj)
              + v10 * weights.wi * (1.0 - weights.wj)
              + v01 * (1.0 - weights.wi) * weights.wj
              + v11 * weights.wi * weights.wj)

    return result.reshape(weights.n_lat, weights.n_lon)


# ---------------------------------------------------------------------------
# Corner-based regridding (smooth wind fields at face boundaries)
# ---------------------------------------------------------------------------


def _pad_corner_field_for_regrid(field: np.ndarray, n: int) -> np.ndarray:
    """Pad (6, n+1, n+1) D-grid corner data → (6, n+3, n+3).

    Corner indices 0 and n sit ON the face boundary (shared with the
    neighbouring face), so the halo copies the *second-from-boundary*
    corner on the neighbour (``strip_inset=1``); see
    :func:`_pad_faces_for_regrid`.
    """
    return _pad_faces_for_regrid(field, strip_inset=1)


def apply_cubedsphere_corners_to_latlon(
    corner_field: np.ndarray,
    weights: CubedSphereToLatLonWeights,
) -> np.ndarray:
    """Regrid D-grid corner data (6, n+1, n+1) → (n_lat, n_lon).

    1. Pad corners with cross-face halo  → (6, n+3, n+3)
    2. Average 4 corners → padded cell centres (6, n+2, n+2)
    3. Apply cell-centre bilinear weights

    Boundary corners are synchronized across faces, so the resulting
    cell-centre field is seamless at face boundaries — no edge artifacts.
    """
    n = weights.n
    corner = np.asarray(corner_field, dtype=np.float64).reshape(6, n + 1, n + 1)
    padded_corners = _pad_corner_field_for_regrid(corner, n)

    padded = 0.25 * (
        padded_corners[:, :-1, :-1] + padded_corners[:, 1:, :-1]
        + padded_corners[:, :-1, 1:] + padded_corners[:, 1:, 1:]
    )  # (6, n+2, n+2) — same layout as _pad_field_for_regrid output

    i1 = weights.i0 + 1
    j1 = weights.j0 + 1

    v00 = padded[weights.face, weights.i0, weights.j0]
    v10 = padded[weights.face, i1, weights.j0]
    v01 = padded[weights.face, weights.i0, j1]
    v11 = padded[weights.face, i1, j1]

    result = (v00 * (1.0 - weights.wi) * (1.0 - weights.wj)
              + v10 * weights.wi * (1.0 - weights.wj)
              + v01 * (1.0 - weights.wi) * weights.wj
              + v11 * weights.wi * weights.wj)

    return result.reshape(weights.n_lat, weights.n_lon)


def apply_cubedsphere_to_latlon_3d(
    field_faces: np.ndarray,
    weights: CubedSphereToLatLonWeights,
) -> np.ndarray:
    """Regrid (6, n, n, nlev) → (n_lat, n_lon, nlev) using face-aware bilinear.

    Parameters
    ----------
    field_faces : array, shape (6, n, n, nlev) or (6*n*n, nlev)
        Source cubed-sphere 3-D field.
    weights : CubedSphereToLatLonWeights
        Precomputed weights.

    Returns
    -------
    array, shape (n_lat, n_lon, nlev)
    """
    arr = np.asarray(field_faces, dtype=np.float64)
    n = weights.n
    if arr.ndim == 2:
        nlev = arr.shape[-1]
        arr = arr.reshape(6, n, n, nlev)
    elif arr.ndim == 4:
        nlev = arr.shape[-1]
    else:
        raise ValueError(f"Expected 2-D or 4-D input, got shape {arr.shape}")

    out = np.empty((weights.n_lat, weights.n_lon, nlev), dtype=np.float64)
    for k in range(nlev):
        out[..., k] = apply_cubedsphere_to_latlon(arr[..., k], weights)
    return out


# ==============================================================================
# Unstructured (SCVT / Voronoi cell) → lat-lon regridding for CMIP output
# ==============================================================================


class VoronoiToLatLonWeights(NamedTuple):
    """Inverse-distance k-nearest weights for SCVT/Voronoi cell → lat-lon.

    For each target lat-lon point, ``idx`` holds the ``k`` nearest source
    cell indices and ``w`` the normalised inverse-(chord-)distance weights
    (rows sum to 1).  k-nearest IDW (not conservative) — adequate for CMIP
    diagnostic output on an unstructured mesh, where the alternatives
    (nearest = blocky; conservative = needs cell polygons) trade simplicity
    for marginal accuracy.  Mirror of :func:`compute_cs_to_gauss_weights`'s
    KD-tree + IDW pattern, specialised to a regular lat-lon target.

    Fields
    ------
    idx : int64 array, shape (n_target, k) — source cell indices.
    w : float64 array, shape (n_target, k) — IDW weights, rows sum to 1.
    n_lat, n_lon : int — target grid shape (row-major ``(n_lat, n_lon)``).
    lat_cent, lon_cent : 1-D arrays [deg] — target cell centres.
    """

    idx: np.ndarray
    w: np.ndarray
    n_lat: int
    n_lon: int
    lat_cent: np.ndarray
    lon_cent: np.ndarray


def compute_voronoi_to_latlon_weights(
    lat_cell: np.ndarray,
    lon_cell: np.ndarray,
    n_lon: int = 360,
    n_lat: int = 181,
    k: int = 3,
    lat_cent: np.ndarray | None = None,
) -> VoronoiToLatLonWeights:
    """Precompute IDW k-nearest weights from Voronoi cell centres to lat-lon.

    Parameters
    ----------
    lat_cell, lon_cell : array, shape (nCells,)
        Voronoi cell-centre latitude / longitude **in radians** (the
        ``VoronoiMesh.latCell`` / ``lonCell`` convention).
    n_lon, n_lat : int
        Output regular lat-lon grid dimensions.
    k : int
        Number of nearest source cells per target point (clamped to nCells).
    lat_cent : 1-D array, optional
        Row latitudes [deg] to sample at (default pole-to-pole linspace).
    """
    from scipy.spatial import cKDTree

    lon_cent = np.linspace(-180.0, 180.0, n_lon, endpoint=False) + 180.0 / n_lon
    lat_cent = _target_lat(n_lat, lat_cent)
    lon2d, lat2d = np.meshgrid(lon_cent, lat_cent)

    src_xyz = _latlon_to_xyz(np.asarray(lat_cell, dtype=np.float64),
                             np.asarray(lon_cell, dtype=np.float64))
    tgt_xyz = _latlon_to_xyz(np.deg2rad(lat2d.ravel()),
                             np.deg2rad(lon2d.ravel()))

    k = int(min(k, src_xyz.shape[0]))
    tree = cKDTree(src_xyz)
    dist, idx = tree.query(tgt_xyz, k=k)
    if k == 1:  # cKDTree drops the trailing axis for k==1
        dist = dist[:, None]
        idx = idx[:, None]
    # Inverse-(chord-)distance weights; eps guards an exact hit.
    w = 1.0 / (dist + 1e-12)
    w /= w.sum(axis=1, keepdims=True)
    return VoronoiToLatLonWeights(
        idx=idx.astype(np.int64), w=w.astype(np.float64),
        n_lat=n_lat, n_lon=n_lon, lat_cent=lat_cent, lon_cent=lon_cent,
    )


def apply_voronoi_to_latlon(
    field_cells: np.ndarray, weights: VoronoiToLatLonWeights,
) -> np.ndarray:
    """Regrid a cell field ``(nCells,)`` → ``(n_lat, n_lon)`` via IDW."""
    f = np.asarray(field_cells, dtype=np.float64)
    if f.ndim != 1:
        raise ValueError(f"Expected (nCells,) field, got shape {f.shape}")
    vals = f[weights.idx]                       # (n_target, k)
    out = np.einsum("tk,tk->t", vals, weights.w)
    return out.reshape(weights.n_lat, weights.n_lon)


def apply_voronoi_to_latlon_3d(
    field_cells: np.ndarray, weights: VoronoiToLatLonWeights,
) -> np.ndarray:
    """Regrid a cell field ``(nCells, nlev)`` → ``(n_lat, n_lon, nlev)``."""
    f = np.asarray(field_cells, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError(f"Expected (nCells, nlev) field, got shape {f.shape}")
    vals = f[weights.idx]                        # (n_target, k, nlev)
    out = np.einsum("tk,tkl->tl", weights.w, vals)
    return out.reshape(weights.n_lat, weights.n_lon, f.shape[-1])


# ==============================================================================
# Legacy KD-tree cubed-sphere to lat-lon regridding (kept for compatibility)
# ==============================================================================


def regrid_faces_to_latlon(
    field_faces: np.ndarray,
    src_lon_rad: np.ndarray,
    src_lat_rad: np.ndarray,
    n_lon: int | None = None,
    n_lat: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate cubed-sphere face data to a regular lat-lon grid.

    Uses Gaussian (RBF) weighting of K nearest neighbors in 3-D Cartesian
    coordinates on the unit sphere.

    Parameters
    ----------
    field_faces : array
        Data on cubed-sphere faces (e.g. shape ``(6, n, n)``).
    src_lon_rad, src_lat_rad : array
        Source grid longitude/latitude in **radians**, same shape as the
        spatial dimensions of *field_faces*.
    n_lon, n_lat : int, optional
        Output grid size.  Defaults to ``max(360, 8*N)`` and ``n_lon // 2``
        where *N* is inferred from the face tile dimension.

    Returns
    -------
    lon_cent : 1-D array, shape (n_lon,)
        Longitude centres [degrees].
    lat_cent : 1-D array, shape (n_lat,)
        Latitude centres [degrees].
    field_ll : 2-D array, shape (n_lat, n_lon)
        Regridded field.
    """
    from scipy.spatial import cKDTree

    # Infer default output resolution from face tile size
    face_shape = np.asarray(field_faces).shape
    n_tile = face_shape[1] if len(face_shape) >= 3 else int(np.sqrt(face_shape[0] / 6))
    if n_lon is None:
        n_lon = max(360, 8 * n_tile)
    if n_lat is None:
        n_lat = n_lon // 2

    cube_lon_deg = np.asarray(src_lon_rad, dtype=np.float64) * 180.0 / np.pi
    cube_lat_deg = np.asarray(src_lat_rad, dtype=np.float64) * 180.0 / np.pi

    lon = cube_lon_deg.reshape(-1)
    lat = cube_lat_deg.reshape(-1)
    val = np.asarray(field_faces, dtype=np.float64).reshape(-1)

    valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(val)
    lon = ((lon[valid] + 180.0) % 360.0) - 180.0
    lat = np.clip(lat[valid], -90.0, 90.0)
    val = val[valid]

    lon_cent = np.linspace(-180.0, 180.0, n_lon, endpoint=False) + 180.0 / n_lon
    lat_cent = np.linspace(-90.0, 90.0, n_lat)
    lon2d, lat2d = np.meshgrid(lon_cent, lat_cent)

    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    cos_lat = np.cos(lat_rad)
    src_xyz = np.column_stack(
        [cos_lat * np.cos(lon_rad), cos_lat * np.sin(lon_rad), np.sin(lat_rad)],
    )

    lon_t = np.deg2rad(lon2d.reshape(-1))
    lat_t = np.deg2rad(lat2d.reshape(-1))
    cos_lat_t = np.cos(lat_t)
    tgt_xyz = np.column_stack(
        [cos_lat_t * np.cos(lon_t), cos_lat_t * np.sin(lon_t), np.sin(lat_t)],
    )

    k = min(16, src_xyz.shape[0])
    tree = cKDTree(src_xyz)
    dist, idx = tree.query(tgt_xyz, k=k)
    if k == 1:
        field_ll = val[idx].reshape(lon2d.shape)
    else:
        dist = np.maximum(dist, 1.0e-12)
        sigma = np.median(dist[:, 0]) * 2.0
        w = np.exp(-0.5 * (dist / sigma) ** 2)
        w /= np.sum(w, axis=1, keepdims=True)
        field_ll = np.sum(val[idx] * w, axis=1).reshape(lon2d.shape)

    return lon_cent, lat_cent, field_ll


def regrid_scalar_nan_aware(
    field: jnp.ndarray,
    regrid_weights: RegridWeights,
) -> jnp.ndarray:
    """NaN-aware version of :func:`regrid_scalar` (KD-tree IDW).

    Missing source cells (``NaN``) are dropped from each target's neighbour set
    and the inverse-distance weights renormalised over the valid neighbours, so a
    target becomes ``NaN`` only when *all* its neighbours are missing.  For
    land-only source data (ocean = NaN) this stops ocean NaN bleeding into coastal
    target cells — the same role conservative regridding plays for regular
    lat-lon, but for arbitrary (cubed-sphere / MPAS) targets via point neighbours.
    """
    spatial_size = regrid_weights.src_flat_size
    if field.size == spatial_size:
        flat = field.ravel(); extra_dims = ()
    else:
        n_trailing = field.size // spatial_size
        flat = field.reshape(spatial_size, n_trailing); extra_dims = flat.shape[1:]

    idx = regrid_weights.src_indices               # (n_target, k)
    w = regrid_weights.weights                     # (n_target, k)
    gathered = flat[idx]                            # (..., k[, n_extra])
    if len(extra_dims) == 0:
        valid = jnp.isfinite(gathered)             # (n_target, k)
        wv = w * valid
        num = jnp.sum(jnp.where(valid, gathered, 0.0) * wv, axis=-1)
        den = jnp.sum(wv, axis=-1)
        res = jnp.where(den > 0.0, num / jnp.where(den > 0.0, den, 1.0), jnp.nan)
        return res.reshape(regrid_weights.target_shape)
    else:
        valid = jnp.isfinite(gathered)             # (n_target, k, n_extra)
        wv = w[..., None] * valid
        num = jnp.sum(jnp.where(valid, gathered, 0.0) * wv, axis=-2)
        den = jnp.sum(wv, axis=-2)
        res = jnp.where(den > 0.0, num / jnp.where(den > 0.0, den, 1.0), jnp.nan)
        return res.reshape(regrid_weights.target_shape + extra_dims)


def fill_missing_nearest_valid(
    data: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    """Fill ``NaN`` entries from the nearest valid point, slice by slice.

    Each row of ``data`` is filled INDEPENDENTLY: a missing entry takes the
    value of the nearest point that is valid *in that same row*.  Two uses in
    the model share this: an AMIP forcing frame filled from the nearest
    unmasked cell at that same time, and an observed T/S level filled from the
    nearest source column that has an observation at that same depth.  Filling
    per row is what makes the second one correct -- a column-wise fill would
    carry a shallow value down into levels it was never observed at.

    Distances are Euclidean in the supplied coordinates.  Passing unit-sphere
    Cartesian coordinates (``x, y, z``) makes the ordering exact on the globe
    with no dateline or pole seam, which is why callers convert lat/lon rather
    than differencing degrees.

    Parameters
    ----------
    data : np.ndarray
        Shape ``(n_slices, n_points)``.  ``NaN`` marks a missing value.
    coords : np.ndarray
        Point coordinates, shape ``(n_points, n_dim)``; unit-sphere Cartesian
        (``n_dim = 3``) for geographic data.

    Returns
    -------
    np.ndarray
        ``data`` with every ``NaN`` replaced, same shape.

    Raises
    ------
    ValueError
        If a slice is entirely ``NaN``.  There is no donor for it, and
        returning it unchanged would leak ``NaN`` into whatever consumes the
        field with no message at all.
    """
    data = np.asarray(data)
    coords = np.asarray(coords)
    if data.ndim != 2:
        raise ValueError(
            f"data must be 2-D (n_slices, n_points); got shape {data.shape}."
        )
    if coords.shape[0] != data.shape[1]:
        raise ValueError(
            f"coords has {coords.shape[0]} points but data has "
            f"{data.shape[1]} per slice."
        )
    if not np.any(np.isnan(data)):
        return data

    from scipy.interpolate import NearestNDInterpolator

    filled = data.copy()
    for i in range(data.shape[0]):
        frame = data[i]
        mask_valid = ~np.isnan(frame)
        if not mask_valid.any():
            raise ValueError(
                f"nearest-valid fill: slice {i} of {data.shape[0]} is "
                "entirely NaN/missing, so there is no valid point to fill it "
                "from. Check the source field for an all-masked time record "
                "or depth level."
            )
        if mask_valid.all():
            continue
        interp = NearestNDInterpolator(coords[mask_valid], frame[mask_valid])
        filled[i] = interp(coords)

    return filled


def _cell_edges(centers):
    """Cell edges (n+1) from 1-D cell centres (works for ascending or descending)."""
    c = np.asarray(centers, dtype=np.float64)
    mid = 0.5 * (c[:-1] + c[1:])
    return np.concatenate([[2.0 * c[0] - mid[0]], mid, [2.0 * c[-1] - mid[-1]]])


def _overlap_matrix(src_edges, tgt_edges, periodic_span=None):
    """``(n_tgt, n_src)`` overlap length of each target cell with each source cell.

    Cells are taken as ``[min(edge_i, edge_{i+1}), max(...)]`` so the result is
    orientation-independent.  ``periodic_span`` (e.g. 360 for longitude) adds the
    wrapped overlaps so cells straddling the seam are handled.
    """
    s_lo = np.minimum(src_edges[:-1], src_edges[1:]); s_hi = np.maximum(src_edges[:-1], src_edges[1:])
    t_lo = np.minimum(tgt_edges[:-1], tgt_edges[1:]); t_hi = np.maximum(tgt_edges[:-1], tgt_edges[1:])

    def _ov(shift):
        lo = np.maximum(t_lo[:, None], s_lo[None, :] + shift)
        hi = np.minimum(t_hi[:, None], s_hi[None, :] + shift)
        return np.clip(hi - lo, 0.0, None)

    if periodic_span:
        return _ov(0.0) + _ov(periodic_span) + _ov(-periodic_span)
    return _ov(0.0)


def conservative_regrid_latlon(field, src_lat, src_lon, tgt_lat, tgt_lon):
    """First-order **conservative**, NaN-aware regrid between regular lat-lon grids.

    Each target cell value is the source-cell-area-weighted mean over the source
    cells it overlaps, using ``sin(lat)`` (true area) for the latitude weight and
    periodic longitude overlap.  **NaN-aware**: missing source cells (e.g. ocean)
    are dropped and the weights renormalised over the valid overlap, so NaNs never
    bleed into a partially-covered (coastal) target cell — a target is NaN only
    when *all* its overlapping source cells are missing.  Conserves the
    area-integral over the valid region.

    Parameters
    ----------
    field : array ``(n_src_lat, n_src_lon)`` or ``(n_src_lat, n_src_lon, L)``
    src_lat, src_lon, tgt_lat, tgt_lon : 1-D cell centres [deg]

    Returns
    -------
    array ``(n_tgt_lat, n_tgt_lon[, L])``
    """
    field = np.asarray(field, dtype=np.float64)
    has_layers = field.ndim == 3
    v = field if has_layers else field[:, :, None]            # (ns_lat, ns_lon, L)

    # Latitude weight uses sin(lat) (true cell-area measure); longitude is periodic.
    # Edges clipped to the poles: a pole-centred row extrapolates to +/-90.5,
    # where sin folds back and the cap would get zero width.
    def _sin_lat_edges(lat):
        return np.sin(np.deg2rad(np.clip(_cell_edges(lat), -90.0, 90.0)))
    w_lat = _overlap_matrix(_sin_lat_edges(src_lat),
                            _sin_lat_edges(tgt_lat))   # (n_tgt_lat, n_src_lat)
    w_lon = _overlap_matrix(_cell_edges(src_lon), _cell_edges(tgt_lon),
                            periodic_span=360.0)                        # (n_tgt_lon, n_src_lon)

    valid = np.isfinite(v).astype(np.float64)
    fv = np.where(valid > 0, v, 0.0)
    # contract source lat then source lon (separable -> cheap)
    num = np.einsum("bj,aj L -> ab L", w_lon, np.einsum("as,sj L -> aj L", w_lat, fv))
    den = np.einsum("bj,aj L -> ab L", w_lon, np.einsum("as,sj L -> aj L", w_lat, valid))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0.0, num / np.maximum(den, 1e-300), np.nan)
    return out if has_layers else out[:, :, 0]
