"""Regridding between CubedSphere and Gaussian grids.

Provides offline weight computation (numpy, done once at init) and
JIT-compatible application for field transfer between grids.
Supports scalar fields and vector fields with rotation correction.

Methods:
- Bilinear interpolation using inverse-distance weighting of
  K nearest neighbors (KD-tree lookup in lat-lon space).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

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
    from scipy.spatial import cKDTree

    # Source points: flatten cubed-sphere (6, n, n) → (6*n*n, 3)
    src_lat = np.asarray(cs_grid.lat).ravel()
    src_lon = np.asarray(cs_grid.lon).ravel()
    src_xyz = _latlon_to_xyz(src_lat, src_lon)

    # Target points: Gaussian grid (n_lat, n_lon) → (n_lat*n_lon, 3)
    tgt_lat = np.asarray(gauss_grid.lat2d).ravel()
    tgt_lon = np.asarray(gauss_grid.lon2d).ravel()
    tgt_xyz = _latlon_to_xyz(tgt_lat, tgt_lon)

    # KD-tree lookup
    tree = cKDTree(src_xyz)
    distances, indices = tree.query(tgt_xyz, k=k_neighbors)

    # Inverse-distance weights
    # Guard against exact matches (distance = 0)
    distances = np.maximum(distances, 1e-12)
    inv_dist = 1.0 / distances
    weights = inv_dist / inv_dist.sum(axis=-1, keepdims=True)

    target_shape = (gauss_grid.n_lat, gauss_grid.n_lon)
    src_flat_size = int(np.prod(np.array(cs_grid.lat.shape)))

    return RegridWeights(
        src_indices=jnp.array(indices, dtype=jnp.int32),
        weights=jnp.array(weights, dtype=jnp.float64),
        target_shape=target_shape,
        src_flat_size=src_flat_size,
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
    from scipy.spatial import cKDTree

    # Source: Gaussian grid
    src_lat = np.asarray(gauss_grid.lat2d).ravel()
    src_lon = np.asarray(gauss_grid.lon2d).ravel()
    src_xyz = _latlon_to_xyz(src_lat, src_lon)

    # Target: cubed-sphere
    tgt_lat = np.asarray(cs_grid.lat).ravel()
    tgt_lon = np.asarray(cs_grid.lon).ravel()
    tgt_xyz = _latlon_to_xyz(tgt_lat, tgt_lon)

    tree = cKDTree(src_xyz)
    distances, indices = tree.query(tgt_xyz, k=k_neighbors)

    distances = np.maximum(distances, 1e-12)
    inv_dist = 1.0 / distances
    weights = inv_dist / inv_dist.sum(axis=-1, keepdims=True)

    target_shape = tuple(int(s) for s in cs_grid.lat.shape)
    src_flat_size = gauss_grid.n_lat * gauss_grid.n_lon

    return RegridWeights(
        src_indices=jnp.array(indices, dtype=jnp.int32),
        weights=jnp.array(weights, dtype=jnp.float64),
        target_shape=target_shape,
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
    trailing_shape = field.shape[len(field.shape) - (field.size // spatial_size):]

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
    weights = regrid_weights.weights        # (n_target, k)

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
# Cubed-sphere to regular lat-lon regridding (for output / plotting)
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
    N_tile = face_shape[1] if len(face_shape) >= 3 else int(np.sqrt(face_shape[0] / 6))
    if n_lon is None:
        n_lon = max(360, 8 * N_tile)
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
