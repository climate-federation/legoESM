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
