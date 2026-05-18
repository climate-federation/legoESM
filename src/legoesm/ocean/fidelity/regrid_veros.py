"""Reference-model regridding helpers (Veros B-grid → legoESM lat-lon C-grid).

The module is named after Veros because it is the immediate consumer, but
its functions are general: they convert any B-grid (corner velocity, cell
tracer) source onto the lat-lon C-grid (face velocity, cell tracer) layout
that the rest of the fidelity layer compares against.

Pure numpy. Heavy lifting (overlap weight computation) is delegated to
:func:`legoesm.grids.conservative_regrid.compute_overlap_weights`, but the
sparse apply step is reimplemented here via :func:`numpy.add.at` so this
module does not pull in JAX at import time.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from legoesm.grids import conservative_regrid as _cr


def regrid_2d_latlon(
    field: np.ndarray,
    src_lat_edges: np.ndarray,
    src_lon_edges: np.ndarray,
    dst_lat_edges: np.ndarray,
    dst_lon_edges: np.ndarray,
) -> np.ndarray:
    """Conservatively regrid a 2-D lat-lon field; pure numpy output."""
    if field.ndim != 2:
        raise ValueError(f"field must be 2-D, got shape {field.shape}")
    weights = _cr.compute_overlap_weights(
        np.asarray(src_lat_edges, dtype=np.float64),
        np.asarray(src_lon_edges, dtype=np.float64),
        np.asarray(dst_lat_edges, dtype=np.float64),
        np.asarray(dst_lon_edges, dtype=np.float64),
    )
    if field.shape != weights.src_shape:
        raise ValueError(
            f"field shape {field.shape} does not match weights.src_shape "
            f"{weights.src_shape}"
        )
    return _apply_2d_numpy(np.asarray(field, dtype=np.float64), weights)


def _apply_2d_numpy(field_2d: np.ndarray, weights) -> np.ndarray:
    src_flat = field_2d.reshape(-1)
    src_idx = np.asarray(weights.src_idx_flat)
    dst_idx = np.asarray(weights.dst_idx_flat)
    w = np.asarray(weights.weights, dtype=np.float64)
    contributions = src_flat[src_idx] * w
    dst_flat = np.zeros(weights.n_dst_cells, dtype=np.float64)
    np.add.at(dst_flat, dst_idx, contributions)
    return dst_flat.reshape(weights.dst_shape)


def bgrid_velocity_to_cgrid(
    u_corner: np.ndarray,
    v_corner: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Average B-grid corner velocities onto C-grid faces.

    Both inputs share shape ``(n_lat + 1, n_lon + 1, ...)`` (corners). On a
    C-grid the zonal velocity ``u`` lives on the east face of each tracer
    cell (shape ``(n_lat, n_lon + 1, ...)``) and the meridional velocity
    ``v`` lives on the north face (shape ``(n_lat + 1, n_lon, ...)``).
    Adjacent corners are averaged to land on the matching face.
    """
    if u_corner.shape != v_corner.shape:
        raise ValueError(
            f"u/v_corner shape mismatch: {u_corner.shape} vs {v_corner.shape}"
        )
    if u_corner.ndim < 2:
        raise ValueError("u_corner must have at least 2 dims (lat, lon)")
    n_lat_c, n_lon_c = u_corner.shape[:2]
    if n_lat_c < 2 or n_lon_c < 2:
        raise ValueError("need >= 2 corner samples along each horizontal axis")
    u_face_east = 0.5 * (u_corner[:-1, :, ...] + u_corner[1:, :, ...])
    v_face_north = 0.5 * (v_corner[:, :-1, ...] + v_corner[:, 1:, ...])
    return u_face_east, v_face_north


def interp_to_target_z(
    field: np.ndarray,
    z_src: np.ndarray,
    z_dst: np.ndarray,
) -> np.ndarray:
    """Per-column linear interpolation along the vertical axis (axis 0)."""
    z_src = np.asarray(z_src, dtype=float)
    z_dst = np.asarray(z_dst, dtype=float)
    if z_src.ndim != 1 or z_dst.ndim != 1:
        raise ValueError("z_src and z_dst must be 1-D")
    if z_src.size != field.shape[0]:
        raise ValueError(
            f"field axis-0 length {field.shape[0]} != z_src length {z_src.size}"
        )
    diffs = np.diff(z_src)
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        raise ValueError("z_src must be strictly monotonic")
    if diffs[0] < 0:
        z_src = z_src[::-1]
        field = field[::-1, ...]
    horiz_shape = field.shape[1:]
    flat = field.reshape(field.shape[0], -1)
    out = np.empty((z_dst.size, flat.shape[1]), dtype=float)
    for col in range(flat.shape[1]):
        out[:, col] = np.interp(z_dst, z_src, flat[:, col])
    return out.reshape((z_dst.size, *horiz_shape))


def report_grid_edges_from_centers(centers: np.ndarray) -> np.ndarray:
    """Edges placed midway between adjacent centres; outer edges extrapolated.

    Useful for landing a Veros / NEMO output (which usually only ships cell
    centres) onto :func:`compute_overlap_weights`, which expects edges.
    """
    c = np.asarray(centers, dtype=float)
    if c.ndim != 1 or c.size < 2:
        raise ValueError("centers must be 1-D with at least 2 entries")
    inner = 0.5 * (c[:-1] + c[1:])
    first = c[0] - 0.5 * (c[1] - c[0])
    last = c[-1] + 0.5 * (c[-1] - c[-2])
    return np.concatenate(([first], inner, [last]))
