"""Cubed-sphere edge continuity blending utilities.

These helpers apply a configurable relaxation across connected cube-face
edges to reduce edge-local numerical artifacts in collocated-grid fields.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.grids.halo import CONNECTIVITY, WEST, EAST, SOUTH, NORTH


def _build_unique_edge_pairs() -> tuple[tuple[int, int, int, int, bool], ...]:
    pairs: list[tuple[int, int, int, int, bool]] = []
    seen = set()
    for face in range(6):
        for edge, (nbr_face, nbr_edge, reversed_idx) in CONNECTIVITY[face].items():
            key = tuple(sorted(((face, edge), (nbr_face, nbr_edge))))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((face, edge, nbr_face, nbr_edge, reversed_idx))
    return tuple(pairs)


_UNIQUE_EDGE_PAIRS = _build_unique_edge_pairs()


def _edge_strip(arr: jax.Array, face: int, edge: int, offset: int = 0) -> jax.Array:
    if edge == WEST:
        return arr[face, offset, :]
    if edge == EAST:
        return arr[face, -(offset + 1), :]
    if edge == SOUTH:
        return arr[face, :, offset]
    if edge == NORTH:
        return arr[face, :, -(offset + 1)]
    raise ValueError(f"Unknown edge: {edge}")


def _set_edge_strip(
    arr: jax.Array,
    face: int,
    edge: int,
    strip: jax.Array,
    offset: int = 0,
) -> jax.Array:
    if edge == WEST:
        return arr.at[face, offset, :].set(strip)
    if edge == EAST:
        return arr.at[face, -(offset + 1), :].set(strip)
    if edge == SOUTH:
        return arr.at[face, :, offset].set(strip)
    if edge == NORTH:
        return arr.at[face, :, -(offset + 1)].set(strip)
    raise ValueError(f"Unknown edge: {edge}")


def blend_scalar_cube_edges_2d(arr: jax.Array, strength: float, width: int = 1) -> jax.Array:
    """Blend one scalar cubed-sphere field with shape (6, n, n).

    Parameters
    ----------
    arr : jax.Array
        Scalar cubed-sphere field, shape (6, n, n).
    strength : float
        Edge relaxation strength in [0, 1].
    width : int
        Number of interior strips from each panel edge to relax.
    """
    if strength <= 0.0:
        return arr
    if width <= 0:
        return arr

    out = arr
    n = int(arr.shape[1])
    width_eff = min(int(width), max(1, n // 2))

    for d in range(width_eff):
        w = jnp.asarray(strength / float(d + 1), dtype=arr.dtype)
        one_minus_w = 1.0 - w

        for face, edge, nbr_face, nbr_edge, reversed_idx in _UNIQUE_EDGE_PAIRS:
            a = _edge_strip(out, face, edge, offset=d)
            b = _edge_strip(out, nbr_face, nbr_edge, offset=d)
            if reversed_idx:
                b = b[::-1]

            avg = 0.5 * (a + b)
            a_new = one_minus_w * a + w * avg
            b_new = one_minus_w * b + w * avg

            if reversed_idx:
                b_new = b_new[::-1]

            out = _set_edge_strip(out, face, edge, a_new, offset=d)
            out = _set_edge_strip(out, nbr_face, nbr_edge, b_new, offset=d)
    return out


def blend_scalar_cube_edges(arr: jax.Array, strength: float, width: int = 1) -> jax.Array:
    """Blend scalar cubed-sphere arrays of rank 3/4/5.

    Supported shapes:
    - (6, n, n)
    - (6, n, n, nlev)
    - (6, n, n, nlev, ntr)
    """
    if strength <= 0.0:
        return arr
    if arr.ndim == 3:
        return blend_scalar_cube_edges_2d(arr, strength, width=width)
    if arr.ndim == 4:
        arr_t = jnp.moveaxis(arr, -1, 0)  # (nlev, 6, n, n)
        out_t = jax.vmap(lambda a: blend_scalar_cube_edges_2d(a, strength, width=width))(arr_t)
        return jnp.moveaxis(out_t, 0, -1)
    if arr.ndim == 5:
        lead = arr.shape[3] * arr.shape[4]
        arr_f = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2], lead)
        arr_t = jnp.moveaxis(arr_f, -1, 0)  # (lead, 6, n, n)
        out_t = jax.vmap(lambda a: blend_scalar_cube_edges_2d(a, strength, width=width))(arr_t)
        out_f = jnp.moveaxis(out_t, 0, -1)
        return out_f.reshape(arr.shape)
    raise ValueError(
        "blend_scalar_cube_edges expects rank-3/4/5 cubed-sphere array, "
        f"got shape={arr.shape}"
    )


def blend_vector_cube_edges(
    u: jax.Array,
    v: jax.Array,
    cos_angle: jax.Array,
    sin_angle: jax.Array,
    strength: float,
    width: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Blend horizontal vector components by rotating to geographic axes."""
    if strength <= 0.0:
        return u, v

    if u.ndim == 3:
        cos_a = cos_angle
        sin_a = sin_angle
    elif u.ndim == 4:
        cos_a = cos_angle[..., None]
        sin_a = sin_angle[..., None]
    else:
        raise ValueError(
            "blend_vector_cube_edges expects rank-3/4 vector arrays, "
            f"got u.shape={u.shape}, v.shape={v.shape}"
        )

    u_east = cos_a * u - sin_a * v
    v_north = sin_a * u + cos_a * v

    u_east_blend = blend_scalar_cube_edges(u_east, strength, width=width)
    v_north_blend = blend_scalar_cube_edges(v_north, strength, width=width)

    u_grid = cos_a * u_east_blend + sin_a * v_north_blend
    v_grid = -sin_a * u_east_blend + cos_a * v_north_blend
    return u_grid, v_grid
