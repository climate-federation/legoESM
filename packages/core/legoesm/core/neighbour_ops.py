"""Neighbour-offset operator layer: divergence, gradient and curl written once.

A :class:`Connectivity` maps every target point (cell, edge or vertex) to its
``K`` source neighbours.  Structured grids realise it with integer index
offsets (``+1``/``-1`` -> array slices, :func:`offsets`); unstructured meshes
with an index table (gather, :func:`table`).  Each differential operator is then
ONE signed neighbour sum, identical for every grid::

    div(F)    = sum_k sign_k * F[n_k] * dv_k * mask_k / area_cell
    grad(phi) = sum_k sign_k * phi[n_k] * 1   * mask_k / dc_edge
    curl(u)   = sum_k sign_k * u[n_k] * dc_k * mask_k / area_vertex

The product order ``sign * value * weight * mask`` and the reduction over the
leading neighbour axis are those of the TRiSK kernels in
:mod:`legoesm.core.operators_voronoi`, so an unstructured mesh reproduces them
exactly.  A field on several staggered families (lat-lon C-grid ``(u, v)``) is a
tuple of arrays; a single-family field may be a bare array.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


class Connectivity(NamedTuple):
    """Target <- source neighbour map with its per-neighbour metric.

    ``neighbours(field)`` returns ``(K, *target_shape)``; ``sign``, ``weight``
    and ``mask`` broadcast against it (``mask`` is 1 for a real neighbour,
    0 for an absent one such as a pole ghost or a padded table slot).
    """

    neighbours: Callable[[Any], jax.Array]
    sign: jax.Array
    weight: jax.Array
    mask: jax.Array


class NeighbourMesh(NamedTuple):
    """The three connectivities and measures the operators need.

    ``e2c`` and ``dc_edge`` hold one entry per staggered edge family.
    """

    c2e: Connectivity
    e2c: tuple[Connectivity, ...]
    v2e: Connectivity
    area_cell: jax.Array
    dc_edge: tuple[jax.Array, ...]
    area_vertex: jax.Array


def _components(field: Any) -> tuple:
    return field if isinstance(field, tuple) else (field,)


def offsets(
    spec: Sequence[tuple[int, int, int]],
    target_shape: tuple[int, int],
    pads: Sequence[tuple[Callable[[jax.Array], jax.Array] | None, tuple[int, int]]],
) -> Callable[[Any], jax.Array]:
    """Structured neighbours from ``(component, dj, di)`` index offsets.

    Neighbour ``(c, dj, di)`` of target ``(j, i)`` is component ``c`` at
    ``(j + dj, i + di)``.  ``pads[c] = (pad_fn, (oj, oi))``: ``pad_fn`` (or
    ``None``) supplies the halo the offsets reach into, and ``(oj, oi)`` is where
    the unpadded origin sits inside the padded array.
    """
    nj, ni = target_shape

    def neighbours(field: Any) -> jax.Array:
        comps = _components(field)
        padded = [comps[c] if fn is None else fn(comps[c])
                  for c, (fn, _) in enumerate(pads)]
        out = []
        for c, dj, di in spec:
            oj, oi = pads[c][1]
            j0, i0 = oj + dj, oi + di
            out.append(padded[c][j0:j0 + nj, i0:i0 + ni])
        return jnp.stack(out)

    return neighbours


def table(index: jax.Array) -> Callable[[Any], jax.Array]:
    """Unstructured neighbours from a ``(K, n_target)`` index table (gather)."""

    def neighbours(field: Any) -> jax.Array:
        return _components(field)[0][index]

    return neighbours


def neighbor_sum(conn: Connectivity, field: Any) -> jax.Array:
    """``sum_k sign_k * field[n_k] * weight_k * mask_k`` over the neighbour axis."""
    return jnp.sum(
        conn.sign * conn.neighbours(field) * conn.weight * conn.mask, axis=0)


def divergence(mesh: NeighbourMesh, flux: Any) -> jax.Array:
    """Cell divergence of an edge-normal flux."""
    return neighbor_sum(mesh.c2e, flux) / mesh.area_cell


def gradient(mesh: NeighbourMesh, phi: jax.Array) -> tuple[jax.Array, ...]:
    """Edge-normal gradient of a cell scalar, one array per edge family."""
    return tuple(neighbor_sum(conn, phi) / dc
                 for conn, dc in zip(mesh.e2c, mesh.dc_edge))


def curl(mesh: NeighbourMesh, u: Any) -> jax.Array:
    """Vertex curl (circulation / area) of an edge-normal velocity."""
    return neighbor_sum(mesh.v2e, u) / mesh.area_vertex
