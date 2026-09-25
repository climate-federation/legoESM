"""Shared Neumann-fill helper for MPAS ocean grids.

Replaces land-cell values with the average of ocean-neighbor values,
preventing spurious gradients at coastlines.  Used by the baroclinic
tendencies, barotropic solver, and model step function.
"""

from __future__ import annotations

import jax.numpy as jnp


def fill_land_cells_mpas(
    field_cell: jnp.ndarray,
    mask_cell: jnp.ndarray,
    c1: jnp.ndarray,
    c2: jnp.ndarray,
    edges_on_cell: jnp.ndarray,
    n_edges_on_cell: jnp.ndarray,
    n_iter: int = 3,
) -> jnp.ndarray:
    """Replace land-cell values with ocean-neighbor average (Neumann BC).

    Iterates the single-edge Neumann fill ``n_iter`` times, updating a
    local copy of the mask after each pass so that land cells that
    received an ocean-neighbor value in iteration *k* act as ocean for
    iteration *k+1*.  After ``n_iter`` iterations, land cells up to
    ``n_iter`` edges from any ocean cell have been filled; deeper
    interior land cells are left unchanged.

    Matches the 3-iteration behaviour of ``neumann_fill_cgrid`` in
    ``ocean_pe_latlon_cgrid.py``.  A single iteration (the previous
    default) left any land cell more than one edge from ocean at its
    stale value, which combined with the post-step tracer masking in
    ``ocean_model_mpas.py`` to produce cold/fresh fronts that
    propagated one cell per step along coastlines.  See issue #164.

    Each cell GATHERS over its own edges (``edges_on_cell`` rows below
    ``n_edges_on_cell``), taking the far-end cell of each edge from
    ``c1``/``c2``, so the neighbour set is exactly the set of edges the
    former per-edge scatter-add visited.  The gather is ~3x faster on CPU
    (serial scatter; measured 38 -> 13 ms, s6 x 40 levels, 8 cores); only
    the summation order changes (last-bit differences on filled cells).

    Parameters
    ----------
    field_cell : (nCells,) or (nCells, nlev)
        Values to fill on land cells.
    mask_cell : (nCells,)
        1 on ocean, 0 on land.  Only a local copy is mutated.
    c1, c2 : (nEdges,)
        ``cellsOnEdge`` connectivity.
    edges_on_cell : (maxEdges, nCells)
        ``edgesOnCell`` connectivity.
    n_edges_on_cell : (nCells,)
        Number of valid ``edges_on_cell`` rows per cell.
    n_iter : int
        Number of Neumann-fill iterations (default 3).  Each iteration
        propagates ocean values one edge further into land.

    Returns
    -------
    filled : same shape as ``field_cell``
    """
    n_cells = field_cell.shape[0]
    e1 = c1[edges_on_cell]                            # (maxEdges, nCells)
    e2 = c2[edges_on_cell]
    here = jnp.arange(n_cells)[jnp.newaxis, :]
    nbr = jnp.where(e1 == here, e2, e1)               # far-end cell per edge
    valid = (jnp.arange(edges_on_cell.shape[0])[:, jnp.newaxis]
             < n_edges_on_cell[jnp.newaxis, :])

    filled = field_cell
    m = mask_cell

    for _ in range(n_iter):
        # Accumulate one edge slot at a time, so no (maxEdges, nCells, nlev)
        # temporary is ever materialized.
        cnt = jnp.zeros_like(m)
        vsum = jnp.zeros_like(filled)
        for k in range(edges_on_cell.shape[0]):
            j, ok = nbr[k], valid[k]
            cnt = cnt + jnp.where(ok, m[j], 0.0)
            if filled.ndim == 1:
                vsum = vsum + jnp.where(ok, filled[j] * m[j], 0.0)
            else:
                vsum = vsum + jnp.where(ok[:, jnp.newaxis],
                                        filled[j] * m[j][:, jnp.newaxis], 0.0)
        has_nbr = cnt > 0.0
        if filled.ndim == 1:
            nbr_avg = vsum / jnp.maximum(cnt, 1.0)
            can_fill = (m < 0.5) & has_nbr
        else:
            nbr_avg = vsum / jnp.maximum(cnt[:, jnp.newaxis], 1.0)
            can_fill = ((m < 0.5) & has_nbr)[:, jnp.newaxis]

        filled = jnp.where(can_fill, nbr_avg, filled)

        # Update the local mask: land cells that got filled become
        # "ocean" for the next iteration, so the fill propagates.
        m = jnp.where((m < 0.5) & has_nbr, 1.0, m)

    return filled
