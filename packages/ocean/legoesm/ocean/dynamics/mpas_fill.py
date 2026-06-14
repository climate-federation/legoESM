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

    Parameters
    ----------
    field_cell : (nCells,) or (nCells, nlev)
        Values to fill on land cells.
    mask_cell : (nCells,)
        1 on ocean, 0 on land.  Only a local copy is mutated.
    c1, c2 : (nEdges,)
        ``cellsOnEdge`` connectivity.
    n_iter : int
        Number of Neumann-fill iterations (default 3).  Each iteration
        propagates ocean values one edge further into land.

    Returns
    -------
    filled : same shape as ``field_cell``
    """
    filled = field_cell
    m = mask_cell

    for _ in range(n_iter):
        # Topology-only neighbor count (1D, level-independent).
        cnt = jnp.zeros_like(m)
        cnt = cnt.at[c1].add(m[c2])
        cnt = cnt.at[c2].add(m[c1])
        has_nbr = cnt > 0.0

        # Weighted value accumulation (matches filled rank).
        vsum = jnp.zeros_like(filled)
        if filled.ndim == 1:
            vsum = vsum.at[c1].add(filled[c2] * m[c2])
            vsum = vsum.at[c2].add(filled[c1] * m[c1])
            nbr_avg = vsum / jnp.maximum(cnt, 1.0)
            can_fill = (m < 0.5) & has_nbr
        else:
            m2 = m[c2, jnp.newaxis]
            m1 = m[c1, jnp.newaxis]
            vsum = vsum.at[c1].add(filled[c2] * m2)
            vsum = vsum.at[c2].add(filled[c1] * m1)
            nbr_avg = vsum / jnp.maximum(cnt[:, jnp.newaxis], 1.0)
            can_fill = ((m < 0.5) & has_nbr)[:, jnp.newaxis]

        filled = jnp.where(can_fill, nbr_avg, filled)

        # Update the local mask: land cells that got filled become
        # "ocean" for the next iteration, so the fill propagates.
        m = jnp.where((m < 0.5) & has_nbr, 1.0, m)

    return filled
