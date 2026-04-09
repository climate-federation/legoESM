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
) -> jnp.ndarray:
    """Replace land-cell values with ocean-neighbor average.

    Parameters
    ----------
    field_cell : (nCells,) or (nCells, nlev)
    mask_cell : (nCells,)  1=ocean, 0=land
    c1, c2 : (nEdges,)  cellsOnEdge connectivity

    Returns
    -------
    filled : same shape as field_cell
    """
    nbr_sum = jnp.zeros_like(field_cell)
    nbr_cnt = jnp.zeros_like(field_cell)
    if field_cell.ndim == 1:
        nbr_sum = nbr_sum.at[c1].add(field_cell[c2] * mask_cell[c2])
        nbr_cnt = nbr_cnt.at[c1].add(mask_cell[c2])
        nbr_sum = nbr_sum.at[c2].add(field_cell[c1] * mask_cell[c1])
        nbr_cnt = nbr_cnt.at[c2].add(mask_cell[c1])
    else:
        m2 = mask_cell[c2, jnp.newaxis]
        m1 = mask_cell[c1, jnp.newaxis]
        nbr_sum = nbr_sum.at[c1].add(field_cell[c2] * m2)
        nbr_cnt = nbr_cnt.at[c1].add(m2)
        nbr_sum = nbr_sum.at[c2].add(field_cell[c1] * m1)
        nbr_cnt = nbr_cnt.at[c2].add(m1)
    nbr_avg = nbr_sum / jnp.maximum(nbr_cnt, 1.0)
    mask_e = mask_cell if field_cell.ndim == 1 else mask_cell[:, jnp.newaxis]
    return jnp.where(mask_e > 0.5, field_cell, nbr_avg)
