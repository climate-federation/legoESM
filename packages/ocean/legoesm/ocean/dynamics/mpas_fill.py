"""Shared Neumann-fill helper for MPAS ocean grids.

Replaces land-cell values with the average of ocean-neighbor values,
preventing spurious gradients at coastlines.  Used by the baroclinic
tendencies, barotropic solver, and model step function.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def fill_land_cells_mpas(
    field_cell: jnp.ndarray,
    mask_cell: jnp.ndarray,
    c1: jnp.ndarray,
    c2: jnp.ndarray,
    edges_on_cell: jnp.ndarray,
    n_edges_on_cell: jnp.ndarray,
    n_iter: int = 3,
    band_capacity: int | None = None,
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
    band_capacity : int or None
        None: every pass runs over all cells.  An int K: the passes run only
        over the land cells within ``n_iter`` edges of ocean (the only cells
        any pass writes), found from ``mask_cell`` on every call and padded
        to K; same arithmetic per cell, so the result is identical.  If
        more than K such cells exist, the all-cells passes run instead.

    Returns
    -------
    filled : same shape as ``field_cell``
    """
    n_cells = field_cell.shape[0]
    nbr, valid = _neighbours(c1, c2, edges_on_cell, n_edges_on_cell)
    if band_capacity is None or n_iter == 0:
        return _fill_passes(field_cell, mask_cell, nbr, valid, n_iter, None)

    band = _band(mask_cell, nbr, valid, n_iter)
    idx = jnp.nonzero(band, size=band_capacity, fill_value=n_cells)[0]
    return jax.lax.cond(
        jnp.sum(band) <= band_capacity,
        lambda f: _fill_passes(f, mask_cell, nbr, valid, n_iter, idx),
        lambda f: _fill_passes(f, mask_cell, nbr, valid, n_iter, None),
        field_cell)


def _neighbours(c1, c2, edges_on_cell, n_edges_on_cell):
    """Far-end cell of every (edge slot, cell) and whether it is a neighbour."""
    n_cells = edges_on_cell.shape[1]
    # A sharded local mesh marks edges the rank does not hold with -1
    # (voronoi_partition remap); those slots are not neighbours here, just as
    # the edge scatter never visited them. Index them safely, then mask.
    # Invariant relied on: an edge is -1 only when the rank does not hold it,
    # so it was absent from the local cellsOnEdge the scatter summed over.
    # Checked 2026-09-26 on 4-device s4 partitions: -1 slots occur, and the
    # old scatter and this gather agree to <= 4e-16 on every sharded call.
    held = edges_on_cell >= 0
    eoc = jnp.where(held, edges_on_cell, 0)
    e1 = c1[eoc]                                      # (maxEdges, nCells)
    e2 = c2[eoc]
    here = jnp.arange(n_cells)[jnp.newaxis, :]
    nbr = jnp.where(e1 == here, e2, e1)               # far-end cell per edge
    # The MPI-per-rank partition keeps an edge when EITHER cell is local and
    # marks the missing far cell -1; that is not a neighbour here (it used to
    # index the last local cell).
    valid = held & (nbr >= 0) & (jnp.arange(edges_on_cell.shape[0])[:, jnp.newaxis]
                                 < n_edges_on_cell[jnp.newaxis, :])
    nbr = jnp.where(valid, nbr, 0)
    return nbr, valid


def _band(mask_cell, nbr, valid, n_iter):
    """Land cells the fill writes: reached from ocean within n_iter passes."""
    m, band = mask_cell, jnp.zeros(mask_cell.shape[0], dtype=bool)
    for _ in range(n_iter):
        cnt = jnp.zeros_like(m)
        for k in range(nbr.shape[0]):
            cnt = cnt + jnp.where(valid[k], m[nbr[k]], 0.0)
        flip = (m < 0.5) & (cnt > 0.0)
        band = band | flip
        m = jnp.where(flip, 1.0, m)
    return band


def land_fill_band(mask_cell, c1, c2, edges_on_cell, n_edges_on_cell,
                   n_iter: int = 3) -> jnp.ndarray:
    """Boolean (nCells,): the cells the land fill writes on this mesh/mask."""
    nbr, valid = _neighbours(jnp.asarray(c1), jnp.asarray(c2),
                             jnp.asarray(edges_on_cell),
                             jnp.asarray(n_edges_on_cell))
    return _band(jnp.asarray(mask_cell), nbr, valid, n_iter)


def land_fill_band_size(mask_cell, c1, c2, edges_on_cell, n_edges_on_cell,
                        n_iter: int = 3) -> int:
    """Smallest ``band_capacity`` that keeps the band path on this mesh/mask."""
    return int(jnp.sum(land_fill_band(mask_cell, c1, c2, edges_on_cell,
                                      n_edges_on_cell, n_iter)))


def _fill_passes(filled, m, nbr, valid, n_iter, idx):
    """The Neumann passes over all cells (``idx`` None) or over the cells
    ``idx`` (entries == nCells are padding: read clipped, never written)."""
    if idx is not None:
        # The all-cells passes promote the field to the mask's dtype (the
        # where() below); the band path must return the same dtype.
        filled = filled.astype(jnp.result_type(filled, m))
        lane = idx < filled.shape[0]
        nbr = jnp.take(nbr, idx, axis=1, mode="clip")
        valid = jnp.take(valid, idx, axis=1, mode="clip") & lane[jnp.newaxis, :]

    for _ in range(n_iter):
        m_t = m if idx is None else jnp.take(m, idx, mode="clip")
        f_t = filled if idx is None else jnp.take(filled, idx, axis=0, mode="clip")
        # Accumulate one edge slot at a time, so no (maxEdges, nCells, nlev)
        # temporary is ever materialized.
        cnt = jnp.zeros_like(m_t)
        vsum = jnp.zeros_like(f_t)
        for k in range(nbr.shape[0]):
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
            can_fill = (m_t < 0.5) & has_nbr
        else:
            nbr_avg = vsum / jnp.maximum(cnt[:, jnp.newaxis], 1.0)
            can_fill = ((m_t < 0.5) & has_nbr)[:, jnp.newaxis]

        f_t = jnp.where(can_fill, nbr_avg, f_t)
        # Land cells that got filled become "ocean" for the next pass, so
        # the fill propagates.
        m_t = jnp.where((m_t < 0.5) & has_nbr, 1.0, m_t)
        if idx is None:
            filled, m = f_t, m_t
        else:
            filled = filled.at[idx].set(f_t, mode="drop")
            m = m.at[idx].set(m_t, mode="drop")

    return filled
