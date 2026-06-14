"""TVD tracer advection on MPAS Voronoi meshes.

Implements Van Leer TVD reconstruction for edge-based tracer fluxes on
unstructured Voronoi grids.  The key challenge vs structured grids is
finding the "upwind-of-upwind" cell for each edge — on a structured grid
this is simply the cell at (i-2) or (j-2), but on a Voronoi mesh we must
use the mesh connectivity to locate the cell roughly opposite the donor
across the edge.

Algorithm
---------
For each edge *e* with adjacent cells c1, c2:
  1. Determine the donor cell (upstream of mass flux).
  2. Look up the upwind-of-upwind cell: the neighbor of the donor that
     is roughly "behind" it relative to the edge (found by taking the
     opposite slot in the CCW edge ordering around the cell).
  3. Compute the smoothness ratio ``r = (T_donor - T_upup) / (T_downstream - T_donor)``.
  4. Apply the Van Leer limiter ``phi(r) = (r + |r|) / (1 + |r|)``.
  5. Reconstruct: ``T_face = T_donor + 0.5 * phi(r) * delta``.

The "opposite neighbor" heuristic is geometrically exact for regular
hexagons (edge + 3 slots) and approximate for pentagons (edge + 2 slots).
The limiter safely falls back to upwind (phi=0) for any geometric
inaccuracy, making the scheme robust on all Voronoi meshes.

References
----------
- Van Leer, B. (1979). Towards the ultimate conservative difference
  scheme. V. A second-order sequel to Godunov's method. J. Comput. Phys.
- Skamarock, W. C. & Gassmann, A. (2011). Conservative transport
  schemes for the climatic version of the MPAS dynamical core. MWR.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.grids.voronoi import VoronoiMesh


def compute_upup_cells(mesh: VoronoiMesh) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Precompute upwind-of-upwind cell indices for TVD on edges.

    For each edge *e* with cells (c1, c2), computes:
    - ``upup_pos[e]``: the cell "behind" c1 when flow goes c1 → c2
    - ``upup_neg[e]``: the cell "behind" c2 when flow goes c2 → c1

    Found by locating the edge in the donor cell's CCW edge list and
    taking the neighbor at the opposite slot (half-way around the cell).

    Invalid neighbors (padded -1, land) are replaced with the donor cell
    index, producing ``r = 0`` and thus pure upwind (safe fallback).

    Parameters
    ----------
    mesh : VoronoiMesh

    Returns
    -------
    upup_pos : jnp.ndarray, shape (nEdges,)
        Upwind-of-upwind cell index for positive (c1→c2) flow.
    upup_neg : jnp.ndarray, shape (nEdges,)
        Upwind-of-upwind cell index for negative (c2→c1) flow.
    """
    nEdges = mesh.nEdges
    c1 = mesh.cellsOnEdge[0]  # (nEdges,)
    c2 = mesh.cellsOnEdge[1]  # (nEdges,)

    edge_ids = jnp.arange(nEdges)  # (nEdges,)

    # --- Positive flow: donor = c1, find upup behind c1 ---
    # edgesOnCell[:, c1] has shape (maxEdges, nEdges): each column is
    # the edge list of c1[e].
    edges_of_c1 = mesh.edgesOnCell[:, c1]           # (maxEdges, nEdges)
    match_c1 = (edges_of_c1 == edge_ids[None, :])   # (maxEdges, nEdges)
    pos_in_c1 = jnp.argmax(match_c1, axis=0)        # (nEdges,)

    n1 = mesh.nEdgesOnCell[c1]                       # (nEdges,)
    opposite_idx_c1 = (pos_in_c1 + n1 // 2) % n1    # (nEdges,)

    neighbors_of_c1 = mesh.cellsOnCell[:, c1]        # (maxEdges, nEdges)
    upup_pos = neighbors_of_c1[opposite_idx_c1, edge_ids]  # (nEdges,)

    # Replace invalid (-1) with donor cell (gives r=0 → upwind)
    upup_pos = jnp.where(upup_pos >= 0, upup_pos, c1)

    # --- Negative flow: donor = c2, find upup behind c2 ---
    edges_of_c2 = mesh.edgesOnCell[:, c2]
    match_c2 = (edges_of_c2 == edge_ids[None, :])
    pos_in_c2 = jnp.argmax(match_c2, axis=0)

    n2 = mesh.nEdgesOnCell[c2]
    opposite_idx_c2 = (pos_in_c2 + n2 // 2) % n2

    neighbors_of_c2 = mesh.cellsOnCell[:, c2]
    upup_neg = neighbors_of_c2[opposite_idx_c2, edge_ids]
    upup_neg = jnp.where(upup_neg >= 0, upup_neg, c2)

    return upup_pos, upup_neg


# Canonical Van Leer limiter from core (redundancy audit) — the same kernel the
# lat-lon TVD path uses; aliased to the local private name so call sites are
# unchanged.
from legoesm.core.flux_limiters import (
    grad_safe_ratio,
    ratio_grad_floor,
    van_leer_limiter as _van_leer_limiter,
)


def tvd_tracer_to_edges(
    tr: jnp.ndarray,
    mass_flux: jnp.ndarray,
    mesh: VoronoiMesh,
    upup_pos: jnp.ndarray,
    upup_neg: jnp.ndarray,
    cell_active: jnp.ndarray | None = None,
    limiter_fn=_van_leer_limiter,
) -> jnp.ndarray:
    """TVD interpolation of cell-center tracer to edges.

    ``limiter_fn`` selects the flux limiter family; defaults to Van Leer
    (used by ``tracer_advection="tvd"``). Pass
    :func:`legoesm.ocean.dynamics._flux_limiters.sweby_limiter` for
    Veros-compatible superbee (``tracer_advection="superbee"``).

    Second-order accurate in smooth regions, monotone (no new extrema).
    Falls back to first-order upwind at sharp fronts and boundaries.

    Parameters
    ----------
    tr : jnp.ndarray, shape (nCells, nlev)
        Tracer at cell centers.
    mass_flux : jnp.ndarray, shape (nEdges, nlev)
        Per-layer mass flux on edges [m²/s].  Positive = c1 → c2.
    mesh : VoronoiMesh
    upup_pos : jnp.ndarray, shape (nEdges,)
        Upwind-of-upwind cell for positive flow (from compute_upup_cells).
    upup_neg : jnp.ndarray, shape (nEdges,)
        Upwind-of-upwind cell for negative flow (from compute_upup_cells).
    cell_active : jnp.ndarray, shape (nCells, nlev), optional
        Per-cell, per-level active mask (1=ocean, 0=sub-seafloor).
        When provided, sub-seafloor upup values are replaced with the
        donor cell value, preventing the TVD limiter from seeing
        T=0/S=0 below the seafloor and producing biased fluxes.

    Returns
    -------
    tr_edge : jnp.ndarray, shape (nEdges, nlev)
        TVD-reconstructed tracer at edges (to be multiplied by mass_flux
        for the tracer flux).
    """
    eps = 1e-30

    c1 = mesh.cellsOnEdge[0]  # (nEdges,)
    c2 = mesh.cellsOnEdge[1]  # (nEdges,)

    tr_c1 = tr[c1]            # (nEdges, nlev)
    tr_c2 = tr[c2]            # (nEdges, nlev)

    # --- Positive flow (c1 → c2): donor = c1 ---
    tr_upup_pos = tr[upup_pos]                       # (nEdges, nlev)
    # Replace sub-seafloor upup with donor value → r=0 → pure upwind
    if cell_active is not None:
        active_upup_pos = cell_active[upup_pos]      # (nEdges, nlev)
        tr_upup_pos = jnp.where(active_upup_pos > 0.5, tr_upup_pos, tr_c1)
    t_grad = ratio_grad_floor(tr.dtype)
    delta_pos = tr_c2 - tr_c1                        # downstream - donor
    r_pos = grad_safe_ratio(
        tr_c1 - tr_upup_pos,
        jnp.where(jnp.abs(delta_pos) > eps, delta_pos, eps),
        jnp.abs(delta_pos) > t_grad,
    )
    tr_face_pos = tr_c1 + 0.5 * limiter_fn(r_pos) * delta_pos

    # --- Negative flow (c2 → c1): donor = c2 ---
    tr_upup_neg = tr[upup_neg]                       # (nEdges, nlev)
    if cell_active is not None:
        active_upup_neg = cell_active[upup_neg]      # (nEdges, nlev)
        tr_upup_neg = jnp.where(active_upup_neg > 0.5, tr_upup_neg, tr_c2)
    delta_neg = tr_c1 - tr_c2                        # downstream - donor
    r_neg = grad_safe_ratio(
        tr_c2 - tr_upup_neg,
        jnp.where(jnp.abs(delta_neg) > eps, delta_neg, eps),
        jnp.abs(delta_neg) > t_grad,
    )
    tr_face_neg = tr_c2 + 0.5 * limiter_fn(r_neg) * delta_neg

    # Select based on mass flux direction
    return jnp.where(mass_flux > 0, tr_face_pos, tr_face_neg)
