"""Topography-aware metric helpers for MPAS partial bottom cells.

P2 of the MPAS realistic-geometry plan (see
``docs/ocean_experiments/realistic_geometry_mpas_plan.md``). These
helpers give the partial-cell versions of the continuity, momentum,
and PV-flux operators their per-edge / per-vertex thickness inputs
and active-cell masks.

Naming follows MPAS-O Fortran (Petersen et al. 2015 OM, 2019 JAMES):

* ``maxLevelEdgeBot = min(bot[c1], bot[c2])`` — flux bottom index
  per edge (limits vertical extent of momentum/tracer flux).
* ``maxLevelEdgeTop = max(bot[c1], bot[c2])`` — perturbation-pressure
  top index per edge (Petersen 2015 §3.4 BCL/BTP split: pressure
  perturbation is reconstructed only for k <= maxLevelEdgeTop).
* Edge thickness uses **min-rule** for flux closure (MITgcm hFacZ
  convention) and **donor-cell upstream** for the continuity
  equation's advected thickness — these are different objects.
* Vertex thickness uses the **active-renormalized kite-area mean**
  (Petersen 2015) by default, with a min-over-active fallback near
  coasts (``vertex_thickness_hybrid``) per the dycore audit
  (kite mean alone lets a thin partial vertex transmit O(1) PV
  flux from a deep neighbor and generate spurious coastal flow).
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.grids.voronoi import VoronoiMesh

# Sentinel for dry-cell thickness in min-rule reductions.  Large enough
# that ``min(BIG_H, h_real)`` returns the real thickness for any
# reasonable depth, small enough to avoid overflow under multiplication.
BIG_H = 1.0e20


# ----- Masks ---------------------------------------------------------------


def compute_edge_mask(
    land_mask: jnp.ndarray, mesh: VoronoiMesh,
) -> jnp.ndarray:
    """Edge mask: 1 where both ``cellsOnEdge`` are wet, else 0.

    Returns
    -------
    (nEdges,) same dtype as ``land_mask``.
    """
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    return land_mask[c1] * land_mask[c2]


def compute_vertex_mask(
    land_mask: jnp.ndarray, mesh: VoronoiMesh,
) -> jnp.ndarray:
    """Vertex mask: 1 where every adjacent cell of the kite is wet.

    ``cellsOnVertex`` may be padded with -1; padded entries are
    treated as wet (do not suppress the AND), matching the convention
    used elsewhere by ``vertex_thickness_3d``.

    Returns
    -------
    (nVertices,) same dtype as ``land_mask``.
    """
    cov = mesh.cellsOnVertex  # (vertexDegree, nVertices)
    valid = cov >= 0
    cov_safe = jnp.maximum(cov, 0)
    cell_wet = (land_mask[cov_safe] > 0.5) | ~valid
    return jnp.all(cell_wet, axis=0).astype(land_mask.dtype)


# ----- Per-edge bottom-level indices --------------------------------------


def compute_max_level_edge_bot(
    bottom_level: jnp.ndarray, mesh: VoronoiMesh,
) -> jnp.ndarray:
    """Per-edge ``min(bot[c1], bot[c2])`` — flux bottom index.

    Limits the vertical extent of momentum and tracer flux through the
    edge: the deeper cell has rock below the shallower seafloor, so
    flux closes there.  Returns -1 (or smaller) for edges with a dry
    neighbor (``bottom_level == -1`` for dry cells)."""
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    return jnp.minimum(bottom_level[c1], bottom_level[c2])


def compute_max_level_edge_top(
    bottom_level: jnp.ndarray, mesh: VoronoiMesh,
) -> jnp.ndarray:
    """Per-edge ``max(bot[c1], bot[c2])`` — perturbation top index.

    BCL/BTP split (Petersen 2015 §3.4): the perturbation pressure is
    reconstructed only for k <= maxLevelEdgeTop.  At a step edge,
    this is the bottom of the *deeper* of the two adjacent columns."""
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    return jnp.maximum(bottom_level[c1], bottom_level[c2])


# ----- Edge thickness ------------------------------------------------------


def min_cell_to_edge(
    h_cell_3d: jnp.ndarray, mesh: VoronoiMesh,
) -> jnp.ndarray:
    """Edge thickness by min-rule (MITgcm hFacZ convention).

    Use case: **flux closure** at step edges.  At a face between
    cells of different depth, the face cross-section at level k is
    the lesser of the two cell thicknesses — ``min(h[c1,k], h[c2,k])``.

    Parameters
    ----------
    h_cell_3d : (nCells, nlev)
    mesh : VoronoiMesh

    Returns
    -------
    (nEdges, nlev)
    """
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    return jnp.minimum(h_cell_3d[c1], h_cell_3d[c2])


def donor_cell_to_edge(
    h_cell_3d: jnp.ndarray,
    u_edge_3d: jnp.ndarray,
    mesh: VoronoiMesh,
) -> jnp.ndarray:
    """Upstream donor-cell thickness for the **continuity** equation.

    Petersen 2015 §3.4: at a step edge the *advected* thickness in
    ``div(h u)`` must come from the upstream cell, not a centered
    average.  Centered ``0.5*(h[c1]+h[c2])`` lets the deeper cell's
    thickness leak through a step into the shallower side, violating
    mass conservation and potentially carrying tracers from below the
    seafloor.

    The canonical edge orientation is from ``cellsOnEdge[0] = c1`` to
    ``cellsOnEdge[1] = c2``: positive ``u_edge`` means flow from c1
    to c2, so c1 is the upstream donor.

    Parameters
    ----------
    h_cell_3d : (nCells, nlev)
    u_edge_3d : (nEdges, nlev)  signed by the canonical edge normal.

    Returns
    -------
    (nEdges, nlev)
    """
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    h1 = h_cell_3d[c1]
    h2 = h_cell_3d[c2]
    return jnp.where(u_edge_3d >= 0.0, h1, h2)


# ----- Vertex thickness ----------------------------------------------------


def kite_area_vertex_thickness(
    h_cell_3d: jnp.ndarray,
    mesh: VoronoiMesh,
    vtx_active_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Active-renormalized kite-area-weighted vertex thickness.

    Petersen 2015 / MPAS-O Fortran convention:

        h_v(v, k) = Σ_{wet} kiteArea(k,v) · h[cov(k,v), k]
                   / Σ_{wet} kiteArea(k,v)

    where ``wet`` = (cov(k,v) is a valid cell) AND (h>0 at this level).
    Returns 0 at vertices where no adjacent cell is wet at level k.

    Differs from the existing ``vertex_thickness_3d`` (which divides
    by the **total** triangle area regardless of mask state):
    renormalizing by the *wet* sub-area gives the correct geometric
    average and equals h_cell exactly on uniform-wet regions.

    Parameters
    ----------
    h_cell_3d : (nCells, nlev)
    mesh : VoronoiMesh
    vtx_active_mask : (nVertices,) optional float
        Vertices with mask < 0.5 are zeroed (use to enforce a Boolean
        AND of cell mask on top of the h>0 wet check).

    Returns
    -------
    (nVertices, nlev)
    """
    cov = mesh.cellsOnVertex
    ka = mesh.kiteAreasOnVertex
    cov_valid = cov >= 0
    cov_safe = jnp.maximum(cov, 0)
    h_gathered = h_cell_3d[cov_safe]  # (vertexDegree, nVertices, nlev)
    wet = cov_valid[:, :, None] & (h_gathered > 0.0)
    weights = ka[:, :, None] * wet.astype(h_cell_3d.dtype)
    numer = jnp.sum(weights * h_gathered, axis=0)
    denom = jnp.sum(weights, axis=0)
    h_v = jnp.where(denom > 0.0, numer / jnp.maximum(denom, 1.0e-30), 0.0)
    if vtx_active_mask is not None:
        h_v = h_v * vtx_active_mask[:, None]
    return h_v


def min_cell_to_vertex(
    h_cell_3d: jnp.ndarray,
    mesh: VoronoiMesh,
    vtx_active_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Min-over-active-cell vertex thickness (MITgcm hFacZ convention).

    Used as a coast-fallback in :func:`vertex_thickness_hybrid` and
    as a possible defensive choice for PV-flux normalization on
    pathological configurations.  Padded cells (cov<0) and dry cells
    (h=0) are excluded via a BIG_H sentinel; vertices with no wet
    neighbor return 0.
    """
    cov = mesh.cellsOnVertex
    cov_valid = cov >= 0
    cov_safe = jnp.maximum(cov, 0)
    h_gathered = h_cell_3d[cov_safe]
    valid = cov_valid[:, :, None] & (h_gathered > 0.0)
    h_for_min = jnp.where(valid, h_gathered, BIG_H)
    h_v = jnp.min(h_for_min, axis=0)
    any_wet = jnp.any(valid, axis=0)
    h_v = jnp.where(any_wet, h_v, 0.0)
    if vtx_active_mask is not None:
        h_v = h_v * vtx_active_mask[:, None]
    return h_v


def partial_cell_pgf_correction_edge(
    centroid_depth: jnp.ndarray,
    rho_prime: jnp.ndarray,
    mesh: VoronoiMesh,
    g: float,
    rho_0: float,
) -> jnp.ndarray:
    """Adcroft & Campin (2004) face correction at Voronoi edges.

    Returns an additive correction (units [m/s^2]) that, when added
    to ``gradient_edge(p_prime / rho_0)``, shifts each cell's
    pressure to the face-reference depth (the shallower of the two
    cell centroids) before differencing.  Mathematically equivalent
    to comparing ``p_eff = p - g*rho_prime*excess`` at the shared
    face-reference depth — eliminates the partial-cell-vs-full PGF
    cancellation error that drives spurious flow on realistic
    bathymetry.

    For full-cell columns where centroids align across cells, both
    excesses are zero and the correction is identically zero —
    flat-bottom z-star regression bit-exact.

    Sign and orientation: the standard ``gradient_edge`` returns
    ``(p[c2] - p[c1]) / dcEdge`` (gradient pointing from c1 to c2,
    along the canonical edge normal).  This helper adds:

        delta = -g * (rho[c2] * excess_c2 - rho[c1] * excess_c1)
                / (dcEdge * rho_0)

    Each cell's pressure is shifted using its **own local density**
    (Adcroft & Campin 2004 §3.2): the deeper cell's pressure is
    extrapolated from its centroid to the shallower face-reference
    depth using the deeper cell's local rho_prime, not an averaged
    edge density.

    CVT mesh assumption (per the dycore audit): on a centroidal
    Voronoi tessellation the cell-centroid line crosses each Voronoi
    edge orthogonally (Ringler et al. 2010), so ``dcEdge`` is the
    correct baseline length.  On a non-CVT mesh (or after a
    metric-perturbing smoothing) the correction is off by
    ``cos(angle)`` between the centroid line and the edge normal.

    Parameters
    ----------
    centroid_depth : (nCells, nlev)
        Geometric centroid depth of each layer below the surface
        (positive downward), accounting for partial cells.  Use
        :func:`legoesm.ocean.vertical.compute_centroid_depth`.
        Convention: pass the **eta=0 reference** centroid (same
        reference as the ``rho_prime`` argument; using live eta
        breaks the "rest-state machine-zero" claim — see lat-lon
        usage in ``ocean_pe_latlon_cgrid.py``).
    rho_prime : (nCells, nlev)
        Density anomaly [kg/m^3].
    mesh : VoronoiMesh
    g : float
        Gravitational acceleration [m/s^2].
    rho_0 : float
        Reference density [kg/m^3] (matches ``MPASOceanConfig.rho_0``).

    Returns
    -------
    (nEdges, nlev)  additive correction in [m/s^2], ready to add
    directly to ``gradient_edge_3d(bernoulli, mesh)`` where
    ``bernoulli = ke + p_prime / rho_0``.
    """
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    centroid_c1 = centroid_depth[c1]
    centroid_c2 = centroid_depth[c2]
    rho_c1 = rho_prime[c1]
    rho_c2 = rho_prime[c2]

    face_ref = jnp.minimum(centroid_c1, centroid_c2)
    excess_c1 = centroid_c1 - face_ref  # >= 0
    excess_c2 = centroid_c2 - face_ref  # >= 0

    correction_pa_per_m = (
        -g * (rho_c2 * excess_c2 - rho_c1 * excess_c1)
        / mesh.dcEdge[:, jnp.newaxis]
    )
    return correction_pa_per_m / rho_0


def vertex_thickness_hybrid(
    h_cell_3d: jnp.ndarray,
    mesh: VoronoiMesh,
    *,
    alpha: float = 0.5,
    vtx_active_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Hybrid vertex thickness — kite-area mean in the interior,
    min-over-active near coasts.

    Switch criterion: at each (v, k), if ``min_wet_h < alpha * max_wet_h``
    use the min; else use the kite mean.  Combines the smoothness of
    the kite identity (Petersen 2015, MPAS-O production) with the
    coast robustness of MITgcm hFacZ.  The dycore audit recommends
    this hybrid because pure kite-mean lets a thin partial vertex
    transmit O(1) PV flux from a deep neighbor and drive spurious
    coastal currents.

    ``alpha`` defaults to 0.5 (a 2x ratio between thinnest and
    thickest wet cell triggers the min fallback).
    """
    cov = mesh.cellsOnVertex
    cov_valid = cov >= 0
    cov_safe = jnp.maximum(cov, 0)
    h_gathered = h_cell_3d[cov_safe]
    valid = cov_valid[:, :, None] & (h_gathered > 0.0)

    h_for_max = jnp.where(valid, h_gathered, 0.0)
    h_max = jnp.max(h_for_max, axis=0)
    h_for_min = jnp.where(valid, h_gathered, BIG_H)
    h_min = jnp.min(h_for_min, axis=0)
    any_wet = jnp.any(valid, axis=0)
    h_min = jnp.where(any_wet, h_min, 0.0)

    use_min = h_min < alpha * h_max  # (nVertices, nlev)
    h_kite = kite_area_vertex_thickness(h_cell_3d, mesh)
    h_v = jnp.where(use_min, h_min, h_kite)
    if vtx_active_mask is not None:
        h_v = h_v * vtx_active_mask[:, None]
    return h_v
