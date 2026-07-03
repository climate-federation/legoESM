"""Topography-aware metric helpers for MPAS partial bottom cells.

P2 of the MPAS realistic-geometry plan (see
``docs/ocean/experiments/realistic_geometry_mpas_plan.md``). These
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


def density_jacobian_pgf_smc03_mpas(
    rho_per_cell: jnp.ndarray,
    h_partial: jnp.ndarray,
    is_active: jnp.ndarray,
    mesh: VoronoiMesh,
    g: float,
) -> jnp.ndarray:
    """Shchepetkin & McWilliams 2003 density-Jacobian PGF on Voronoi.

    MPAS edge-loop wrapper around the grid-neutral SMC03 building
    blocks in :mod:`legoesm.ocean.dynamics.pgf_smc03`.  Mirrors the
    lat-lon ``density_jacobian_pgf_smc03_x/y`` operators exactly; the
    only difference is that the W/E (or S/N) neighbour gather is
    replaced with a ``cellsOnEdge`` index gather, and the metric
    divisor is ``dcEdge`` instead of ``R · dlon · cos_lat``.

    Algorithm (per edge between cells c1 = ``cellsOnEdge[0]`` and
    c2 = ``cellsOnEdge[1]``; per level k):

    1. Per-column ``z_centroid = cumsum(h_partial) − 0.5·h_partial``
       (η=0 reference, matching the rho_prime / p_prime reference).
    2. Per-column ``σ`` from ``reconstruct_harmonic_slopes``.
    3. Face-reference depth ``z_target_face = min(z_c[c1], z_c[c2])``
       — the shallower of the two centroids (Adcroft & Campin 2004
       convention; the lat-lon SMC03 docstring at
       ``latlon_cgrid_operators.density_jacobian_pgf_smc03_x`` has the
       full rationale, including why ``min`` is essential rather than
       midpoint — clamp asymmetry on thin partial cells leaves a
       ``ρ·g·(dz_ref − 3·h_partial)/4`` per-face residual otherwise).
    4. Per-column pressure at the SAME face depth via
       ``compute_pressure_at_target_smc03``.
    5. Edge-normal pressure gradient = ``(P_c2 − P_c1) / dcEdge``.

    Sign convention matches ``gradient_edge_3d``:
    ``(phi[c2] - phi[c1]) / dcEdge`` (gradient pointing from c1 to c2
    along the canonical edge normal).  Returns the **raw** pressure
    gradient in Pa/m; the dispatcher in ``ocean_pe_mpas`` divides by
    ``rho_0`` before adding to the Bernoulli gradient (matching the
    lat-lon SMC03 contract; ``partial_cell_pgf_correction_edge``
    follows the alternative convention of pre-dividing).

    For full-cell columns (``h_partial == dz_ref`` everywhere) and
    horizontally uniform ρ, the result is bit-exact zero — the same
    rest-state property the centered+AC stack provides on z-star.
    For *linear* ρ(z) over arbitrary partial-cell topography, the
    result is also exactly zero (the property that motivates SMC03
    over AC).

    Parameters
    ----------
    rho_per_cell : (nCells, nlev)
        Cell-mean density anomaly ρ' [kg/m³] (from
        ``iterate_eos_and_pressure_anomaly``).
    h_partial : (nCells, nlev)
        Per-cell layer thickness [m] (from
        ``OceanPartialCellCoordinate.h_partial``).
    is_active : (nCells, nlev)
        Wet-cell mask (1.0 / 0.0); from
        ``OceanPartialCellCoordinate.is_active``.
    mesh : VoronoiMesh
    g : float
        Gravitational acceleration [m/s²].

    Returns
    -------
    (nEdges, nlev)
        Edge-normal pressure-gradient force in Pa/m.  Divide by ρ_0 to
        get acceleration [m/s²] before adding to the Bernoulli edge
        gradient.

    References
    ----------
    Shchepetkin & McWilliams (2003), JGR Oceans 108(C9), §4.
    See ``docs/ocean/experiments/density_jacobian_pgf_mpas.md`` for
    the port design and validation phases.
    """
    # Local import — avoid pulling SMC03 into this module's import
    # chain when MPAS is loaded without a partial-cell coordinate.
    from legoesm.ocean.dynamics.pgf_smc03 import (
        compute_pressure_at_target_smc03,
        reconstruct_harmonic_slopes,
    )

    # Per-column geometry and slopes (grid-agnostic).
    z_centroid = jnp.cumsum(h_partial, axis=-1) - 0.5 * h_partial
    sigma = reconstruct_harmonic_slopes(rho_per_cell, z_centroid, is_active)

    # Edge-pair gather: c1 = cellsOnEdge[0], c2 = cellsOnEdge[1].
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    rho_1 = rho_per_cell[c1]
    rho_2 = rho_per_cell[c2]
    h_1 = h_partial[c1]
    h_2 = h_partial[c2]
    z_c_1 = z_centroid[c1]
    z_c_2 = z_centroid[c2]
    sigma_1 = sigma[c1]
    sigma_2 = sigma[c2]

    # Face-reference depth: shallower of the two centroids.  See
    # ``density_jacobian_pgf_smc03_x`` docstring for the C1-bug
    # rationale.  By construction this z lies inside both columns'
    # active range (the shallower-centroid cell encloses its own
    # centroid; the deeper column's cell at the same level extends at
    # least to that depth).  Reduces to the standard centroid on
    # full-cell faces.
    z_target_face = jnp.minimum(z_c_1, z_c_2)        # (nEdges, nlev)

    # Per-column pressure at the SAME face depth.
    P_1 = compute_pressure_at_target_smc03(
        rho_1, h_1, z_c_1, sigma_1, z_target_face, g,
    )                                                # (nEdges, nlev)
    P_2 = compute_pressure_at_target_smc03(
        rho_2, h_2, z_c_2, sigma_2, z_target_face, g,
    )                                                # (nEdges, nlev)

    # Edge-normal pressure gradient (Pa/m).  Matches
    # ``gradient_edge_3d`` sign: (phi[c2] - phi[c1]) / dcEdge.
    return (P_2 - P_1) / mesh.dcEdge[:, jnp.newaxis]


def density_jacobian_pgf_ahh08_mpas(
    T_3d: jnp.ndarray,
    S_3d: jnp.ndarray,
    h_partial: jnp.ndarray,
    mesh: VoronoiMesh,
    g: float,
    p_atmos: float = 0.0,
) -> jnp.ndarray:
    """Adcroft-Hallberg-Hill 2008 analytic finite-volume PGF on Voronoi.

    Builds on the grid-neutral Wright-EOS analytic primitives in
    :mod:`legoesm.ocean.dynamics.pgf_ahh08`.  Unlike SMC03 (which works
    with ρ' and a harmonic-slope reconstruction), AHH08 uses the
    Wright EOS directly to integrate ``∫p(z) dz`` analytically per
    cell, then differences face-averaged pressures over the common
    wet face on the two sides of an edge.

    Algorithm (per edge between c1 = ``cellsOnEdge[0]`` and
    c2 = ``cellsOnEdge[1]``; per level k):

    1. Per-column AHH08 walk: solve the implicit hydrostatic relation
       ``λ·ln(u_b/u_t) + α₀·(u_b−u_t) = g·dz`` cell by cell, where
       ``u = p + p₀(T,S)``.  Two Newton iterations from a linear-pressure
       starting guess get us to ~1e-12 relative.
    2. Per edge per level, define ``h_face = min(h_c1[k], h_c2[k])``
       (the common wet face vertical extent).  Levels below the step
       on the shallower side have ``h_face = 0`` and contribute zero;
       the dispatcher's ``edge_mask_3d`` masks the corresponding
       tendency contribution out.
    3. Face-averaged pressure on each side: ``P̄_n = ∫_{z_top}^{z_top+
       h_face} p_n(z) dz / h_face`` using the same analytic integral.
    4. Edge-normal pressure gradient = ``(P̄_2 − P̄_1) / dcEdge``.

    Why this gives machine-zero on a rest state with partial-cell steps
    --------------------------------------------------------------------
    On a state with horizontally uniform T(z), S(z):

    * The column walk produces identical ``u_top_k`` per cell on every
      column (same T, S, h_full at every level above the step).
    * At a step level k where one side has h_partial < h_full,
      ``h_face = h_partial`` and the Newton solve on each side yields
      identical ``u_face`` because (T_k, S_k, u_top, h_face) are
      identical.
    * Both ``∫p dz`` integrals over the same z-range with the same
      integrand are equal → ``P̄_1 = P̄_2`` exactly.

    The face-averaged differencing avoids the cell-centroid pressure
    mismatch that drives Adcroft & Campin's O(h_step) extrapolation
    error.  In contrast to SMC03's slope-reconstruction approach, the
    Wright EOS provides the FULL ρ(z) curvature analytically, so the
    rest-state cancellation is exact (not just exact-on-linear-ρ).

    Parameters
    ----------
    T_3d, S_3d : (nCells, nlev)
        Cell-mean potential temperature [°C] and salinity [PSU].
        Note that AHH08 uses T,S directly rather than the ρ' array
        used by SMC03 — the analytic integral evaluates the Wright EOS
        internally.
    h_partial : (nCells, nlev)
        Per-cell layer thickness [m] (from
        ``OceanPartialCellCoordinate.h_partial``).
    mesh : VoronoiMesh
    g : float
        Gravitational acceleration [m/s²].
    p_atmos : float, default 0
        Atmospheric pressure at the surface [Pa].  The barotropic
        mode handles ``g·η`` separately, so 0 is the right reference.

    Returns
    -------
    (nEdges, nlev)
        Edge-normal pressure-gradient force in Pa/m.  Matches the
        SMC03 wrapper's contract: divide by ρ_0 to get acceleration
        [m/s²] before adding to the Bernoulli edge gradient.

    References
    ----------
    Adcroft, A., R. Hallberg, and M. Hill (2008): A finite volume
    discretization of the pressure gradient force using analytic
    integration.  Ocean Modelling, 22(3-4), 106-113.

    Implementation notes
    --------------------
    The face Newton solve is done at *every* edge / level even when
    ``h_face = 0`` (dry below step).  The Newton update with ``g·dz =
    0`` converges to ``u_face = u_top`` on the first iteration, so
    these dry-edge calls are cheap and safe.  Their contribution is
    masked out by the dispatcher's ``edge_mask_3d``.
    """
    from legoesm.ocean.dynamics.pgf_ahh08 import (
        column_pressure_integrals_ahh08,
        integral_p_dz_cell,
        solve_u_bottom,
    )

    # Per-column walk → u_top_cell per cell.
    u_top_cell, _u_bot_cell, _F_cell = column_pressure_integrals_ahh08(
        T_3d, S_3d, h_partial, g, p_atmos=p_atmos,
    )                                                # (nCells, nlev)

    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]

    T_1 = T_3d[c1]
    T_2 = T_3d[c2]
    S_1 = S_3d[c1]
    S_2 = S_3d[c2]
    h_1 = h_partial[c1]
    h_2 = h_partial[c2]
    u_top_1 = u_top_cell[c1]
    u_top_2 = u_top_cell[c2]

    h_face = jnp.minimum(h_1, h_2)                   # (nEdges, nlev)

    # Newton-solve u at z_top + h_face on each side.
    u_face_1 = solve_u_bottom(T_1, S_1, u_top_1, h_face, g)
    u_face_2 = solve_u_bottom(T_2, S_2, u_top_2, h_face, g)

    # Face integrals over the COMMON wet face on each side.
    F_face_1 = integral_p_dz_cell(T_1, S_1, u_top_1, u_face_1, g)
    F_face_2 = integral_p_dz_cell(T_2, S_2, u_top_2, u_face_2, g)

    # Cell-averaged pressure over the face.  ``h_safe`` avoids 0/0 at
    # dry edges (h_face = 0) — those edges are masked by the
    # dispatcher's ``edge_mask_3d`` so the per-edge value here is
    # discarded.  Use a small floor so the division is finite under
    # both float32 and float64 precision.
    h_safe = jnp.maximum(h_face, jnp.float32(1e-12))
    P_bar_1 = F_face_1 / h_safe
    P_bar_2 = F_face_2 / h_safe

    return (P_bar_2 - P_bar_1) / mesh.dcEdge[:, jnp.newaxis]


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
