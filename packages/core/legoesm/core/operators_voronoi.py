"""TRiSK discrete operators on Voronoi (MPAS-style) meshes.

All operators from Ringler et al. (2010) for arbitrarily-structured
C-grids. Uses JAX scatter-gather operations (no Python loops) for
full JIT and autodiff compatibility.

References
----------
- Ringler, T. D., Thuburn, J., Klemp, J. B., & Skamarock, W. C. (2010).
  A unified approach to energy conservation and potential vorticity dynamics
  for arbitrarily-structured C-grids. J. Comput. Phys., 229(9), 3065-3090.
"""

from __future__ import annotations

import jax.numpy as jnp


# ============================================================================
# Primary operators
# ============================================================================

def divergence_cell(u_edge, mesh):
    """Divergence at cell centers (Ringler 2010, Eq. 21).

    div(c) = (1/A_c) * Σ_e n_{e,c} * u_e * l_e

    where n_{e,c} = edgeSignOnCell, l_e = dvEdge.

    Parameters
    ----------
    u_edge : jax.Array, shape (nEdges,)
        Normal velocity at edges.
    mesh : VoronoiMesh

    Returns
    -------
    jax.Array, shape (nCells,)
    """

    # edgesOnCell: (maxEdges, nCells)
    eoc = mesh.edgesOnCell  # (maxEdges, nCells)
    sign = mesh.edgeSignOnCell  # (maxEdges, nCells)

    # Mask invalid entries
    mask = (eoc >= 0).astype(u_edge.dtype)  # (maxEdges, nCells)

    # Gather edge values and dvEdge
    eoc_safe = jnp.maximum(eoc, 0)  # clamp for safe indexing
    u_gathered = u_edge[eoc_safe]  # (maxEdges, nCells)
    dv_gathered = mesh.dvEdge[eoc_safe]  # (maxEdges, nCells)

    # Sum contributions
    flux = sign * u_gathered * dv_gathered * mask  # (maxEdges, nCells)
    div = jnp.sum(flux, axis=0) / mesh.areaCell  # (nCells,)
    return div


def gradient_edge(phi_cell, mesh):
    """Gradient at edges (Ringler 2010, Eq. 22).

    grad(e) = (phi[cell2] - phi[cell1]) / dcEdge

    Parameters
    ----------
    phi_cell : jax.Array, shape (nCells,)
        Scalar field at cell centers.
    mesh : VoronoiMesh

    Returns
    -------
    jax.Array, shape (nEdges,)
    """
    c1 = mesh.cellsOnEdge[0]  # (nEdges,)
    c2 = mesh.cellsOnEdge[1]  # (nEdges,)
    return (phi_cell[c2] - phi_cell[c1]) / mesh.dcEdge


def curl_vertex(u_edge, mesh):
    """Curl (relative vorticity) at vertices (Ringler 2010, Eq. 23).

    curl(v) = (1/A_v) * Σ_e t_{e,v} * u_e * d_e

    where t_{e,v} = edgeSignOnVertex, d_e = dcEdge.

    Parameters
    ----------
    u_edge : jax.Array, shape (nEdges,)
        Normal velocity at edges.
    mesh : VoronoiMesh

    Returns
    -------
    jax.Array, shape (nVertices,)
    """

    eov = mesh.edgesOnVertex  # (vertexDegree, nVertices)
    sign = mesh.edgeSignOnVertex  # (vertexDegree, nVertices)

    mask = (eov >= 0).astype(u_edge.dtype)
    eov_safe = jnp.maximum(eov, 0)

    u_gathered = u_edge[eov_safe]  # (vertexDegree, nVertices)
    dc_gathered = mesh.dcEdge[eov_safe]  # (vertexDegree, nVertices)

    circ = sign * u_gathered * dc_gathered * mask
    curl = jnp.sum(circ, axis=0) / mesh.areaTriangle
    return curl


def tangential_velocity(u_edge, mesh):
    """Tangential velocity reconstruction (Ringler 2010, Eq. 24).

    v_t(e) = Σ_{e'} w(e,e') * u(e')

    Uses Thuburn et al. (2009) weights for energy conservation.

    Parameters
    ----------
    u_edge : jax.Array, shape (nEdges,)
    mesh : VoronoiMesh

    Returns
    -------
    jax.Array, shape (nEdges,)
    """
    eoe = mesh.edgesOnEdge  # (maxEdges2, nEdges)
    woe = mesh.weightsOnEdge  # (maxEdges2, nEdges)

    mask = (eoe >= 0).astype(u_edge.dtype)
    eoe_safe = jnp.maximum(eoe, 0)

    u_gathered = u_edge[eoe_safe]  # (maxEdges2, nEdges)
    vt = jnp.sum(woe * u_gathered * mask, axis=0)
    return vt


# ============================================================================
# Cell-to-edge interpolation
# ============================================================================

def cell_to_edge_avg(phi_cell, mesh):
    """Interpolate a cell-centered scalar to edges by simple averaging.

    phi_edge = (phi[cell1] + phi[cell2]) / 2

    Parameters
    ----------
    phi_cell : jax.Array, shape (nCells, ...) or (nCells,)
        Scalar field at cell centers.
    mesh : VoronoiMesh

    Returns
    -------
    jax.Array, shape (nEdges, ...) or (nEdges,)
    """
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    return 0.5 * (phi_cell[c1] + phi_cell[c2])


def edge_thickness(h_cell, mesh):
    """Thickness at edges by simple averaging (Ringler 2010, Eq. 52).

    h_edge = (h[cell1] + h[cell2]) / 2

    Parameters
    ----------
    h_cell : jax.Array, shape (nCells,)
    mesh : VoronoiMesh

    Returns
    -------
    jax.Array, shape (nEdges,)
    """
    return cell_to_edge_avg(h_cell, mesh)


def vertex_thickness(h_cell, mesh):
    """Thickness at vertices via kite-area-weighted average.

    h_vertex(v) = Σ_k kiteArea(k,v) * h[cellsOnVertex(k,v)] / areaTriangle(v)

    Parameters
    ----------
    h_cell : jax.Array, shape (nCells,)
    mesh : VoronoiMesh

    Returns
    -------
    jax.Array, shape (nVertices,)
    """
    cov = mesh.cellsOnVertex  # (vertexDegree, nVertices)
    ka = mesh.kiteAreasOnVertex  # (vertexDegree, nVertices)

    mask = (cov >= 0).astype(h_cell.dtype)
    cov_safe = jnp.maximum(cov, 0)

    h_gathered = h_cell[cov_safe]  # (vertexDegree, nVertices)
    h_v = jnp.sum(ka * h_gathered * mask, axis=0) / mesh.areaTriangle
    return h_v


# ============================================================================
# Kinetic energy and potential vorticity
# ============================================================================

def kinetic_energy_cell(u_edge, mesh):
    """Kinetic energy at cell centers (Ringler 2010, Eq. 63).

    KE(c) = (1/(2*A_c)) * Σ_e (A_e/2) * u_e²

    where A_e = dcEdge * dvEdge is the edge "area".

    Parameters
    ----------
    u_edge : jax.Array, shape (nEdges,)
    mesh : VoronoiMesh

    Returns
    -------
    jax.Array, shape (nCells,)
    """
    eoc = mesh.edgesOnCell  # (maxEdges, nCells)
    mask = (eoc >= 0).astype(u_edge.dtype)
    eoc_safe = jnp.maximum(eoc, 0)

    u_sq = u_edge[eoc_safe] ** 2  # (maxEdges, nCells)
    dc = mesh.dcEdge[eoc_safe]
    dv = mesh.dvEdge[eoc_safe]
    area_e = dc * dv  # edge "area"

    ke = jnp.sum(0.25 * area_e * u_sq * mask, axis=0) / mesh.areaCell
    return ke


def potential_vorticity_vertex(u_edge, h_cell, f_vertex, mesh):
    """Potential vorticity at vertices (Ringler 2010, Eq. 34).

    q(v) = (curl(u)(v) + f(v)) / h_vertex(v)

    Parameters
    ----------
    u_edge : jax.Array, shape (nEdges,)
    h_cell : jax.Array, shape (nCells,)
    f_vertex : jax.Array, shape (nVertices,)
    mesh : VoronoiMesh

    Returns
    -------
    jax.Array, shape (nVertices,)
    """
    zeta = curl_vertex(u_edge, mesh)
    h_v = vertex_thickness(h_cell, mesh)
    # Guard against division by zero
    h_v_safe = jnp.maximum(h_v, 1e-10)
    return (zeta + f_vertex) / h_v_safe


def pv_edge(q_vertex, mesh):
    """PV at edges: simple average of the two adjacent vertices.

    Parameters
    ----------
    q_vertex : jax.Array, shape (nVertices,)
    mesh : VoronoiMesh

    Returns
    -------
    jax.Array, shape (nEdges,)
    """
    v0 = mesh.verticesOnEdge[0]  # (nEdges,)
    v1 = mesh.verticesOnEdge[1]
    return 0.5 * (q_vertex[v0] + q_vertex[v1])


# ============================================================================
# PV flux terms
# ============================================================================

def pv_flux_energy_conserving(u_edge, h_cell, q_vertex, mesh):
    """Energy-conserving PV flux (Ringler 2010, Eq. 49).

    F_q(e) = Σ_{e'} w(e,e') * q_avg(e') * h_edge(e') * u(e')

    Parameters
    ----------
    u_edge : jax.Array, shape (nEdges,)
    h_cell : jax.Array, shape (nCells,)
    q_vertex : jax.Array, shape (nVertices,)
    mesh : VoronoiMesh

    Returns
    -------
    jax.Array, shape (nEdges,)
    """
    h_e = edge_thickness(h_cell, mesh)
    q_e = pv_edge(q_vertex, mesh)

    eoe = mesh.edgesOnEdge  # (maxEdges2, nEdges)
    woe = mesh.weightsOnEdge  # (maxEdges2, nEdges)
    mask = (eoe >= 0).astype(u_edge.dtype)
    eoe_safe = jnp.maximum(eoe, 0)

    # Gather quantities at stencil edges
    u_g = u_edge[eoe_safe]
    h_g = h_e[eoe_safe]
    q_g = q_e[eoe_safe]

    flux = jnp.sum(woe * q_g * h_g * u_g * mask, axis=0)
    return flux


def pv_flux_enstrophy_conserving(u_edge, h_cell, q_vertex, mesh):
    """Enstrophy-conserving PV flux (Ringler 2010, Eq. 71-72).

    F_q(e) = q_e * Σ_{e'} w(e,e') * h_e(e') * u(e')

    The tangential *thickness flux* is reconstructed via TRiSK weights,
    i.e., Σ w(e,e') * F(e') where F = h_edge * u_normal.  This is
    distinct from h_e(e) * v_t(e) because the weights must act on
    the full flux, not just the velocity.

    Parameters
    ----------
    u_edge : jax.Array, shape (nEdges,)
    h_cell : jax.Array, shape (nCells,)
    q_vertex : jax.Array, shape (nVertices,)
    mesh : VoronoiMesh

    Returns
    -------
    jax.Array, shape (nEdges,)
    """
    h_e = edge_thickness(h_cell, mesh)
    q_e = pv_edge(q_vertex, mesh)

    # Tangential thickness flux: Σ w(e,e') * h_e(e') * u(e')
    F_normal = h_e * u_edge  # thickness flux at each edge
    F_tangential = tangential_velocity(F_normal, mesh)  # reuse TRiSK reconstruction

    return q_e * F_tangential


# ============================================================================
# Diffusion operators
# ============================================================================

def vector_laplacian_del2(u_edge, mesh):
    """Vector Laplacian del2: grad(div) - k×grad(curl).

    Parameters
    ----------
    u_edge : jax.Array, shape (nEdges,)
    mesh : VoronoiMesh

    Returns
    -------
    jax.Array, shape (nEdges,)
    """
    div_c = divergence_cell(u_edge, mesh)   # (nCells,)
    curl_v = curl_vertex(u_edge, mesh)      # (nVertices,)

    # grad(div) at edges
    grad_div = gradient_edge(div_c, mesh)   # (nEdges,)

    # k×grad(curl) at edges: this is the tangential gradient of curl
    # For C-grid: tangent of curl gradient = (curl[v1] - curl[v0]) / dvEdge
    v0 = mesh.verticesOnEdge[0]
    v1 = mesh.verticesOnEdge[1]
    grad_curl_tangent = (curl_v[v1] - curl_v[v0]) / mesh.dvEdge

    return grad_div - grad_curl_tangent


def vector_laplacian_del4(u_edge, mesh):
    """Biharmonic vector Laplacian: -del2(del2(u)).

    Parameters
    ----------
    u_edge : jax.Array, shape (nEdges,)
    mesh : VoronoiMesh

    Returns
    -------
    jax.Array, shape (nEdges,)
    """
    del2_u = vector_laplacian_del2(u_edge, mesh)
    return -vector_laplacian_del2(del2_u, mesh)


# ============================================================================
# Thickness flux (higher-order)
# ============================================================================

def thickness_flux(h_cell, u_edge, mesh, order=2):
    """Thickness flux at edges for mass equation.

    Parameters
    ----------
    h_cell : jax.Array, shape (nCells,)
    u_edge : jax.Array, shape (nEdges,)
    mesh : VoronoiMesh
    order : int
        Order of reconstruction: 2 (centered) or 3 (upwind-biased).

    Returns
    -------
    jax.Array, shape (nEdges,)
        h_edge * u_edge
    """
    if order == 2:
        h_e = edge_thickness(h_cell, mesh)
    elif order >= 3:
        # 3rd order: upwind-biased
        h_e_centered = edge_thickness(h_cell, mesh)
        c1 = mesh.cellsOnEdge[0]
        c2 = mesh.cellsOnEdge[1]
        # Upwind correction
        h_upwind = jnp.where(u_edge > 0, h_cell[c1], h_cell[c2])
        # Blend: 3/4 centered + 1/4 upwind for 3rd order
        h_e = 0.75 * h_e_centered + 0.25 * h_upwind
    else:
        h_e = edge_thickness(h_cell, mesh)

    return h_e * u_edge


# ============================================================================
# APVM (Anticipated Potential Vorticity Method)
# ============================================================================

def apvm_correction(q_vertex, u_edge, mesh, dt):
    """Anticipated PV Method upwinding correction.

    Modifies PV at vertices by an upstream correction:
    q_apvm(v) = q(v) - dt/2 * (u·∇q)(v)

    Parameters
    ----------
    q_vertex : jax.Array, shape (nVertices,)
    u_edge : jax.Array, shape (nEdges,)
    mesh : VoronoiMesh
    dt : float

    Returns
    -------
    jax.Array, shape (nVertices,)
        Corrected PV at vertices.
    """
    # Approximate gradient of q at vertices using adjacent vertex values
    eov = mesh.edgesOnVertex  # (vertexDegree, nVertices)
    voe = mesh.verticesOnEdge  # (2, nEdges)

    mask = (eov >= 0).astype(q_vertex.dtype)
    eov_safe = jnp.maximum(eov, 0)

    # For each vertex, compute advective derivative from connected edges
    # u_edge * dq/ds along each edge (where s is along the edge tangent)
    u_edge[eov_safe]  # (vertexDegree, nVertices)

    # q difference along each edge
    v0_of_edge = voe[0][eov_safe]  # vertex 0 of each adjacent edge
    v1_of_edge = voe[1][eov_safe]  # vertex 1 of each adjacent edge
    dq = q_vertex[v1_of_edge] - q_vertex[v0_of_edge]
    dv = mesh.dvEdge[eov_safe]
    dv_safe = jnp.maximum(dv, 1e-10)
    dq_ds = dq / dv_safe

    # Tangential velocity contribution to q advection
    vt = tangential_velocity(u_edge, mesh)
    vt_at_edges = vt[eov_safe]

    # Advective derivative at vertex: average of edge contributions —
    # both reductions sum along the ``maxEdges`` axis with weight ``mask``.
    _pair = jnp.sum(
        jnp.stack([vt_at_edges * dq_ds, jnp.ones_like(mask)], axis=-1) * mask[..., None],
        axis=0,
    )
    advection = _pair[..., 0]
    count = jnp.maximum(_pair[..., 1], 1.0)
    u_dot_grad_q = advection / count

    return q_vertex - 0.5 * dt * u_dot_grad_q


# ============================================================================
# Batched 3D operators — one gather for all levels
# ============================================================================
# These replace jax.lax.scan over levels: adjacency is gathered once
# across the full (nEdges/nCells/nVertices, nlev) arrays, eliminating
# nlev repeated indirect indexing calls.


def divergence_cell_3d(u_edge_3d, mesh):
    """Divergence at cell centers for all levels.

    Parameters
    ----------
    u_edge_3d : jax.Array, shape (nEdges, nlev)
    mesh : VoronoiMesh

    Returns
    -------
    jax.Array, shape (nCells, nlev)
    """
    eoc = mesh.edgesOnCell  # (maxEdges, nCells)
    sign = mesh.edgeSignOnCell  # (maxEdges, nCells)
    mask = (eoc >= 0).astype(u_edge_3d.dtype)
    eoc_safe = jnp.maximum(eoc, 0)

    u_gathered = u_edge_3d[eoc_safe]          # (maxEdges, nCells, nlev)
    dv_gathered = mesh.dvEdge[eoc_safe]        # (maxEdges, nCells)

    flux = (sign[:, :, None] * u_gathered
            * dv_gathered[:, :, None] * mask[:, :, None])
    return jnp.sum(flux, axis=0) / mesh.areaCell[:, None]


def gradient_edge_3d(phi_cell_3d, mesh):
    """Gradient at edges for all levels.

    Parameters
    ----------
    phi_cell_3d : jax.Array, shape (nCells, nlev)
    mesh : VoronoiMesh

    Returns
    -------
    jax.Array, shape (nEdges, nlev)
    """
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    return (phi_cell_3d[c2] - phi_cell_3d[c1]) / mesh.dcEdge[:, None]


def curl_vertex_3d(u_edge_3d, mesh):
    """Curl (relative vorticity) at vertices for all levels.

    Parameters
    ----------
    u_edge_3d : jax.Array, shape (nEdges, nlev)
    mesh : VoronoiMesh

    Returns
    -------
    jax.Array, shape (nVertices, nlev)
    """
    eov = mesh.edgesOnVertex          # (vertexDegree, nVertices)
    sign = mesh.edgeSignOnVertex
    mask = (eov >= 0).astype(u_edge_3d.dtype)
    eov_safe = jnp.maximum(eov, 0)

    u_gathered = u_edge_3d[eov_safe]   # (vertexDegree, nVertices, nlev)
    dc_gathered = mesh.dcEdge[eov_safe]

    circ = (sign[:, :, None] * u_gathered
            * dc_gathered[:, :, None] * mask[:, :, None])
    return jnp.sum(circ, axis=0) / mesh.areaTriangle[:, None]


def tangential_velocity_3d(u_edge_3d, mesh):
    """Tangential velocity reconstruction for all levels.

    Parameters
    ----------
    u_edge_3d : jax.Array, shape (nEdges, nlev)
    mesh : VoronoiMesh

    Returns
    -------
    jax.Array, shape (nEdges, nlev)
    """
    eoe = mesh.edgesOnEdge        # (maxEdges2, nEdges)
    woe = mesh.weightsOnEdge
    mask = (eoe >= 0).astype(u_edge_3d.dtype)
    eoe_safe = jnp.maximum(eoe, 0)

    u_gathered = u_edge_3d[eoe_safe]  # (maxEdges2, nEdges, nlev)
    return jnp.sum(woe[:, :, None] * u_gathered * mask[:, :, None], axis=0)


def cell_to_edge_avg_3d(phi_cell_3d, mesh):
    """Interpolate cell-centered 3D field to edges by averaging.

    Parameters
    ----------
    phi_cell_3d : jax.Array, shape (nCells, nlev) or (nCells, ...)
    mesh : VoronoiMesh

    Returns
    -------
    jax.Array, shape (nEdges, nlev) or (nEdges, ...)
    """
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    return 0.5 * (phi_cell_3d[c1] + phi_cell_3d[c2])


def edge_thickness_3d(h_cell_3d, mesh):
    """Thickness at edges for all levels."""
    return cell_to_edge_avg_3d(h_cell_3d, mesh)


def vertex_thickness_3d(h_cell_3d, mesh):
    """Kite-area-weighted thickness at vertices for all levels.

    Parameters
    ----------
    h_cell_3d : jax.Array, shape (nCells, nlev)
    mesh : VoronoiMesh

    Returns
    -------
    jax.Array, shape (nVertices, nlev)
    """
    cov = mesh.cellsOnVertex       # (vertexDegree, nVertices)
    ka = mesh.kiteAreasOnVertex
    mask = (cov >= 0).astype(h_cell_3d.dtype)
    cov_safe = jnp.maximum(cov, 0)

    h_gathered = h_cell_3d[cov_safe]  # (vertexDegree, nVertices, nlev)
    return (jnp.sum(ka[:, :, None] * h_gathered * mask[:, :, None], axis=0)
            / mesh.areaTriangle[:, None])


def kinetic_energy_cell_3d(u_edge_3d, mesh):
    """Kinetic energy at cell centers for all levels.

    Parameters
    ----------
    u_edge_3d : jax.Array, shape (nEdges, nlev)
    mesh : VoronoiMesh

    Returns
    -------
    jax.Array, shape (nCells, nlev)
    """
    eoc = mesh.edgesOnCell
    mask = (eoc >= 0).astype(u_edge_3d.dtype)
    eoc_safe = jnp.maximum(eoc, 0)

    u_sq = u_edge_3d[eoc_safe] ** 2   # (maxEdges, nCells, nlev)
    dc = mesh.dcEdge[eoc_safe]
    dv = mesh.dvEdge[eoc_safe]
    area_e = dc * dv

    return (jnp.sum(0.25 * area_e[:, :, None] * u_sq * mask[:, :, None], axis=0)
            / mesh.areaCell[:, None])


def potential_vorticity_vertex_3d(u_edge_3d, h_cell_3d, f_vertex, mesh):
    """Potential vorticity at vertices for all levels.

    Parameters
    ----------
    u_edge_3d : jax.Array, shape (nEdges, nlev)
    h_cell_3d : jax.Array, shape (nCells, nlev)
    f_vertex : jax.Array, shape (nVertices,)
    mesh : VoronoiMesh

    Returns
    -------
    jax.Array, shape (nVertices, nlev)
    """
    zeta = curl_vertex_3d(u_edge_3d, mesh)
    h_v = vertex_thickness_3d(h_cell_3d, mesh)
    h_v_safe = jnp.maximum(h_v, 1e-10)
    return (zeta + f_vertex[:, None]) / h_v_safe


def pv_edge_3d(q_vertex_3d, mesh):
    """PV at edges for all levels."""
    v0 = mesh.verticesOnEdge[0]
    v1 = mesh.verticesOnEdge[1]
    return 0.5 * (q_vertex_3d[v0] + q_vertex_3d[v1])


def pv_flux_energy_conserving_3d(
    u_edge_3d, h_cell_3d, q_vertex_3d, mesh,
    h_edge_3d=None,
):
    """Energy-conserving PV flux for all levels.

    Parameters
    ----------
    u_edge_3d : jax.Array, shape (nEdges, nlev)
    h_cell_3d : jax.Array, shape (nCells, nlev)
    q_vertex_3d : jax.Array, shape (nVertices, nlev)
    mesh : VoronoiMesh
    h_edge_3d : jax.Array, shape (nEdges, nlev), optional
        Pre-computed edge thickness ``cell_to_edge_avg_3d(h_cell_3d)``.
        When provided the internal ``edge_thickness_3d`` call is
        skipped — share the gather across pv_flux and any other op
        that already needed ``h`` at edges.

    Returns
    -------
    jax.Array, shape (nEdges, nlev)
    """
    h_e = h_edge_3d if h_edge_3d is not None else edge_thickness_3d(h_cell_3d, mesh)
    q_e = pv_edge_3d(q_vertex_3d, mesh)

    eoe = mesh.edgesOnEdge
    woe = mesh.weightsOnEdge
    mask = (eoe >= 0).astype(u_edge_3d.dtype)
    eoe_safe = jnp.maximum(eoe, 0)

    u_g = u_edge_3d[eoe_safe]   # (maxEdges2, nEdges, nlev)
    h_g = h_e[eoe_safe]
    q_g = q_e[eoe_safe]

    return jnp.sum(woe[:, :, None] * q_g * h_g * u_g * mask[:, :, None], axis=0)


def pv_flux_enstrophy_conserving_3d(
    u_edge_3d, h_cell_3d, q_vertex_3d, mesh,
    h_edge_3d=None,
):
    """Enstrophy-conserving PV flux for all levels.

    Parameters
    ----------
    u_edge_3d : jax.Array, shape (nEdges, nlev)
    h_cell_3d : jax.Array, shape (nCells, nlev)
    q_vertex_3d : jax.Array, shape (nVertices, nlev)
    mesh : VoronoiMesh
    h_edge_3d : jax.Array, shape (nEdges, nlev), optional
        Pre-computed edge thickness — see :func:`pv_flux_energy_conserving_3d`
        for usage.

    Returns
    -------
    jax.Array, shape (nEdges, nlev)
    """
    h_e = h_edge_3d if h_edge_3d is not None else edge_thickness_3d(h_cell_3d, mesh)
    q_e = pv_edge_3d(q_vertex_3d, mesh)

    F_normal = h_e * u_edge_3d
    F_tangential = tangential_velocity_3d(F_normal, mesh)
    return q_e * F_tangential


def vector_laplacian_del2_3d(u_edge_3d, mesh):
    """Vector Laplacian del2 for all levels.

    Parameters
    ----------
    u_edge_3d : jax.Array, shape (nEdges, nlev)
    mesh : VoronoiMesh

    Returns
    -------
    jax.Array, shape (nEdges, nlev)
    """
    div_c = divergence_cell_3d(u_edge_3d, mesh)
    curl_v = curl_vertex_3d(u_edge_3d, mesh)

    grad_div = gradient_edge_3d(div_c, mesh)

    v0 = mesh.verticesOnEdge[0]
    v1 = mesh.verticesOnEdge[1]
    grad_curl_tangent = (curl_v[v1] - curl_v[v0]) / mesh.dvEdge[:, None]

    return grad_div - grad_curl_tangent


def vector_laplacian_del4_3d(u_edge_3d, mesh, *, mid_refresh=None):
    """Biharmonic vector Laplacian for all levels.

    Parameters
    ----------
    u_edge_3d : jax.Array, shape (nEdges, nlev)
    mesh : VoronoiMesh
    mid_refresh : Callable(*edge_fields) -> tuple, optional
        Distributed-only halo refresh applied to the INTERMEDIATE
        Laplacian: the two-pass stencil consumes 4 hops, twice a
        ``halo_depth=2`` partition budget, so without a mid-operator
        refresh the outer pass reads a corrupted ``del2`` ring and owned
        edges at partition boundaries silently diverge from serial
        (MPAS stage-correctness audit).  ``None`` (serial) is the
        byte-identical default.

    Returns
    -------
    jax.Array, shape (nEdges, nlev)
    """
    del2_u = vector_laplacian_del2_3d(u_edge_3d, mesh)
    if mid_refresh is not None:
        (del2_u,) = mid_refresh(del2_u)
    return -vector_laplacian_del2_3d(del2_u, mesh)


def laplacian_cell_3d(f_cell_3d, mesh, *, mask=None):
    """Scalar Laplacian ``∇²f = div(grad(f))`` at cell centres, all levels.

    Uses the native MPAS TRiSK C-grid stencil:
    ``gradient_edge_3d`` then ``divergence_cell_3d``.  When a cell mask is
    supplied, the edge mask is derived as ``mask[c1] * mask[c2]`` so that
    gradients vanish at coastlines, and the final Laplacian is zeroed on
    land cells — mirroring the latlon ``laplacian_cgrid`` convention.

    Parameters
    ----------
    f_cell_3d : jax.Array, shape (nCells, nlev)
    mesh : VoronoiMesh
    mask : jax.Array, optional, shape (nCells,)
        Cell mask (1 on ocean, 0 on land). When supplied, the edge mask
        is derived internally to zero gradients across land-ocean
        boundaries, and the returned Laplacian is multiplied by ``mask``.

    Returns
    -------
    jax.Array, shape (nCells, nlev)
    """
    grad_f = gradient_edge_3d(f_cell_3d, mesh)  # (nEdges, nlev)
    if mask is not None:
        c1 = mesh.cellsOnEdge[0]
        c2 = mesh.cellsOnEdge[1]
        edge_mask = mask[c1] * mask[c2]         # (nEdges,)
        grad_f = grad_f * edge_mask[:, None]

    lap = divergence_cell_3d(grad_f, mesh)      # (nCells, nlev)

    if mask is not None:
        lap = lap * mask[:, None]
    return lap


def bilaplacian_cell_3d(f_cell_3d, mesh, *, mask=None):
    """Scalar bilaplacian ``∇⁴f = ∇²(∇²f)`` at cell centres, all levels.

    Applies :func:`laplacian_cell_3d` twice.  Used as the kernel of
    biharmonic tracer diffusion: ``dtr/dt -= K_bih * bilaplacian_cell_3d(tr)``.
    Same sign convention as the latlon ``bilaplacian_cgrid``: the operator
    returns ``∇²(∇²f)`` (positive-definite eigenvalues), so the physical
    dissipation sign is applied at the call site.

    Parameters
    ----------
    f_cell_3d : jax.Array, shape (nCells, nlev)
    mesh : VoronoiMesh
    mask : jax.Array, optional, shape (nCells,)
        Cell mask applied inside each Laplacian pass (boundary gradient
        zeroing + land-cell zeroing of the intermediate Laplacian).

    Returns
    -------
    jax.Array, shape (nCells, nlev)
    """
    lap_f = laplacian_cell_3d(f_cell_3d, mesh, mask=mask)
    return laplacian_cell_3d(lap_f, mesh, mask=mask)


def smagorinsky_biharmonic_3d(u_edge_3d, mesh, C_smag, *, mid_refresh=None):
    """Smagorinsky biharmonic viscosity: ``-del2(A_smag * del2(u))``.

    Flow-dependent biharmonic viscosity using the Smagorinsky (1963)
    formulation.  The strain rate is decomposed into divergence (tension)
    at cell centres and curl (shearing) at vertices, then averaged to
    edges where the velocity lives.

    The two-pass structure places the spatially varying coefficient
    *between* the two Laplacian applications, following the standard
    MPAS-Ocean / ICON-O approach::

        del2(u)  ->  multiply by A_smag  ->  del2 again  ->  negate

    This is NOT equivalent to ``A_smag * del4(u)`` when A_smag varies
    in space.

    Parameters
    ----------
    u_edge_3d : jax.Array, shape (nEdges, nlev)
        Normal velocity at edges.
    mesh : VoronoiMesh
        Mesh connectivity and geometry.
    C_smag : float
        Dimensionless Smagorinsky coefficient (typical 0.01-0.15).

    Returns
    -------
    jax.Array, shape (nEdges, nlev)
        Viscous tendency (to be ADDED to du/dt).
    """
    # --- Strain rate at native TRiSK locations ---
    div_c = divergence_cell_3d(u_edge_3d, mesh)   # (nCells, nlev) — tension
    curl_v = curl_vertex_3d(u_edge_3d, mesh)       # (nVertices, nlev) — shearing

    # --- Average to edges ---
    c1, c2 = mesh.cellsOnEdge[0], mesh.cellsOnEdge[1]
    D_T_edge = 0.5 * (div_c[c1] + div_c[c2])      # (nEdges, nlev)

    v0, v1 = mesh.verticesOnEdge[0], mesh.verticesOnEdge[1]
    D_S_edge = 0.5 * (curl_v[v0] + curl_v[v1])    # (nEdges, nlev)

    # Small epsilon prevents NaN gradient of sqrt at zero (boundary edges).
    deformation = jnp.sqrt(D_T_edge**2 + D_S_edge**2 + 1e-30)

    # --- Smagorinsky coefficient [m²/s] at edges ---
    # Geometric mean of primal/dual edge lengths as grid scale.
    delta_edge = jnp.sqrt(mesh.dcEdge * mesh.dvEdge)  # (nEdges,)
    A_smag = (C_smag * delta_edge[:, None]) ** 2 * deformation  # (nEdges, nlev)

    # --- Two-pass biharmonic: -del2(A_smag * del2(u)) ---
    del2_u = vector_laplacian_del2_3d(u_edge_3d, mesh)  # (nEdges, nlev)
    intermediate = A_smag * del2_u                        # (nEdges, nlev)
    if mid_refresh is not None:
        # Distributed mid-operator refresh — see vector_laplacian_del4_3d.
        (intermediate,) = mid_refresh(intermediate)
    return -vector_laplacian_del2_3d(intermediate, mesh)  # (nEdges, nlev)


def smagorinsky_laplacian_3d(u_edge_3d, mesh, C_smag_lap):
    """Smagorinsky Laplacian viscosity: ``A_smag * del2(u)``.

    Flow-dependent Laplacian viscosity using the Smagorinsky (1963)
    formulation.  Unlike the biharmonic variant, this applies the
    spatially varying coefficient directly to a single Laplacian —
    stronger dissipation at all scales where strain is large.

    The effective viscosity at each edge is::

        A_smag = (C_smag_lap · Δ)² · |D|

    where Δ is the geometric-mean grid scale and |D| is the total
    deformation (strain rate magnitude = sqrt(tension² + shearing²)).

    This is the MPAS Voronoi equivalent of the lat-lon ``C_smag_lap``
    scheme in ``ocean_pe_latlon_cgrid.py``.

    Parameters
    ----------
    u_edge_3d : jax.Array, shape (nEdges, nlev)
        Normal velocity at edges.
    mesh : VoronoiMesh
    C_smag_lap : float
        Dimensionless Smagorinsky coefficient (typical 0.1-0.3 for
        Laplacian; higher than biharmonic because it's less scale-
        selective).

    Returns
    -------
    jax.Array, shape (nEdges, nlev)
        Viscous tendency (to be ADDED to du/dt).
    """
    # --- Strain rate at native TRiSK locations ---
    div_c = divergence_cell_3d(u_edge_3d, mesh)   # (nCells, nlev) — tension
    curl_v = curl_vertex_3d(u_edge_3d, mesh)       # (nVertices, nlev) — shearing

    # --- Average to edges ---
    c1, c2 = mesh.cellsOnEdge[0], mesh.cellsOnEdge[1]
    D_T_edge = 0.5 * (div_c[c1] + div_c[c2])      # (nEdges, nlev)

    v0, v1 = mesh.verticesOnEdge[0], mesh.verticesOnEdge[1]
    D_S_edge = 0.5 * (curl_v[v0] + curl_v[v1])    # (nEdges, nlev)

    # Small epsilon prevents NaN gradient of sqrt at zero.
    deformation = jnp.sqrt(D_T_edge**2 + D_S_edge**2 + 1e-30)

    # --- Smagorinsky coefficient [m²/s] at edges ---
    delta_edge = jnp.sqrt(mesh.dcEdge * mesh.dvEdge)  # (nEdges,)
    A_smag = (C_smag_lap * delta_edge[:, None]) ** 2 * deformation  # (nEdges, nlev)

    # --- Single-pass Laplacian: A_smag * del2(u) ---
    del2_u = vector_laplacian_del2_3d(u_edge_3d, mesh)  # (nEdges, nlev)
    return A_smag * del2_u  # (nEdges, nlev)


# ---------------------------------------------------------------------------
# Leith viscosity for MPAS TRiSK C-grid (Leith 1996;
# Fox-Kemper & Menemenlis 2008).
# ---------------------------------------------------------------------------
# On the TRiSK staggering ζ lives at vertices (dual cells) and δ lives at
# cell centres, so the natural place to evaluate the gradients that make up
# A_L is at the edges where u is stored:
#
#   ∂ζ/∂t   ≈  (ζ[v1] − ζ[v0]) / dvEdge         (tangential along dual edge)
#   ∂δ/∂n   ≈  (δ[c2] − δ[c1]) / dcEdge          (normal along primal edge)
#
# A fully-rigorous TRiSK reconstruction of both components of ∇ζ and ∇δ at
# an edge is non-trivial because the orthogonal components of each live on
# different grid elements.  We keep the practically-used approximation
# (common in MPAS-Ocean Leith implementations): use the tangential ζ
# gradient as the dominant contribution, multiplied by √2 to account for
# an isotropic-mesh assumption that |∂ζ/∂n| ≈ |∂ζ/∂t|.  For the modified
# form the normal δ gradient (which *is* native at edges) is added in
# quadrature.  The C_L coefficient should be tuned with this convention
# in mind; the default C_L ≈ 1–2 matches MPAS-Ocean.

def leith_viscosity_edge_3d(u_edge_3d, mesh, C_leith, *, modified=False):
    """Leith viscosity coefficient at edges.

    ``A_L = (C_L * Δ_e)³ * |∇ζ|``  (classical) or
    ``A_L = (C_L * Δ_e)³ * sqrt(|∇ζ|² + |∇δ|²)``  (modified),

    with the edge length scale ``Δ_e = sqrt(dcEdge * dvEdge)`` and
    ``|∇ζ|_edge ≈ √2 · |ζ[v1] − ζ[v0]| / dvEdge`` (isotropic approximation).

    Parameters
    ----------
    u_edge_3d : jax.Array, shape (nEdges, nlev)
    mesh : VoronoiMesh
    C_leith : float
        Dimensionless Leith coefficient (typical 1.0–2.0).
    modified : bool, default False
        Include the divergence-gradient term (Fox-Kemper & Menemenlis 2008).

    Returns
    -------
    jax.Array, shape (nEdges, nlev) — ``A_L`` in m²/s.
    """
    v0 = mesh.verticesOnEdge[0]
    v1 = mesh.verticesOnEdge[1]

    zeta_v = curl_vertex_3d(u_edge_3d, mesh)                 # (nVertices, nlev)
    dv = jnp.maximum(mesh.dvEdge, 1e-10)
    grad_zeta_tan = (zeta_v[v1] - zeta_v[v0]) / dv[:, None]  # (nEdges, nlev)

    # Isotropic-mesh factor of 2: |∇ζ|² ≈ 2 · |∂ζ/∂t|².
    grad_mag_sq = 2.0 * grad_zeta_tan ** 2

    if modified:
        div_c = divergence_cell_3d(u_edge_3d, mesh)          # (nCells, nlev)
        grad_div_norm = gradient_edge_3d(div_c, mesh)         # (nEdges, nlev)
        # The normal component is native at edges; symmetrise with the same
        # isotropy factor as the vorticity gradient for consistency.
        grad_mag_sq = grad_mag_sq + 2.0 * grad_div_norm ** 2

    norm = jnp.sqrt(grad_mag_sq + 1e-30)

    delta_edge = jnp.sqrt(mesh.dcEdge * mesh.dvEdge)          # (nEdges,)
    return (C_leith * delta_edge[:, None]) ** 3 * norm        # (nEdges, nlev)


def leith_biharmonic_3d(u_edge_3d, mesh, C_leith, *, modified=False,
                        mid_refresh=None):
    """Leith-biharmonic viscosity: ``-del2(A_L * del2(u))``.

    Two-pass structure mirroring ``smagorinsky_biharmonic_3d``: the
    spatially varying Leith coefficient is inserted between the two
    vector-Laplacians.

    Parameters
    ----------
    u_edge_3d : jax.Array, shape (nEdges, nlev)
    mesh : VoronoiMesh
    C_leith : float
    modified : bool
        Include the divergence-gradient term.
    mid_refresh : Callable(*edge_fields) -> tuple, optional
        Distributed mid-operator halo refresh of the intermediate
        (see ``vector_laplacian_del4_3d``); ``None`` = serial identity.

    Returns
    -------
    jax.Array, shape (nEdges, nlev) — viscous tendency to ADD to du/dt.
    """
    A_L = leith_viscosity_edge_3d(u_edge_3d, mesh, C_leith, modified=modified)
    del2_u = vector_laplacian_del2_3d(u_edge_3d, mesh)
    intermediate = A_L * del2_u
    if mid_refresh is not None:
        # Distributed mid-operator refresh — see vector_laplacian_del4_3d
        # (two-pass = 4 hops > halo_depth; refresh the coefficient-scaled
        # intermediate so the outer pass reads a fresh ring).
        (intermediate,) = mid_refresh(intermediate)
    return -vector_laplacian_del2_3d(intermediate, mesh)


def apvm_correction_3d(q_vertex_3d, u_edge_3d, mesh, dt):
    """Anticipated PV Method correction for all levels.

    Parameters
    ----------
    q_vertex_3d : jax.Array, shape (nVertices, nlev)
    u_edge_3d : jax.Array, shape (nEdges, nlev)
    mesh : VoronoiMesh
    dt : float

    Returns
    -------
    jax.Array, shape (nVertices, nlev)
    """
    eov = mesh.edgesOnVertex
    voe = mesh.verticesOnEdge

    mask = (eov >= 0).astype(q_vertex_3d.dtype)
    eov_safe = jnp.maximum(eov, 0)

    v0_of_edge = voe[0][eov_safe]
    v1_of_edge = voe[1][eov_safe]
    dq = q_vertex_3d[v1_of_edge] - q_vertex_3d[v0_of_edge]
    dv = mesh.dvEdge[eov_safe]
    dv_safe = jnp.maximum(dv, 1e-10)
    dq_ds = dq / dv_safe[:, :, None]

    vt = tangential_velocity_3d(u_edge_3d, mesh)
    vt_at_edges = vt[eov_safe]

    advection = jnp.sum((vt_at_edges * dq_ds) * mask[:, :, None], axis=0)
    count = jnp.maximum(jnp.sum(mask, axis=0), 1.0)
    u_dot_grad_q = advection / count[:, None]

    return q_vertex_3d - 0.5 * dt * u_dot_grad_q


# ============================================================================
# Biharmonic dissipation on relative vorticity (∇⁴ζ on the dual grid)
# ============================================================================

def vertex_laplacian_3d(phi_vertex_3d, mesh):
    """Laplacian of a vertex-centered scalar on the triangular dual grid.

    Finite-volume discretisation on the dual (vertex-centred) control
    volume::

        (∇²φ)_v = (1/A_v) · Σ_{e∈E(v)} (φ_{v_other(e)} − φ_v) / dvEdge_e · dcEdge_e

    where E(v) is the set of edges incident on vertex v, v_other(e) is the
    other endpoint of edge e, A_v is the triangle area, dvEdge is the
    vertex-to-vertex length along the edge, and dcEdge is the primal
    (cell-to-cell) length perpendicular to it.  This is the dual of the
    standard cell Laplacian ``divergence_cell(gradient_edge(·))`` and is
    exact on hexagonal MPAS meshes to second order.

    Parameters
    ----------
    phi_vertex_3d : jax.Array, shape (nVertices, nlev)
        Vertex-centred scalar.
    mesh : VoronoiMesh

    Returns
    -------
    jax.Array, shape (nVertices, nlev)
        ``∇²φ`` at vertices.
    """
    eov = mesh.edgesOnVertex            # (vertexDegree, nVertices)
    voe = mesh.verticesOnEdge           # (2, nEdges)
    dvEdge = mesh.dvEdge                # (nEdges,)
    dcEdge = mesh.dcEdge                # (nEdges,)
    area_tri = mesh.areaTriangle        # (nVertices,)

    mask = (eov >= 0).astype(phi_vertex_3d.dtype)[:, :, None]  # (vD, nV, 1)
    eov_safe = jnp.maximum(eov, 0)

    v0_of_edge = voe[0][eov_safe]       # (vD, nV)
    v1_of_edge = voe[1][eov_safe]       # (vD, nV)

    # For each (v, local edge i), gather the other vertex.
    v_index = jnp.arange(mesh.nVertices)[None, :]   # (1, nV)
    is_v_at_v0 = (v0_of_edge == v_index)            # (vD, nV)

    phi_v0_gathered = phi_vertex_3d[v0_of_edge]     # (vD, nV, nlev)
    phi_v1_gathered = phi_vertex_3d[v1_of_edge]
    phi_other = jnp.where(is_v_at_v0[..., None],
                          phi_v1_gathered, phi_v0_gathered)
    phi_self = jnp.where(is_v_at_v0[..., None],
                          phi_v0_gathered, phi_v1_gathered)

    dv_gathered = dvEdge[eov_safe][:, :, None]      # (vD, nV, 1)
    dc_gathered = dcEdge[eov_safe][:, :, None]
    # Guard against zero-length halo edges.
    dv_safe = jnp.maximum(dv_gathered, 1e-30)

    contrib = (phi_other - phi_self) / dv_safe * dc_gathered * mask
    lap = jnp.sum(contrib, axis=0) / area_tri[:, None]
    return lap


def biharmonic_vorticity_del4_3d(u_edge_3d, mesh, *, mid_refresh=None):
    """Edge-normal force from biharmonic damping on relative vorticity ζ.

    Returns the edge-normal force that, when added to ``du/dt`` in the
    momentum equation, produces ``−∇⁴ζ`` in the corresponding vorticity
    equation::

        F_e = −∂_τ̂ (∇²ζ_v) = −(∇²ζ_{v1} − ∇²ζ_{v0}) / dvEdge_e

    Taking the curl of this force (i.e. reading back the vorticity
    tendency) yields::

        (curl F)_v = −∇²(∇²ζ_v) = −∇⁴ζ_v

    which is a biharmonic damping of ζ on the dual grid.  Unlike
    ``vector_laplacian_del4_3d`` (which applies biharmonic to the velocity
    u and would also damp ζ through the vector identity *in the continuum*),
    this operator acts on the ζ field *directly* at vertices.  It therefore
    captures the ζ-checkerboard null mode of the energy-conserving PV flux,
    which lives in the kernel of the discrete velocity-to-vorticity map and
    is invisible to the velocity-based biharmonic.

    Note on sign/scaling: the returned array is normalised so that the
    physical tendency is ``du/dt += K_ζ · biharmonic_vorticity_del4_3d(u)``.
    The caller supplies ``K_ζ`` with units ``[m⁴/s]`` (same as ``B_h``).

    Parameters
    ----------
    u_edge_3d : jax.Array, shape (nEdges, nlev)
        Edge-normal velocity.
    mesh : VoronoiMesh

    Returns
    -------
    jax.Array, shape (nEdges, nlev)
        Edge force per unit ``K_ζ``.
    """
    zeta_v = curl_vertex_3d(u_edge_3d, mesh)        # (nVertices, nlev)
    lap_zeta_v = vertex_laplacian_3d(zeta_v, mesh)  # (nVertices, nlev)
    if mid_refresh is not None:
        # Distributed mid-operator refresh (VERTEX channel): the
        # curl -> vertex-Laplacian -> tangential-gradient chain is 3
        # stencil hops from u — one beyond a halo_depth=2 budget — so
        # the vertex Laplacian's ring is re-armed before the final
        # gradient (see vector_laplacian_del4_3d).
        (lap_zeta_v,) = mid_refresh(lap_zeta_v)

    v0 = mesh.verticesOnEdge[0]
    v1 = mesh.verticesOnEdge[1]
    dv_safe = jnp.maximum(mesh.dvEdge, 1e-30)[:, None]
    # Tangential gradient of ∇²ζ along the edge (from v0 to v1).
    grad_tangent = (lap_zeta_v[v1] - lap_zeta_v[v0]) / dv_safe  # (nEdges, nlev)

    # Sign: the discrete operator satisfies
    # ``curl_vertex(grad_tangent(φ)) = −∇²φ`` on this mesh (verified in
    # ``tests/ocean/unit/test_biharmonic_vorticity.py``; correlation +1
    # between ``curl(F)`` and ``−∇⁴ζ`` when F = +grad_tangent(∇²ζ)).
    # Hence returning ``+grad_tangent(∇²ζ)`` gives ``curl(K_ζ·F) = −K_ζ·∇⁴ζ``
    # in the vorticity equation — damping for K_ζ > 0.
    #
    # Note: the caller is responsible for applying ``edge_mask`` to the
    # returned force (same convention as ``vector_laplacian_del4_3d``,
    # ``smagorinsky_biharmonic_3d`` etc.). See ocean-expert audit
    # 2026-04-24 for a discussion of land-contaminated ζ bleeding across
    # partial-land triangles — this is a codebase-wide concern for all
    # curl-based operators, not specific to this one.
    return grad_tangent
