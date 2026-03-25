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

import jax
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
    nCells = mesh.nCells
    maxEdges = mesh.maxEdges

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
    nVertices = mesh.nVertices
    vertexDegree = mesh.vertexDegree

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
    nCells = mesh.nCells
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
    u_at_edges = u_edge[eov_safe]  # (vertexDegree, nVertices)

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

    # Advective derivative at vertex: average of edge contributions
    advection = jnp.sum((vt_at_edges * dq_ds) * mask, axis=0)
    count = jnp.maximum(jnp.sum(mask, axis=0), 1.0)
    u_dot_grad_q = advection / count

    return q_vertex - 0.5 * dt * u_dot_grad_q
