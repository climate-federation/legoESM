"""Build :class:`~legoesm.core.neighbour_ops.NeighbourMesh` for concrete grids.

Lat-lon C-grid: structured ``+1``/``-1`` offsets (array slices).  Metrics are
taken from the same expressions and halo helpers the lat-lon C-grid operators
use (:mod:`legoesm.grids.operators_latlon_cgrid`), so the generic operators
agree with them to round-off.  Scope: regular global lat-lon on the local halo
backend; tripolar, north-fold, stored-metric, meridionally-flat and distributed
grids raise.

MPAS / Voronoi: index tables straight from the mesh connectivity, reproducing
the TRiSK kernels of :mod:`legoesm.core.operators_voronoi`.

RBF-FD on the sphere (arXiv 2610.25484 App. B, Listings 8-9): every node gets a
stencil of its ``n`` nearest nodes; polyharmonic-spline + polynomial weights
for the surface gradient (Cartesian components) and the Laplace-Beltrami
operator, by the tangent-plane method, plus conservatively corrected copies
of the divergence and Laplacian for the mass equation.  Collocated, so one index table serves
every operator; the apply is the table gather + neighbour sum.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.core.neighbour_ops import Connectivity, NeighbourMesh, offsets, table


def _check_latlon_scope(grid: Any) -> None:
    from legoesm.grids.halo import get_halo_backend
    from legoesm.grids.halo_latlon import (
        get_meridionally_flat,
        get_meridionally_periodic,
    )
    from legoesm.grids.operators_latlon_cgrid import (
        fold_is_local,
        is_tripolar,
        north_fold_mask,
    )
    from legoesm.parallel.latlon_spmd import spmd_pole_end_masks

    unsupported = {
        "tripolar grid": is_tripolar(grid),
        "north fold": fold_is_local(grid) or north_fold_mask(grid) is not None,
        "stored grid metric (rich geometry)": hasattr(grid, "dx_v"),
        "not pole-to-pole": not math.isclose(grid.n_lat * float(grid.dlat), math.pi,
                                             rel_tol=1e-12),
        "meridionally flat topology": get_meridionally_flat(),
        "meridionally periodic topology": get_meridionally_periodic(),
        "non-local halo backend": get_halo_backend() != "local",
        "SPMD latitude bands": spmd_pole_end_masks() is not None,
    }
    bad = [name for name, hit in unsupported.items() if hit]
    if bad:
        raise ValueError(
            "latlon_neighbour_mesh supports only a regular global lat-lon grid "
            f"on one device; got: {', '.join(bad)}")


def latlon_neighbour_supported(grid: Any) -> bool:
    """True when :func:`latlon_neighbour_mesh` can build ``grid``'s mesh
    (False for out-of-scope grids and for traced grid geometry)."""
    try:
        _check_latlon_scope(grid)
    except (ValueError, jax.errors.ConcretizationTypeError):
        return False
    return True


def latlon_neighbour_mesh(grid: Any) -> NeighbourMesh:
    """Neighbour mesh of a regular global lat-lon C-grid.

    Edge fields are ``(u, v)``: ``u`` (n_lat, n_lon+1) on longitude faces
    (periodic duplicate last column), ``v`` (n_lat+1, n_lon) on latitude faces
    (zero at the poles).  Vertices are (n_lat+1, n_lon+1).  The scope check
    reads the geometry as Python values, so the grid must be concrete (not
    traced), as it is when a model closes over it.
    """
    from legoesm.grids.halo_latlon import (
        pad_halo_latlon,
        pad_with_pole_bc_lat,
        zero_polar_lat_ends,
    )
    from legoesm.grids.operators_latlon_cgrid import pad_lon_cgrid

    _check_latlon_scope(grid)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    radius, dlon = grid.radius, grid.dlon
    dtype = grid.lat.dtype

    def _pm(*s):
        return jnp.asarray(s, dtype=dtype)[:, None, None]

    def _rows(a, ncol):
        return jnp.broadcast_to(a[:, None], (a.shape[0], ncol))

    # --- cell <- 4 faces (divergence): east u, west u, north v, south v ---
    face_dy = grid.dy * 0.5                                   # (n_lat,)
    lat_pad0 = pad_with_pole_bc_lat(grid.lat, halo=1, south_value=0.0,
                                    north_value=0.0)
    cos_lat_v = zero_polar_lat_ends(jnp.cos(0.5 * (lat_pad0[:-1] + lat_pad0[1:])))
    face_dx = radius * cos_lat_v * dlon                            # (n_lat+1,)
    c2e = Connectivity(
        neighbours=offsets(
            [(0, 0, 1), (0, 0, 0), (1, 1, 0), (1, 0, 0)], (n_lat, n_lon),
            pads=[(None, (0, 0)), (None, (0, 0))]),
        sign=_pm(1, -1, 1, -1),
        weight=jnp.stack([_rows(face_dy, n_lon), _rows(face_dy, n_lon),
                          _rows(face_dx[1:], n_lon), _rows(face_dx[:-1], n_lon)]),
        mask=jnp.ones((1, 1, 1), dtype=dtype),
    )

    # --- u face <- west/east cell (periodic in longitude) ---
    e2c_u = Connectivity(
        neighbours=offsets(
            [(0, 0, -1), (0, 0, 0)], (n_lat, n_lon + 1),
            pads=[(lambda h: pad_lon_cgrid(h, halo=1), (0, 1))]),
        sign=_pm(-1, 1),
        weight=jnp.ones((1, 1, 1), dtype=dtype),
        mask=jnp.ones((1, 1, 1), dtype=dtype),
    )
    dx_u = _rows(radius * dlon * grid.cos_lat, n_lon + 1)

    # --- v face <- south/north cell (wall at the poles: mask 0) ---
    pole_mask_v = zero_polar_lat_ends(jnp.ones((n_lat + 1, n_lon), dtype=dtype))
    e2c_v = Connectivity(
        neighbours=offsets(
            [(0, -1, 0), (0, 0, 0)], (n_lat + 1, n_lon),
            pads=[(lambda h: pad_halo_latlon(h, halo=1)[:, 1:-1], (1, 0))]),
        sign=_pm(-1, 1),
        weight=jnp.ones((1, 1, 1), dtype=dtype),
        mask=pole_mask_v[None],
    )
    dy_h = grid.dy * 0.5
    dy_h_pad = jnp.pad(dy_h, (1, 1), mode="edge")
    dy_v = _rows(0.5 * (dy_h_pad[1:] + dy_h_pad[:-1]), n_lon)

    # --- vertex <- south u, north u, east v, west v (circulation) ---
    cos_ext_q = pad_with_pole_bc_lat(grid.cos_lat, halo=1, south_value=0.0,
                                     north_value=0.0)
    dx_ext = radius * cos_ext_q * dlon                             # (n_lat+2,)
    dy_edge = jnp.pad(0.5 * (dy_h[1:] + dy_h[:-1]), (1, 1), mode="edge")
    pole_mask_q = zero_polar_lat_ends(jnp.ones((n_lat + 1, n_lon + 1), dtype=dtype))
    v2e = Connectivity(
        neighbours=offsets(
            [(0, -1, 0), (0, 0, 0), (1, 0, 0), (1, 0, -1)], (n_lat + 1, n_lon + 1),
            pads=[(lambda u: pad_with_pole_bc_lat(u, halo=1, south_value=0.0,
                                                  north_value=0.0), (1, 0)),
                  (lambda v: pad_lon_cgrid(v, halo=1), (0, 1))]),
        sign=_pm(1, -1, 1, -1),
        weight=jnp.stack([_rows(dx_ext[:-1], n_lon + 1), _rows(dx_ext[1:], n_lon + 1),
                          _rows(dy_edge, n_lon + 1), _rows(dy_edge, n_lon + 1)]),
        mask=pole_mask_q[None],
    )
    if hasattr(grid, "area_q"):
        a_vertex = jnp.abs(grid.area_q[:, 0]).astype(dtype)
    else:
        lat_ext_q = pad_with_pole_bc_lat(grid.lat, halo=1,
                                         south_value=-math.pi / 2.0,
                                         north_value=math.pi / 2.0)
        sin_ext = jnp.sin(lat_ext_q)
        sin_ext = jnp.concatenate([jnp.full((1,), -1.0, dtype=sin_ext.dtype),
                                   sin_ext[1:-1],
                                   jnp.full((1,), 1.0, dtype=sin_ext.dtype)])
        a_vertex = radius**2 * dlon * jnp.abs(sin_ext[1:] - sin_ext[:-1])
    a_vertex = jnp.where(a_vertex > 0, a_vertex, 1.0)

    return NeighbourMesh(
        c2e=c2e, e2c=(e2c_u, e2c_v), v2e=v2e,
        area_cell=grid.area,
        dc_edge=(dx_u, dy_v),
        area_vertex=_rows(a_vertex, n_lon + 1),
    )


def mpas_neighbour_mesh(mesh: Any) -> NeighbourMesh:
    """Neighbour mesh of an MPAS/Voronoi mesh (one edge family, ``(u_edge,)``)."""
    dtype = mesh.dvEdge.dtype
    eoc, eov = mesh.edgesOnCell, mesh.edgesOnVertex
    eoc_safe, eov_safe = jnp.maximum(eoc, 0), jnp.maximum(eov, 0)
    one = jnp.ones((1, 1), dtype=dtype)
    return NeighbourMesh(
        c2e=Connectivity(table(eoc_safe), mesh.edgeSignOnCell,
                         mesh.dvEdge[eoc_safe], (eoc >= 0).astype(dtype)),
        # No -1 guard, exactly like ``gradient_edge``: global meshes have two
        # cells on every edge; a regional mesh's boundary edges would gather
        # the last cell, as the reference kernel does.
        e2c=(Connectivity(table(mesh.cellsOnEdge),
                          jnp.asarray([-1, 1], dtype=dtype)[:, None], one, one),),
        v2e=Connectivity(table(eov_safe), mesh.edgeSignOnVertex,
                         mesh.dcEdge[eov_safe], (eov >= 0).astype(dtype)),
        area_cell=mesh.areaCell,
        dc_edge=(mesh.dcEdge,),
        area_vertex=mesh.areaTriangle,
    )


# ============================================================================
# RBF-FD (collocated, sphere)
# ============================================================================

class RBFFDMesh(NamedTuple):
    """Collocated RBF-FD operators on a sphere of radius ``radius``.

    ``grad[i]`` is the Cartesian component ``i`` of the surface gradient,
    ``lap`` the Laplace-Beltrami operator, ``normal`` the outward unit normal
    ``(nx, ny, nz)`` at every node.  ``div`` and ``lap_mass`` are the same
    operators with the polynomial-preserving conservative correction
    (:func:`polynomial_preserving_correction`): ``sum(area * div(F)) = 0`` for
    every tangent field ``F`` and ``sum(area * lap_mass(phi)) = 0`` for every
    ``phi``, to round-off.  All share one ``(n, n_nodes)`` table.

    Known limits: the connectivities hold the neighbour layer's closure-based
    index tables, so the mesh cannot be passed as a ``jax.jit`` argument (close
    over it instead), as for the lat-lon and MPAS neighbour meshes; ``div`` and
    ``lap_mass`` come from a host-side sparse solve, so no derivative with
    respect to the node positions or areas flows through the correction.
    """

    grad: tuple[Connectivity, Connectivity, Connectivity]
    lap: Connectivity
    div: tuple[Connectivity, Connectivity, Connectivity]
    lap_mass: Connectivity
    normal: tuple[jax.Array, jax.Array, jax.Array]
    radius: float


def rbffd_stencils(xyz: Any, n: int) -> np.ndarray:
    """``(n_nodes, n)`` indices of each node's ``n`` nearest nodes (self first).

    Setup-time (SciPy k-d tree on the positions normalised to the unit
    sphere); not traced.
    """
    from scipy.spatial import cKDTree

    x = np.asarray(xyz, dtype=np.float64)
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    _, idx = cKDTree(x).query(x, k=n)
    return np.asarray(idx, dtype=np.int32)


def _poly_exponents(degree: int) -> list[tuple[int, int]]:
    return [(i - j, j) for i in range(degree + 1) for j in range(i + 1)]


def phs_planar_weights(xs: jax.Array, *, phs_degree: int,
                       poly_degree: int) -> jax.Array:
    """RBF-FD weights at the origin for batched planar stencils (Listing 8).

    ``xs (B, n, 2)`` are stencil points (scaled, centre at the origin);
    returns ``(B, n, 3)`` weights for ``d/dxi1``, ``d/dxi2`` and the planar
    Laplacian, from PHS ``r**phs_degree`` augmented with all monomials of total
    degree ``<= poly_degree``.
    """
    m = phs_degree
    n = xs.shape[1]

    diff = xs[:, :, None, :] - xs[:, None, :, :]
    a_phs = jnp.sum(diff**2, -1) ** (m / 2)                              # (N, n, n)
    exps = _poly_exponents(poly_degree)
    p = polynomial_matrix(xs, poly_degree)                               # (N, n, M)
    n_poly = len(exps)
    zeros = jnp.zeros((xs.shape[0], n_poly, n_poly), dtype=xs.dtype)
    lhs = jnp.concatenate([
        jnp.concatenate([a_phs, p], -1),
        jnp.concatenate([jnp.swapaxes(p, 1, 2), zeros], -1)], axis=1)

    # Operators applied to phi(|y - x_j|) at y = 0 (the centre).
    r2 = jnp.sum(xs**2, -1)                                              # (N, n)
    r_m2 = r2 ** ((m - 2) / 2)
    rhs_phs = jnp.stack([-m * r_m2 * xs[..., 0], -m * r_m2 * xs[..., 1],
                         m * m * r_m2], -1)                              # (N, n, 3)
    rhs_poly = np.zeros((n_poly, 3))
    for k, (i, j) in enumerate(exps):
        rhs_poly[k] = [float((i, j) == (1, 0)), float((i, j) == (0, 1)),
                       2.0 * float((i, j) in ((2, 0), (0, 2)))]
    rhs = jnp.concatenate([
        rhs_phs,
        jnp.broadcast_to(jnp.asarray(rhs_poly, dtype=xs.dtype),
                         (xs.shape[0], n_poly, 3))], axis=1)

    return jnp.linalg.solve(lhs, rhs)[:, :n, :]


def rbffd_tangent_coordinates(xyz: Any, stencil: Any, xp: Any = jnp):
    """Per-stencil tangent frame and projected coordinates on the unit sphere.

    Returns ``e1, e2 (N, 3)`` (orthonormal tangent frame at each centre),
    ``xs (N, n, 2)`` (stencil projected orthogonally onto the tangent plane,
    scaled by the stencil radius) and that radius ``s (N,)``.  ``xp`` is the
    array namespace (``numpy`` for host float64 independent of JAX x64).
    """
    x = xyz / xp.linalg.norm(xyz, axis=-1, keepdims=True)               # (N, 3)
    # Tangent frame from the coordinate axis least aligned with the normal.
    # The axis choice is piecewise constant (zero derivative) and the weights
    # do not depend on the in-plane rotation of the frame.
    axis = xp.eye(3, dtype=x.dtype)[xp.argmin(xp.abs(x), axis=-1)]
    e1 = axis - xp.sum(axis * x, -1, keepdims=True) * x
    e1 = e1 / xp.linalg.norm(e1, axis=-1, keepdims=True)
    e2 = xp.cross(x, e1)
    d = x[xp.asarray(stencil)] - x[:, None, :]                           # (N, n, 3)
    xi = xp.stack([xp.sum(d * e1[:, None, :], -1),
                   xp.sum(d * e2[:, None, :], -1)], axis=-1)             # (N, n, 2)
    # Scale by the stencil radius (Listing 8); weights are scale-covariant.
    s = xp.sqrt(xp.max(xp.sum(xi**2, -1), axis=-1))                     # (N,)
    return e1, e2, xi / s[:, None, None], s


def polynomial_matrix(xs: Any, poly_degree: int, xp: Any = jnp) -> Any:
    """``(B, n, M)`` monomials of total degree ``<= poly_degree`` at ``xs``."""
    return xp.stack([xs[..., 0]**i * xs[..., 1]**j
                     for i, j in _poly_exponents(poly_degree)], -1)


def rbffd_sphere_weights(
    xyz: jax.Array, stencil: Any, *, phs_degree: int, poly_degree: int,
) -> tuple[jax.Array, jax.Array]:
    """Tangent-plane RBF-FD weights on the UNIT sphere (differentiable in ``xyz``).

    Each stencil is projected orthogonally onto the tangent plane at its
    centre; 2-D PHS ``r**phs_degree`` + polynomials of total degree
    ``poly_degree`` give weights for ``d/dxi1``, ``d/dxi2`` and the planar
    Laplacian at the centre.  Projection makes the chart metric ``I + O(xi^2)``,
    so these are the surface derivatives and the Laplace-Beltrami operator at
    the centre.  Returns ``grad (n_nodes, n, 3)`` (Cartesian components of the
    surface gradient) and ``lap (n_nodes, n)``.  Divide by the radius
    (squared for ``lap``) for a sphere of radius ``a``.
    """
    e1, e2, xs, s = rbffd_tangent_coordinates(xyz, stencil)
    w = phs_planar_weights(xs, phs_degree=phs_degree, poly_degree=poly_degree)
    w1 = w[..., 0] / s[:, None]
    w2 = w[..., 1] / s[:, None]
    lap = w[..., 2] / s[:, None] ** 2
    grad = e1[:, None, :] * w1[..., None] + e2[:, None, :] * w2[..., None]
    return grad, lap


def conservative_column_correction(
    w: jax.Array, stencil: Any, area: jax.Array, normal: jax.Array | None = None,
) -> jax.Array:
    """Minimal-norm change of the stencil weights ``w (n_nodes, n[, 3])`` so the
    area-weighted column sums vanish: ``sum_i area_i w~_ij = 0`` for a scalar
    operator, or only its component tangent to ``normal_j`` for a vector
    (surface-derivative) operator, since ``int grad_s(phi) dA`` is normal.

    ``w~_ij = w_ij - area_i t_j / sum_{i: j in stencil(i)} area_i^2`` with
    ``t_j`` the (tangential part of the) column sum.  Used to remove the small
    residual of :func:`polynomial_preserving_correction`; on its own it moves
    every weight by ~1/n (the column sums are O(1)), which ruins accuracy.
    """
    idx = jnp.asarray(stencil)
    n_nodes = idx.shape[0]
    a = area[:, None]
    q = jax.ops.segment_sum(jnp.broadcast_to(a**2, idx.shape).ravel(), idx.ravel(),
                            num_segments=n_nodes)
    if w.ndim == 2:
        t = jax.ops.segment_sum((a * w).ravel(), idx.ravel(), num_segments=n_nodes)
        return w - a * (t / q)[idx]
    col = jax.ops.segment_sum((a[..., None] * w).reshape(-1, 3), idx.ravel(),
                              num_segments=n_nodes)
    t = col - jnp.sum(col * normal, -1, keepdims=True) * normal
    return w - a[..., None] * (t / q[:, None])[idx]


def polynomial_preserving_correction(
    w: Any, stencil: Any, xyz: Any, area: Any, *, poly_degree: int,
    refine_steps: int,
) -> tuple[jax.Array, float]:
    """Conservative weights that keep every row exact on its polynomials.

    Minimises ``sum_i ||delta_i||^2`` subject to ``P_i^T delta_i = 0`` (row ``i``
    stays exact on the degree-``<= poly_degree`` polynomials of its tangent
    plane) and the column condition of :func:`conservative_column_correction`.
    Lagrange: ``delta_i = -area_i Pi_i lambda[S_i]`` with ``Pi_i`` the projector
    off ``span(P_i)``, and ``M lambda = t``, ``M = sum_i area_i^2 E_i^T Pi_i E_i``.
    A vector (surface-derivative) operator ``w (N, n, 3)`` is corrected in each
    row's tangent frame (both tangent derivatives stay exact) with the
    tangential column condition in each column's frame (``2N`` unknowns).
    Scalar case: ``M`` annihilates constants and so does every ``Pi_i``, so
    ``lambda_0 = 0`` is pinned (``delta`` is the same for every solution).

    Solved on the host at setup: sparse LU (SciPy) plus ``refine_steps`` steps
    of iterative refinement; the remaining residual is removed by
    :func:`conservative_column_correction`.  Returns the weights and the
    relative residual ``||M lambda - t|| / ||t||``.

    KNOWN LIMITS: (1) the result is a constant for JAX, so derivatives with
    respect to the node positions and areas do not flow through the correction
    (derivatives with respect to the fields the weights act on are exact).
    (2) Accuracy: the corrected divergence's max-norm error converges at order
    4.4 (642 -> 2562 nodes) then 1.5 (2562 -> 10242), against 4.0 / 3.9
    uncorrected; even with LU and refinement the solve residual grows with
    node count, 8e-12 / 8e-10 / 9e-10 (divergence) and 3e-11 / 1.5e-9 / 4.6e-9
    (Laplacian) at 642 / 2562 / 10242 nodes, and rows stay polynomial-exact
    only to 1e-10 / 6e-9 / 2.5e-8 relative.  Cause: the correction system is
    ill-conditioned (inferred, condition number not measured).  Williamson TC2 keeps order 4.0
    and error 1.3e-7 at 10242 nodes.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu

    idx = np.asarray(stencil)
    n_nodes, n = idx.shape
    e1, e2, xs, _ = rbffd_tangent_coordinates(
        np.asarray(xyz, dtype=np.float64), idx, xp=np)
    q, _ = np.linalg.qr(polynomial_matrix(xs, poly_degree, xp=np))
    pi = np.eye(n)[None] - q @ np.swapaxes(q, 1, 2)                    # (N, n, n)
    area = np.asarray(area, dtype=np.float64)
    a = area / area.mean()                                              # delta is invariant
    w_np = np.asarray(w, dtype=np.float64)

    def solve(blocks, var_idx, t, pin):
        rows = np.repeat(var_idx, var_idx.shape[1], axis=1).ravel()
        cols = np.tile(var_idx, (1, var_idx.shape[1])).ravel()
        m = sp.csc_matrix((blocks.ravel(), (rows, cols)), shape=(t.size, t.size))
        keep = np.arange(t.size) != pin if pin is not None else slice(None)
        mk = m[keep][:, keep].tocsc()
        lu = splu(mk)
        lam = np.zeros_like(t)
        lam[keep] = lu.solve(t[keep])
        for _ in range(refine_steps):
            lam[keep] += lu.solve(t[keep] - mk @ lam[keep])
        return lam, float(np.linalg.norm(m @ lam - t) / np.linalg.norm(t))

    if w_np.ndim == 2:
        t = np.zeros(n_nodes)
        np.add.at(t, idx, a[:, None] * w_np)
        t -= t.mean()
        lam, resid = solve((a**2)[:, None, None] * pi, idx, t, pin=0)
        out = w_np - a[:, None] * np.einsum("bkl,bl->bk", pi, lam[idx])
        return conservative_column_correction(
            jnp.asarray(out), stencil, jnp.asarray(area)), resid

    # g[i, k, a, b] = e_a(i) . e_b(S_i[k]) maps column-frame multipliers to row frame.
    ei = np.stack([e1, e2], 1)                                           # (N, 2, 3)
    g = np.einsum("iax,ikbx->ikab", ei, ei[idx])
    blocks = (a**2)[:, None, None, None, None] * np.einsum(
        "ikab,ikl,ilac->ikblc", g, pi, g)                                 # (N, n, 2, n, 2)
    var_idx = (2 * idx[:, :, None] + np.arange(2)).reshape(n_nodes, 2 * n)
    col = np.zeros((n_nodes, 3))
    np.add.at(col, idx, a[:, None, None] * w_np)
    t = np.einsum("jax,jx->ja", ei, col).ravel()
    lam, resid = solve(blocks.reshape(n_nodes, 2 * n, 2 * n), var_idx, t, pin=None)
    c = np.einsum("ikab,ikb->ika", g, lam.reshape(n_nodes, 2)[idx])     # row-frame
    c = np.einsum("ikl,ila->ika", pi, c)
    out = w_np - a[:, None, None] * np.einsum("ika,iax->ikx", c, ei)
    normal = np.cross(e1, e2)
    return conservative_column_correction(
        jnp.asarray(out), stencil, jnp.asarray(area), jnp.asarray(normal)), resid


def rbffd_sphere_mesh(
    xyz: Any, radius: float, area: Any, *, n: int, phs_degree: int, poly_degree: int,
    refine_steps: int,
) -> RBFFDMesh:
    """RBF-FD mesh on a sphere of ``radius`` with nodes ``xyz (n_nodes, 3)`` and
    node quadrature weights ``area (n_nodes,)`` (the weights mass is measured
    with, e.g. Voronoi cell areas)."""
    stencil = rbffd_stencils(xyz, n)
    x = jnp.asarray(xyz)
    area = jnp.asarray(area, dtype=x.dtype)
    grad, lap = rbffd_sphere_weights(x, stencil, phs_degree=phs_degree,
                                     poly_degree=poly_degree)
    unit = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
    kw = dict(poly_degree=poly_degree, refine_steps=refine_steps)
    div, _ = polynomial_preserving_correction(grad, stencil, x, area, **kw)
    lap_mass, _ = polynomial_preserving_correction(lap, stencil, x, area, **kw)
    nb = table(jnp.asarray(stencil.T))
    one = jnp.ones((1, 1), dtype=x.dtype)

    def conn(w):
        return Connectivity(nb, one, w.T, one)

    return RBFFDMesh(
        grad=tuple(conn(grad[..., i] / radius) for i in range(3)),
        lap=conn(lap / radius**2),
        div=tuple(conn(div[..., i] / radius) for i in range(3)),
        lap_mass=conn(lap_mass / radius**2),
        normal=(unit[:, 0], unit[:, 1], unit[:, 2]),
        radius=radius,
    )
