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
operator, by the tangent-plane method.  Collocated, so one index table serves
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
    ``(nx, ny, nz)`` at every node.  All four share one ``(n, n_nodes)`` table.
    """

    grad: tuple[Connectivity, Connectivity, Connectivity]
    lap: Connectivity
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
    p = jnp.stack([xs[..., 0]**i * xs[..., 1]**j for i, j in exps], -1)  # (N, n, M)
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
    m = phs_degree
    x = xyz / jnp.linalg.norm(xyz, axis=-1, keepdims=True)              # (N, 3)
    # Tangent frame from the coordinate axis least aligned with the normal.
    # The axis choice is piecewise constant (zero derivative) and the weights
    # do not depend on the in-plane rotation of the frame.
    axis = jax.nn.one_hot(jnp.argmin(jnp.abs(x), axis=-1), 3, dtype=x.dtype)
    e1 = axis - jnp.sum(axis * x, -1, keepdims=True) * x
    e1 = e1 / jnp.linalg.norm(e1, axis=-1, keepdims=True)
    e2 = jnp.cross(x, e1)
    d = x[jnp.asarray(stencil)] - x[:, None, :]                          # (N, n, 3)
    xi = jnp.stack([jnp.sum(d * e1[:, None, :], -1),
                    jnp.sum(d * e2[:, None, :], -1)], axis=-1)           # (N, n, 2)
    # Scale by the stencil radius (Listing 8); weights are scale-covariant.
    s = jnp.sqrt(jnp.max(jnp.sum(xi**2, -1), axis=-1))                  # (N,)
    xs = xi / s[:, None, None]
    w = phs_planar_weights(xs, phs_degree=m, poly_degree=poly_degree)
    w1 = w[..., 0] / s[:, None]
    w2 = w[..., 1] / s[:, None]
    lap = w[..., 2] / s[:, None] ** 2
    grad = e1[:, None, :] * w1[..., None] + e2[:, None, :] * w2[..., None]
    return grad, lap


def rbffd_sphere_mesh(
    xyz: Any, radius: float, *, n: int, phs_degree: int, poly_degree: int,
) -> RBFFDMesh:
    """RBF-FD mesh on a sphere of ``radius`` with nodes ``xyz (n_nodes, 3)``."""
    stencil = rbffd_stencils(xyz, n)
    x = jnp.asarray(xyz)
    grad, lap = rbffd_sphere_weights(x, stencil, phs_degree=phs_degree,
                                     poly_degree=poly_degree)
    unit = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
    nb = table(jnp.asarray(stencil.T))
    one = jnp.ones((1, 1), dtype=x.dtype)

    def conn(w):
        return Connectivity(nb, one, w.T, one)

    return RBFFDMesh(
        grad=tuple(conn(grad[..., i] / radius) for i in range(3)),
        lap=conn(lap / radius**2),
        normal=(unit[:, 0], unit[:, 1], unit[:, 2]),
        radius=radius,
    )
