"""Conservative, differentiable remap for unstructured spherical meshes (MPAS
Voronoi), via fine-quadrature first-order conservative overlap weights.

The weight precompute runs ONCE on the host (numpy + scipy ``cKDTree``) and emits
the SAME :class:`~legoesm.grids.conservative_regrid.ConservativeRegridWeights`
triple as the regular lat-lon path, so the differentiable apply kernel
(``apply_conservative_regrid`` — a ``segment_sum`` linear in the field values)
and the coupler dispatch (``coupler.grid_remap``) are reused verbatim.  No new
traced numerics are introduced; gradients flow w.r.t. the field exactly as for
lat-lon (the weights are compile-time constants).

Method — fine-quadrature first-order conservative (the SCRIP/ESMF "conserve"
accuracy class):

* Each TARGET cell is split into spherical triangles (fan triangulation of its
  Voronoi polygon), and each triangle into ``n_sub**2`` sub-triangles whose exact
  spherical areas (``spherical_triangle_area``) sum to the cell area.
* Each sub-triangle's area is assigned to the SOURCE cell that owns its centroid.
  For a Voronoi source mesh the owning cell is the one whose generator is the
  nearest center (the definition of a Voronoi cell), found exactly by a
  ``cKDTree`` on the source cell centers.
* The per-pair weight is ``overlap_area / target_cell_area`` where the target
  area is the quadrature-summed sub-triangle area.  Normalising by the SAME area
  the weights integrate makes the weights sum to EXACTLY 1 over any fully-covered
  target cell (so constant fields are preserved exactly).

ACCURACY / CONSERVATION (read before using for FLUXES).  Two properties are
MACHINE-EXACT at any ``n_sub``: constant-field preservation and per-cell
weight-sum == 1 (partition of unity).  The GLOBAL INTEGRAL, however, is conserved
only to the quadrature accuracy of the overlap areas — it is FIRST-ORDER in the
sub-cell size, NOT machine-exact, because each sub-triangle is assigned whole to
the source cell owning its centroid (sub-triangles straddling a source boundary
are mis-attributed).  On coarse meshes / large coarsening ratios the global flux
error can be several percent at feasible ``n_sub`` (it converges as ``n_sub``
grows and shrinks with finer meshes).  This is therefore well-suited to STATE
remap (SST, currents — conservation not required) and to flux remap only where
first-order global conservation is acceptable; for MACHINE-EXACT flux
conservation use the deferred EXACT spherical-polygon-clip generator (same
``ConservativeRegridWeights`` interface, higher implementation cost).

Cube<->cube and cross-family (cube<->MPAS) reuse the same quadrature core and are
deferred (cube corner reconstruction needs a public projector; the core already
generalizes).  Scalar remap only — differentiable vector (u,v) rotation across
grids with different local bases is deferred (see ``coupler.grid_remap``).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

import jax.numpy as jnp

from legoesm.grids.conservative_regrid import ConservativeRegridWeights
from legoesm.grids.voronoi import spherical_triangle_area, shift_near


def _unit(v: np.ndarray) -> np.ndarray:
    """Normalise the last axis to unit length (sphere projection)."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-30)


def _triangle_subcells(a, b, c, n_sub):
    """Tile spherical triangle (unit vectors ``a``,``b``,``c``) into ``n_sub**2``
    sub-triangles; return ``(centroids (M,3) unit, areas (M,))``.

    Areas are exact spherical sub-triangle areas (sum == the triangle's exact
    spherical area).  Centroids are the normalised mean of the three corners
    (used only for source-cell point-location, so an approximate centroid is
    fine).
    """
    def P(i, j):
        # Barycentric (i, j, n_sub-i-j)/n_sub mapped to the sphere.
        w0 = (n_sub - i - j) / n_sub
        w1 = i / n_sub
        w2 = j / n_sub
        return _unit(w0 * a + w1 * b + w2 * c)

    # Cache lattice points so each is built once.
    lattice = {}
    for i in range(n_sub + 1):
        for j in range(n_sub + 1 - i):
            lattice[(i, j)] = P(i, j)

    cents = []
    areas = []
    for i in range(n_sub):
        for j in range(n_sub - i):
            A, B, C = lattice[(i, j)], lattice[(i + 1, j)], lattice[(i, j + 1)]
            cents.append(_unit(A + B + C))
            areas.append(spherical_triangle_area(A, B, C, 1.0))
            if i + j < n_sub - 1:  # the downward-pointing companion triangle
                A2 = lattice[(i + 1, j)]
                B2 = lattice[(i + 1, j + 1)]
                C2 = lattice[(i, j + 1)]
                cents.append(_unit(A2 + B2 + C2))
                areas.append(spherical_triangle_area(A2, B2, C2, 1.0))
    return np.asarray(cents, dtype=np.float64), np.asarray(areas, dtype=np.float64)


def _voronoi_cell_polygons(mesh):
    """Per-cell ordered (CCW) unit-sphere vertex arrays + counts for a VoronoiMesh."""
    vx = np.asarray(mesh.xVertex, dtype=np.float64)
    vy = np.asarray(mesh.yVertex, dtype=np.float64)
    vz = np.asarray(mesh.zVertex, dtype=np.float64)
    v_xyz = _unit(np.stack([vx, vy, vz], axis=1))     # (nVertices, 3)
    voc = np.asarray(mesh.verticesOnCell)             # (maxEdges, nCells)
    neoc = np.asarray(mesh.nEdgesOnCell).astype(int)  # (nCells,)
    return v_xyz, voc, neoc


def compute_mpas_to_mpas_weights(src, dst, *, n_sub: int = 8) -> ConservativeRegridWeights:
    """Conservative, differentiable remap weights ``src -> dst`` for two MPAS
    Voronoi meshes.

    Parameters
    ----------
    src, dst : VoronoiMesh
        Source and target meshes (any resolutions; global).
    n_sub : int
        Per-cell triangle sub-division (``n_sub**2`` sub-triangles per fan
        triangle).  Larger ``n_sub`` => tighter global-integral conservation
        (first-order, convergent).  Conservation gates (constant-field preserved,
        weights sum to 1) are exact at any ``n_sub``.
    """
    # Source point-locator: nearest generator == owning Voronoi cell (exact).
    src_centers = _unit(np.stack([
        np.asarray(src.xCell, dtype=np.float64),
        np.asarray(src.yCell, dtype=np.float64),
        np.asarray(src.zCell, dtype=np.float64),
    ], axis=1))
    tree = cKDTree(src_centers)
    n_src = int(src.nCells)
    n_dst = int(dst.nCells)

    v_xyz, voc, neoc = _voronoi_cell_polygons(dst)
    # Global meshes => shift is a no-op; carried for the regional case.
    L_rad = 2.0 * np.pi

    # Accumulate overlap area per (dst, src) pair.
    overlap: dict[tuple[int, int], float] = {}
    dst_area_q = np.zeros(n_dst, dtype=np.float64)

    for d in range(n_dst):
        k = int(neoc[d])
        if k < 3:
            continue
        verts = v_xyz[voc[:k, d]]                 # (k, 3) unit, CCW
        v0 = verts[0]
        verts = np.stack([shift_near(v0, verts[m], L_rad) for m in range(k)])
        v0 = verts[0]
        cent_list = []
        area_list = []
        for i in range(1, k - 1):                 # fan triangulation from v0
            c_q, a_q = _triangle_subcells(v0, verts[i], verts[i + 1], n_sub)
            cent_list.append(c_q)
            area_list.append(a_q)
        cents = np.concatenate(cent_list, axis=0)
        areas = np.concatenate(area_list, axis=0)
        _, s_idx = tree.query(cents, k=1)
        for s, a in zip(np.asarray(s_idx).astype(int), areas):
            key = (d, int(s))
            overlap[key] = overlap.get(key, 0.0) + float(a)
        dst_area_q[d] = float(areas.sum())

    src_idx = np.empty(len(overlap), dtype=np.int32)
    dst_idx = np.empty(len(overlap), dtype=np.int32)
    wts = np.empty(len(overlap), dtype=np.float64)
    for p, ((d, s), a) in enumerate(overlap.items()):
        src_idx[p] = s
        dst_idx[p] = d
        wts[p] = a / max(dst_area_q[d], 1e-30)

    return ConservativeRegridWeights(
        src_idx_flat=jnp.asarray(src_idx, dtype=jnp.int32),
        dst_idx_flat=jnp.asarray(dst_idx, dtype=jnp.int32),
        weights=jnp.asarray(wts, dtype=jnp.float64),
        src_shape=(n_src,),
        dst_shape=(n_dst,),
        n_dst_cells=n_dst,
    )
