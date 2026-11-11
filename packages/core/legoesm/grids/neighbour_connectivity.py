"""Build :class:`~legoesm.core.neighbour_ops.NeighbourMesh` for concrete grids.

Lat-lon C-grid: structured ``+1``/``-1`` offsets (array slices).  Metrics are
taken from the same expressions and halo helpers the lat-lon C-grid operators
use (:mod:`legoesm.grids.operators_latlon_cgrid`), so the generic operators
agree with them to round-off.  Scope: regular global lat-lon on the local halo
backend; tripolar, north-fold, stored-metric, meridionally-flat and distributed
grids raise.

MPAS / Voronoi: index tables straight from the mesh connectivity, reproducing
the TRiSK kernels of :mod:`legoesm.core.operators_voronoi`.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
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
