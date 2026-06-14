"""Differentiable grid coupling between heterogeneous atmosphere and ocean grids.

The coupler exchanges surface fields (ocean SST / currents -> atmosphere; surface
fluxes atmosphere -> ocean).  When the atmosphere and ocean run on the SAME grid
(the default coupled-driver configuration) no remap is needed.  To let the model
mix-and-match a different atmosphere grid with a different ocean grid, those
exchanges must pass through a regridding layer that is **differentiable**
(end-to-end ``jax.grad`` is a hard goal of legoESM) and, for the flux direction,
**conservative** (the ESM energy / freshwater budgets depend on it).

This module is a thin, numerics-free wrapper around the already-shipped,
already-AD-tested conservative kernel in
``legoesm.grids.conservative_regrid``: the overlap weights are precomputed once
on the host (static int32 indices + float64 area-overlap weights) and applied via
``jax.ops.segment_sum``, which is LINEAR in the field values — so gradients flow
with respect to the field, the weights being compile-time constants.  No new
remap numerics are introduced here.

Scope (first slice): regular lat-lon atmosphere <-> regular lat-lon ocean at
arbitrary (possibly different) resolutions.  Conservative remap to/from
cubed-sphere or MPAS/Voronoi targets, and vector-field (u, v) rotation across
grids with different local east/north bases, are deferred (they need a
spherical-polygon overlap generator and a differentiable rotation respectively).
Scalar remap of u/v on lat-lon <-> lat-lon is valid because the local basis is
shared.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from legoesm.grids.conservative_regrid import (
    ConservativeRegridWeights,
    compute_overlap_weights,
    apply_conservative_regrid,
    cell_edges_1d,
)


def _is_regular_latlon(grid) -> bool:
    """True if ``grid`` exposes the regular lat-lon geometry the remap needs."""
    return (
        hasattr(grid, "lat_v")
        and hasattr(grid, "lon")
        and hasattr(grid, "n_lat")
        and hasattr(grid, "n_lon")
    )


def _is_voronoi(grid) -> bool:
    """True if ``grid`` is an MPAS Voronoi mesh (duck-typed)."""
    return (
        hasattr(grid, "verticesOnCell")
        and hasattr(grid, "nEdgesOnCell")
        and hasattr(grid, "xCell")
        and hasattr(grid, "nCells")
    )


def make_latlon_remapper(src_grid, dst_grid) -> ConservativeRegridWeights:
    """Conservative, differentiable remap weights ``src_grid -> dst_grid``.

    Both grids must be regular lat-lon (e.g. ``LatLonGrid``).  Latitude edges are
    taken from each grid's pole-clamped v-faces (``grid.lat_v``); longitude edges
    are inferred from the uniform centres.  The periodic-longitude seam is closed
    by padding the SOURCE with one wrapped ghost column on each side and folding
    the ghost indices back onto the real source columns, so the returned weights
    index the UNPADDED source field and ``apply_conservative_regrid`` is used
    unchanged.  (One ghost column closes the seam when the destination cells are
    not coarser than ~one source cell — the regime of interest; extreme
    coarsening ratios would need more ghosts.)
    """
    if not (_is_regular_latlon(src_grid) and _is_regular_latlon(dst_grid)):
        raise NotImplementedError(
            "make_latlon_remapper requires regular lat-lon src and dst grids. "
            "Conservative remap to/from cubed-sphere or MPAS/Voronoi grids is not "
            "yet implemented (it needs a spherical-polygon area-overlap weight "
            "generator emitting the same ConservativeRegridWeights triple); see "
            "the deferred grid-coupling work."
        )

    n_src_lat = int(src_grid.n_lat)
    n_src_lon = int(src_grid.n_lon)

    # Latitude: pole-clamped v-faces span exactly [-pi/2, pi/2] for a global grid.
    src_lat_e = np.asarray(src_grid.lat_v, dtype=np.float64)
    dst_lat_e = np.asarray(dst_grid.lat_v, dtype=np.float64)

    # Destination longitude: full periodic circle.
    dst_lon_e = cell_edges_1d(np.asarray(dst_grid.lon), periodic_lon=True)

    # Source longitude: pad with one wrapped ghost column on each side so a
    # destination cell straddling the 0/2pi seam sees full source coverage.
    src_lon = np.asarray(src_grid.lon, dtype=np.float64)
    src_lon_padded = np.concatenate(
        ([src_lon[-1] - 2.0 * np.pi], src_lon, [src_lon[0] + 2.0 * np.pi])
    )
    src_lon_e_padded = cell_edges_1d(src_lon_padded, periodic_lon=False)

    w = compute_overlap_weights(src_lat_e, src_lon_e_padded, dst_lat_e, dst_lon_e)

    # Fold padded-source longitude indices back onto the real (unpadded) columns:
    # padded col 0 -> real (n_src_lon-1); cols 1..n_src_lon -> 0..n_src_lon-1;
    # col n_src_lon+1 -> 0.  i.e. real = (padded - 1) mod n_src_lon.
    n_pad_lon = n_src_lon + 2
    src_flat = np.asarray(w.src_idx_flat)
    j_src = src_flat // n_pad_lon
    i_src_padded = src_flat % n_pad_lon
    i_src_real = (i_src_padded - 1) % n_src_lon
    src_idx_real = (j_src * n_src_lon + i_src_real).astype(np.int32)

    return ConservativeRegridWeights(
        src_idx_flat=jnp.asarray(src_idx_real, dtype=jnp.int32),
        dst_idx_flat=w.dst_idx_flat,
        weights=w.weights,
        src_shape=(n_src_lat, n_src_lon),
        dst_shape=w.dst_shape,
        n_dst_cells=w.n_dst_cells,
    )


class GridRemapper(NamedTuple):
    """Bidirectional differentiable atm<->ocean remap.

    ``a2o`` maps atmosphere-grid fields (surface fluxes) onto the ocean grid;
    ``o2a`` maps ocean-grid fields (SST, surface currents) onto the atmosphere
    grid.  Both are static :class:`ConservativeRegridWeights` (or ``None`` when
    the grids coincide).  ``identity`` is True iff atm and ocean share a grid, in
    which case callers SKIP the remap entirely (preserving byte-identity for the
    standard single-grid coupled run).
    """

    a2o: ConservativeRegridWeights | None
    o2a: ConservativeRegridWeights | None
    identity: bool


def _grids_equivalent(a, b) -> bool:
    """True if the two grids are the same grid (no remap needed).

    Same object always qualifies (the common case: ocean defaults to the atm
    grid).  Two independently-built but identical regular lat-lon grids also
    qualify.  Structural equality of cubed-sphere / MPAS meshes is deferred to
    those grids' own identity (the ``is`` check), so two separate-but-identical
    MPAS meshes currently fall through to the same-family direct-remap branch.
    """
    if a is b:
        return True
    if _is_regular_latlon(a) and _is_regular_latlon(b):
        if (int(a.n_lat), int(a.n_lon)) != (int(b.n_lat), int(b.n_lon)):
            return False
        return bool(
            np.allclose(np.asarray(a.lat), np.asarray(b.lat))
            and np.allclose(np.asarray(a.lon), np.asarray(b.lon))
        )
    return False


def make_grid_remapper(atm_grid, ocean_grid) -> GridRemapper:
    """Construct an atm<->ocean :class:`GridRemapper`, dispatching on grid type.

    - Same grid  -> identity (no weights); the default single-grid coupled
      driver pays nothing and stays byte-identical.
    - Both regular lat-lon (different resolutions) -> direct conservative remap.
    - Heterogeneous grids that share a FAMILY (e.g. both MPAS, both cubed-sphere)
      must be coupled by a DIRECT same-family conservative remap — never routed
      through an intermediate lat-lon grid (which would add error, break
      conservation, and cost a second remap).  Not yet implemented -> explicit
      ``NotImplementedError`` (no silent lat-lon fallback).
    - Cross-family grids -> cross-family overlap remap, also not yet implemented.
    """
    if _grids_equivalent(atm_grid, ocean_grid):
        return GridRemapper(a2o=None, o2a=None, identity=True)
    if _is_regular_latlon(atm_grid) and _is_regular_latlon(ocean_grid):
        return GridRemapper(
            a2o=make_latlon_remapper(atm_grid, ocean_grid),
            o2a=make_latlon_remapper(ocean_grid, atm_grid),
            identity=False,
        )
    # Same-family MPAS Voronoi <-> Voronoi: a DIRECT conservative remap (never
    # routed through an intermediate lat-lon grid).
    if _is_voronoi(atm_grid) and _is_voronoi(ocean_grid):
        from legoesm.grids.conservative_regrid_unstructured import (
            compute_mpas_to_mpas_weights,
        )
        return GridRemapper(
            a2o=compute_mpas_to_mpas_weights(atm_grid, ocean_grid),
            o2a=compute_mpas_to_mpas_weights(ocean_grid, atm_grid),
            identity=False,
        )
    same_family = type(atm_grid) is type(ocean_grid)
    raise NotImplementedError(
        f"Differentiable atm<->ocean coupling between "
        f"{type(atm_grid).__name__} (atm) and {type(ocean_grid).__name__} "
        f"(ocean) is not implemented yet. "
        + (
            "Build a DIRECT same-family conservative remap for this grid type "
            "(do NOT route same-family grids through an intermediate lat-lon "
            "grid)."
            if same_family
            else "A cross-family spherical-overlap conservative remap is "
            "required (the deferred any-to-any grid-coupling work)."
        )
    )


def remap_field(field, weights: ConservativeRegridWeights | None):
    """Apply remap ``weights`` to ``field`` (differentiable); pass-through if None.

    ``field`` has its TRAILING axes equal to ``weights.src_shape`` (rank 1 MPAS,
    2 lat-lon, or 3 cube); leading axes (levels, tracers, ensemble) are
    vmap-broadcast by the underlying kernel.
    """
    if weights is None:
        return field
    return apply_conservative_regrid(field, weights)


def remap_surface_fields(obj, weights: ConservativeRegridWeights | None):
    """Remap every 2-D surface field of a pytree ``obj`` onto the target grid.

    A leaf is remapped iff its last two axes equal ``weights.src_shape`` (i.e. it
    is a surface field on the source grid); leaves of any other shape (scalars,
    config, vertical profiles) pass through unchanged.  Differentiable; a
    ``None`` ``weights`` (identity coupling) is an exact pass-through, so the
    standard single-grid coupled run is byte-identical.
    """
    if weights is None:
        return obj
    src_shape = tuple(weights.src_shape)
    nd = len(src_shape)

    def _maybe(x):
        # Remap a leaf iff its trailing axes are the source grid (rank 1 for
        # MPAS (nCells,), 2 for lat-lon, 3 for cube (6,n,n)); scalars, vertical
        # profiles, and other shapes pass through unchanged.
        if (
            hasattr(x, "shape")
            and getattr(x, "ndim", 0) >= nd
            and tuple(x.shape[-nd:]) == src_shape
        ):
            return apply_conservative_regrid(x, weights)
        return x

    return jax.tree_util.tree_map(_maybe, obj)
