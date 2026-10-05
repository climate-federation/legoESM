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

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.grids.conservative_regrid import (
    ConservativeRegridWeights,
    apply_conservative_regrid,
    cell_edges_1d,
    compute_overlap_weights,
)


def _is_regular_latlon(grid) -> bool:
    """True if ``grid`` exposes the regular lat-lon geometry the remap needs.

    "Regular lat-lon" here means the raw 1-D :class:`LatLonGrid`, NOT a C-grid
    geometry.  ``lat_v`` used to separate the two on its own, and stopped when
    ``LatLonCGridGeometry`` gained an optional ``lat_v`` of its own: every
    geometry then answered ``hasattr`` True, including a TRIPOLE built by
    folding a regular one (``create_synthetic_tripole``), which silently broke
    the mutual exclusion with :func:`_is_tripole` documented below and sent a
    tripolar ocean through the separable remap -- measured, 1300 weight pairs
    against the curvilinear generator's 1216.  ``lat_T`` is the discriminator
    that does not rot the same way: it is the 2-D centre array EVERY C-grid
    geometry carries and the raw grid never does, so this restores the exact
    prior answer for all three cases (raw grid True, regular geometry False,
    tripole False) instead of merely patching the tripole one."""
    return (
        getattr(grid, "lat_v", None) is not None
        and not hasattr(grid, "lat_T")
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


def _is_cube(grid) -> bool:
    """True if ``grid`` is a cubed-sphere grid (duck-typed: a ``(6, n, n)``
    ``grid_shape_2d`` with stored Cartesian cell centres)."""
    shp = getattr(grid, "grid_shape_2d", None)
    return (
        shp is not None
        and len(tuple(shp)) == 3
        and int(tuple(shp)[0]) == 6
        and hasattr(grid, "x_cart")
        and not _is_voronoi(grid)
    )


def _is_tripole(grid) -> bool:
    """True if ``grid`` is a curvilinear tripole C-grid with an ACTIVE bipolar
    fold (duck-typed: a :class:`LatLonCGridGeometry` whose ``fold.is_active``).

    A tripole grid carries 2-D centres (``lat_T``/``lon_T``) but no 1-D
    ``lat_v`` ARRAY (the optional field stays ``None``), so
    ``_is_regular_latlon`` is False for it; the two detectors are mutually
    exclusive.  The active-fold gate means a regular
    lat-lon C-grid geometry (``fold.is_active = False``) is NOT treated as
    tripole — it stays on the separable regular-lat-lon remap path."""
    fold = getattr(grid, "fold", None)
    return (
        fold is not None
        and getattr(fold, "is_active", False)
        and hasattr(grid, "lat_T")
        and hasattr(grid, "lon_T")
        and hasattr(grid, "n_lat")
        and hasattr(grid, "n_lon")
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

    # Source longitude: pad with ceil(dst_width / src_width) wrapped ghost
    # columns on EACH side so a destination cell straddling the 0/2pi seam sees
    # full source coverage even when the destination is much coarser than the
    # source (dd/ds > ~3, e.g. 0.25deg -> 1deg).  ONE ghost (the old code) spans
    # only one source cell ds and under-covers once dd exceeds ~3 ds.
    n_dst_lon = int(dst_grid.n_lon)
    n_ghost = max(1, int(np.ceil(n_src_lon / n_dst_lon)))
    src_lon = np.asarray(src_grid.lon, dtype=np.float64)
    src_lon_padded = np.concatenate(
        (src_lon[-n_ghost:] - 2.0 * np.pi, src_lon, src_lon[:n_ghost] + 2.0 * np.pi)
    )
    src_lon_e_padded = cell_edges_1d(src_lon_padded, periodic_lon=False)

    # DEFAULT 'dstarea' normalisation and NO polar_fill, deliberately.  This path
    # carries the conservative FLUX direction (the ESM energy / freshwater budgets
    # depend on it), and both 'fracarea' and polar_fill trade strict conservation
    # for correct magnitude -- the right trade for intensive forcing fields, the
    # wrong one here.  So every destination cell must be fully covered and any
    # shortfall is a bug: require_full_coverage is the strict invariant, identical
    # in strength to the original check.
    #
    # This does reject a non-global source band, which is correct rather than
    # incidental: Mercator/DINO satisfy `_is_regular_latlon` too (their lat_v is
    # bounded by lat_max_deg, not +-pi/2), and a lat_max=70 deg Mercator source into
    # a global lat-lon destination has 0.0 coverage in its outer rows -- a constant
    # 290 K SST would arrive as 0 K.
    #
    # No span assertions here: a lat assertion would newly reject legitimate
    # Mercator -> Mercator pairs, and a lon assertion would reject DINO's
    # regional-longitude frame (span 0.84 rad) that passes today.  Those
    # preconditions belong to the forcing callers, whose sources are global
    # datasets and whose ghost padding would otherwise mask a partial axis.
    w = compute_overlap_weights(
        src_lat_e, src_lon_e_padded, dst_lat_e, dst_lon_e,
        require_full_coverage=True,
    )

    # Fold padded-source longitude indices back onto the real (unpadded) columns:
    # the n_ghost left ghosts -> real cols (n_src_lon-n_ghost)..(n_src_lon-1);
    # the n_src_lon middle cols -> 0..n_src_lon-1; the n_ghost right ghosts -> 0..
    # i.e. real = (padded - n_ghost) mod n_src_lon.
    n_pad_lon = n_src_lon + 2 * n_ghost
    src_flat = np.asarray(w.src_idx_flat)
    j_src = src_flat // n_pad_lon
    i_src_padded = src_flat % n_pad_lon
    i_src_real = (i_src_padded - n_ghost) % n_src_lon
    src_idx_real = (j_src * n_src_lon + i_src_real).astype(np.int32)

    return ConservativeRegridWeights(
        src_idx_flat=jnp.asarray(src_idx_real, dtype=jnp.int32),
        dst_idx_flat=w.dst_idx_flat,
        weights=w.weights,
        src_shape=(n_src_lat, n_src_lon),
        dst_shape=w.dst_shape,
        n_dst_cells=w.n_dst_cells,
    )


def make_curvilinear_latlon_remapper(
        src_grid, dst_grid, *, wet_tripole=None) -> ConservativeRegridWeights:
    """Conservative, differentiable remap weights ``src_grid -> dst_grid`` where
    EXACTLY ONE of the two grids is a curvilinear tripole C-grid and the other
    is a regular lat-lon grid.

    Dispatches to the fine-quadrature first-order conservative generator in
    :mod:`legoesm.grids.conservative_regrid_curvilinear` (only the regular grid
    is tiled; the tripole side is located by nearest cell centre).  The returned
    weights are the same :class:`ConservativeRegridWeights` triple as the
    separable lat-lon path, applied unchanged by ``apply_conservative_regrid``.
    """
    from legoesm.grids.conservative_regrid_curvilinear import (
        make_curvilinear_to_regular_weights,
        make_regular_to_curvilinear_weights,
    )
    if _is_regular_latlon(src_grid) and _is_tripole(dst_grid):
        # a2o (atm FLUX -> tripole ocean): UNMASKED conservative partition-of-unity
        # weights.  The flux is gated to the wet ocean domain by the open-water
        # fraction (cross_grid_open_water_fraction) in the driver, which conserves
        # the flux integral there WITHOUT masking the weights (masking a2o would be
        # equivalent for the wet cells and would corrupt fraction remaps).
        return make_regular_to_curvilinear_weights(src_grid, dst_grid)
    if _is_tripole(src_grid) and _is_regular_latlon(dst_grid):
        # o2a (ocean STATE -> atm): EXCLUDE ocean-grid LAND source cells (H4) so a
        # coastal atm SST/current never averages the ocean land fill value.
        return make_curvilinear_to_regular_weights(
            src_grid, dst_grid, src_wet=wet_tripole)
    raise NotImplementedError(
        "make_curvilinear_latlon_remapper requires exactly one regular lat-lon "
        "grid and one tripole (active-fold) curvilinear grid; got "
        f"src={type(src_grid).__name__}, dst={type(dst_grid).__name__}. "
        "Tripole<->tripole and cube<->tripole remaps are not implemented."
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


def make_grid_remapper(atm_grid, ocean_grid, *, ocean_wet=None) -> GridRemapper:
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
    # Regular lat-lon ATM <-> curvilinear TRIPOLE ocean (Phase-2 cross-grid
    # coupling).  Only this orientation is supported — the atmosphere dycore is
    # regular lat-lon; a cube-atm <-> tripole-ocean coupling is cross-family and
    # still deferred (see the NotImplementedError below).
    if _is_regular_latlon(atm_grid) and _is_tripole(ocean_grid):
        return GridRemapper(
            a2o=make_curvilinear_latlon_remapper(atm_grid, ocean_grid),
            o2a=make_curvilinear_latlon_remapper(
                ocean_grid, atm_grid, wet_tripole=ocean_wet),
            identity=False,
        )
    if _is_tripole(atm_grid) and _is_regular_latlon(ocean_grid):
        # Symmetric orientation (regular ocean, tripole "atm") — not a real
        # configuration, but handle it rather than fall through to the generic
        # error so the dispatch is total over {regular, tripole}.
        return GridRemapper(
            a2o=make_curvilinear_latlon_remapper(atm_grid, ocean_grid),
            o2a=make_curvilinear_latlon_remapper(ocean_grid, atm_grid),
            identity=False,
        )
    # Cross-family cubed-sphere ATM <-> regular lat-lon OCEAN (a cube/spectral
    # atmosphere driving a lat-lon 3-D ocean).  First-order conservative
    # spherical-overlap remap (legoesm.grids.conservative_regrid_cubedsphere),
    # constant-preserving + global-integral conserving to quadrature order —
    # the deferred cross-family coupling.  The cube ocean dycore does not exist,
    # so only cube-atm x latlon-ocean (and the symmetric spelling) is wired.
    if _is_cube(atm_grid) and _is_regular_latlon(ocean_grid):
        from legoesm.grids.conservative_regrid_cubedsphere import (
            make_cube_to_latlon_weights,
            make_latlon_to_cube_weights,
        )
        return GridRemapper(
            a2o=make_cube_to_latlon_weights(atm_grid, ocean_grid),
            o2a=make_latlon_to_cube_weights(ocean_grid, atm_grid),
            identity=False,
        )
    if _is_regular_latlon(atm_grid) and _is_cube(ocean_grid):
        from legoesm.grids.conservative_regrid_cubedsphere import (
            make_cube_to_latlon_weights,
            make_latlon_to_cube_weights,
        )
        return GridRemapper(
            a2o=make_latlon_to_cube_weights(atm_grid, ocean_grid),
            o2a=make_cube_to_latlon_weights(ocean_grid, atm_grid),
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


def attach_wet_masks(remapper, atm_grid, ocean_grid, ocean_wet):
    """Re-bake the cross-grid remap weights with the OCEAN's static wet mask.

    Returns a NEW :class:`GridRemapper` whose ``o2a`` (ocean STATE -> atm) weights
    EXCLUDE ocean-grid LAND source cells, so a coastal atm SST/current average
    never ingests the ocean land fill value (H4).  The ``a2o`` (atm FLUX -> ocean)
    weights are the UNCHANGED conservative partition-of-unity remap: the a2o flux
    is gated to the ocean's own wet domain by the open-water fraction
    (``cross_grid_open_water_fraction``) in the driver, which conserves the flux
    integral over the wet ocean WITHOUT masking the weights (masking a2o would be
    equivalent for the wet cells and would corrupt fraction remaps).

    ``ocean_wet`` is the ocean-grid wet mask (1 = ocean, 0 = land), STATIC and
    host-known.  MUST be called AFTER the ocean (and its mask) are initialised --
    the mask does not exist at the first ``make_grid_remapper`` call.  Identity
    (shared-grid) remapper -> returned UNCHANGED (byte-identical single-grid run;
    the traced apply stays a pure ``segment_sum`` in every case).
    """
    if remapper is None or remapper.identity:
        return remapper
    if ocean_wet is None:
        return remapper
    return make_grid_remapper(
        atm_grid, ocean_grid, ocean_wet=np.asarray(ocean_wet))


def cross_grid_open_water_fraction(ocean_wet, sic_on_ocean):
    """Open-water fraction on the OCEAN grid = ocean's OWN wet mask x ice-free.

    ``f_ocean = ocean_wet * (1 - sic_on_ocean)``.  The STATIC land/sea split is
    the ocean's OWN wet mask (1 = ocean) -- NOT a remapped ATM open-water fraction
    (which would impose the atmosphere coastline on the ocean grid and leak /
    starve coastal ocean cells; H3).  Only the DYNAMIC sea-ice concentration is
    remapped from the atmosphere.  Multiplying every conservative a2o flux by this
    factor also GATES the flux to the wet ocean domain (land cells -> 0), so the
    energy / freshwater integral is conserved there.  Differentiable: an
    elementwise multiply of traced fields by the static wet mask.
    """
    sic = jnp.clip(sic_on_ocean, 0.0, 1.0)
    return jnp.asarray(ocean_wet, dtype=sic.dtype) * (1.0 - sic)


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


def rotate_tpoint_currents_to_geographic(u_c, v_c, cos_alpha_u, sin_alpha_u):
    """Rotate grid-aligned T-point currents to geographic east/north.

    On a curvilinear C-grid (the tripole bipolar cap), the prognostic
    velocity components ``(u_c, v_c)`` are aligned with the LOCAL grid axes
    (i, j), whereas the atmosphere/coupler expects geographic (east, north).
    This rotation MUST be applied on the ocean grid BEFORE remapping each
    component as a scalar (once both components share the geographic basis the
    remap is per-component scalar — see the module docstring on shared-basis
    scalar remap of u/v).

    The rotation angle is the SAME ``i``-axis -> geographic-east angle the
    ocean core uses (in the inverse direction) for wind stress
    (``ocean_pe_latlon_cgrid._apply_external_surface_forcing``: it maps
    geographic stress to grid-aligned via ``tau_i = tau_e cosα + tau_n sinα``).
    Currents are the INVERSE of that rotation::

        u_east  = u_i cosα − v_j sinα
        v_north = u_i sinα + v_j cosα

    The angle is supplied at u-faces (``cos_alpha_u``/``sin_alpha_u``, shape
    ``(n_lat, n_lon+1)``); it is averaged to T-centres and renormalised to unit
    length.  OUTSIDE the bipolar cap ``cosα = 1``, ``sinα = 0`` so this is the
    IDENTITY — a regular lat-lon C-grid (and the regular sub-domain of a
    tripole) is bit-exact unchanged.  Differentiable: the rotation is an
    elementwise multiply/add of the traced currents by the static angle arrays.

    Parameters
    ----------
    u_c, v_c : jax.Array, shape ``(n_lat, n_lon)``
        Grid-aligned currents at T-centres (i-, j-components).
    cos_alpha_u, sin_alpha_u : jax.Array, shape ``(n_lat, n_lon+1)``
        u-face rotation angles (i-axis -> geographic east).

    Returns
    -------
    (u_east, v_north) : tuple of jax.Array, shape ``(n_lat, n_lon)``
    """
    cos_a_u = jnp.asarray(cos_alpha_u, dtype=u_c.dtype)
    sin_a_u = jnp.asarray(sin_alpha_u, dtype=u_c.dtype)
    # Average the two bracketing u-faces of each T-cell, then renormalise to a
    # unit (cosα, sinα) so averaging unit vectors does not shrink the rotation.
    cos_T = 0.5 * (cos_a_u[:, :-1] + cos_a_u[:, 1:])
    sin_T = 0.5 * (sin_a_u[:, :-1] + sin_a_u[:, 1:])
    norm = jnp.sqrt(cos_T * cos_T + sin_T * sin_T)
    norm = jnp.where(norm > 1e-12, norm, 1.0)   # safety floor (degenerate cell)
    cos_T = cos_T / norm
    sin_T = sin_T / norm
    u_east = u_c * cos_T - v_c * sin_T
    v_north = u_c * sin_T + v_c * cos_T
    return u_east, v_north


_NN_CHUNK_ELEMS = 2 ** 26   # dot products per chunk (~0.5 GB float64)


def nearest_column_map(
    src_lat, src_lon, dst_lat, dst_lon, *, src_valid=None,
) -> np.ndarray:
    """Index of the nearest source column for every target column.

    Generic point-cloud nearest-neighbour on the sphere, for moving PER-COLUMN
    STATE between any two of the coupler's grids (lat-lon, Voronoi mesh, cube
    face — anything that yields per-column lat/lon in radians).  Chord distance
    through unit vectors: exact ordering, no pole or dateline seams.  This is
    for state carried point-to-point (soil columns, snow, per-column
    parameters); FLUXES need the conservative remappers above, never this.

    ``src_valid``: optional boolean mask; targets then map onto the nearest
    VALID source column (e.g. land state selects only land donors, so ocean
    placeholders can never leak into a land column).
    """
    src_lat = np.asarray(src_lat, np.float64).ravel()
    src_lon = np.asarray(src_lon, np.float64).ravel()
    dst_lat = np.asarray(dst_lat, np.float64).ravel()
    dst_lon = np.asarray(dst_lon, np.float64).ravel()
    if src_valid is not None:
        idx = np.flatnonzero(np.asarray(src_valid).ravel())
        if idx.size == 0:
            raise ValueError("src_valid excludes every source column")
        return idx[nearest_column_map(src_lat[idx], src_lon[idx],
                                      dst_lat, dst_lon)]

    def unit(lat, lon):
        return np.stack([np.cos(lat) * np.cos(lon),
                         np.cos(lat) * np.sin(lon),
                         np.sin(lat)], axis=-1)

    # Chunked over targets: the dense (n_dst, n_src) product is ~52 GB for an
    # ERA5 N320 source onto the res-6 mesh.  Row chunks give the identical argmax.
    src_t = unit(src_lat, src_lon).T
    dst = unit(dst_lat, dst_lon)
    step = max(1, _NN_CHUNK_ELEMS // max(src_t.shape[1], 1))
    return np.concatenate(
        [np.argmax(dst[i:i + step] @ src_t, axis=1)
         for i in range(0, dst.shape[0], step)]
        or [np.zeros(0, dtype=np.intp)])
