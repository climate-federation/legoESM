"""General immersed-boundary geometry for the lat-lon C-grid ocean.

**NOT YET INTEGRATED** — this module is the validated general-immersed-boundary
*core* (Oceananigans-style ``GridFittedBoundary``: carve ARBITRARY solid cells
and/or thin-wall face barriers into an otherwise-active ocean, on top of the
bathymetry-following ``GridFittedBottom`` — partial bottom cells +
``bottom_level`` + the 2-D ``land_mask`` — the model already supports).  It is
not yet imported by any production ocean experiment: every current experiment
(``overflow``, ``dino``, ``isomip_plus``, …) builds a SMOOTH bathymetry field
and uses the already-wired ``GridFittedBottom`` partial-cell path, so none needs
an interior obstacle / thin-wall.  Wire this in when an experiment requires a
general immersed boundary by:

1. building the ``solid_3d`` mask (and/or ``u_barrier``/``v_barrier``) for the
   geometry in the experiment's ``create_initial_conditions``;
2. calling :func:`carve_immersed_state` with the freshly-built
   ``(state, z_coord, solid_3d)`` to get a CONSISTENT ``(state, z_coord)`` pair
   (``H_bathy``, ``land_mask``, face masks and the partial-cell coordinate all
   updated together) — never carve the coordinate alone (see the footgun note on
   :func:`carve_immersed_state`);
3. threading the barrier-augmented face masks through the experiment's operator
   setup via :func:`immersed_face_masks` when a thin-wall sill/dam is needed;
4. adding a matrix case + a conservation/impermeability regression before the
   experiment is declared production-ready.

The C-grid tracer/momentum operators already enforce no flux through a face
whose per-level mask is zero (fluxes are ``h_face * u * u_mask_3d``), and
:func:`legoesm.ocean.dynamics.latlon_cgrid_operators.compute_face_masks_3d`
already builds those per-level face masks from an arbitrary 3-D cell-activity
mask.  This module is the user-facing layer on top of that leak-free machinery:

* :func:`combine_solid_mask` — merge an arbitrary 3-D ``solid`` mask into the
  coordinate's ``is_active`` (a cell is active iff wet AND not solid);
* :func:`add_face_barriers` — impose THIN-WALL barriers on specific u/v faces
  (a sub-grid sill / dam / strait closure) WITHOUT masking either adjacent cell,
  so the vertical column stays fully active and the hydrostatic / vertical-mixing
  logic is untouched — the safe way to add interior walls;
* :func:`immersed_partial_cell_coordinate` — a masked
  :class:`OceanPartialCellCoordinate` from a solid-cell mask, VALIDATED to keep
  every column surface-connected;
* :func:`carve_immersed_state` — the ATOMIC entry: carve the coordinate AND the
  live C-grid state (``H_bathy`` + the three masks) together;
* :func:`mask_immersed_field` — zero a prognostic field on solid cells.

SCOPE / SAFETY.  The masked cells must keep every water column
SURFACE-CONNECTED (active cells top-anchored and contiguous):
:func:`assert_columns_surface_connected` raises on an overhang / sub-surface
cavity (an active cell beneath a solid cell).  Those non-monotonic geometries
(ice-shelf cavities, overhangs) need a reworked vertical column — the
hydrostatic pressure integrates from the free surface downward and the vertical
mixing solves a top-anchored tridiagonal — and are the deliberately-deferred
remaining ``GridFittedBoundary`` gap.  Thin-wall barriers are ALWAYS safe (they
mask only faces, never cells, so connectivity is preserved).
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d
from legoesm.ocean.vertical import OceanPartialCellCoordinate


def combine_solid_mask(is_active_3d: jnp.ndarray,
                       solid_3d: jnp.ndarray) -> jnp.ndarray:
    """Active-cell mask with ``solid_3d`` cells carved out.

    A cell is active iff it was already active (wet, above the seafloor) AND
    not flagged solid.  Both arrays are ``(n_lat, n_lon, nlev)``; returns bool.
    """
    return is_active_3d.astype(bool) & ~solid_3d.astype(bool)


def assert_columns_surface_connected(is_active_3d: jnp.ndarray) -> None:
    """Raise if any water column is NOT surface-connected.

    Enforces the top-anchored contiguity invariant the hydrostatic vertical
    column relies on: an active cell may not sit BENEATH an inactive cell
    (that is an overhang / sub-surface cavity).  Host-side check on the static
    geometry mask (not traced), so a Python ``if`` is correct here.
    """
    a = is_active_3d.astype(bool)
    # An active cell at level k (k>=1) whose neighbour ABOVE (k-1) is inactive
    # is an overhang: active-below-inactive.
    overhang = a[..., 1:] & ~a[..., :-1]
    if bool(jnp.any(overhang)):
        n = int(jnp.sum(overhang))
        raise ValueError(
            f"immersed geometry is not surface-connected: {n} active cell(s) "
            "sit beneath a solid/inactive cell (an overhang / sub-surface "
            "cavity).  The hydrostatic-PGF and vertical-mixing column logic "
            "assumes top-anchored contiguous active cells; non-monotonic "
            "vertical geometry (ice-shelf cavities, overhangs) is the deferred "
            "GridFittedBoundary vertical-rework gap.  Use thin-wall face "
            "barriers (add_face_barriers) for interior walls instead, or carve "
            "solid cells only from the column bottom upward."
        )


def add_face_barriers(u_mask_3d: jnp.ndarray, v_mask_3d: jnp.ndarray,
                      u_barrier: jnp.ndarray | None = None,
                      v_barrier: jnp.ndarray | None = None,
                      ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Impose thin-wall barriers on u/v faces (``barrier==1`` -> wall).

    A barrier zeroes the face mask so the flux-form operators pass NO flux
    through it, WITHOUT masking either adjacent cell — a sub-grid sill / dam /
    strait closure that leaves both columns fully active (vertical logic
    untouched).  Shapes match the face masks: ``u_barrier`` ``(n_lat, n_lon+1
    [, nlev])``, ``v_barrier`` ``(n_lat+1, n_lon [, nlev])`` (2-D per-face
    barriers broadcast over levels).  Returns the masked
    ``(u_mask_3d, v_mask_3d)``.
    """
    def _apply(mask, barrier, name):
        if barrier is None:
            return mask
        # Static host-side geometry: validate the FULL shape up front so a
        # transposed / mis-sized barrier fails loudly instead of broadcasting a
        # wall onto the wrong faces or levels.  A per-level barrier must match
        # the face mask exactly (a ``(..., 1)`` or ``(..., nlev±1)`` barrier
        # would otherwise silently broadcast one level onto the whole column);
        # a face-plane (level-less) barrier must match the mask's spatial plane.
        if barrier.ndim == mask.ndim:
            if tuple(barrier.shape) != tuple(mask.shape):
                raise ValueError(
                    f"{name} per-level shape {tuple(barrier.shape)} != face-mask "
                    f"shape {tuple(mask.shape)}")
        elif barrier.ndim == mask.ndim - 1:
            if tuple(barrier.shape) != tuple(mask.shape[:-1]):
                raise ValueError(
                    f"{name} face-plane shape {tuple(barrier.shape)} != face-mask "
                    f"plane {tuple(mask.shape[:-1])}")
        else:
            raise ValueError(
                f"{name} must be {mask.ndim - 1}-D (face plane) or {mask.ndim}-D "
                f"(per-level), got ndim={barrier.ndim}")
        b = barrier[..., None] if barrier.ndim == mask.ndim - 1 else barrier
        return mask * (1.0 - b.astype(mask.dtype))

    return _apply(u_mask_3d, u_barrier, "u_barrier"), \
        _apply(v_mask_3d, v_barrier, "v_barrier")


def immersed_face_masks(is_active_3d: jnp.ndarray, grid=None,
                        u_barrier: jnp.ndarray | None = None,
                        v_barrier: jnp.ndarray | None = None,
                        ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-level u/v face masks for a general immersed boundary.

    Builds the leak-free per-level face masks from the 3-D activity mask (a
    face is wet iff BOTH adjacent cells are wet at that level) via
    :func:`compute_face_masks_3d`, then applies any thin-wall barriers.  The
    C-grid tracer/momentum operators consume these directly (``h*u*u_mask_3d``),
    so no flux crosses a masked face — impermeability by construction.
    """
    um, vm = compute_face_masks_3d(is_active_3d, grid)
    return add_face_barriers(um, vm, u_barrier, v_barrier)


def mask_immersed_field(field: jnp.ndarray, is_active_3d: jnp.ndarray,
                        value: float = 0.0) -> jnp.ndarray:
    """Set ``field`` to ``value`` on solid (inactive) cells.

    ``is_active_3d`` broadcasts against ``field`` (e.g. a 2-D mask over a 3-D
    field, or the exact 3-D mask).  Used to keep tracers/velocity zero inside
    solid geometry so diagnostics and re-derived quantities never read stale
    values from a masked cell.
    """
    return jnp.where(is_active_3d.astype(bool), field, value)


def immersed_column_depth(coord: OceanPartialCellCoordinate) -> jnp.ndarray:
    """Per-column water depth ``H_bathy = Σ_k h_partial`` [m], shape ``(...,)``.

    The coordinate invariant is ``Σ_k h_partial == H_bathy``, so after a solid
    carve reduces some cells' thickness to zero THIS is the depth callers MUST
    pass to :func:`legoesm.ocean.vertical.compute_layer_thickness` (which scales
    partial cells by ``(eta + H_bathy)/H_bathy``) — never a separately-cached
    pre-carve bathymetry, which would mis-scale the live column thickness.
    """
    return jnp.sum(coord.h_partial, axis=-1)


def immersed_partial_cell_coordinate(
    coord: OceanPartialCellCoordinate,
    solid_3d: jnp.ndarray,
) -> OceanPartialCellCoordinate:
    """A masked :class:`OceanPartialCellCoordinate` with ``solid_3d`` carved out.

    Combines the coordinate's bathymetry ``is_active`` with the arbitrary
    ``solid_3d`` mask, zeroes ``h_partial`` on the newly-solid cells (so their
    volume and every flux-form transport through them vanish), RECOMPUTES
    ``bottom_level`` as the deepest still-active level per column (so
    bottom-cell physics — bottom drag, the bottom boundary layer — target the
    NEW seafloor, not a carved-out level), and VALIDATES surface-connectivity
    (raises on an overhang / cavity).

    NOTE on depth: a bottom-anchored carve makes the column shallower, so its
    ``H_bathy = Σ h_partial`` shrinks — callers computing layer thickness must
    use :func:`immersed_column_depth` (the coordinate invariant), NOT a cached
    pre-carve bathymetry.  ``H_max`` / the reference z-grid are unchanged.  When
    a live C-grid state exists, prefer :func:`carve_immersed_state`, which also
    keeps ``state.H_bathy`` / ``land_mask`` / face masks consistent.
    """
    new_active = combine_solid_mask(coord.is_active, solid_3d)
    assert_columns_surface_connected(new_active)
    new_h = jnp.where(new_active, coord.h_partial, 0.0)
    # Deepest still-active level per column (-1 for a fully-solid/dry column),
    # so bottom_level tracks the carved seafloor.
    nlev = new_active.shape[-1]
    k_idx = jnp.arange(nlev, dtype=coord.bottom_level.dtype)
    active_k = jnp.where(new_active, k_idx, jnp.asarray(-1, coord.bottom_level.dtype))
    new_bottom = jnp.max(active_k, axis=-1)
    return coord._replace(
        is_active=new_active, h_partial=new_h, bottom_level=new_bottom)


def carve_immersed_state(state, coord: OceanPartialCellCoordinate,
                         solid_3d: jnp.ndarray):
    """Atomically carve ``solid_3d`` into BOTH the coordinate and the C-grid state.

    A bottom-anchored solid carve makes columns shallower (or fully dry).  The
    cgrid model reads ``state.H_bathy`` — a field SEPARATE from the partial-cell
    coordinate — in every ``compute_layer_thickness`` call, so carving only the
    coordinate leaves ``state.H_bathy`` at its pre-carve value and silently
    mis-scales the live column thickness (partial cells scale by
    ``(eta + H_bathy)/H_bathy``): a mass-conservation footgun.  This returns a
    CONSISTENT ``(new_state, new_coord)`` pair:

    * ``new_coord`` = :func:`immersed_partial_cell_coordinate` (masked
      ``is_active``, zeroed ``h_partial`` on solids, ``bottom_level`` tracking
      the new seafloor, surface-connectivity validated);
    * ``new_state.H_bathy`` <- :func:`immersed_column_depth` (= ``Σ h_partial``),
      the depth callers must use for layer thickness;
    * ``new_state.land_mask`` <- 0 on any column carved FULLY solid (no active
      cell left, ``bottom_level == -1``), with ``u_mask``/``v_mask`` recomputed
      via :func:`legoesm.ocean.init_latlon_cgrid.replace_land_mask` so the three
      masks stay consistent (never ``_replace(land_mask=...)`` alone — stale
      face masks leak flux through walls).

    Sign convention: ``H_bathy`` is positive-downward depth [m]; the carved
    depth ``Σ h_partial >= 0`` and a fully-solid column has depth 0 AND
    ``land_mask = 0`` (consistent land).  A partially-carved column keeps
    ``land_mask = 1`` (still ocean) with a reduced ``H_bathy``.

    Returns ``(new_state, new_coord)``.
    """
    # Function-scope import: init_latlon_cgrid imports latlon_cgrid_operators
    # (this module's sibling), so a top-level import would be an ordering hazard.
    from legoesm.ocean.init_latlon_cgrid import replace_land_mask

    new_coord = immersed_partial_cell_coordinate(coord, solid_3d)
    new_H = immersed_column_depth(new_coord)                    # (n_lat, n_lon)

    # A column carved fully solid (no active cell at any level -> bottom_level
    # == -1) becomes LAND: drop it from land_mask so the recomputed face masks
    # wall it off.  A still-wet column keeps its existing land/ocean flag.
    column_wet = (new_coord.bottom_level >= 0).astype(new_H.dtype)
    new_land_mask = state.land_mask.data.astype(new_H.dtype) * column_wet

    new_state = replace_land_mask(state, new_land_mask)
    new_state = new_state._replace(
        H_bathy=Field(data=new_H, name="H_bathy",
                      dims=state.H_bathy.dims, units=state.H_bathy.units),
    )
    return new_state, new_coord
