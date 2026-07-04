"""General immersed-boundary geometry for the lat-lon C-grid ocean.

Oceananigans-style ``GridFittedBoundary``: carve ARBITRARY solid cells and/or
thin-wall face barriers into an otherwise-active ocean, on top of the
bathymetry-following ``GridFittedBottom`` (partial bottom cells + ``bottom_level``
+ the 2-D ``land_mask``) the model already supports.

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
    untouched).  Shapes match the face masks: ``u_barrier`` ``(n_lat, n_lon+1,
    nlev)``, ``v_barrier`` ``(n_lat+1, n_lon, nlev)`` (2-D per-face barriers
    broadcast over levels).  Returns the masked ``(u_mask_3d, v_mask_3d)``.
    """
    um, vm = u_mask_3d, v_mask_3d
    if u_barrier is not None:
        ub = u_barrier[..., None] if u_barrier.ndim == um.ndim - 1 else u_barrier
        um = um * (1.0 - ub.astype(um.dtype))
    if v_barrier is not None:
        vb = v_barrier[..., None] if v_barrier.ndim == vm.ndim - 1 else v_barrier
        vm = vm * (1.0 - vb.astype(vm.dtype))
    return um, vm


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


def immersed_partial_cell_coordinate(
    coord: OceanPartialCellCoordinate,
    solid_3d: jnp.ndarray,
) -> OceanPartialCellCoordinate:
    """A masked :class:`OceanPartialCellCoordinate` with ``solid_3d`` carved out.

    Combines the coordinate's bathymetry ``is_active`` with the arbitrary
    ``solid_3d`` mask, zeroes ``h_partial`` on the newly-solid cells (so their
    volume and every flux-form transport through them vanish), and VALIDATES
    that the result keeps every column surface-connected (raises otherwise).
    ``bottom_level`` is left unchanged — it still marks the bathymetric
    seafloor; the solid mask carves additional interior/lateral geometry above
    it that the face masks + zero thickness enforce.
    """
    new_active = combine_solid_mask(coord.is_active, solid_3d)
    assert_columns_surface_connected(new_active)
    new_h = jnp.where(new_active, coord.h_partial, 0.0)
    return coord._replace(is_active=new_active, h_partial=new_h)
