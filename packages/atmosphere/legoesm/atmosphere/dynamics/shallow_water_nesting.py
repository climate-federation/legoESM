"""One-way (parent -> child) nested shallow-water stepping on the lat-lon C-grid.

This is the dynamics-side companion to :mod:`legoesm.grids.nesting`: it drives a
coarse global PARENT and a refined regional CHILD shallow-water integration, with
the child's lateral boundary PRESCRIBED each step from the interpolated parent
state (Davies 1976 specified boundary; the lat-lon analogue of the FV3 one-way
nest, Harris & Lin 2013).

Why a dedicated stepper (not ``CGridLatLonShallowWaterModel.step``)
------------------------------------------------------------------
The standalone C-grid SW model assumes a GLOBAL grid: periodic longitude
(``jnp.roll``), polar walls (``v = 0`` at the pole rows) and a domain-total mass
fixer.  A regional child has none of those — its lateral boundary is OPEN and
specified from the parent.  So we reuse the model's PURE tendency function
(:func:`cgrid_latlon_sw_tendencies`) and the shared time integrators
(:func:`dispatch_integrator`) verbatim, but replace the global BCs with:

1. **Specified boundary band.**  After each RK update, the outer ``n_halo``
   rows/columns of ``h`` (and the matching u/v faces) are OVERWRITTEN with the
   bilinearly-interpolated parent state.  The interior evolves freely.
2. **Interior mass conservation.**  The mass fixer acts on the INTERIOR cells
   only, anchored to the running interior-mass budget so a steady boundary
   forcing (Williamson-2) leaves the interior mass invariant to round-off.  The
   fixer correction is a uniform interior depth offset (conservative, smooth).

Pure JAX / pytree throughout — ``lax.scan``-able and ``jax.grad``-safe.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.grids.nesting import (
    NestedLatLonGrid,
    apply_boundary_interp,
)
from legoesm.atmosphere.dynamics.shallow_water_latlon_cgrid import (
    CGridLatLonShallowWaterState,
    CGridLatLonShallowWaterConfig,
    cgrid_latlon_sw_tendencies,
)
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.core.conservation import conservation_accumulator


class NestedSWState(NamedTuple):
    """Coupled parent + child shallow-water state for the one-way nest."""

    parent: CGridLatLonShallowWaterState
    child: CGridLatLonShallowWaterState


# ----------------------------------------------------------------------------
# Parent -> child boundary prescription
# ----------------------------------------------------------------------------


def interpolate_parent_to_child(
    parent: CGridLatLonShallowWaterState,
    nest: NestedLatLonGrid,
) -> CGridLatLonShallowWaterState:
    """Interpolate the full parent state onto the child C-grid (h, u, v, h_s).

    Returns a child-shaped :class:`CGridLatLonShallowWaterState`.  ``h``/``h_s``
    use the cell-centre weights; ``u`` the u-face weights; ``v`` the v-face
    weights — each staggered location interpolated from the parent CELL-CENTRE
    fields (the parent u/v live on its own faces, so we reconstruct cell-centre
    winds first, then prolong to the child faces; this keeps a single, smooth
    bilinear operator per field rather than face-to-face index gymnastics).
    """
    # Parent cell-centre winds from its C-grid faces (periodic in lon).
    pu_c = 0.5 * (parent.u[:, :-1] + parent.u[:, 1:])      # (n_lat_p, n_lon_p)
    pv_c = 0.5 * (parent.v[:-1, :] + parent.v[1:, :])      # (n_lat_p, n_lon_p)

    h_child = apply_boundary_interp(parent.h, nest.interp_centers)
    hs_child = apply_boundary_interp(parent.h_s, nest.interp_centers)
    u_child = apply_boundary_interp(pu_c, nest.interp_uface)
    v_child = apply_boundary_interp(pv_c, nest.interp_vface)

    return CGridLatLonShallowWaterState(
        h=h_child, u=u_child, v=v_child, h_s=hs_child,
    )


def _apply_boundary_band(
    child: CGridLatLonShallowWaterState,
    bc: CGridLatLonShallowWaterState,
    nest: NestedLatLonGrid,
) -> CGridLatLonShallowWaterState:
    """Prescribe the boundary via a Davies relaxation blend toward the parent.

    Each field is blended ``field = (1 - w) * child_free + w * bc`` with the
    relaxation weight ``w``: ``w = 1`` on the hard ``n_halo`` band (exact
    overwrite — the boundary fully specifies the flux into the relaxation zone),
    raised-cosine ``1 -> 0`` across the next ``n_relax`` cells (nudge, absorbing
    outgoing gravity waves), ``w = 0`` in the free interior.  ``h_s`` is held at
    the interpolated parent topography everywhere (static for SW test cases).

    The blend is the SAME weighted-average form on cell centres (``h``), u-faces
    (``u``) and v-faces (``v``), using the location-matched relaxation weight, so
    the staggered C-grid is forced consistently (no face/centre mismatch that
    would seed a checkerboard mode at the boundary).
    """
    wc = nest.relax_weight                # (n_lat, n_lon)   cell centres
    wu = nest.relax_weight_uface          # (n_lat, n_lon+1) u-faces
    wv = nest.relax_weight_vface          # (n_lat+1, n_lon) v-faces

    wc = wc.astype(child.h.dtype)
    wu = wu.astype(child.u.dtype)
    wv = wv.astype(child.v.dtype)

    h = (1.0 - wc) * child.h + wc * bc.h
    u = (1.0 - wu) * child.u + wu * bc.u
    v = (1.0 - wv) * child.v + wv * bc.v

    return CGridLatLonShallowWaterState(h=h, u=u, v=v, h_s=bc.h_s)


# ----------------------------------------------------------------------------
# Interior mass diagnostics / fixer
# ----------------------------------------------------------------------------


def interior_mass(
    child: CGridLatLonShallowWaterState,
    nest: NestedLatLonGrid,
) -> jax.Array:
    """fp64 total mass over the child's FREE INTERIOR (band + relax zone excluded).

    Conservation is defined on the freely-evolving cells (``relax_weight == 0``):
    the hard band is prescribed and the relaxation zone is nudged, so neither is a
    closed conserved region.  This is the load-bearing budget for the W2 test.
    """
    acc = conservation_accumulator()
    area = nest.child.area.astype(acc)
    mask = nest.free_interior_mask.astype(acc)
    return jnp.sum(child.h.astype(acc) * area * mask)


def _fix_interior_mass(
    child: CGridLatLonShallowWaterState,
    nest: NestedLatLonGrid,
    target_mass: jax.Array,
) -> CGridLatLonShallowWaterState:
    """Restore free-interior mass to *target_mass* via a uniform depth offset.

    Conservative: adds a constant ``delta`` to ``h`` on the FREE-interior cells
    only (where no nudging occurs), chosen so ``sum(h*area) == target_mass``
    there.  Mirrors the domain fixer in :class:`CGridLatLonShallowWaterModel` but
    masked to the free interior (the band is prescribed and the relaxation zone is
    nudged — both are excluded from the conserved budget).
    """
    acc = conservation_accumulator()
    area = nest.child.area.astype(acc)
    mask = nest.free_interior_mask.astype(acc)
    interior_area = jnp.sum(area * mask)
    mass_now = jnp.sum(child.h.astype(acc) * area * mask)
    delta = (target_mass - mass_now) / interior_area
    h_new = child.h + (delta * mask).astype(child.h.dtype)
    h_new = jnp.maximum(h_new, 0.0)
    return child._replace(h=h_new)


# ----------------------------------------------------------------------------
# Single nested step
# ----------------------------------------------------------------------------


def _free_child_step(
    child: CGridLatLonShallowWaterState,
    nest: NestedLatLonGrid,
    config: CGridLatLonShallowWaterConfig,
    dt: float,
) -> CGridLatLonShallowWaterState:
    """One RK update of the child interior using the shared SW tendencies.

    Uses :func:`cgrid_latlon_sw_tendencies` on the child grid unchanged — its
    periodic-roll longitude and pole rows are harmless here because the outer
    band is overwritten immediately afterward, so those stencils only ever feed
    the band cells we discard.  ``h_s`` tendency is held at zero (static topo).
    """
    def tendency_fn(s):
        dh, du, dv = cgrid_latlon_sw_tendencies(s, nest.child, config)
        return CGridLatLonShallowWaterState(
            h=dh, u=du, v=dv, h_s=jnp.zeros_like(s.h_s),
        )

    return dispatch_integrator(child, tendency_fn, dt, config.time_integrator)


def step_child(
    child: CGridLatLonShallowWaterState,
    parent_bc: CGridLatLonShallowWaterState,
    nest: NestedLatLonGrid,
    dt: float,
    interior_target: jax.Array,
    config: CGridLatLonShallowWaterConfig = CGridLatLonShallowWaterConfig(),
) -> CGridLatLonShallowWaterState:
    """Advance ONLY the child one step under prescribed parent boundary forcing.

    The one-way coupling means the parent is independent — a caller may integrate
    it with the FULL standalone :class:`CGridLatLonShallowWaterModel` (polar
    filter, etc.) and pass the resulting state as *parent_bc* here.  This is the
    boundary-forced child update: free interior RK step, re-prescribe the outer
    ``n_halo`` band from the interpolated parent, then restore interior mass.
    Pure pytree (``jax.grad`` / ``lax.scan`` safe).
    """
    bc = interpolate_parent_to_child(parent_bc, nest)
    child_free = _free_child_step(child, nest, config, dt)
    child_bc = _apply_boundary_band(child_free, bc, nest)
    if config.fix_mass:
        child_bc = _fix_interior_mass(child_bc, nest, interior_target)
    return child_bc


def nested_sw_step(
    state: NestedSWState,
    parent_bc: CGridLatLonShallowWaterState,
    nest: NestedLatLonGrid,
    dt: float,
    interior_target: jax.Array,
    config: CGridLatLonShallowWaterConfig = CGridLatLonShallowWaterConfig(),
) -> NestedSWState:
    """Advance the one-way nest one child step (concurrent-time, ratio 1 in time).

    PURE pytree function — ``lax.scan``-able and ``jax.grad``-safe.  ``nest`` and
    ``config`` are treated as closed-over CONSTANTS w.r.t. differentiation
    (geometry / static config), so callers that JIT should close over them via
    :func:`make_nested_stepper` rather than pass them as JIT static args (the
    cached interp-weight arrays inside ``nest`` are unhashable, so they cannot be
    JIT statics).

    Parameters
    ----------
    state
        Coupled parent + child state at the start of the step.
    parent_bc
        The parent state to use for the child boundary forcing over this step
        (typically the parent state at the same time level — passed explicitly so
        a single-time-level driver and a sub-cycling driver share one step fn).
    nest
        The :class:`NestedLatLonGrid` (carries the cached interp weights and the
        parent/child grids).
    dt
        Time step [s] (same for parent and child here — temporal refinement is a
        documented follow-up; spatial refinement is what conservation is proven
        on).
    interior_target
        fp64 target interior mass for the conservation fixer (compute once via
        :func:`interior_mass` on the initial child state).
    config
        Shared SW config.

    Returns
    -------
    NestedSWState
        Parent advanced with the standalone global SW step semantics; child
        advanced one interior step with its band re-prescribed from *parent_bc*.
    """
    parent, child = state

    # --- Parent: standalone global SW update (periodic lon, polar walls). ---
    # NOTE: this in-built parent step uses the BARE tendencies (no polar filter);
    # for pole-sensitive cases (Williamson-5) integrate the parent with the full
    # CGridLatLonShallowWaterModel and drive the child via step_child instead.
    def parent_tendency(s):
        dh, du, dv = cgrid_latlon_sw_tendencies(s, nest.parent, config)
        return CGridLatLonShallowWaterState(
            h=dh, u=du, v=dv, h_s=jnp.zeros_like(s.h_s),
        )

    parent_new = dispatch_integrator(
        parent, parent_tendency, dt, config.time_integrator,
    )
    # Enforce v = 0 at the parent poles (wall BC), as in the standalone model.
    parent_new = parent_new._replace(
        v=jnp.pad(parent_new.v[1:-1, :], ((1, 1), (0, 0))),
    )

    # --- Child: free interior update + re-prescribed band + interior mass fix. ---
    child_bc = step_child(child, parent_bc, nest, dt, interior_target, config)

    return NestedSWState(parent=parent_new, child=child_bc)


def make_nested_stepper(
    nest: NestedLatLonGrid,
    config: CGridLatLonShallowWaterConfig = CGridLatLonShallowWaterConfig(),
    *,
    jit: bool = True,
):
    """Build a stepper closing over the (static) ``nest`` + ``config``.

    Returns ``step(state, parent_bc, dt, interior_target) -> NestedSWState``.
    Closing over ``nest`` keeps its cached interp-weight arrays as compile-time
    constants (they cannot be JIT statics — arrays are unhashable — nor do we
    want to retrace when only the state changes).  Build ONCE outside any
    training/time loop (CLAUDE.md: never build closures per iteration).

    With ``jit=False`` the returned closure is the raw pure function — use it
    inside ``jax.grad`` / ``eqx.filter_value_and_grad`` (no buffer donation,
    AD-safe), mirroring the ``build_segment_fn(...).raw`` convention.
    """

    def _step(state, parent_bc, dt, interior_target):
        return nested_sw_step(state, parent_bc, nest, dt, interior_target, config)

    return jax.jit(_step) if jit else _step


def initial_nested_state(
    nest: NestedLatLonGrid,
    parent_ic: CGridLatLonShallowWaterState,
) -> NestedSWState:
    """Build the initial coupled state: parent IC + child = interpolated parent.

    The child is initialised by prolonging the parent IC onto the child grid, so
    the nest starts perfectly consistent with the parent (no spin-up shock at the
    boundary).  For a steady parent IC (Williamson-2) this is also the child's
    exact steady solution, which is what the interior mass-conservation test
    exploits.
    """
    child0 = interpolate_parent_to_child(parent_ic, nest)
    return NestedSWState(parent=parent_ic, child=child0)
