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
(:func:`cgrid_latlon_sw_tendencies`) and the SAME SSP-RK3 stage arithmetic as
:mod:`legoesm.timestepping.ssp_rk3` verbatim, but replace the global BCs with:

1. **Specified boundary band, enforced PER SUBSTAGE.**  The outer ``n_halo``
   rows/columns of ``h`` (and the matching u/v faces) are OVERWRITTEN with the
   stagger-correctly interpolated parent state, and a Davies (1976) relaxation
   zone just inside it is nudged toward the parent.  Because the global tendency
   stencil advances the band with periodic/pole operators that are WRONG for a
   regional child, the band (+ relaxation) is re-prescribed to the INITIAL state
   and AFTER EVERY SSP-RK3 substage — before the next tendency evaluation — so no
   globally-polluted boundary value ever feeds a later stage into the interior.
   The interior evolves freely.
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
from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
    CGridLatLonShallowWaterState,
    CGridLatLonShallowWaterConfig,
    cgrid_latlon_sw_tendencies,
)
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.pytree_ops import (
    pytree_axpy,
    pytree_linear_combination,
)
from legoesm.core.conservation import conservation_accumulator


# SSP-RK3 (Shu & Osher 1988) stage coefficients, shared with
# ``legoesm.timestepping.ssp_rk3.ssp_rk3_step`` (same scheme; we re-expand the
# stages here ONLY so the prescribed boundary band can be re-enforced AFTER each
# substage — see :func:`_bc_enforcing_child_step`).  Kept as the SSP convex
# weights (alpha on the running stage state, beta on the forward-Euler update).
_SSP_RK3_KEYS = frozenset({"ssp_rk3", "ssp3", "rk3"})


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

    Returns a child-shaped :class:`CGridLatLonShallowWaterState`.  Each field is
    prolonged from the MATCHING parent staggered location: ``h``/``h_s`` from the
    parent CELL CENTRES (``interp_centers``), ``u`` from the parent U-FACES
    (``interp_uface``) and ``v`` from the parent V-FACES (``interp_vface``).

    This is STAGGER-CORRECT face-to-face prolongation.  We do NOT average the
    parent u/v faces to cell centres and bilinearly map those smoothed centre
    fields back onto the child faces: that cell-centre round-trip is not
    stagger-correct (a child face coincident with a parent face would return a
    blend of adjacent parent-face averages, not the parent face value), which
    destroys the discrete geostrophic/divergence balance at the boundary and
    seeds checkerboard / grid-scale wind noise (the W2 v-wind + W5 wind_speed edge
    artifact).  Sourcing each face directly from the parent face field reproduces
    a coincident parent face value to round-off and keeps the balance intact.
    The face-source weights live in ``nest`` (built once at construction, geometry
    only), so this stays a pure-JAX gather (JIT / ``jax.grad`` safe).
    """
    h_child = apply_boundary_interp(parent.h, nest.interp_centers)
    hs_child = apply_boundary_interp(parent.h_s, nest.interp_centers)
    u_child = apply_boundary_interp(parent.u, nest.interp_uface)
    v_child = apply_boundary_interp(parent.v, nest.interp_vface)

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


def _overwrite_hard_band(
    child: CGridLatLonShallowWaterState,
    bc: CGridLatLonShallowWaterState,
    nest: NestedLatLonGrid,
) -> CGridLatLonShallowWaterState:
    """Hard-overwrite ONLY the prescribed band (relaxation weight == 1) with *bc*.

    This is the IDEMPOTENT specified-boundary part of :func:`_apply_boundary_band`
    (the ``w == 1`` hard ``n_halo`` band), with NO Davies relaxation nudge.  It is
    what gets re-applied after EVERY RK substage (BUG 2 fix): the hard band is a
    pure overwrite, so applying it once per substage is mathematically the same as
    applying it once (no accumulation), and it keeps the band — which the global
    tendency stencil would otherwise advance with periodic/pole operators — pinned
    to the parent BC before each tendency evaluation.  Crucially the soft Davies
    RELAXATION nudge (``0 < w < 1``) is NOT applied here: that is a source term
    that must be integrated ONCE per step (applying the raised-cosine nudge every
    substage would triple its effective strength and inject imbalance), so the
    full :func:`_apply_boundary_band` is applied only once, after the step.

    Hard-band masks are ``relax_weight == 1`` on each staggered location; we build
    them from the relaxation weights so the band geometry stays single-sourced.
    """
    mc = (nest.relax_weight == 1.0).astype(child.h.dtype)
    mu = (nest.relax_weight_uface == 1.0).astype(child.u.dtype)
    mv = (nest.relax_weight_vface == 1.0).astype(child.v.dtype)

    h = (1.0 - mc) * child.h + mc * bc.h
    u = (1.0 - mu) * child.u + mu * bc.u
    v = (1.0 - mv) * child.v + mv * bc.v
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


def _bc_enforcing_child_step(
    child: CGridLatLonShallowWaterState,
    bc: CGridLatLonShallowWaterState,
    nest: NestedLatLonGrid,
    config: CGridLatLonShallowWaterConfig,
    dt: float,
) -> CGridLatLonShallowWaterState:
    """One BOUNDARY-AWARE SSP-RK3 child update: hard band pinned PER SUBSTAGE.

    The child shares the global SW tendency (:func:`cgrid_latlon_sw_tendencies`),
    whose periodic-roll longitude and pole rows are WRONG for a regional child:
    they advance the prescribed boundary band with global operators.  A
    multi-stage integrator evaluates the tendency on INTERMEDIATE stage states, so
    if the band is only fixed AFTER the full step (the old free step + post-step
    blend), those globally-polluted band values feed the later stages' tendencies
    into the relaxation zone and free interior — injecting boundary noise /
    spurious reflection even though the FINAL band is later overwritten (it still
    passes a final-band-equality test).

    Fix (BUG 2): re-OVERWRITE the HARD band (the ``n_halo`` ``w == 1`` cells,
    :func:`_overwrite_hard_band`) on the INITIAL state and AFTER EVERY SSP-RK3
    substage, so every tendency evaluation reads the correct prescribed boundary
    before it advances the interior.  The hard overwrite is IDEMPOTENT, so doing
    it per substage is identical to doing it once — no accumulation.  The soft
    Davies RELAXATION nudge (``0 < w < 1``) is deliberately NOT applied here: it is
    a source term that must be integrated ONCE per step (the calibrated
    raised-cosine strength that ``test_relaxation_reduces_boundary_noise`` checks);
    applying it every substage would triple its strength and inject imbalance.  So
    the caller applies the full :func:`_apply_boundary_band` once, after this step.

    This is the same SSP-RK3 (Shu & Osher 1988) as
    :func:`legoesm.timestepping.ssp_rk3.ssp_rk3_step` — identical stage
    coefficients and ``_pytree`` arithmetic — with the idempotent hard-band
    overwrite composed after each stage; it is NOT a new integrator.  Only SSP-RK3
    is supported for the nest (the validated scheme); any other ``time_integrator``
    raises, rather than silently per-stage-enforcing an integrator whose substages
    have a different meaning.  ``h_s`` tendency is held at zero (static
    topography).  Pure pytree (``jax.grad`` / ``lax.scan`` safe).
    """
    key = config.time_integrator.lower()
    if key not in _SSP_RK3_KEYS:
        raise ValueError(
            "Nested child stepping requires a per-substage boundary-enforced "
            f"SSP-RK3 integrator; got time_integrator={config.time_integrator!r}. "
            f"Supported: {sorted(_SSP_RK3_KEYS)} (the validated nest scheme)."
        )
    # The biharmonic L(L(u)) stencil reaches 2 cells: with n_halo < 2 the
    # first Laplacian at the pinned band reads the child's invalid
    # global-edge neighbor and the second propagates it into the first
    # unpinned row/column (codex review).  Fail loud rather than corrupt
    # the relaxation zone silently.
    if config.nu_del4 > 0.0 and nest.n_halo < 2:
        raise ValueError(
            "nu_del4 > 0 on a nested child requires n_halo >= 2 (the del-4 "
            f"stencil is 2 cells wide); got n_halo={nest.n_halo}.")

    def tendency_fn(s):
        dh, du, dv = cgrid_latlon_sw_tendencies(s, nest.child, config, dt)
        return CGridLatLonShallowWaterState(
            h=dh, u=du, v=dv, h_s=jnp.zeros_like(s.h_s),
        )

    def pin_band(s):
        return _overwrite_hard_band(s, bc, nest)

    # Initial state must already satisfy the prescribed hard band so stage 1's
    # tendency reads it (the caller may pass a child whose band has drifted).
    state = pin_band(child)

    # Stage 1: k1 = state + dt*F(state); re-pin band before stage 2's tendency.
    k1 = pytree_axpy(state, tendency_fn(state), dt)
    k1 = pin_band(k1)

    # Stage 2: k2 = 3/4 state + 1/4 (k1 + dt*F(k1)); re-pin.
    k1_eul = pytree_axpy(k1, tendency_fn(k1), dt)
    k2 = pytree_linear_combination(state, k1_eul, 0.75, 0.25)
    k2 = pin_band(k2)

    # Stage 3: k3 = 1/3 state + 2/3 (k2 + dt*F(k2)); final hard-band pin.
    k2_eul = pytree_axpy(k2, tendency_fn(k2), dt)
    k3 = pytree_linear_combination(state, k2_eul, 1.0 / 3.0, 2.0 / 3.0)
    k3 = pin_band(k3)
    return k3


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
    boundary-forced child update: a BOUNDARY-AWARE SSP-RK3 interior step that
    pins the HARD ``n_halo`` band to the interpolated parent AFTER EVERY substage
    (so no globally-polluted boundary feeds a later stage), then applies the
    Davies relaxation blend ONCE (the calibrated raised-cosine nudge, integrated
    once per step), then restores interior mass.  Pure pytree (``jax.grad`` /
    ``lax.scan`` safe).
    """
    bc = interpolate_parent_to_child(parent_bc, nest)
    # Boundary-aware integrator: HARD band pinned on the initial state and after
    # every RK substage (idempotent overwrite — no accumulation).
    child_bc = _bc_enforcing_child_step(child, bc, nest, config, dt)
    # Davies relaxation blend applied ONCE per step (hard band + soft nudge);
    # the hard band is already at bc, so this only adds the once-per-step nudge in
    # the relaxation zone.
    child_bc = _apply_boundary_band(child_bc, bc, nest)
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
        dh, du, dv = cgrid_latlon_sw_tendencies(s, nest.parent, config, dt)
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
